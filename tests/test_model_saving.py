import pytest
import os
import h5py
import tempfile
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_raises

from keras import backend as K
from keras.engine.saving import preprocess_weights_for_loading
from keras.models import Model, Sequential
from keras.layers import Dense, Lambda, RepeatVector, TimeDistributed, Bidirectional, GRU, LSTM, CuDNNGRU, CuDNNLSTM
from keras.layers import Conv2D, Flatten
from keras.layers import Input, InputLayer
from keras.initializers import Constant
from keras import optimizers
from keras import losses
from keras import metrics
from keras.utils.test_utils import keras_test
from keras.models import save_model, load_model


skipif_no_tf_gpu = pytest.mark.skipif(
    (K.backend() != 'tensorflow') or (not K.tensorflow_backend._get_available_gpus()),
    reason='Requires TensorFlow backend and a GPU')


@keras_test
def test_sequential_model_saving():
    model = Sequential()
    model.add(Dense(2, input_shape=(3,)))
    model.add(RepeatVector(3))
    model.add(TimeDistributed(Dense(3)))
    model.compile(loss=losses.MSE,
                  optimizer=optimizers.RMSprop(lr=0.0001),
                  metrics=[metrics.categorical_accuracy],
                  sample_weight_mode='temporal')
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)

    new_model = load_model(fname)
    os.remove(fname)

    out2 = new_model.predict(x)
    assert_allclose(out, out2, atol=1e-05)

    # test that new updates are the same with both models
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)
    new_model.train_on_batch(x, y)
    out = model.predict(x)
    out2 = new_model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


@keras_test
def test_sequential_model_saving_2():
    # test with custom optimizer, loss
    custom_opt = optimizers.rmsprop
    custom_loss = losses.mse
    model = Sequential()
    model.add(Dense(2, input_shape=(3,)))
    model.add(Dense(3))
    model.compile(loss=custom_loss, optimizer=custom_opt(), metrics=['acc'])

    x = np.random.random((1, 3))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)

    model = load_model(fname,
                       custom_objects={'custom_opt': custom_opt,
                                       'custom_loss': custom_loss})
    os.remove(fname)

    out2 = model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


@keras_test
def test_functional_model_saving():
    inputs = Input(shape=(3,))
    x = Dense(2)(inputs)
    outputs = Dense(3)(x)

    model = Model(inputs, outputs)
    model.compile(loss=losses.MSE,
                  optimizer=optimizers.Adam(),
                  metrics=[metrics.categorical_accuracy])
    x = np.random.random((1, 3))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)

    model = load_model(fname)
    os.remove(fname)

    out2 = model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


@keras_test
def test_model_saving_to_pre_created_h5py_file():
    inputs = Input(shape=(3,))
    x = Dense(2)(inputs)
    outputs = Dense(3)(x)

    model = Model(inputs, outputs)
    model.compile(loss=losses.MSE,
                  optimizer=optimizers.Adam(),
                  metrics=[metrics.categorical_accuracy])
    x = np.random.random((1, 3))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    _, fname = tempfile.mkstemp('.h5')
    with h5py.File(fname, mode='r+') as h5file:
        save_model(model, h5file)
        loaded_model = load_model(h5file)
        out2 = loaded_model.predict(x)
    assert_allclose(out, out2, atol=1e-05)

    # test non-default options in h5
    with h5py.File('does not matter', driver='core',
                   backing_store=False) as h5file:
        save_model(model, h5file)
        loaded_model = load_model(h5file)
        out2 = loaded_model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


@keras_test
def test_model_saving_to_binary_stream():
    inputs = Input(shape=(3,))
    x = Dense(2)(inputs)
    outputs = Dense(3)(x)

    model = Model(inputs, outputs)
    model.compile(loss=losses.MSE,
                  optimizer=optimizers.Adam(),
                  metrics=[metrics.categorical_accuracy])
    x = np.random.random((1, 3))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    _, fname = tempfile.mkstemp('.h5')
    with h5py.File(fname, mode='r+') as h5file:
        save_model(model, h5file)
        loaded_model = load_model(h5file)
        out2 = loaded_model.predict(x)
    assert_allclose(out, out2, atol=1e-05)

    # Save the model to an in-memory-only h5 file.
    with h5py.File('does not matter', driver='core',
                   backing_store=False) as h5file:
        save_model(model, h5file)
        h5file.flush()  # Very important! Otherwise you get all zeroes below.
        binary_data = h5file.fid.get_file_image()

        # Make sure the binary data is correct by saving it to a file manually
        # and then loading it the usual way.
        with open(fname, 'wb') as raw_file:
            raw_file.write(binary_data)

    # Load the manually-saved binary data, and make sure the model is intact.
    with h5py.File(fname, mode='r') as h5file:
        loaded_model = load_model(h5file)
        out2 = loaded_model.predict(x)

    assert_allclose(out, out2, atol=1e-05)


@keras_test
def test_saving_multiple_metrics_outputs():
    inputs = Input(shape=(5,))
    x = Dense(5)(inputs)
    output1 = Dense(1, name='output1')(x)
    output2 = Dense(1, name='output2')(x)

    model = Model(inputs=inputs, outputs=[output1, output2])

    metrics = {'output1': ['mse', 'binary_accuracy'],
               'output2': ['mse', 'binary_accuracy']
               }
    loss = {'output1': 'mse', 'output2': 'mse'}

    model.compile(loss=loss, optimizer='sgd', metrics=metrics)

    # assure that model is working
    x = np.array([[1, 1, 1, 1, 1]])
    out = model.predict(x)
    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)

    model = load_model(fname)
    os.remove(fname)

    out2 = model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


@keras_test
def test_saving_without_compilation():
    """Test saving model without compiling.
    """
    model = Sequential()
    model.add(Dense(2, input_shape=(3,)))
    model.add(Dense(3))

    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)
    model = load_model(fname)
    os.remove(fname)


@keras_test
def test_saving_right_after_compilation():
    model = Sequential()
    model.add(Dense(2, input_shape=(3,)))
    model.add(Dense(3))
    model.compile(loss='mse', optimizer='sgd', metrics=['acc'])
    model._make_train_function()

    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)
    model = load_model(fname)
    os.remove(fname)


@keras_test
def test_saving_unused_layers_is_ok():
    a = Input(shape=(256, 512, 6))
    b = Input(shape=(256, 512, 1))
    c = Lambda(lambda x: x[:, :, :, :1])(a)

    model = Model(inputs=[a, b], outputs=c)

    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)
    load_model(fname)
    os.remove(fname)


@keras_test
def test_loading_weights_by_name_and_reshape():
    """
    test loading model weights by name on:
        - sequential model
    """

    # test with custom optimizer, loss
    custom_opt = optimizers.rmsprop
    custom_loss = losses.mse

    # sequential model
    model = Sequential()
    model.add(Conv2D(2, (1, 1), input_shape=(1, 1, 1), name='rick'))
    model.add(Flatten())
    model.add(Dense(3, name='morty'))
    model.compile(loss=custom_loss, optimizer=custom_opt(), metrics=['acc'])

    x = np.random.random((1, 1, 1, 1))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    old_weights = [layer.get_weights() for layer in model.layers]
    _, fname = tempfile.mkstemp('.h5')

    model.save_weights(fname)

    # delete and recreate model
    del(model)
    model = Sequential()
    model.add(Conv2D(2, (1, 1), input_shape=(1, 1, 1), name='rick'))
    model.add(Conv2D(3, (1, 1), name='morty'))
    model.compile(loss=custom_loss, optimizer=custom_opt(), metrics=['acc'])

    # load weights from first model
    with pytest.raises(ValueError):
        model.load_weights(fname, by_name=True, reshape=False)
    with pytest.raises(ValueError):
        model.load_weights(fname, by_name=False, reshape=False)
    model.load_weights(fname, by_name=False, reshape=True)
    model.load_weights(fname, by_name=True, reshape=True)

    out2 = model.predict(x)
    assert_allclose(np.squeeze(out), np.squeeze(out2), atol=1e-05)
    for i in range(len(model.layers)):
        new_weights = model.layers[i].get_weights()
        for j in range(len(new_weights)):
            # only compare layers that have weights, skipping Flatten()
            if old_weights[i]:
                assert_allclose(old_weights[i][j], new_weights[j], atol=1e-05)

    # delete and recreate model with `use_bias=False`
    del(model)
    model = Sequential()
    model.add(Conv2D(2, (1, 1), input_shape=(1, 1, 1), use_bias=False, name='rick'))
    model.add(Flatten())
    model.add(Dense(3, name='morty'))
    with pytest.raises(ValueError,
                       match=r'.* expects [0-9]+ .* but the saved .* [0-9]+ .*'):
        model.load_weights(fname)
    with pytest.raises(ValueError,
                       match=r'.* expects [0-9]+ .* but the saved .* [0-9]+ .*'):
        model.load_weights(fname, by_name=True)
    with pytest.warns(UserWarning,
                      match=r'Skipping loading .* due to mismatch .*'):
        model.load_weights(fname, by_name=True, skip_mismatch=True)

    # delete and recreate model with `filters=10`
    del(model)
    model = Sequential()
    model.add(Conv2D(10, (1, 1), input_shape=(1, 1, 1), name='rick'))
    with pytest.raises(ValueError,
                       match=r'.* has shape .* but the saved .* shape .*'):
        model.load_weights(fname, by_name=True)
    with pytest.raises(ValueError,
                       match=r'.* load .* [0-9]+ layers into .* [0-9]+ layers.'):
        model.load_weights(fname)

    os.remove(fname)


@keras_test
def test_loading_weights_by_name_2():
    """
    test loading model weights by name on:
        - both sequential and functional api models
        - different architecture with shared names
    """

    # test with custom optimizer, loss
    custom_opt = optimizers.rmsprop
    custom_loss = losses.mse

    # sequential model
    model = Sequential()
    model.add(Dense(2, input_shape=(3,), name='rick'))
    model.add(Dense(3, name='morty'))
    model.compile(loss=custom_loss, optimizer=custom_opt(), metrics=['acc'])

    x = np.random.random((1, 3))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    old_weights = [layer.get_weights() for layer in model.layers]
    _, fname = tempfile.mkstemp('.h5')

    model.save_weights(fname)

    # delete and recreate model using Functional API
    del(model)
    data = Input(shape=(3,))
    rick = Dense(2, name='rick')(data)
    jerry = Dense(3, name='jerry')(rick)  # add 2 layers (but maintain shapes)
    jessica = Dense(2, name='jessica')(jerry)
    morty = Dense(3, name='morty')(jessica)

    model = Model(inputs=[data], outputs=[morty])
    model.compile(loss=custom_loss, optimizer=custom_opt(), metrics=['acc'])

    # load weights from first model
    model.load_weights(fname, by_name=True)
    os.remove(fname)

    out2 = model.predict(x)
    assert np.max(np.abs(out - out2)) > 1e-05

    rick = model.layers[1].get_weights()
    jerry = model.layers[2].get_weights()
    jessica = model.layers[3].get_weights()
    morty = model.layers[4].get_weights()

    assert_allclose(old_weights[0][0], rick[0], atol=1e-05)
    assert_allclose(old_weights[0][1], rick[1], atol=1e-05)
    assert_allclose(old_weights[1][0], morty[0], atol=1e-05)
    assert_allclose(old_weights[1][1], morty[1], atol=1e-05)
    assert_allclose(np.zeros_like(jerry[1]), jerry[1])  # biases init to 0
    assert_allclose(np.zeros_like(jessica[1]), jessica[1])  # biases init to 0


@keras_test
def test_loading_weights_by_name_skip_mismatch():
    """
    test skipping layers while loading model weights by name on:
        - sequential model
    """

    # test with custom optimizer, loss
    custom_opt = optimizers.rmsprop
    custom_loss = losses.mse

    # sequential model
    model = Sequential()
    model.add(Dense(2, input_shape=(3,), name='rick'))
    model.add(Dense(3, name='morty'))
    model.compile(loss=custom_loss, optimizer=custom_opt(), metrics=['acc'])

    x = np.random.random((1, 3))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    old_weights = [layer.get_weights() for layer in model.layers]
    _, fname = tempfile.mkstemp('.h5')

    model.save_weights(fname)

    # delete and recreate model
    del(model)
    model = Sequential()
    model.add(Dense(2, input_shape=(3,), name='rick'))
    model.add(Dense(4, name='morty'))  # different shape w.r.t. previous model
    model.compile(loss=custom_loss, optimizer=custom_opt(), metrics=['acc'])

    # load weights from first model
    with pytest.warns(UserWarning):  # expect UserWarning for skipping weights
        model.load_weights(fname, by_name=True, skip_mismatch=True)
    os.remove(fname)

    # assert layers 'rick' are equal
    for old, new in zip(old_weights[0], model.layers[0].get_weights()):
        assert_allclose(old, new, atol=1e-05)

    # assert layers 'morty' are not equal, since we skipped loading this layer
    for old, new in zip(old_weights[1], model.layers[1].get_weights()):
        assert_raises(AssertionError, assert_allclose, old, new, atol=1e-05)


# a function to be called from the Lambda layer
def square_fn(x):
    return x * x


@keras_test
def test_saving_lambda_custom_objects():
    inputs = Input(shape=(3,))
    x = Lambda(lambda x: square_fn(x), output_shape=(3,))(inputs)
    outputs = Dense(3)(x)

    model = Model(inputs, outputs)
    model.compile(loss=losses.MSE,
                  optimizer=optimizers.RMSprop(lr=0.0001),
                  metrics=[metrics.categorical_accuracy])
    x = np.random.random((1, 3))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)

    model = load_model(fname, custom_objects={'square_fn': square_fn})
    os.remove(fname)

    out2 = model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


@keras_test
def test_saving_lambda_numpy_array_arguments():
    mean = np.random.random((4, 2, 3))
    std = np.abs(np.random.random((4, 2, 3))) + 1e-5
    inputs = Input(shape=(4, 2, 3))
    outputs = Lambda(lambda image, mu, std: (image - mu) / std,
                     arguments={'mu': mean, 'std': std})(inputs)
    model = Model(inputs, outputs)
    model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)

    model = load_model(fname)
    os.remove(fname)

    assert_allclose(mean, model.layers[1].arguments['mu'])
    assert_allclose(std, model.layers[1].arguments['std'])


@keras_test
def test_saving_custom_activation_function():
    x = Input(shape=(3,))
    output = Dense(3, activation=K.cos)(x)

    model = Model(x, output)
    model.compile(loss=losses.MSE,
                  optimizer=optimizers.RMSprop(lr=0.0001),
                  metrics=[metrics.categorical_accuracy])
    x = np.random.random((1, 3))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)

    model = load_model(fname, custom_objects={'cos': K.cos})
    os.remove(fname)

    out2 = model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


@keras_test
def test_saving_model_with_long_layer_names():
    # This layer name will make the `layers_name` HDF5 attribute blow
    # out of proportion. Note that it fits into the internal HDF5
    # attribute memory limit on its own but because h5py converts
    # the list of layer names into numpy array, which uses the same
    # amout of memory for every item, it increases the memory
    # requirements substantially.
    x = Input(shape=(2,), name='input_' + ('x' * (2**15)))
    f = x
    for i in range(4):
        f = Dense(2, name='dense_%d' % (i,))(f)

    model = Model(inputs=[x], outputs=[f])

    model.compile(loss='mse', optimizer='adam', metrics=['acc'])

    x = np.random.random((1, 2))
    y = np.random.random((1, 2))
    model.train_on_batch(x, y)

    out = model.predict(x)

    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)

    model = load_model(fname)

    # Check that the HDF5 files contains chunked array
    # of layer names.
    with h5py.File(fname, 'r') as h5file:
        n_layer_names_arrays = len([attr for attr in h5file['model_weights'].attrs
                                    if attr.startswith('layer_names')])

    os.remove(fname)

    # The chunking of layer names array should have happened.
    assert n_layer_names_arrays > 0

    out2 = model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


@keras_test
def test_saving_model_with_long_weights_names():
    x = Input(shape=(2,), name='nested_model_input')
    f = x
    for i in range(4):
        f = Dense(2, name='nested_model_dense_%d' % (i,))(f)
    # This layer name will make the `weights_name`
    # HDF5 attribute blow out of proportion.
    f = Dense(2, name='nested_model_output' + ('x' * (2**15)))(f)
    nested_model = Model(inputs=[x], outputs=[f], name='nested_model')

    x = Input(shape=(2,), name='outer_model_input')
    f = nested_model(x)
    f = Dense(2, name='outer_model_output')(f)

    model = Model(inputs=[x], outputs=[f])

    model.compile(loss='mse', optimizer='adam', metrics=['acc'])

    x = np.random.random((1, 2))
    y = np.random.random((1, 2))
    model.train_on_batch(x, y)

    out = model.predict(x)

    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)

    model = load_model(fname)

    # Check that the HDF5 files contains chunked array
    # of weight names.
    with h5py.File(fname, 'r') as h5file:
        n_weight_names_arrays = len([attr for attr in h5file['model_weights']['nested_model'].attrs
                                     if attr.startswith('weight_names')])

    os.remove(fname)

    # The chunking of layer names array should have happened.
    assert n_weight_names_arrays > 0

    out2 = model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


@keras_test
def test_saving_recurrent_layer_with_init_state():
    vector_size = 8
    input_length = 20

    input_initial_state = Input(shape=(vector_size,))
    input_x = Input(shape=(input_length, vector_size))

    lstm = LSTM(vector_size, return_sequences=True)(
        input_x, initial_state=[input_initial_state, input_initial_state])

    model = Model(inputs=[input_x, input_initial_state], outputs=[lstm])

    _, fname = tempfile.mkstemp('.h5')
    model.save(fname)

    loaded_model = load_model(fname)
    os.remove(fname)


@keras_test
def test_saving_recurrent_layer_without_bias():
    vector_size = 8
    input_length = 20

    input_x = Input(shape=(input_length, vector_size))
    lstm = LSTM(vector_size, use_bias=False)(input_x)
    model = Model(inputs=[input_x], outputs=[lstm])

    _, fname = tempfile.mkstemp('.h5')
    model.save(fname)

    loaded_model = load_model(fname)
    os.remove(fname)


@keras_test
def test_saving_constant_initializer_with_numpy():
    """Test saving and loading model of constant initializer with numpy ndarray as input.
    """
    model = Sequential()
    model.add(Dense(2, input_shape=(3,), kernel_initializer=Constant(np.ones((3, 2)))))
    model.add(Dense(3))
    model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)
    model = load_model(fname)
    os.remove(fname)


@keras_test
@pytest.mark.parametrize('implementation', [1, 2], ids=['impl1', 'impl2'])
@pytest.mark.parametrize('bidirectional', [False, True], ids=['single', 'bidirectional'])
@pytest.mark.parametrize('to_cudnn', [False, True], ids=['from_cudnn', 'to_cudnn'])
@pytest.mark.parametrize('rnn_type', ['LSTM', 'GRU'], ids=['LSTM', 'GRU'])
@pytest.mark.parametrize('model_nest_level', [1, 2], ids=['model_plain', 'model_nested'])
@pytest.mark.parametrize('model_type', ['func', 'seq'], ids=['model_func', 'model_seq'])
@skipif_no_tf_gpu
def test_load_weights_between_noncudnn_rnn(rnn_type, to_cudnn, bidirectional, implementation,
                                           model_nest_level, model_type):
    input_size = 10
    timesteps = 6
    input_shape = (timesteps, input_size)
    units = 2
    num_samples = 32
    inputs = np.random.random((num_samples, timesteps, input_size))

    rnn_layer_kwargs = {
        'recurrent_activation': 'sigmoid',
        # ensure biases are non-zero and properly converted
        'bias_initializer': 'random_uniform',
        'implementation': implementation
    }
    if rnn_type == 'LSTM':
        rnn_layer_class = LSTM
        cudnn_rnn_layer_class = CuDNNLSTM
    else:
        rnn_layer_class = GRU
        cudnn_rnn_layer_class = CuDNNGRU
        rnn_layer_kwargs['reset_after'] = True

    layer = rnn_layer_class(units, **rnn_layer_kwargs)
    if bidirectional:
        layer = Bidirectional(layer)

    cudnn_layer = cudnn_rnn_layer_class(units)
    if bidirectional:
        cudnn_layer = Bidirectional(cudnn_layer)

    model = _make_nested_model(input_shape, layer, model_nest_level, model_type)
    cudnn_model = _make_nested_model(input_shape, cudnn_layer, model_nest_level, model_type)

    if to_cudnn:
        _convert_model_weights(model, cudnn_model)
    else:
        _convert_model_weights(cudnn_model, model)

    assert_allclose(model.predict(inputs), cudnn_model.predict(inputs), atol=1e-4)


def _make_nested_model(input_shape, layer, level=1, model_type='func'):
    # example: make_nested_seq_model((1,), Dense(10), level=2).summary()
    def make_nested_seq_model(input_shape, layer, level=1):
        model = layer
        for i in range(1, level + 1):
            layers = [InputLayer(input_shape), model] if (i == 1) else [model]
            model = Sequential(layers)
        return model

    # example: make_nested_func_model((1,), Dense(10), level=2).summary()
    def make_nested_func_model(input_shape, layer, level=1):
        input = Input(input_shape)
        model = layer
        for i in range(level):
            model = Model(input, model(input))
        return model

    if model_type == 'func':
        return make_nested_func_model(input_shape, layer, level)
    elif model_type == 'seq':
        return make_nested_seq_model(input_shape, layer, level)


def _convert_model_weights(source_model, target_model):
    _, fname = tempfile.mkstemp('.h5')
    source_model.save_weights(fname)
    target_model.load_weights(fname)
    os.remove(fname)


@keras_test
@pytest.mark.parametrize('to_cudnn', [False, True], ids=['from_cudnn', 'to_cudnn'])
@pytest.mark.parametrize('rnn_type', ['LSTM', 'GRU'], ids=['LSTM', 'GRU'])
@skipif_no_tf_gpu
def test_load_weights_between_noncudnn_rnn_time_distributed(rnn_type, to_cudnn):
    """
    Similar test as  test_load_weights_between_noncudnn_rnn() but has different
    rank of input due to usage of TimeDistributed. Issue: #10356.
    """
    input_size = 10
    steps = 6
    timesteps = 6
    input_shape = (timesteps, steps, input_size)
    units = 2
    num_samples = 32
    inputs = np.random.random((num_samples,) + input_shape)

    rnn_layer_kwargs = {
        'recurrent_activation': 'sigmoid',
        # ensure biases are non-zero and properly converted
        'bias_initializer': 'random_uniform',
    }
    if rnn_type == 'LSTM':
        rnn_layer_class = LSTM
        cudnn_rnn_layer_class = CuDNNLSTM
    else:
        rnn_layer_class = GRU
        cudnn_rnn_layer_class = CuDNNGRU
        rnn_layer_kwargs['reset_after'] = True

    layer = rnn_layer_class(units, **rnn_layer_kwargs)
    layer = TimeDistributed(layer)

    cudnn_layer = cudnn_rnn_layer_class(units)
    cudnn_layer = TimeDistributed(cudnn_layer)

    model = _make_nested_model(input_shape, layer)
    cudnn_model = _make_nested_model(input_shape, cudnn_layer)

    if to_cudnn:
        _convert_model_weights(model, cudnn_model)
    else:
        _convert_model_weights(cudnn_model, model)

    assert_allclose(model.predict(inputs), cudnn_model.predict(inputs), atol=1e-4)


@skipif_no_tf_gpu
def test_preprocess_weights_for_loading_gru_incompatible():
    """
    Loading weights between incompatible layers should fail fast with an exception.
    """
    def gru(cudnn=False, **kwargs):
        layer_class = CuDNNGRU if cudnn else GRU
        return layer_class(2, input_shape=[3, 5], **kwargs)

    def initialize_weights(layer):
        # A model is needed to initialize weights.
        _ = Sequential([layer])
        return layer

    def assert_not_compatible(src, dest, message):
        with pytest.raises(ValueError) as ex:
            preprocess_weights_for_loading(dest, initialize_weights(src).get_weights())
        assert message in ex.value.message

    assert_not_compatible(gru(), gru(cudnn=True),
                          'GRU(reset_after=False) is not compatible with CuDNNGRU')
    assert_not_compatible(gru(cudnn=True), gru(),
                          'CuDNNGRU is not compatible with GRU(reset_after=False)')
    assert_not_compatible(gru(), gru(reset_after=True),
                          'GRU(reset_after=False) is not compatible with GRU(reset_after=True)')
    assert_not_compatible(gru(reset_after=True), gru(),
                          'GRU(reset_after=True) is not compatible with GRU(reset_after=False)')


if __name__ == '__main__':
    pytest.main([__file__])
