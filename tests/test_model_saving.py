import io
import pytest
import os
import h5py
import tempfile
import warnings
from contextlib import contextmanager
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_raises

from keras import backend as K
from keras.engine.saving import preprocess_weights_for_loading
from keras.models import Model, Sequential
from keras.layers import Dense, Lambda, RepeatVector, TimeDistributed
from keras.layers import Bidirectional, GRU, LSTM, CuDNNGRU, CuDNNLSTM
from keras.layers import Conv2D, Flatten
from keras.layers import Input, InputLayer
from keras.initializers import Constant
from keras import optimizers
from keras import losses
from keras import metrics
from keras.models import save_model, load_model
from keras.utils.test_utils import tf_file_io_proxy
try:
    from unittest.mock import patch
except:
    from mock import patch


skipif_no_tf_gpu = pytest.mark.skipif(
    (K.backend() != 'tensorflow' or
     not K.tensorflow_backend._get_available_gpus()),
    reason='Requires TensorFlow backend and a GPU')


def test_sequential_model_saving():
    model = Sequential()
    model.add(Dense(2, input_shape=(3,)))
    model.add(RepeatVector(3))
    model.add(TimeDistributed(Dense(3)))
    model.compile(loss=losses.MeanSquaredError(),
                  optimizer=optimizers.RMSprop(lr=0.0001),
                  metrics=[metrics.categorical_accuracy],
                  sample_weight_mode='temporal')
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)

    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)
    new_model_disk = load_model(fname)
    os.remove(fname)

    with tf_file_io_proxy('keras.engine.saving.tf_file_io') as file_io_proxy:
        gcs_filepath = file_io_proxy.get_filepath(filename=fname)
        save_model(model, gcs_filepath)
        file_io_proxy.assert_exists(gcs_filepath)
        new_model_gcs = load_model(gcs_filepath)
        file_io_proxy.delete_file(gcs_filepath)  # cleanup

    x2 = np.random.random((1, 3))
    y2 = np.random.random((1, 3, 3))
    model.train_on_batch(x2, y2)
    out_2 = model.predict(x2)

    for new_model in [new_model_disk, new_model_gcs]:
        new_out = new_model.predict(x)
        assert_allclose(out, new_out, atol=1e-05)
        # test that new updates are the same with both models
        new_model.train_on_batch(x2, y2)
        new_out_2 = new_model.predict(x2)
        assert_allclose(out_2, new_out_2, atol=1e-05)


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

    load_kwargs = {'custom_objects': {'custom_opt': custom_opt,
                                      'custom_loss': custom_loss}}
    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)
    new_model_disk = load_model(fname, **load_kwargs)
    os.remove(fname)

    with tf_file_io_proxy('keras.engine.saving.tf_file_io') as file_io_proxy:
        gcs_filepath = file_io_proxy.get_filepath(filename=fname)
        save_model(model, gcs_filepath)
        file_io_proxy.assert_exists(gcs_filepath)
        new_model_gcs = load_model(gcs_filepath, **load_kwargs)
        file_io_proxy.delete_file(gcs_filepath)  # cleanup

    for new_model in [new_model_disk, new_model_gcs]:
        new_out = new_model.predict(x)
        assert_allclose(out, new_out, atol=1e-05)


def _get_sample_model_and_input():
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

    return model, x


def test_functional_model_saving():
    model, x = _get_sample_model_and_input()
    out = model.predict(x)
    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)
    new_model_disk = load_model(fname)
    os.remove(fname)

    with tf_file_io_proxy('keras.engine.saving.tf_file_io') as file_io_proxy:
        gcs_filepath = file_io_proxy.get_filepath(filename=fname)
        save_model(model, gcs_filepath)
        file_io_proxy.assert_exists(gcs_filepath)
        new_model_gcs = load_model(gcs_filepath)
        file_io_proxy.delete_file(gcs_filepath)  # cleanup

    for new_model in [new_model_disk, new_model_gcs]:
        new_out = new_model.predict(x)
        assert_allclose(out, new_out, atol=1e-05)


def test_model_saving_to_pre_created_h5py_file():
    model, x = _get_sample_model_and_input()

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

    with h5py.File(fname, mode='r+') as h5file:
        g = h5file.create_group('model')
        save_model(model, g)
        loaded_model = load_model(g)
        out2 = loaded_model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


@contextmanager
def temp_filename(filename):
    """Context that returns a temporary filename and deletes the file on exit if
    it still exists (so that this is not forgotten).
    """
    _, temp_fname = tempfile.mkstemp(filename)
    yield temp_fname
    if os.path.exists(temp_fname):
        os.remove(temp_fname)


def test_model_saving_to_binary_stream():
    model, x = _get_sample_model_and_input()
    out = model.predict(x)

    with temp_filename('h5') as fname:
        # save directly to binary file
        with open(fname, 'wb') as raw_file:
            save_model(model, raw_file)
        # Load the data the usual way, and make sure the model is intact.
        with h5py.File(fname, mode='r') as h5file:
            loaded_model = load_model(h5file)
    out2 = loaded_model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


def test_model_loading_from_binary_stream():
    model, x = _get_sample_model_and_input()
    out = model.predict(x)

    with temp_filename('h5') as fname:
        # save the model the usual way
        with h5py.File(fname, mode='w') as h5file:
            save_model(model, h5file)
        # Load the data binary, and make sure the model is intact.
        with open(fname, 'rb') as raw_file:
            loaded_model = load_model(raw_file)
    out2 = loaded_model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


def test_model_save_load_binary_in_memory():
    model, x = _get_sample_model_and_input()
    out = model.predict(x)

    stream = io.BytesIO()
    save_model(model, stream)
    stream.seek(0)
    loaded_model = load_model(stream)
    out2 = loaded_model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


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


def test_saving_unused_layers_is_ok():
    a = Input(shape=(256, 512, 6))
    b = Input(shape=(256, 512, 1))
    c = Lambda(lambda x: x[:, :, :, :1])(a)

    model = Model(inputs=[a, b], outputs=c)

    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)
    load_model(fname)
    os.remove(fname)


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


def test_saving_model_with_long_weights_names():
    x = Input(shape=(2,), name='nested_model_input')
    f = x
    for i in range(4):
        f = Dense(2, name='nested_model_dense_%d' % (i,))(f)
    f = Dense(2, name='nested_model_dense_4', trainable=False)(f)
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
        attrs = [attr for attr in h5file['model_weights']['nested_model'].attrs
                 if attr.startswith('weight_names')]
        n_weight_names_arrays = len(attrs)

    os.remove(fname)

    # The chunking of layer names array should have happened.
    assert n_weight_names_arrays > 0

    out2 = model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


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


def test_loop_model_saving():
    model = Sequential()
    model.add(Dense(2, input_shape=(3,)))
    model.compile(loss=losses.MSE,
                  optimizer=optimizers.RMSprop(lr=0.0001),
                  metrics=[metrics.categorical_accuracy])

    x = np.random.random((1, 3))
    y = np.random.random((1, 2))
    _, fname = tempfile.mkstemp('.h5')

    for _ in range(3):
        model.train_on_batch(x, y)
        save_model(model, fname, overwrite=True)
        out = model.predict(x)

    new_model = load_model(fname)
    os.remove(fname)

    out2 = new_model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


def test_saving_constant_initializer_with_numpy():
    """Test saving and loading model of constant initializer with numpy inputs.
    """
    model = Sequential()
    model.add(Dense(2, input_shape=(3,),
                    kernel_initializer=Constant(np.ones((3, 2)))))
    model.add(Dense(3))
    model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)
    model = load_model(fname)
    os.remove(fname)


def test_save_load_weights_gcs():
    model = Sequential()
    model.add(Dense(2, input_shape=(3,)))
    org_weights = model.get_weights()

    with tf_file_io_proxy('keras.engine.saving.tf_file_io') as file_io_proxy:
        gcs_filepath = file_io_proxy.get_filepath(
            filename='test_save_load_weights_gcs.h5')
        # we should not use same filename in several tests to allow for parallel
        # execution
        model.save_weights(gcs_filepath)
        model.set_weights([np.random.random(w.shape) for w in org_weights])
        for w, org_w in zip(model.get_weights(), org_weights):
            assert not (w == org_w).all()
        model.load_weights(gcs_filepath)
        for w, org_w in zip(model.get_weights(), org_weights):
            assert_allclose(w, org_w)

        file_io_proxy.delete_file(gcs_filepath)  # cleanup


def test_saving_overwrite_option():
    model = Sequential()
    model.add(Dense(2, input_shape=(3,)))
    org_weights = model.get_weights()
    new_weights = [np.random.random(w.shape) for w in org_weights]

    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)
    model.set_weights(new_weights)

    with patch('keras.engine.saving.ask_to_proceed_with_overwrite') as ask:
        ask.return_value = False
        save_model(model, fname, overwrite=False)
        ask.assert_called_once()
        new_model = load_model(fname)
        for w, org_w in zip(new_model.get_weights(), org_weights):
            assert_allclose(w, org_w)

        ask.return_value = True
        save_model(model, fname, overwrite=False)
        assert ask.call_count == 2
        new_model = load_model(fname)
        for w, new_w in zip(new_model.get_weights(), new_weights):
            assert_allclose(w, new_w)

    os.remove(fname)


def test_saving_overwrite_option_gcs():
    model = Sequential()
    model.add(Dense(2, input_shape=(3,)))
    org_weights = model.get_weights()
    new_weights = [np.random.random(w.shape) for w in org_weights]

    with tf_file_io_proxy('keras.engine.saving.tf_file_io') as file_io_proxy:
        gcs_filepath = file_io_proxy.get_filepath(
            filename='test_saving_overwrite_option_gcs.h5')
        # we should not use same filename in several tests to allow for parallel
        # execution
        save_model(model, gcs_filepath)
        model.set_weights(new_weights)

        with patch('keras.engine.saving.ask_to_proceed_with_overwrite') as ask:
            ask.return_value = False
            save_model(model, gcs_filepath, overwrite=False)
            ask.assert_called_once()
            new_model = load_model(gcs_filepath)
            for w, org_w in zip(new_model.get_weights(), org_weights):
                assert_allclose(w, org_w)

            ask.return_value = True
            save_model(model, gcs_filepath, overwrite=False)
            assert ask.call_count == 2
            new_model = load_model(gcs_filepath)
            for w, new_w in zip(new_model.get_weights(), new_weights):
                assert_allclose(w, new_w)

        file_io_proxy.delete_file(gcs_filepath)  # cleanup


@pytest.mark.parametrize('implementation', [1, 2], ids=['impl1', 'impl2'])
@pytest.mark.parametrize('bidirectional',
                         [False, True],
                         ids=['single', 'bidirectional'])
@pytest.mark.parametrize('to_cudnn', [False, True], ids=['from_cudnn', 'to_cudnn'])
@pytest.mark.parametrize('rnn_type', ['LSTM', 'GRU'], ids=['LSTM', 'GRU'])
@pytest.mark.parametrize('model_nest_level',
                         [1, 2],
                         ids=['model_plain', 'model_nested'])
@pytest.mark.parametrize('model_type',
                         ['func', 'seq'],
                         ids=['model_func', 'model_seq'])
@skipif_no_tf_gpu
def test_load_weights_between_noncudnn_rnn(rnn_type, to_cudnn, bidirectional,
                                           implementation, model_nest_level,
                                           model_type):
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
    cudnn_model = _make_nested_model(input_shape, cudnn_layer,
                                     model_nest_level, model_type)

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
            preprocess_weights_for_loading(dest,
                                           initialize_weights(src).get_weights())
        assert message in ex.value.message

    assert_not_compatible(gru(), gru(cudnn=True),
                          'GRU(reset_after=False) is not compatible with CuDNNGRU')
    assert_not_compatible(gru(cudnn=True), gru(),
                          'CuDNNGRU is not compatible with GRU(reset_after=False)')
    assert_not_compatible(gru(), gru(reset_after=True),
                          'GRU(reset_after=False) is not compatible with '
                          'GRU(reset_after=True)')
    assert_not_compatible(gru(reset_after=True), gru(),
                          'GRU(reset_after=True) is not compatible with '
                          'GRU(reset_after=False)')


def test_model_saving_with_rnn_initial_state_and_args():
    class CustomRNN(LSTM):
        def call(self, inputs, arg=1, mask=None, training=None, initial_state=None):
            if isinstance(inputs, list):
                inputs = inputs[:]
                shape = K.int_shape(inputs[0])
                inputs[0] *= arg
                inputs[0]._keras_shape = shape  # for theano backend
            else:
                shape = K.int_shape(inputs)
                inputs *= arg
                inputs._keras_shape = shape  # for theano backend
            return super(CustomRNN, self).call(inputs, mask, training, initial_state)

    inp = Input((3, 2))
    rnn_out, h, c = CustomRNN(2, return_state=True, return_sequences=True)(inp)
    assert hasattr(rnn_out, '_keras_history')
    assert hasattr(h, '_keras_history')
    assert hasattr(c, '_keras_history')
    rnn2_out = CustomRNN(2)(rnn_out, arg=2, initial_state=[h, c])
    assert hasattr(rnn2_out, '_keras_history')
    model = Model(inputs=inp, outputs=rnn2_out)
    x = np.random.random((2, 3, 2))
    y1 = model.predict(x)
    _, fname = tempfile.mkstemp('.h5')
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        model.save(fname)
    model2 = load_model(fname, custom_objects={'CustomRNN': CustomRNN})
    y2 = model2.predict(x)
    assert_allclose(y1, y2, atol=1e-5)
    os.remove(fname)


if __name__ == '__main__':
    pytest.main([__file__])
