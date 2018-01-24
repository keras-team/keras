import pytest
import os
import tempfile
import numpy as np
from numpy.testing import assert_allclose, assert_raises

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Lambda, RepeatVector, TimeDistributed, LSTM
from keras.layers import Conv2D, Flatten
from keras.layers import Input
from keras import optimizers
from keras import losses
from keras import metrics
from keras.utils.test_utils import keras_test
from keras.models import save_model, load_model


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
    model.model._make_train_function()

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
    os.remove(fname)

    out2 = model.predict(x)
    assert_allclose(np.squeeze(out), np.squeeze(out2), atol=1e-05)
    for i in range(len(model.layers)):
        new_weights = model.layers[i].get_weights()
        for j in range(len(new_weights)):
            # only compare layers that have weights, skipping Flatten()
            if old_weights[i]:
                assert_allclose(old_weights[i][j], new_weights[j], atol=1e-05)


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


if __name__ == '__main__':
    pytest.main([__file__])
