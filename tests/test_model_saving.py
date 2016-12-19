import pytest
import os
import tempfile
import numpy as np
from numpy.testing import assert_allclose

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Lambda, RepeatVector, TimeDistributed
from keras.layers import Input
from keras import optimizers
from keras import objectives
from keras import metrics
from keras.utils.test_utils import keras_test
from keras.models import save_model, load_model


@keras_test
def test_sequential_model_saving():
    model = Sequential()
    model.add(Dense(2, input_dim=3))
    model.add(RepeatVector(3))
    model.add(TimeDistributed(Dense(3)))
    model.compile(loss=objectives.MSE,
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
    custom_loss = objectives.mse
    model = Sequential()
    model.add(Dense(2, input_dim=3))
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
def test_fuctional_model_saving():
    input = Input(shape=(3,))
    x = Dense(2)(input)
    output = Dense(3)(x)

    model = Model(input, output)
    model.compile(loss=objectives.MSE,
                  optimizer=optimizers.RMSprop(lr=0.0001),
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
def test_saving_without_compilation():
    model = Sequential()
    model.add(Dense(2, input_dim=3))
    model.add(Dense(3))
    model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)
    model = load_model(fname)
    os.remove(fname)


@keras_test
def test_saving_right_after_compilation():
    model = Sequential()
    model.add(Dense(2, input_dim=3))
    model.add(Dense(3))
    model.compile(loss='mse', optimizer='sgd', metrics=['acc'])
    model.model._make_train_function()

    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)
    model = load_model(fname)
    os.remove(fname)


@keras_test
def test_loading_weights_by_name():
    """
    test loading model weights by name on:
        - sequential model
    """

    # test with custom optimizer, loss
    custom_opt = optimizers.rmsprop
    custom_loss = objectives.mse

    # sequential model
    model = Sequential()
    model.add(Dense(2, input_dim=3, name="rick"))
    model.add(Dense(3, name="morty"))
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
    model.add(Dense(2, input_dim=3, name="rick"))
    model.add(Dense(3, name="morty"))
    model.compile(loss=custom_loss, optimizer=custom_opt(), metrics=['acc'])

    # load weights from first model
    model.load_weights(fname, by_name=True)
    os.remove(fname)

    out2 = model.predict(x)
    assert_allclose(out, out2, atol=1e-05)
    for i in range(len(model.layers)):
        new_weights = model.layers[i].get_weights()
        for j in range(len(new_weights)):
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
    custom_loss = objectives.mse

    # sequential model
    model = Sequential()
    model.add(Dense(2, input_dim=3, name="rick"))
    model.add(Dense(3, name="morty"))
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
    rick = Dense(2, name="rick")(data)
    jerry = Dense(3, name="jerry")(rick)  # add 2 layers (but maintain shapes)
    jessica = Dense(2, name="jessica")(jerry)
    morty = Dense(3, name="morty")(jessica)

    model = Model(input=[data], output=[morty])
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


# a function to be called from the Lambda layer
def square_fn(x):
    return x * x


@keras_test
def test_saving_lambda_custom_objects():
    input = Input(shape=(3,))
    x = Lambda(lambda x: square_fn(x), output_shape=(3,))(input)
    output = Dense(3)(x)

    model = Model(input, output)
    model.compile(loss=objectives.MSE,
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


if __name__ == '__main__':
    pytest.main([__file__])
