import pytest
import os
import numpy as np
from numpy.testing import assert_allclose

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, RepeatVector, TimeDistributed
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
    model.add(Dense(3))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])

    x = np.random.random((1, 3))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    fname = 'tmp_' + str(np.random.randint(10000)) + '.h5'
    save_model(model, fname)

    new_model = load_model(fname)

    out2 = new_model.predict(x)
    assert_allclose(out, out2, atol=1e-05)

    # test that new updates are the same with both models
    x = np.random.random((1, 3))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)
    new_model.train_on_batch(x, y)
    out = model.predict(x)
    out2 = new_model.predict(x)
    assert_allclose(out, out2, atol=1e-05)

    # test load_weights on model file
    model.load_weights(fname)
    os.remove(fname)


@keras_test
def test_sequential_model_saving_2():
    # test with funkier config
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
    fname = 'tmp_' + str(np.random.randint(10000)) + '.h5'
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
def test_sequential_model_saving_3():
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
    fname = 'tmp_' + str(np.random.randint(10000)) + '.h5'
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
    fname = 'tmp_' + str(np.random.randint(10000)) + '.h5'
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

    fname = 'tmp_' + str(np.random.randint(10000)) + '.h5'
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

    fname = 'tmp_' + str(np.random.randint(10000)) + '.h5'
    save_model(model, fname)
    model = load_model(fname)
    os.remove(fname)


if __name__ == '__main__':
    pytest.main([__file__])
