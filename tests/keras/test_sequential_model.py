from __future__ import absolute_import
from __future__ import print_function
import pytest
import os
import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.utils.test_utils import get_test_data, keras_test
from keras.models import model_from_json, model_from_yaml
from keras import losses
from keras.engine.training import _make_batches


input_dim = 16
num_hidden = 8
num_class = 4
batch_size = 32
epochs = 1


@pytest.fixture
def in_tmpdir(tmpdir):
    """Runs a function in a temporary directory.

    Checks that the directory is empty afterwards.
    """
    with tmpdir.as_cwd():
        yield None
    assert not tmpdir.listdir()


@keras_test
def test_sequential_pop():
    model = Sequential()
    model.add(Dense(num_hidden, input_dim=input_dim))
    model.add(Dense(num_class))
    model.compile(loss='mse', optimizer='sgd')
    x = np.random.random((batch_size, input_dim))
    y = np.random.random((batch_size, num_class))
    model.fit(x, y, epochs=1)
    model.pop()
    assert len(model.layers) == 1
    assert model.output_shape == (None, num_hidden)
    model.compile(loss='mse', optimizer='sgd')
    y = np.random.random((batch_size, num_hidden))
    model.fit(x, y, epochs=1)


def _get_test_data():
    np.random.seed(1234)

    train_samples = 100
    test_samples = 50

    (x_train, y_train), (x_test, y_test) = get_test_data(num_train=train_samples,
                                                         num_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         num_classes=4)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    return (x_train, y_train), (x_test, y_test)


@keras_test
def test_sequential_fit_generator():
    (x_train, y_train), (x_test, y_test) = _get_test_data()

    def data_generator(train):
        if train:
            max_batch_index = len(x_train) // batch_size
        else:
            max_batch_index = len(x_test) // batch_size
        i = 0
        while 1:
            if train:
                yield (x_train[i * batch_size: (i + 1) * batch_size], y_train[i * batch_size: (i + 1) * batch_size])
            else:
                yield (x_test[i * batch_size: (i + 1) * batch_size], y_test[i * batch_size: (i + 1) * batch_size])
            i += 1
            i = i % max_batch_index

    model = Sequential()
    model.add(Dense(num_hidden, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(num_class))
    model.pop()
    model.add(Dense(num_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit_generator(data_generator(True), 5, epochs)
    model.fit_generator(data_generator(True), 5, epochs,
                        validation_data=(x_test, y_test))
    model.fit_generator(data_generator(True), 5, epochs,
                        validation_data=data_generator(False),
                        validation_steps=3)
    model.fit_generator(data_generator(True), 5, epochs, max_queue_size=2)
    model.evaluate(x_train, y_train)


@keras_test
def test_sequential(in_tmpdir):
    (x_train, y_train), (x_test, y_test) = _get_test_data()

    # TODO: factor out
    def data_generator(x, y, batch_size=50):
        index_array = np.arange(len(x))
        while 1:
            batches = _make_batches(len(x_test), batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                x_batch = x[batch_ids]
                y_batch = y[batch_ids]
                yield (x_batch, y_batch)

    model = Sequential()
    model.add(Dense(num_hidden, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(num_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.1)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=False)

    model.train_on_batch(x_train[:32], y_train[:32])

    loss = model.evaluate(x_test, y_test)

    prediction = model.predict_generator(data_generator(x_test, y_test), 1, max_queue_size=2, verbose=1)
    gen_loss = model.evaluate_generator(data_generator(x_test, y_test, 50), 1, max_queue_size=2)
    pred_loss = K.eval(K.mean(losses.get(model.loss)(K.variable(y_test), K.variable(prediction))))

    assert(np.isclose(pred_loss, loss))
    assert(np.isclose(gen_loss, loss))

    model.predict(x_test, verbose=0)
    model.predict_classes(x_test, verbose=0)
    model.predict_proba(x_test, verbose=0)

    fname = 'test_sequential_temp.h5'
    model.save_weights(fname, overwrite=True)
    model = Sequential()
    model.add(Dense(num_hidden, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(num_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.load_weights(fname)
    os.remove(fname)

    nloss = model.evaluate(x_test, y_test, verbose=0)
    assert(loss == nloss)

    # test serialization
    config = model.get_config()
    Sequential.from_config(config)

    model.summary()
    json_str = model.to_json()
    model_from_json(json_str)

    yaml_str = model.to_yaml()
    model_from_yaml(yaml_str)


@keras_test
def test_nested_sequential(in_tmpdir):
    (x_train, y_train), (x_test, y_test) = _get_test_data()

    inner = Sequential()
    inner.add(Dense(num_hidden, input_shape=(input_dim,)))
    inner.add(Activation('relu'))
    inner.add(Dense(num_class))

    middle = Sequential()
    middle.add(inner)

    model = Sequential()
    model.add(middle)
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.1)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=False)

    model.train_on_batch(x_train[:32], y_train[:32])

    loss = model.evaluate(x_test, y_test, verbose=0)

    model.predict(x_test, verbose=0)
    model.predict_classes(x_test, verbose=0)
    model.predict_proba(x_test, verbose=0)

    fname = 'test_nested_sequential_temp.h5'
    model.save_weights(fname, overwrite=True)

    inner = Sequential()
    inner.add(Dense(num_hidden, input_shape=(input_dim,)))
    inner.add(Activation('relu'))
    inner.add(Dense(num_class))

    middle = Sequential()
    middle.add(inner)

    model = Sequential()
    model.add(middle)
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.load_weights(fname)
    os.remove(fname)

    nloss = model.evaluate(x_test, y_test, verbose=0)
    assert(loss == nloss)

    # test serialization
    config = model.get_config()
    Sequential.from_config(config)

    model.summary()
    json_str = model.to_json()
    model_from_json(json_str)

    yaml_str = model.to_yaml()
    model_from_yaml(yaml_str)


@keras_test
def test_sequential_count_params():
    input_dim = 20
    num_units = 10
    num_classes = 2

    n = input_dim * num_units + num_units
    n += num_units * num_units + num_units
    n += num_units * num_classes + num_classes

    model = Sequential()
    model.add(Dense(num_units, input_shape=(input_dim,)))
    model.add(Dense(num_units))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.build()

    assert(n == model.count_params())

    model.compile('sgd', 'binary_crossentropy')
    assert(n == model.count_params())


@keras_test
def test_rebuild_model():
    model = Sequential()
    model.add(Dense(128, input_shape=(784,)))
    model.add(Dense(64))
    assert(model.get_layer(index=-1).output_shape == (None, 64))

    model.add(Dense(32))
    assert(model.get_layer(index=-1).output_shape == (None, 32))


if __name__ == '__main__':
    pytest.main([__file__])
