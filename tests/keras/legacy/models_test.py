from __future__ import absolute_import
from __future__ import print_function
import pytest
import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.legacy.layers import Merge
from keras.utils import np_utils
from keras.utils.test_utils import get_test_data, keras_test
from keras.models import model_from_json, model_from_yaml


input_dim = 16
num_hidden = 8
num_classes = 4
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


def _get_test_data():
    np.random.seed(1234)

    train_samples = 100
    test_samples = 50

    (x_train, y_train), (x_test, y_test) = get_test_data(num_train=train_samples,
                                                         num_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         num_classes=num_classes)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    return (x_train, y_train), (x_test, y_test)


@keras_test
def test_merge_sum(in_tmpdir):
    (x_train, y_train), (x_test, y_test) = _get_test_data()
    left = Sequential()
    left.add(Dense(num_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Dense(num_hidden, input_shape=(input_dim,)))
    right.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([left, right], mode='sum'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit([x_train, x_train], y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=([x_test, x_test], y_test))
    model.fit([x_train, x_train], y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_split=0.1)
    model.fit([x_train, x_train], y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    model.fit([x_train, x_train], y_train, batch_size=batch_size, epochs=epochs, verbose=0, shuffle=False)

    loss = model.evaluate([x_test, x_test], y_test, verbose=0)

    model.predict([x_test, x_test], verbose=0)
    model.predict_classes([x_test, x_test], verbose=0)
    model.predict_proba([x_test, x_test], verbose=0)

    # test weight saving
    fname = 'test_merge_sum_temp.h5'
    model.save_weights(fname, overwrite=True)
    left = Sequential()
    left.add(Dense(num_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))
    right = Sequential()
    right.add(Dense(num_hidden, input_shape=(input_dim,)))
    right.add(Activation('relu'))
    model = Sequential()
    model.add(Merge([left, right], mode='sum'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.load_weights(fname)
    os.remove(fname)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    nloss = model.evaluate([x_test, x_test], y_test, verbose=0)
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
def test_merge_dot():
    (x_train, y_train), (x_test, y_test) = _get_test_data()

    left = Sequential()
    left.add(Dense(num_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Dense(num_hidden, input_shape=(input_dim,)))
    right.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([left, right], mode='dot', dot_axes=1))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    left = Sequential()
    left.add(Dense(num_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Dense(num_hidden, input_shape=(input_dim,)))
    right.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([left, right], mode='dot', dot_axes=[1, 1]))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


@keras_test
def test_merge_concat(in_tmpdir):
    (x_train, y_train), (x_test, y_test) = _get_test_data()

    left = Sequential(name='branch_1')
    left.add(Dense(num_hidden, input_shape=(input_dim,), name='dense_1'))
    left.add(Activation('relu', name='relu_1'))

    right = Sequential(name='branch_2')
    right.add(Dense(num_hidden, input_shape=(input_dim,), name='dense_2'))
    right.add(Activation('relu', name='relu_2'))

    model = Sequential(name='merged_branches')
    model.add(Merge([left, right], mode='concat', name='merge'))
    model.add(Dense(num_classes, name='final_dense'))
    model.add(Activation('softmax', name='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit([x_train, x_train], y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=([x_test, x_test], y_test))
    model.fit([x_train, x_train], y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_split=0.1)
    model.fit([x_train, x_train], y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    model.fit([x_train, x_train], y_train, batch_size=batch_size, epochs=epochs, verbose=0, shuffle=False)

    loss = model.evaluate([x_test, x_test], y_test, verbose=0)

    model.predict([x_test, x_test], verbose=0)
    model.predict_classes([x_test, x_test], verbose=0)
    model.predict_proba([x_test, x_test], verbose=0)
    model.get_config()

    fname = 'test_merge_concat_temp.h5'
    model.save_weights(fname, overwrite=True)
    model.fit([x_train, x_train], y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    model.load_weights(fname)
    os.remove(fname)

    nloss = model.evaluate([x_test, x_test], y_test, verbose=0)
    assert(loss == nloss)


@keras_test
def test_merge_recursivity(in_tmpdir):
    (x_train, y_train), (x_test, y_test) = _get_test_data()
    left = Sequential()
    left.add(Dense(num_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Dense(num_hidden, input_shape=(input_dim,)))
    right.add(Activation('relu'))

    righter = Sequential()
    righter.add(Dense(num_hidden, input_shape=(input_dim,)))
    righter.add(Activation('relu'))

    intermediate = Sequential()
    intermediate.add(Merge([left, right], mode='sum'))
    intermediate.add(Dense(num_hidden))
    intermediate.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([intermediate, righter], mode='sum'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit([x_train, x_train, x_train], y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=([x_test, x_test, x_test], y_test))
    model.fit([x_train, x_train, x_train], y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_split=0.1)
    model.fit([x_train, x_train, x_train], y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    model.fit([x_train, x_train, x_train], y_train, batch_size=batch_size, epochs=epochs, verbose=0, shuffle=False)

    loss = model.evaluate([x_test, x_test, x_test], y_test, verbose=0)

    model.predict([x_test, x_test, x_test], verbose=0)
    model.predict_classes([x_test, x_test, x_test], verbose=0)
    model.predict_proba([x_test, x_test, x_test], verbose=0)

    fname = 'test_merge_recursivity_temp.h5'
    model.save_weights(fname, overwrite=True)
    model.load_weights(fname)
    os.remove(fname)

    nloss = model.evaluate([x_test, x_test, x_test], y_test, verbose=0)
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
def test_merge_overlap(in_tmpdir):
    (x_train, y_train), (x_test, y_test) = _get_test_data()
    left = Sequential()
    left.add(Dense(num_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([left, left], mode='sum'))
    model.add(Dense(num_classes))
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

    fname = 'test_merge_overlap_temp.h5'
    print(model.layers)
    model.save_weights(fname, overwrite=True)
    print(model.trainable_weights)

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


if __name__ == '__main__':
    pytest.main([__file__])
