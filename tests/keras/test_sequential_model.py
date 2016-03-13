from __future__ import absolute_import
from __future__ import print_function
import pytest
import os
import numpy as np
np.random.seed(1337)

from keras import backend as K
from keras.models import Graph, Sequential, model_from_json, model_from_yaml
from keras.layers.core import Dense, Activation, Merge, Lambda, LambdaMerge, Siamese, add_shared_layer
from keras.layers import containers
from keras.utils import np_utils
from keras.utils.test_utils import get_test_data


input_dim = 32
nb_hidden = 16
nb_class = 4
batch_size = 32
nb_epoch = 1


def _get_test_data():
    np.random.seed(1234)

    train_samples = 2000
    test_samples = 500

    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=train_samples,
                                                         nb_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         nb_class=4)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    return (X_train, y_train), (X_test, y_test)


def test_sequential_fit_generator():
    (X_train, y_train), (X_test, y_test) = _get_test_data()

    def data_generator(train):
        if train:
            max_batch_index = len(X_train) // batch_size
        else:
            max_batch_index = len(X_test) // batch_size
        i = 0
        while 1:
            if train:
                yield (X_train[i * batch_size: (i + 1) * batch_size], y_train[i * batch_size: (i + 1) * batch_size])
            else:
                yield (X_test[i * batch_size: (i + 1) * batch_size], y_test[i * batch_size: (i + 1) * batch_size])
            i += 1
            i = i % max_batch_index

    model = Sequential()
    model.add(Dense(nb_hidden, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit_generator(data_generator(True), len(X_train), nb_epoch, show_accuracy=False)
    model.fit_generator(data_generator(True), len(X_train), nb_epoch, show_accuracy=True)
    model.fit_generator(data_generator(True), len(X_train), nb_epoch, show_accuracy=False, validation_data=(X_test, y_test))
    model.fit_generator(data_generator(True), len(X_train), nb_epoch, show_accuracy=True, validation_data=(X_test, y_test))
    model.fit_generator(data_generator(True), len(X_train), nb_epoch, show_accuracy=False,
                        validation_data=data_generator(False), nb_val_samples=batch_size * 3)
    model.fit_generator(data_generator(True), len(X_train), nb_epoch, show_accuracy=True,
                        validation_data=data_generator(False), nb_val_samples=batch_size * 3)

    loss = model.evaluate(X_train, y_train, verbose=0)
    assert(loss < 0.9)


def test_sequential():
    (X_train, y_train), (X_test, y_test) = _get_test_data()

    # TODO: factor out
    def data_generator(train):
        if train:
            max_batch_index = len(X_train) // batch_size
        else:
            max_batch_index = len(X_test) // batch_size
        i = 0
        while 1:
            if train:
                yield (X_train[i * batch_size: (i + 1) * batch_size], y_train[i * batch_size: (i + 1) * batch_size])
            else:
                yield (X_test[i * batch_size: (i + 1) * batch_size], y_test[i * batch_size: (i + 1) * batch_size])
            i += 1
            i = i % max_batch_index

    model = Sequential()
    model.add(Dense(nb_hidden, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.summary()

    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, y_test))
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=2, validation_data=(X_test, y_test))
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_split=0.1)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=1, validation_split=0.1)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=False)

    model.train_on_batch(X_train[:32], y_train[:32])

    gen_loss = model.evaluate_generator(data_generator(True), 256, verbose=0)
    assert(gen_loss < 0.8)

    loss = model.evaluate(X_test, y_test, verbose=0)
    assert(loss < 0.8)

    model.predict(X_test, verbose=0)
    model.predict_classes(X_test, verbose=0)
    model.predict_proba(X_test, verbose=0)
    model.get_config(verbose=0)

    fname = 'test_sequential_temp.h5'
    model.save_weights(fname, overwrite=True)
    model = Sequential()
    model.add(Dense(nb_hidden, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.load_weights(fname)
    os.remove(fname)

    nloss = model.evaluate(X_test, y_test, verbose=0)
    assert(loss == nloss)

    # test json serialization
    json_data = model.to_json()
    model = model_from_json(json_data)

    # test yaml serialization
    yaml_data = model.to_yaml()
    model = model_from_yaml(yaml_data)


def test_nested_sequential():
    (X_train, y_train), (X_test, y_test) = _get_test_data()

    inner = Sequential()
    inner.add(Dense(nb_hidden, input_shape=(input_dim,)))
    inner.add(Activation('relu'))
    inner.add(Dense(nb_class))

    middle = Sequential()
    middle.add(inner)

    model = Sequential()
    model.add(middle)
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.summary()

    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, y_test))
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=2, validation_data=(X_test, y_test))
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_split=0.1)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=1, validation_split=0.1)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=False)

    model.train_on_batch(X_train[:32], y_train[:32])

    loss = model.evaluate(X_test, y_test, verbose=0)
    assert(loss < 0.8)

    model.predict(X_test, verbose=0)
    model.predict_classes(X_test, verbose=0)
    model.predict_proba(X_test, verbose=0)
    model.get_config(verbose=0)

    fname = 'test_nested_sequential_temp.h5'
    model.save_weights(fname, overwrite=True)

    inner = Sequential()
    inner.add(Dense(nb_hidden, input_shape=(input_dim,)))
    inner.add(Activation('relu'))
    inner.add(Dense(nb_class))

    middle = Sequential()
    middle.add(inner)

    model = Sequential()
    model.add(middle)
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.load_weights(fname)
    os.remove(fname)

    nloss = model.evaluate(X_test, y_test, verbose=0)
    assert(loss == nloss)

    # test json serialization
    json_data = model.to_json()
    model = model_from_json(json_data)

    # test yaml serialization
    yaml_data = model.to_yaml()
    model = model_from_yaml(yaml_data)


def test_merge_sum():
    (X_train, y_train), (X_test, y_test) = _get_test_data()
    left = Sequential()
    left.add(Dense(nb_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Dense(nb_hidden, input_shape=(input_dim,)))
    right.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([left, right], mode='sum'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_data=([X_test, X_test], y_test))
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=([X_test, X_test], y_test))
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_split=0.1)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_split=0.1)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, shuffle=False)

    loss = model.evaluate([X_test, X_test], y_test, verbose=0)
    assert(loss < 0.8)

    model.predict([X_test, X_test], verbose=0)
    model.predict_classes([X_test, X_test], verbose=0)
    model.predict_proba([X_test, X_test], verbose=0)
    model.get_config(verbose=0)

    # test weight saving
    fname = 'test_merge_sum_temp.h5'
    model.save_weights(fname, overwrite=True)
    left = Sequential()
    left.add(Dense(nb_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))
    right = Sequential()
    right.add(Dense(nb_hidden, input_shape=(input_dim,)))
    right.add(Activation('relu'))
    model = Sequential()
    model.add(Merge([left, right], mode='sum'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    model.load_weights(fname)
    os.remove(fname)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    nloss = model.evaluate([X_test, X_test], y_test, verbose=0)
    assert(loss == nloss)


@pytest.mark.skipif(K._BACKEND == 'tensorflow',
                    reason='currently not working with TensorFlow')
def test_merge_dot():
    (X_train, y_train), (X_test, y_test) = _get_test_data()

    left = Sequential()
    left.add(Dense(input_dim=input_dim, output_dim=nb_hidden))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Dense(input_dim=input_dim, output_dim=nb_hidden))
    right.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([left, right], mode='dot', dot_axes=1))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    left = Sequential()
    left.add(Dense(input_dim=input_dim, output_dim=nb_hidden))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Dense(input_dim=input_dim, output_dim=nb_hidden))
    right.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([left, right], mode='dot', dot_axes=([1], [1])))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


def test_merge_concat():
    (X_train, y_train), (X_test, y_test) = _get_test_data()

    left = Sequential()
    left.add(Dense(nb_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Dense(nb_hidden, input_shape=(input_dim,)))
    right.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([left, right], mode='concat'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_data=([X_test, X_test], y_test))
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=([X_test, X_test], y_test))
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_split=0.1)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_split=0.1)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, shuffle=False)

    loss = model.evaluate([X_test, X_test], y_test, verbose=0)
    assert(loss < 0.8)

    model.predict([X_test, X_test], verbose=0)
    model.predict_classes([X_test, X_test], verbose=0)
    model.predict_proba([X_test, X_test], verbose=0)
    model.get_config(verbose=0)

    fname = 'test_merge_concat_temp.h5'
    model.save_weights(fname, overwrite=True)
    left = Sequential()
    left.add(Dense(nb_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Dense(nb_hidden, input_shape=(input_dim,)))
    right.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([left, right], mode='concat'))

    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.load_weights(fname)
    os.remove(fname)

    nloss = model.evaluate([X_test, X_test], y_test, verbose=0)
    assert(loss == nloss)


def test_merge_recursivity():
    (X_train, y_train), (X_test, y_test) = _get_test_data()
    left = Sequential()
    left.add(Dense(nb_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Dense(nb_hidden, input_shape=(input_dim,)))
    right.add(Activation('relu'))

    righter = Sequential()
    righter.add(Dense(nb_hidden, input_shape=(input_dim,)))
    righter.add(Activation('relu'))

    intermediate = Sequential()
    intermediate.add(Merge([left, right], mode='sum'))
    intermediate.add(Dense(nb_hidden))
    intermediate.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([intermediate, righter], mode='sum'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit([X_train, X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_data=([X_test, X_test, X_test], y_test))
    model.fit([X_train, X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=([X_test, X_test, X_test], y_test))
    model.fit([X_train, X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_split=0.1)
    model.fit([X_train, X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_split=0.1)
    model.fit([X_train, X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    model.fit([X_train, X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, shuffle=False)

    loss = model.evaluate([X_test, X_test, X_test], y_test, verbose=0)
    assert(loss < 0.8)

    model.predict([X_test, X_test, X_test], verbose=0)
    model.predict_classes([X_test, X_test, X_test], verbose=0)
    model.predict_proba([X_test, X_test, X_test], verbose=0)
    model.get_config(verbose=0)

    fname = 'test_merge_recursivity_temp.h5'
    model.save_weights(fname, overwrite=True)
    model.load_weights(fname)
    os.remove(fname)

    nloss = model.evaluate([X_test, X_test, X_test], y_test, verbose=0)
    assert(loss == nloss)


def test_merge_overlap():
    (X_train, y_train), (X_test, y_test) = _get_test_data()
    left = Sequential()
    left.add(Dense(nb_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([left, left], mode='sum'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, y_test))
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=2, validation_data=(X_test, y_test))
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_split=0.1)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=1, validation_split=0.1)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=False)

    model.train_on_batch(X_train[:32], y_train[:32])

    loss = model.evaluate(X_test, y_test, verbose=0)
    assert(loss < 0.9)
    model.predict(X_test, verbose=0)
    model.predict_classes(X_test, verbose=0)
    model.predict_proba(X_test, verbose=0)
    model.get_config(verbose=0)

    fname = 'test_merge_overlap_temp.h5'
    model.save_weights(fname, overwrite=True)
    model.load_weights(fname)
    os.remove(fname)

    nloss = model.evaluate(X_test, y_test, verbose=0)
    assert(loss == nloss)


def test_lambda():
    (X_train, y_train), (X_test, y_test) = _get_test_data()

    def func(X):
        s = X[0]
        for i in range(1, len(X)):
            s += X[i]
        return s

    def activation(X):
        return K.softmax(X)

    def output_shape(input_shapes):
        return input_shapes[0]

    left = Sequential()
    left.add(Dense(nb_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Dense(nb_hidden, input_shape=(input_dim,)))
    right.add(Activation('relu'))

    model = Sequential()
    model.add(LambdaMerge([left, right], function=func,
                          output_shape=output_shape))
    model.add(Dense(nb_class))
    model.add(Lambda(activation))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_data=([X_test, X_test], y_test))
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=([X_test, X_test], y_test))
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_split=0.1)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_split=0.1)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, shuffle=False)

    loss = model.evaluate([X_test, X_test], y_test, verbose=0)
    assert(loss < 0.8)

    model.predict([X_test, X_test], verbose=0)
    model.predict_classes([X_test, X_test], verbose=0)
    model.predict_proba([X_test, X_test], verbose=0)
    model.get_config(verbose=0)

    # test weight saving
    fname = 'test_lambda_temp.h5'
    model.save_weights(fname, overwrite=True)
    left = Sequential()
    left.add(Dense(nb_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))
    right = Sequential()
    right.add(Dense(nb_hidden, input_shape=(input_dim,)))
    right.add(Activation('relu'))
    model = Sequential()
    model.add(LambdaMerge([left, right], function=func,
                          output_shape=output_shape))
    model.add(Dense(nb_class))
    model.add(Lambda(activation))
    model.load_weights(fname)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    os.remove(fname)

    nloss = model.evaluate([X_test, X_test], y_test, verbose=0)
    assert(loss == nloss)

    # test "join" mode in Lambda
    def difference(input_dict):
        assert(len(input_dict) == 2)
        keys = list(input_dict.keys())
        return input_dict[keys[0]] - input_dict[keys[1]]

    g = Graph()
    g.add_input(name='input_a', input_shape=(2,))
    g.add_input(name='input_b', input_shape=(2,))
    g.add_node(Lambda(difference, output_shape=(2,)),
               inputs=['input_a', 'input_b'],
               merge_mode='join',
               name='d')
    g.add_output(name='output', input='d')
    g.compile(loss={'output': 'categorical_crossentropy'}, optimizer='rmsprop')


def test_sequential_count_params():
    input_dim = 20
    nb_units = 10
    nb_classes = 2

    n = input_dim * nb_units + nb_units
    n += nb_units * nb_units + nb_units
    n += nb_units * nb_classes + nb_classes

    model = Sequential()
    model.add(Dense(nb_units, input_shape=(input_dim,)))
    model.add(Dense(nb_units))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    assert(n == model.count_params())

    model.compile('sgd', 'binary_crossentropy')
    assert(n == model.count_params())


def test_siamese_1():
    (X_train, y_train), (X_test, y_test) = _get_test_data()
    left = Sequential()
    left.add(Dense(nb_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Dense(nb_hidden, input_shape=(input_dim,)))
    right.add(Activation('relu'))

    model = Sequential()
    model.add(Siamese(Dense(nb_hidden), [left, right], merge_mode='sum'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_data=([X_test, X_test], y_test))
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=([X_test, X_test], y_test))
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_split=0.1)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_split=0.1)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, shuffle=False)

    loss = model.evaluate([X_test, X_test], y_test, verbose=0)
    assert(loss < 0.8)

    model.predict([X_test, X_test], verbose=0)
    model.predict_classes([X_test, X_test], verbose=0)
    model.predict_proba([X_test, X_test], verbose=0)
    model.get_config(verbose=0)

    # test weight saving
    fname = 'test_siamese_1.h5'
    model.save_weights(fname, overwrite=True)
    left = Sequential()
    left.add(Dense(nb_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Dense(nb_hidden, input_shape=(input_dim,)))
    right.add(Activation('relu'))

    model = Sequential()
    model.add(Siamese(Dense(nb_hidden), [left, right], merge_mode='sum'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    model.load_weights(fname)
    os.remove(fname)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    nloss = model.evaluate([X_test, X_test], y_test, verbose=0)
    assert(loss == nloss)


def test_siamese_2():
    (X_train, y_train), (X_test, y_test) = _get_test_data()
    left = Sequential()
    left.add(Dense(nb_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Dense(nb_hidden, input_shape=(input_dim,)))
    right.add(Activation('relu'))

    add_shared_layer(Dense(nb_hidden), [left, right])

    left.add(Dense(nb_hidden))
    right.add(Dense(nb_hidden))

    add_shared_layer(Dense(nb_hidden), [left, right])

    model = Sequential()
    model.add(Merge([left, right], mode='sum'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_data=([X_test, X_test], y_test))
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=([X_test, X_test], y_test))
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_split=0.1)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_split=0.1)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, shuffle=False)

    loss = model.evaluate([X_test, X_test], y_test, verbose=0)
    assert(loss < 0.8)

    model.predict([X_test, X_test], verbose=0)
    model.predict_classes([X_test, X_test], verbose=0)
    model.predict_proba([X_test, X_test], verbose=0)
    model.get_config(verbose=0)

    # test weight saving
    fname = 'test_siamese_2.h5'
    model.save_weights(fname, overwrite=True)
    left = Sequential()
    left.add(Dense(nb_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Dense(nb_hidden, input_shape=(input_dim,)))
    right.add(Activation('relu'))

    add_shared_layer(Dense(nb_hidden), [left, right])

    left.add(Dense(nb_hidden))
    right.add(Dense(nb_hidden))

    add_shared_layer(Dense(nb_hidden), [left, right])

    model = Sequential()
    model.add(Merge([left, right], mode='sum'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    model.load_weights(fname)
    os.remove(fname)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    nloss = model.evaluate([X_test, X_test], y_test, verbose=0)
    assert(loss == nloss)


if __name__ == '__main__':
    pytest.main([__file__])
