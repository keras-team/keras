from __future__ import absolute_import
from __future__ import print_function
import pytest
import numpy as np
np.random.seed(1337)

from keras import backend as K
from keras.models import Graph, Sequential, model_from_json, model_from_yaml
from keras.layers.core import Dense, Activation, Merge, Lambda, LambdaMerge, Siamese, add_shared_layer
from keras.layers import containers
from keras.utils import np_utils
from keras.utils.test_utils import get_test_data

import os


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


####################
# SEQUENTIAL TEST  #
####################

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

    loss = model.evaluate(X_train, y_train, verbose=0)
    assert(loss < 0.9)


def test_sequential():
    (X_train, y_train), (X_test, y_test) = _get_test_data()

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


###############
# GRAPH TEST  #
###############

(X_train_graph, y_train_graph), (X_test_graph, y_test_graph) = get_test_data(nb_train=1000,
                                                                             nb_test=200,
                                                                             input_shape=(32,),
                                                                             classification=False,
                                                                             output_shape=(4,))
(X2_train_graph, y2_train_graph), (X2_test_graph, y2_test_graph) = get_test_data(nb_train=1000,
                                                                                 nb_test=200,
                                                                                 input_shape=(32,),
                                                                                 classification=False,
                                                                                 output_shape=(1,))


def test_graph_fit_generator():
    def data_generator_graph(train):
        while 1:
            if train:
                yield {'input1': X_train_graph, 'output1': y_train_graph}
            else:
                yield {'input1': X_test_graph, 'output1': y_test_graph}

    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))

    graph.add_node(Dense(16), name='dense1', input='input1')
    graph.add_node(Dense(4), name='dense2', input='input1')
    graph.add_node(Dense(4), name='dense3', input='dense1')

    graph.add_output(name='output1',
                     inputs=['dense2', 'dense3'],
                     merge_mode='sum')
    graph.compile('rmsprop', {'output1': 'mse'})

    graph.fit_generator(data_generator_graph(True), 1000, nb_epoch=4)
    graph.fit_generator(data_generator_graph(True), 1000, nb_epoch=4)
    graph.fit_generator(data_generator_graph(True), 1000, nb_epoch=4, validation_data={'input1': X_test_graph, 'output1': y_test_graph})
    graph.fit_generator(data_generator_graph(True), 1000, nb_epoch=4, validation_data={'input1': X_test_graph, 'output1': y_test_graph})

    loss = graph.evaluate({'input1': X_test_graph, 'output1': y_test_graph}, verbose=0)
    assert(loss < 3.)


def test_1o_1i():
    # test a non-sequential graph with 1 input and 1 output
    np.random.seed(1337)

    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))

    graph.add_node(Dense(16), name='dense1', input='input1')
    graph.add_node(Dense(4), name='dense2', input='input1')
    graph.add_node(Dense(4), name='dense3', input='dense1')

    graph.add_output(name='output1',
                     inputs=['dense2', 'dense3'],
                     merge_mode='sum')
    graph.compile('rmsprop', {'output1': 'mse'})

    graph.fit({'input1': X_train_graph, 'output1': y_train_graph},
              nb_epoch=10)
    out = graph.predict({'input1': X_test_graph})
    assert(type(out == dict))
    assert(len(out) == 1)
    loss = graph.test_on_batch({'input1': X_test_graph, 'output1': y_test_graph})
    loss = graph.train_on_batch({'input1': X_test_graph, 'output1': y_test_graph})
    loss = graph.evaluate({'input1': X_test_graph, 'output1': y_test_graph}, verbose=0)
    assert(loss < 2.5)

    # test validation split
    graph.fit({'input1': X_train_graph, 'output1': y_train_graph},
              validation_split=0.2, nb_epoch=1)
    # test validation data
    graph.fit({'input1': X_train_graph, 'output1': y_train_graph},
              validation_data={'input1': X_train_graph, 'output1': y_train_graph},
              nb_epoch=1)


def test_1o_1i_2():
    # test a more complex non-sequential graph with 1 input and 1 output
    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))

    graph.add_node(Dense(16), name='dense1', input='input1')
    graph.add_node(Dense(4), name='dense2-0', input='input1')
    graph.add_node(Activation('relu'), name='dense2', input='dense2-0')

    graph.add_node(Dense(16), name='dense3', input='dense2')
    graph.add_node(Dense(4), name='dense4', inputs=['dense1', 'dense3'],
                   merge_mode='sum')

    graph.add_output(name='output1', inputs=['dense2', 'dense4'],
                     merge_mode='sum')
    graph.compile('rmsprop', {'output1': 'mse'})

    graph.fit({'input1': X_train_graph, 'output1': y_train_graph},
              nb_epoch=10)
    out = graph.predict({'input1': X_train_graph})
    assert(type(out == dict))
    assert(len(out) == 1)

    loss = graph.test_on_batch({'input1': X_test_graph, 'output1': y_test_graph})
    loss = graph.train_on_batch({'input1': X_test_graph, 'output1': y_test_graph})
    loss = graph.evaluate({'input1': X_test_graph, 'output1': y_test_graph})
    assert(loss < 2.5)

    graph.get_config(verbose=1)
    graph.summary()


def test_1o_2i():
    # test a non-sequential graph with 2 inputs and 1 output
    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))
    graph.add_input(name='input2', input_shape=(32,))

    graph.add_node(Dense(16), name='dense1', input='input1')
    graph.add_node(Dense(4), name='dense2', input='input2')
    graph.add_node(Dense(4), name='dense3', input='dense1')

    graph.add_output(name='output1', inputs=['dense2', 'dense3'],
                     merge_mode='sum')
    graph.compile('rmsprop', {'output1': 'mse'})

    graph.fit({'input1': X_train_graph, 'input2': X2_train_graph, 'output1': y_train_graph},
              nb_epoch=10)
    out = graph.predict({'input1': X_test_graph, 'input2': X2_test_graph})
    assert(type(out == dict))
    assert(len(out) == 1)

    loss = graph.test_on_batch({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    loss = graph.train_on_batch({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    loss = graph.evaluate({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    assert(loss < 3.0)

    graph.get_config(verbose=1)


def test_siamese_3():
    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))
    graph.add_input(name='input2', input_shape=(32,))

    graph.add_shared_node(Dense(16), name='shared', inputs=['input1', 'input2'], merge_mode='sum')
    graph.add_node(Dense(4), name='dense1', input='shared')
    graph.add_node(Dense(4), name='dense2', input='dense1')

    graph.add_output(name='output1', input='dense2')
    graph.compile('rmsprop', {'output1': 'mse'})

    graph.fit({'input1': X_train_graph, 'input2': X2_train_graph, 'output1': y_train_graph},
              nb_epoch=10)
    out = graph.predict({'input1': X_test_graph, 'input2': X2_test_graph})
    assert(type(out == dict))
    assert(len(out) == 1)

    loss = graph.test_on_batch({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    loss = graph.train_on_batch({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    loss = graph.evaluate({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    assert(loss < 3.0)

    graph.get_config(verbose=1)


def test_siamese_4():
    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))
    graph.add_input(name='input2', input_shape=(32,))

    graph.add_shared_node(Dense(16), name='shared1', inputs=['input1', 'input2'])
    graph.add_shared_node(Dense(4), name='shared2', inputs=['shared1'])
    graph.add_shared_node(Dense(4), name='shared3', inputs=['shared2'], merge_mode='sum')
    graph.add_node(Dense(4), name='dense', input='shared3')

    graph.add_output(name='output1', input='dense',
                     merge_mode='sum')
    graph.compile('rmsprop', {'output1': 'mse'})

    graph.fit({'input1': X_train_graph, 'input2': X2_train_graph, 'output1': y_train_graph},
              nb_epoch=10)
    out = graph.predict({'input1': X_test_graph, 'input2': X2_test_graph})
    assert(type(out == dict))
    assert(len(out) == 1)

    loss = graph.test_on_batch({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    loss = graph.train_on_batch({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    loss = graph.evaluate({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    assert(loss < 3.0)

    graph.get_config(verbose=1)


def test_siamese_5():
    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))
    graph.add_input(name='input2', input_shape=(32,))

    graph.add_shared_node(Dense(16), name='shared1', inputs=['input1', 'input2'])
    graph.add_shared_node(Dense(4), name='shared2', inputs=['shared1'])
    graph.add_shared_node(Dense(4), name='shared3', inputs=['shared2'], outputs=['shared_output1','shared_output2'])
    graph.add_node(Dense(4), name='dense1',  input='shared_output1')
    graph.add_node(Dense(4), name='dense2',  input='shared_output2')

    graph.add_output(name='output1', inputs=['dense1', 'dense2'],
                     merge_mode='sum')
    graph.compile('rmsprop', {'output1': 'mse'})

    graph.fit({'input1': X_train_graph, 'input2': X2_train_graph, 'output1': y_train_graph},
              nb_epoch=10)
    out = graph.predict({'input1': X_test_graph, 'input2': X2_test_graph})
    assert(type(out == dict))
    assert(len(out) == 1)

    loss = graph.test_on_batch({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    loss = graph.train_on_batch({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    loss = graph.evaluate({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    assert(loss < 3.0)

    graph.get_config(verbose=1)


def test_2o_1i_weights():
    # test a non-sequential graph with 1 input and 2 outputs
    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))

    graph.add_node(Dense(16), name='dense1', input='input1')
    graph.add_node(Dense(4), name='dense2', input='input1')
    graph.add_node(Dense(1), name='dense3', input='dense1')

    graph.add_output(name='output1', input='dense2')
    graph.add_output(name='output2', input='dense3')
    graph.compile('rmsprop', {'output1': 'mse', 'output2': 'mse'})

    graph.fit({'input1': X_train_graph, 'output1': y_train_graph, 'output2': y2_train_graph},
              nb_epoch=10)
    out = graph.predict({'input1': X_test_graph})
    assert(type(out == dict))
    assert(len(out) == 2)
    loss = graph.test_on_batch({'input1': X_test_graph, 'output1': y_test_graph, 'output2': y2_test_graph})
    loss = graph.train_on_batch({'input1': X_test_graph, 'output1': y_test_graph, 'output2': y2_test_graph})
    loss = graph.evaluate({'input1': X_test_graph, 'output1': y_test_graph, 'output2': y2_test_graph})
    assert(loss < 4.)

    # test weight saving
    fname = 'test_2o_1i_weights_temp.h5'
    graph.save_weights(fname, overwrite=True)

    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))
    graph.add_node(Dense(16), name='dense1', input='input1')
    graph.add_node(Dense(4), name='dense2', input='input1')
    graph.add_node(Dense(1), name='dense3', input='dense1')
    graph.add_output(name='output1', input='dense2')
    graph.add_output(name='output2', input='dense3')
    graph.compile('rmsprop', {'output1': 'mse', 'output2': 'mse'})
    graph.load_weights('test_2o_1i_weights_temp.h5')
    os.remove(fname)

    nloss = graph.evaluate({'input1': X_test_graph, 'output1': y_test_graph, 'output2': y2_test_graph})
    assert(loss == nloss)


def test_2o_1i_sample_weights():
    # test a non-sequential graph with 1 input and 2 outputs with sample weights
    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))

    graph.add_node(Dense(16), name='dense1', input='input1')
    graph.add_node(Dense(4), name='dense2', input='input1')
    graph.add_node(Dense(1), name='dense3', input='dense1')

    graph.add_output(name='output1', input='dense2')
    graph.add_output(name='output2', input='dense3')

    weights1 = np.random.uniform(size=y_train_graph.shape[0])
    weights2 = np.random.uniform(size=y2_train_graph.shape[0])
    weights1_test = np.random.uniform(size=y_test_graph.shape[0])
    weights2_test = np.random.uniform(size=y2_test_graph.shape[0])

    graph.compile('rmsprop', {'output1': 'mse', 'output2': 'mse'})

    graph.fit({'input1': X_train_graph, 'output1': y_train_graph, 'output2': y2_train_graph},
              nb_epoch=10,
              sample_weight={'output1': weights1, 'output2': weights2})
    out = graph.predict({'input1': X_test_graph})
    assert(type(out == dict))
    assert(len(out) == 2)
    loss = graph.test_on_batch({'input1': X_test_graph, 'output1': y_test_graph, 'output2': y2_test_graph},
                               sample_weight={'output1': weights1_test, 'output2': weights2_test})
    loss = graph.train_on_batch({'input1': X_train_graph, 'output1': y_train_graph, 'output2': y2_train_graph},
                                sample_weight={'output1': weights1, 'output2': weights2})
    loss = graph.evaluate({'input1': X_train_graph, 'output1': y_train_graph, 'output2': y2_train_graph},
                          sample_weight={'output1': weights1, 'output2': weights2})


def test_recursive():
    # test layer-like API

    graph = containers.Graph()
    graph.add_input(name='input1', input_shape=(32,))
    graph.add_node(Dense(16), name='dense1', input='input1')
    graph.add_node(Dense(4), name='dense2', input='input1')
    graph.add_node(Dense(4), name='dense3', input='dense1')
    graph.add_output(name='output1', inputs=['dense2', 'dense3'],
                     merge_mode='sum')

    seq = Sequential()
    seq.add(Dense(32, input_shape=(32,)))
    seq.add(graph)
    seq.add(Dense(4))

    seq.compile('rmsprop', 'mse')

    seq.fit(X_train_graph, y_train_graph, batch_size=10, nb_epoch=10)
    loss = seq.evaluate(X_test_graph, y_test_graph)
    assert(loss < 2.5)

    loss = seq.evaluate(X_test_graph, y_test_graph, show_accuracy=True)
    seq.predict(X_test_graph)
    seq.get_config(verbose=1)


def test_create_output():
    # test create_output argument
    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))

    graph.add_node(Dense(16), name='dense1', input='input1')
    graph.add_node(Dense(4), name='dense2', input='input1')
    graph.add_node(Dense(4), name='dense3', input='dense1')
    graph.add_node(Dense(4), name='output1', inputs=['dense2', 'dense3'],
                   merge_mode='sum', create_output=True)
    graph.compile('rmsprop', {'output1': 'mse'})

    history = graph.fit({'input1': X_train_graph, 'output1': y_train_graph},
                        nb_epoch=10)
    out = graph.predict({'input1': X_test_graph})
    assert(type(out == dict))
    assert(len(out) == 1)

    loss = graph.test_on_batch({'input1': X_test_graph, 'output1': y_test_graph})
    loss = graph.train_on_batch({'input1': X_test_graph, 'output1': y_test_graph})
    loss = graph.evaluate({'input1': X_test_graph, 'output1': y_test_graph})
    assert(loss < 2.5)


def test_count_params():
    # test count params

    nb_units = 100
    nb_classes = 2

    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))
    graph.add_input(name='input2', input_shape=(32,))
    graph.add_node(Dense(nb_units),
                   name='dense1', input='input1')
    graph.add_node(Dense(nb_classes),
                   name='dense2', input='input2')
    graph.add_node(Dense(nb_classes),
                   name='dense3', input='dense1')
    graph.add_output(name='output', inputs=['dense2', 'dense3'],
                     merge_mode='sum')

    n = 32 * nb_units + nb_units
    n += 32 * nb_classes + nb_classes
    n += nb_units * nb_classes + nb_classes

    assert(n == graph.count_params())

    graph.compile('rmsprop', {'output': 'binary_crossentropy'})

    assert(n == graph.count_params())


if __name__ == '__main__':
    pytest.main([__file__])
