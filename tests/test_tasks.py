from __future__ import print_function
import numpy as np
import pytest
np.random.seed(1337)

from keras.utils.test_utils import get_test_data
from keras.models import Sequential
from keras.layers.core import Dense, TimeDistributedDense, Flatten
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution2D
from keras.utils.np_utils import to_categorical


def test_vector_classification():
    nb_hidden = 10

    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=500,
                                                         nb_test=200,
                                                         input_shape=(20,),
                                                         classification=True,
                                                         nb_class=2)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential([
        Dense(nb_hidden, input_shape=(X_train.shape[-1],), activation='relu'),
        Dense(y_train.shape[-1], activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    history = model.fit(X_train, y_train, nb_epoch=15, batch_size=16,
                        validation_data=(X_test, y_test),
                        show_accuracy=True, verbose=0)
    assert(history.history['val_acc'][-1] > 0.8)


def test_vector_regression():
    nb_hidden = 10
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=500,
                                                         nb_test=200,
                                                         input_shape=(20,),
                                                         output_shape=(2,),
                                                         classification=False)

    model = Sequential([
        Dense(nb_hidden, input_shape=(X_train.shape[-1],), activation='tanh'),
        Dense(y_train.shape[-1])
    ])

    model.compile(loss='hinge', optimizer='adagrad')
    history = model.fit(X_train, y_train, nb_epoch=20, batch_size=16,
                        validation_data=(X_test, y_test), verbose=0)
    assert (history.history['val_loss'][-1] < 0.9)


def test_temporal_classification():
    np.random.seed(1337)
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=500,
                                                         nb_test=200,
                                                         input_shape=(3, 5),
                                                         classification=True,
                                                         nb_class=2)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential()
    model.add(GRU(y_train.shape[-1],
                  input_shape=(X_train.shape[1], X_train.shape[2]),
                  activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    history = model.fit(X_train, y_train, nb_epoch=20, batch_size=16,
                        validation_data=(X_test, y_test),
                        show_accuracy=True, verbose=0)
    assert(history.history['val_acc'][-1] > 0.9)


def test_temporal_regression():
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=500,
                                                         nb_test=200,
                                                         input_shape=(3, 5),
                                                         output_shape=(2,),
                                                         classification=False)
    model = Sequential()
    model.add(GRU(y_train.shape[-1],
              input_shape=(X_train.shape[1], X_train.shape[2])))
    model.compile(loss='hinge', optimizer='adam')
    history = model.fit(X_train, y_train, nb_epoch=20, batch_size=16,
                        validation_data=(X_test, y_test), verbose=0)
    assert(history.history['val_loss'][-1] < 0.8)


def test_sequence_to_sequence():
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=500,
                                                         nb_test=200,
                                                         input_shape=(3, 5),
                                                         output_shape=(3, 5),
                                                         classification=False)

    model = Sequential()
    model.add(TimeDistributedDense(y_train.shape[-1],
              input_shape=(X_train.shape[1], X_train.shape[2])))
    model.compile(loss='hinge', optimizer='rmsprop')
    history = model.fit(X_train, y_train, nb_epoch=20, batch_size=16,
                        validation_data=(X_test, y_test), verbose=0)
    assert(history.history['val_loss'][-1] < 0.8)


def test_image_classification():
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=500,
                                                         nb_test=200,
                                                         input_shape=(3, 8, 8),
                                                         classification=True,
                                                         nb_class=2)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential([
        Convolution2D(8, 8, 8, input_shape=(3, 8, 8), activation='sigmoid'),
        Flatten(),
        Dense(y_test.shape[-1], activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    history = model.fit(X_train, y_train, nb_epoch=20, batch_size=16,
                        validation_data=(X_test, y_test),
                        show_accuracy=True, verbose=0)
    assert(history.history['val_acc'][-1] > 0.9)


if __name__ == '__main__':
    pytest.main([__file__])
