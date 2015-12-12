from __future__ import print_function
import numpy as np
import pytest

from keras.utils.test_utils import get_test_data
from keras.models import Sequential
from keras.layers.core import TimeDistributedDense
from keras.layers.recurrent import GRU
from keras.utils.np_utils import to_categorical


def test_temporal_classification():
    '''
    Classify temporal sequences of float numbers of length 3 into 2 classes using
    single layer of GRU units and softmax applied to the last activations of the units
    '''
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
    history = model.fit(X_train, y_train, nb_epoch=5, batch_size=16,
                        validation_data=(X_test, y_test),
                        show_accuracy=True, verbose=0)
    assert(history.history['val_acc'][-1] > 0.9)


def test_temporal_regression():
    '''
    Predict float numbers (regression) based on sequences of float numbers of length 3 using
    single layer of GRU units
    '''
    np.random.seed(1337)
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=500,
                                                         nb_test=200,
                                                         input_shape=(3, 5),
                                                         output_shape=(2,),
                                                         classification=False)
    model = Sequential()
    model.add(GRU(y_train.shape[-1],
              input_shape=(X_train.shape[1], X_train.shape[2])))
    model.compile(loss='hinge', optimizer='adam')
    history = model.fit(X_train, y_train, nb_epoch=5, batch_size=16,
                        validation_data=(X_test, y_test), verbose=0)
    assert(history.history['val_loss'][-1] < 0.75)


def test_sequence_to_sequence():
    '''
    Apply a same Dense layer for each element of time dimension of the input
    and make predictions of the output sequence elements.
    This does not make use of the temporal structure of the sequence
    (see TimeDistributedDense for more details)
    '''
    np.random.seed(1337)
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


if __name__ == '__main__':
    pytest.main([__file__])
