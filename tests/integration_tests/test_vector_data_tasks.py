from __future__ import print_function
import numpy as np
import pytest

from keras.utils.test_utils import get_test_data
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils.np_utils import to_categorical


def test_vector_classification():
    '''
    Classify random float vectors into 2 classes with logistic regression
    using 2 layer neural network with ReLU hidden units.
    '''
    np.random.seed(1337)
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
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, nb_epoch=15, batch_size=16,
                        validation_data=(X_test, y_test),
                        verbose=0)
    assert(history.history['val_acc'][-1] > 0.8)


def test_vector_regression():
    '''
    Perform float data prediction (regression) using 2 layer MLP
    with tanh and sigmoid activations.
    '''
    np.random.seed(1337)
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


if __name__ == '__main__':
    pytest.main([__file__])
