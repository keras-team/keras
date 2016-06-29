from __future__ import print_function
import numpy as np
import pytest

from keras.utils.test_utils import get_test_data
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils.np_utils import to_categorical


def test_image_classification():
    '''
    Classify random 16x16 color images into several classes using logistic regression
    with convolutional hidden layer.
    '''
    np.random.seed(1337)
    input_shape = (3, 16, 16)
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=500,
                                                         nb_test=200,
                                                         input_shape=input_shape,
                                                         classification=True,
                                                         nb_class=4)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # convolution kernel size
    nb_conv = 3
    # size of pooling area for max pooling
    nb_pool = 2

    model = Sequential([
        Convolution2D(nb_filter=8, nb_row=nb_conv, nb_col=nb_conv, input_shape=input_shape),
        MaxPooling2D(pool_size=(nb_pool, nb_pool)),
        Flatten(),
        Activation('relu'),
        Dense(y_test.shape[-1], activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, nb_epoch=10, batch_size=16,
                        validation_data=(X_test, y_test),
                        verbose=0)
    assert(history.history['val_acc'][-1] > 0.85)


if __name__ == '__main__':
    pytest.main([__file__])
