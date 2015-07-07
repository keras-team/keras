from __future__ import print_function
import numpy as np
np.random.seed(1337)
from keras.utils.test_utils import get_test_data
from keras.models import Sequential
from keras.layers.core import Dense, Activation, TimeDistributedDense, Flatten
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution2D
from keras.utils.np_utils import to_categorical
import unittest

class TestRegularizers(unittest.TestCase):
    def test_vector_clf(self):
        nb_hidden = 10

        print('vector classification data:')
        (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=1000, nb_test=200, input_shape=(10,),
            classification=True, nb_class=2)
        print('X_train:', X_train.shape)
        print('X_test:', X_test.shape)
        print('y_train:', y_train.shape)
        print('y_test:', y_test.shape)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        model = Sequential()
        model.add(Dense(X_train.shape[-1], nb_hidden))
        model.add(Activation('relu'))
        model.add(Dense(nb_hidden, y_train.shape[-1]))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        history = model.fit(X_train, y_train, nb_epoch=12, batch_size=16, validation_data=(X_test, y_test), show_accuracy=True, verbose=2)
        print(history.history)
        self.assertTrue(history.history['val_acc'][-1] > 0.9)

    def test_vector_reg(self):
        nb_hidden = 10
        print('vector regression data:')
        (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=1000, nb_test=200, input_shape=(10,), output_shape=(2,),
            classification=False)
        print('X_train:', X_train.shape)
        print('X_test:', X_test.shape)
        print('y_train:', y_train.shape)
        print('y_test:', y_test.shape)

        model = Sequential()
        model.add(Dense(X_train.shape[-1], nb_hidden))
        model.add(Activation('tanh'))
        model.add(Dense(nb_hidden, y_train.shape[-1]))
        model.compile(loss='hinge', optimizer='adagrad')
        history = model.fit(X_train, y_train, nb_epoch=12, batch_size=16, validation_data=(X_test, y_test), verbose=2)
        self.assertTrue(history.history['val_loss'][-1] < 0.9)

    def test_temporal_clf(self):
        print('temporal classification data:')
        (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=1000, nb_test=200, input_shape=(5,10), 
            classification=True, nb_class=2)
        print('X_train:', X_train.shape)
        print('X_test:', X_test.shape)
        print('y_train:', y_train.shape)
        print('y_test:', y_test.shape)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        model = Sequential()
        model.add(GRU(X_train.shape[-1], y_train.shape[-1]))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adadelta')
        history = model.fit(X_train, y_train, nb_epoch=12, batch_size=16, validation_data=(X_test, y_test), show_accuracy=True, verbose=2)
        self.assertTrue(history.history['val_acc'][-1] > 0.9)

    def test_temporal_reg(self):
        print('temporal regression data:')
        (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=1000, nb_test=200, input_shape=(5, 10), output_shape=(2,),
            classification=False)
        print('X_train:', X_train.shape)
        print('X_test:', X_test.shape)
        print('y_train:', y_train.shape)
        print('y_test:', y_test.shape)

        model = Sequential()
        model.add(GRU(X_train.shape[-1], y_train.shape[-1]))
        model.compile(loss='hinge', optimizer='adam')
        history = model.fit(X_train, y_train, nb_epoch=12, batch_size=16, validation_data=(X_test, y_test), verbose=2)
        self.assertTrue(history.history['val_loss'][-1] < 0.75)


    def test_seq_to_seq(self):
        print('sequence to sequence data:')
        (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=1000, nb_test=200, input_shape=(5, 10), output_shape=(5, 10),
            classification=False)
        print('X_train:', X_train.shape)
        print('X_test:', X_test.shape)
        print('y_train:', y_train.shape)
        print('y_test:', y_test.shape)

        model = Sequential()
        model.add(TimeDistributedDense(X_train.shape[-1], y_train.shape[-1]))
        model.compile(loss='hinge', optimizer='rmsprop')
        history = model.fit(X_train, y_train, nb_epoch=12, batch_size=16, validation_data=(X_test, y_test), verbose=2)
        self.assertTrue(history.history['val_loss'][-1] < 0.75)


    def test_img_clf(self):
        print('image classification data:')
        (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=1000, nb_test=200, input_shape=(3, 32, 32), 
            classification=True, nb_class=2)
        print('X_train:', X_train.shape)
        print('X_test:', X_test.shape)
        print('y_train:', y_train.shape)
        print('y_test:', y_test.shape)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        model = Sequential()
        model.add(Convolution2D(32, 3, 32, 32))
        model.add(Activation('sigmoid'))
        model.add(Flatten())
        model.add(Dense(32, y_test.shape[-1]))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd')
        history = model.fit(X_train, y_train, nb_epoch=12, batch_size=16, validation_data=(X_test, y_test), show_accuracy=True, verbose=2)
        self.assertTrue(history.history['val_acc'][-1] > 0.9)


if __name__ == '__main__':
    print('Test different types of classification and regression tasks')
    unittest.main()