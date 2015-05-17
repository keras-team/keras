from __future__ import absolute_import
from __future__ import print_function
import keras
from keras.datasets import mnist
import keras.models
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.regularizers import l2, l1
from keras.constraints import maxnorm, nonneg
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils, generic_utils
import theano
import theano.tensor as T
import numpy as np
import scipy

import cPickle
import gzip

batch_size = 100
nb_classes = 10
nb_epoch = 10

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# self define new activation function
def LeakyReLU(alpha=0.3):
	def LeakyReLUwrap(X):
		return ((X + T.abs_(X)) / 2.0) + alpha * ((X - T.abs_(X)) / 2.0)
	return LeakyReLUwrap

model = Sequential()
model.add(Dense(784, 128, activation=LeakyReLU(0.3)))
# model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 10))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
