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

batch_size = 100
nb_classes = 10
nb_epoch = 10

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train=X_train.reshape(60000,784)
X_test=X_test.reshape(10000,784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(784, 20, W_constraint=maxnorm(1)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(20, 20, W_constraint=nonneg()))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(20, 10, W_constraint=maxnorm(1)))
model.add(Activation('softmax'))


rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0)

a=model.params[0].eval()
if np.isclose(np.max(np.sqrt(np.sum(a**2, axis=0))),1):
	print('Maxnorm test passed')
else:
	raise ValueError('Maxnorm test failed!')
		
b=model.params[2].eval()
if np.min(b)==0 and np.min(a)!=0:
	print('Nonneg test passed')
else:
	raise ValueError('Nonneg test failed!')
	

model = Sequential()
model.add(Dense(784, 20))
model.add(Activation('relu'))
model.add(Dense(20, 20, W_regularizer=l1(.01)))
model.add(Activation('relu'))
model.add(Dense(20, 10))
model.add(Activation('softmax'))


rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=20, show_accuracy=True, verbose=0)

a=model.params[2].eval().reshape(400)
(D, p1) = scipy.stats.kurtosistest(a)

model = Sequential()
model.add(Dense(784, 20))
model.add(Activation('relu'))
model.add(Dense(20, 20, W_regularizer=l2(.01)))
model.add(Activation('relu'))
model.add(Dense(20, 10))
model.add(Activation('softmax'))


rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=20, show_accuracy=True, verbose=0)

a=model.params[2].eval().reshape(400)
(D, p2) = scipy.stats.kurtosistest(a)

if p1<.01 and p2>.01:
	print('L1 and L2 regularization tests passed')
else:
	raise ValueError('L1 and L2 regularization tests failed!')