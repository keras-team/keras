__author__ = 'mccolgan'


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten, ActivityRegularization
from keras.utils import np_utils
from keras.regularizers import l2, activity_l1

import numpy as np

np.random.seed(1337)

max_train_samples = 128*8
max_test_samples = 1000

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((-1, 28*28))
X_test = X_test.reshape((-1, 28*28))

nb_classes = len(np.unique(y_train))
Y_train = np_utils.to_categorical(y_train, nb_classes)[:max_train_samples]
Y_test = np_utils.to_categorical(y_test, nb_classes)[:max_test_samples]

model_noreg = Sequential()
model_noreg.add(Flatten())
model_noreg.add(Dense(28*28, 20, activation='sigmoid'))
model_noreg.add(Dense(20, 10, activation='sigmoid'))

model_noreg.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model_noreg.fit(X_train, Y_train)

score_noreg = model_noreg.evaluate(X_test, Y_test)
score_train_noreg = model_noreg.evaluate(X_train, Y_train)

model_reg = Sequential()
model_reg.add(Flatten())
model_reg.add(Dense(28*28, 20, activation='sigmoid'))
model_reg.add(ActivityRegularization(activity_l1(0.1)))
model_reg.add(Dense(20, 10, activation='sigmoid'))

model_reg.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model_reg.fit(X_train, Y_train)

score_reg = model_reg.evaluate(X_test, Y_test)
score_train_reg = model_reg.evaluate(X_train, Y_train)

print
print
print "Overfitting without regularisation: %f - %f = %f" % ( score_noreg , score_train_noreg , score_noreg-score_train_noreg)
print "Overfitting with regularisation: %f - %f = %f" % ( score_reg , score_train_reg , score_reg-score_train_reg)
