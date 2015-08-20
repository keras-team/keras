from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras.wrappers.scikit_learn import *
import numpy as np

batch_size = 128
nb_epoch = 1

nb_classes = 10
max_train_samples = 5000
max_test_samples = 1000

np.random.seed(1337) # for reproducibility

############################################
# scikit-learn classification wrapper test #
############################################
print('Beginning scikit-learn classification wrapper test')

print('Loading data')
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)[:max_train_samples]
X_test = X_test.reshape(10000, 784)[:max_test_samples]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)[:max_train_samples]
Y_test = np_utils.to_categorical(y_test, nb_classes)[:max_test_samples]

print('Defining model')
model = Sequential()
model.add(Dense(784, 50))
model.add(Activation('relu'))
model.add(Dense(50, 10))
model.add(Activation('softmax'))

print('Creating wrapper')
classifier = KerasClassifier(model, train_batch_size=batch_size, nb_epoch=nb_epoch)

print('Fitting model')
classifier.fit(X_train, Y_train)

print('Testing score function')
score = classifier.score(X_train, Y_train)
print('Score: ', score)

print('Testing predict function')
preds = classifier.predict(X_test)
print('Preds.shape: ', preds.shape)

print('Testing predict proba function')
proba = classifier.predict_proba(X_test)
print('Proba.shape: ', proba.shape)

print('Testing get params')
print(classifier.get_params())

print('Testing set params')
classifier.set_params(optimizer='sgd', loss='binary_crossentropy')
print(classifier.get_params())

print('Testing attributes')
print('Classes')
print(classifier.classes_)
print('Config')
print(classifier.config_)
print('Weights')
print(classifier.weights_)
print('Compiled model')
print(classifier.compiled_model_)

########################################
# scikit-learn regression wrapper test #
########################################
print('Beginning scikit-learn regression wrapper test')

print('Generating data')
X_train = np.random.random((5000, 100))
X_test = np.random.random((1000, 100))
y_train = np.random.random(5000)
y_test = np.random.random(1000)

print('Defining model')
model = Sequential()
model.add(Dense(100, 50))
model.add(Activation('relu'))
model.add(Dense(50, 1))
model.add(Activation('linear'))

print('Creating wrapper')
regressor = KerasRegressor(model, train_batch_size=batch_size, nb_epoch=nb_epoch)

print('Fitting model')
regressor.fit(X_train, y_train)

print('Testing score function')
score = regressor.score(X_train, y_train)
print('Score: ', score)

print('Testing predict function')
preds = regressor.predict(X_test)
print('Preds.shape: ', preds.shape)

print('Testing get params')
print(regressor.get_params())

print('Testing set params')
regressor.set_params(optimizer='sgd', loss='mean_absolute_error')
print(regressor.get_params())

print('Testing attributes')
print('Config')
print(regressor.config_)
print('Weights')
print(regressor.weights_)
print('Compiled model')
print(regressor.compiled_model_)

print('Test script complete.')
