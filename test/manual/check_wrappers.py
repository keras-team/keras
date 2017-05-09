from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np

nb_classes = 10
batch_size = 128
nb_epoch = 1

max_train_samples = 5000
max_test_samples = 1000

np.random.seed(1337) # for reproducibility

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000,784)[:max_train_samples]
X_test = X_test.reshape(10000,784)[:max_test_samples]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)[:max_train_samples]
Y_test = np_utils.to_categorical(y_test, nb_classes)[:max_test_samples]

#############################
# scikit-learn wrapper test #
#############################
print('Beginning scikit-learn wrapper test')

print('Defining model')
model = Sequential()
model.add(Dense(784, 50))
model.add(Activation('relu'))
model.add(Dense(50, 10))
model.add(Activation('softmax'))

print('Creating wrapper')
classifier = KerasClassifier(model)

print('Fitting model')
classifier.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)

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
classifier.set_params(optimizer='sgd', loss='mse')
print(classifier.get_params())

print('Testing attributes')
print('Classes')
print(classifier.classes_)
print('Config')
print(classifier.config_)
print('Weights')
print(classifier.weights_)

print('Test script complete.')
