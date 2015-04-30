from __future__ import absolute_import
from __future__ import print_function
import keras
from keras.datasets import mnist
import keras.models
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.regularizers import l2, l1
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, RMSprop

'''
    Train a (fairly simple) deep NN on the MNIST dataset.

'''

batch_size = 100
nb_classes = 10
nb_epoch = 200

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data(test_split=0.1)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(784, 100, W_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(100, 100))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(100, 10, W_constraint=maxnorm(3)))
model.add(Activation('softmax'))


# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=10)
score = model.evaluate(X_test, Y_test, batch_size=batch_size)
print('Test score:', score)
