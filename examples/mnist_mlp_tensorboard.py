'''Trains NN with batch normalization
on the MNIST dataset.

Uses tensorboard callback to write cross entropy and accuracy
at epoch which can be visualized using tensorboard
'''


from __future__ import print_function
import numpy as np
np.random.seed(20)


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from keras.utils import np_utils


# Model parameters
batch_size = 256
nb_classes = 10
nb_epoch = 20


# Load MNIST data and shuffle & split into train & test
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

# Convert to one hot encoding of classes
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


# Neural network with no dropouts only Batch normalization
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

print(model.summary())


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callback to tensorboard to write logs
# Visualize model and its performance using tensorboard --logdir=_mnist
# write_graph=True if you want to visualize model
tensorboard = keras.callbacks.TensorBoard(log_dir="_mnist", write_graph=False, write_images=True)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test), callbacks=[tensorboard])


test_performance = model.evaluate(X_test, Y_test, verbose=0)
print('Test Categorical crossentropy:', test_performance[0])
print('Test accuracy:', test_performance[1])
