'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # For reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

BATCH_SIZE = 128
NB_CLASSES = 10
NB_EPOCH = 12

# Input image dimensions.
IMG_ROWS, IMG_COLS = 28, 28
# Number of convolutional filters to use.
NB_FILTERS = 32
# Size of pooling area for max pooling.
POOL_SIZE = (2, 2)
# Convolution kernel size.
KERNEL_SIZE = (3, 3)

# The data, shuffled and split between train and test sets.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Input image is grayscale and the shape is is (nb_samples, 28, 28). Place
# channel dimension where appropriate.
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, IMG_ROWS, IMG_COLS)
    X_test = X_test.reshape(X_test.shape[0], 1, IMG_ROWS, IMG_COLS)
    input_shape = (1, IMG_ROWS, IMG_COLS)
else:
    X_train = X_train.reshape(X_train.shape[0], IMG_ROWS, IMG_COLS, 1)
    X_test = X_test.reshape(X_test.shape[0], IMG_ROWS, IMG_COLS, 1)
    input_shape = (IMG_ROWS, IMG_COLS, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

model = Sequential()

model.add(Convolution2D(NB_FILTERS, KERNEL_SIZE[0], KERNEL_SIZE[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(NB_FILTERS, KERNEL_SIZE[0], KERNEL_SIZE[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=POOL_SIZE))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
