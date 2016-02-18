'''Train a simple deep NN on the MNIST dataset.

Get to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1337)  # for reproducibility

import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers.variational import VariationalDense as VAE


batch_size = 128
nb_classes = 10
nb_epoch = 100
code_size = 100

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

encoder = Sequential()
encoder.add(Dense(1000, input_shape=(784,), activation='tanh'))
encoder.add(BatchNormalization())
encoder.add(Dense(1000, activation='tanh'))
encoder.add(BatchNormalization())
encoder.add(VAE(code_size, batch_size=batch_size, activation="linear"))
# encoder.add(Dense(code_size, activation='linear'))
decoder = Sequential()
decoder.add(Dense(1000, input_shape=(code_size,), activation='tanh'))
decoder.add(Dense(1000, activation='tanh'))
decoder.add(Dense(784, activation='softmax'))

model = Sequential()
model.add(encoder)
model.add(decoder)
model.compile(loss='binary_crossentropy', optimizer="adam")

model.fit(X_train, X_train,
          batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=2,
          validation_data=(X_test, X_test))
score = model.evaluate(X_test, X_test,
                       show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Visualize samples
sample = K.function([decoder.get_input()], [decoder.get_output()])

noise = np.random.normal(0, 1, size=(batch_size, code_size))
sampled = sample([noise])
for i in xrange(9):
    plt.subplot(3, 3, i)
    plt.imshow(sampled[0][i].reshape((28, 28)), cmap='gray')
    plt.axis("off")
plt.show()
