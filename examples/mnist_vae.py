'''Train a deep Variational Autoencoder on the MNIST dataset.
Here, we want to learn hidden factors z from a dataset x, optimally we would do
that by sampling from the posterior p(z|x), but when that distribution is
unknown, VAE proposes calculating instead an approximation q(z|x) as a
parametirized approximation by maximizing the lower bound:

L = - KL(q(z|x) || p(z)) + E[log p(x|z)]

In practical terms, this means that we have to train a good decoder to maximize
E[log p(x|z)], which can be done by minimizing the error between true samples
and generated samples, and minimize the KL-divergence between q(z|x) and a prior
distribution p(z). Here we assume the prior is a zero mean, unit std Gaussian,
thus, the output of the encoder must have the same Gaussian distribution. This
Gaussian distribution constraint is imposed by the `keras.layers.VariationalDense`
layer.

After the model is trained, the decoder can be used to transform samples from a
zero mean, unit std Gaussian into realistic looking samples. We sample data
using this technique in the last lines of this script.

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
from keras.layers.variational import VariationalDense as VAE


batch_size = 128
nb_epoch = 300  # VAEs usually take long to converge
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

encoder = Sequential()
encoder.add(Dense(1000, input_shape=(784,), activation='tanh'))
encoder.add(BatchNormalization())
encoder.add(Dense(1000, activation='tanh'))
encoder.add(BatchNormalization())
encoder.add(VAE(code_size, activation="linear"))

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
