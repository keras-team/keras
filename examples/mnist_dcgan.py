"""
Trains a DCGAN on the MNIST dataset.
"""

from __future__ import print_function, division
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LeakyReLU, MinibatchDiscrimination
from keras.layers import Convolution2D, UpSampling2D, Reshape
from keras.optimizers import Adam
import keras.backend as K
try:
    import matplotlib.pyplot as plt
    plotting_enabled = True
except ImportError:
    plotting_enabled = False

np.random.seed(42)  # for reproducibility


def append_minibatch_discrimination_features(activation, x, nb_kernels, kernel_dim):
    activation = K.reshape(activation, (-1, nb_kernels, kernel_dim))
    diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), axis=2)
    minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
    return K.concatenate([x, minibatch_features], 1)


def set_trainable(net, boolean):
    net.trainable = boolean
    for layer in net.layers:
        layer.trainable = boolean

batch_size = 128
nb_epoch = 5
nb_zdim = 100

# number of convolutional filters to use
nb_filters = 128
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
kernel_size = (5, 5)

# the data, shuffled and split between train and test sets
# here we only need the training data
(X_train_load, y_train), (X_test, y_test) = mnist.load_data()

X_train_load = X_train_load.reshape(X_train_load.shape[0], 1, 28, 28)
# pad MNIST data to 32x32 to have powers of 2
X_train = np.zeros([X_train_load.shape[0], 1, 32, 32], dtype=np.float32)
X_train[:, :, 2:-2, 2:-2] = X_train_load[:, :, :, :]
X_train /= 255.
print('\nX_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')

optimizer = Adam(lr=2e-4, beta_1=0.5)
discrim_optimizer = optimizer

# Ideally we would have BatchNormalization layers on all but the generator output
# and discriminator input layers, but I was not able to get this to work.
discriminator = Sequential()
discriminator.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], input_shape=(1, 32, 32),
                                subsample=(2, 2), border_mode='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Convolution2D(nb_filters * 2, kernel_size[0], kernel_size[1],
                                subsample=(nb_pool, nb_pool), border_mode='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Convolution2D(nb_filters * 4, kernel_size[0], kernel_size[1],
                                subsample=(nb_pool, nb_pool), border_mode='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Flatten())
discriminator.add(MinibatchDiscrimination(8, 4))
discriminator.add(Dense(output_dim=1))
discriminator.add(Activation('sigmoid'))

discriminator.compile(loss='binary_crossentropy', optimizer=discrim_optimizer)

generator = Sequential()
generator.add(Dense(output_dim=(nb_filters * 4) * 4 * 4, input_dim=nb_zdim))
generator.add(Activation('relu'))
generator.add(Reshape(target_shape=(nb_filters * 4, 4, 4)))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(nb_filters * 2, kernel_size[0], kernel_size[1], border_mode='same'))
generator.add(Activation('relu'))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same'))
generator.add(Activation('relu'))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(1, kernel_size[0], kernel_size[1], border_mode='same'))
generator.add(Activation('tanh'))

generator.compile(loss='binary_crossentropy', optimizer=optimizer)

gan = Sequential()
gan.add(generator)
set_trainable(discriminator, False)
gan.add(discriminator)

gan.compile(loss='binary_crossentropy', optimizer=optimizer)

losses = {'discriminator': [], 'generator': []}
zeros = np.zeros(batch_size)
ones = np.ones(batch_size)
nb_batches = int((X_train.shape[0]-batch_size)/batch_size)

print('Beginning training. First batch may take a while due to compilation.')
for i_epoch in range(nb_epoch):
    print('\nEpoch {}/{}'.format(i_epoch+1, nb_epoch))
    for i_start in range(0, X_train.shape[0]-batch_size, batch_size):
        set_trainable(discriminator, True)

        noise_gen = np.random.uniform(0, 1, size=(batch_size, nb_zdim))
        X_generated_batch = generator.predict(noise_gen)
        X_train_batch = X_train[i_start:i_start+batch_size, :, :, :]

        X_batch = np.concatenate([X_generated_batch, X_train_batch], axis=0)
        y = np.concatenate([ones, zeros], axis=0)
        discrim_loss = discriminator.train_on_batch(X_batch, y)
        losses['discriminator'].append(discrim_loss)

        set_trainable(discriminator, False)
        noise_train = np.random.uniform(0, 1, size=(batch_size, nb_zdim))
        gen_loss = gan.train_on_batch(noise_train, zeros)
        losses['generator'].append(gen_loss)

        print('\rbatch {} of {}... losses - D: {:.4f}, G: {:.4f}'.format(
            int(i_start/batch_size), nb_batches, discrim_loss.item(), gen_loss.item()), end='')

if plotting_enabled:
    print('Plotting examples. Will continue until program is interrupted.')
    while True:
        nb_examples = 5
        fig, ax = plt.subplots(2, 5, figsize=(12, 5))
        noise_gen = np.random.uniform(0, 1, size=(nb_examples, nb_zdim))
        X_generated = generator.predict(noise_gen)
        X_real = X_train[np.random.randint(0, X_train.shape[0], nb_examples), :, :, :]
        for i in range(nb_examples):
            ax[0, i].imshow(X_generated[i, 0, :, :], aspect='equal', cmap='Greys', interpolation='none')
            ax[1, i].imshow(X_real[i, 0, :, :], aspect='equal', cmap='Greys', interpolation='none')
        ax[0, 0].set_ylabel('Generated')
        ax[1, 0].set_ylabel('Real')
        plt.show()
        plt.close(fig)
