"""
Trains a DCGAN on the MNIST dataset.
"""

from __future__ import print_function, division
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, LeakyReLU, MinibatchDiscrimination
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Reshape
from keras.optimizers import Adam

np.random.seed(1337)  # for reproducibility


def set_trainable(net, boolean):
    net.trainable = boolean
    for layer in net.layers:
        layer.trainable = boolean

batch_size = 128
nb_epoch = 12
nb_zdim = 100

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
# here we only need the training data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_train /= 255
print('\nX_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')

optimizer = Adam(lr=2e-4, beta_1=0.5)
discrim_optimizer = optimizer

discriminator = Sequential()
discriminator.add(ZeroPadding2D(padding=(4, 4), input_shape=(1, 28, 28)))  # pad to 32x32 to get power of 2
discriminator.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                subsample=(2, 2), border_mode='same'))
discriminator.add(BatchNormalization(mode=2))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Convolution2D(nb_filters * 2, kernel_size[0], kernel_size[1],
                                subsample=(nb_pool, nb_pool), border_mode='same'))
discriminator.add(BatchNormalization(mode=2))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Convolution2D(nb_filters * 4, kernel_size[0], kernel_size[1],
                                subsample=(nb_pool, nb_pool), border_mode='same'))
discriminator.add(BatchNormalization(mode=2))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Flatten())
discriminator.add(MinibatchDiscrimination(5, 3))
discriminator.add(Dense(output_dim=1))
discriminator.add(Activation('sigmoid'))

discriminator.compile(loss='binary_crossentropy', optimizer=discrim_optimizer)

print('Setting up generator')
generator = Sequential()
generator.add(Dense(output_dim=(nb_filters * 4) * 4 * 4, input_dim=nb_zdim))
generator.add(BatchNormalization(mode=2))
generator.add(Activation('relu'))
generator.add(Reshape(target_shape=(nb_filters * 4, 4, 4)))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(nb_filters * 2, kernel_size[0], kernel_size[1], border_mode='same'))
generator.add(BatchNormalization(mode=2))
generator.add(Activation('relu'))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same'))
generator.add(BatchNormalization(mode=2))
generator.add(Activation('relu'))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(1, kernel_size[0], kernel_size[1], border_mode='same'))
generator.add(Activation('tanh'))

generator.compile(loss='binary_crossentropy', optimizer=optimizer)

print('Setting up combined GAN')
gan = Sequential()
gan.add(generator)
set_trainable(discriminator, False)
gan.add(discriminator)

gan.compile(loss='binary_crossentropy', optimizer=optimizer)

losses = {'discriminator': [], 'generator': []}
zeros = np.zeros(batch_size)
ones = np.ones(batch_size)
nb_batches = int((X_train.shape[0]-batch_size)/batch_size)

for i_epoch in range(nb_epoch):
    print('\nEpoch {}/{}'.format(i_epoch, nb_epoch))
    for i_start in range(0, X_train.shape[0]-batch_size, batch_size):
        set_trainable(discriminator, True)

        noise_gen = np.random.uniform(-1, 1, size=(batch_size, nb_zdim))
        X_generated_batch = generator.predict(noise_gen)
        X_train_batch = X_train[i_start:i_start+batch_size, :, :, :]

        X_batch = np.concatenate([X_generated_batch, X_train_batch], axis=0)
        y = np.concatenate([zeros, ones], axis=0)
        discrim_loss = discriminator.train_on_batch(X_batch, y)
        losses['discriminator'].append(discrim_loss)

        set_trainable(discriminator, False)
        noise_train = np.random.uniform(-1, 1, size=(batch_size, nb_zdim))
        gen_loss = gan.train_on_batch(noise_train, zeros)
        losses['generator'].append(gen_loss)

        print('\rbatch {} of {}... losses - D: {:.4f}, G: {:.4f}'.format(
            int(i_start/batch_size), nb_batches, discrim_loss, gen_loss), end='')
