'''Trains DCGAN on MNIST using Keras

DCGAN is a Generative Adversarial Network (GAN) using CNN.
The generator tries to fool the discriminator by generating fake images.
The discriminator learns to discriminate real from fake images.
The generator + discriminator form an adversarial network.
DCGAN trains the discriminator and adversarial networks alternately.
During training, not only the discriminator learns to distinguish real from
fake images, it also coaches the generator part of the adversarial on how
to improve its ability to generate fake images.

[1] Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional
generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import math
import matplotlib.pyplot as plt


def generator(inputs, image_size):
    """Build a Generator Model

    Stacks of BN-ReLU-Conv2DTranpose to generate fake images
    Output activation is sigmoid instead of tanh in [1].
    Sigmoid converges easily.

    # Arguments
        inputs (Layer): Input layer of the generator (the z-vector)
        image_size: Target size of one side (assuming square image)

    # Returns
        Model: Generator Model
    """
    image_resize = image_size // 4
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]

    x = Dense(image_resize * image_resize * layer_filters[0])(inputs)
    x = Reshape((image_resize, image_resize, layer_filters[0]))(x)

    for filters in layer_filters:
        if filters > layer_filters[-2]:
            strides = 2
        else:
            strides = 1
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same')(x)

    x = Activation('sigmoid')(x)
    generator = Model(inputs, x, name='generator')
    return generator


def discriminator(inputs):
    """Build a Discriminator Model

    Stacks of LeakyReLU-Conv2D to discriminate real from fake
    The network does not converge with BN so it is not used here
    unlike in [1]

    # Arguments
        inputs (Layer): Input layer of the discriminator (the image)

    # Returns
        Model: Discriminator Model
    """
    kernel_size = 5
    layer_filters = [32, 64, 128, 256]

    x = inputs
    for filters in layer_filters:
        if filters == layer_filters[-1]:
            strides = 1
        else:
            strides = 2
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same')(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    discriminator = Model(inputs, x, name='discriminator')
    return discriminator


def train(models,
          x_train,
          batch_size=256,
          train_steps=10000,
          latent_size=100):
    """Train the Discriminator and Adversarial Networks

    Alternately train Discriminator and Adversarial networks by batch
    Discriminator is trained first with properly real and fake images
    Adversarial is trained next with fake images pretending to be real
    Generate sample images per save_interval

    # Arguments
        models (list): Generator, Discriminator, Adversarial models
        x_train (tensor): Train images
        batch_size (int): Batch size in train_on_batch
        train_steps (int): Number of steps of training
        latent_size (int): The z-vector dim

    """
    generator, discriminator, adversarial = models
    save_interval = 500
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    for i in range(train_steps):
        # Pick random real images
        rand_indexes = np.random.randint(0, x_train.shape[0], size=batch_size)
        train_images = x_train[rand_indexes, :, :, :]
        # Generate fake images
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_images = generator.predict(noise)
        x = np.concatenate((train_images, fake_images))
        # Label real and fake images
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0
        # Train the Discriminator network
        metrics = discriminator.train_on_batch(x, y)
        loss = metrics[0]
        accuracy = metrics[1]
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, accuracy)

        # Generate fake images
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        # Label fake images as real
        y = np.ones([batch_size, 1])
        # Train the Adversarial network
        metrics = adversarial.train_on_batch(noise, y)
        loss = metrics[0]
        accuracy = metrics[1]
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, accuracy)
        print(log)
        if (i + 1) % save_interval == 0:
            if (i + 1) == train_steps:
                show = True
            else:
                show = False
            plot_images(generator,
                        noise_input=noise_input,
                        show=show,
                        step=(i + 1))


def plot_images(generator,
                noise_input,
                show=False,
                step=0):
    """Generate fake images and plot them

    For visualization purposes, generate fake images
    then plot them in a square grid

    # Arguments
        generator (Model): The Generator Model for fake images generation
        noise_input (ndarray): Array of z-vectors
        show (bool): Whether to show plot or not
        step (int): Appended to filename of the save images

    """
    filename = "mnist_dcgan_%d.png" % step
    images = generator.predict(noise_input)
    plt.figure(figsize=(2.4, 2.4))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_input.shape[0]))
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = images[i, :, :, :]
        image = np.reshape(image, [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close('all')


# MNIST dataset
(x_train, _), (_, _) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255

# The latent or z vector is 100-dim
latent_size = 100
input_shape = (image_size, image_size, 1)

# Build Discriminator Model
inputs = Input(shape=input_shape, name='discriminator_input')
discriminator = discriminator(inputs)
# [1] uses Adam, but discriminator converges easily with RMSprop
optimizer = RMSprop(lr=0.0002)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
discriminator.summary()

# Build Generator Model
input_shape = (latent_size, )
inputs = Input(shape=input_shape, name='z_input')
generator = generator(inputs, image_size)
generator.summary()

# Build Adversarial Model = Generator + Discriminator
optimizer = RMSprop(lr=0.0001)
adversarial = Model(inputs, discriminator(generator(inputs)), name='dcgan')
adversarial.compile(loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])
adversarial.summary()

# Train Discriminator and Adversarial Networks
models = (generator, discriminator, adversarial)
train(models, x_train, latent_size=latent_size)
