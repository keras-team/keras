'''This script demonstrates how to build a variational autoencoder with Keras and deconvolution layers.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist

# input image dimensions
img_rows, img_cols, img_chns = 28, 28, 1
# number of convolutional filters to use
nb_filters = 64
# convolution kernel size
nb_conv = 3

batch_size = 128
original_dim = (img_chns, img_rows, img_cols)
latent_dim = 2
intermediate_dim = 128
epsilon_std = 1.0
nb_epoch = 5


x = Input(batch_shape=(batch_size,) + original_dim)
c_1 = Convolution2D(img_chns, 2, 2, border_mode='same', activation='relu')(x)
c_2 = Convolution2D(nb_filters, 2, 2,
                    border_mode='same', activation='relu',
                    subsample=(2, 2))(c_1)
c_3 = Convolution2D(nb_filters, nb_conv, nb_conv,
                    border_mode='same', activation='relu',
                    subsample=(1, 1))(c_2)
c_4 = Convolution2D(nb_filters, nb_conv, nb_conv,
                    border_mode='same', activation='relu',
                    subsample=(1, 1))(c_3)
f = Flatten()(c_4)
h = Dense(intermediate_dim, activation='relu')(f)

z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_f = Dense(nb_filters*14*14, activation='relu')
decoder_c_1 = Reshape((nb_filters, 14, 14))
decoder_c_2 = Deconvolution2D(nb_filters, nb_conv, nb_conv,
                              (batch_size, nb_filters, 14, 14),
                              border_mode='same',
                              subsample=(1, 1))
decoder_c_3 = Deconvolution2D(nb_filters, nb_conv, nb_conv,
                              (batch_size, nb_filters, 14, 14),
                              border_mode='same',
                              subsample=(1, 1))
decoder_mean = Deconvolution2D(nb_filters, 2, 2,
                               (batch_size, nb_filters, 29, 29),
                               border_mode='valid',
                               subsample=(2, 2))
decoder_mean_c = Convolution2D(img_chns, 2, 2, border_mode='valid', activation='relu')

h_decoded = decoder_h(z)
f_decoded = decoder_f(h_decoded)
c_1_decoded = decoder_c_1(f_decoded)
c_2_decoded = decoder_c_2(c_1_decoded)
c_3_decoded = decoder_c_3(c_2_decoded)
x_decoded_mean = decoder_mean(c_3_decoded)
x_decoded_mean = decoder_mean_c(x_decoded_mean)

def vae_loss(x, x_decoded_mean):
    # NOTE: binary_crossentropy expects a batch_size by dim for x and x_decoded_mean, so we MUST flatten these!
    x = K.flatten(x)
    x_decoded_mean = K.flatten(x_decoded_mean)
    xent_loss = 28 * 28 * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)
vae.summary()

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')[:, None, :, :] / 255.
x_test = x_test.astype('float32')[:, None, :, :] / 255.

end = len(x_test) - len(x_test) % batch_size
x_test = x_test[:end]

end = len(x_train) - len(x_train) % batch_size
x_train = x_train[:end]

end = len(y_test) - len(y_test) % batch_size
y_test = y_test[:end]

print(x_train.shape)

vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test))


# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_f_decoded = decoder_f(_h_decoded)
_c_1_decoded = decoder_c_1(_f_decoded)
_c_2_decoded = decoder_c_2(_c_1_decoded)
_c_3_decoded = decoder_c_3(_c_2_decoded)
_x_decoded_mean = decoder_mean(_c_3_decoded)
_x_decoded_mean_c = decoder_mean_c(_x_decoded_mean)
generator = Model(decoder_input, _x_decoded_mean_c)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# we will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = generator.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()
