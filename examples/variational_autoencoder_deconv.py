'''This script demonstrates how to build a variational autoencoder with Keras and deconvolution layers.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Convolution2D, TransposedConvolution2D, MaxPooling2D
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.optimizers import Adam, RMSprop
from keras.datasets import mnist

# input image dimensions
img_rows, img_cols, img_chns = 28, 28, 1
# number of convolutional filters to use
nb_filters = 32
# convolution kernel size
nb_conv = 3

batch_size = 16
original_dim = (img_chns, img_rows, img_cols)
latent_dim = 2
intermediate_dim = 128
epsilon_std = 0.01
nb_epoch = 5


def pp(l, x):
    print(l.eval({x:np.random.randn(batch_size, img_chns, img_rows, img_cols).astype('float32')}))


x = Input(batch_shape=(batch_size,) + original_dim)
pp(x.shape, x)
c = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', activation='relu')(x)
pp(c.shape, x)
f = Flatten()(c)
pp(f.shape, x)
h = Dense(intermediate_dim, activation='relu')(f)
pp(h.shape, x)

z_mean = Dense(latent_dim)(h)
pp(z_mean.shape, x)
z_log_std = Dense(latent_dim)(h)
pp(z_log_std.shape, x)

def sampling(args):
    z_mean, z_log_std = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_std) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_std])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_std])
pp(z.shape, x)

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
pp(decoder_h(z).shape, x)
decoder_f = Dense(nb_filters*img_rows*img_cols, activation='relu')
pp(decoder_f(decoder_h(z)).shape, x)
decoder_c = Reshape((nb_filters, img_rows, img_cols))
pp(decoder_c(decoder_f(decoder_h(z))).shape, x)
decoder_mean = TransposedConvolution2D(img_chns, nb_conv, nb_conv, 
                                       (batch_size, img_chns, img_rows, img_cols), border_mode='same')
pp(decoder_mean(decoder_c(decoder_f(decoder_h(z)))).shape, x)

h_decoded = decoder_h(z)
f_decoded = decoder_f(h_decoded)
c_decoded = decoder_c(f_decoded)
x_decoded_mean = decoder_mean(c_decoded)


def vae_loss(x, x_decoded_mean):
    # NOTE: binary_crossentropy expects a batch_size by dim for x and x_decoded_mean, so we MUST flatten these!!!!!
    x = Flatten()(x)
    x_decoded_mean = Flatten()(x_decoded_mean)
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, x_decoded_mean)
opt = RMSprop()
vae.compile(optimizer=opt, loss=vae_loss)

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')[:,None,:,:] / 255.
x_test = x_test.astype('float32')[:,None,:,:] / 255.

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
_c_decoded = decoder_c(_f_decoded)
_x_decoded_mean = decoder_mean(_c_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# we will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]]) * epsilon_std
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()
