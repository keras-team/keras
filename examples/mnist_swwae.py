'''Trains a stacked what-where autoencoder built on residual blocks on the
MNIST dataset.  It exemplifies two influential methods that have been developed
in the past few years.

The first is the idea of properly "unpooling." During any max pool, the
exact location (the "where") of the maximal value in a pooled receptive field
is lost, however it can be very useful in the overall reconstruction of an
input image.  Therefore, if the "where" is handed from the encoder
to the corresponding decoder layer, features being decoded can be "placed" in
the right location, allowing for reconstructions of much higher fidelity.

References:
[1]
"Visualizing and Understanding Convolutional Networks"
Matthew D Zeiler, Rob Fergus
https://arxiv.org/abs/1311.2901v3

[2]
"Stacked What-Where Auto-encoders"
Junbo Zhao, Michael Mathieu, Ross Goroshin, Yann LeCun
https://arxiv.org/abs/1506.02351v8

The second idea exploited here is that of residual learning.  Residual blocks
ease the training process by allowing skip connections that give the network
the ability to be as linear (or non-linear) as the data sees fit.  This allows
for much deep networks to be easily trained.  The residual element seems to
be advantageous in the context of this example as it allows a nice symmetry
between the encoder and decoder.  Normally, in the decoder, the final
projection to the space where the image is reconstructed is linear, however
this does not have to be the case for a residual block as the degree to which
its output is linear or non-linear is determined by the data it is fed.
However, in order to cap the reconstruction in this example, a hard softmax is
applied as a bias because we know the MNIST digits are mapped to [0,1].

References:
[3]
"Deep Residual Learning for Image Recognition"
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
https://arxiv.org/abs/1512.03385v1

[4]
"Identity Mappings in Deep Residual Networks"
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
https://arxiv.org/abs/1603.05027v3

'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Activation, merge
from keras.layers import UpSampling2D, Convolution2D, MaxPooling2D
from keras.layers import Input, BatchNormalization
import matplotlib.pyplot as plt
import keras.backend as K


def convresblock(x, nfeats=8, ksize=3, nskipped=2):
    ''' The proposed residual block from [4]'''
    y0 = Convolution2D(nfeats, ksize, ksize, border_mode='same')(x)
    y = y0
    for i in range(nskipped):
        y = BatchNormalization(mode=0, axis=1)(y)
        y = Activation('relu')(y)
        y = Convolution2D(nfeats, ksize, ksize, border_mode='same')(y)
    return merge([y0, y], mode='sum')


def getwhere(x):
    ''' Calculate the "where" mask that contains switches indicating which
    index contained the max value when MaxPool2D was applied.  Using the
    gradient of the sum is a nice trick to keep everything high level.'''
    y_prepool, y_postpool = x
    return K.gradients(K.sum(y_postpool), y_prepool)

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# The size of the kernel used for the MaxPooling2D
pool_size = 2
# The total number of feature maps at each layer
nfeats = [8, 16, 32, 64, 128]
# The sizes of the pooling kernel at each layer
pool_sizes = np.array([1, 1, 1, 1, 1]) * pool_size
# The convolution kernel size
ksize = 3
# Number of epochs to train for
nb_epoch = 5
# Batch size during training
batch_size = 128

if pool_size == 2:
    # if using a 5 layer net of pool_size = 2
    X_train = np.pad(X_train, [[0, 0], [0, 0], [2, 2], [2, 2]],
                     mode='constant')
    X_test = np.pad(X_test, [[0, 0], [0, 0], [2, 2], [2, 2]], mode='constant')
    nlayers = 5
elif pool_size == 3:
    # if using a 3 layer net of pool_size = 3
    X_train = X_train[:, :, :-1, :-1]
    X_test = X_test[:, :, :-1, :-1]
    nlayers = 3
else:
    import sys
    sys.exit("Script supports pool_size of 2 and 3.")

# Shape of input to train on (note that model is fully convolutional however)
input_shape = X_train.shape[1:]
# The final list of the size of axis=1 for all layers, including input
nfeats_all = [input_shape[0]] + nfeats

# First build the encoder, all the while keeping track of the "where" masks
img_input = Input(shape=input_shape)

# We push the "where" masks to the following list
wheres = [None] * nlayers
y = img_input
for i in range(nlayers):
    y_prepool = convresblock(y, nfeats=nfeats_all[i + 1], ksize=ksize)
    y = MaxPooling2D(pool_size=(pool_sizes[i], pool_sizes[i]))(y_prepool)
    wheres[i] = merge([y_prepool, y], mode=getwhere,
                      output_shape=lambda x: x[0])

# Now build the decoder, and use the stored "where" masks to place the features
for i in range(nlayers):
    ind = nlayers - 1 - i
    y = UpSampling2D(size=(pool_sizes[ind], pool_sizes[ind]))(y)
    y = merge([y, wheres[ind]], mode='mul')
    y = convresblock(y, nfeats=nfeats_all[ind], ksize=ksize)

# Use hard_simgoid to clip range of reconstruction
y = Activation('hard_sigmoid')(y)

# Define the model and it's mean square error loss, and compile it with Adam
model = Model(img_input, y)
model.compile('adam', 'mse')

# Fit the model
model.fit(X_train, X_train, validation_data=(X_test, X_test),
          batch_size=batch_size, nb_epoch=nb_epoch)

# Plot
X_recon = model.predict(X_test[:25])
X_plot = np.concatenate((X_test[:25], X_recon), axis=1)
X_plot = X_plot.reshape((5, 10, input_shape[-2], input_shape[-1]))
X_plot = np.vstack([np.hstack(x) for x in X_plot])
plt.figure()
plt.axis('off')
plt.title('Test Samples: Originals/Reconstructions')
plt.imshow(X_plot, interpolation='none', cmap='gray')
plt.savefig('reconstructions.png')
