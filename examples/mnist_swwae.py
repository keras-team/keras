'''Trains a simple stacked what-where autoencoder on the MNIST dataset.'''

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

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convolutional res block
def convresblock(x,nfeats=8,ksize=3,nskipped=2):
    y0 = Convolution2D(nfeats,ksize,ksize,border_mode='same')(x)
    y = y0
    for i in range(nskipped):
        y = BatchNormalization(mode=0,axis=1)(y)
        y = Activation('relu')(y)
        y = Convolution2D(nfeats,ksize,ksize,border_mode='same')(y)
    return merge([y0,y],mode='sum')

def getwhere(x):
    y_prepool,y_postpool = x
    return K.gradients(K.sum(y_postpool),y_prepool)
    
def getwhereshape(input_shape):
    return input_shape[0]
    
pool_size = 2
nfeats = [16,32,64,128,256]
pool_sizes = np.array([1,1,1,1,1])*pool_size
ksize = 3
nb_epoch = 10
batch_size = 128
nfeats = [1]+nfeats

if pool_size == 2:
    # if using a 5 layer net of pool_size = 2
    X_train = np.pad(X_train,[[0,0],[0,0],[2,2],[2,2]],mode='constant')
    X_test = np.pad(X_test,[[0,0],[0,0],[2,2],[2,2]],mode='constant')
    nlayers = 5
elif pool_size == 3:
    # if using a 3 layer net of pool_size = 3
    X_train = X_train[:,:,:-1,:-1]
    X_test = X_test[:,:,:-1,:-1]    
    nlayers = 3
else:
    import sys
    sys.exit("Script supports pool_size of 2 and 3.")
    
input_shape = X_train.shape[1:]
x_in = Input(shape=input_shape)    
    
wheres = [None]*nlayers
# Create the encoder
y = x_in
for i in range(nlayers):
    y_prepool = convresblock(y,nfeats=nfeats[i+1],ksize=ksize)
    y = MaxPooling2D(pool_size=(pool_sizes[i],pool_sizes[i]))(y_prepool)
    wheres[i] = merge([y_prepool,y],mode=getwhere,output_shape=getwhereshape)
    
# Create the decoder
for i in range(nlayers):
    y = UpSampling2D(size=(pool_sizes[nlayers-1-i],pool_sizes[nlayers-1-i]))(y)
    y = merge([y,wheres[nlayers-1-i]],mode='mul')
    y = convresblock(y,nfeats=nfeats[nlayers-1-i],ksize=ksize)
    
# Use hard_simgoid to clip range of reconstruction    
y = Activation('hard_sigmoid')(y)    
    
model = Model(x_in,y)

model.compile('adam','mse')
model.optimizer.lr.set_value(.001)
model.fit(X_train,X_train,validation_data=(X_test,X_test),
          batch_size=batch_size,nb_epoch=nb_epoch)

# Plot
X_recon = model.predict(X_test[:25])
X_plot = np.concatenate((X_test[:25],X_recon),axis=1)
X_plot = X_plot.reshape((5,10,input_shape[-2],input_shape[-1]))
X_plot = np.vstack([np.hstack(x) for x in X_plot])
plt.figure()
plt.imshow(X_plot,interpolation='none',cmap='gray')
