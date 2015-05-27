from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential, Autoencoder
from keras.layers.core import Dense, Dropout, Activation, Layer, Merge
from keras.optimizers import RMSprop
import theano.tensor as T
import numpy as np

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams()


(X_train, y_train), (_, _) = mnist.load_data()

X_train = X_train.reshape((X_train.shape[0],-1)).astype(float)/255.

encoder = Sequential()
decoder = Sequential()

n_hidden = 64

layer_enc = Dense(X_train.shape[1],n_hidden,W_regularizer=None)
layer_dec = Dense(n_hidden,X_train.shape[1])

# Tie the weights of encoder and decoder
layer_dec.params.remove(layer_dec.W)
layer_dec.W = T.transpose(layer_enc.W)

encoder.add(layer_enc)
encoder.add(Dropout(p=0.3))
encoder.add(Activation('relu'))
encoder.add(Activation('softmax'))

decoder.add(layer_dec)
decoder.add(Activation('relu'))

autoencoder = Autoencoder(encoder,decoder)

rms = RMSprop()
autoencoder.compile(loss='mean_squared_error', optimizer=rms)

autoencoder.fit(X_train, verbose=1, nb_epoch=10)

autoencoder.freeze_encoder()

testnet = Sequential()
testnet.add(Merge([autoencoder.encoder]))
testnet.add(Dense(n_hidden,10))
testnet.add(Activation('sigmoid'))

W_pretrain = layer_enc.W.get_value()

y_train_full = np.zeros((y_train.shape[0],10))
for n in range(len(y_train)):
    y_train_full[n,y_train[n]] = 1.
testnet.compile(loss='mean_squared_error', optimizer=rms)
testnet.fit(X_train,y_train_full,nb_epoch=10)

W_posttrain = layer_enc.W.get_value()

assert(np.abs(W_posttrain-W_pretrain).max() < 1e-9)
