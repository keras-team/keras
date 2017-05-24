'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function
import numpy as np
import keras.backend as K

np.random.seed(1337)  # for reproducibility
import logging
import os

from keras.layers import *
from keras.models import model_from_json, Model
from keras.optimizers import Adam, RMSprop, Nadam, Adadelta, SGD, Adagrad, Adamax
from keras.regularizers import l2
from keras_wrapper.cnn_model import Model_Wrapper
from keras_wrapper.extra.regularize import Regularize
import argparse
import ast
import copy
import sys
import time
from timeit import default_timer as timer
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Model, Sequential
from keras.layers import *
from keras.datasets import imdb

max_features = 20000
maxlen = 20  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')

input1 = Input(name='Input_1', shape=tuple([None]), dtype='int32')
# If we don't mask the emb, the error doesn't appear
# If we mask it, it appears
emb =  Embedding(max_features, 128, dropout=0.2, mask_zero=False)(input1)
lstm  = AttLSTMCond(128, dropout_W=0.2, dropout_U=0.2)([emb, emb])
masked_lstm = RemoveMask()(lstm)
out_lstm = Dense(1, activation='sigmoid')(masked_lstm)
model = Model(input=input1, output=out_lstm)
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam')#,
              #metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
          validation_data=(X_test, y_test))
