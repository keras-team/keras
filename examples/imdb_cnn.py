'''This example demonstrates the use of Convolution1D for text classification.

Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # For reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.datasets import imdb


# Set parameters.
NUM_WORDS = 5000
MAXLEN = 400
BATCH_SIZE = 32
EMBEDDING_DIMS = 50
NB_FILTER = 250
FILTER_LENGTH = 3
HIDDEN_DIMS = 250
NB_EPOCH = 2

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=NUM_WORDS)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad (and truncate) sequences to make them (num samples, MAXLEN)')
X_train = sequence.pad_sequences(X_train, maxlen=MAXLEN)
X_test = sequence.pad_sequences(X_test, maxlen=MAXLEN)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()

# Start off with an efficient embedding layer which maps
# our vocab indices into EMBEDDING_DIMS dimensions.
model.add(Embedding(NUM_WORDS,
                    EMBEDDING_DIMS,
                    input_length=MAXLEN,
                    dropout=0.2))

# Add a Convolution1D, which will learn NB_FILTER
# word group filters of size FILTER_LENGTH:
model.add(Convolution1D(nb_filter=NB_FILTER,
                        filter_length=FILTER_LENGTH,
                        border_mode='valid',
                        activation='relu'))

# Get max value for each filter.
model.add(GlobalMaxPooling1D())

# Add a vanialla hidden layer.
model.add(Dense(HIDDEN_DIMS))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# Project onto a single unit output layer, and squash it with a sigmoid.
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          nb_epoch=NB_EPOCH,
          validation_data=(X_test, y_test))
