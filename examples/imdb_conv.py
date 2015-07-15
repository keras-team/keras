from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb

# set parameters:
max_features = 5000
maxlen = 100
batch_size = 10
embedding_dims = 100
nb_filters = 250
filter_length = 3
hidden_dims = 250
nb_epochs = 10

print("Loading data...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
                                                      test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features, embedding_dims))

# we add a Convolution1D, which will learn nb_filters
# word group filters of size filter_length:
model.add(Convolution1D(input_dim=embedding_dims,
                        nb_filter=nb_filters,
                        filter_length=filter_length,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1))

# we use standard max pooling (halving the output of the previous layer):
model.add(MaxPooling1D(pool_length=2))

# We flatten the output of the conv layer, so that we can add a vanilla dense layer:
model.add(Flatten())

# Computing the output shape of a conv layer can be tricky;
# for an good tutorial, see: http://cs231n.github.io/convolutional-networks/
output_size = nb_filters * (((maxlen - filter_length)/1)+1)/2

# We add a vanilla hidden layer:
model.add(Dense(output_size, hidden_dims))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(hidden_dims, 1))
model.add(Activation('sigmoid'))

# we use some mild clipping to protect our gradients from vanishing:
rms = RMSprop(clipnorm=0.1)
model.compile(loss='binary_crossentropy', optimizer=rms, class_mode="binary")

# The performance of this toy model seems comparable to that of the LSTM in imdb_lstm.py,
# reaching an acceptable 83.16% on the test data at epoch 10.
# One epoch took around 268 seconds on a Quadro K600.
print("Train...")
for e in range(nb_epochs):
    print(">>> Epoch ", e+1)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, show_accuracy=True)
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
    print('\t- Test loss:', score)
    print('\t- Test accuracy:', acc)
