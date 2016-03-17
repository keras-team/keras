'''This scripts implements Kim's paper "Convolutional Neural Networks for Sentence Classification"

Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_cnn.py

Get to 0.835 test accuracy after 2 epochs. 100s/epoch on K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras.layers import containers
from keras.constraints import MaxNorm


# set parameters:
max_features = 5000
maxlen = 100
batch_size = 32
embedding_dims = 100
nb_filter = 250
filter_length = [3, 4, 5]
hidden_dims = 250
nb_epoch = 2

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
                                                      test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
# This dropout layer is not adopted in the original paper
model.add(Dropout(0.25))

c = containers.Graph()
c.add_input(name='input', input_shape=(maxlen, embedding_dims))
inps = []
for i in filter_length:
    c.add_node(containers.Sequential([Convolution1D(nb_filter=nb_filter,
                                                    filter_length=i,
                                                    border_mode='valid',
                                                    activation='relu',
                                                    subsample_length=1,
                                                    input_shape=(maxlen, embedding_dims),),
                                      MaxPooling1D(pool_length=maxlen-i+1),
                                      Flatten()]),
               name='Conv{}'.format(i), input='input')
    inps.append('Conv{}'.format(i))

if len(inps) == 1:
    c.add_output('output', input=inps[0])
else:
    c.add_output('output', inputs=inps)

model.add(c)

# We flatten the output of the conv layer,
# so that we can add a vanilla dense layer:
model.add(Flatten())

# Add dropout at penultimate layer
model.add(Dropout(0.5))

# Fully connected with clipping regularization
model.add(Dense(1,  W_constraint=MaxNorm(m=3, axis=0)))
model.add(Activation('sigmoid'))

# The paper adopt adadelta.
# Here, we adopt adagrad, which achieves higher accuracy in 2 epoch
model.compile(loss='binary_crossentropy',
              optimizer='adagrad')
model.fit(X_train, y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, show_accuracy=True,
          validation_data=(X_test, y_test))
