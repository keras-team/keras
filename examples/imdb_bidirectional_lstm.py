from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils.np_utils import accuracy
from keras.models import Graph
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb

'''
    Train a Bidirectional LSTM on the IMDB sentiment classification task.

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_bidirectional_lstm.py

    Output after 4 epochs on CPU: ~0.8146
    Time per epoch on CPU (Core i7): ~150s.
'''

max_features = 20000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

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
y_train = np.array(y_train)
y_test = np.array(y_test)

print('Build model...')
model = Graph()
model.add_input(name='input', input_shape=(maxlen,), dtype=int)
model.add_node(Embedding(max_features, 128, input_length=maxlen),
               name='embedding', input='input')
model.add_node(LSTM(64), name='forward', input='embedding')
model.add_node(LSTM(64, go_backwards=True), name='backward', input='embedding')
model.add_node(Dropout(0.5), name='dropout', inputs=['forward', 'backward'])
model.add_node(Dense(1, activation='sigmoid'), name='sigmoid', input='dropout')
model.add_output(name='output', input='sigmoid')

# try using different optimizers and different optimizer configs
model.compile('adam', {'output': 'binary_crossentropy'})

print("Train...")
model.fit({'input': X_train, 'output': y_train},
          batch_size=batch_size,
          nb_epoch=4)
acc = accuracy(y_test,
               np.round(np.array(model.predict({'input': X_test},
                                               batch_size=batch_size)['output'])))
print('Test accuracy:', acc)
