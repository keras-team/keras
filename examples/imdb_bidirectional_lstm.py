from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb

from sklearn.metrics import accuracy_score as accuracy

'''
    Train a Bidirectional LSTM on the IMDB sentiment classification task.
    The dataset is actually too small for LSTM to be of any advantage
    compared to simpler, much faster methods such as TF-IDF+LogReg.
    Notes:
    - RNNs are tricky. Choice of batch size is important,
    choice of loss and optimizer is critical, etc.
    Some configurations won't converge.
    - LSTM loss decrease patterns during training can be quite different
    from what you see with CNNs/MLPs/etc.
    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py
'''

max_features = 20000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print("Loading data...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)
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
model.add_input(name='input', input_shape=(1,), dtype=int)
model.add_node(Embedding(max_features, 128, input_length=maxlen), name='embedding', input='input')
model.add_node(LSTM(64), name='forward', input='embedding' )  # You can change these two layers with GRU
model.add_node(LSTM(64, backwards=True), name='backward', input='embedding' )
model.add_node(Dropout(0.5), name='drop', inputs=['forward', 'backward'])
model.add_node(Dense(1, activation='sigmoid'), name='out_layer', input='drop')
model.add_output(name='output', input='out_layer')

# try using different optimizers and different optimizer configs
model.compile('adam',{'output':'binary_crossentropy'})

print("Train...")
model.fit({'input':X_train, 'output':y_train}, batch_size=batch_size, nb_epoch=4)
acc = accuracy(y_test, np.round(np.array(model.predict({'input':X_test}, batch_size=batch_size)['output'])))
print('Test score:', acc)

