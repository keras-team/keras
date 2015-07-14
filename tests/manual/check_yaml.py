from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb
from keras.models import sequential_from_yaml

'''
This is essentially the IMDB test. Deserialized models should yield 
the same config as the original one.
'''

max_features = 10000
maxlen = 100 
batch_size = 32

(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, 128)) 
model.add(Dropout(0.5))
model.add(Dense(128, 1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

model.get_config(verbose=1)

######################
# save model to yaml #
######################
yamlString = model.to_yaml()

recovered_model = sequential_from_yaml(yamlString)
recovered_model.get_config(verbose=1)

#####################################
# save model w/o parameters to yaml #
#####################################

yaml_no_params = model.to_yaml(storeParams=False)

no_param_model = sequential_from_yaml(yaml_no_params)
no_param_model.get_config(verbose=1)