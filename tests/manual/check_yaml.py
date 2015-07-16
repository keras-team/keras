from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.utils.test_utils import get_test_data
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb
from keras.models import model_from_yaml

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
model.add(Dense(128, 1, W_regularizer='identity', b_constraint='maxnorm'))
model.add(Activation('sigmoid'))

model.get_config(verbose=1)

#####################################
# save model w/o parameters to yaml #
#####################################

yaml_no_params = model.to_yaml()

no_param_model = model_from_yaml(yaml_no_params)
no_param_model.get_config(verbose=1)

######################################
# save multi-branch sequential model #
######################################

seq = Sequential()
seq.add(Merge([model, model], mode='sum'))
seq.get_config(verbose=1)
merge_yaml = seq.to_yaml()
merge_model = model_from_yaml(merge_yaml)

large_model = Sequential()
large_model.add(Merge([seq,model], mode='concat'))
large_model.get_config(verbose=1)
large_model.to_yaml()

####################
# save graph model #
####################

X = np.random.random((100, 32))
X2 = np.random.random((100, 32))
y = np.random.random((100, 4))
y2 = np.random.random((100,))

(X_train, y_train), (X_test, y_test) = get_test_data(nb_train=1000, nb_test=200, input_shape=(32,),
    classification=False, output_shape=(4,))

graph = Graph()

graph.add_input(name='input1', ndim=2)

graph.add_node(Dense(32, 16), name='dense1', input='input1')
graph.add_node(Dense(32, 4), name='dense2', input='input1')
graph.add_node(Dense(16, 4), name='dense3', input='dense1')

graph.add_output(name='output1', inputs=['dense2', 'dense3'], merge_mode='sum')
graph.compile('rmsprop', {'output1':'mse'})

graph.get_config(verbose=1)

history = graph.fit({'input1':X_train, 'output1':y_train}, nb_epoch=10)
original_pred = graph.predict({'input1':X_test})

graph_yaml = graph.to_yaml()
graph.save_weights('temp.h5', overwrite=True)

reloaded_graph = model_from_yaml(graph_yaml)
reloaded_graph.load_weights('temp.h5')
reloaded_graph.get_config(verbose=1)

reloaded_graph.compile('rmsprop', {'output1':'mse'})
new_pred = reloaded_graph.predict({'input1':X_test})

assert(new_pred['output1'][3][1] == original_pred['output1'][3][1])
