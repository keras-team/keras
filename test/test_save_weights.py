from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

import sys
sys.setrecursionlimit(10000) # to be able to pickle Theano compiled functions

import pickle, numpy

def create_model():
    model = Sequential()
    model.add(Dense(256, 2048, init='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, 2048, init='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, 2048, init='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, 2048, init='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, 256, init='uniform', activation='linear'))
    return model

model = create_model()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=sgd)

pickle.dump(model, open('/tmp/model.pkl', 'wb'))
model.save_weights('/tmp/model_weights.hdf5')

model_loaded = create_model()
model_loaded.load_weights('/tmp/model_weights.hdf5')

for k in range(len(model.layers)):
    weights_orig = model.layers[k].get_weights()
    weights_loaded = model_loaded.layers[k].get_weights()
    for x, y in zip(weights_orig, weights_loaded):
        if numpy.any(x != y):
            raise ValueError('Loaded weights are different from pickled weights!')


