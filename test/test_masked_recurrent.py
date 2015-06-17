# Dummy test data as input to RNN. This input is 3 timesteps long where the third timestep always matches the
# first. Without masking it should be able to learn it, with masking it should fail.

import numpy as np
from keras.utils.theano_utils import sharedX
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge
from keras.layers.recurrent import SimpleRNN
import theano

theano.config.exception_verbosity='high' 

# (nb_samples, timesteps, dimensions)
X = np.random.random_integers(20, size=(100000,3,2))

def my_eye(scale=1.0):
    def inner(shape):
        return sharedX(scale*np.eye(shape[0], shape[1]))
    return inner

unmasked_model = Sequential()
unmasked_model.add(SimpleRNN(2,3, activation='relu', return_sequences=True))
unmasked_model.add(SimpleRNN(3,3, activation='relu'))
unmasked_model.add(Dense(3,2))
unmasked_model.compile(loss='mse', optimizer='rmsprop', theano_mode=theano.compile.mode.FAST_COMPILE)
print("Compiled unmasked_model")

unmasked_model_b = Sequential()
unmasked_model_b.add(SimpleRNN(2,3, activation='relu', return_sequences=True))
unmasked_model_b.add(SimpleRNN(3,3, activation='relu'))
unmasked_model_b.add(Dense(3,2))
unmasked_model_b.compile(loss='mse', optimizer='rmsprop', theano_mode=theano.compile.mode.FAST_COMPILE)
print("Compiled unmasked_model")

masked_model = Sequential()
masked_model.add(SimpleRNN(2,3, activation='relu', time_mask=True, return_sequences=True))
masked_model.add(SimpleRNN(3,3, activation='relu', time_mask=True))
masked_model.add(Dense(3,2))
masked_model.compile(loss='mse', optimizer='rmsprop', theano_mode=theano.compile.mode.FAST_COMPILE)
print("Compiled masked_model")

# This masked model is expected to learn
masked_model_b = Sequential()
masked_model_b.add(SimpleRNN(2,3, activation='relu', time_mask=True, return_sequences=True))
masked_model_b.add(SimpleRNN(3,3, activation='relu', time_mask=True))
masked_model_b.add(Dense(3,2))
masked_model_b.compile(loss='mse', optimizer='rmsprop', theano_mode=theano.compile.mode.FAST_COMPILE)
print("Compiled masked_model")


mask = np.ones(X.shape)
mask[:,0,:] = 0 # mask out the first timestep
withmask = np.concatenate((X, mask), axis=1)

unmasked_model.fit(X, X[:,0,:], nb_epoch=1)
score = unmasked_model.evaluate(X, X[:,0,:])
if np.sqrt(score) > 5.744:
    raise Exception('Failed to learn to copy timestep 0, score %f' % score)

unmasked_model_b.fit(X[:,1:,:], X[:,1,:], nb_epoch=1)
score = unmasked_model_b.evaluate(X[:,1:,:], X[:,1,:])
if np.sqrt(score) > 5.744:
    raise Exception('Failed to learn to copy timestep 1, score %f' % score)

masked_model.fit(withmask, X[:,0,:], nb_epoch=1)
score = masked_model.evaluate(withmask, X[:,0,:])
if np.sqrt(score) < 5.744:
    raise Exception('Somehow learned to copy timestep 0 despite mask, score %f' % score)


masked_model_b.fit(withmask, X[:,1,:], nb_epoch=1)
score = masked_model_b.evaluate(withmask, X[:,1,:])
if np.sqrt(score) > 5.744:
    raise Exception('Failed to learn to copy timestep 1 in masked model, score %f' % score)

