# Dummy test data as input to RNN. This input is 3 timesteps long where the third timestep always matches the
# first. Without masking it should be able to learn it, with masking it should fail.

import numpy as np
from keras.utils.theano_utils import sharedX
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
import theano

theano.config.exception_verbosity='high' 

# (nb_samples, timesteps, dimensions)
X = np.random.random_integers(0, 3, size=(40000,3))

def my_eye(scale=1.0):
    def inner(shape):
        return sharedX(scale*np.eye(shape[0], shape[1]))
    return inner

unmasked_model = Sequential()
unmasked_model.add(Embedding(4, 2))
unmasked_model.add(SimpleRNN(2,3, activation='relu', return_sequences=True))
unmasked_model.add(SimpleRNN(3,3, activation='relu'))
unmasked_model.add(Dense(3,4, activation='softmax'))
unmasked_model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop', theano_mode=theano.compile.mode.FAST_COMPILE)
print("Compiled unmasked_model")

unmasked_model_b = Sequential()
unmasked_model_b.add(Embedding(4, 2))
unmasked_model_b.add(SimpleRNN(2,3, activation='relu', return_sequences=True))
unmasked_model_b.add(SimpleRNN(3,3, activation='relu'))
unmasked_model_b.add(Dense(3,4, activation='softmax'))
unmasked_model_b.compile(loss='categorical_crossentropy', optimizer='rmsprop', theano_mode=theano.compile.mode.FAST_COMPILE)
print("Compiled unmasked_model")

masked_model = Sequential()
masked_model.add(Embedding(4, 2, pass_mask=True))
masked_model.add(SimpleRNN(2,3, activation='relu', time_mask=True, return_sequences=True))
masked_model.add(SimpleRNN(3,3, activation='relu', time_mask=True))
masked_model.add(Dense(3,4, activation='softmax'))
masked_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', theano_mode=theano.compile.mode.FAST_COMPILE)
print("Compiled masked_model")

# This masked model is expected to learn
masked_model_b = Sequential()
masked_model_b.add(Embedding(4, 2, pass_mask=True))
masked_model_b.add(SimpleRNN(2,3, activation='relu', time_mask=True, return_sequences=True))
masked_model_b.add(SimpleRNN(3,3, activation='relu', time_mask=True))
masked_model_b.add(Dense(3,4, activation='softmax'))
masked_model_b.compile(loss='categorical_crossentropy', optimizer='rmsprop', theano_mode=theano.compile.mode.FAST_COMPILE)
print("Compiled masked_model")


# Uniform score: 4 options = ln(4) nats (2 bits)
# we should not do better than this when we mask out the part of the input
# that gives us the correct answer
uniform_score = np.log(4)

mask = np.ones((X.shape[0], X.shape[1], 1))
mask[:,0,:] = 0 # mask out the first timestep
withmask = np.concatenate((X[:,:,np.newaxis], mask), axis=2)

X0_onehot = np.zeros((X.shape[0], 4))
X1_onehot = np.zeros((X.shape[0], 4))
for i, row in enumerate(X):
    X0_onehot[i, row[0]] = 1
    X1_onehot[i, row[1]] = 1

#print(X0_onehot[:10])
#print("+++++")
#print(withmask[:10])

unmasked_model.fit(X, X0_onehot, nb_epoch=1)
score = unmasked_model.evaluate(X, X0_onehot)
if score > uniform_score*0.9:
    raise Exception('Failed to learn to copy timestep 0, score %f' % score)

unmasked_model_b.fit(X[:,1:], X1_onehot, nb_epoch=1)
score = unmasked_model_b.evaluate(X[:,1:], X1_onehot)
if score > uniform_score*0.9:
    raise Exception('Failed to learn to copy timestep 1, score %f' % score)

masked_model.fit(withmask, X0_onehot, nb_epoch=1)
score = masked_model.evaluate(withmask, X0_onehot)
if score < uniform_score*0.9:
   raise Exception('Somehow learned to copy timestep 0 despite mask, score %f' % score)


masked_model_b.fit(withmask, X1_onehot, nb_epoch=1)
score = masked_model_b.evaluate(withmask, X1_onehot)
if score > uniform_score*0.9:
    raise Exception('Failed to learn to copy timestep 1 in masked model, score %f' % score)

