# Dummy test data as input to RNN. This input is 3 timesteps long where the third timestep always matches the
# first. Without masking it should be able to learn it, with masking it should fail.

import numpy as np
from keras.utils.theano_utils import sharedX
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import default_mask_val
import theano

theano.config.exception_verbosity='high' 

# (nb_samples, timesteps, dimensions)
X = np.random.random_integers(1, 4, size=(400000,3))

model = Sequential()
model.add(Embedding(5, 2, zero_is_mask=True))
model.add(SimpleRNN(2,3, activation='relu', return_sequences=True))
model.add(SimpleRNN(3,3, activation='relu'))
model.add(Dense(3,4, activation='softmax'))
model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop', theano_mode=theano.compile.mode.FAST_RUN)
print("Compiled model")

if model.layers[0].W.get_value()[0,:] != default_mask_val:
    raise Exception("Did not set the mask val properly into the Embedding W matrix, got: ",
            model.layers[0].W.get_value()[0,:]

W = model.get_weights() # We'll save these so we can reset it later


Xmask0 = X.copy()
Xmask0[:,0] = 0

X0_onehot = np.zeros((X.shape[0], 4))
X1_onehot = np.zeros((X.shape[0], 4))
for i, row in enumerate(X):
    X0_onehot[i, row[0]-1] = 1
    X1_onehot[i, row[1]-1] = 1

# Uniform score: 4 options = ln(4) nats (2 bits)
# we should not do better than this when we mask out the part of the input
# that gives us the correct answer
uniform_score = np.log(4)

# Train it to guess 0th dim
model.fit(X, X0_onehot, nb_epoch=1)
score = model.evaluate(X, X0_onehot)
if score > uniform_score*0.9:
    raise Exception('Failed to learn to copy timestep 0, score %f' % score)

if model.layers[0].W.get_value()[0,:] != default_mask_val:
    raise Exception("After training, the W of the Embedding's mask value changed to: ",
            model.layers[0].W.get_value()[0,:]
    

model.set_weights(W)

# Train without showing it the 0th dim to learn 1st dim
model.fit(X[:,1:], X1_onehot, nb_epoch=1)
score = model.evaluate(X[:,1:], X1_onehot)
if score > uniform_score*0.9:
    raise Exception('Failed to learn to copy timestep 1, score %f' % score)

model.set_weights(W)

# Train to guess 0th dim when 0th dim has been masked (should fail)
model.fit(Xmask0, X0_onehot, nb_epoch=1)
score = model.evaluate(Xmask0, X0_onehot)
if score < uniform_score*0.9:
   raise Exception('Somehow learned to copy timestep 0 despite mask, score %f' % score)

model.set_weights(W)

# Train to guess 1st dim when 0th dim has been masked (should succeed)
model.fit(Xmask0, X1_onehot, nb_epoch=1)
score = model.evaluate(Xmask0, X1_onehot)
if score > uniform_score*0.9:
    raise Exception('Failed to learn to copy timestep 1 in masked model, score %f' % score)

