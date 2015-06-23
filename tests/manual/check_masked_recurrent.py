# Dummy test data as input to RNN. This input is 3 timesteps long where the third timestep always matches the
# first. Without masking it should be able to learn it, with masking it should fail.

import numpy as np
from keras.utils.theano_utils import sharedX
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Dropout, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, SimpleDeepRNN, LSTM, GRU
from keras.layers.core import default_mask_val
import theano

theano.config.exception_verbosity='high' 

# (nb_samples, timesteps, dimensions)
X = np.random.random_integers(1, 4, size=(500000,15))

print("About to compile the first model")
model = Sequential()
model.add(Embedding(5, 4, zero_is_mask=True))
model.add(TimeDistributedDense(4,4)) # obviously this is redundant. Just testing.
model.add(SimpleRNN(4,4, activation='relu', return_sequences=True))
model.add(Dropout(0.5))
model.add(SimpleDeepRNN(4,4, depth=2, activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(4,4, activation='softmax'))
model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop', theano_mode=theano.compile.mode.FAST_RUN)
print("Compiled model")

W0 = model.layers[0].W.get_value()[0,:]
if (W0 != default_mask_val).any():
    raise Exception("Did not set the mask val properly into the Embedding W matrix, got: ",
            W0)

W = model.get_weights() # We'll save these so we can reset it later


X[:,:10] = 0
Xmask0 = X.copy()
Xmask0[:,10] = 0

Xmask12 = X.copy()
Xmask12[:,11] = 0
Xmask12[:,12] = 0

X0_onehot = np.zeros((X.shape[0], 4))
X1_onehot = np.zeros((X.shape[0], 4))
for i, row in enumerate(X):
    X0_onehot[i, row[10]-1] = 1
    X1_onehot[i, row[11]-1] = 1

# Uniform score: 4 options = ln(4) nats (2 bits)
# we should not do better than this when we mask out the part of the input
# that gives us the correct answer
uniform_score = np.log(4)
batch_size=512

# Train it to guess 0th dim
model.fit(X, X0_onehot, nb_epoch=1, batch_size=batch_size)
score = model.evaluate(X, X0_onehot, batch_size=batch_size)
if score > uniform_score*0.9:
    raise Exception('Failed to learn to copy timestep 0, score %f' % score)

W0 = model.layers[0].W.get_value()[0,:]
if (W0 != default_mask_val).any():
    raise Exception("After training, the W0 of the Embedding's mask value changed to: ",
            W0)
    

model.set_weights(W)

# Train without showing it the 0th dim to learn 1st dim
model.fit(X[:,1:], X1_onehot, nb_epoch=1, batch_size=batch_size)
score = model.evaluate(X[:,1:], X1_onehot, batch_size=batch_size)
if score > uniform_score*0.9:
    raise Exception('Failed to learn to copy timestep 1, score %f' % score)

model.set_weights(W)

# Train to guess 0th dim when 0th dim has been masked (should fail)
model.fit(Xmask0, X0_onehot, nb_epoch=1, batch_size=batch_size)
score = model.evaluate(Xmask0, X0_onehot, batch_size=batch_size)
if score < uniform_score*0.9:
   raise Exception('Somehow learned to copy timestep 0 despite mask, score %f' % score)

model.set_weights(W)

# Train to guess 1st dim when 0th dim has been masked (should succeed)
model.fit(Xmask0, X1_onehot, nb_epoch=1, batch_size=batch_size)
score = model.evaluate(Xmask0, X1_onehot, batch_size=batch_size)
if score > uniform_score*0.9:
    raise Exception('Failed to learn to copy timestep 1 in masked model, score %f' % score)



model.set_weights(W)

# Finally, make sure the mask is actually blocking input, mask out timesteps 1 and 2, and see if
# it can learn timestep 0 (should fail)
model.fit(Xmask12, X0_onehot, nb_epoch=1, batch_size=batch_size)

W0 = model.layers[0].W.get_value()[0,:]
if (W0 != default_mask_val).any():
    raise Exception("After masked training, the W0 of the Embedding's mask value changed to: ",
            W0)

score = model.evaluate(Xmask12, X0_onehot, batch_size=batch_size)
if score < uniform_score*0.9:
    raise Exception('Somehow learned to copy timestep 0 despite masking 1, score %f' % score)

# Another testing approach, just initialize models and make sure that prepending zeros doesn't affect
# their output
print("About to compile the second model")
model2 = Sequential()
model2.add(Embedding(5, 4, zero_is_mask=True))
model2.add(TimeDistributedDense(4,4))
model2.add(Activation('time_distributed_softmax'))
model2.add(LSTM(4,4, return_sequences=True))
model2.add(Activation('tanh'))
model2.add(GRU(4,4, activation='softmax', return_sequences=True))
model2.add(SimpleDeepRNN(4,4, depth=2, activation='relu', return_sequences=True)) 
model2.add(SimpleRNN(4,4, activation='relu', return_sequences=False))
model2.compile(loss='categorical_crossentropy',
        optimizer='rmsprop', theano_mode=theano.compile.mode.FAST_COMPILE)
print("Compiled model2")

X2 = np.random.random_integers(1, 4, size=(2,5))
ref = model2.predict(X2)
for pre_zeros in range(1,10):
    padded = np.concatenate((np.zeros((2, pre_zeros)), X2), axis=1)
    pred = model2.predict(padded)
    if not np.allclose(ref, pred):
        raise Exception("Different result after left-padding %d zeros. Ref: %s, Pred: %s" % (pre_zeros, ref, pred))
        
