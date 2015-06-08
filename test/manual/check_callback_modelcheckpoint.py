from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge
from keras.utils import np_utils
import keras.callbacks as cbks
import numpy as np

nb_classes = 10
batch_size = 128
nb_epoch = 20

# small sample size to overfit on training data
max_train_samples = 50
max_test_samples = 1000

np.random.seed(1337) # for reproducibility

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000,784)[:max_train_samples]
X_test = X_test.reshape(10000,784)[:max_test_samples]
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)[:max_train_samples]
Y_test = np_utils.to_categorical(y_test, nb_classes)[:max_test_samples]

##########################
# model checkpoint tests #
##########################

# Create a slightly larger network than required to test best validation save only 
model = Sequential()
model.add(Dense(784, 500))
model.add(Activation('relu'))
model.add(Dense(500, 10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# test file location
path = "/tmp"
filename = "model_weights.hdf5"
import os
f = os.path.join(path, filename)

print("Test model checkpointer")
# only store best validation model in checkpointer
checkpointer = cbks.ModelCheckpoint(filepath=f, verbose=1, save_best_only=True)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_data=(X_test, Y_test), callbacks =[checkpointer])

if not os.path.isfile(f):
    raise Exception("Model weights were not saved to %s" % (f))

print("Test model checkpointer without validation data")
import warnings
warnings.filterwarnings('error')
try:
    # this should issue a warning
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, callbacks =[checkpointer])
except:
    print("Tests passed")
    import sys
    sys.exit(0)

raise Exception("Tests did not pass")

