'''Trains a multi-output deep NN on the MNIST dataset using crossentropy and
   policy gradients (REINFORCE).

The goal of this example is twofold:
* Show how to use policy graidents for training
* Show how to use generators with multioutput models

# Policy graidients
This is a Reinforcement Learning technique [1] that trains the model
following the gradient of the logarithm of action taken scaled by the advantage
(reward - baseline) of that action.

# Generators
They are useful for data augmentation on the fly and for pulling off core data
from disk as you go. This example shows a simple generator signature for
multioutput models

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.

There are two outputs: out1 that is trained with categorical crossentropy and
out2 that is trained with REINFORCE. The crosstropy branch does slightly better
but the REINFORCE branch is mind opening and shows alternatives for non
differentiable desired functions like number of correct counts (the one used
here), BLEU scores, etc.

[1] Simple Statistical Gradient-Following Algorithms for Connectionist
Reinforcement Learning. R. J. Williams
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K

batch_size = 100
nb_classes = 10
nb_epoch = 20

def datagen(X, Y):
    while 1:
        for i in range(0, 60000, 100):
            x = X[i:i+100]
            y = Y[i:i+100]
            yield ([x], [y, y])  # multiple inputs/outputs go packed in lists

def REINFORCE(y_true, y_pred):
    correct = K.argmax(y_true, axis=1)
    guess = K.argmax(y_pred, axis=1)  # gradients don't flow through this
    adv = K.equal(correct, guess)  # reward
    baseline = K.mean(adv)  # baseline
    adv = adv - baseline  # advantage
    logit = K.log(K.max(y_pred, axis=1)) # log probability of action taken, this makes our model and advantage actor critic
    # Keras does cost minimization, but we want to maximize reward probability, thus this minus sign
    return -adv*logit  # gradient will be -(r-b)*grad(log(pi))



# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X = Input(shape=(784,), name="X")

l1 = Dense(512, activation="relu")(X)
l2 = Dropout(.2)(l1)
l3 = Dense(512, activation="relu")(l2)
l4 = Dropout(.2)(l3)
out1 = Dense(10, activation="softmax", name="out1")(l4)
out2 = Dense(10, activation="softmax", name="out2")(l4)


model = Model(input=[X], output=[out1, out2])
model.summary()

model.compile(loss={'out1': 'categorical_crossentropy',
                    'out2': REINFORCE}, optimizer=RMSprop(),
    metrics={'out1': 'accuracy', 'out2': 'accuracy'})

model.fit_generator(datagen(X_train, Y_train), len(X_train), nb_epoch=nb_epoch)

score = model.evaluate_generator(datagen(X_test, Y_test), len(X_test))
print('Test score:', score[0])
print('Test accuracy:', score[1])
