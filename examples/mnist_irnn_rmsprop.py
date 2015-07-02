from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.initializations import normal, identity
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import RMSprop
from keras.utils import np_utils

'''
    This is a variant of IRNN experiment with sequential MNIST in
    "A Simple Way to Initialize Recurrent Networks of Rectified Linear Units "
    by Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton

    arXiv:1504.00941v2 [cs.NE] 7 Apr 201
    http://arxiv.org/pdf/1504.00941v2.pdf

    Optimizer is replaced with RMSprop which give more stable and steady
    improvement.

    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_irnn_rmsprop.py

    Reaches to 80% train/test accuracy and 0.55 train/test loss after 70 epochs
    (it's still underfitting at that point, though).

    About 15 minuts per epoch on a GRID K520 GPU.
'''

batch_size = 16
nb_classes = 10
nb_epochs = 200
hidden_units = 100

learning_rate = 1e-6
clip_norm = 1.0
BPTT_trancate = 28*28

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1, 1)
X_test = X_test.reshape(X_test.shape[0], -1, 1)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(SimpleRNN(input_dim=1, output_dim=hidden_units,
                    init=lambda shape: normal(shape, scale=0.001),
                    inner_init=lambda shape: identity(shape, scale=1.0),
                    activation='relu', truncate_gradient=BPTT_trancate))
model.add(Dense(hidden_units, nb_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

model.fit(X_train, Y_train, batch_size=16, nb_epoch=nb_epochs,
          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
