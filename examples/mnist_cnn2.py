from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

'''
    Train a simple convnet on the MNIST dataset.

    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py

    Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
    16 seconds per epoch on a GRID K520 GPU.
'''

batch_size = 128
nb_classes = 10
nb_epoch = 12

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
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

graph = Graph()
graph.add_input(name='input', ndim=4)
nb_filters = 3
graph.add_node(Convolution2D(nb_filters, 1, 3, 3, border_mode='same'),
        name='scores_layer_1a', input='input')
graph.add_node(Convolution2D(nb_filters, 1, 5, 5, border_mode='same'),
        name='scores_layer_1b', input='input')

# double duty layer, computing activations and concatenating pathways 'a' and 'b'. 
# Not sure concatenation is in the right order for what I want.
# In this example, I only have 2 filters per pathway
graph.add_node(Activation('relu'),
        name='activations_layer_1', inputs=['scores_layer_1a', 'scores_layer_1b'], axis=1)

#graph.add_node(Reshape(2*nb_filters, 28, 28), 
#        name='layer_1_4d', input='activations_layer_1')

#graph.add_node(Convolution2D(3, 2*nb_filters, 3, 3, border_mode='same'),
#        name='scores_layer_2', input='layer_1_4d')
graph.add_node(Convolution2D(3, 2*nb_filters, 3, 3, border_mode='same'),
        name='scores_layer_2', input='activations_layer_1')

graph.add_node(Activation('relu'),
        name='activations_layer_2', input='scores_layer_2')

graph.add_node(Flatten(),
        name='flatten_layer_2', input='activations_layer_2')

graph.add_node(Dense(3*28*28, 128),
        name='scores_layer_3', input='flatten_layer_2')

graph.add_node(Activation('relu'), 
        name='activations_layer_3', input='scores_layer_3')

graph.add_node(Dense(128, nb_classes),
        name='scores_layer_4', input='activations_layer_3')

graph.add_node(Activation('softmax'),
        name='activations_layer_4', input='scores_layer_4')

graph.add_output(name='output', input='activations_layer_4')

graph.compile('adadelta', {'output':'categorical_crossentropy'})

graph.fit({'input':X_train, 'output':Y_train}, batch_size=batch_size, nb_epoch=nb_epoch)
