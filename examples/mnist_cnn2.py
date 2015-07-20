from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, Callback, SnapshotPrediction
from keras.utils import np_utils
import pdb

'''
    Train a simple convnet on the MNIST dataset.

    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py

    Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
    16 seconds per epoch on a GRID K520 GPU.
'''

class model(object):
    def __init__(self):
        self.batch_size = 128*5
        self.nb_classes = 10
        self.nb_epoch = 3

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
        Y_train = np_utils.to_categorical(y_train, self.nb_classes)
        Y_test = np_utils.to_categorical(y_test, self.nb_classes)
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.y_train = y_train
        self.y_test = y_test

    def build_model(self):
        graph = Graph()
        graph.add_input(name='input', ndim=4)
        nb_filters = 3

        # First convolutional pathway using 3x3 filters
        graph.add_node(Convolution2D(nb_filters, 1, 3, 3, border_mode='same'),
                name='scores_layer_1a', input='input')

        # Second convolutional pathway using 5x5 filters
        graph.add_node(Convolution2D(nb_filters, 1, 7, 7, border_mode='same'),
                name='scores_layer_1b', input='input')

        graph.add_node(Permute((2,3,1)),
                name='scores_layer_1a_permuted', input='scores_layer_1a')

        graph.add_node(Permute((2,3,1)), 
                name='scores_layer_1b_permuted', input='scores_layer_1b')

        # double duty layer, computing activations and concatenating pathways 'a' and 'b'. 
        # Not sure concatenation is in the right order for what I want.
        graph.add_node(Activation('relu'),
                name='activations_layer_1_permuted', inputs=['scores_layer_1a_permuted', 'scores_layer_1b_permuted'])

        graph.add_node(Permute((3,1,2)),
                name='activations_layer_1', input='activations_layer_1_permuted')

        #graph.add_node(Reshape(2*nb_filters, 28, 28), 
        #        name='layer_1_4d', input='activations_layer_1')

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

        graph.add_node(Dense(128, self.nb_classes),
                name='scores_layer_4', input='activations_layer_3')

        graph.add_node(Activation('softmax'),
                name='activations_layer_4', input='scores_layer_4')

        graph.add_output(name='output', input='activations_layer_4')

        graph.compile('adadelta', {'output':'categorical_crossentropy'})

        self.graph = graph

    def fit(self):
        checkpoint = ModelCheckpoint(filepath='graph.hdf5', verbose=1, save_best_only=False)
        #checkpred = SnapshotPrediction(filepath="graph_prediction.hdf5")

        checkAccuracy = GetAccuracy(self.X_test, self.y_test)
        self.graph.fit({'input':self.X_train, 'output':self.Y_train}, batch_size=self.batch_size, nb_epoch=self.nb_epoch, verbose=1,
                validation_data={'input':self.X_test, 'output':self.Y_test}, callbacks=[checkpoint, checkAccuracy])

        score = self.graph.evaluate({'input':self.X_test, 'output':self.Y_test}, verbose=0)
        #pdb.set_trace()
        print('Test score:', score)

class GetAccuracy(Callback):
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs={}):
        #pdb.set_trace()
        predictions = self.model.predict({'input':self.X_test})
        classes = np_utils.probas_to_classes(predictions['output'])
        print("accuracy on test data: {0}".format((classes==self.y_test).mean()))
