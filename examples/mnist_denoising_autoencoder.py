from __future__ import absolute_import
from __future__ import print_function

import numpy as np

import theano
import theano.tensor as T


from keras.datasets import mnist
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils.theano_utils import shared_zeros
from keras.layers.core import Layer, Dense, Activation
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from keras import initializations, activations, regularizers, constraints

class DenoisingAutoEncoder(Layer):
    '''
        Denoising AutoEncoder
	---------------------
	Stacking of the autoencoder layers must be done manually.
	This layer was build by modifying the source code of Dense layer in keras.
	Weights between input and output are tied and output_dim refers to number of units in hidden layer.
	While training, output of the net is the reconstruction of the input.
	In testing, output of the net is the value output by the hidden layer.
    '''
    def __init__(self, input_dim, output_dim, init='uniform', activation='sigmoid', weights=None, name=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, corruption_level=0.0):

        super(DenoisingAutoEncoder, self).__init__()
        self.srng = RandomStreams(seed=np.random.randint(10e6))
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.corruption_level = corruption_level

        self.input = T.matrix()
        self.W = self.init((self.input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))
        self.bT = shared_zeros((self.input_dim))

        self.params = [self.W, self.b, self.bT]

        self.regularizers = []
        self.W_regularizer = regularizers.get(W_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        if weights is not None:
            self.set_weights(weights)

        if name is not None:
            self.set_name(name)

    def set_name(self, name):
        self.W.name = '%s_W' % name
        self.b.name = '%s_b' % name
        self.bT.name = '%s_bT' % name

    def get_output(self, train=False):
        X = self.get_input(train)
        if train:
            X *= self.srng.binomial(X.shape, p=1-self.corruption_level, dtype=theano.config.floatX)
            output = self.activation(T.dot(self.activation(T.dot(X, self.W) + self.b), self.W.T) + self.bT)
        else:
            output = self.activation(T.dot(X, self.W) + self.b)
        return output

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "activation": self.activation.__name__,
                "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                "b_constraint": self.b_constraint.get_config() if self.b_constraint else None,
                "corruption_level": self.corruption_level}

def test_dAE_on_mnist():
    nb_classes = 10
    layer_sizes = [784, 450, 350, 100]
    nb_pretrain_epochs = [5, 5, 5]
    nb_finetune_epochs = 5
    batch_sizes = [300, 300, 300]
    finetune_batch_size = 300

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 784)
    X_test = X_test.reshape(X_test.shape[0], 784)
    X_train = X_train.astype(theano.config.floatX)
    X_test = X_test.astype(theano.config.floatX)
    X_train /= 255
    X_test /= 255
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    all_model_params = []

    X_wholeset = np.vstack([X_train, X_test])
    X_wholeset_tmp = X_wholeset

    for i in range(len(layer_sizes)-1):
        print('PRETRAINING LAYER: {}'.format(i+1))
        pretrained_dAE_model = Sequential()
        pretrained_dAE_model.add(DenoisingAutoEncoder(layer_sizes[i], layer_sizes[i+1], activation='sigmoid', corruption_level=0.3))
        pretrained_dAE_model.compile(loss='mean_squared_error', optimizer='adadelta')
        pretrained_dAE_model.fit(X_wholeset_tmp, X_wholeset_tmp, nb_epoch=nb_pretrain_epochs[i], batch_size=batch_sizes[i])
        X_wholeset_tmp = pretrained_dAE_model.predict(X_wholeset_tmp)
        W, b, bT = pretrained_dAE_model.get_weights()
        all_model_params.append((W, b, bT))

    print('CREATE AND COMPILE CLASSIFIER')
    final_dAE_model = Sequential()
    for i in range(len(layer_sizes)-1):
        pretrained_dense_layer = Dense(layer_sizes[i], layer_sizes[i+1], activation='sigmoid')
        final_dAE_model.add(pretrained_dense_layer)
    final_dAE_model.add(Dense(layer_sizes[-1], nb_classes, activation='sigmoid'))
    final_dAE_model.add(Activation('softmax'))
    final_dAE_model.compile(loss='categorical_crossentropy', optimizer='adam')

    print('INITIALIZE CLASSIFIER USING PRE-TRAINED WEIGHTS')
    # initialize weights
    for i in range(len(layer_sizes)-1):
        W, b, bT = all_model_params[i]
        final_dAE_model.layers[i].set_weights([W, b])

    # finetune
    print('FINETUNING')
    final_dAE_model.fit(X_train, y_train, nb_epoch=nb_finetune_epochs, batch_size=200, show_accuracy=True, validation_data=[X_test, y_test])

    # evaluate performance
    score = final_dAE_model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
    print ('Test accuracy: {}'.format(score[1]))


if __name__ == '__main__':
    test_dAE_on_mnist()
