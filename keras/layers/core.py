# -*- coding: utf-8 -*-
import theano
import theano.tensor as T

from .. import activations, initializations
from ..utils.theano_utils import shared_zeros, floatX
from ..utils.generic_utils import make_tuple

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams()

class Layer(object):
    def connect(self, previous_layer):
        self.previous_layer = previous_layer

    def output(self, train):
        raise NotImplementedError

    def get_input(self, train):
        if hasattr(self, 'previous_layer'):
            return self.previous_layer.output(train=train)
        else:
            return self.input

    def set_weights(self, weights):
        for p, w in zip(self.params, weights):
            p.set_value(floatX(w))

    def get_weights(self):
        weights = []
        for p in self.params:
            weights.append(p.get_value())
        return weights


class Dropout(Layer):
    '''
        Hinton's dropout. 
    '''
    def __init__(self, p):
        self.p = p
        self.params = []

    def output(self, train):
        X = self.get_input(train)
        if self.p > 0.:
            retain_prob = 1. - self.p
            if train:
                X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            else:
                X *= retain_prob
        return X


class Activation(Layer):
    '''
        Apply an activation function to an output.
    '''
    def __init__(self, activation):
        self.activation = activations.get(activation)
        self.params = []

    def output(self, train):
        X = self.get_input(train)
        return self.activation(X)


class Reshape(Layer):
    '''
        Reshape an output to a certain shape.
        Can't be used as first layer in a model (no fixed input!)
        First dimension is assumed to be nb_samples.
    '''
    def __init__(self, *dims):
        self.dims = dims
        self.params = []

    def output(self, train):
        X = self.get_input(train)
        nshape = make_tuple(X.shape[0], *self.dims)
        return theano.tensor.reshape(X, nshape)


class Flatten(Layer):
    '''
        Reshape input to flat shape.
        First dimension is assumed to be nb_samples.
    '''
    def __init__(self, size):
        self.size = size
        self.params = []

    def output(self, train):
        X = self.get_input(train)
        nshape = (X.shape[0], self.size)
        return theano.tensor.reshape(X, nshape)


class RepeatVector(Layer):
    '''
        Repeat input n times.

        Dimensions of input are assumed to be (nb_samples, dim).
        Return tensor of shape (nb_samples, n, dim).
    '''
    def __init__(self, n):
        self.n = n
        self.params = []

    def output(self, train):
        X = self.get_input(train)
        tensors = [X]*self.n
        stacked = theano.tensor.stack(*tensors)
        return stacked.dimshuffle((1,0,2))


class Dense(Layer):
    '''
        Just your regular fully connected NN layer.
    '''
    def __init__(self, input_dim, output_dim, init='uniform', activation='linear', weights=None):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = T.matrix()
        self.W = self.init((self.input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.params = [self.W, self.b]

        if weights is not None:
            self.set_weights(weights)

    def output(self, train):
        X = self.get_input(train)
        output = self.activation(T.dot(X, self.W) + self.b)
        return output


class Embedding(Layer):
    '''
        Turn a list of integers >=0 into a dense vector of fixed size. 
        eg. [4, 50, 123, 26] -> [0.25, 0.1]

        @input_dim: size of vocabulary (highest input integer + 1)
        @out_dim: size of dense representation
    '''
    def __init__(self, input_dim, output_dim, init='uniform', weights=None):
        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = T.imatrix()
        self.W = self.init((self.input_dim, self.output_dim))
        self.params = [self.W]

        if weights is not None:
            self.set_weights(weights)

    def output(self, train):
        X = self.get_input(train)
        return self.W[X]

