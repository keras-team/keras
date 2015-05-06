# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import theano
import theano.tensor as T

from .. import activations, initializations
from ..utils.theano_utils import shared_zeros, floatX
from ..utils.generic_utils import make_tuple

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from six.moves import zip
srng = RandomStreams()

class Layer(object):
    def __init__(self):
        self.params = []

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

    def get_config(self):
        return {"name":self.__class__.__name__}


class Dropout(Layer):
    '''
        Hinton's dropout.
    '''
    def __init__(self, p):
        super(Dropout,self).__init__()
        self.p = p

    def output(self, train):
        X = self.get_input(train)
        if self.p > 0.:
            retain_prob = 1. - self.p
            if train:
                X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            else:
                X *= retain_prob
        return X

    def get_config(self):
        return {"name":self.__class__.__name__,
            "p":self.p}


class Activation(Layer):
    '''
        Apply an activation function to an output.
    '''
    def __init__(self, activation, target=0, beta=0.1):
        super(Activation,self).__init__()
        self.activation = activations.get(activation)
        self.target = target
        self.beta = beta

    def output(self, train):
        X = self.get_input(train)
        return self.activation(X)

    def get_config(self):
        return {"name":self.__class__.__name__,
            "activation":self.activation.__name__,
            "target":self.target,
            "beta":self.beta}


class Reshape(Layer):
    '''
        Reshape an output to a certain shape.
        Can't be used as first layer in a model (no fixed input!)
        First dimension is assumed to be nb_samples.
    '''
    def __init__(self, *dims):
        super(Reshape,self).__init__()
        self.dims = dims

    def output(self, train):
        X = self.get_input(train)
        nshape = make_tuple(X.shape[0], *self.dims)
        return theano.tensor.reshape(X, nshape)

    def get_config(self):
        return {"name":self.__class__.__name__,
            "dims":self.dims}


class Flatten(Layer):
    '''
        Reshape input to flat shape.
        First dimension is assumed to be nb_samples.
    '''
    def __init__(self):
        super(Flatten,self).__init__()

    def output(self, train):
        X = self.get_input(train)
        size = theano.tensor.prod(X.shape) // X.shape[0]
        nshape = (X.shape[0], size)
        return theano.tensor.reshape(X, nshape)


class RepeatVector(Layer):
    '''
        Repeat input n times.

        Dimensions of input are assumed to be (nb_samples, dim).
        Return tensor of shape (nb_samples, n, dim).
    '''
    def __init__(self, n):
        super(RepeatVector,self).__init__()
        self.n = n

    def output(self, train):
        X = self.get_input(train)
        tensors = [X]*self.n
        stacked = theano.tensor.stack(*tensors)
        return stacked.dimshuffle((1,0,2))

    def get_config(self):
        return {"name":self.__class__.__name__,
            "n":self.n}


class Dense(Layer):
    '''
        Just your regular fully connected NN layer.
    '''
    def __init__(self, input_dim, output_dim, init='glorot_uniform', activation='linear', weights=None, 
        W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None):

        super(Dense,self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = T.matrix()
        self.W = self.init((self.input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.params = [self.W, self.b]

        self.regularizers = [W_regularizer, b_regularizer]
        self.constraints = [W_constraint, b_constraint]

        if weights is not None:
            self.set_weights(weights)

    def output(self, train):
        X = self.get_input(train)
        output = self.activation(T.dot(X, self.W) + self.b)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "activation":self.activation.__name__}


class TimeDistributedDense(Layer):
    '''
       Apply a same DenseLayer for each dimension[1] (shared_dimension) input
       Especially useful after a recurrent network with 'return_sequence=True'
       Tensor input dimensions:   (nb_sample, shared_dimension, input_dim)
       Tensor output dimensions:  (nb_sample, shared_dimension, output_dim)

    '''
    def __init__(self, input_dim, output_dim, init='glorot_uniform', activation='linear', weights=None, 
        W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None):

        super(TimeDistributedDense,self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = T.tensor3()
        self.W = self.init((self.input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.params = [self.W, self.b]

        self.regularizers = [W_regularizer, b_regularizer]
        self.constraints = [W_constraint, b_constraint]

        if weights is not None:
            self.set_weights(weights)

    def output(self, train):
        X = self.get_input(train)

        def act_func(X):
            return self.activation(T.dot(X, self.W) + self.b)

        output, _ = theano.scan(fn = act_func,
                                sequences = X.dimshuffle(1,0,2),
                                outputs_info=None)
        return output.dimshuffle(1,0,2)

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "activation":self.activation.__name__}
