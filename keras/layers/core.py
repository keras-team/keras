# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import theano
import theano.tensor as T
import numpy as np

from .. import activations, initializations
from ..utils.theano_utils import shared_zeros, floatX
from ..utils.generic_utils import make_tuple
from .. import regularizers
from .. import constraints

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from six.moves import zip
srng = RandomStreams(seed=np.random.randint(10e6))

class Layer(object):
    def __init__(self, name, prev):
        self.params = []
        self.name = name
        self.prev_name = prev

        if type(self.prev_name) is str:
            self.prev_name = [self.prev_name] # single string or a list of strings under one op

    def connect(self, node):
        self.previous = node

        if len(node) == 1:
            self.previous = self.previous[0]    # for backward compatibility

    def get_output(self, train):
        raise NotImplementedError

    def get_input(self, train):
        if hasattr(self, 'previous'):
            if type(self.previous) is not list:
                return self.previous.get_output(train=train)
            else:
                output = []
                for node in self.previous:
                    output.append(node.get_output(train=train))
                return output
        else:
            return self.input


    def set_weights(self, weights):
        for p, w in zip(self.params, weights):
            if p.eval().shape != w.shape:
                raise Exception("Layer shape %s not compatible with weight shape %s." % (p.eval().shape, w.shape))
            p.set_value(floatX(w))

    def get_weights(self):
        weights = []
        for p in self.params:
            weights.append(p.get_value())
        return weights

    def get_config(self):
        return {"name":self.__class__.__name__}

    def get_params(self):
        regs = []
        consts = []

        if hasattr(self, 'regularizers') and len(self.regularizers) == len(self.params):
            for r in self.regularizers:
                if r:
                    regs.append(r)
                else:
                    regs.append(regularizers.identity)
        elif hasattr(self, 'regularizer') and self.regularizer:
            regs += [self.regularizer for _ in range(len(self.params))]
        else:
            regs += [regularizers.identity for _ in range(len(self.params))]

        if hasattr(self, 'constraints') and len(self.constraints) == len(self.params):
            for c in self.constraints:
                if c:
                    consts.append(c)
                else:
                    consts.append(constraints.identity)
        elif hasattr(self, 'constraint') and self.constraint:
            consts += [self.constraint for _ in range(len(self.params))]
        else:
            consts += [constraints.identity for _ in range(len(self.params))]

        return self.params, regs, consts


class Merge(Layer):
    def __init__(self, prev, mode='sum', name=None):
        super(Merge,self).__init__(name, prev)
        self.mode = mode

    def get_output(self, train):
        inputs = self.get_input(train)
        if self.mode == 'sum':
            s = inputs[0]
            for inp in inputs[1:]:
                s += inp
            return s
        elif self.mode == 'concat':
            return T.concatenate(inputs, axis=-1)
        else:
            raise Exception('Unknown merge mode')

class Dropout(Layer):
    '''
        Hinton's dropout.
    '''
    def __init__(self, p, name=None, prev=None):
        super(Dropout,self).__init__(name, prev)
        self.p = p

    def get_output(self, train):
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
    def __init__(self, activation, target=0, beta=0.1, name=None, prev=None):
        super(Activation,self).__init__(name, prev)
        self.activation = activations.get(activation)
        self.target = target
        self.beta = beta

    def get_output(self, train):
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
        super(Reshape,self).__init__(name, prev)
        self.dims = dims

    def get_output(self, train):
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
    def __init__(self, name=None, prev=None):
        super(Flatten,self).__init__(name, prev)

    def get_output(self, train):
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
    def __init__(self, n, name=None, prev=None):
        super(RepeatVector,self).__init__(name, prev)
        self.n = n

    def get_output(self, train):
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
    def __init__(self, input_shape, output_shape,init='glorot_uniform', activation='linear', weights=None,
        W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, name=None, prev=None):

        super(Dense,self).__init__(name, prev)
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.regularizers = [W_regularizer, b_regularizer]
        self.constraints = [W_constraint, b_constraint]

        self.input = T.matrix()

        self.W = self.init((self.input_shape, self.output_shape))
        self.b = shared_zeros((self.output_shape))

        self.params = [self.W, self.b]

        if weights is not None:
            self.set_weights(weights)

    def get_output(self, train):
        X = self.get_input(train)
        output = self.activation(T.dot(X, self.W) + self.b)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_shape":self.input_shape,
            "output_shape":self.output_shape,
            "init":self.init.__name__,
            "activation":self.activation.__name__}


class TimeDistributedDense(Layer):
    '''
       Apply a same DenseLayer for each dimension[1] (shared_dimension) input
       Especially useful after a recurrent network with 'return_sequence=True'
       Tensor input dimensions:   (nb_sample, shared_dimension, input_shape)
       Tensor output dimensions:  (nb_sample, shared_dimension, output_shape)

    '''
    def __init__(self, input_shape,  output_shape,init='glorot_uniform', activation='linear', weights=None,
        W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, name=None, prev=None):

        super(TimeDistributedDense,self).__init__(name, prev)
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.regularizers = [W_regularizer, b_regularizer]
        self.constraints = [W_constraint, b_constraint]

        self.input = T.tensor3()
        self.W = self.init((self.input_shape, self.output_shape))
        self.b = shared_zeros((self.output_shape))

        self.params = [self.W, self.b]

        if weights is not None:
            self.set_weights(weights)


    def get_output(self, train):
        X = self.get_input(train)

        def act_func(X):
            return self.activation(T.dot(X, self.W) + self.b)

        output, _ = theano.scan(fn = act_func,
                                sequences = X.dimshuffle(1,0,2),
                                outputs_info=None)
        return output.dimshuffle(1,0,2)

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_shape":self.input_shape,
            "output_shape":self.output_shape,
            "init":self.init.__name__,
            "activation":self.activation.__name__}




class MaxoutDense(Layer):
    '''
        Max-out layer, nb_feature is the number of pieces in the piecewise linear approx.
        Refer to http://arxiv.org/pdf/1302.4389.pdf
    '''
    def __init__(self, input_shape, output_shape,nb_feature=4, init='glorot_uniform', weights=None,
        W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, name=None, prev=None):

        super(MaxoutDense,self).__init__(name, prev)
        self.init = initializations.get(init)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.nb_feature = nb_feature

        self.regularizers = [W_regularizer, b_regularizer]
        self.constraints = [W_constraint, b_constraint]


        self.input = T.matrix()
        self.W = self.init((self.nb_feature, self.input_shape, self.output_shape))
        self.b = shared_zeros((self.nb_feature, self.output_shape))

        self.params = [self.W, self.b]

        if weights is not None:
            self.set_weights(weights)


    def get_output(self, train):
        X = self.get_input(train)
        # -- don't need activation since it's just linear.
        output = T.max(T.dot(X, self.W) + self.b, axis=1)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_shape":self.input_shape,
            "output_shape":self.output_shape,
            "init":self.init.__name__,
            "nb_feature" : self.nb_feature}
