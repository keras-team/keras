# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import theano
import theano.tensor as T
import numpy as np

from .. import activations, initializations
from ..utils.theano_utils import shared_zeros, floatX, shared_scalar, get_mask, default_mask_val
from ..utils.generic_utils import make_tuple
from .. import regularizers
from .. import constraints

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from six.moves import zip
srng = RandomStreams(seed=np.random.randint(10e6))

class Layer(object):
    def __init__(self):
        self.params = []

    def connect(self, layer):
        self.previous = layer

    def get_output(self, train):
        raise NotImplementedError

    def get_input(self, train):
        if hasattr(self, 'previous'):
            return self.previous.get_output(train=train)
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


class Merge(object): 
    def __init__(self, models, mode='sum'):
        ''' Merge the output of a list of models into a single tensor.
            mode: {'sum', 'concat'}
        '''
        if len(models) < 2:
            raise Exception("Please specify two or more input models to merge")
        self.mode = mode
        self.models = models
        self.params = []
        self.regularizers = []
        self.constraints = []
        for m in self.models:
            self.params += m.params
            self.regularizers += m.regularizers
            self.constraints += m.constraints

    def get_params(self):
        return self.params, self.regularizers, self.constraints

    def get_output(self, train=False):
        if self.mode == 'sum':
            s = self.models[0].get_output(train)
            for i in range(1, len(self.models)):
                s += self.models[i].get_output(train)
            return s
        elif self.mode == 'concat':
            inputs = [self.models[i].get_output(train) for i in range(len(self.models))]
            return T.concatenate(inputs, axis=-1)
        else:
            raise Exception('Unknown merge mode')

    def get_input(self, train=False):
        res = []
        for i in range(len(self.models)):
            o = self.models[i].get_input(train)
            if type(o) == list:
                res += o
            else:
                res.append(o)
        return res

    @property
    def input(self):
        return self.get_input()

    def get_weights(self):
        weights = []
        for m in self.models:
            weights += m.get_weights()
        return weights

    def set_weights(self, weights):
        for i in range(len(self.models)):
            nb_param = len(self.models[i].params)
            self.models[i].set_weights(weights[:nb_param])
            weights = weights[nb_param:]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "models":[m.get_config() for m in self.models],
            "mode":self.mode}


class Dropout(Layer):
    '''
        Hinton's dropout.
    '''
    def __init__(self, p, mask_val=default_mask_val):
        super(Dropout,self).__init__()
        self.p = p
        self.mask_val = shared_scalar(mask_val)

    def get_output(self, train):
        X = self.get_input(train)
        mask = get_mask(X, self.mask_val)
        if self.p > 0.:
            retain_prob = 1. - self.p
            if train:
                X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            else:
                X *= retain_prob
        return mask * X + (1 - mask) * self.mask_val

    def get_config(self):
        return {"name":self.__class__.__name__,
            "p":self.p}


class Activation(Layer):
    '''
        Apply an activation function to an output.
    '''
    def __init__(self, activation, target=0, beta=0.1, mask_val=default_mask_val):
        super(Activation,self).__init__()
        self.activation = activations.get(activation)
        self.target = target
        self.beta = beta
        self.mask_val = shared_scalar(mask_val)

    def get_output(self, train):
        X = self.get_input(train)
        mask = get_mask(X, self.mask_val)
        return mask * self.activation(X) + (1 - mask) * self.mask_val

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
    def __init__(self):
        super(Flatten,self).__init__()

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
    def __init__(self, n):
        super(RepeatVector,self).__init__()
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

    def get_output(self, train):
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
        W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, mask_val=default_mask_val):

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
        self.mask_val = shared_scalar(mask_val)

        if weights is not None:
            self.set_weights(weights)

    def get_output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle(1,0,2)
        mask = get_mask(X, self.mask_val)

        def act_func(X, mask):
            return mask * self.activation(T.dot(X, self.W) + self.b) + (1 - mask) * self.mask_val

        output, _ = theano.scan(fn = act_func,
                                sequences = [X, mask],
                                outputs_info=None)
        return output.dimshuffle(1,0,2)

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "activation":self.activation.__name__}

class AutoEncoder(Layer):
    '''
        A customizable autoencoder model.
        If output_reconstruction then dim(input) = dim(output)
        else dim(output) = dim(hidden)
    '''
    def __init__(self, encoder, decoder, output_reconstruction=True, tie_weights=False, weights=None):

        super(AutoEncoder,self).__init__()

        self.output_reconstruction = output_reconstruction
        self.tie_weights = tie_weights
        self.encoder = encoder
        self.decoder = decoder

        self.decoder.connect(self.encoder)

        self.params = []
        self.regularizers = []
        self.constraints = []
        for layer in [self.encoder, self.decoder]:
            self.params += layer.params
            if hasattr(layer, 'regularizers'):
                self.regularizers += layer.regularizers
            if hasattr(layer, 'constraints'):
                self.constraints += layer.constraints

        if weights is not None:
            self.set_weights(weights)

    def connect(self, node):
        self.encoder.previous = node

    def get_weights(self):
        weights = []
        for layer in [self.encoder, self.decoder]:
            weights += layer.get_weights()
        return weights

    def set_weights(self, weights):
        nb_param = len(self.encoder.params)
        self.encoder.set_weights(weights[:nb_param])
        self.decoder.set_weights(weights[nb_param:])

    def get_input(self, train=False):
        return self.encoder.get_input(train)

    @property
    def input(self):
        return self.encoder.input

    def _get_hidden(self, train):
        return self.encoder.get_output(train)

    def get_output(self, train):
        if not train and not self.output_reconstruction:
            return self.encoder.get_output(train)

        decoded = self.decoder.get_output(train)

        if self.tie_weights:
            encoder_params = self.encoder.get_weights()
            decoder_params = self.decoder.get_weights()
            for dec_param, enc_param in zip(decoder_params, encoder_params):
                if len(dec_param.shape) > 1:
                    enc_param = dec_param.T

        return decoded

    def get_config(self):
        return {"name":self.__class__.__name__,
                "encoder_config":self.encoder.get_config(),
                "decoder_config":self.decoder.get_config(),
                "output_reconstruction":self.output_reconstruction,
                "tie_weights":self.tie_weights}


class DenoisingAutoEncoder(AutoEncoder):
    '''
        A denoising autoencoder model that inherits the base features from autoencoder
    '''
    def __init__(self, encoder=None, decoder=None, output_reconstruction=True, tie_weights=False, weights=None, corruption_level=0.3):
        super(DenoisingAutoEncoder, self).__init__(encoder, decoder, output_reconstruction, tie_weights, weights)
        self.corruption_level = corruption_level

    def _get_corrupted_input(self, input):
        """
            http://deeplearning.net/tutorial/dA.html
        """
        return srng.binomial(size=(self.input_dim, 1), n=1,
                             p=1-self.corruption_level,
                             dtype=theano.config.floatX) * input

    def get_input(self, train=False):
        uncorrupted_input = super(DenoisingAutoEncoder, self).get_input(train)
        return self._get_corrupted_input(uncorrupted_input)

    def get_config(self):
        return {"name":self.__class__.__name__,
                "encoder_config":self.encoder.get_config(),
                "decoder_config":self.decoder.get_config(),
                "corruption_level":self.corruption_level,
                "output_reconstruction":self.output_reconstruction,
                "tie_weights":self.tie_weights}


class MaxoutDense(Layer):
    '''
        Max-out layer, nb_feature is the number of pieces in the piecewise linear approx.
        Refer to http://arxiv.org/pdf/1302.4389.pdf
    '''
    def __init__(self, input_dim, output_dim, nb_feature=4, init='glorot_uniform', weights=None, 
        W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None):

        super(MaxoutDense,self).__init__()
        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nb_feature = nb_feature

        self.input = T.matrix()
        self.W = self.init((self.nb_feature, self.input_dim, self.output_dim))
        self.b = shared_zeros((self.nb_feature, self.output_dim))

        self.params = [self.W, self.b]

        self.regularizers = [W_regularizer, b_regularizer]
        self.constraints = [W_constraint, b_constraint]

        if weights is not None:
            self.set_weights(weights)

    def get_output(self, train):
        X = self.get_input(train)
        # -- don't need activation since it's just linear.
        output = T.max(T.dot(X, self.W) + self.b, axis=1)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "nb_feature" : self.nb_feature}
