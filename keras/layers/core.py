# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import theano
import theano.tensor as T
import numpy as np

from .. import activations, initializations, regularizers, constraints
from ..utils.theano_utils import shared_zeros, floatX
from ..utils.generic_utils import make_tuple
from ..regularizers import ActivityRegularizer, Regularizer

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from six.moves import zip


class Layer(object):
    def __init__(self):
        self.params = []

    def init_updates(self):
        self.updates = []

    def set_previous(self, layer, connection_map={}):
        assert self.nb_input == layer.nb_output == 1, "Cannot connect layers: input count and output count should be 1."
        if not self.supports_masked_input() and layer.get_output_mask() is not None:
            raise Exception("Cannot connect non-masking layer to layer with masked output")
        self.previous = layer

    @property
    def nb_input(self):
        return 1

    @property
    def nb_output(self):
        return 1

    def get_output(self, train=False):
        return self.get_input(train)

    def get_input(self, train=False):
        if hasattr(self, 'previous'):
            return self.previous.get_output(train=train)
        else:
            return self.input

    def supports_masked_input(self):
        ''' Whether or not this layer respects the output mask of its previous layer in its calculations. If you try
        to attach a layer that does *not* support masked_input to a layer that gives a non-None output_mask() that is
        an error'''
        return False

    def get_output_mask(self, train=None):
        '''
        For some models (such as RNNs) you want a way of being able to mark some output data-points as
        "masked", so they are not used in future calculations. In such a model, get_output_mask() should return a mask
        of one less dimension than get_output() (so if get_output is (nb_samples, nb_timesteps, nb_dimensions), then the mask
        is (nb_samples, nb_timesteps), with a one for every unmasked datapoint, and a zero for every masked one.

        If there is *no* masking then it shall return None. For instance if you attach an Activation layer (they support masking)
        to a layer with an output_mask, then that Activation shall also have an output_mask. If you attach it to a layer with no
        such mask, then the Activation's get_output_mask shall return None.

        Some layers have an output_mask even if their input is unmasked, notably Embedding which can turn the entry "0" into
        a mask.
        '''
        return None

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
        return {"name": self.__class__.__name__}

    def get_params(self):
        consts = []
        updates = []

        if hasattr(self, 'regularizers'):
            regularizers = self.regularizers
        else:
            regularizers = []

        if hasattr(self, 'constraints') and len(self.constraints) == len(self.params):
            for c in self.constraints:
                if c:
                    consts.append(c)
                else:
                    consts.append(constraints.identity())
        elif hasattr(self, 'constraint') and self.constraint:
            consts += [self.constraint for _ in range(len(self.params))]
        else:
            consts += [constraints.identity() for _ in range(len(self.params))]

        if hasattr(self, 'updates') and self.updates:
            updates += self.updates

        return self.params, regularizers, consts, updates

    def set_name(self, name):
        for i in range(len(self.params)):
            self.params[i].name = '%s_p%d' % (name, i)

    def count_params(self):
        return sum([np.prod(p.shape.eval()) for p in self.params])

class MaskedLayer(Layer):
    '''
    If your layer trivially supports masking (by simply copying the input mask to the output), then subclass MaskedLayer
    instead of Layer, and make sure that you incorporate the input mask into your calculation of get_output()
    '''
    def supports_masked_input(self):
        return True

    def get_input_mask(self, train=False):
        if hasattr(self, 'previous'):
            return self.previous.get_output_mask(train)
        else:
            return None

    def get_output_mask(self, train=False):
        ''' The default output mask is just the input mask unchanged. Override this in your own
        implementations if, for instance, you are reshaping the input'''
        return self.get_input_mask(train)


class Masking(MaskedLayer):
    """Mask an input sequence by using a mask value to identify padding.

    This layer copies the input to the output layer with identified padding
    replaced with 0s and creates an output mask in the process.

    At each timestep, if the values all equal `mask_value`,
    then the corresponding mask value for the timestep is 0 (skipped),
    otherwise it is 1.

    """
    def __init__(self, mask_value=0.):
        super(Masking, self).__init__()
        self.mask_value = mask_value
        self.input = T.tensor3()

    def get_output_mask(self, train=False):
        X = self.get_input(train)
        return T.any(T.ones_like(X) * (1. - T.eq(X, self.mask_value)), axis=-1)

    def get_output(self, train=False):
        X = self.get_input(train)
        return X * T.shape_padright(T.any((1. - T.eq(X, self.mask_value)), axis=-1))

    def get_config(self):
        return {"name": self.__class__.__name__,
                "mask_value": self.mask_value}

class TimeDistributedMerge(Layer):
    def __init__(self, mode='sum'):
        '''
        Sum/multiply/avearge over a time distributed layer's outputs.
        mode: {'sum', 'mul', 'ave'}
        Tensor input dimensions:   (nb_sample, shared_dimension, input_dim)
        Tensor output dimensions:  (nb_sample, output_dim)
        '''
        self.mode = mode
        self.params = []
        self.regularizers = []
        self.constraints = []
        self.updates = []

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.mode == 'sum' or self.mode == 'ave':
            s = theano.tensor.sum(X, axis=1)
            if self.mode == 'ave':
                s /= X.shape[1]
            return s
        elif self.mode == 'mul':
            s = theano.tensor.mul(X, axis=1)
            return s
        else:
            raise Exception('Unknown merge mode')

    def get_config(self):
        return {"name": self.__class__.__name__,
                "mode": self.mode}


class Merge(Layer):
    def __init__(self, layers, mode='sum', concat_axis=-1):
        ''' Merge the output of a list of layers or containers into a single tensor.
            mode: {'sum', 'mul', 'concat', 'ave'}
        '''
        if len(layers) < 2:
            raise Exception("Please specify two or more input layers (or containers) to merge")
        self.mode = mode
        self.concat_axis = concat_axis
        self.layers = layers
        self.params = []
        self.regularizers = []
        self.constraints = []
        self.updates = []
        for l in self.layers:
            params, regs, consts, updates = l.get_params()
            self.regularizers += regs
            self.updates += updates
            # params and constraints have the same size
            for p, c in zip(params, consts):
                if p not in self.params:
                    self.params.append(p)
                    self.constraints.append(c)

    def get_params(self):
        return self.params, self.regularizers, self.constraints, self.updates

    def get_output(self, train=False):
        if self.mode == 'sum' or self.mode == 'ave':
            s = self.layers[0].get_output(train)
            for i in range(1, len(self.layers)):
                s += self.layers[i].get_output(train)
            if self.mode == 'ave':
                s /= len(self.layers)
            return s
        elif self.mode == 'concat':
            inputs = [self.layers[i].get_output(train) for i in range(len(self.layers))]
            return T.concatenate(inputs, axis=self.concat_axis)
        elif self.mode == 'mul':
            s = self.layers[0].get_output(train)
            for i in range(1, len(self.layers)):
                s *= self.layers[i].get_output(train)
            return s
        else:
            raise Exception('Unknown merge mode')

    def get_input(self, train=False):
        res = []
        for i in range(len(self.layers)):
            o = self.layers[i].get_input(train)
            if not type(o) == list:
                o = [o]
            for output in o:
                if output not in res:
                    res.append(output)
        return res

    @property
    def input(self):
        return self.get_input()

    def supports_masked_input(self):
        return False

    def get_output_mask(self, train=None):
        return None

    def get_weights(self):
        weights = []
        for l in self.layers:
            weights += l.get_weights()
        return weights

    def set_weights(self, weights):
        for i in range(len(self.layers)):
            nb_param = len(self.layers[i].params)
            self.layers[i].set_weights(weights[:nb_param])
            weights = weights[nb_param:]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "layers": [l.get_config() for l in self.layers],
                "mode": self.mode,
                "concat_axis": self.concat_axis}


class Dropout(MaskedLayer):
    '''
        Hinton's dropout.
    '''
    def __init__(self, p):
        super(Dropout, self).__init__()
        self.p = p
        self.srng = RandomStreams(seed=np.random.randint(10e6))

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.p > 0.:
            retain_prob = 1. - self.p
            if train:
                X *= self.srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            else:
                X *= retain_prob
        return X

    def get_config(self):
        return {"name": self.__class__.__name__,
                "p": self.p}


class Activation(MaskedLayer):
    '''
        Apply an activation function to an output.
    '''
    def __init__(self, activation, target=0, beta=0.1):
        super(Activation, self).__init__()
        self.activation = activations.get(activation)
        self.target = target
        self.beta = beta

    def get_output(self, train=False):
        X = self.get_input(train)
        return self.activation(X)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "activation": self.activation.__name__,
                "target": self.target,
                "beta": self.beta}


class Reshape(Layer):
    '''
        Reshape an output to a certain shape.
        Can't be used as first layer in a model (no fixed input!)
        First dimension is assumed to be nb_samples.
    '''
    def __init__(self, *dims):
        super(Reshape, self).__init__()
        self.dims = dims

    def get_output(self, train=False):
        X = self.get_input(train)
        nshape = make_tuple(X.shape[0], *self.dims)
        return theano.tensor.reshape(X, nshape)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "dims": self.dims}


class Permute(Layer):
    '''
        Permute the dimensions of the data according to the given tuple
    '''
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def get_output(self, train):
        X = self.get_input(train)
        return X.dimshuffle((0,) + self.dims)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "dims": self.dims}


class Flatten(Layer):
    '''
        Reshape input to flat shape.
        First dimension is assumed to be nb_samples.
    '''
    def __init__(self):
        super(Flatten, self).__init__()

    def get_output(self, train=False):
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
        super(RepeatVector, self).__init__()
        self.n = n

    def get_output(self, train=False):
        X = self.get_input(train)
        tensors = [X]*self.n
        stacked = theano.tensor.stack(*tensors)
        return stacked.dimshuffle((1, 0, 2))

    def get_config(self):
        return {"name": self.__class__.__name__,
                "n": self.n}


class Dense(Layer):
    '''
        Just your regular fully connected NN layer.
    '''
    def __init__(self, input_dim, output_dim, init='glorot_uniform', activation='linear', weights=None, name=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None):

        super(Dense, self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = T.matrix()
        self.W = self.init((self.input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.params = [self.W, self.b]

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

    def get_output(self, train=False):
        X = self.get_input(train)
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
                "b_constraint": self.b_constraint.get_config() if self.b_constraint else None}


class ActivityRegularization(Layer):
    '''
        Layer that passes through its input unchanged, but applies an update
        to the cost function based on the activity.
    '''
    def __init__(self, l1=0., l2=0.):
        super(ActivityRegularization, self).__init__()
        self.l1 = l1
        self.l2 = l2

        activity_regularizer = ActivityRegularizer(l1=l1, l2=l2)
        activity_regularizer.set_layer(self)
        self.regularizers = [activity_regularizer]

    def get_output(self, train=False):
        return self.get_input(train)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l1": self.l1,
                "l2": self.l2}


class TimeDistributedDense(MaskedLayer):
    '''
       Apply a same DenseLayer for each dimension[1] (shared_dimension) input
       Especially useful after a recurrent network with 'return_sequence=True'
       Tensor input dimensions:   (nb_sample, shared_dimension, input_dim)
       Tensor output dimensions:  (nb_sample, shared_dimension, output_dim)

    '''
    def __init__(self, input_dim, output_dim, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None):

        super(TimeDistributedDense, self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = T.tensor3()
        self.W = self.init((self.input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.params = [self.W, self.b]

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

    def get_output(self, train=False):
        X = self.get_input(train)
        output = self.activation(T.dot(X.dimshuffle(1, 0, 2), self.W) + self.b)
        return output.dimshuffle(1, 0, 2)

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
                "b_constraint": self.b_constraint.get_config() if self.b_constraint else None}


class AutoEncoder(Layer):
    '''
        A customizable autoencoder model.
        If output_reconstruction then dim(input) = dim(output)
        else dim(output) = dim(hidden)
    '''
    def __init__(self, encoder, decoder, output_reconstruction=True, weights=None):

        super(AutoEncoder, self).__init__()

        self.output_reconstruction = output_reconstruction
        self.encoder = encoder
        self.decoder = decoder

        self.decoder.set_previous(self.encoder)

        self.params = []
        self.regularizers = []
        self.constraints = []
        self.updates = []
        for layer in [self.encoder, self.decoder]:
            params, regularizers, constraints, updates = layer.get_params()
            self.regularizers += regularizers
            self.updates += updates
            for p, c in zip(params, constraints):
                if p not in self.params:
                    self.params.append(p)
                    self.constraints.append(c)

        if weights is not None:
            self.set_weights(weights)

    def set_previous(self, node):
        self.encoder.set_previous(node)

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

    def _get_hidden(self, train=False):
        return self.encoder.get_output(train)

    def get_output(self, train=False):
        if not train and not self.output_reconstruction:
            return self.encoder.get_output(train)

        return self.decoder.get_output(train)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "encoder_config": self.encoder.get_config(),
                "decoder_config": self.decoder.get_config(),
                "output_reconstruction": self.output_reconstruction}


class MaxoutDense(Layer):
    '''
        Max-out layer, nb_feature is the number of pieces in the piecewise linear approx.
        Refer to http://arxiv.org/pdf/1302.4389.pdf
    '''
    def __init__(self, input_dim, output_dim, nb_feature=4, init='glorot_uniform', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None):

        super(MaxoutDense, self).__init__()
        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nb_feature = nb_feature

        self.input = T.matrix()
        self.W = self.init((self.nb_feature, self.input_dim, self.output_dim))
        self.b = shared_zeros((self.nb_feature, self.output_dim))

        self.params = [self.W, self.b]

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

    def get_output(self, train=False):
        X = self.get_input(train)
        # -- don't need activation since it's just linear.
        output = T.max(T.dot(X, self.W) + self.b, axis=1)
        return output

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "nb_feature": self.nb_feature,
                "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                "b_constraint": self.b_constraint.get_config() if self.b_constraint else None}
