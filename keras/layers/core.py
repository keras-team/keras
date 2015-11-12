# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import theano
import theano.tensor as T
import numpy as np

from collections import OrderedDict
import copy

from .. import activations, initializations, regularizers, constraints
from ..utils.theano_utils import shared_zeros, floatX, ndim_tensor
from ..utils.generic_utils import make_tuple
from ..regularizers import ActivityRegularizer, Regularizer

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from six.moves import zip

import marshal
import types
import sys


class Layer(object):
    def __init__(self, **kwargs):
        for kwarg in kwargs:
            assert kwarg in {'input_shape', 'trainable'}, "Keyword argument not understood: " + kwarg
        if 'input_shape' in kwargs:
            self.set_input_shape(kwargs['input_shape'])
        if 'trainable' in kwargs:
            self._trainable = kwargs['trainable']
        if not hasattr(self, 'params'):
            self.params = []

    def set_previous(self, layer, connection_map={}):
        assert self.nb_input == layer.nb_output == 1, "Cannot connect layers: input count and output count should be 1."
        if hasattr(self, 'input_ndim'):
            assert self.input_ndim == len(layer.output_shape), "Incompatible shapes: layer expected input with ndim=" +\
                str(self.input_ndim) + " but previous layer has output_shape " + str(layer.output_shape)
        if layer.get_output_mask() is not None:
            assert self.supports_masked_input(), "Cannot connect non-masking layer to layer with masked output"
        self.previous = layer
        self.build()

    def build(self):
        '''Instantiation of layer weights.

        Called after `set_previous`, or after `set_input_shape`,
        once the layer has a defined input shape.
        Must be implemented on all layers that have weights.
        '''
        pass

    @property
    def trainable(self):
        if hasattr(self, '_trainable'):
            return self._trainable
        else:
            return True

    @trainable.setter
    def trainable(self, value):
        self._trainable = value

    @property
    def nb_input(self):
        return 1

    @property
    def nb_output(self):
        return 1

    @property
    def input_shape(self):
        # if layer is not connected (e.g. input layer),
        # input shape can be set manually via _input_shape attribute.
        if hasattr(self, 'previous'):
            return self.previous.output_shape
        elif hasattr(self, '_input_shape'):
            return self._input_shape
        else:
            raise Exception('Layer is not connected. Did you forget to set "input_shape"?')

    def set_input_shape(self, input_shape):
        if type(input_shape) not in [tuple, list]:
            raise Exception('Invalid input shape - input_shape should be a tuple of int.')
        input_shape = (None,) + tuple(input_shape)
        if hasattr(self, 'input_ndim') and self.input_ndim:
            if self.input_ndim != len(input_shape):
                raise Exception('Invalid input shape - Layer expects input ndim=' +
                                str(self.input_ndim) + ', was provided with input shape ' + str(input_shape))
        self._input_shape = input_shape
        self.input = ndim_tensor(len(self._input_shape))
        self.build()

    @property
    def output_shape(self):
        # default assumption: tensor shape unchanged.
        return self.input_shape

    def get_output(self, train=False):
        return self.get_input(train)

    def get_input(self, train=False):
        if hasattr(self, 'previous'):
            return self.previous.get_output(train=train)
        elif hasattr(self, 'input'):
            return self.input
        else:
            raise Exception('Layer is not connected\
                and is not an input layer.')

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
        assert len(self.params) == len(weights), 'Provided weight array does not match layer weights (' + \
            str(len(self.params)) + ' layer params vs. ' + str(len(weights)) + ' provided weights)'
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
        config = {"name": self.__class__.__name__}
        if hasattr(self, '_input_shape'):
            config['input_shape'] = self._input_shape[1:]
        if hasattr(self, '_trainable'):
            config['trainable'] = self._trainable
        return config

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
    def __init__(self, mask_value=0., **kwargs):
        super(Masking, self).__init__(**kwargs)
        self.mask_value = mask_value
        self.input = T.tensor3()

    def get_output_mask(self, train=False):
        X = self.get_input(train)
        return T.any(T.ones_like(X) * (1. - T.eq(X, self.mask_value)), axis=-1)

    def get_output(self, train=False):
        X = self.get_input(train)
        return X * T.shape_padright(T.any((1. - T.eq(X, self.mask_value)), axis=-1))

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "mask_value": self.mask_value}
        base_config = super(Masking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TimeDistributedMerge(Layer):
    '''Sum/multiply/average over the outputs of a TimeDistributed layer.

    mode: {'sum', 'mul', 'ave'}
    Tensor input dimensions:   (nb_sample, time, features)
    Tensor output dimensions:  (nb_sample, features)
    '''
    input_ndim = 3

    def __init__(self, mode='sum', **kwargs):
        super(TimeDistributedMerge, self).__init__(**kwargs)
        self.mode = mode
        self.params = []
        self.regularizers = []
        self.constraints = []
        self.updates = []

    @property
    def output_shape(self):
        return (None, self.input_shape[2])

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.mode == 'ave':
            s = theano.tensor.mean(X, axis=1)
            return s
        if self.mode == 'sum':
            s = theano.tensor.sum(X, axis=1)
            return s
        elif self.mode == 'mul':
            s = theano.tensor.mul(X, axis=1)
            return s
        else:
            raise Exception('Unknown merge mode')

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "mode": self.mode}
        base_config = super(TimeDistributedMerge, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Merge(Layer):
    def __init__(self, layers, mode='sum', concat_axis=-1, dot_axes=-1):
        ''' Merge the output of a list of layers or containers into a single tensor.
            mode: {'sum', 'mul', 'concat', 'ave', 'join'}
        '''
        if len(layers) < 2:
            raise Exception("Please specify two or more input layers (or containers) to merge")

        if mode not in {'sum', 'mul', 'concat', 'ave', 'join', 'cos', 'dot'}:
            raise Exception("Invalid merge mode: " + str(mode))

        if mode in {'sum', 'mul', 'ave', 'cos'}:
            input_shapes = set([l.output_shape for l in layers])
            if len(input_shapes) > 1:
                raise Exception("Only layers of same output shape can be merged using " + mode + " mode. " +
                                "Layer shapes: %s" % ([l.output_shape for l in layers]))
        if mode in {'cos', 'dot'}:
            if len(layers) > 2:
                raise Exception(mode + " merge takes exactly 2 layers")
            shape1 = layers[0].output_shape
            shape2 = layers[1].output_shape
            n1 = len(shape1)
            n2 = len(shape2)
            if mode == 'dot':
                if type(dot_axes) == int:
                    if dot_axes < 0:
                        dot_axes = [range(dot_axes % n1, n1), range(dot_axes % n2, n2)]
                    else:
                        dot_axes = [range(n1 - dot_axes, n2), range(1, dot_axes + 1)]
                if type(dot_axes) not in [list, tuple]:
                    raise Exception("Invalid type for dot_axes - should be a list.")
                if len(dot_axes) != 2:
                    raise Exception("Invalid format for dot_axes - should contain two elements.")
                if type(dot_axes[0]) not in [list, tuple, range] or type(dot_axes[1]) not in [list, tuple, range]:
                    raise Exception("Invalid format for dot_axes - list elements should have type 'list' or 'tuple'.")
                for i in range(len(dot_axes[0])):
                    if shape1[dot_axes[0][i]] != shape2[dot_axes[1][i]]:
                        raise Exception("Dimension incompatibility using dot mode: " +
                                        "%s != %s. " % (shape1[dot_axes[0][i]], shape2[dot_axes[1][i]]) +
                                        "Layer shapes: %s, %s" % (shape1, shape2))
        elif mode == 'concat':
            input_shapes = set()
            for l in layers:
                oshape = list(l.output_shape)
                oshape.pop(concat_axis)
                oshape = tuple(oshape)
                input_shapes.add(oshape)
            if len(input_shapes) > 1:
                raise Exception("'concat' mode can only merge layers with matching " +
                                "output shapes except for the concat axis. " +
                                "Layer shapes: %s" % ([l.output_shape for l in layers]))

        self.mode = mode
        self.concat_axis = concat_axis
        self.dot_axes = dot_axes
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

    @property
    def output_shape(self):
        input_shapes = [layer.output_shape for layer in self.layers]
        if self.mode in ['sum', 'mul', 'ave']:
            return input_shapes[0]
        elif self.mode == 'concat':
            output_shape = list(input_shapes[0])
            for shape in input_shapes[1:]:
                output_shape[self.concat_axis] += shape[self.concat_axis]
            return tuple(output_shape)
        elif self.mode == 'join':
            return None
        elif self.mode == 'dot':
            shape1 = list(input_shapes[0])
            shape2 = list(input_shapes[1])
            dot_axes = []
            for axes in self.dot_axes:
                dot_axes.append([index-1 for index in axes])
            tensordot_output = np.tensordot(np.zeros(tuple(shape1[1:])),
                                            np.zeros(tuple(shape2[1:])),
                                            axes=dot_axes)
            if len(tensordot_output.shape) == 0:
                shape = (1,)
            else:
                shape = tensordot_output.shape
            return (shape1[0],) + shape
        elif self.mode == 'cos':
            return tuple(input_shapes[0][0], 1)

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
        elif self.mode == 'join':
            inputs = OrderedDict()
            for i in range(len(self.layers)):
                X = self.layers[i].get_output(train)
                if X.name is None:
                    raise ValueError("merge_mode='join' only works with named inputs")
                else:
                    inputs[X.name] = X
            return inputs
        elif self.mode == 'mul':
            s = self.layers[0].get_output(train)
            for i in range(1, len(self.layers)):
                s *= self.layers[i].get_output(train)
            return s
        elif self.mode == 'dot':
            l1 = self.layers[0].get_output(train)
            l2 = self.layers[1].get_output(train)
            output = T.batched_tensordot(l1, l2, self.dot_axes)
            output_shape = list(self.output_shape)
            output_shape[0] = l1.shape[0]
            output = output.reshape(tuple(output_shape))
            return output
        elif self.mode == 'cos':
            l1 = self.layers[0].get_output(train)
            l2 = self.layers[1].get_output(train)
            output, _ = theano.scan(lambda v1, v2: T.dot(v1, v2) / T.sqrt(T.dot(v1, v1) * T.dot(v2, v2)),
                                    sequences=[l1, l2],
                                    outputs_info=None)
            return output
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
        config = {"name": self.__class__.__name__,
                  "layers": [l.get_config() for l in self.layers],
                  "mode": self.mode,
                  "concat_axis": self.concat_axis,
                  "dot_axes": self.dot_axes}
        base_config = super(Merge, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Dropout(MaskedLayer):
    '''
        Hinton's dropout.
    '''
    def __init__(self, p, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.p = p
        self.srng = RandomStreams(seed=np.random.randint(10e6))

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.p > 0.:
            retain_prob = 1. - self.p
            if train:
                X *= self.srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX) / retain_prob
        return X

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "p": self.p}
        base_config = super(Dropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Activation(MaskedLayer):
    '''
        Apply an activation function to an output.
    '''
    def __init__(self, activation, target=0, beta=0.1, **kwargs):
        super(Activation, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.target = target
        self.beta = beta

    def get_output(self, train=False):
        X = self.get_input(train)
        return self.activation(X)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "activation": self.activation.__name__,
                  "target": self.target,
                  "beta": self.beta}
        base_config = super(Activation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Reshape(Layer):
    '''
        Reshape an output to a certain shape.
        Can't be used as first layer in a model (no fixed input!)
        First dimension is assumed to be nb_samples.
    '''
    def __init__(self, dims, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self.dims = tuple(dims)

    @property
    def output_shape(self):
        return (self.input_shape[0],) + self.dims

    def get_output(self, train=False):
        X = self.get_input(train)
        new_shape = (X.shape[0],) + self.dims
        return theano.tensor.reshape(X, new_shape)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "dims": self.dims}
        base_config = super(Reshape, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Permute(Layer):
    '''
        Permute the dimensions of the input according to the given tuple.
    '''
    def __init__(self, dims, **kwargs):
        super(Permute, self).__init__(**kwargs)
        self.dims = tuple(dims)

    @property
    def output_shape(self):
        input_shape = list(self.input_shape)
        output_shape = copy.copy(input_shape)
        for i, dim in enumerate(self.dims):
            target_dim = input_shape[dim]
            output_shape[i+1] = target_dim
        return tuple(output_shape)

    def get_output(self, train=False):
        X = self.get_input(train)
        return X.dimshuffle((0,) + self.dims)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "dims": self.dims}
        base_config = super(Permute, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Flatten(Layer):
    '''
        Reshape input to flat shape.
        First dimension is assumed to be nb_samples.
    '''
    def __init__(self, **kwargs):
        super(Flatten, self).__init__(**kwargs)

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], np.prod(input_shape[1:]))

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
    def __init__(self, n, **kwargs):
        super(RepeatVector, self).__init__(**kwargs)
        self.n = n

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], self.n, input_shape[1])

    def get_output(self, train=False):
        X = self.get_input(train)
        tensors = [X]*self.n
        stacked = theano.tensor.stack(*tensors)
        return stacked.dimshuffle((1, 0, 2))

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "n": self.n}
        base_config = super(RepeatVector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Dense(Layer):
    '''
        Just your regular fully connected NN layer.
    '''
    input_ndim = 2

    def __init__(self, output_dim, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Dense, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[1]

        self.input = T.matrix()
        self.W = self.init((input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim,))

        self.params = [self.W, self.b]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        output = self.activation(T.dot(X, self.W) + self.b)
        return output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "activation": self.activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                  "b_constraint": self.b_constraint.get_config() if self.b_constraint else None,
                  "input_dim": self.input_dim}
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ActivityRegularization(Layer):
    '''
        Layer that passes through its input unchanged, but applies an update
        to the cost function based on the activity.
    '''
    def __init__(self, l1=0., l2=0., **kwargs):
        super(ActivityRegularization, self).__init__(**kwargs)
        self.l1 = l1
        self.l2 = l2

        activity_regularizer = ActivityRegularizer(l1=l1, l2=l2)
        activity_regularizer.set_layer(self)
        self.regularizers = [activity_regularizer]

    def get_output(self, train=False):
        return self.get_input(train)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "l1": self.l1,
                  "l2": self.l2}
        base_config = super(ActivityRegularization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TimeDistributedDense(MaskedLayer):
    '''
       Apply a same Dense layer for each dimension[1] (time_dimension) input.
       Especially useful after a recurrent network with 'return_sequence=True'.
       Tensor input dimensions:   (nb_sample, time_dimension, input_dim)
       Tensor output dimensions:  (nb_sample, time_dimension, output_dim)

    '''
    input_ndim = 3

    def __init__(self, output_dim, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(TimeDistributedDense, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        self.input = T.tensor3()
        self.W = self.init((input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.params = [self.W, self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        output = self.activation(T.dot(X.dimshuffle(1, 0, 2), self.W) + self.b)
        return output.dimshuffle(1, 0, 2)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "activation": self.activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                  "b_constraint": self.b_constraint.get_config() if self.b_constraint else None,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(TimeDistributedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AutoEncoder(Layer):
    '''A customizable autoencoder model.

    Tensor input dimensions: same as encoder input
    Tensor output dimensions:
        if output_reconstruction:
            same as encoder output
        else:
            same as decoder output
    '''
    def __init__(self, encoder, decoder, output_reconstruction=True, weights=None, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)

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

    @property
    def input_shape(self):
        return self.encoder.input_shape

    @property
    def output_shape(self):
        if self.output_reconstruction:
            return self.encoder.previous.output_shape
        else:
            return self.decoder.previous.output_shape

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
    input_ndim = 2

    def __init__(self, output_dim, nb_feature=4, init='glorot_uniform', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, input_dim=None, **kwargs):
        self.output_dim = output_dim
        self.nb_feature = nb_feature
        self.init = initializations.get(init)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights
        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MaxoutDense, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[1]

        self.input = T.matrix()
        self.W = self.init((self.nb_feature, input_dim, self.output_dim))
        self.b = shared_zeros((self.nb_feature, self.output_dim))

        self.params = [self.W, self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        # -- don't need activation since it's just linear.
        output = T.max(T.dot(X, self.W) + self.b, axis=1)
        return output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "nb_feature": self.nb_feature,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                  "b_constraint": self.b_constraint.get_config() if self.b_constraint else None,
                  "input_dim": self.input_dim}
        base_config = super(MaxoutDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Lambda(Layer):

    """Lambda layer for evaluating arbitrary function

    Input shape
    -----------
    output_shape of previous layer

    Output shape
    ------------
    Specified by output_shape argument

    Arguments
    ---------
    function - The function to be evaluated. Takes one argument : output of previous layer
    output_shape - Expected output shape from function. Could be a tuple or a function of the shape of the input
    """

    def __init__(self, function, output_shape=None):
        super(Lambda, self).__init__()
        py3 = sys.version_info[0] == 3
        if py3:
            self.function = marshal.dumps(function.__code__)
        else:
            self.function = marshal.dumps(function.func_code)
        if output_shape is None:
            self._output_shape = None
        elif type(output_shape) in {tuple, list} :
            self._output_shape = tuple(output_shape)
        else:
            if py3:
                self._output_shape = marshal.dumps(output_shape.__code__)
            else:
                self._output_shape = marshal.dumps(output_shape.func_code)

    @property
    def output_shape(self):
        if self._ouput_shape is None:
            return self.input_shape
        elif type(self._output_shape) == tuple:
            return (self.input_shape[0], ) + self._output_shape
        else:
            output_shape_func = marshal.loads(self._output_shape)
            output_shape_func = types.FunctionType(output_shape_func, globals())
            shape = output_shape_func(self.previous.output_shape)
            if type(shape) not in {list, tuple}:
                raise Exception("output_shape function must return a tuple")
            return tuple(shape)

    def get_output(self, train=False):
        func = marshal.loads(self.function)
        func = types.FunctionType(func, globals())
        if hasattr(self, 'previous'):
            return func(self.previous.get_output(train))
        else:
            return func(self.input)


class MaskedLambda(MaskedLayer, Lambda):
    pass


class LambdaMerge(Lambda):
    """LambdaMerge layer for evaluating arbitrary function over multiple inputs

    Input shape
    -----------
    None

    Output shape
    ------------
    Specified by output_shape argument

    Arguments
    ---------
    layers - Input layers. Similar to layers argument of Merge
    function - The function to be evaluated. Takes one argument : list of outputs from input layers
    output_shape - Expected output shape from function. Could be a tuple or a function of list of input shapes
    """
    def __init__(self, layers, function, output_shape=None):
        if len(layers) < 2:
            raise Exception("Please specify two or more input layers (or containers) to merge")
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
        py3 = sys.version_info[0] == 3
        if py3:
            self.function = marshal.dumps(function.__code__)
        else:
            self.function = marshal.dumps(function.func_code)
        if output_shape is None:
            self._output_shape = None
        elif type(output_shape) in {tuple, list}:
            self._output_shape = tuple(output_shape)
        else:
            if py3:
                self._output_shape = marshal.dumps(output_shape.__code__)
            else:
                self._output_shape = marshal.dumps(output_shape.func_code)

    @property
    def output_shape(self):
        input_shapes = [layer.output_shape for layer in self.layers]
        if self._output_shape is None:
            return input_shapes[0]
        elif type(self._output_shape) == tuple:
            return (input_shapes[0][0], ) + self._output_shape
        else:
            output_shape_func = marshal.loads(self._output_shape)
            output_shape_func = types.FunctionType(output_shape_func, globals())
            shape = output_shape_func(input_shapes)
            if type(shape) not in {list, tuple}:
                raise Exception("output_shape function must return a tuple")
            return tuple(shape)

    def get_params(self):
        return self.params, self.regularizers, self.constraints, self.updates

    def get_output(self, train=False):
        func = marshal.loads(self.function)
        func = types.FunctionType(func, globals())
        inputs = [layer.get_output(train) for layer in self.layers]
        return func(inputs)

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
        config = {"name": self.__class__.__name__,
                  "layers": [l.get_config() for l in self.layers],
                  "function": self.function,
                  "output_shape": self._output_shape
                  }
        base_config = super(LambdaMerge, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
