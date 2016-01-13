# -*- coding: utf-8 -*-
from __future__ import absolute_import
from ..layers.core import Layer
from ..import backend as K
from ..import activations, initializations, regularizers, constraints

class Wrapper(Layer):
    '''
    Abstract Wrapper class
    '''
    def __init__(self, layer, initial_weights=None, **kwargs):
        self.layer = layer
        super(Wrapper, self).__init__(**kwargs)
        if hasattr(layer, '_input_shape'): #Layer is already built.
            self.set_input_shape(self.get_input_shape())
        if initial_weights:
            self.set_weights(initial_weights)
            del initial_weights

    @property
    def output_shape(self):
        return self.get_output_shape()

    @property
    def cache_enabled(self):
        return self._cache_enabled

    @cache_enabled.setter
    def cache_enabled(self, value):
        self._cache_enabled = value
        self.layer.cache_enabled = value

    def reset_states(self):
        self.layer.reset_states()

    def set_params(self):
        layer_params = self.layer.get_params()
        self.params = layer_params[0]
        self.regularizers = layer_params[1]
        self.constraints = layer_params[2]
        self.updates = layer_params[3]

    def build(self):
        self.layer.set_input_shape(self.input_shape)
        self.set_params()

    def get_input_shape(self):
        '''
        Get input shape of the wrapper from the layer's input shape
        '''
        return self.layer.input_shape

    def get_output_shape(self):
        '''
        Get output shape of the wrapper from the layer's output shape
        '''
        return self.layer.output_shape

    def get_output(self, train=False):
        x = self.get_input(train)
        return self.layer(x, None, train)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'layer': self.layer.get_config()}
        base_config = super(Wrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class TimeDistributed(MaskedLayer, Wrapper):

    input_ndim = 3

    def get_output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1], self.layer.output_shape[-1])

    def get_input_shape(self):
        layer_input_shape = self.layer.input_shape
        timesteps = None if not hasattr(self, '_input_shape') else self._input_shape[1]
        return  (layer_input_shape[0], timesteps, layer_input_shape[1])

    def build(self):
        input_shape = self.input_shape
        self.layer.set_input_shape((input_shape[0], input_shape[-1]))
        self.set_params()


    def get_output(self, train=False):
        X = self.get_input(train)

        def step(x, states):
            return self.layer(x), []

        return K.rnn(step, X, [])[1]

class Highway(Wrapper):

    input_ndim = 2

    def __init__(self, layer, init='glorot_uniform', transform_bias=-2, **kwargs):
        self.init = initializations.get(init)
        self.transform_bias = transform_bias
        super(Highway, self).__init__(layer, **kwargs)

    def build(self):
        super(Highway, self).build()
        input_dim = self.input_shape[1]
        output_dim = self.output_shape[1]
        assert input_dim ==  output_dim, 'Input and output shapes should be equal for highway layers.'
        self.input_dim = input_dim
        self.W = self.init((input_dim, input_dim))
        self.b = K.variable(np.ones((input_dim,)) * self.transform_bias)
        self.params += [self.W, self.b]

    def get_output(self, train=False):
        X = self.get_input(train)
        alpha = K.sigmoid(K.dot(X, self.W) + self.b)
        Y_hat = self.layer(X)
        Y  = (Y_hat * alpha) + (X * (1 - alpha))
        return Y

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'layer': self.layer.get_config(),
                  'init': self.init.__name_,
                  'transform_bias': self.transform_bias}
        base_config = super(Highway, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
