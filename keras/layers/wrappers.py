# -*- coding: utf-8 -*-
from __future__ import absolute_import
from ..layers.core import Layer, MaskedLayer
from ..import backend as K


class Wrapper(MaskedLayer):
    '''
    Abstract Wrapper class
    '''
    def __init__(self, layer, initial_weights=None, **kwargs):
        self.layer = layer
        super(Wrapper, self).__init__(**kwargs)
        if hasattr(layer, '_input_shape'): #Layer is already built.
            self.set_input_shape(self.get_input_shape())
            self.set_params()
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
        # If child layer is a stateful RNN
        self.layer.reset_states()

    def set_params(self):
        layer_params = self.layer.get_params()
        self.trainable_weights = layer_params[0]
        self.regularizers = layer_params[1]
        self.constraints = layer_params[2]
        self.updates = layer_params[3]

    def build(self):
        if not hasattr(self.layer, '_input_shape'):
            self.layer.set_input_shape(self.get_child_input_shape())
            self.set_params()

    def get_input_shape(self):
        '''
        Get input shape of the wrapper from the child layer's input shape
        '''
        return self.layer.input_shape

    def get_child_input_shape(self):
        '''
        Get the child layer's input shape from the wrapper's input shape
        '''
        return self.input_shape

    def get_output_shape(self):
        '''
        Get output shape of the wrapper from the child layer's output shape
        '''
        return self.layer.output_shape

    @property
    def trainable_weights(self):
        return self.layer.trainable_weights

    @trainable_weights.setter
    def trainable_weights(self, value):
        self.layer.trainable_weights = value

    @property
    def non_trainable_weights(self):
        return self.layer.non_trainable_weights

    @non_trainable_weights.setter
    def non_trainable_weights(self, value):
        self.layer.non_trainable_weights = value

    def get_weights(self):
        return self.layer.get_weights()

    def set_weights(self, weights):
        self.layer.set_weights(weights)

    def get_output(self, train=False):
        x = self.get_input(train)
        return self.layer(x, None, train)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'layer': self.layer.get_config()}
        base_config = super(Wrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
