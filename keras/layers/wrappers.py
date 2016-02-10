# -*- coding: utf-8 -*-
from __future__ import absolute_import
from ..layers.core import Layer, MaskedLayer
from ..import backend as K
from ..import activations, initializations, regularizers, constraints
import cPickle

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
        self.trainable_weights = layer_params[0]
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
        return tuple([input_shape[0], input_shape[1]] + list(self.layer.output_shape[1:]))

    def get_input_shape(self):
        layer_input_shape = self.layer.input_shape
        self.input_ndim = 1 + len(layer_input_shape)
        timesteps = None if not hasattr(self, '_input_shape') else self._input_shape[1]
        return  tuple([layer_input_shape[0], timesteps] + list(layer_input_shape[1:]))

    def build(self):
        input_shape = self.input_shape
        self.layer.set_input_shape(tuple([input_shape[0]] + list(input_shape[2:])))
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
        self.trainable_weights += [self.W, self.b]

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
        

class Bidirectional(MaskedLayer):
    ''' Bidirectional wrapper for RNNs

    # Arguments:
        rnn: `Recurrent` object. 
        merge_mode: Mode by which outputs of the forward and reverse RNNs will be combined. One of {sum, mul, concat, ave}

    # Examples:
    ```python
    model = Sequential()
    model.add(Bidirectional(LSTM(10, input_shape=(10, 20))))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit([X_train,], Y_train, batch_size=32, nb_epoch=20,
              validation_data=([X_test], Y_test))
    ```
    '''
    def __init__(self, rnn, merge_mode='concat', weights=None):

        self.forward = rnn
        self.reverse = cPickle.loads(cPickle.dumps(rnn))
        self.merge_mode = merge_mode
        if weights:
            nw = len(weights)
            self.forward.initial_weights = weights[:nw/2]
            self.reverse.initial_weights = weights[nw/2:]
        self._cache_enabled = True
        self.stateful = rnn.stateful
        self.return_sequences = rnn.return_sequences
        if hasattr(rnn, '_input_shape'):
            self._input_shape = rnn.input_shape
        elif hasattr(rnn, 'previous') and rnn.previous:
            self.previous = rnn.previous

    def get_weights(self):
        return self.forward.get_weights() + self.reverse.get_weights()

    def set_weights(self, weights):
        nw = len(weights)
        self.forward.set_weights(weights[:nw/2])
        self.reverse.set_weights(weights[:nw/2])

    def set_previous(self, layer):
        self.previous = layer
        self.forward.set_previous(layer)
        self.reverse.set_previous(layer)
        self._input_shape = layer.output_shape

    @property
    def cache_enabled(self):
        return self._cache_enabled

    @cache_enabled.setter
    def cache_enabled(self, value):
        self._cache_enabled = value
        self.forward.cache_enabled = value
        self.reverse.cache_enabled = value

    @property
    def output_shape(self):
        if self.merge_mode in ['sum', 'ave', 'mul']:
            return self.forward.output_shape
        elif self.merge_mode == 'concat':
            shape = list(self.forward.output_shape)
            shape[-1] *= 2
            return tuple(shape)

    def get_output(self, train=False):
        X = self.get_input(train) # 0,0,0,1,2,3,4
        mask = self.get_input_mask(train) # 0,0,0,1,1,1,1

        def reverse(x):
            rev = K.permute_dimensions(x, (1, 0, 2))[::-1]
            return K.permute_dimensions(rev, (1, 0, 2))

        X_rev = reverse(X) # 4,3,2,1,0,0,0
        Y = self.forward(X, mask) # 0,0,0,1,3,6,10
        mask_rev = reverse(mask) if mask else None # 1,1,1,1,0,0,0
        Y_rev = self.reverse(X_rev, mask_rev) # 4,7,9,10,10,10,10

        #Fix allignment
        if self.return_sequences:
            Y_rev = reverse(Y_rev) # 10,10,10,10,9,7,4

        if self.merge_mode == 'concat':
            return K.concatenate([Y, Y_rev])
        elif self.merge_mode == 'sum':
            return Y + Y_rev
        elif self.merge_mode == 'ave':
            return (Y + Y_rev) / 2
        elif self.merge_mode == 'mul':
            return Y * Y_rev

    def get_output_mask(self, train=False):
        if self.forward.return_sequences:
            return self.get_input_mask(train)
        else:
            return None

    @property
    def input_shape(self):
        return self.forward.input_shape

    def get_input(self, train=False):
        return self.forward.get_input(train)

    @property
    def non_trainable_weights(self):
        return self.forward.non_trainable_weights + self.reverse.non_trainable_weights

    @property
    def trainable_weights(self):
        return self.forward.trainable_weights + self.reverse.trainable_weights

    @property
    def regularizers(self):
        return self.forward.get_params()[1] + self.reverse.get_params()[1] 

    @property
    def constraints(self):
        return self.forward.get_params()[2] + self.reverse.get_params()[2]

    @property
    def updates(self):
        return self.forward.get_params()[3] + self.reverse.get_params()[3]

    def reset_states(self):
        self.forward.reset_states()
        self.reverse.reset_states()

    def build(self):
        if not hasattr(self.forward, '_input_shape'):
            if hasattr(self, '_input_shape'):
                self.forward._input_shape = self._input_shape
                self.reverse._input_shape = self._input_shape
                self.forward.previous = self.previous
                self.reverse.previous = self.previous
                self.forward.trainable_weights = []
                self.reverse.trainable_weights = []
                self.forward.build()
                self.reverse.build()

    def get_config(self):
        config = {
                  "name": self.__class__.__name__,
                  "rnn": self.forward.get_config(),
                  "merge_mode": self.merge_mode}
        base_config = super(Bidirectional, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
