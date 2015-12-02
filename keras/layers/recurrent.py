# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from .. import backend as K
from .. import activations, initializations
from ..layers.core import MaskedLayer


class Recurrent(MaskedLayer):
    input_ndim = 3

    def __init__(self, weights=None,
                 return_sequences=False, go_backwards=False, stateful=False,
                 input_dim=None, input_length=None, **kwargs):
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.go_backwards = go_backwards
        self.stateful = stateful

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(Recurrent, self).__init__(**kwargs)

    def get_output_mask(self, train=False):
        if self.return_sequences:
            return super(Recurrent, self).get_output_mask(train)
        else:
            return None

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.output_dim)
        else:
            return (input_shape[0], self.output_dim)

    def step(self, x, states):
        raise NotImplementedError

    def get_output(self, train=False):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        X = self.get_input(train)
        assert K.ndim(X) == 3
        if K._BACKEND == 'tensorflow':
            if not self.input_shape[1]:
                raise Exception('When using TensorFlow, you should define ' +
                                'explicitely the number of timesteps of ' +
                                'your sequences. Make sure the first layer ' +
                                'has a "batch_input_shape" argument ' +
                                'including the samples axis.')

        mask = self.get_output_mask(train)
        if mask:
            # apply mask
            X *= K.expand_dims(mask)
            masking = True
        else:
            masking = False

        if self.stateful:
            initial_states = self.states
        else:
            # build an all-zero tensor of shape (samples, output_dim)
            initial_state = K.zeros_like(X)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
            reducer = K.zeros((self.input_dim, self.output_dim))
            initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
            initial_states = [initial_state for _ in range(len(self.states))]

        last_output, outputs, states = K.rnn(self.step, X, initial_states,
                                             go_backwards=self.go_backwards,
                                             masking=masking)
        if self.stateful:
            for i in range(len(states)):
                K.set_value(self.states[i], states[i])

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length,
                  "go_backwards": self.go_backwards,
                  "stateful": self.stateful}
        base_config = super(Recurrent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SimpleRNN(Recurrent):
    '''
        Fully-connected RNN where the output is to fed back to input.
        Takes inputs with shape:
        (nb_samples, max_sample_length, input_dim)
        (samples shorter than `max_sample_length`
         are padded with zeros at the end)
        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        super(SimpleRNN, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        if self.stateful:
            if not input_shape[0]:
                raise Exception('If a RNN is stateful, a complete ' +
                                'input_shape must be provided ' +
                                '(including batch size).')
            self.states = [K.zeros(input_shape[0], self.output_dim)]
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.W = self.init((input_dim, self.output_dim))
        self.U = self.inner_init((self.output_dim, self.output_dim))
        self.b = K.zeros((self.output_dim))
        self.params = [self.W, self.U, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def step(self, x, states):
        # states only contains the previous output.
        assert len(states) == 1
        prev_output = states[0]
        h = K.dot(x, self.W) + self.b
        output = self.activation(h * K.dot(prev_output, self.U))
        return output, [output]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__}
        base_config = super(SimpleRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GRU(Recurrent):
    '''
        Gated Recurrent Unit - Cho et al. 2014
        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.
        Takes inputs with shape:
        (nb_samples, max_sample_length, input_dim)
        (samples shorter than `max_sample_length`
         are padded with zeros at the end)
        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)
        References:
            On the Properties of Neural Machine Translation:
            Encoder–Decoder Approaches
                http://www.aclweb.org/anthology/W14-4012
            Empirical Evaluation of Gated Recurrent Neural Networks
            on Sequence Modeling
                http://arxiv.org/pdf/1412.3555v1.pdf
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        super(GRU, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        input_dim = input_shape[2]
        self.input_dim = input_dim
        self.input = K.placeholder(input_shape)

        self.W_z = self.init((input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = K.zeros((self.output_dim,))

        self.W_r = self.init((input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = K.zeros((self.output_dim,))

        self.W_h = self.init((input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = K.zeros((self.output_dim,))

        self.params = [self.W_z, self.U_z, self.b_z,
                       self.W_r, self.U_r, self.b_r,
                       self.W_h, self.U_h, self.b_h]

        if self.stateful:
            if not input_shape[0]:
                raise Exception('If a RNN is stateful, a complete ' +
                                'input_shape must be provided ' +
                                '(including batch size).')
            self.states = [K.zeros(input_shape[0], self.output_dim)]
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def step(self, x, states):
        assert len(states) == 1
        x_z = K.dot(x, self.W_z) + self.b_z
        x_r = K.dot(x, self.W_r) + self.b_r
        x_h = K.dot(x, self.W_h) + self.b_h

        h_tm1 = states[0]
        z = self.inner_activation(x_z + K.dot(h_tm1, self.U_z))
        r = self.inner_activation(x_r + K.dot(h_tm1, self.U_r))

        hh = self.inner_activation(x_h + K.dot(r * h_tm1, self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__}
        base_config = super(GRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LSTM(Recurrent):
    '''
        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.
        Takes inputs with shape:
        (nb_samples, max_sample_length, input_dim)
        (samples shorter than `max_sample_length`
         are padded with zeros at the end)
        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)
        For a step-by-step description of the algorithm, see:
        http://deeplearning.net/tutorial/lstm.html
        References:
            Long short-term memory (original 97 paper)
                http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
            Learning to forget: Continual prediction with LSTM
                http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015
            Supervised sequence labelling with recurrent neural networks
                http://www.cs.toronto.edu/~graves/preprint.pdf
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid', **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        super(LSTM, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        input_dim = input_shape[2]
        self.input_dim = input_dim
        self.input = K.placeholder(input_shape)

        if self.stateful:
            if not input_shape[0]:
                raise Exception('If a RNN is stateful, a complete ' +
                                'input_shape must be provided ' +
                                '(including batch size).')
            self.states = [K.zeros(input_shape[0], self.output_dim),
                           K.zeros(input_shape[0], self.output_dim)]
        else:
            # initial states: 2 all-zero tensor of shape (output_dim)
            self.states = [None, None]

        self.W_i = self.init((input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = K.zeros((self.output_dim))

        self.W_f = self.init((input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.output_dim))

        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = K.zeros((self.output_dim))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = K.zeros((self.output_dim))

        self.params = [self.W_i, self.U_i, self.b_i,
                       self.W_c, self.U_c, self.b_c,
                       self.W_f, self.U_f, self.b_f,
                       self.W_o, self.U_o, self.b_o]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def step(self, x, states):
        assert len(states) == 2
        h_tm1 = states[0]
        c_tm1 = states[1]

        x_i = K.dot(x, self.W_i) + self.b_i
        x_f = K.dot(x, self.W_f) + self.b_f
        x_c = K.dot(x, self.W_c) + self.b_c
        x_o = K.dot(x, self.W_o) + self.b_o

        i = self.inner_activation(x_i + K.dot(h_tm1, self.U_i))
        f = self.inner_activation(x_f + K.dot(h_tm1, self.U_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1, self.U_c))
        o = self.inner_activation(x_o + K.dot(h_tm1, self.U_o))
        h = o * self.activation(c)
        return h, [h, c]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__}
        base_config = super(LSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
