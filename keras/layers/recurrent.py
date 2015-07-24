# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from .. import activations, initializations
from ..utils.theano_utils import shared_zeros, alloc_zeros_matrix
from ..layers.core import MaskedLayer
from six.moves import range


class Recurrent(MaskedLayer):
    def get_output_mask(self, train=None):
        if self.return_sequences:
            return super(Recurrent, self).get_output_mask(train)
        else:
            return None

    def get_padded_shuffled_mask(self, train, X, pad=0):
        mask = self.get_input_mask(train)
        if mask is None:
            mask = T.ones_like(X.sum(axis=-1))  # is there a better way to do this without a sum?

        # mask is (nb_samples, time)
        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)

        if pad > 0:
            # left-pad in time with 0
            padding = alloc_zeros_matrix(pad, mask.shape[1], 1)
            mask = T.concatenate([padding, mask], axis=0)
        return mask.astype('int8')


class SimpleRNN(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, input_dim, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', activation='sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 go_backwards=False):

        super(SimpleRNN, self).__init__()
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards

        self.input = T.tensor3()

        self.W = self.init((self.input_dim, self.output_dim))
        self.U = self.inner_init((self.output_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))
        self.params = [self.W, self.U, self.b]

        if weights is not None:
            self.set_weights(weights)

    def _step(self, x_t, mask_tm1, h_tm1, u):
        '''
            Variable names follow the conventions from:
            http://deeplearning.net/software/theano/library/scan.html

        '''
        return self.activation(x_t + mask_tm1 * T.dot(h_tm1, u))

    def get_output(self, train=False):
        X = self.get_input(train)  # shape: (nb_samples, time (padded with zeros), input_dim)
        # new shape: (time, nb_samples, input_dim)
        # because theano.scan iterates over main dimension
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))
        x = T.dot(X, self.W) + self.b

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        outputs, updates = theano.scan(
            self._step,  # this will be called with arguments (sequences[i], outputs[i-1], non_sequences[i])
            sequences=[x, dict(input=padded_mask, taps=[-1])],  # tensors to iterate over, inputs to _step
            # initialization of the output. Input to _step with default tap=-1.
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=self.U,  # static inputs to _step
            truncate_gradient=self.truncate_gradient,
            go_backwards=self.go_backwards)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "init": self.init.__name__,
            "inner_init": self.inner_init.__name__,
            "activation": self.activation.__name__,
            "truncate_gradient": self.truncate_gradient,
            "return_sequences": self.return_sequences,
            "go_backwards": self.go_backwards
        }


class SimpleDeepRNN(Recurrent):
    '''
        Fully connected RNN where the output of multiple timesteps
        (up to "depth" steps in the past) is fed back to the input:

        output = activation( W.x_t + b + inner_activation(U_1.h_tm1) + inner_activation(U_2.h_tm2) + ... )

        This demonstrates how to build RNNs with arbitrary lookback.
        Also (probably) not a super useful model.
    '''
    def __init__(self, input_dim, output_dim, depth=3,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 go_backwards=False):

        super(SimpleDeepRNN, self).__init__()
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.depth = depth
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards

        self.input = T.tensor3()

        self.W = self.init((self.input_dim, self.output_dim))
        self.Us = [
            self.inner_init((self.output_dim, self.output_dim))
            for _ in range(self.depth)
        ]
        self.b = shared_zeros((self.output_dim))
        self.params = [self.W] + self.Us + [self.b]

        if weights is not None:
            self.set_weights(weights)

    def _step(self, x_t, *args):
        o = x_t
        for i in range(self.depth):
            mask_tmi = args[i]
            h_tmi = args[i + self.depth]
            U_tmi = args[i + 2 * self.depth]
            o += mask_tmi * self.inner_activation(T.dot(h_tmi, U_tmi))
        return self.activation(o)

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=self.depth)
        X = X.dimshuffle((1, 0, 2))

        x = T.dot(X, self.W) + self.b

        if self.depth == 1:
            initial = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
        else:
            initial = T.unbroadcast(
                T.unbroadcast(
                    alloc_zeros_matrix(self.depth, X.shape[1], self.output_dim), 0), 2)

        outputs, updates = theano.scan(
            self._step,
            sequences=[x, dict(
                input=padded_mask,
                taps=[(-i) for i in range(self.depth)]
            )],
            outputs_info=[dict(
                initial=initial,
                taps=[(-i - 1) for i in range(self.depth)]
            )],
            non_sequences=self.Us,
            truncate_gradient=self.truncate_gradient,
            go_backwards=self.go_backwards
        )

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "depth": self.depth,
            "init": self.init.__name__,
            "inner_init": self.inner_init.__name__,
            "activation": self.activation.__name__,
            "inner_activation": self.inner_activation.__name__,
            "truncate_gradient": self.truncate_gradient,
            "return_sequences": self.return_sequences,
            "go_backwards": self.go_backwards,
        }


class GRU(Recurrent):
    '''
        Gated Recurrent Unit - Cho et al. 2014

        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            On the Properties of Neural Machine Translation: Encoder–Decoder Approaches
                http://www.aclweb.org/anthology/W14-4012
            Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
                http://arxiv.org/pdf/1412.3555v1.pdf
    '''
    def __init__(self, input_dim, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 go_backwards=False):

        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.input = T.tensor3()

        self.W_z = self.init((self.input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((self.input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((self.input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        h_mask_tm1 = mask_tm1 * h_tm1
        z = self.inner_activation(xz_t + T.dot(h_mask_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = z * h_mask_tm1 + (1 - z) * hh_t
        return h_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U_z, self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient,
            go_backwards=self.go_backwards)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "init": self.init.__name__,
            "inner_init": self.inner_init.__name__,
            "activation": self.activation.__name__,
            "inner_activation": self.inner_activation.__name__,
            "truncate_gradient": self.truncate_gradient,
            "return_sequences": self.return_sequences,
            "go_backwards": self.go_backwards,
        }


class LSTM(Recurrent):
    '''
        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

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
    def __init__(self, input_dim, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 go_backwards=False):

        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.input = T.tensor3()

        self.W_i = self.init((self.input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = shared_zeros((self.output_dim))

        self.W_f = self.init((self.input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.output_dim))

        self.W_c = self.init((self.input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = shared_zeros((self.output_dim))

        self.W_o = self.init((self.input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              xi_t, xf_t, xo_t, xc_t, mask_tm1,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c):
        h_mask_tm1 = mask_tm1 * h_tm1
        c_mask_tm1 = mask_tm1 * c_tm1

        i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1, u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1, u_f))
        c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1, u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o

        [outputs, memories], updates = theano.scan(
            self._step,
            sequences=[xi, xf, xo, xc, padded_mask],
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
            ],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
            truncate_gradient=self.truncate_gradient,
            go_backwards=self.go_backwards)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "init": self.init.__name__,
            "inner_init": self.inner_init.__name__,
            "forget_bias_init": self.forget_bias_init.__name__,
            "activation": self.activation.__name__,
            "inner_activation": self.inner_activation.__name__,
            "truncate_gradient": self.truncate_gradient,
            "return_sequences": self.return_sequences,
            "go_backwards": self.go_backwards
        }


class JZS1(Recurrent):
    '''
        Evolved recurrent neural network architectures from the evaluation of thousands
        of models, serving as alternatives to LSTMs and GRUs. See Jozefowicz et al. 2015.

        This corresponds to the `MUT1` architecture described in the paper.

        Takes inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            An Empirical Exploration of Recurrent Network Architectures
                http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    '''
    def __init__(self, input_dim, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 go_backwards=False):

        super(JZS1, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.input = T.tensor3()

        self.W_z = self.init((self.input_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((self.input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        # P_h used to project X onto different dimension, using sparse random projections
        if self.input_dim == self.output_dim:
            self.Pmat = theano.shared(np.identity(self.output_dim, dtype=theano.config.floatX), name=None)
        else:
            P_shape = (self.input_dim, self.output_dim)
            P = np.random.binomial(1, 0.5, size=P_shape).astype(theano.config.floatX) * 2 - 1
            P = 1 / np.sqrt(self.input_dim) * P
            self.Pmat = theano.shared(P, name=None)

        self.params = [
            self.W_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.U_h, self.b_h,
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_r, u_h):
        h_mask_tm1 = mask_tm1 * h_tm1
        z = self.inner_activation(xz_t)
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = hh_t * z + h_mask_tm1 * (1 - z)
        return h_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.tanh(T.dot(X, self.Pmat)) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient,
            go_backwards=self.go_backwards)
        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "init": self.init.__name__,
            "inner_init": self.inner_init.__name__,
            "activation": self.activation.__name__,
            "inner_activation": self.inner_activation.__name__,
            "truncate_gradient": self.truncate_gradient,
            "return_sequences": self.return_sequences,
            "go_backwards": self.go_backwards
        }


class JZS2(Recurrent):
    '''
        Evolved recurrent neural network architectures from the evaluation of thousands
        of models, serving as alternatives to LSTMs and GRUs. See Jozefowicz et al. 2015.

        This corresponds to the `MUT2` architecture described in the paper.

        Takes inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            An Empirical Exploration of Recurrent Network Architectures
                http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    '''
    def __init__(self, input_dim, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 go_backwards=False):

        super(JZS2, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.input = T.tensor3()

        self.W_z = self.init((self.input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((self.input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        # P_h used to project X onto different dimension, using sparse random projections
        if self.input_dim == self.output_dim:
            self.Pmat = theano.shared(np.identity(self.output_dim, dtype=theano.config.floatX), name=None)
        else:
            P = np.random.binomial(1, 0.5, size=(self.input_dim, self.output_dim)).astype(theano.config.floatX) * 2 - 1
            P = 1 / np.sqrt(self.input_dim) * P
            self.Pmat = theano.shared(P, name=None)

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        h_mask_tm1 = mask_tm1 * h_tm1
        z = self.inner_activation(xz_t + T.dot(h_mask_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = hh_t * z + h_mask_tm1 * (1 - z)
        return h_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.Pmat) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U_z, self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient,
            go_backwards=self.go_backwards)
        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "init": self.init.__name__,
            "inner_init": self.inner_init.__name__,
            "activation": self.activation.__name__,
            "inner_activation": self.inner_activation.__name__,
            "truncate_gradient": self.truncate_gradient,
            "return_sequences": self.return_sequences,
            "go_backwards": self.go_backwards
        }


class JZS3(Recurrent):
    '''
        Evolved recurrent neural network architectures from the evaluation of thousands
        of models, serving as alternatives to LSTMs and GRUs. See Jozefowicz et al. 2015.

        This corresponds to the `MUT3` architecture described in the paper.

        Takes inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            An Empirical Exploration of Recurrent Network Architectures
                http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    '''
    def __init__(self, input_dim, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 go_backwards=False):

        super(JZS3, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.input = T.tensor3()

        self.W_z = self.init((self.input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((self.input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((self.input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        h_mask_tm1 = mask_tm1 * h_tm1
        z = self.inner_activation(xz_t + T.dot(T.tanh(h_mask_tm1), u_z))
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = hh_t * z + h_mask_tm1 * (1 - z)
        return h_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U_z, self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient,
            go_backwards=self.go_backwards,
        )
        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "init": self.init.__name__,
            "inner_init": self.inner_init.__name__,
            "activation": self.activation.__name__,
            "inner_activation": self.inner_activation.__name__,
            "truncate_gradient": self.truncate_gradient,
            "return_sequences": self.return_sequences,
            "go_backwards": self.go_backwards
        }


class Bidirectional(Recurrent):
    """
    Construct a bidirectional RNN out of an underlying RNN class. Traverses
    the input sequence both forwards and backwards and then concatenates RNN
    outputs from both directions.
    """

    def __init__(self, rnn_class, input_dim, output_dim=128, return_sequences=False, **kwargs):
        """
        All extra arguments are passed to the `rnn_class` constructor.
        """
        super(Bidirectional, self).__init__()
        self.rnn_class = rnn_class
        self.input_dim = input_dim
        if output_dim % 2 != 0:
            # since we're splitting the output dim between two underlying
            # RNNs, the total must be divisible by 2
            raise ValueError(
                "Output dimension of bidirectional RNN must be even")
        self.output_dim = output_dim
        self.return_sequences = return_sequences
        self.kwargs = kwargs

        self.forward_model = rnn_class(
            input_dim=self.input_dim,
            output_dim=self.output_dim // 2,
            return_sequences=self.return_sequences,
            go_backwards=False,
            **kwargs)
        self.backward_model = rnn_class(
            input_dim=self.input_dim,
            output_dim=self.output_dim // 2,
            return_sequences=self.return_sequences,
            go_backwards=True,
            **kwargs)

        # create input tensor in case this is the first layer in a network
        self.input = T.tensor3()

        self.layers = [self.forward_model, self.backward_model]
        self.params = []
        self.regularizers = []
        self.constraints = []

        for l in self.layers:
            params, regs, consts = l.get_params()
            self.regularizers += regs
            # params and constraints have the same size
            for p, c in zip(params, consts):
                if p not in self.params:
                    self.params.append(p)
                    self.constraints.append(c)

    def get_params(self):
        return self.params, self.regularizers, self.constraints

    def set_previous(self, layer):
        self.forward_model.set_previous(layer)
        self.backward_model.set_previous(layer)

    def get_output(self, train=False):
        forward_output = self.forward_model.get_output(train)
        backward_output = self.backward_model.get_output(train)
        if self.return_sequences:
            # reverse the output of the backward model along the
            # time dimension so that it's aligned with the forward model's
            # output
            backward_output = backward_output[:, ::-1]
        # both forward_output and backward_output have shapes like
        # (n_samples, n_timesteps, output_dim) in the case of self.return_sequences=True
        # or otherwise like (n_samples, output_dim)
        # In either case, concatenate the two outputs to get a final dimension
        # of output_dim * 2
        return T.concatenate([forward_output, backward_output], axis=-1)

    def get_output_mask(self, train=False):
        if self.return_sequences:
            return self.get_input_mask(train)
        else:
            return None

    def get_config(self):
        config_dict = {
            "name": self.__class__.__name__,
            "rnn_class": self.rnn_class.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }
        config_dict.update(self.kwargs)
        return config_dict
