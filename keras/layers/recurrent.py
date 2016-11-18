# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
np.set_printoptions(threshold=np.inf)
import types
import theano
import theano.tensor as T

from .. import backend as K
from .. import activations, initializations, regularizers
from ..engine import Layer, InputSpec

def time_distributed_dense(x, w, b=None, dropout=None,
                           input_dim=None, output_dim=None, timesteps=None):
    '''Apply y.w + b for every temporal slice y of x.
    '''
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b:
        x = x + b
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.pack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x


class Recurrent(Layer):
    '''Abstract base class for recurrent layers.
    Do not use in a model -- it's not a valid layer!
    Use its children classes `LSTM`, `GRU` and `SimpleRNN` instead.

    All recurrent layers (`LSTM`, `GRU`, `SimpleRNN`) also
    follow the specifications of this class and accept
    the keyword arguments listed below.

    # Example

    ```python
        # as the first layer in a Sequential model
        model = Sequential()
        model.add(LSTM(32, input_shape=(10, 64)))
        # now model.output_shape == (None, 32)
        # note: `None` is the batch dimension.

        # the following is identical:
        model = Sequential()
        model.add(LSTM(32, input_dim=64, input_length=10))

        # for subsequent layers, not need to specify the input size:
        model.add(LSTM(16))
    ```

    # Arguments
        weights: list of Numpy arrays to set as initial weights.
            The list should have 3 elements, of shapes:
            `[(input_dim, output_dim), (output_dim, output_dim), (output_dim,)]`.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False). If True, the network will be unrolled,
            else a symbolic loop will be used. When using TensorFlow, the network
            is always unrolled, so this argument does not do anything.
            Unrolling can speed-up a RNN, although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        consume_less: one of "cpu", "mem", or "gpu" (LSTM/GRU only).
            If set to "cpu", the RNN will use
            an implementation that uses fewer, larger matrix products,
            thus running faster on CPU but consuming more memory.
            If set to "mem", the RNN will use more matrix products,
            but smaller ones, thus running slower (may actually be faster on GPU)
            while consuming less memory.
            If set to "gpu" (LSTM/GRU only), the RNN will combine the input gate,
            the forget gate and the output gate into a single matrix,
            enabling more time-efficient parallelization on the GPU. Note: RNN
            dropout must be shared for all gates, resulting in a slightly
            reduced regularization.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        input_length: Length of input sequences, to be specified
            when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).
            Note that if the recurrent layer is not the first layer
            in your model, you would need to specify the input length
            at the level of the first layer
            (e.g. via the `input_shape` argument)

    # Input shape
        3D tensor with shape `(nb_samples, timesteps, input_dim)`.

    # Output shape
        - if `return_sequences`: 3D tensor with shape
            `(nb_samples, timesteps, output_dim)`.
        - else, 2D tensor with shape `(nb_samples, output_dim)`.

    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.

    # Note on performance
        You are likely to see better performance with RNNs in Theano compared
        to TensorFlow. Additionally, when using TensorFlow, it is often
        preferable to set `unroll=True` for better performance.

    # Note on using statefulness in RNNs
        You can set RNN layers to be 'stateful', which means that the states
        computed for the samples in one batch will be reused as initial states
        for the samples in the next batch.
        This assumes a one-to-one mapping between
        samples in different successive batches.

        To enable statefulness:
            - specify `stateful=True` in the layer constructor.
            - specify a fixed batch size for your model, by passing
                if sequential model:
                  a `batch_input_shape=(...)` to the first layer in your model.
                else for functional model with 1 or more Input layers:
                  a `batch_shape=(...)` to all the first layers in your model.
                This is the expected shape of your inputs *including the batch size*.
                It should be a tuple of integers, e.g. `(32, 10, 100)`.

        To reset the states of your model, call `.reset_states()` on either
        a specific layer, or on your entire model.
    '''
    def __init__(self, weights=None,
                 return_sequences=False, go_backwards=False, stateful=False,
                 unroll=False, consume_less='gpu',
                 input_dim=None, input_length=None, **kwargs):
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.consume_less = consume_less

        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(Recurrent, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.output_dim)
        else:
            return (input_shape[0], self.output_dim)

    def compute_mask(self, input, mask):
        if self.return_sequences:
            return mask
        else:
            return None

    def step(self, x, states):
        raise NotImplementedError

    def get_constants(self, x):
        return []

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.output_dim])  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def preprocess_input(self, x):
        return x

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)
        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def get_config(self):
        config = {'return_sequences': self.return_sequences,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful,
                  'unroll': self.unroll,
                  'consume_less': self.consume_less}
        if self.stateful:
            config['batch_input_shape'] = self.input_spec[0].shape
        else:
            config['input_dim'] = self.input_dim
            config['input_length'] = self.input_length

        base_config = super(Recurrent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SimpleRNN(Recurrent):
    '''Fully-connected RNN where the output is to be fed back to input.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(SimpleRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.W = self.init((input_dim, self.output_dim),
                           name='{}_W'.format(self.name))
        self.U = self.inner_init((self.output_dim, self.output_dim),
                                 name='{}_U'.format(self.name))
        self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.trainable_weights = [self.W, self.U, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]
            return time_distributed_dense(x, self.W, self.b, self.dropout_W,
                                          input_dim, self.output_dim,
                                          timesteps)
        else:
            return x

    def step(self, x, states):
        prev_output = states[0]
        B_U = states[1]
        B_W = states[2]

        if self.consume_less == 'cpu':
            h = x
        else:
            h = K.dot(x * B_W, self.W) + self.b

        output = self.activation(h + K.dot(prev_output * B_U, self.U))
        return output, [output]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = K.in_train_phase(K.dropout(ones, self.dropout_U), ones)
            constants.append(B_U)
        else:
            constants.append(K.cast_to_floatx(1.))
        if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, input_dim))
            B_W = K.in_train_phase(K.dropout(ones, self.dropout_W), ones)
            constants.append(B_W)
        else:
            constants.append(K.cast_to_floatx(1.))
        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U}
        base_config = super(SimpleRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GRU(Recurrent):
    '''Gated Recurrent Unit - Cho et al. 2014.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [On the Properties of Neural Machine Translation: Encoder–Decoder Approaches](http://www.aclweb.org/anthology/W14-4012)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/pdf/1412.3555v1.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(GRU, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

        if self.consume_less == 'gpu':

            self.W = self.init((self.input_dim, 3 * self.output_dim),
                               name='{}_W'.format(self.name))
            self.U = self.inner_init((self.output_dim, 3 * self.output_dim),
                                     name='{}_U'.format(self.name))

            self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                           np.zeros(self.output_dim),
                                           np.zeros(self.output_dim))),
                                name='{}_b'.format(self.name))

            self.trainable_weights = [self.W, self.U, self.b]
        else:

            self.W_z = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_z'.format(self.name))
            self.U_z = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_z'.format(self.name))
            self.b_z = K.zeros((self.output_dim,), name='{}_b_z'.format(self.name))

            self.W_r = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_r'.format(self.name))
            self.U_r = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_r'.format(self.name))
            self.b_r = K.zeros((self.output_dim,), name='{}_b_r'.format(self.name))

            self.W_h = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_h'.format(self.name))
            self.U_h = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_h'.format(self.name))
            self.b_h = K.zeros((self.output_dim,), name='{}_b_h'.format(self.name))

            self.trainable_weights = [self.W_z, self.U_z, self.b_z,
                                      self.W_r, self.U_r, self.b_r,
                                      self.W_h, self.U_h, self.b_h]

            self.W = K.concatenate([self.W_z, self.W_r, self.W_h])
            self.U = K.concatenate([self.U_z, self.U_r, self.U_h])
            self.b = K.concatenate([self.b_z, self.b_r, self.b_h])

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_z = time_distributed_dense(x, self.W_z, self.b_z, self.dropout_W,
                                         input_dim, self.output_dim, timesteps)
            x_r = time_distributed_dense(x, self.W_r, self.b_r, self.dropout_W,
                                         input_dim, self.output_dim, timesteps)
            x_h = time_distributed_dense(x, self.W_h, self.b_h, self.dropout_W,
                                         input_dim, self.output_dim, timesteps)
            return K.concatenate([x_z, x_r, x_h], axis=2)
        else:
            return x

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrices for recurrent units
        B_W = states[2]

        if self.consume_less == 'gpu':

            matrix_x = K.dot(x * B_W[0], self.W) + self.b
            matrix_inner = K.dot(h_tm1 * B_U[0], self.U[:, :2 * self.output_dim])

            x_z = matrix_x[:, :self.output_dim]
            x_r = matrix_x[:, self.output_dim: 2 * self.output_dim]
            inner_z = matrix_inner[:, :self.output_dim]
            inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]

            z = self.inner_activation(x_z + inner_z)
            r = self.inner_activation(x_r + inner_r)

            x_h = matrix_x[:, 2 * self.output_dim:]
            inner_h = K.dot(r * h_tm1 * B_U[0], self.U[:, 2 * self.output_dim:])
            hh = self.activation(x_h + inner_h)
        else:
            if self.consume_less == 'cpu':
                x_z = x[:, :self.output_dim]
                x_r = x[:, self.output_dim: 2 * self.output_dim]
                x_h = x[:, 2 * self.output_dim:]
            elif self.consume_less == 'mem':
                x_z = K.dot(x * B_W[0], self.W_z) + self.b_z
                x_r = K.dot(x * B_W[1], self.W_r) + self.b_r
                x_h = K.dot(x * B_W[2], self.W_h) + self.b_h
            else:
                raise Exception('Unknown `consume_less` mode.')
            z = self.inner_activation(x_z + K.dot(h_tm1 * B_U[0], self.U_z))
            r = self.inner_activation(x_r + K.dot(h_tm1 * B_U[1], self.U_r))

            hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(3)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, input_dim))
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(3)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])
        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U}
        base_config = super(GRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class GRUCond(GRU):
    '''Gated Recurrent Unit - Cho et al. 2014. with the previously generated word fed to the current timestep.
    You should give two inputs to this layer:
        1. The shifted sequence of words (shape: (mini_batch_size, output_timesteps, embedding_size))

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        return_states: boolean indicating if we want the intermediate states (hidden_state and memory) as additional outputs
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
        w_a_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        W_a_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_a_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_a_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_w_a: float between 0 and 1.
        dropout_W_a: float between 0 and 1.
        dropout_U_a: float between 0 and 1.


    # Formulation

        The resulting attention vector 'phi' at time 't' is formed by applying a weighted sum over
        the set of inputs 'x_i' contained in 'X':

            phi(X, t) = ∑_i alpha_i(t) * x_i,

        where each 'alpha_i' at time 't' is a weighting vector over all the input dimension that
        accomplishes the following condition:

            ∑_i alpha_i = 1

        and is dynamically adapted at each timestep w.r.t. the following formula:

            alpha_i(t) = exp{e_i(t)} /  ∑_j exp{e_j(t)}

        where each 'e_i' at time 't' is calculated as:

            e_i(t) = wa' * tanh( Wa * x_i  +  Ua * h(t-1)  +  ba ),

        where the following are learnable with the respectively named sizes:
                wa                Wa                     Ua                 ba
            [input_dim] [input_dim, input_dim] [output_dim, input_dim] [input_dim]

        The names of 'Ua' and 'Wa' are exchanged w.r.t. the provided reference as well as 'v' being renamed
        to 'x' for matching Keras LSTM's nomenclature.


    # References
        - [On the Properties of Neural Machine Translation: Encoder–Decoder Approaches](http://www.aclweb.org/anthology/W14-4012)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/pdf/1412.3555v1.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 return_states=False,
                 activation='tanh', inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, V_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., dropout_V=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.return_states = return_states
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.V_regularizer = regularizers.get(V_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U, self.dropout_V  = dropout_W, dropout_U, dropout_V

        if self.dropout_W or self.dropout_U or self.dropout_V:
            self.uses_learning_phase = True
        super(GRUCond, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2 or len(input_shape) == 3, 'You should pass two inputs to LSTMAttnCond ' \
                                                               '(previous_embedded_words and context) and ' \
                                                               'one optional input (init_memory)'

        if len(input_shape) == 2:
            self.input_spec = [InputSpec(shape=input_shape[0]), InputSpec(shape=input_shape[1])]
            self.num_inputs = 2
        elif len(input_shape) == 3:
            self.input_spec = [InputSpec(shape=input_shape[0]),
                               InputSpec(shape=input_shape[1]),
                               InputSpec(shape=input_shape[2])]
            self.num_inputs = 3
        self.input_dim = input_shape[0][2]
        self.context_dim = input_shape[1][1]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

        if self.consume_less == 'gpu':

            self.W = self.init((self.context_dim, 3 * self.output_dim),
                               name='{}_W'.format(self.name))
            self.U = self.inner_init((self.output_dim, 3 * self.output_dim),
                                     name='{}_U'.format(self.name))
            self.V = self.inner_init((self.input_dim, 3 * self.output_dim),
                                     name='{}_V'.format(self.name))

            self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                           np.zeros(self.output_dim),
                                           np.zeros(self.output_dim))),
                                name='{}_b'.format(self.name))

            self.trainable_weights =  [self.V, # Cond weights
                                       self.W, self.U, self.b]
        else:
            self.V_z = self.init((self.input_dim, self.output_dim),
                                 name='{}_V_z'.format(self.name))
            self.W_z = self.init((self.context_dim, self.output_dim),
                                 name='{}_W_z'.format(self.name))
            self.U_z = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_z'.format(self.name))
            self.b_z = K.zeros((self.output_dim,), name='{}_b_z'.format(self.name))

            self.V_r = self.init((self.input_dim, self.output_dim),
                                 name='{}_V_r'.format(self.name))
            self.W_r = self.init((self.context_dim, self.output_dim),
                                 name='{}_W_r'.format(self.name))
            self.U_r = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_r'.format(self.name))
            self.b_r = K.zeros((self.output_dim,), name='{}_b_r'.format(self.name))

            self.V_h = self.init((self.input_dim, self.output_dim),
                                 name='{}_V_h'.format(self.name))
            self.W_h = self.init((self.context_dim, self.output_dim),
                                 name='{}_W_h'.format(self.name))
            self.U_h = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_h'.format(self.name))
            self.b_h = K.zeros((self.output_dim,), name='{}_b_h'.format(self.name))

            self.trainable_weights = [self.V_z, self.W_z, self.U_z, self.b_z,
                                      self.V_r, self.W_r, self.U_r, self.b_r,
                                      self.V_h, self.W_h, self.U_h, self.b_h]

            self.V = K.concatenate([self.V_z, self.V_r, self.V_h])
            self.W = K.concatenate([self.W_z, self.W_r, self.W_h])
            self.U = K.concatenate([self.U_z, self.U_r, self.U_h])
            self.b = K.concatenate([self.b_z, self.b_r, self.b_h])

        self.regularizers = []
        # GRU regularizers
        if self.V_regularizer:
            self.V_regularizer.set_param(self.V)
            self.regularizers.append(self.V_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[2],
                        np.zeros((input_shape[0], input_shape[3])))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
               K.zeros((input_shape[0], self.output_dim)),
               K.zeros((input_shape[0], input_shape[3]))]


    def preprocess_input(self, x, B_V):
            return K.dot(x * B_V[0], self.V) + self.b


    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            main_out = (input_shape[0][0], input_shape[0][1], self.output_dim)
        else:
            main_out = (input_shape[0][0], self.output_dim)

        if self.return_states:
            states_dim = (input_shape[0][0], input_shape[0][1], self.output_dim)
            main_out = [main_out, states_dim]

        return main_out


    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.

        input_shape = self.input_spec[0].shape
        state_below = x[0]
        self.context = x[1]
        if self.num_inputs == 2: # input: [state_below, context]
            self.init_state = None
            self.init_memory = None
        elif self.num_inputs == 3: # input: [state_below, context, init_hidden_state]
            self.init_state = x[2]
            self.init_memory = None

        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(state_below)

        constants, B_V = self.get_constants(state_below)
        preprocessed_input = self.preprocess_input(state_below, B_V)
        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask[0],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=state_below.shape[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            ret = outputs
        else:
            ret = last_output

        # intermediate states as additional outputs
        if self.return_states:
            ret = [ret, states[0]]

        return ret

    def compute_mask(self, input, mask):
        if self.return_sequences:
            ret = mask[0]
        else:
            ret = None
        if self.return_states:
            ret = [ret, None]
        return ret

    def step(self, x, states):
        h_tm1 = states[0]  # previous hidden state

        # dropout matrices for recurrent units
        B_U = states[1]     # Dropout U
        B_W = states[2]     # Dropout W

        # Context (input sequence)
        context = states[3]     # Context

        if self.consume_less == 'gpu':
            matrix_x = x + K.dot(context * B_W[0], self.W)
            matrix_inner = K.dot(h_tm1 * B_U[0], self.U[:, :2 * self.output_dim])

            x_z = matrix_x[:, :self.output_dim]
            x_r = matrix_x[:, self.output_dim: 2 * self.output_dim]
            inner_z = matrix_inner[:, :self.output_dim]
            inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]

            z = self.inner_activation(x_z + inner_z)
            r = self.inner_activation(x_r + inner_r)

            x_h = matrix_x[:, 2 * self.output_dim:]
            inner_h = K.dot(r * h_tm1 * B_U[0], self.U[:, 2 * self.output_dim:])
            hh = self.activation(x_h + inner_h)
        h = z * h_tm1 + (1 - z) * hh

        return h, [h]

    def get_constants(self, x):
        constants = []
        # States[1]
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(3)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        # States[2]
        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0][0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(3)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0 < self.dropout_V < 1:
            input_dim = self.input_dim
            ones = K.ones_like(K.reshape(x[:, :, 0], (-1, x.shape[1], 1))) # (bs, timesteps, 1)
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_V = [K.in_train_phase(K.dropout(ones, self.dropout_V), ones) for _ in range(3)]
        else:
            B_V = [K.cast_to_floatx(1.) for _ in range(3)]

        # States[3]
        constants.append(self.context)

        return constants, B_V

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        if self.init_state is None:
        # build an all-zero tensor of shape (samples, output_dim)
            initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
            initial_state = K.expand_dims(initial_state)  # (samples, 1)
            initial_state = K.tile(initial_state, [1, self.output_dim]) # (samples, output_dim)
        else:
            initial_state = self.init_state
        initial_states = [initial_state]

        return initial_states

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'V_regularizer': self.V_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U,
                  'dropout_V': self.dropout_V,
                  }
        base_config = super(GRUCond, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class AttGRUCond(GRU):
    '''Gated Recurrent Unit - Cho et al. 2014. with Attention + the previously generated word fed to the current timestep.
    You should give two inputs to this layer:
        1. The shifted sequence of words (shape: (mini_batch_size, output_timesteps, embedding_size))
        2. The complete input sequence (shape: (mini_batch_size, input_timesteps, input_dim))

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        return_extra_variables: indicates if we only need the LSTM hidden state (False) or we want
            additional internal variables as outputs (True). The additional variables provided are:
            - x_att (None, out_timesteps, dim_encoder): feature vector computed after the Att.Model at each timestep
            - alphas (None, out_timesteps, in_timesteps): weights computed by the Att.Model at each timestep
        return_states: boolean indicating if we want the intermediate states (hidden_state and memory) as additional outputs
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
        w_a_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        W_a_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_a_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_a_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_w_a: float between 0 and 1.
        dropout_W_a: float between 0 and 1.
        dropout_U_a: float between 0 and 1.


    # Formulation

        The resulting attention vector 'phi' at time 't' is formed by applying a weighted sum over
        the set of inputs 'x_i' contained in 'X':

            phi(X, t) = ∑_i alpha_i(t) * x_i,

        where each 'alpha_i' at time 't' is a weighting vector over all the input dimension that
        accomplishes the following condition:

            ∑_i alpha_i = 1

        and is dynamically adapted at each timestep w.r.t. the following formula:

            alpha_i(t) = exp{e_i(t)} /  ∑_j exp{e_j(t)}

        where each 'e_i' at time 't' is calculated as:

            e_i(t) = wa' * tanh( Wa * x_i  +  Ua * h(t-1)  +  ba ),

        where the following are learnable with the respectively named sizes:
                wa                Wa                     Ua                 ba
            [input_dim] [input_dim, input_dim] [output_dim, input_dim] [input_dim]

        The names of 'Ua' and 'Wa' are exchanged w.r.t. the provided reference as well as 'v' being renamed
        to 'x' for matching Keras LSTM's nomenclature.


    # References
        - [On the Properties of Neural Machine Translation: Encoder–Decoder Approaches](http://www.aclweb.org/anthology/W14-4012)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/pdf/1412.3555v1.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    def __init__(self, output_dim, return_extra_variables=False, return_states=False,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, V_regularizer=None, b_regularizer=None,
                 wa_regularizer=None, Wa_regularizer=None, Ua_regularizer=None, ba_regularizer=None,
                 dropout_W=0., dropout_U=0., dropout_V=0., dropout_wa=0., dropout_Wa=0., dropout_Ua=0., **kwargs):
        self.output_dim = output_dim
        self.return_extra_variables = return_extra_variables
        self.return_states = return_states
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.V_regularizer = regularizers.get(V_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        # attention model learnable params
        self.wa_regularizer = regularizers.get(wa_regularizer)
        self.Wa_regularizer = regularizers.get(Wa_regularizer)
        self.Ua_regularizer = regularizers.get(Ua_regularizer)
        self.ba_regularizer = regularizers.get(ba_regularizer)

        self.dropout_W, self.dropout_U, self.dropout_V  = dropout_W, dropout_U, dropout_V
        self.dropout_wa, self.dropout_Wa, self.dropout_Ua = dropout_wa, dropout_Wa, dropout_Ua

        if self.dropout_W or self.dropout_U or  self.dropout_V or self.dropout_wa or self.dropout_Wa or self.dropout_Ua:
            self.uses_learning_phase = True
        super(AttGRUCond, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2 or len(input_shape) == 3, 'You should pass two inputs to LSTMAttnCond ' \
                                                               '(previous_embedded_words and context) and ' \
                                                               'one optional input (init_memory)'

        if len(input_shape) == 2:
            self.input_spec = [InputSpec(shape=input_shape[0]), InputSpec(shape=input_shape[1])]
            self.num_inputs = 2
        elif len(input_shape) == 3:
            self.input_spec = [InputSpec(shape=input_shape[0]),
                               InputSpec(shape=input_shape[1]),
                               InputSpec(shape=input_shape[2])]
            self.num_inputs = 3
        self.input_dim = input_shape[0][2]
        self.context_steps = input_shape[1][1] #if input_shape[0][1] is not None else self.max_ctx_len
        self.context_dim = input_shape[1][2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

        # Initialize Att model params (following the same format for any option of self.consume_less)
        self.wa = self.init((self.context_dim,),
                            name='{}_wa'.format(self.name))

        self.Ua = self.init((self.context_dim, self.context_dim),
                            name='{}_Ua'.format(self.name))

        self.Wa = self.init((self.output_dim, self.context_dim),
                                  name='{}_Wa'.format(self.name))

        self.ba = K.variable((np.zeros(self.context_dim)),
                             name='{}_ba'.format(self.name))
        self.ca = K.variable((np.zeros(self.context_steps)), name='{}_ca'.format(self.name))


        if self.consume_less == 'gpu':

            self.W = self.init((self.context_dim, 3 * self.output_dim),
                               name='{}_W'.format(self.name))
            self.U = self.inner_init((self.output_dim, 3 * self.output_dim),
                                     name='{}_U'.format(self.name))
            self.V = self.inner_init((self.input_dim, 3 * self.output_dim),
                                     name='{}_V'.format(self.name))

            self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                           np.zeros(self.output_dim),
                                           np.zeros(self.output_dim))),
                                name='{}_b'.format(self.name))

            self.trainable_weights =  [self.wa, self.Wa, self.Ua, self.ba, self.ca, # AttModel parameters
                                       self.V, # Cond weights
                                       self.W, self.U, self.b]
        else:
            self.V_z = self.init((self.input_dim, self.output_dim),
                                 name='{}_V_z'.format(self.name))
            self.W_z = self.init((self.context_dim, self.output_dim),
                                 name='{}_W_z'.format(self.name))
            self.U_z = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_z'.format(self.name))
            self.b_z = K.zeros((self.output_dim,), name='{}_b_z'.format(self.name))

            self.V_r = self.init((self.input_dim, self.output_dim),
                                 name='{}_V_r'.format(self.name))
            self.W_r = self.init((self.context_dim, self.output_dim),
                                 name='{}_W_r'.format(self.name))
            self.U_r = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_r'.format(self.name))
            self.b_r = K.zeros((self.output_dim,), name='{}_b_r'.format(self.name))

            self.V_h = self.init((self.input_dim, self.output_dim),
                                 name='{}_V_h'.format(self.name))
            self.W_h = self.init((self.context_dim, self.output_dim),
                                 name='{}_W_h'.format(self.name))
            self.U_h = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_h'.format(self.name))
            self.b_h = K.zeros((self.output_dim,), name='{}_b_h'.format(self.name))

            self.trainable_weights = [self.wa, self.Wa, self.Ua, self.ba, self.ca,
                                      self.V_z, self.W_z, self.U_z, self.b_z,
                                      self.V_r, self.W_r, self.U_r, self.b_r,
                                      self.V_h, self.W_h, self.U_h, self.b_h]

            self.V = K.concatenate([self.V_z, self.V_r, self.V_h])
            self.W = K.concatenate([self.W_z, self.W_r, self.W_h])
            self.U = K.concatenate([self.U_z, self.U_r, self.U_h])
            self.b = K.concatenate([self.b_z, self.b_r, self.b_h])

        self.regularizers = []
        # Att regularizers
        if self.wa_regularizer:
            self.wa_regularizer.set_param(self.wa)
            self.regularizers.append(self.wa_regularizer)
        if self.Wa_regularizer:
            self.Wa_regularizer.set_param(self.Wa)
            self.regularizers.append(self.Wa_regularizer)
        if self.Ua_regularizer:
            self.Ua_regularizer.set_param(self.Ua)
            self.regularizers.append(self.Ua_regularizer)
        if self.ba_regularizer:
            self.ba_regularizer.set_param(self.ba)
            self.regularizers.append(self.ba_regularizer)
        # GRU regularizers
        if self.V_regularizer:
            self.V_regularizer.set_param(self.V)
            self.regularizers.append(self.V_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[2],
                        np.zeros((input_shape[0], input_shape[3])))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
               K.zeros((input_shape[0], self.output_dim)),
               K.zeros((input_shape[0], input_shape[3]))]


    def preprocess_input(self, x, B_V):
            return K.dot(x * B_V[0], self.V) + self.b


    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            main_out = (input_shape[0][0], input_shape[0][1], self.output_dim)
        else:
            main_out = (input_shape[0][0], self.output_dim)

        if self.return_extra_variables:
            dim_x_att = (input_shape[0][0], input_shape[0][1], self.context_dim)
            dim_alpha_att = (input_shape[0][0], input_shape[0][1], input_shape[1][1])
            main_out = [main_out, dim_x_att, dim_alpha_att]

        if self.return_states:
            if not isinstance(main_out, list):
                main_out = [main_out]
            states_dim = (input_shape[0][0], input_shape[0][1], self.output_dim)
            main_out += [states_dim]

        return main_out


    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.

        input_shape = self.input_spec[0].shape
        state_below = x[0]
        self.context = x[1]
        if self.num_inputs == 2: # input: [state_below, context]
            self.init_state = None
            self.init_memory = None
        elif self.num_inputs == 3: # input: [state_below, context, init_hidden_state]
            self.init_state = x[2]
            self.init_memory = None
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(state_below)

        constants, B_V = self.get_constants(state_below, mask[1])
        preprocessed_input = self.preprocess_input(state_below, B_V)
        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask[0],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=state_below.shape[1],
                                             pos_extra_outputs_states=[1, 2])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            ret = outputs
        else:
            ret = last_output

        if self.return_extra_variables:
            ret = [ret, states[1], states[2]]

        # intermediate states as additional outputs
        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [states[0]]

        return ret

    def compute_mask(self, input, mask):
        if self.return_extra_variables:
            ret = [mask[0], mask[0], mask[0]]
        else:
            ret = mask[0]

        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [mask[0]]

        return ret

    def get_alignments(self, pctx_, h_tm1):
        # AttModel (see Formulation in class header)
        p_state_ = K.dot(h_tm1, self.Wa)
        pctx_ = K.tanh(pctx_ +  p_state_[:, None, :])
        e = K.dot(pctx_, self.wa) + self.ca
        alphas_shape = e.shape
        return K.softmax(e.reshape([alphas_shape[0], alphas_shape[1]]))

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        non_used_x_att = states[1]  # Placeholder for returning extra variables
        non_used_alphas_att = states[2]  # Placeholder for returning extra variables
        # dropout matrices for recurrent units
        B_U = states[3]     # Dropout U
        B_W = states[4]     # Dropout W
        # Att model dropouts
        B_wa = states[5]
        B_Wa = states[6]

        # Context (input sequence)
        pctx_ = states[7]       # Projected context (i.e. context * Ua + ba)
        context = states[8]     # Original context
        mask_input = states[9]  # Context mask
        if mask_input.ndim > 1: # Mask the context (only if necessary)
            pctx_ = mask_input[:, :, None] * pctx_
            context = mask_input[:, :, None] * context    # Masked context

        # AttModel (see Formulation in class header)
        p_state_ = K.dot(h_tm1 * B_Wa[0], self.Wa)
        pctx_ = K.tanh(pctx_ +  p_state_[:, None, :])
        e = K.dot(pctx_ * B_wa[0], self.wa) + self.ca
        alphas_shape = e.shape
        alphas = K.softmax(e.reshape([alphas_shape[0], alphas_shape[1]]))
        ctx_ = (context * alphas[:, :, None]).sum(axis=1) # sum over the in_timesteps dimension resulting in [batch_size, input_dim]

        if self.consume_less == 'gpu':
            matrix_x = x + K.dot(ctx_ * B_W[0], self.W)
            matrix_inner = K.dot(h_tm1 * B_U[0], self.U[:, :2 * self.output_dim])

            x_z = matrix_x[:, :self.output_dim]
            x_r = matrix_x[:, self.output_dim: 2 * self.output_dim]
            inner_z = matrix_inner[:, :self.output_dim]
            inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]

            z = self.inner_activation(x_z + inner_z)
            r = self.inner_activation(x_r + inner_r)

            x_h = matrix_x[:, 2 * self.output_dim:]
            inner_h = K.dot(r * h_tm1 * B_U[0], self.U[:, 2 * self.output_dim:])
            hh = self.activation(x_h + inner_h)
        h = z * h_tm1 + (1 - z) * hh

        return h, [h, ctx_, alphas]

    def get_constants(self, x, mask_input):
        constants = []
        # States[3]
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(3)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        # States[4]
        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0][0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(3)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0 < self.dropout_V < 1:
            input_dim = self.input_dim
            ones = K.ones_like(K.reshape(x[:, :, 0], (-1, x.shape[1], 1))) # (bs, timesteps, 1)
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_V = [K.in_train_phase(K.dropout(ones, self.dropout_V), ones) for _ in range(3)]
        else:
            B_V = [K.cast_to_floatx(1.) for _ in range(3)]

        # AttModel
        # States[5]
        if 0 < self.dropout_wa < 1:
            ones = K.ones_like(K.reshape(self.context[:, :, 0], (-1, self.context.shape[1], 1)))
            #ones = K.concatenate([ones], 1)
            B_wa = [K.in_train_phase(K.dropout(ones, self.dropout_wa), ones)]
            constants.append(B_wa)
        else:
            constants.append([K.cast_to_floatx(1.)])

        # States[6]
        if 0 < self.dropout_Wa < 1:
            input_dim = self.output_dim
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_Wa = [K.in_train_phase(K.dropout(ones, self.dropout_Wa), ones)]
            constants.append(B_Wa)
        else:
            constants.append([K.cast_to_floatx(1.)])

        if 0 < self.dropout_Ua < 1:
            input_dim = self.context_dim
            ones = K.ones_like(K.reshape(self.context[:, :, 0], (-1, self.context.shape[1], 1)))
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_Ua = [K.in_train_phase(K.dropout(ones, self.dropout_Ua), ones)]
            pctx = K.dot(self.context * B_Ua[0], self.Ua) + self.ba
        else:
            pctx = K.dot(self.context, self.Ua) + self.ba

        # States[7]
        constants.append(pctx)

        # States[8]
        constants.append(self.context)

        # States[9]
        if mask_input is None:
            mask_input = K.variable([])
        constants.append(mask_input)

        return constants, B_V


    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        if self.init_state is None:
            # build an all-zero tensor of shape (samples, output_dim)
            initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
            initial_state = K.expand_dims(initial_state)  # (samples, 1)
            initial_state = K.tile(initial_state, [1, self.output_dim]) # (samples, output_dim)
        else:
            initial_state = self.init_state

        initial_states = [initial_state]

        initial_state = K.zeros_like(self.context)            # (samples, intput_timesteps, ctx_dim)
        initial_state_alphas = K.sum(initial_state, axis=2)   # (samples, input_timesteps)
        initial_state = K.sum(initial_state, axis=1)          # (samples, ctx_dim)
        extra_states = [initial_state, initial_state_alphas]  # (samples, ctx_dim)

        return initial_states + extra_states


    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'return_extra_variables': self.return_extra_variables,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'V_regularizer': self.V_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'wa_regularizer': self.wa_regularizer.get_config() if self.wa_regularizer else None,
                  'Wa_regularizer': self.Wa_regularizer.get_config() if self.Wa_regularizer else None,
                  'Ua_regularizer': self.Ua_regularizer.get_config() if self.Ua_regularizer else None,
                  'ba_regularizer': self.ba_regularizer.get_config() if self.ba_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U,
                  'dropout_V': self.dropout_V,
                  'dropout_wa': self.dropout_wa,
                  'dropout_Wa': self.dropout_Wa,
                  'dropout_Ua': self.dropout_Ua,
                  }
        base_config = super(AttGRUCond, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class LSTM(Recurrent):
    '''Long-Short Term Memory unit - Hochreiter 1997.

    For a step-by-step description of the algorithm, see
    [this tutorial](http://deeplearning.net/tutorial/lstm.html).

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',  init_state=None, init_memory=None,
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.init_state = init_state
        self.init_memory = init_memory
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(LSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None]

        if self.consume_less == 'gpu':
            self.W = self.init((self.input_dim, 4 * self.output_dim),
                               name='{}_W'.format(self.name))
            self.U = self.inner_init((self.output_dim, 4 * self.output_dim),
                                     name='{}_U'.format(self.name))

            self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                           K.get_value(self.forget_bias_init((self.output_dim,))),
                                           np.zeros(self.output_dim),
                                           np.zeros(self.output_dim))),
                                name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.U, self.b]
        else:
            self.W_i = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_i'.format(self.name))
            self.U_i = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_i'.format(self.name))
            self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

            self.W_f = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_f'.format(self.name))
            self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_f'.format(self.name))
            self.b_f = self.forget_bias_init((self.output_dim,),
                                             name='{}_b_f'.format(self.name))

            self.W_c = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_c'.format(self.name))
            self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_c'.format(self.name))
            self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))

            self.W_o = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_o'.format(self.name))
            self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_o'.format(self.name))
            self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))

            self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                      self.W_c, self.U_c, self.b_c,
                                      self.W_f, self.U_f, self.b_f,
                                      self.W_o, self.U_o, self.b_o]

            self.W = K.concatenate([self.W_i, self.W_f, self.W_c, self.W_o])
            self.U = K.concatenate([self.U_i, self.U_f, self.U_c, self.U_o])
            self.b = K.concatenate([self.b_i, self.b_f, self.b_c, self.b_o])

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights




    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            if 0 < self.dropout_W < 1:
                dropout = self.dropout_W
            else:
                dropout = 0
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_i = time_distributed_dense(x, self.W_i, self.b_i, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_f = time_distributed_dense(x, self.W_f, self.b_f, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_c = time_distributed_dense(x, self.W_c, self.b_c, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_o = time_distributed_dense(x, self.W_o, self.b_o, dropout,
                                         input_dim, self.output_dim, timesteps)
            return K.concatenate([x_i, x_f, x_c, x_o], axis=2)
        else:
            return x

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        if self.init_state is None:
            # build an all-zero tensor of shape (samples, output_dim)
            initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
            initial_state = K.expand_dims(initial_state)  # (samples, 1)
            initial_state = K.tile(initial_state, [1, self.output_dim]) # (samples, output_dim)
            if self.init_memory is None:
                initial_states = [initial_state for _ in range(len(self.states))]
                return initial_states
            else:
                if len(self.states) == 2: # We have state and memory
                    initial_memory = self.init_memory
                    reducer = K.ones((self.output_dim, self.output_dim))
                    initial_memory = K.dot(initial_memory, reducer)  # (samples, output_dim)
                    initial_states = [initial_state, initial_memory]
                    return initial_states
                else: # We have more states (Why?)
                    initial_states = [initial_state for _ in range(len(self.states))]
                    return initial_states
        else:
            initial_state = self.init_state
            reducer = K.ones((self.output_dim, self.output_dim))
            initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
            if len(self.states) == 2 and self.init_memory is not None: # We have state and memory
                initial_memory = self.init_memory
                reducer = K.ones((self.output_dim, self.output_dim))
                initial_memory = K.dot(initial_memory, reducer)  # (samples, output_dim)
                initial_states = [initial_state, initial_memory]
            else:
                initial_states = [initial_state for _ in range(len(self.states))]
            return initial_states


    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]

        if self.consume_less == 'gpu':
            z = K.dot(x * B_W[0], self.W) + K.dot(h_tm1 * B_U[0], self.U) + self.b

            z0 = z[:, :self.output_dim]
            z1 = z[:, self.output_dim: 2 * self.output_dim]
            z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
            z3 = z[:, 3 * self.output_dim:]

            i = self.inner_activation(z0)
            f = self.inner_activation(z1)
            c = f * c_tm1 + i * self.activation(z2)
            o = self.inner_activation(z3)
        else:
            if self.consume_less == 'cpu':
                x_i = x[:, :self.output_dim]
                x_f = x[:, self.output_dim: 2 * self.output_dim]
                x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
                x_o = x[:, 3 * self.output_dim:]
            elif self.consume_less == 'mem':
                x_i = K.dot(x * B_W[0], self.W_i) + self.b_i
                x_f = K.dot(x * B_W[1], self.W_f) + self.b_f
                x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
                x_o = K.dot(x * B_W[3], self.W_o) + self.b_o
            else:
                raise Exception('Unknown `consume_less` mode.')

            i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
            f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
            c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
            o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))

        h = o * self.activation(c)
        return h, [h, c]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, input_dim))
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(4)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'forget_bias_init': self.forget_bias_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U}
        base_config = super(LSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LSTMCond(LSTM):
    '''Conditional LSTM: The previously generated word is fed to the current timestep

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        return_states: boolean indicating if we want the intermediate states (hidden_state and memory) as additional outputs
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 return_states=False,
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid', consume_less='gpu',
                 W_regularizer=None, V_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., dropout_V=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.return_states = return_states
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.consume_less = consume_less
        self.W_regularizer = W_regularizer
        self.V_regularizer = V_regularizer
        self.U_regularizer = U_regularizer
        self.b_regularizer = b_regularizer

        self.dropout_W, self.dropout_U, self.dropout_V = dropout_W, dropout_U, dropout_V
        if self.dropout_W or self.dropout_U or self.dropout_V:
            self.uses_learning_phase = True

        super(LSTMCond, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2 or len(input_shape) == 4, 'You should pass two inputs to LSTMCond ' \
                                                               '(context and previous_embedded_words) and ' \
                                                               'two optional inputs (init_state and init_memory)'

        if len(input_shape) == 2:
            self.input_spec = [InputSpec(shape=input_shape[0]), InputSpec(shape=input_shape[1])]
            self.num_inputs = 2
        elif len(input_shape) == 4:
            self.input_spec = [InputSpec(shape=input_shape[0]), InputSpec(shape=input_shape[1]),
                               InputSpec(shape=input_shape[2]), InputSpec(shape=input_shape[3])]
            self.num_inputs = 4
        self.input_dim = input_shape[0][2]
        if self.input_spec[1].ndim == 3:
            self.context_dim = input_shape[1][2]
            self.static_ctx = False
            assert input_shape[1][1] == input_shape[0][1], 'When using a 3D ctx in LSTMCond, it has to have the same ' \
                                                          'number of timesteps (dimension 1) as the input. Currently,' \
                                                          'the number of input timesteps is: ' \
                                                           + str(input_shape[0][1]) + \
                                                          ', while the number of ctx timesteps is ' \
                                                           + str(input_shape[1][1]) + ' (complete shapes: '\
                                                           + str(input_shape[0]) + ', ' + str(input_shape[1]) + ')'
        else:
            self.context_dim = input_shape[1][1]
            self.static_ctx = True

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None] # [h, c]

        if self.consume_less == 'gpu':
            self.W = self.init((self.context_dim, 4 * self.output_dim),
                               name='{}_W'.format(self.name))
            self.U = self.inner_init((self.output_dim, 4 * self.output_dim),
                                     name='{}_U'.format(self.name))
            self.V = self.init((self.input_dim, 4 * self.output_dim),
                                     name='{}_V'.format(self.name))
            self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                           K.get_value(self.forget_bias_init(self.output_dim)),
                                           np.zeros(self.output_dim),
                                           np.zeros(self.output_dim))),
                                name='{}_b'.format(self.name))
            self.trainable_weights = [self.W,
                                      self.U,
                                      self.V,
                                      self.b]

        else:
            self.V_i = self.init((self.input_dim, self.output_dim), name='{}_V_i'.format(self.name))
            self.W_i = self.init((self.context_dim, self.output_dim), name='{}_W_i'.format(self.name))
            self.U_i = self.inner_init((self.output_dim, self.output_dim), name='{}_U_i'.format(self.name))
            self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

            self.V_f = self.init((self.input_dim, self.output_dim), name='{}_V_f'.format(self.name))
            self.W_f = self.init((self.context_dim, self.output_dim), name='{}_W_f'.format(self.name))
            self.U_f = self.inner_init((self.output_dim, self.output_dim),name='{}_U_f'.format(self.name))
            self.b_f = self.forget_bias_init((self.output_dim,), name='{}_b_f'.format(self.name))


            self.V_c = self.init((self.input_dim, self.output_dim),name='{}_V_c'.format(self.name))
            self.W_c = self.init((self.context_dim, self.output_dim), name='{}_W_c'.format(self.name))
            self.U_c = self.inner_init((self.output_dim, self.output_dim),name='{}_U_c'.format(self.name))
            self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))

            self.V_o = self.init((self.input_dim, self.output_dim), name='{}_V_o'.format(self.name))
            self.W_o = self.init((self.context_dim, self.output_dim), name='{}_W_o'.format(self.name))
            self.U_o = self.inner_init((self.output_dim, self.output_dim),name='{}_U_o'.format(self.name))
            self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))

            self.V_x = self.init((self.output_dim, self.input_dim), name='{}_V_x'.format(self.name))
            self.W_x = self.init((self.output_dim, self.context_dim), name='{}_W_x'.format(self.name))
            self.b_x = K.zeros((self.context_dim,), name='{}_b_x'.format(self.name))
            self.trainable_weights = [self.V_i, self.W_i, self.U_i, self.b_i,
                                      self.V_c, self.W_c, self.U_c, self.b_c,
                                      self.V_f, self.W_f, self.U_f, self.b_f,
                                      self.V_o, self.W_o, self.U_o, self.b_o,
                                      self.V_x, self.W_x, self.b_x
                                      ]

            self.W = K.concatenate([self.W_i, self.W_f, self.W_c, self.W_o])
            self.U = K.concatenate([self.U_i, self.U_f, self.U_c, self.U_o])
            self.V = K.concatenate([self.V_i, self.V_f, self.V_c, self.V_o])
            self.b = K.concatenate([self.b_i, self.b_f, self.b_c, self.b_o])

        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.V_regularizer:
            self.V_regularizer.set_param(self.V)
            self.regularizers.append(self.V_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[2],
                        np.zeros((input_shape[0], input_shape[3])))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]#,

    def preprocess_input(self, x, context, dropouts):
        if self.static_ctx:
            return K.dot(x * dropouts[0][0], self.V)
        else:
            return K.dot(context * dropouts[0][0], self.W) + K.dot(x * dropouts[1][0], self.V)

    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            main_out = (input_shape[0][0], input_shape[0][1], self.output_dim)
        else:
            main_out = (input_shape[0][0], self.output_dim)

        if self.return_states:
            states_dim = (input_shape[0][0], input_shape[0][1], self.output_dim)
            main_out = [main_out, states_dim, states_dim]

        return main_out


    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.

        input_shape = self.input_spec[0].shape
        state_below = x[0]
        self.context = x[1]
        if self.num_inputs == 2: # input: [state_below, context]
            self.init_state = None
            self.init_memory = None
        elif self.num_inputs == 3: # input: [state_below, context, init_generic]
            self.init_state = x[2]
            self.init_memory = x[2]
        elif self.num_inputs == 4: # input: [state_below, context, init_state, init_memory]
            self.init_state = x[2]
            self.init_memory = x[3]
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(state_below)
        constants, dropouts = self.get_constants(state_below)
        preprocessed_input = self.preprocess_input(state_below, self.context, dropouts)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask[0],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=state_below.shape[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))
        if self.return_sequences:
            ret = outputs
        else:
            ret = last_output

        # intermediate states as additional outputs
        if self.return_states:
            ret = [ret, states[0], states[1]]

        return ret

    def compute_mask(self, input, mask):
        if self.return_sequences:
            ret = mask[0]
        else:
            ret = None
        if self.return_states:
            ret = [ret, None, None]
        return ret

    def step(self, x, states):

        h_tm1 = states[0]  # State
        c_tm1 = states[1]  # Memory

        B_U = states[2]    # Dropout U
        if self.static_ctx:
            B_W = states[3]    # Dropout W
            context = states[4]
            z = x + K.dot(context * B_W[0], self.W) + K.dot(h_tm1 * B_U[0], self.U) + self.b
        else:
            z = x + K.dot(h_tm1 * B_U[0], self.U) + self.b
        z0 = z[:, :self.output_dim]
        z1 = z[:, self.output_dim: 2 * self.output_dim]
        z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
        z3 = z[:, 3 * self.output_dim:]

        i = self.inner_activation(z0)
        f = self.inner_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.inner_activation(z3)
        h = o * self.activation(c)
        return h, [h, c]


    def get_constants(self, x):
        constants = []
        dropouts = []
        # States[2]
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # States[3]
        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[1][0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(4)]
            constants.append(B_W)
        else:
            B_W = [K.cast_to_floatx(1.) for _ in range(4)]
        if self.static_ctx:
            constants.append(B_W)
        else:
            dropouts.append(B_W)

        if 0 < self.dropout_V < 1:
            input_dim = self.input_dim
            ones = K.ones_like(K.reshape(x[:, :, 0], (-1, x.shape[1], 1))) # (bs, timesteps, 1)
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_V = [K.in_train_phase(K.dropout(ones, self.dropout_V), ones) for _ in range(4)]
        else:
            B_V = [K.cast_to_floatx(1.) for _ in range(4)]
        dropouts.append(B_V)
        # States[4]
        if self.static_ctx:
            constants.append(self.context)

        return constants, dropouts

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        if self.init_state is None:
            # build an all-zero tensor of shape (samples, output_dim)
            initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
            initial_state = K.expand_dims(initial_state)  # (samples, 1)
            initial_state = K.tile(initial_state, [1, self.output_dim])  # (samples, output_dim)
            if self.init_memory is None:
                initial_states = [initial_state for _ in range(2)]
            else:
                initial_memory = self.init_memory
                #reducer = K.ones((self.output_dim, self.output_dim))
                #initial_memory = K.dot(initial_memory, reducer)  # (samples, output_dim)
                initial_states = [initial_state, initial_memory]
        else:
            initial_state = self.init_state
            #reducer = K.ones((self.output_dim, self.output_dim))
            #initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
            if self.init_memory is not None: # We have state and memory
                initial_memory = self.init_memory
                #reducer = K.ones((self.output_dim, self.output_dim))
                #initial_memory = K.dot(initial_memory, reducer)  # (samples, output_dim)
                initial_states = [initial_state, initial_memory]
            else:
                initial_states = [initial_state for _ in range(2)]

        return initial_states

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "V_regularizer": self.V_regularizer.get_config() if self.U_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U,
                  "dropout_V": self.dropout_V}
        base_config = super(LSTMCond, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttLSTM(LSTM):
    '''Long-Short Term Memory unit with Attention.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        output_timesteps: number of output timesteps (# of output vectors generated)
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
        w_a_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        W_a_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_a_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_a_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_w_a: float between 0 and 1.
        dropout_W_a: float between 0 and 1.
        dropout_U_a: float between 0 and 1.

    # Formulation

        The resulting attention vector 'phi' at time 't' is formed by applying a weighted sum over
        the set of inputs 'x_i' contained in 'X':

            phi(X, t) = ∑_i alpha_i(t) * x_i,

        where each 'alpha_i' at time 't' is a weighting vector over all the input dimension that
        accomplishes the following condition:

            ∑_i alpha_i = 1

        and is dynamically adapted at each timestep w.r.t. the following formula:

            alpha_i(t) = exp{e_i(t)} /  ∑_j exp{e_j(t)}

        where each 'e_i' at time 't' is calculated as:

            e_i(t) = wa' * tanh( Wa * x_i  +  Ua * h(t-1)  +  ba ),

        where the following are learnable with the respectively named sizes:
                wa                Wa                     Ua                 ba
            [input_dim] [input_dim, input_dim] [output_dim, input_dim] [input_dim]

        The names of 'Ua' and 'Wa' are exchanged w.r.t. the provided reference as well as 'v' being renamed
        to 'x' for matching Keras LSTM's nomenclature.

    # References
        -   Yao L, Torabi A, Cho K, Ballas N, Pal C, Larochelle H, Courville A.
            Describing videos by exploiting temporal structure.
            InProceedings of the IEEE International Conference on Computer Vision 2015 (pp. 4507-4515).
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', init_state=None, init_memory=None,
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., dropout_wa=0., dropout_Wa=0., dropout_Ua=0.,
                 wa_regularizer=None, Wa_regularizer=None, Ua_regularizer=None, ba_regularizer=None,
                 **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.init_state = init_state
        self.init_memory = init_memory
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U
        # attention model learnable params
        self.wa_regularizer = regularizers.get(wa_regularizer)
        self.Wa_regularizer = regularizers.get(Wa_regularizer)
        self.Ua_regularizer = regularizers.get(Ua_regularizer)
        self.ba_regularizer = regularizers.get(ba_regularizer)
        self.dropout_wa, self.dropout_Wa, self.dropout_Ua = dropout_wa, dropout_Wa, dropout_Ua

        if self.dropout_W or self.dropout_U or self.dropout_wa or self.dropout_Wa or self.dropout_Ua:
            self.uses_learning_phase = True
        super(AttLSTM, self).__init__(output_dim, **kwargs)
        self.input_spec = [InputSpec(ndim=4)]


    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape, ndim=4)]
        self.input_dim = input_shape[-1]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None]

        # Initialize Att model params (following the same format for any option of self.consume_less)
        self.wa = self.init((self.input_dim,),
                               name='{}_wa'.format(self.name))

        self.Wa = self.init((self.input_dim, self.input_dim),
                               name='{}_Wa'.format(self.name))
        self.Ua = self.inner_init((self.output_dim, self.input_dim),
                                     name='{}_Ua'.format(self.name))

        self.ba = K.variable((np.zeros(self.input_dim)),
                                name='{}_ba'.format(self.name))

        if self.consume_less == 'gpu':
            self.W = self.init((self.input_dim, 4 * self.output_dim),
                               name='{}_W'.format(self.name))
            self.U = self.inner_init((self.output_dim, 4 * self.output_dim),
                                     name='{}_U'.format(self.name))

            self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                           K.get_value(self.forget_bias_init(self.output_dim)),
                                           np.zeros(self.output_dim),
                                           np.zeros(self.output_dim))),
                                name='{}_b'.format(self.name))

            self.trainable_weights = [self.wa, self.Wa, self.Ua, self.ba, # AttModel parameters
                                      self.W, self.U, self.b]
        else:
            self.W_i = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_i'.format(self.name))
            self.U_i = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_i'.format(self.name))
            self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

            self.W_f = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_f'.format(self.name))
            self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_f'.format(self.name))
            self.b_f = self.forget_bias_init((self.output_dim,),
                                             name='{}_b_f'.format(self.name))

            self.W_c = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_c'.format(self.name))
            self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_c'.format(self.name))
            self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))

            self.W_o = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_o'.format(self.name))
            self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_o'.format(self.name))
            self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))

            self.trainable_weights = [self.wa, self.Wa, self.Ua, self.ba, # AttModel parameters
                                      self.W_i, self.U_i, self.b_i,
                                      self.W_c, self.U_c, self.b_c,
                                      self.W_f, self.U_f, self.b_f,
                                      self.W_o, self.U_o, self.b_o]

            self.W = K.concatenate([self.W_i, self.W_f, self.W_c, self.W_o])
            self.U = K.concatenate([self.U_i, self.U_f, self.U_c, self.U_o])
            self.b = K.concatenate([self.b_i, self.b_f, self.b_c, self.b_o])

        self.regularizers = []
        # Att regularizers
        if self.wa_regularizer:
            self.wa_regularizer.set_param(self.wa)
            self.regularizers.append(self.wa_regularizer)
        if self.Wa_regularizer:
            self.Wa_regularizer.set_param(self.Wa)
            self.regularizers.append(self.Wa_regularizer)
        if self.Ua_regularizer:
            self.Ua_regularizer.set_param(self.Ua)
            self.regularizers.append(self.Ua_regularizer)
        if self.ba_regularizer:
            self.ba_regularizer.set_param(self.ba)
            self.regularizers.append(self.ba_regularizer)
        # LSTM regularizers
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def preprocess_input(self, x):
        return x

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=None,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def step(self, x, states):
        # After applying a RepeatMatrix before this AttLSTM the following way:
        #    x = RepeatMatrix(out_timesteps, dim=1)(x)
        #    x will have the following size:
        #        [batch_size, out_timesteps, in_timesteps, dim_encoder]
        #    which means that in step() our x will be:
        #        [batch_size, in_timesteps, dim_encoder]
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]
        # Att model dropouts
        B_wa = states[4]
        context = states[5] # pre-calculated Wa*x term (common for all output timesteps)
        B_Ua = states[6]

        # AttModel (see Formulation in class header)
        e = K.dot(K.tanh(context + K.dot(h_tm1[:, None, :] * B_Ua, self.Ua) + self.ba) * B_wa, self.wa)
        alpha = K.softmax(e)
        x_ = (x * alpha[:,:,None]).sum(axis=1) # sum over the in_timesteps dimension resulting in [batch_size, input_dim]

        # LSTM
        if self.consume_less == 'gpu':
            z = K.dot(x_ * B_W[0], self.W) + K.dot(h_tm1 * B_U[0], self.U) + self.b

            z0 = z[:, :self.output_dim]
            z1 = z[:, self.output_dim: 2 * self.output_dim]
            z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
            z3 = z[:, 3 * self.output_dim:]

            i = self.inner_activation(z0)
            f = self.inner_activation(z1)
            c = f * c_tm1 + i * self.activation(z2)
            o = self.inner_activation(z3)

        else:
            if self.consume_less == 'cpu':
                x_i = x_[:, :self.output_dim]
                x_f = x_[:, self.output_dim: 2 * self.output_dim]
                x_c = x_[:, 2 * self.output_dim: 3 * self.output_dim]
                x_o = x_[:, 3 * self.output_dim:]
            elif self.consume_less == 'mem':
                x_i = K.dot(x_ * B_W[0], self.W_i) + self.b_i
                x_f = K.dot(x_ * B_W[1], self.W_f) + self.b_f
                x_c = K.dot(x_ * B_W[2], self.W_c) + self.b_c
                x_o = K.dot(x_ * B_W[3], self.W_o) + self.b_o
            else:
                raise Exception('Unknown `consume_less` mode.')

            i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
            f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
            c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
            o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))

        h = o * self.activation(c)
        return h, [h, c]


    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(4)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # AttModel
        if 0 < self.dropout_wa < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, :, 0, 0], (-1, input_shape[1], 1)))
            ones = K.concatenate([ones] * input_dim, 2)
            B_wa = K.in_train_phase(K.dropout(ones, self.dropout_wa), ones)
            constants.append(B_wa)
        else:
            constants.append(K.cast_to_floatx(1.))

        if 0 < self.dropout_Wa < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, :, 0, 0], (-1, input_shape[1], 1)))
            ones = K.concatenate([ones] * input_dim, 2)
            B_Wa = K.in_train_phase(K.dropout(ones, self.dropout_Wa), ones)
            constants.append(K.dot(x[:, 0, :, :] * B_Wa, self.Wa))
        else:
            constants.append(K.dot(x[:, 0, :, :], self.Wa))

        if 0 < self.dropout_Ua < 1:
            input_shape = self.input_spec[0].shape
            ones = K.ones_like(K.reshape(x[:, :, 0, 0], (-1, input_shape[1], 1)))
            ones = K.concatenate([ones] * self.output_dim, 2)
            B_Ua = K.in_train_phase(K.dropout(ones, self.dropout_Ua), ones)
            constants.append(B_Ua)
        else:
            constants.append([K.cast_to_floatx(1.)])

        return constants


    """
    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        if self.init_state is None:
            initial_state = K.zeros_like(x)  # (samples, timesteps_trg, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=1)  # (samples, timesteps_trg, input_dim)
            initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
            reducer = K.ones((self.input_dim, self.output_dim))
            initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
            if self.init_memory is None:
                initial_states = [initial_state for _ in range(len(self.states))]
                return initial_states
            else:
                if len(self.states) == 2: # We have state and memory
                    initial_memory = self.init_memory
                    reducer = K.ones((self.output_dim, self.output_dim))
                    initial_memory = K.dot(initial_memory, reducer)  # (samples, output_dim)
                    initial_states = [initial_state, initial_memory]
                    return initial_states
                else: # We have more states (Why?)
                    initial_states = [initial_state for _ in range(len(self.states))]
                    return initial_states
        else:
            initial_state = self.init_state
            reducer = K.ones((self.output_dim, self.output_dim))
            initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
            if len(self.states) == 2 and self.init_memory is not None: # We have state and memory
                initial_memory = self.init_memory
                reducer = K.ones((self.output_dim, self.output_dim))
                initial_memory = K.dot(initial_memory, reducer)  # (samples, output_dim)
                initial_states = [initial_state, initial_memory]
            else:
                initial_states = [initial_state for _ in range(len(self.states))]
            return initial_states
        """


    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'forget_bias_init': self.forget_bias_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'wa_regularizer': self.wa_regularizer.get_config() if self.wa_regularizer else None,
                  'Wa_regularizer': self.Wa_regularizer.get_config() if self.Wa_regularizer else None,
                  'Ua_regularizer': self.Ua_regularizer.get_config() if self.Ua_regularizer else None,
                  'ba_regularizer': self.ba_regularizer.get_config() if self.ba_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U,
                  'dropout_wa': self.dropout_wa,
                  'dropout_Wa': self.dropout_Wa,
                  'dropout_Ua': self.dropout_Ua}
        base_config = super(AttLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttLSTMCond(LSTM):
    '''Long-Short Term Memory unit with Attention + the previously generated word fed to the current timestep.
    You should give two inputs to this layer:
        1. The shifted sequence of words (shape: (mini_batch_size, output_timesteps, embedding_size))
        2. The complete input sequence (shape: (mini_batch_size, input_timesteps, input_dim))
    # Arguments
        output_dim: dimension of the internal projections and the final output.
        embedding_size: dimension of the word embedding module used for the enconding of the generated words.
        return_extra_variables: indicates if we only need the LSTM hidden state (False) or we want 
            additional internal variables as outputs (True). The additional variables provided are:
            - x_att (None, out_timesteps, dim_encoder): feature vector computed after the Att.Model at each timestep
            - alphas (None, out_timesteps, in_timesteps): weights computed by the Att.Model at each timestep
        return_states: boolean indicating if we want the intermediate states (hidden_state and memory) as additional outputs
        output_timesteps: number of output timesteps (# of output vectors generated)
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
        w_a_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        W_a_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_a_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_a_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_w_a: float between 0 and 1.
        dropout_W_a: float between 0 and 1.
        dropout_U_a: float between 0 and 1.

    # Formulation

        The resulting attention vector 'phi' at time 't' is formed by applying a weighted sum over
        the set of inputs 'x_i' contained in 'X':

            phi(X, t) = ∑_i alpha_i(t) * x_i,

        where each 'alpha_i' at time 't' is a weighting vector over all the input dimension that
        accomplishes the following condition:

            ∑_i alpha_i = 1

        and is dynamically adapted at each timestep w.r.t. the following formula:

            alpha_i(t) = exp{e_i(t)} /  ∑_j exp{e_j(t)}

        where each 'e_i' at time 't' is calculated as:

            e_i(t) = wa' * tanh( Wa * x_i  +  Ua * h(t-1)  +  ba ),

        where the following are learnable with the respectively named sizes:
                wa                Wa                     Ua                 ba
            [input_dim] [input_dim, input_dim] [output_dim, input_dim] [input_dim]

        The names of 'Ua' and 'Wa' are exchanged w.r.t. the provided reference as well as 'v' being renamed
        to 'x' for matching Keras LSTM's nomenclature.

    # References
        -   Yao L, Torabi A, Cho K, Ballas N, Pal C, Larochelle H, Courville A.
            Describing videos by exploiting temporal structure.
            InProceedings of the IEEE International Conference on Computer Vision 2015 (pp. 4507-4515).
    '''
    def __init__(self, output_dim, return_extra_variables=False, return_states=False,
                 init='glorot_uniform', inner_init='orthogonal', #context_dim=None, input_dim=None,
                 forget_bias_init='one', activation='tanh', inner_activation='sigmoid',
                 W_regularizer=None, U_regularizer=None, V_regularizer=None, b_regularizer=None,
                 wa_regularizer=None, Wa_regularizer=None, Ua_regularizer=None, ba_regularizer=None,
                 dropout_W=0., dropout_U=0., dropout_V=0., dropout_wa=0., dropout_Wa=0., dropout_Ua=0.,
                 **kwargs):
        self.output_dim = output_dim
        self.return_extra_variables = return_extra_variables
        self.return_states = return_states
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.V_regularizer = regularizers.get(V_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        # attention model learnable params
        self.wa_regularizer = regularizers.get(wa_regularizer)
        self.Wa_regularizer = regularizers.get(Wa_regularizer)
        self.Ua_regularizer = regularizers.get(Ua_regularizer)
        self.ba_regularizer = regularizers.get(ba_regularizer)
        # Dropouts
        self.dropout_W, self.dropout_U, self.dropout_V = dropout_W, dropout_U, dropout_V
        self.dropout_wa, self.dropout_Wa, self.dropout_Ua = dropout_wa, dropout_Wa, dropout_Ua
        #self.context_dim = context_dim
        #self.input_dim = input_dim
        if self.dropout_W or self.dropout_U or self.dropout_wa or self.dropout_Wa or self.dropout_Ua:
            self.uses_learning_phase = True
        super(AttLSTMCond, self).__init__(output_dim, **kwargs)


    def build(self, input_shape):
        assert len(input_shape) == 2 or len(input_shape) == 4, 'You should pass two inputs to AttLSTMCond ' \
                                                               '(previous_embedded_words and context) ' \
                                                               'and two optional inputs (init_state and init_memory)'

        if len(input_shape) == 2:
            self.input_spec = [InputSpec(shape=input_shape[0]), InputSpec(shape=input_shape[1])]
            self.num_inputs = 2
        elif len(input_shape) == 4:
            self.input_spec = [InputSpec(shape=input_shape[0]), InputSpec(shape=input_shape[1]),
                               InputSpec(shape=input_shape[2]), InputSpec(shape=input_shape[3])]
            self.num_inputs = 4
        self.input_dim = input_shape[0][2]
        self.context_steps = input_shape[1][1] #if input_shape[0][1] is not None else self.max_ctx_len
        self.context_dim = input_shape[1][2]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensors of shape (output_dim)
            #self.states = [None, None, None, None] # [h, c, x_att, alpha_att]
            self.states = [None, None, None] # [h, c, x_att]

        # Initialize Att model params (following the same format for any option of self.consume_less)
        self.wa = self.init((self.context_dim,),
                            name='{}_wa'.format(self.name))

        self.Ua = self.init((self.context_dim, self.context_dim),
                            name='{}_Ua'.format(self.name))

        self.Wa = self.init((self.output_dim, self.context_dim),
                                  name='{}_Wa'.format(self.name))

        self.ba = K.variable((np.zeros(self.context_dim)),
                             name='{}_ba'.format(self.name))
        self.ca = K.variable((np.zeros(self.context_steps)), name='{}_ca'.format(self.name))

        if self.consume_less == 'gpu':
            self.W = self.init((self.context_dim, 4 * self.output_dim),
                               name='{}_W'.format(self.name))
            self.U = self.inner_init((self.output_dim, 4 * self.output_dim),
                                     name='{}_U'.format(self.name))
            self.V = self.init((self.input_dim, 4 * self.output_dim),
                                     name='{}_V'.format(self.name))
            self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                           K.get_value(self.forget_bias_init(self.output_dim)),
                                           np.zeros(self.output_dim),
                                           np.zeros(self.output_dim))),
                                name='{}_b'.format(self.name))

            self.trainable_weights = [self.wa, self.Wa, self.Ua, self.ba, self.ca, # AttModel parameters
                                      self.V, # LSTMCond weights
                                      self.W, self.U, self.b]
        else:
            self.V_i = self.init((self.input_dim, self.output_dim), name='{}_V_i'.format(self.name))
            self.W_i = self.init((self.context_dim, self.output_dim), name='{}_W_i'.format(self.name))
            self.U_i = self.inner_init((self.output_dim, self.output_dim), name='{}_U_i'.format(self.name))
            self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

            self.V_f = self.init((self.input_dim, self.output_dim), name='{}_V_f'.format(self.name))
            self.W_f = self.init((self.context_dim, self.output_dim), name='{}_W_f'.format(self.name))
            self.U_f = self.inner_init((self.output_dim, self.output_dim),name='{}_U_f'.format(self.name))
            self.b_f = self.forget_bias_init((self.output_dim,), name='{}_b_f'.format(self.name))


            self.V_c = self.init((self.input_dim, self.output_dim),name='{}_V_c'.format(self.name))
            self.W_c = self.init((self.context_dim, self.output_dim), name='{}_W_c'.format(self.name))
            self.U_c = self.inner_init((self.output_dim, self.output_dim),name='{}_U_c'.format(self.name))
            self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))

            self.V_o = self.init((self.input_dim, self.output_dim), name='{}_V_o'.format(self.name))
            self.W_o = self.init((self.context_dim, self.output_dim), name='{}_W_o'.format(self.name))
            self.U_o = self.inner_init((self.output_dim, self.output_dim),name='{}_U_o'.format(self.name))
            self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))

            self.V_x = self.init((self.output_dim, self.input_dim), name='{}_V_x'.format(self.name))
            self.W_x = self.init((self.output_dim, self.context_dim), name='{}_W_x'.format(self.name))
            self.b_x = K.zeros((self.context_dim,), name='{}_b_x'.format(self.name))
            self.trainable_weights = [self.wa, self.Wa, self.Ua, self.ba, self.ca, # AttModel parameters
                                      self.V_i, self.W_i, self.U_i, self.b_i,
                                      self.V_c, self.W_c, self.U_c, self.b_c,
                                      self.V_f, self.W_f, self.U_f, self.b_f,
                                      self.V_o, self.W_o, self.U_o, self.b_o,
                                      self.V_x, self.W_x, self.b_x
                                      ]

            self.W = K.concatenate([self.W_i, self.W_f, self.W_c, self.W_o])
            self.U = K.concatenate([self.U_i, self.U_f, self.U_c, self.U_o])
            self.V = K.concatenate([self.V_i, self.V_f, self.V_c, self.V_o])
            self.b = K.concatenate([self.b_i, self.b_f, self.b_c, self.b_o])

        self.regularizers = []
        # Att regularizers
        if self.wa_regularizer:
            self.wa_regularizer.set_param(self.wa)
            self.regularizers.append(self.wa_regularizer)
        if self.Wa_regularizer:
            self.Wa_regularizer.set_param(self.Wa)
            self.regularizers.append(self.Wa_regularizer)
        if self.Ua_regularizer:
            self.Ua_regularizer.set_param(self.Ua)
            self.regularizers.append(self.Ua_regularizer)
        if self.ba_regularizer:
            self.ba_regularizer.set_param(self.ba)
            self.regularizers.append(self.ba_regularizer)
        # LSTM regularizers
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.V_regularizer:
            self.V_regularizer.set_param(self.V)
            self.regularizers.append(self.V_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[2],
                        np.zeros((input_shape[0], input_shape[3])))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], input_shape[3]))]

    def preprocess_input(self, x, B_V):
        return K.dot(x * B_V[0], self.V)

    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            main_out = (input_shape[0][0], input_shape[0][1], self.output_dim)
        else:
            main_out = (input_shape[0][0], self.output_dim)

        if self.return_extra_variables:
            dim_x_att = (input_shape[0][0], input_shape[0][1], self.context_dim)
            dim_alpha_att = (input_shape[0][0], input_shape[0][1], input_shape[1][1])
            main_out = [main_out, dim_x_att, dim_alpha_att]

        if self.return_states:
            if not isinstance(main_out, list):
                main_out = [main_out]
            states_dim = (input_shape[0][0], input_shape[0][1], self.output_dim)
            main_out += [states_dim, states_dim]

        return main_out

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.

        input_shape = self.input_spec[0].shape
        state_below = x[0]
        self.context = x[1]
        if self.num_inputs == 2: # input: [state_below, context]
            self.init_state = None
            self.init_memory = None
        elif self.num_inputs == 3: # input: [state_below, context, init_generic]
            self.init_state = x[2]
            self.init_memory = x[2]
        elif self.num_inputs == 4: # input: [state_below, context, init_state, init_memory]
            self.init_state = x[2]
            self.init_memory = x[3]
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(state_below)

        constants, B_V = self.get_constants(state_below, mask[1])
        preprocessed_input = self.preprocess_input(state_below, B_V)
        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask[0],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=state_below.shape[1],
                                             pos_extra_outputs_states=[2, 3])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            ret = outputs
        else:
            ret = last_output

        if self.return_extra_variables:
            ret = [ret, states[2], states[3]]

        # intermediate states as additional outputs
        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [states[0], states[1]]

        return ret


    def compute_mask(self, input, mask):
        if self.return_extra_variables:
            ret = [mask[0], mask[0], mask[0]]
        else:
            ret = mask[0]

        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [mask[0], mask[0]]

        return ret


    def step(self, x, states):
        h_tm1 = states[0]                                 # State
        c_tm1 = states[1]                                 # Memory
        non_used_x_att = states[2]                        # Placeholder for returning extra variables
        non_used_alphas_att = states[3]                   # Placeholder for returning extra variables
        B_U = states[4]                                   # Dropout U
        B_W = states[5]                                   # Dropout W
        # Att model dropouts
        B_wa = states[6]                                  # Dropout wa
        B_Wa = states[7]                                  # Dropout Wa
        pctx_ = states[8]                                 # Projected context (i.e. context * Ua + ba)
        context = states[9]                               # Original context
        mask_input = states[10]                           # Context mask
        if mask_input.ndim > 1:                           # Mask the context (only if necessary)
            pctx_ = mask_input[:, :, None] * pctx_
            context = mask_input[:, :, None] * context

        # Attention model (see Formulation in class header)
        p_state_ = K.dot(h_tm1 * B_Wa[0], self.Wa)
        pctx_ = K.tanh(pctx_ +  p_state_[:, None, :])
        e = K.dot(pctx_ * B_wa[0], self.wa) + self.ca
        alphas_shape = e.shape
        alphas = K.softmax(e.reshape([alphas_shape[0], alphas_shape[1]]))
        # sum over the in_timesteps dimension resulting in [batch_size, input_dim]
        ctx_ = (context * alphas[:, :, None]).sum(axis=1)
        # LSTM
        if self.consume_less == 'gpu':
            z = x + \
                K.dot(h_tm1 * B_U[0], self.U)  + \
                K.dot(ctx_ * B_W[0], self.W) + \
                self.b

            z0 = z[:, :self.output_dim]
            z1 = z[:, self.output_dim: 2 * self.output_dim]
            z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
            z3 = z[:, 3 * self.output_dim:]
            i = self.inner_activation(z0)
            f = self.inner_activation(z1)
            o = self.inner_activation(z3)
            c = f * c_tm1 + i * self.activation(z2)
        h = o * self.activation(c)

        return h, [h, c, ctx_, alphas]


    def get_constants(self, x, mask_input):
        constants = []
        # States[4]
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # States[5]
        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0][0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(4)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.dropout_V < 1:
            input_dim = self.input_dim
            ones = K.ones_like(K.reshape(x[:, :, 0], (-1, x.shape[1], 1))) # (bs, timesteps, 1)
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_V = [K.in_train_phase(K.dropout(ones, self.dropout_V), ones) for _ in range(4)]
        else:
            B_V = [K.cast_to_floatx(1.) for _ in range(4)]

        # AttModel
        # States[6]
        if 0 < self.dropout_wa < 1:
            ones = K.ones_like(K.reshape(self.context[:, :, 0], (-1, self.context.shape[1], 1)))
            #ones = K.concatenate([ones], 1)
            B_wa = [K.in_train_phase(K.dropout(ones, self.dropout_wa), ones)]
            constants.append(B_wa)
        else:
            constants.append([K.cast_to_floatx(1.)])

        # States[7]
        if 0 < self.dropout_Wa < 1:
            input_dim = self.output_dim
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_Wa = [K.in_train_phase(K.dropout(ones, self.dropout_Wa), ones)]
            constants.append(B_Wa)
        else:
            constants.append([K.cast_to_floatx(1.)])

        if 0 < self.dropout_Ua < 1:
            input_dim = self.context_dim
            ones = K.ones_like(K.reshape(self.context[:, :, 0], (-1, self.context.shape[1], 1)))
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_Ua = [K.in_train_phase(K.dropout(ones, self.dropout_Ua), ones)]
            pctx = K.dot(self.context * B_Ua[0], self.Ua) + self.ba
        else:
            pctx = K.dot(self.context, self.Ua) + self.ba

        # States[8]
        constants.append(pctx)

        # States[9]
        constants.append(self.context)

        # States[10]
        if mask_input is None:
            mask_input = K.variable([])
        constants.append(mask_input)

        return constants, B_V


    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        if self.init_state is None:
            initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
            initial_state = K.expand_dims(initial_state)  # (samples, 1)
            initial_state = K.tile(initial_state, [1, self.output_dim])  # (samples, output_dim)
            if self.init_memory is None:
                initial_states = [initial_state for _ in range(2)]
            else:
                initial_memory = self.init_memory
                initial_states = [initial_state, initial_memory]
        else:
            initial_state = self.init_state
            if self.init_memory is not None: # We have state and memory
                initial_memory = self.init_memory
                initial_states = [initial_state, initial_memory]
            else:
                initial_states = [initial_state for _ in range(2)]

        initial_state = K.zeros_like(self.context)            # (samples, intput_timesteps, ctx_dim)
        initial_state_alphas = K.sum(initial_state, axis=2)   # (samples, input_timesteps)
        initial_state = K.sum(initial_state, axis=1)          # (samples, ctx_dim)
        extra_states = [initial_state, initial_state_alphas]  # (samples, ctx_dim)

        return initial_states + extra_states


    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'return_extra_variables': self.return_extra_variables,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'forget_bias_init': self.forget_bias_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'V_regularizer': self.V_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'wa_regularizer': self.wa_regularizer.get_config() if self.wa_regularizer else None,
                  'Wa_regularizer': self.Wa_regularizer.get_config() if self.Wa_regularizer else None,
                  'Ua_regularizer': self.Ua_regularizer.get_config() if self.Ua_regularizer else None,
                  'ba_regularizer': self.ba_regularizer.get_config() if self.ba_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U,
                  'dropout_V': self.dropout_V,
                  'dropout_wa': self.dropout_wa,
                  'dropout_Wa': self.dropout_Wa,
                  'dropout_Ua': self.dropout_Ua}
        base_config = super(AttLSTMCond, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

