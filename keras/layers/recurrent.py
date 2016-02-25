# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from .. import backend as K
from .. import activations, initializations, regularizers
from ..layers.core import MaskedLayer


class Recurrent(MaskedLayer):
    """
    Streamlined version of keras.layers.recurrent.Recurrent.
    May have some reduced capability ... TBD ... so far all tests passed.
    Main goals are:

    --- doc from keras.layers.recurrent.Recurrent ---

    Abstract base class for recurrent layers.
    Do not use in a model -- it's not a functional layer!

    All recurrent layers (GRU, LSTM, SimpleRNN) also
    follow the specifications of this class and accept
    the keyword arguments listed below.

    # Input shape
        3D tensor with shape `(nb_samples, timesteps, input_dim)`.

    # Output shape
        - if `return_sequences`: 3D tensor with shape
            `(nb_samples, timesteps, output_dim)`.
        - else, 2D tensor with shape `(nb_samples, output_dim)`.

    # Arguments
        weights: list of numpy arrays to set as initial weights.
            The list should have 3 elements, of shapes:
            `[(input_dim, output_dim), (output_dim, output_dim), (output_dim,)]`.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
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

    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.

    # TensorFlow warning
        For the time being, when using the TensorFlow backend,
        the number of timesteps used must be specified in your model.
        Make sure to pass an `input_length` int argument to your
        recurrent layer (if it comes first in your model),
        or to pass a complete `input_shape` argument to the first layer
        in your model otherwise.


    # Note on using statefulness in RNNs
        You can set RNN layers to be 'stateful', which means that the states
        computed for the samples in one batch will be reused as initial states
        for the samples in the next batch.
        This assumes a one-to-one mapping between
        samples in different successive batches.

        To enable statefulness:
            - specify `stateful=True` in the layer constructor.
            - specify a fixed batch size for your model, by passing
                a `batch_input_shape=(...)` to the first layer in your model.
                This is the expected shape of your inputs *including the batch size*.
                It should be a tuple of integers, e.g. `(32, 10, 100)`.

        To reset the states of your model, call `.reset_states()` on either
        a specific layer, or on your entire model.

    # Note on using dropout with TensorFlow
        When using the TensorFlow backend, specify a fixed batch size for your model
        following the notes on statefulness RNNs.

    # Updates
        - improve run-time performance by reducing the amount of computation done in each recurrence step.
        - Initial state trainable in subclasses
    """
    # Nbr dimensions of input data
    input_ndim = 3

    # Used in generating unique names
    instance_ctr = 0

    def __init__(self, weights=None,
                 return_sequences=False, go_backwards=False, stateful=False,
                 input_dim=None, input_length=None,
                 **kwargs):
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.go_backwards = go_backwards
        self.stateful = stateful

        # Used in generating unique instance names
        self.instance_ctr += 1
        self.instance_nbr = self.instance_ctr

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)

        super(Recurrent, self).__init__(**kwargs)
        return

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

    def build(self):
        """
        Build all the Layer's parameters.
        """
        # This is where local params are going to be created, so ensure Class instance has a name
        if not hasattr(self, "_name"):
            self.name = self.__class__.__name__ + "_" + str(self.instance_nbr)
        return

    def reset_states(self):
        """Reset the initial state(s) of the Recurrent layer, for stateful layers."""
        assert self.stateful, "This method supported only for `stateful` layers."

        # This method was called from `keras.layers.recurrent.*.build()` when `self.stateful` is True.
        # Instead we will follow a different design where params (e.g. `self.h0`) for initial state
        # are created in `self.build_initial_state_params()`. This allows all initial-state-param
        # building and initialization to sit in one method.
        #   `reset_states` is now needed only for external calls, and still supported only
        # when a layer is stateful.
        self.build_initial_state_params()
        return

    def build_initial_state_params(self):
        """
        This is where each subclass builds and/or (re)initializes params from which the
        initial state is built.
        """
        raise NotImplementedError

    def pre_recurrence_txform(self, x, pre_recurrence_constants):
        """
        Input transformation done before the recurrence step.
        Doing some of the computation here reduces the load on the recurrence step, and reduces run-time.

        # Arguments
            x: Input to the layer
            pre_recurrence_constants: a list containing dropout tensors,
                as returned by `self.get_pre_recurrence_constants()`.
        """
        # Default is no transformation
        return x

    def step(self, z_t, states):
        """
        Recurrent step, receives a step in the input sequence as generated by `pre_recurrence_txform`,
         and a list of the previous hidden state(s) and constants.
        Returns: Output, new_states
        """
        raise NotImplementedError

    def get_pre_recurrence_constants(self, X, train=False):
        """
        Returns additional data needed for `pre_recurrence_txform`.
        At present these are one or more dropout tensors.

        # Arguments

            X: 3D Tensor, input to the layer.
            train: bool. True if in training mode, False for prediction mode.

        # Returns
            List of Tensors, as needed for each sub-class layer.
        """
        return []

    def get_recurrence_constants(self, Z, train=False):
        """
        Returns additional data needed for each recurrence `step`.
        At present these are one or more dropout tensors.

        # Arguments

            Z: 3D Tensor, batch of time-steps as generated by `pre_recurrence_txform`.
            train: bool. True if in training mode, False for prediction mode.

        # Returns
            List of Tensors, as needed for each sub-class layer.
        """
        return []

    def get_initial_states(self, x):
        """
        Initial hidden state provided as input to the recurrent network's hidden units.
        Called only when layer is NOT stateful.
        Override, and derive from params (e.g. `self.h0`) created during `self.build()`.
        """
        raise NotImplementedError

    def get_output(self, train=False):
        """
        Generates the output (symbolic expression) of this recurrent layer.

        # Arguments

            train: bool. True if in training mode, False for prediction mode.
        """
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        X = self.get_input(train)
        mask = self.get_input_mask(train)

        assert K.ndim(X) == 3
        if K._BACKEND == 'tensorflow':
            if not self.input_shape[1]:
                raise Exception('When using TensorFlow, you should define ' +
                                'explicitly the number of timesteps of ' +
                                'your sequences.\n' +
                                'If your first layer is an Embedding, ' +
                                'make sure to pass it an "input_length" ' +
                                'argument. Otherwise, make sure ' +
                                'the first layer has ' +
                                'an "input_shape" or "batch_input_shape" ' +
                                'argument, including the time axis.')

        # Pre-recurrence step
        pre_recurrence_constants = self.get_pre_recurrence_constants(X, train)
        Z = self.pre_recurrence_txform(X, pre_recurrence_constants)

        recurrence_constants = self.get_recurrence_constants(Z, train)

        # IF stateful THEN
        #   the state at the end of one batch is the input to the next batch
        #   This is taken care of by adding self.states[*] to self.updates below
        #   The initial state is built during `self.build()`.
        #   Note that in this situation, the initial state cannot be trainable.
        # ELSE
        #   derive it from current input

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(Z)

        last_output, outputs, states = K.rnn(self.step, Z,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             # reverse_on_backwards=True,
                                             mask=mask,
                                             constants=recurrence_constants)
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "return_sequences": self.return_sequences,
                  "go_backwards": self.go_backwards,
                  "stateful": self.stateful
                  }
        if self.stateful:
            config['batch_input_shape'] = self.input_shape
        else:
            config['input_dim'] = self.input_dim
            config['input_length'] = self.input_length

        base_config = super(Recurrent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SimpleRNN(Recurrent):
    """
    Fully-connected RNN where the output is to be fed back to input.

    Core recurrence unit supporting batch inputs, for use as the recurrent portion of a traditional RNN.
    Traditional RNN does:
        h_t = tanh( dot(x, W) + b + dot(h_tm1, U) )
        o_t = softmax( dot(h_t, Wo) + bo )
    Traditional RNN flow decomposes to:
        StreamlinedRNN[ Pre-Recurrence=Linear(in=x, out=h). Recurrence-Step(in=h, out=h) ]
        -> TimeDistributedDense(activation='softmax', in=h, out=nc)
    where:
        StreamlinedRNN:
            Linear:             lin_out = dot(x, W) + b
            Recurrence-Step:    h[t] = tanh( lin_out[t] + dot(h[t-1], U) )
        TimeDistributedDense:   out = softmax( dot(h, Wo) + bo )

    # Arguments

        As accepted by superclasses, esp. class:`Recurrent`,
        and the following:

    # Additional Arguments

        output_dim: Number of hidden units.

        activation: activation function which produces the output of the hidden units.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).

        add_bias: bool. Whether to add a bias before activation. Default is True.

        initial_state_trainable: bool. Whether the initial state (h0) is a trainable quantity. Default is False.

        init: weight initialization function for the weight matrix `W`.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).

        inner_init: weight initialization function for the weight matrix `U`.

        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrix `W`.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrix `U`.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias `b`.

        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks]
            (http://arxiv.org/abs/1512.05287)
    """

    def __init__(self, output_dim,
                 activation='tanh',
                 add_bias=True, initial_state_trainable=False,
                 init='glorot_uniform', inner_init='orthogonal',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0.,
                 **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)

        self.W_regularizer = W_regularizer
        self.U_regularizer = U_regularizer
        self.b_regularizer = b_regularizer
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        self.initial_state_trainable = initial_state_trainable
        self.add_bias = add_bias

        super(SimpleRNN, self).__init__(**kwargs)

        # See note in `build_initial_state_params()`
        assert not (self.stateful and self.initial_state_trainable), \
            "If Stateful then initial state cannot be Trainable"

        return

    def build(self):
        """
        Instantiation of layer weights and other parameters.

        Called after `set_previous`, or after `set_input_shape`,
        once the layer has a defined input shape.
        Must be implemented on all layers that have weights.
        """
        super(SimpleRNN, self).build()

        self.input_dim = self.input_shape[2]

        # Build the Weights and Bias

        self.W = self.init((self.input_dim, self.output_dim), name=self.name + ".W")
        self.U = self.inner_init((self.output_dim, self.output_dim), name=self.name + ".U")

        self.trainable_weights = [self.W, self.U]

        if self.add_bias:
            self.b = K.zeros((self.output_dim,), name=self.name + ".b")
            self.trainable_weights.append(self.b)

        # Build the initial state
        self.build_initial_state_params()

        if self.initial_weights is not None:
            # Care needed if using this to also set self.h0, which is the last param in the sequence
            self.set_weights(self.initial_weights)
            del self.initial_weights

        # Add any regularizers

        def append_regulariser(input_regulariser, param, regularizers_list):
            regulariser = regularizers.get(input_regulariser)
            if regulariser:
                regulariser.set_param(param)
                regularizers_list.append(regulariser)

        self.regularizers = []
        append_regulariser(self.W_regularizer, self.W, self.regularizers)
        append_regulariser(self.U_regularizer, self.U, self.regularizers)
        append_regulariser(self.b_regularizer, self.b, self.regularizers)

        return

    def build_initial_state_params(self):
        """
        Initial state for RNN has a single component, derived from self.h0.
        """
        assert not (self.stateful and self.initial_state_trainable), \
            "If Stateful then initial state cannot be Trainable"
        input_shape = self.input_shape

        # IF self.stateful
        #   THEN batch-size must be specified and be a fixed constant.
        #        self.h0 is then the initial state of corresponding size,
        #        and it can NOT be Trainable,
        #        as the next batch gets as its input state the last state of the previous batch.
        #   ELSE
        #       initial state is derived from self.h0,
        #       and it CAN be Trainable.

        if self.stateful:
            if not input_shape[0]:
                raise Exception('If a RNN is stateful, a complete ' +
                                'input_shape must be provided (including batch size).')
            if hasattr(self, 'h0'):
                # This path followed when called from `reset_states()`
                K.set_value(self.h0, np.zeros((input_shape[0], self.output_dim)))
            else:
                self.h0 = K.zeros((input_shape[0], self.output_dim), name=self.name + ".h0")
            self.states = [self.h0]
            self.non_trainable_weights = [self.h0]
        else:
            self.h0 = K.zeros((self.output_dim,), name=self.name + ".h0")
            if self.initial_state_trainable:
                self.trainable_weights.append(self.h0)
            else:
                self.non_trainable_weights = [self.h0]
            self.states = [None]
        return

    def get_initial_states(self, x):
        """
        In SimpleRNN, there is only one component to the hidden state, so one element in the list.
        """
        assert not self.stateful, "Layer cannot be stateful when calling this method."
        # tile self.h0 as many times as batch_size ... faster than the original implementation
        return [ K.tile(self.h0, (K.shape(x)[0], 1)) ]

    def get_pre_recurrence_constants(self, X, train=False):
        retain_p_W = 1. - self.dropout_W
        if train and self.dropout_W > 0:
            nb_samples = K.shape(X)[0]
            if K._BACKEND == 'tensorflow':
                if not self.input_shape[0]:
                    raise Exception('For RNN dropout in tensorflow, a complete ' +
                                    'input_shape must be provided (including batch size).')
                nb_samples = self.input_shape[0]
            # Each sample in the batch gets a different mask,
            #   each time step in a sample gets the same mask.
            # B_W is applied to X during `pre_recurrence_xform()`, so its shape should match X
            if K._BACKEND == 'theano':
                # Broadcasting the 2nd dimension is faster than tiled version.
                B_W = K.addbroadcast(K.random_binomial((nb_samples, 1, self.input_dim), p=retain_p_W), 1)
            else:
                B_W = K.tile(K.random_binomial((nb_samples, 1, self.input_dim), p=retain_p_W),
                             (1, K.shape(X)[1], 1))
        else:
            B_W = np.ones(1, dtype=K.floatx()) * retain_p_W
        return [B_W]

    def get_recurrence_constants(self, Z, train=False):
        retain_p_U = 1. - self.dropout_U
        if train and self.dropout_U > 0:
            nb_samples = K.shape(Z)[0]
            if K._BACKEND == 'tensorflow':
                if not self.input_shape[0]:
                    raise Exception('For RNN dropout in tensorflow, a complete ' +
                                    'input_shape must be provided (including batch size).')
                nb_samples = self.input_shape[0]
            # Each sample in the batch gets a different mask,
            #   each time step in a sample gets the same mask.
            # B_U is applied to z_t during recurrent `step`
            B_U = K.random_binomial((nb_samples, self.output_dim), p=retain_p_U)
        else:
            B_U = np.ones(1, dtype=K.floatx()) * retain_p_U
        return [B_U]

    def pre_recurrence_txform(self, x, pre_recurrence_constants):
        """
        lin_out = dot(x * B_W, W) + b

        # Arguments
            x: Input to the layer
            constants: as returned by `self.get_constants()`, a list containing dropout matrices.
       """
        B_W = pre_recurrence_constants[0]
        z = K.dot(x * B_W, self.W)
        if self.add_bias:
            z = z + self.b
        return z

    def step(self, z_t, states):
        # h_t = activation( dot(h_tm1, W) + z_t [+ b] )
        assert len(states) == 2  # = [h_tm1, B_U]
        h_tm1 = states[0]
        B_U = states[1]
        h_t = self.activation(z_t + K.dot(h_tm1 * B_U, self.U))
        # The output h_t is also the new hidden state
        return h_t, [h_t]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "add_bias": self.add_bias,
                  "initial_state_trainable": self.initial_state_trainable,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U
                  }
        base_config = super(SimpleRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class GRU(Recurrent):
    '''
    Gated Recurrent Unit - Cho et al. 2014.

    # Arguments
        (See class:`Recurrent` for common arguments to all Recurrent layers).

        output_dim: dimension of the internal projections and the final output.

        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).

        inner_activation: activation function for the inner cells.

        add_bias: bool. Whether to add a bias before activation. Default is True.

        initial_state_trainable: bool. Whether the initial state (h0) is a trainable quantity. Default is False.

        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).

        inner_init: initialization function of the inner cells.

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
                 activation='tanh', inner_activation='hard_sigmoid',
                 add_bias=True, initial_state_trainable=False,
                 init='glorot_uniform', inner_init='orthogonal',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0.,
                 **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.W_regularizer = W_regularizer
        self.U_regularizer = U_regularizer
        self.b_regularizer = b_regularizer
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        self.initial_state_trainable = initial_state_trainable
        self.add_bias = add_bias

        super(GRU, self).__init__(**kwargs)

        assert not (self.stateful and self.initial_state_trainable), \
            "If Stateful then initial state cannot be Trainable"

        return

    def build(self):
        super(GRU, self).build()

        input_shape = self.input_shape
        input_dim = input_shape[2]
        self.input_dim = input_dim

        #self.input = K.placeholder(input_shape) # Why is this needed?

        # Build the Weights and Bias

        self.W_z = self.init((input_dim, self.output_dim), name=self.name + "W_z")
        self.U_z = self.inner_init((self.output_dim, self.output_dim), name=self.name + "U_z")

        self.W_r = self.init((input_dim, self.output_dim), name=self.name + "W_r")
        self.U_r = self.inner_init((self.output_dim, self.output_dim), name=self.name + "U_r")

        self.W_h = self.init((input_dim, self.output_dim), name=self.name + "W_h")
        self.U_h = self.inner_init((self.output_dim, self.output_dim), name=self.name + "U_h")

        self.trainable_weights = [self.W_z, self.U_z,
                                  self.W_r, self.U_r,
                                  self.W_h, self.U_h]

        if self.add_bias:
            self.b_z = K.zeros((self.output_dim,), name=self.name + ".b_z")
            self.b_r = K.zeros((self.output_dim,), name=self.name + ".b_r")
            self.b_h = K.zeros((self.output_dim,), name=self.name + ".b_h")
            self.b = K.concatenate([self.b_z, self.b_r, self.b_h], axis=-1)
            self.trainable_weights += [self.b_z, self.b_r, self.b_h]

        # Build the initial state
        self.build_initial_state_params()

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        # Add any regularizers

        def append_regulariser(input_regulariser, param, regularizers_list):
            regulariser = regularizers.get(input_regulariser)
            if regulariser:
                regulariser.set_param(param)
                regularizers_list.append(regulariser)

        self.regularizers = []
        for W in [self.W_z, self.W_r, self.W_h]:
            append_regulariser(self.W_regularizer, W, self.regularizers)
        for U in [self.U_z, self.U_r, self.U_h]:
            append_regulariser(self.U_regularizer, U, self.regularizers)
        for b in [self.b_z, self.b_r, self.b_h]:
            append_regulariser(self.b_regularizer, b, self.regularizers)

        return

    def build_initial_state_params(self):
        """
        GRU has one component to the initial state.
        """
        assert not (self.stateful and self.initial_state_trainable), \
                    "If Stateful then initial state cannot be Trainable"
        input_shape = self.input_shape

        if self.stateful:
            if not input_shape[0]:
                        raise Exception('If a RNN is stateful, a complete ' +
                                        'input_shape must be provided (including batch size).')
            if hasattr(self, 'h0'):
                # This path followed when called from `reset_states()`
                K.set_value(self.h0, np.zeros((input_shape[0], self.output_dim)))
            else:
                self.h0 = K.zeros((input_shape[0], self.output_dim), name=self.name + ".h0")
            self.states = [self.h0]
            self.non_trainable_weights = [self.h0]
        else:
            self.h0 = K.zeros((self.output_dim,), name=self.name + ".h0")
            if self.initial_state_trainable:
                self.trainable_weights += [self.h0]
            else:
                self.non_trainable_weights = [self.h0]
            self.states = [None]
        return

    def get_initial_states(self, x):
        assert not self.stateful, "Layer cannot be stateful when calling this method."
        # tile as many times as batch_size
        return [ K.tile(self.h0,  (K.shape(x)[0], 1)) ]

    def get_pre_recurrence_constants(self, X, train=False):
        retain_p_W = 1. - self.dropout_W
        if train and self.dropout_W > 0:
            nb_samples = K.shape(X)[0]
            if K._BACKEND == 'tensorflow':
                if not self.input_shape[0]:
                    raise Exception('For RNN dropout in tensorflow, a complete ' +
                                    'input_shape must be provided (including batch size).')
                nb_samples = self.input_shape[0]
            # Each sample in the batch gets a different mask,
            #   each time step in a sample gets the same mask,
            #   each mask is actually a composite of 3 independent masks, one for each of the W_*
            # Each component mask is applied to X during `pre_recurrence_xform()`, so its shape matches X
            if K._BACKEND == 'theano':
                # Broadcasting the 2nd dimension is faster than tiled version.
                B_W = [K.addbroadcast(K.random_binomial((nb_samples, 1, self.input_dim), p=retain_p_W), 1)
                       for _ in range(3)]
            else:
                B_W = [K.tile(K.random_binomial((nb_samples, 1, self.input_dim), p=retain_p_W),
                              (1, K.shape(X)[1], 1))
                       for _ in range(3)]
        else:
            B_W = [np.ones(1, dtype=K.floatx()) * retain_p_W for _ in range(3)]
        return [B_W]

    def get_recurrence_constants(self, Z, train=False):
        retain_p_U = 1. - self.dropout_U
        if train and self.dropout_U > 0:
            nb_samples = K.shape(Z)[0]
            if K._BACKEND == 'tensorflow':
                if not self.input_shape[0]:
                    raise Exception('For RNN dropout in tensorflow, a complete ' +
                                    'input_shape must be provided (including batch size).')
                nb_samples = self.input_shape[0]
            # Each sample in the batch gets a different mask,
            #   each time step in a sample gets the same mask,
            #   each mask is actually a composite of 3 independent masks, one for each of the W_*
            # B_U is applied to z_t during recurrent `step`
            B_U = [K.random_binomial((nb_samples, self.output_dim), p=retain_p_U) for _ in range(3)]
        else:
            B_U = [np.ones(1, dtype=K.floatx()) * retain_p_U for _ in range(3)]
        return [B_U]

    def pre_recurrence_txform(self, x, pre_recurrence_constants):
        B_W = pre_recurrence_constants[0]

        x_z = K.dot(x * B_W[0], self.W_z)
        x_r = K.dot(x * B_W[1], self.W_r)
        x_h = K.dot(x * B_W[2], self.W_h)

        z = K.concatenate([x_z, x_r, x_h], axis=-1)

        # b is a concatenation of b_z, b_r, b_h
        if self.add_bias:
            z = z + self.b
        return z

    def step(self, z_t, states):
        assert len(states) == 2  # 1 state and 1 constant (B_U)
        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrix for recurrent units

        # Split z_t (2D) into z_t_z, z_t_r, z_t_h
        z_t_z = z_t[:, 0:self.output_dim]
        z_t_r = z_t[:, self.output_dim:2 * self.output_dim]
        z_t_h = z_t[:, 2 * self.output_dim:]

        z = self.inner_activation(z_t_z + K.dot(h_tm1 * B_U[0], self.U_z))
        r = self.inner_activation(z_t_r + K.dot(h_tm1 * B_U[1], self.U_r))

        hh = self.activation(z_t_h + K.dot(r * h_tm1 * B_U[2], self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "add_bias": self.add_bias,
                  "initial_state_trainable": self.initial_state_trainable,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(GRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LSTM(Recurrent):
    """
    Long-Short Term Memory unit - Hochreiter 1997.

    For a step-by-step description of the algorithm, see
    [this tutorial](http://deeplearning.net/tutorial/lstm.html).

    # Arguments
        (See class:`Recurrent` for common arguments to all Recurrent layers).

        output_dim: dimension of the internal projections and the final output.

        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).

        inner_activation: activation function for the inner cells.

        add_bias: bool. Whether to add a bias before activation. Default is True.

        initial_state_trainable: bool. Whether the initial state (h0) is a trainable quantity. Default is False.

        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).

        inner_init: initialization function of the inner cells.

        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.

    # References
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
    """
    def __init__(self, output_dim,
                 activation='tanh', inner_activation='hard_sigmoid',
                 add_bias=True, initial_state_trainable=False,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0.,
                 **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.W_regularizer = W_regularizer
        self.U_regularizer = U_regularizer
        self.b_regularizer = b_regularizer
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        self.initial_state_trainable = initial_state_trainable
        self.add_bias = add_bias

        super(LSTM, self).__init__(**kwargs)

        assert not (self.stateful and self.initial_state_trainable), \
            "If Stateful then initial state cannot be Trainable"

        return

    def build(self):
        super(LSTM, self).build()

        input_shape = self.input_shape
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.input = K.placeholder(input_shape) # Why is this needed?

        # Build the Weights and Bias

        self.W_i = self.init((input_dim, self.output_dim), name=self.name + ".W_i")
        self.U_i = self.inner_init((self.output_dim, self.output_dim), name=self.name + ".U_i")

        self.W_f = self.init((input_dim, self.output_dim), name=self.name + ".W_f")
        self.U_f = self.inner_init((self.output_dim, self.output_dim), name=self.name + ".U_f")

        self.W_c = self.init((input_dim, self.output_dim), name=self.name + ".W_c")
        self.U_c = self.inner_init((self.output_dim, self.output_dim), name=self.name + ".U_c")

        self.W_o = self.init((input_dim, self.output_dim), name=self.name + ".W_o")
        self.U_o = self.inner_init((self.output_dim, self.output_dim), name=self.name + ".U_o")

        # The concatenated form allows combining 4 matrix mults into one operation
        # W = [W_i | W_f | W_c | W_o]
        # self.W = K.concatenate([self.W_i, self.W_f, self.W_c, self.W_o], axis=-1)

        self.trainable_weights = [self.W_i, self.U_i,
                                  self.W_c, self.U_c,
                                  self.W_f, self.U_f,
                                  self.W_o, self.U_o]

        if self.add_bias:
            self.b_i = K.zeros((self.output_dim,), name=self.name + ".b_i")
            self.b_f = self.forget_bias_init((self.output_dim,), name=self.name + ".b_f")
            self.b_c = K.zeros((self.output_dim,), name=self.name + ".b_c")
            self.b_o = K.zeros((self.output_dim,), name=self.name + ".b_o")

            self.b = K.concatenate([self.b_i, self.b_f, self.b_c, self.b_o], axis=-1)
            self.trainable_weights += [self.b_i, self.b_f, self.b_c, self.b_o]

        # Build the initial state
        self.build_initial_state_params()

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        # Add any regularizers

        def append_regulariser(input_regulariser, param, regularizers_list):
            regulariser = regularizers.get(input_regulariser)
            if regulariser:
                regulariser.set_param(param)
                regularizers_list.append(regulariser)

        self.regularizers = []
        for W in [self.W_i, self.W_f, self.W_i, self.W_o]:
            append_regulariser(self.W_regularizer, W, self.regularizers)
        for U in [self.U_i, self.U_f, self.U_i, self.U_o]:
            append_regulariser(self.U_regularizer, U, self.regularizers)
        for b in [self.b_i, self.b_f, self.b_i, self.b_o]:
            append_regulariser(self.b_regularizer, b, self.regularizers)

        return

    def build_initial_state_params(self):
        """
        StreamlinedLSTM has two components to the initial state: h0, \tilde{c}0 (here called cc0)
        """
        assert not (self.stateful and self.initial_state_trainable), \
                    "If Stateful then initial state cannot be Trainable"
        input_shape = self.input_shape

        if self.stateful:
            if not input_shape[0]:
                        raise Exception('If a RNN is stateful, a complete ' +
                                        'input_shape must be provided (including batch size).')
            if hasattr(self, 'h0'):
                # This path followed when called from `reset_states()`
                K.set_value(self.h0, np.zeros((input_shape[0], self.output_dim)))
            else:
                self.h0  = K.zeros((input_shape[0], self.output_dim), name=self.name + ".h0")
            if hasattr(self, 'cc0'):
                # This path followed when called from `reset_states()`
                K.set_value(self.cc0, np.zeros((input_shape[0], self.output_dim)))
            else:
                self.cc0 = K.zeros((input_shape[0], self.output_dim), name=self.name + ".cc0")
            self.states = [self.h0, self.cc0]
            self.non_trainable_weights = [self.h0, self.cc0]
        else:
            self.h0  = K.zeros((self.output_dim,), name=self.name + ".h0")
            self.cc0 = K.zeros((self.output_dim,), name=self.name + ".cc0")
            if self.initial_state_trainable:
                self.trainable_weights += [self.h0, self.cc0]
            else:
                self.non_trainable_weights = [self.h0, self.cc0]
            self.states = [None, None]
        return

    def get_initial_states(self, x):
        assert not self.stateful, "Layer cannot be stateful when calling this method."
        # tile as many times as batch_size
        return [ K.tile(self.h0,  (K.shape(x)[0], 1)),
                 K.tile(self.cc0, (K.shape(x)[0], 1)) ]

    def get_pre_recurrence_constants(self, X, train=False):
        retain_p_W = 1. - self.dropout_W
        if train and self.dropout_W > 0:
            nb_samples = K.shape(X)[0]
            if K._BACKEND == 'tensorflow':
                if not self.input_shape[0]:
                    raise Exception('For RNN dropout in tensorflow, a complete ' +
                                    'input_shape must be provided (including batch size).')
                nb_samples = self.input_shape[0]
            # Each sample in the batch gets a different mask,
            #   each time step in a sample gets the same mask,
            #   each mask is actually a composite of 4 independent masks, one for each of the W_*
            # Each component mask is applied to X during `pre_recurrence_xform()`, so its shape matches X
            if K._BACKEND == 'theano':
                # Broadcasting the 2nd dimension is faster than tiled version.
                B_W = [K.addbroadcast(K.random_binomial((nb_samples, 1, self.input_dim), p=retain_p_W), 1)
                       for _ in range(4)]
            else:
                B_W = [K.tile(K.random_binomial((nb_samples, 1, self.input_dim), p=retain_p_W),
                              (1, K.shape(X)[1], 1))
                       for _ in range(4)]
        else:
            B_W = [np.ones(1, dtype=K.floatx()) * retain_p_W for _ in range(4)]
        return [B_W]

    def get_recurrence_constants(self, Z, train=False):
        retain_p_U = 1. - self.dropout_U
        if train and self.dropout_U > 0:
            nb_samples = K.shape(Z)[0]
            if K._BACKEND == 'tensorflow':
                if not self.input_shape[0]:
                    raise Exception('For RNN dropout in tensorflow, a complete ' +
                                    'input_shape must be provided (including batch size).')
                nb_samples = self.input_shape[0]
            # Each sample in the batch gets a different mask,
            #   each time step in a sample gets the same mask,
            #   each mask is actually a composite of 4 independent masks, one for each of the W_*
            # B_U is applied to z_t during recurrent `step`
            B_U = [K.random_binomial((nb_samples, self.output_dim), p=retain_p_U) for _ in range(4)]
        else:
            B_U = [np.ones(1, dtype=K.floatx()) * retain_p_U for _ in range(4)]
        return [B_U]

    def pre_recurrence_txform(self, x, pre_recurrence_constants):
        B_W = pre_recurrence_constants[0]

        x_i = K.dot(x * B_W[0], self.W_i)
        x_f = K.dot(x * B_W[1], self.W_f)
        x_c = K.dot(x * B_W[2], self.W_c)
        x_o = K.dot(x * B_W[3], self.W_o)

        z = K.concatenate([x_i, x_f, x_c, x_o], axis=-1)

        # b is a concatenation of b_i, b_f, b_c, b_o
        if self.add_bias:
            z = z + self.b
        return z

    def step(self, z_t, states):
        assert len(states) == 3    # 2 states and B_U
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]

        # Split z_t (2D) into z_t_i, z_t_f, z_t_c, z_t_o
        z_t_i = z_t[:, 0:self.output_dim]
        z_t_f = z_t[:, self.output_dim:2 * self.output_dim]
        z_t_c = z_t[:, 2 * self.output_dim:3 * self.output_dim]
        z_t_o = z_t[:, 3 * self.output_dim:]

        i = self.inner_activation(z_t_i + K.dot(h_tm1 * B_U[0], self.U_i))
        f = self.inner_activation(z_t_f + K.dot(h_tm1 * B_U[1], self.U_f))
        c = f * c_tm1 + i * self.activation(z_t_c + K.dot(h_tm1 * B_U[2], self.U_c))
        o = self.inner_activation(z_t_o + K.dot(h_tm1 * B_U[3], self.U_o))
        h = o * self.activation(c)
        return h, [h, c]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "add_bias": self.add_bias,
                  "initial_state_trainable": self.initial_state_trainable,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U
                  }
        base_config = super(LSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
#/
