# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras import activations
from .. import backend as K
from ..engine import InputSpec
from ..engine import Layer
from ..layers import concatenate
from ..layers import has_arg
from .. import initializers
from .. import regularizers
from .. import constraints


class _RNNAttentionCell(Layer):
    """Base class for recurrent attention mechanisms.

    This base class implements the RNN cell interface and defines a standard
    way for attention mechanisms to interact with a (wrapped) "core" RNN cell
    (such as the `SimpleRNNCell`, `GRUCell` or `LSTMCell`).

    The main idea is that the attention mechanism, implemented by
    `attention_call` in extensions of this class, computes an "attention
    encoding", based on the attended input as well as the input and the core
    cell state(s) at the current time step, which will be used as modified
    input for the core cell.

    # Arguments
        cell: A RNN cell instance. The cell to wrap by the attention mechanism.
            A RNN cell is a class that has:
            - a `call(input_at_t, states_at_t)` method, returning
                `(output_at_t, states_at_t_plus_1)`.
            - a `state_size` attribute. This can be a single integer
                (single state) in which case it is the size of the recurrent
                state (which should be the same as the size of the cell
                output). This can also be a list/tuple of integers (one size
                per state). In this case, the first entry (`state_size[0]`)
                should be the same as the size of the cell output.
        attend_after: Boolean (default False). If True, the attention
            transformation defined by `attention_call` will be applied after
            the core cell transformation (and the attention encoding will be
            used as input for core cell transformation next time step).
        concatenate_input: Boolean (default True). If True the concatenation of
            the attention encoding and the original input will be used as input
            for the core cell transformation. If set to False, only the
            attention encoding will be used as input for the core cell
            transformation.

    # Abstract Methods and Properties
        Extension of this class must implement:
            - `attention_build` (method): Builds the attention transformation
              based on input shapes.
            - `attention_call` (method): Defines the attention transformation
              returning the attention encoding.
            - `attention_size` (property): After `attention_build` has been
              called, this property should return the size (int) of the
              attention encoding. Do this by setting `_attention_size` in scope
              of `attention_build` or by implementing `attention_size`
              property.
        Extension of this class can optionally implement:
            - `attention_state_size` (property): Default [`attention_size`].
              If the attention mechanism has it own internal states (besides
              the attention encoding which is by default the only part of
              `attention_states`) override this property accordingly.
        See docs of the respective method/property for further details.

    # Details of interaction between attention and cell transformations
        Let "cell" denote core (wrapped) RNN cell and "att(cell)" the complete
        attentive RNN cell defined by this class. We write the core cell
        transformation as:

            y{t}, s_cell{t+1} = cell.call(x{t}, s_cell{t})

        where y{t} denotes the output, x{t} the input at and s_cell{t} the core
        cell state(s) at time t and s_cell{t+1} the updated state(s).

        We can then write the complete "attentive" cell transformation as:

            y{t}, s_att(cell){t+1} = att(cell).call(x{t}, s_att(cell){t},
                                                    constants=attended)

        where s_att(cell) denotes the complete states of the attentive cell,
        which consists of the core cell state(s) followed but the attention
        state(s), and attended denotes the tensor attended to (note: no time
        indexing as this is the same constant input at each time step).

        Internally, this is how the attention transformation, implemented by
        `attention_call`, interacts with the core cell transformation
        `cell.call`:

        - with `attend_after=False` (default):
            a{t}, s_att{t+1} = att(cell).attention_call(x_t, s_cell{t},
                                                        attended, s_att{t})
            with `concatenate_input=True` (default):
                x'{t} = [x{t}, a{t}]
            else:
                x'{t} = a{t}
            y{t}, s_cell{t+1} = cell.call(x'{t}, s_cell{t})

        - with `attend_after=True`:
            with `concatenate_input=True` (default):
                x'{t} = [x{t}, a{t-1}]
            else:
                x'{t} = a{t-1}
            y{t}, s_cell{t+1} = cell.call(x'{t}, s_cell{t})
            a{t}, s_att{t+1} = att(cell).attention_call(x_t, s_cell{t+1},
                                                        attended, s_att{t})

        where a{t} denotes the attention encoding, s_att{t} the attention
        state(s), x'{t} the modified core cell input and [x{.}, a{.}] the
        (tensor) concatenation of the input and attention encoding.
    """

    def __init__(self, cell,
                 attend_after=False,
                 concatenate_input=False,
                 **kwargs):
        self.cell = cell  # must be set before calling super
        super(_RNNAttentionCell, self).__init__(**kwargs)
        self.attend_after = attend_after
        self.concatenate_input = concatenate_input
        self.attended_spec = None
        self._attention_size = None

    def attention_call(self,
                       inputs,
                       cell_states,
                       attended,
                       attention_states,
                       training=None):
        """The main logic for computing the attention encoding.

        # Arguments
            inputs: The input at current time step.
            cell_states: States for the core RNN cell.
            attended: The same tensor(s) to attend at each time step.
            attention_states: States dedicated for the attention mechanism.
            training: whether run in training mode or not

        # Returns
            attention_h: The computed attention encoding at current time step.
            attention_states: States to be passed to next `attention_call`. By
                default this should be [`attention_h`].
                NOTE: if additional states are used, these should be appended
                after `attention_h`, i.e. `attention_states[0]` should always
                be `attention_h`.
        """
        raise NotImplementedError(
            '`attention_call` must be implemented by extensions of `{}`'.format(
                self.__class__.__name__))

    def attention_build(self, input_shape, cell_state_size, attended_shape):
        """Build the attention mechanism.

        NOTE: `self._attention_size` should be set in this method to the size
        of the attention encoding (i.e. size of first `attention_states`)
        unless `attention_size` property is implemented in another way.

        # Arguments
            input_shape: Tuple of integers. Shape of the input at a single time
                step.
            cell_state_size: List of tuple of integers.
            attended_shape: List of tuple of integers.

            NOTE: both `cell_state_size` and `attended_shape` will always be
            lists - for simplicity. For example: even if (wrapped)
            `cell.state_size` is an integer, `cell_state_size` will be a list
            of this one element.
        """
        raise NotImplementedError(
            '`attention_build` must be implemented by extensions of `{}`'.format(
                self.__class__.__name__))

    @property
    def attention_size(self):
        """Size off attention encoding, an integer.
        """
        if self._attention_size is None and self.built:
            raise NotImplementedError(
                'extensions of `{}` must either set property `_attention_size`'
                ' in `attention_build` or implement the or implement'
                ' `attention_size` in some other way'.format(
                    self.__class__.__name__))

        return self._attention_size

    @property
    def attention_state_size(self):
        """Size of attention states, defaults to `attention_size`, an integer.

        Modify this property to return list of integers if the attention
        mechanism has several internal states. Note that the first size should
        always be the size of the attention encoding, i.e.:
            `attention_state_size[0]` = `attention_size`
        """
        return self.attention_size

    @property
    def state_size(self):
        """Size of states of the complete attentive cell, a tuple of integers.

        The attentive cell's states consists of the core RNN cell state size(s)
        followed by attention state size(s). NOTE it is important that the core
        cell states are first as the first state of any RNN cell should be same
        as the cell's output.
        """
        state_size_s = []
        for state_size in [self.cell.state_size, self.attention_state_size]:
            if hasattr(state_size, '__len__'):
                state_size_s += list(state_size)
            else:
                state_size_s.append(state_size)

        return tuple(state_size_s)

    def call(self, inputs, states, constants, training=None):
        """Complete attentive cell transformation.
        """
        attended = constants
        cell_states = states[:self._num_wrapped_states]
        attention_states = states[self._num_wrapped_states:]

        if self.attend_after:
            attention_call = self.call_attend_after
        else:
            attention_call = self.call_attend_before

        return attention_call(inputs=inputs,
                              cell_states=cell_states,
                              attended=attended,
                              attention_states=attention_states,
                              training=training)

    def call_attend_before(self,
                           inputs,
                           cell_states,
                           attended,
                           attention_states,
                           training=None):
        """Complete attentive cell transformation, if `attend_after=False`.
        """
        attention_h, new_attention_states = self.attention_call(
            inputs=inputs,
            cell_states=cell_states,
            attended=attended,
            attention_states=attention_states,
            training=training)

        if self.concatenate_input:
            cell_input = concatenate([attention_h, inputs])
        else:
            cell_input = attention_h

        if has_arg(self.cell.call, 'training'):
            output, new_cell_states = self.cell.call(cell_input, cell_states,
                                                     training=training)
        else:
            output, new_cell_states = self.cell.call(cell_input, cell_states)

        return output, new_cell_states + new_attention_states

    def call_attend_after(self,
                          inputs,
                          cell_states,
                          attended,
                          attention_states,
                          training=None):
        """Complete attentive cell transformation, if `attend_after=True`.
        """
        attention_h_previous = attention_states[0]

        if self.concatenate_input:
            cell_input = concatenate([attention_h_previous, inputs])
        else:
            cell_input = attention_h_previous

        if has_arg(self.cell.call, 'training'):
            output, new_cell_states = self.cell.call(cell_input, cell_states,
                                                     training=training)
        else:
            output, new_cell_states = self.cell.call(cell_input, cell_states)

        attention_h, new_attention_states = self.attention_call(
            inputs=inputs,
            cell_states=new_cell_states,
            attended=attended,
            attention_states=attention_states,
            training=training)

        return output, new_cell_states, new_attention_states

    @staticmethod
    def _num_elements(x):
        if hasattr(x, '__len__'):
            return len(x)
        else:
            return 1

    @property
    def _num_wrapped_states(self):
        return self._num_elements(self.cell.state_size)

    @property
    def _num_attention_states(self):
        return self._num_elements(self.attention_state_size)

    def build(self, input_shape):
        """Builds attention mechanism and wrapped cell (if keras layer).

        Arguments:
            input_shape: list of tuples of integers, the input feature shape
                (inputs sequence shape without time dimension) followed by
                constants (i.e. attended) shapes.
        """
        if not isinstance(input_shape, list):
            raise ValueError('input shape should contain shape of both cell '
                             'inputs and constants (attended)')

        attended_shape = input_shape[1:]
        input_shape = input_shape[0]
        self.attended_spec = [InputSpec(shape=shape) for shape in attended_shape]
        if isinstance(self.cell.state_size, int):
            cell_state_size = [self.cell.state_size]
        else:
            cell_state_size = list(self.cell.state_size)
        self.attention_build(
            input_shape=input_shape,
            cell_state_size=cell_state_size,
            attended_shape=attended_shape,
        )

        if isinstance(self.cell, Layer):
            cell_input_shape = (input_shape[0],
                                self.attention_size +
                                input_shape[-1] if self.concatenate_input
                                else self._attention_size)
            self.cell.build(cell_input_shape)

        self.built = True

    def compute_output_shape(self, input_shape):
        if hasattr(self.cell.state_size, '__len__'):
            cell_output_dim = self.cell.state_size[0]
        else:
            cell_output_dim = self.cell.state_size

        return input_shape[0], cell_output_dim

    @property
    def trainable_weights(self):
        return super(_RNNAttentionCell, self).trainable_weights + \
               self.cell.trainable_weights

    @property
    def non_trainable_weights(self):
        return super(_RNNAttentionCell, self).non_trainable_weights + \
               self.cell.non_trainable_weights

    def get_config(self):
        config = {'attend_after': self.attend_after,
                  'concatenate_input': self.concatenate_input}

        cell_config = self.cell.get_config()
        config['cell'] = {'class_name': self.cell.__class__.__name__,
                          'config': cell_config}
        base_config = super(_RNNAttentionCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MixtureOfGaussian1DAttention(_RNNAttentionCell):
    """RNN attention mechanism for attending sequences.

    The attention encoding (passed to the wrapped core RNN cell) is obtained by
    letting the attention mechanism predict a Mixture of Gaussian distribution
    (MoG) over the time dimension of the attended feature sequence. The
    attention encoding is taken as the weighted sum of all features - where the
    weight is given by the probability density function (evaluated in the
    respective time step) according the predicted MoG distribution.

    # Arguments
        components: Positive integer, the number of mixture components to use
            (for each head, see below).
        heads: Positive integer (Default 1), the number of independent "read
            heads" to use. Each head produces an independent (sub) attention
            encoding, by predicting an independent MoG each. The (full)
            attention encoding passed to the wrapped core RNN cell is the
            concatenation of the attention encodings from each head. See "Notes
            on multiple heads vs multiple components" below.
        mu_activation: The activation function applied (after learnt linear
            transformation) for mu:s (expectation value/location) of each
            Gaussian component.
        sigma_activation: The activation function applied (after learnt linear
            transformation) for sigma:s (standard deviation) of each Gaussian
            component. *NOTE* that this function should only return values > 0.
        sigma_epsilon: Positive Float, this value is added to sigma to force it
            to be at least this value.
        predict_delta_mu: Boolean (Default True), whether or not to let the
            attention mechanism to predict the _change_ in location (mu) of
            each mixture component. This is recommended as it usually leads to
            more stable convergence. By passing a `mu_activation` that always
            returns a value > 0 and having `predict_delta_mu=True` it is
            enforced that the attention mechanism "parses" the attended
            sequence "from start to end" as the attention can not be moved
            backwards.
        For initializers, regularizers & constraints: See docs of Dense layer.

    # Notes on multiple heads vs multiple components
        A single head can "attend to multiple parts of the sequence" by
        using multiple components. However, the features from the location of
        the components are averaged together by a weighted sum (no
        information is kept on their internal ordering for example). With
        multiple heads, on the other side, the attention mechanism can "pick
        out" features from multiple locations without averaging them, and
        passing them "intact" to the core RNN cell. This is done at the cost of
        a larger input vector to, and thereby more parameters of, the core RNN
        cell.

    # Example - Machine Translation with Attention and "teacher forcing"
        # NOTE that this is a minimal naive example, this setup will not
        # perform well for machine translation in general.
        # TODO add `examples/machine_translation_with_attention.py`
        # with performing setup

        input_english = Input((None, tokens_english))
        target_french_tm1 = Input((None, tokens_french))

        cell = MixtureOfGaussian1DAttention(LSTMCell(64), components=3, heads=3)
        attention_lstm = RNN(cell, return_sequences=True)
        h_sequence = attention_lstm(target_french_tm1, constants=input_english)
        output_layer = TimeDistributed(Dense(tokens_french, activation='softmax'))
        predicted_french = output_layer(h_sequence)

        train_model = Model(
            inputs=[target_french_tm1, input_english],
            outputs=predicted_french
        )
        model.compile(optimizer='Adam', loss='categorical_crossentropy')
        model.fit(
            x=[french_text[:, :-1], english_text],
            y=french_text[:, 1:],
            epochs=10
        )
    """
    def __init__(self, cell,
                 components,
                 heads=1,
                 mu_activation=None,
                 sigma_activation='exponential',
                 sigma_epsilon=1e-3,
                 predict_delta_mu=True,  # TODO alternative name `cumulative_mu`?
                 kernel_initializer='glorot_uniform',  # FIXME most likely not optimal
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MixtureOfGaussian1DAttention, self).__init__(cell, **kwargs)
        self.components = components
        self.heads = heads
        self.mu_activation = activations.get(mu_activation)
        self.sigma_activation = activations.get(sigma_activation)
        self.sigma_epsilon = sigma_epsilon
        self.predict_delta_mu = predict_delta_mu
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    @property
    def attention_state_size(self):
        """Size of states dedicated for the attention mechanism.

        If self.predict_delta_mu is True, mu (the "location") for all heads'
        components needs to be forwarded to next time step and is therefore
        added to the attention states.
        """
        attention_state_size = [self.attention_size]
        if self.predict_delta_mu:
            mu_size = self.components * self.heads
            attention_state_size.append(mu_size)

        return attention_state_size

    def attention_call(self,
                       inputs,
                       cell_states,
                       attended,
                       attention_states,
                       training=None):
        # only one attended sequence for now (verified in build)
        [attended] = attended
        mu_tm1 = attention_states[1] if self.predict_delta_mu else None

        mog_input = concatenate([inputs, cell_states[0]])
        params = K.bias_add(K.dot(mog_input, self.kernel), self.bias)

        # dynamic creation of time index
        # TODO check support by all backends
        # TODO faster with non-dynamic if size of time dimension is fixed?
        time_idx = K.arange(K.shape(attended)[1], dtype='float32')
        time_idx = K.expand_dims(K.expand_dims(time_idx, 0), -1)

        if self.heads == 1:
            attention_h, mu = self._get_attention_h_and_mu(params, attended,
                                                           mu_tm1, time_idx)
        else:
            c = self.components
            attention_h_s, mu_s = zip(*[
                self._get_attention_h_and_mu(
                    params=params[..., c * i * 3:c * (i+1) * 3],
                    attended=attended,
                    mu_tm1=(mu_tm1[..., c * i:c * (i+1)]
                            if self.predict_delta_mu else None),
                    time_idx=time_idx
                ) for i in range(self.heads)
            ])
            attention_h = concatenate(list(attention_h_s))
            mu = concatenate(list(mu_s))

        new_attention_states = [attention_h]
        if self.predict_delta_mu:
            new_attention_states.append(mu)

        return attention_h, new_attention_states

    def _get_attention_h_and_mu(self, params, attended, mu_tm1, time_idx):
        """Computes the attention encoding for "one head".

        # Arguments
            params: The MoG params (before activation) for one head.
            attended: The attended sequence (tensor).
            mu_tm1: mu from previous time step (tensor) if self.use_delta is
                True otherwise None.
            time_idx: Time index of the attended (tensor).

        # Returns
            attention_h: The attention encoding for the attention of one head.
            mu: the location(s) of each mixture component for one head.
        """
        def sigma_activation(x):
            return self.sigma_activation(x) + self.sigma_epsilon

        mixture_weights, mu, sigma = [
            activation(params[..., i * self.components:(i + 1) * self.components])
            for i, activation in enumerate(
                [K.softmax, self.mu_activation, sigma_activation])]

        if self.predict_delta_mu:
            mu += mu_tm1

        mixture_weights_, mu_, sigma_ = [
            K.expand_dims(p, 1) for p in [mixture_weights, mu, sigma]]

        attention_w = K.sum(
            mixture_weights_ * K.exp(- sigma_ * K.square(mu_ - time_idx)),
            # NOTE no normalisation was carried out in original paper by A. Graves
            axis=-1,
            keepdims=True
        )
        attention_h = K.sum(attention_w * attended, axis=1)

        return attention_h, mu

    def attention_build(self, input_shape, cell_state_size, attended_shape):
        if not len(attended_shape) == 1:
            raise ValueError('only a single attended supported')
        attended_shape = attended_shape[0]
        if not len(attended_shape) == 3:
            raise ValueError('only support attending tensors with dim=3')

        # NOTE _attention_size must always be set in `attention_build`
        self._attention_size = attended_shape[-1] * self.heads
        mog_in_dim = (input_shape[-1] + cell_state_size[0])
        mog_out_dim = self.heads * self.components * 3
        self.kernel = self.add_weight(
            shape=(mog_in_dim, mog_out_dim),
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.bias = self.add_weight(shape=(mog_out_dim,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

    def get_config(self):
        config = {
            'components': self.components,
            'heads': self.heads,
            'mu_activation': activations.serialize(self.mu_activation),
            'sigma_activation': activations.serialize(self.sigma_activation),
            'sigma_epsilon': self.sigma_epsilon,
            'predict_delta_mu': self.predict_delta_mu,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MixtureOfGaussian1DAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
