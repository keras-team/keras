# -*- coding: utf-8 -*-
from __future__ import absolute_import

import abc

from .. import backend as K
from ..distribution import MixtureOfGaussian1D
from ..engine import InputSpec
from ..engine import Layer
from ..layers import concatenate
from ..layers import has_arg
from .. import initializers
from .. import regularizers
from .. import constraints


class CellAttentionWrapperABC(Layer):
    """Base class for implementing recurrent attention mechanisms.

    TODO docs and example
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, cell,
                 attend_after=False,
                 concatenate_input=False,
                 **kwargs):
        self.cell = cell  # must be set before calling super
        super(CellAttentionWrapperABC, self).__init__(**kwargs)
        self.attend_after = attend_after
        self.concatenate_input = concatenate_input
        self.attended_spec = None
        self._attention_size = None

    @abc.abstractmethod
    def attention_call(self,
                       inputs,
                       cell_states,
                       attended,
                       attention_states,
                       training=None):
        """This method implements the core logic for computing the attention
        representation.
        # Arguments
            inputs: the input at current time step
            cell_states: states for the wrapped RNN cell from previous state
                if attend_after=False otherwise from current time step.
            attended: the same tensor at each timestep
            attention_states: states from previous attention step, by
                default attention from last step but can be extended.
            training: whether run in training mode or not

        # Returns
            attention_h: the computed attention representation at current
                timestep
            attention_states: states to be passed to next attention_step, by
                default this is just [attention_h]. NOTE if more states are
                used, these should be _appended_ to attention states,
                attention_states[0] should always be attention_h.
        """
        pass

    @abc.abstractmethod
    def attention_build(self, input_shape, cell_state_size, attended_shape):
        """Build the attention mechanism.

        NOTE: should set self._attention_size (unless attention_size property
        is over implemented).

        # Arguments
            input_shape (tuple(int)):
            cell_state_size ([tuple(int)]): note: always list
            attended_shape ([tuple(int)]): note: always list
        """
        pass

    @property
    def attention_size(self):
        """Should return size off attention encoding (int).
        """
        return self._attention_size

    @property
    def attention_state_size(self):
        """Declares size of attention states.

        Returns:
            int or list of int. If the attention mechanism arr using multiple
            states, the first should always be the attention encoding, i.e.
            have size `units`
        """
        return self.attention_size

    @property
    def state_size(self):
        """
        # Returns
            tuple of (wrapped) RNN cell state size(s) followed by attention
            state size(s). NOTE important that wrapped cell states are first
            as size of cell output should be same as state_size[0]
        """
        state_size_s = []
        for state_size in [self.cell.state_size, self.attention_state_size]:
            if hasattr(state_size, '__len__'):
                state_size_s += list(state_size)
            else:
                state_size_s.append(state_size)

        return tuple(state_size_s)

    def call(self, inputs, states, constants, training=None):
        attended = constants
        cell_states = states[:self._n_wrapped_states]
        attention_states = states[self._n_wrapped_states:]

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

    @property
    def _n_wrapped_states(self):
        if hasattr(self.cell.state_size, '__len__'):
            return len(self.cell.state_size)
        else:
            return 1

    @property
    def _n_attention_states(self):
        if hasattr(self.attention_state_size, '__len__'):
            return len(self.attention_state_size)
        else:
            return 1

    def build(self, input_shape):
        """Builds attention mechanism and wrapped cell.

        Arguments:
            input_shape: list of tuples of integers, the input feature shape
                (inputs sequence shape without time dimension) followed by
                attended shapes.
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
        return super(CellAttentionWrapperABC, self).trainable_weights + \
               self.cell.trainable_weights

    @property
    def non_trainable_weights(self):
        return super(CellAttentionWrapperABC, self).non_trainable_weights + \
               self.cell.non_trainable_weights

    def get_config(self):
        config = {'attend_after': self.attend_after,
                  'concatenate_input': self.concatenate_input}

        cell_config = self.cell.get_config()
        config['cell'] = {'class_name': self.cell.__class__.__name__,
                          'config': cell_config}
        base_config = super(CellAttentionWrapperABC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MixtureOfGaussian1DAttention(CellAttentionWrapperABC):

    def __init__(self, cell,
                 n_components,
                 mu_activation=None,
                 sigma_activation=None,
                 use_delta=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MixtureOfGaussian1DAttention, self).__init__(cell, **kwargs)
        self.distribution = MixtureOfGaussian1D(
            n_components=n_components,
            mu_activation=mu_activation,
            sigma_activation=sigma_activation)
        self.use_delta = use_delta
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    @property
    def attention_state_size(self):
        attention_state_size = [self.attention_size]
        if self.use_delta:
            attention_state_size.append(self.distribution.n_components)

        return attention_state_size

    def attention_call(self,
                       inputs,
                       cell_states,
                       attended,
                       attention_states,
                       training=None):
        mog_input = concatenate([inputs, cell_states[0]])
        mog_params = self.distribution.activation(
            K.bias_add(K.dot(mog_input, self.kernel), self.bias))

        mixture_weights, mu, sigma, = \
            self.distribution.split_param_types(mog_params)
        if self.use_delta:
            mu_tm1 = attention_states[1]
            mu += mu_tm1
        mixture_weights_, mu_, sigma_ = [
            K.expand_dims(p, 1) for p in [mixture_weights, mu, sigma]]

        time_idx = K.arange(K.shape(attended[0])[1], dtype='float32')
        time_idx = K.expand_dims(K.expand_dims(time_idx, 0), -1)
        attention_w = K.sum(
            mixture_weights_ * K.exp(- sigma_ * K.square(mu_ - time_idx)),
            # TODO normalisation needed?
            axis=-1,
            keepdims=True
        )
        attention_h = K.sum(attention_w * attended[0], axis=1)
        new_attention_states = [attention_h]
        if self.use_delta:
            new_attention_states.append(mu)

        return attention_h, new_attention_states

    def attention_build(self, input_shape, cell_state_size, attended_shape):
        if not len(attended_shape) == 1:
            raise ValueError('only a single attended supported')
        attended_shape = attended_shape[0]
        if not len(attended_shape) == 3:
            raise ValueError('only support attending tensors with dim=3')

        # NOTE _attention_size must always be set in `attention_build`
        self._attention_size = attended_shape[-1]

        mog_input_dim = (input_shape[-1] + cell_state_size[0])

        self.kernel = self.add_weight(
            shape=(mog_input_dim, self.distribution.n_params),
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        self.bias = self.add_weight(shape=(self.distribution.n_params,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

    def get_config(self):
        pass  # TODO
