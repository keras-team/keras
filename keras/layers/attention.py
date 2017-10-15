from __future__ import division, print_function

import abc

from warnings import warn
from collections import OrderedDict

import numpy as np

from keras import backend as K
from keras.distribution import MixtureOfGaussian1D, DistributionOutputLayer
from keras.engine import InputSpec
from keras.engine import Layer
from keras.layers import concatenate


class MultiLayerWrappingMixin(object):
    """Mixin for using internal layers in arbitrary ways internally in a Layer.

    (Should be inherited first!)
    TODO complete docs.
    """
    @property
    def layers(self):
        if not hasattr(self, '_layers'):
            self._layers = OrderedDict([])
        return self._layers

    @layers.setter
    def layers(self, value):
        raise NotImplementedError(
            'property layers should not be set, use method add_child'
        )

    def add_layer(self, identifier, layer):
        self.layers[identifier] = layer

        return layer

    @property
    def trainable(self):
        return getattr(self, '_trainable', True)

    @trainable.setter
    def trainable(self, value):
        if not value == self.trainable:
            warn('changing trainable property of {} does not modify layers')
            # FIXME how to deal with this, some layers might intentionally
            # not be trainable?

        self._trainable = value

    @property
    def trainable_weights(self):
        if self.trainable:
            return self._trainable_weights + sum(
                [layer.trainable_weights for layer in self.layers.values()],
                []
            )
        else:
            return []

    @property
    def non_trainable_weights(self):
        layers_non_trainable_weights = sum(
            [layer.non_trainable_weights for layer in self.layers.values()],
            []
        )
        if self.trainable:
            return self._non_trainable_weights + layers_non_trainable_weights
        else:
            return (
                self._trainable_weights +
                self._non_trainable_weights +
                layers_non_trainable_weights
            )

    @property
    def trainable_variables(self):
        return self.trainable_weights

    @property
    def non_trainable_variables(self):
        return self.non_trainable_weights

    @property
    def weights(self):
        """Returns the list of all layer variables/weights.
        Returns:
          A list of variables.
        """
        return self.trainable_weights + self.non_trainable_weights

    @property
    def variables(self):
        """Returns the list of all layer variables/weights.
        Returns:
          A list of variables.
        """
        return self.weights

    # TODO implement get/set_weights!


class CellLayerABC(Layer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def call(self, inputs, states, constants=None):
        """
        # Args
            inputs: input tensor
            states: list of state tensor(s)
            constants: list of constant (not time dependent) tensor(s)

        # Returns
            outputs: output tensor
            new_states: updated states
        """
        pass

    @abc.abstractproperty
    def state_size(self):
        pass

    @abc.abstractmethod
    def build(self, input_shape):
        """Builds the cell.
        # Args
            input_shape (tuple | [tuple]): expects shapes of inputs
            followed by constants if passed in RNN.__call__.
        """
        pass


class CellAttentionWrapperABC(MultiLayerWrappingMixin, CellLayerABC):
    """Base class for implementing recurrent attention mechanisms

    TODO docs and example
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, cell,
                 attend_after=False,
                 concatenate_input=False,
                 return_attention=False,
                 **kwargs):

        super(CellAttentionWrapperABC, self).__init__(**kwargs)
        self.add_layer('cell', cell)
        self.attend_after = attend_after
        self.concatenate_input = concatenate_input
        self.return_attention = return_attention
        self.attended_spec = None
        self._attention_size = None

    @abc.abstractmethod
    def attention_call(
        self,
        inputs,
        cell_states,
        attended,
        attention_states,
    ):
        """This method implements the core logic for computing the attention
        representation.
        # Arguments
            inputs: the input at current time step
            cell_states: states for the wrapped RNN cell from previous state
                if attend_after=False otherwise from current time step.
            attended: the same tensor at each timestep
            attention_states: states from previous attention step, by
                default attention from last step but can be extended.

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
    def attention_build(
        self,
        input_shape,
        cell_state_size,
        attended_shape,
    ):
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
    def cell(self):
        return self.layers['cell']

    @property
    def state_size(self):
        """
        # Returns
            tuple of (wrapped) RNN cell state size(s) followed by attention
            state size(s). NOTE important that wrapped cell states are first
            as size of cell output should be same as state_size[0]
        """
        state_size_s = []
        for state_size in [
            self.cell.state_size,
            self.attention_state_size
        ]:
            if hasattr(state_size, '__len__'):
                state_size_s += list(state_size)
            else:
                state_size_s.append(state_size)

        return tuple(state_size_s)

    def call(self, inputs, states, constants=None):
        attended = constants
        if attended is None:
            raise RuntimeError(
                'attended must either be passed in call or set as property'
            )
        cell_states = states[:self._n_wrapped_states]
        attention_states = states[self._n_wrapped_states:]

        if self.attend_after:
            attention_call = self.call_attend_after
        else:
            attention_call = self.call_attend_before

        return attention_call(inputs=inputs,
                              cell_states=cell_states,
                              attended=attended,
                              attention_states=attention_states)

    def call_attend_before(
        self,
        inputs,
        cell_states,
        attended,
        attention_states,
    ):
        attention_h, new_attention_states = self.attention_call(
            inputs=inputs,
            cell_states=cell_states,
            attended=attended,
            attention_states=attention_states)

        if self.concatenate_input:
            cell_input = concatenate([attention_h, inputs])
        else:
            cell_input = attention_h

        output, new_cell_states = self.cell.call(cell_input, cell_states)

        if self.return_attention:
            output = concatenate([output, attention_h])
            # TODO we must have states[0] == output

        return output, new_cell_states + new_attention_states

    def call_attend_after(
        self,
        inputs,
        cell_states,
        attended,
        attention_states
    ):
        attention_h_previous = attention_states[0]

        if self.concatenate_input:
            cell_input = concatenate([attention_h_previous, inputs])
        else:
            cell_input = attention_h_previous

        output, new_cell_states = self.cell.call(cell_input, cell_states)

        attention_h, new_attention_states = self.attention_call(
            inputs=inputs,
            cell_states=new_cell_states,
            attended=attended,
            attention_states=attention_states)

        if self.return_attention:
            output = concatenate([output, attention_h])
            # TODO we must have states[0] == output

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

        if self.return_attention:
            return input_shape[0], cell_output_dim + self.attention_size
        else:
            return input_shape[0], cell_output_dim


class MixtureOfGaussian1DAttention(CellAttentionWrapperABC):

    def __init__(self, cell,
                 n_components,
                 mu_activation=None,
                 sigma_activation=None,
                 use_delta=True):
        super(MixtureOfGaussian1DAttention, self).__init__(cell, )
        self.mog_layer = self.add_layer(
            'mog_out_layer',
            DistributionOutputLayer(
                distribution=MixtureOfGaussian1D(
                    n_components=n_components,
                    mu_activation=mu_activation,
                    sigma_activation=sigma_activation)))
        self.use_delta = use_delta

    @property
    def attention_state_size(self):
        attention_state_size = [self.attention_size]
        if self.use_delta:
            attention_state_size.append(
                self.mog_layer.distribution.n_components)

        return attention_state_size

    def attention_call(self, inputs, cell_states, attended, attention_states):
        mog_input = concatenate([inputs, cell_states[0]])
        mog_params = self.mog_layer(mog_input)
        mixture_weights, mu, sigma, = \
            self.mog_layer.distribution.split_param_types(mog_params)
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

        self._attention_size = attended_shape[-1]
        mog_input_dim = (input_shape[-1] + cell_state_size[0])
        self.mog_layer.build(input_shape=(input_shape[0], mog_input_dim))
