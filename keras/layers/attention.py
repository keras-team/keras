from __future__ import division, print_function


import abc

from keras.engine import Layer
from keras.layers import Wrapper, concatenate


class CellWithConstantsLayerABC(object):

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
            input_shape (tuple | [tuple]): will contain shapes of initial
                states and constants if passed in __call__.
        """
        pass


class AttentionCellBase(Wrapper):
    """

    """
    def __init__(
        self,
        units,
        cell,
        attend_after=False,
        concatenate_input=False,
        return_attention=False,
        **kwargs
    ):
        super(AttentionCellBase, self).__init__(layer=cell, **kwargs)
        self.units = units
        self.attend_after = attend_after
        self.concatenate_input = concatenate_input
        self.return_attention = return_attention

        # set either in call or by setting attended property
        self._attended = None

    def attention_state_size(self):
        """Declares size of attention states.

        Returns:
            int or list of int. If the attention mechanism arr using multiple
            states, the first should always be the attention encoding, i.e.
            have size `units`
        """
        return self.units

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
        pass

    @property
    def cell(self):
        return self.layer

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
                state_size += list(state_size)
            else:
                state_size.append(state_size)

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

        self.attention_build(
            input_shape=input_shape,
            cell_state_size=self.cell.state_size,
            attended_shape=attended_shape,
        )
        if isinstance(self.cell, Layer):
            cell_input_shape = (input_shape[0],
                                self.units + input_shape[-1]
                                if self.concatenate_input else self.units)
            self.cell.build(cell_input_shape)

        self.built = True

    def compute_output_shape(self, input_shape):
        if hasattr(self.cell.state_size, '__len__'):
            cell_output_dim = self.cell.state_size[0]
        else:
            cell_output_dim = self.cell.state_size

        if self.return_attention:
            return input_shape[0], cell_output_dim + self.units
        else:
            return input_shape[0], cell_output_dim


def to_list_or_none(x):
    if x is None or isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def get_shape(inputs):
    # TODO duplicates code in Layer...
    if isinstance(inputs, list):
        xs = inputs
        return_list = True
    else:
        xs = [inputs]
        return_list = False

    inputs_shape = []
    for x in xs:
        if hasattr(x, '_keras_shape'):
            inputs_shape.append(x._keras_shape)
        elif hasattr(K, 'int_shape'):
            inputs_shape.append(K.int_shape(x))
        else:
            raise ValueError('cannot infer shape of {}'.format(x))
    if return_list:
        return inputs_shape
    else:
        # must be only one
        return inputs_shape[0]