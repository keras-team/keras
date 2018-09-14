# -*- coding: utf-8 -*-
from __future__ import absolute_import

from ..engine import InputSpec
from ..engine import Layer
from ..layers import concatenate
from ..layers import has_arg
from ..utils.generic_utils import to_list


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
        raise NotImplementedError

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
        raise NotImplementedError

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
        state_size_s = to_list(self.cell.state_size, allow_tuple=True)
        state_size_s += to_list(self.attention_state_size, allow_tuple=True)
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

    @property
    def _num_wrapped_states(self):
        return _num_elements(self.cell.state_size)

    @property
    def _num_attention_states(self):
        return _num_elements(self.attention_state_size)

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
        cell_output_dim = to_list(self.cell.state_size, allow_tuple=True)[0]
        return input_shape[0], cell_output_dim

    @property
    def trainable_weights(self):
        layer_weights = super(_RNNAttentionCell, self).trainable_weights
        return layer_weights + self.cell.trainable_weights

    @property
    def non_trainable_weights(self):
        layer_weights = super(_RNNAttentionCell, self).non_trainable_weights
        return layer_weights + self.cell.non_trainable_weights

    def get_config(self):
        config = {'attend_after': self.attend_after,
                  'concatenate_input': self.concatenate_input}

        cell_config = self.cell.get_config()
        config['cell'] = {'class_name': self.cell.__class__.__name__,
                          'config': cell_config}
        base_config = super(_RNNAttentionCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _num_elements(x):
    return len(to_list(x, allow_tuple=True))
