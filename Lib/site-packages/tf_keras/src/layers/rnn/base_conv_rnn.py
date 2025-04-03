# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base class for convolutional-recurrent layers."""


import numpy as np
import tensorflow.compat.v2 as tf

from tf_keras.src import backend
from tf_keras.src.engine import base_layer
from tf_keras.src.engine.input_spec import InputSpec
from tf_keras.src.layers.rnn.base_rnn import RNN
from tf_keras.src.utils import conv_utils
from tf_keras.src.utils import generic_utils
from tf_keras.src.utils import tf_utils


class ConvRNN(RNN):
    """N-Dimensional Base class for convolutional-recurrent layers.

    Args:
      rank: Integer, rank of the convolution, e.g. "2" for 2D convolutions.
      cell: A RNN cell instance. A RNN cell is a class that has: - a
        `call(input_at_t, states_at_t)` method, returning `(output_at_t,
        states_at_t_plus_1)`. The call method of the cell can also take the
        optional argument `constants`, see section "Note on passing external
        constants" below. - a `state_size` attribute. This can be a single
        integer (single state) in which case it is the number of channels of the
        recurrent state (which should be the same as the number of channels of
        the cell output). This can also be a list/tuple of integers (one size
        per state).  In this case, the first entry (`state_size[0]`) should be
        the same as the size of the cell output.
      return_sequences: Boolean. Whether to return the last output. in the
        output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state in addition to the
        output.
      go_backwards: Boolean (default False). If True, process the input sequence
        backwards and return the reversed sequence.
      stateful: Boolean (default False). If True, the last state for each sample
        at index i in a batch will be used as initial state for the sample of
        index i in the following batch.
      input_shape: Use this argument to specify the shape of the input when this
        layer is the first one in a model.
    Call arguments:
      inputs: A (2 + `rank`)D tensor.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether a
        given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is for use with cells that use dropout.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.
      constants: List of constant tensors to be passed to the cell at each
        timestep.
    Input shape:
      (3 + `rank`)D tensor with shape: `(samples, timesteps, channels,
        img_dimensions...)`
      if data_format='channels_first' or shape: `(samples, timesteps,
        img_dimensions..., channels)` if data_format='channels_last'.
    Output shape:
      - If `return_state`: a list of tensors. The first tensor is the output.
        The remaining tensors are the last states,
        each (2 + `rank`)D tensor with shape: `(samples, filters,
          new_img_dimensions...)` if data_format='channels_first'
        or shape: `(samples, new_img_dimensions..., filters)` if
          data_format='channels_last'. img_dimension values might have changed
          due to padding.
      - If `return_sequences`: (3 + `rank`)D tensor with shape: `(samples,
        timesteps, filters, new_img_dimensions...)` if
        data_format='channels_first'
        or shape: `(samples, timesteps, new_img_dimensions..., filters)` if
          data_format='channels_last'.
      - Else, (2 + `rank`)D tensor with shape: `(samples, filters,
        new_img_dimensions...)` if data_format='channels_first'
        or shape: `(samples, new_img_dimensions..., filters)` if
          data_format='channels_last'.
    Masking: This layer supports masking for input data with a variable number
      of timesteps.
    Note on using statefulness in RNNs: You can set RNN layers to be 'stateful',
      which means that the states computed for the samples in one batch will be
      reused as initial states for the samples in the next batch. This assumes a
      one-to-one mapping between samples in different successive batches.
      To enable statefulness: - Specify `stateful=True` in the layer
      constructor.
        - Specify a fixed batch size for your model, by passing
            - If sequential model: `batch_input_shape=(...)` to the first layer
              in your model.
            - If functional model with 1 or more Input layers:
              `batch_shape=(...)` to all the first layers in your model. This is
              the expected shape of your inputs *including the batch size*. It
              should be a tuple of integers, e.g. `(32, 10, 100, 100, 32)`. for
              rank 2 convolution Note that the image dimensions should be
              specified too. - Specify `shuffle=False` when calling fit(). To
              reset the states of your model, call `.reset_states()` on either a
              specific layer, or on your entire model.
    Note on specifying the initial state of RNNs: You can specify the initial
      state of RNN layers symbolically by calling them with the keyword argument
      `initial_state`. The value of `initial_state` should be a tensor or list
      of tensors representing the initial state of the RNN layer. You can
      specify the initial state of RNN layers numerically by calling
      `reset_states` with the keyword argument `states`. The value of `states`
      should be a numpy array or list of numpy arrays representing the initial
      state of the RNN layer.
    Note on passing external constants to RNNs: You can pass "external"
      constants to the cell using the `constants` keyword argument of
      `RNN.__call__` (as well as `RNN.call`) method. This requires that the
      `cell.call` method accepts the same keyword argument `constants`. Such
      constants can be used to condition the cell transformation on additional
      static inputs (not changing over time), a.k.a. an attention mechanism.
    """

    def __init__(
        self,
        rank,
        cell,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        **kwargs,
    ):
        if unroll:
            raise TypeError(
                "Unrolling is not possible with convolutional RNNs. "
                f"Received: unroll={unroll}"
            )
        if isinstance(cell, (list, tuple)):
            # The StackedConvRNN3DCells isn't implemented yet.
            raise TypeError(
                "It is not possible at the moment to"
                "stack convolutional cells. Only pass a single cell "
                "instance as the `cell` argument. Received: "
                f"cell={cell}"
            )
        super().__init__(
            cell,
            return_sequences,
            return_state,
            go_backwards,
            stateful,
            unroll,
            **kwargs,
        )
        self.rank = rank
        self.input_spec = [InputSpec(ndim=rank + 3)]
        self.states = None
        self._num_constants = None

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        cell = self.cell
        if cell.data_format == "channels_first":
            img_dims = input_shape[3:]
        elif cell.data_format == "channels_last":
            img_dims = input_shape[2:-1]

        norm_img_dims = tuple(
            [
                conv_utils.conv_output_length(
                    img_dims[idx],
                    cell.kernel_size[idx],
                    padding=cell.padding,
                    stride=cell.strides[idx],
                    dilation=cell.dilation_rate[idx],
                )
                for idx in range(len(img_dims))
            ]
        )

        if cell.data_format == "channels_first":
            output_shape = input_shape[:2] + (cell.filters,) + norm_img_dims
        elif cell.data_format == "channels_last":
            output_shape = input_shape[:2] + norm_img_dims + (cell.filters,)

        if not self.return_sequences:
            output_shape = output_shape[:1] + output_shape[2:]

        if self.return_state:
            output_shape = [output_shape]
            if cell.data_format == "channels_first":
                output_shape += [
                    (input_shape[0], cell.filters) + norm_img_dims
                    for _ in range(2)
                ]
            elif cell.data_format == "channels_last":
                output_shape += [
                    (input_shape[0],) + norm_img_dims + (cell.filters,)
                    for _ in range(2)
                ]
        return output_shape

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        # Note input_shape will be list of shapes of initial states and
        # constants if these are passed in __call__.
        if self._num_constants is not None:
            constants_shape = input_shape[-self._num_constants :]
        else:
            constants_shape = None

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_spec[0] = InputSpec(
            shape=(batch_size, None) + input_shape[2 : self.rank + 3]
        )

        # allow cell (if layer) to build before we set or validate state_spec
        if isinstance(self.cell, base_layer.Layer):
            step_input_shape = (input_shape[0],) + input_shape[2:]
            if constants_shape is not None:
                self.cell.build([step_input_shape] + constants_shape)
            else:
                self.cell.build(step_input_shape)

        # set or validate state_spec
        if hasattr(self.cell.state_size, "__len__"):
            state_size = list(self.cell.state_size)
        else:
            state_size = [self.cell.state_size]

        if self.state_spec is not None:
            # initial_state was passed in call, check compatibility
            if self.cell.data_format == "channels_first":
                ch_dim = 1
            elif self.cell.data_format == "channels_last":
                ch_dim = self.rank + 1
            if [spec.shape[ch_dim] for spec in self.state_spec] != state_size:
                raise ValueError(
                    "An `initial_state` was passed that is not compatible with "
                    "`cell.state_size`. Received state shapes "
                    f"{[spec.shape for spec in self.state_spec]}. "
                    f"However `cell.state_size` is {self.cell.state_size}"
                )
        else:
            img_dims = tuple((None for _ in range(self.rank)))
            if self.cell.data_format == "channels_first":
                self.state_spec = [
                    InputSpec(shape=(None, dim) + img_dims)
                    for dim in state_size
                ]
            elif self.cell.data_format == "channels_last":
                self.state_spec = [
                    InputSpec(shape=(None,) + img_dims + (dim,))
                    for dim in state_size
                ]
        if self.stateful:
            self.reset_states()
        self.built = True

    def get_initial_state(self, inputs):
        # (samples, timesteps, img_dims..., filters)
        initial_state = backend.zeros_like(inputs)
        # (samples, img_dims..., filters)
        initial_state = backend.sum(initial_state, axis=1)
        shape = list(self.cell.kernel_shape)
        shape[-1] = self.cell.filters
        initial_state = self.cell.input_conv(
            initial_state,
            tf.zeros(tuple(shape), initial_state.dtype),
            padding=self.cell.padding,
        )

        if hasattr(self.cell.state_size, "__len__"):
            return [initial_state for _ in self.cell.state_size]
        else:
            return [initial_state]

    def call(
        self,
        inputs,
        mask=None,
        training=None,
        initial_state=None,
        constants=None,
    ):
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        inputs, initial_state, constants = self._process_inputs(
            inputs, initial_state, constants
        )

        if isinstance(mask, list):
            mask = mask[0]
        timesteps = backend.int_shape(inputs)[1]

        kwargs = {}
        if generic_utils.has_arg(self.cell.call, "training"):
            kwargs["training"] = training

        if constants:
            if not generic_utils.has_arg(self.cell.call, "constants"):
                raise ValueError(
                    f"RNN cell {self.cell} does not support constants. "
                    f"Received: constants={constants}"
                )

            def step(inputs, states):
                constants = states[-self._num_constants :]
                states = states[: -self._num_constants]
                return self.cell.call(
                    inputs, states, constants=constants, **kwargs
                )

        else:

            def step(inputs, states):
                return self.cell.call(inputs, states, **kwargs)

        last_output, outputs, states = backend.rnn(
            step,
            inputs,
            initial_state,
            constants=constants,
            go_backwards=self.go_backwards,
            mask=mask,
            input_length=timesteps,
            return_all_outputs=self.return_sequences,
        )
        if self.stateful:
            updates = [
                backend.update(self_state, state)
                for self_state, state in zip(self.states, states)
            ]
            self.add_update(updates)

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        if self.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return [output] + states
        return output

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError("Layer must be stateful.")
        input_shape = self.input_spec[0].shape
        state_shape = self.compute_output_shape(input_shape)
        if self.return_state:
            state_shape = state_shape[0]
        if self.return_sequences:
            state_shape = state_shape[:1].concatenate(state_shape[2:])
        if None in state_shape:
            raise ValueError(
                "If a RNN is stateful, it needs to know "
                "its batch size. Specify the batch size "
                "of your input tensors: \n"
                "- If using a Sequential model, "
                "specify the batch size by passing "
                "a `batch_input_shape` "
                "argument to your first layer.\n"
                "- If using the functional API, specify "
                "the time dimension by passing a "
                "`batch_shape` argument to your Input layer.\n"
                "The same thing goes for the number of rows and "
                "columns."
            )

        # helper function
        def get_tuple_shape(nb_channels):
            result = list(state_shape)
            if self.cell.data_format == "channels_first":
                result[1] = nb_channels
            elif self.cell.data_format == "channels_last":
                result[self.rank + 1] = nb_channels
            else:
                raise KeyError(
                    "Cell data format must be one of "
                    '{"channels_first", "channels_last"}. Received: '
                    f"cell.data_format={self.cell.data_format}"
                )
            return tuple(result)

        # initialize state if None
        if self.states[0] is None:
            if hasattr(self.cell.state_size, "__len__"):
                self.states = [
                    backend.zeros(get_tuple_shape(dim))
                    for dim in self.cell.state_size
                ]
            else:
                self.states = [
                    backend.zeros(get_tuple_shape(self.cell.state_size))
                ]
        elif states is None:
            if hasattr(self.cell.state_size, "__len__"):
                for state, dim in zip(self.states, self.cell.state_size):
                    backend.set_value(state, np.zeros(get_tuple_shape(dim)))
            else:
                backend.set_value(
                    self.states[0],
                    np.zeros(get_tuple_shape(self.cell.state_size)),
                )
        else:
            if not isinstance(states, (list, tuple)):
                states = [states]
            if len(states) != len(self.states):
                raise ValueError(
                    f"Layer {self.name} expects {len(self.states)} states, "
                    f"but it received {len(states)} state values. "
                    f"States received: {states}"
                )
            for index, (value, state) in enumerate(zip(states, self.states)):
                if hasattr(self.cell.state_size, "__len__"):
                    dim = self.cell.state_size[index]
                else:
                    dim = self.cell.state_size
                if value.shape != get_tuple_shape(dim):
                    raise ValueError(
                        "State {index} is incompatible with layer "
                        f"{self.name}: expected shape={get_tuple_shape(dim)}, "
                        f"found shape={value.shape}"
                    )
                backend.set_value(state, value)

