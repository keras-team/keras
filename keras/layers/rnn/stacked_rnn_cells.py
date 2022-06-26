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
"""Wrapper allowing a stack of RNN cells to behave as a single cell."""


import functools

import tensorflow.compat.v2 as tf

from keras import backend
from keras.engine import base_layer
from keras.layers.rnn import rnn_utils
from keras.utils import generic_utils
from keras.utils import tf_utils

# isort: off
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.StackedRNNCells")
class StackedRNNCells(base_layer.Layer):
    """Wrapper allowing a stack of RNN cells to behave as a single cell.

    Used to implement efficient stacked RNNs.

    Args:
      cells: List of RNN cell instances.

    Examples:

    ```python
    batch_size = 3
    sentence_max_length = 5
    n_features = 2
    new_shape = (batch_size, sentence_max_length, n_features)
    x = tf.constant(np.reshape(np.arange(30), new_shape), dtype = tf.float32)

    rnn_cells = [tf.keras.layers.LSTMCell(128) for _ in range(2)]
    stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
    lstm_layer = tf.keras.layers.RNN(stacked_lstm)

    result = lstm_layer(x)
    ```
    """

    def __init__(self, cells, **kwargs):
        for cell in cells:
            if "call" not in dir(cell):
                raise ValueError(
                    "All cells must have a `call` method. "
                    f"Received cell without a `call` method: {cell}"
                )
            if "state_size" not in dir(cell):
                raise ValueError(
                    "All cells must have a `state_size` attribute. "
                    f"Received cell without a `state_size`: {cell}"
                )
        self.cells = cells
        # reverse_state_order determines whether the state size will be in a
        # reverse order of the cells' state. User might want to set this to True
        # to keep the existing behavior. This is only useful when use
        # RNN(return_state=True) since the state will be returned as the same
        # order of state_size.
        self.reverse_state_order = kwargs.pop("reverse_state_order", False)
        if self.reverse_state_order:
            logging.warning(
                "reverse_state_order=True in StackedRNNCells will soon "
                "be deprecated. Please update the code to work with the "
                "natural order of states if you rely on the RNN states, "
                "eg RNN(return_state=True)."
            )
        super().__init__(**kwargs)

    @property
    def state_size(self):
        return tuple(
            c.state_size
            for c in (
                self.cells[::-1] if self.reverse_state_order else self.cells
            )
        )

    @property
    def output_size(self):
        if getattr(self.cells[-1], "output_size", None) is not None:
            return self.cells[-1].output_size
        elif rnn_utils.is_multiple_state(self.cells[-1].state_size):
            return self.cells[-1].state_size[0]
        else:
            return self.cells[-1].state_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        initial_states = []
        for cell in (
            self.cells[::-1] if self.reverse_state_order else self.cells
        ):
            get_initial_state_fn = getattr(cell, "get_initial_state", None)
            if get_initial_state_fn:
                initial_states.append(
                    get_initial_state_fn(
                        inputs=inputs, batch_size=batch_size, dtype=dtype
                    )
                )
            else:
                initial_states.append(
                    rnn_utils.generate_zero_filled_state_for_cell(
                        cell, inputs, batch_size, dtype
                    )
                )

        return tuple(initial_states)

    def call(self, inputs, states, constants=None, training=None, **kwargs):
        # Recover per-cell states.
        state_size = (
            self.state_size[::-1]
            if self.reverse_state_order
            else self.state_size
        )
        nested_states = tf.nest.pack_sequence_as(
            state_size, tf.nest.flatten(states)
        )

        # Call the cells in order and store the returned states.
        new_nested_states = []
        for cell, states in zip(self.cells, nested_states):
            states = states if tf.nest.is_nested(states) else [states]
            # TF cell does not wrap the state into list when there is only one
            # state.
            is_tf_rnn_cell = getattr(cell, "_is_tf_rnn_cell", None) is not None
            states = (
                states[0] if len(states) == 1 and is_tf_rnn_cell else states
            )
            if generic_utils.has_arg(cell.call, "training"):
                kwargs["training"] = training
            else:
                kwargs.pop("training", None)
            # Use the __call__ function for callable objects, eg layers, so that
            # it will have the proper name scopes for the ops, etc.
            cell_call_fn = cell.__call__ if callable(cell) else cell.call
            if generic_utils.has_arg(cell.call, "constants"):
                inputs, states = cell_call_fn(
                    inputs, states, constants=constants, **kwargs
                )
            else:
                inputs, states = cell_call_fn(inputs, states, **kwargs)
            new_nested_states.append(states)

        return inputs, tf.nest.pack_sequence_as(
            state_size, tf.nest.flatten(new_nested_states)
        )

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        def get_batch_input_shape(batch_size, dim):
            shape = tf.TensorShape(dim).as_list()
            return tuple([batch_size] + shape)

        for cell in self.cells:
            if isinstance(cell, base_layer.Layer) and not cell.built:
                with backend.name_scope(cell.name):
                    cell.build(input_shape)
                    cell.built = True
            if getattr(cell, "output_size", None) is not None:
                output_dim = cell.output_size
            elif rnn_utils.is_multiple_state(cell.state_size):
                output_dim = cell.state_size[0]
            else:
                output_dim = cell.state_size
            batch_size = tf.nest.flatten(input_shape)[0]
            if tf.nest.is_nested(output_dim):
                input_shape = tf.nest.map_structure(
                    functools.partial(get_batch_input_shape, batch_size),
                    output_dim,
                )
                input_shape = tuple(input_shape)
            else:
                input_shape = tuple(
                    [batch_size] + tf.TensorShape(output_dim).as_list()
                )
        self.built = True

    def get_config(self):
        cells = []
        for cell in self.cells:
            cells.append(generic_utils.serialize_keras_object(cell))
        config = {"cells": cells}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from keras.layers import deserialize as deserialize_layer

        cells = []
        for cell_config in config.pop("cells"):
            cells.append(
                deserialize_layer(cell_config, custom_objects=custom_objects)
            )
        return cls(cells, **config)
