from keras.src import ops
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving import serialization_lib


@keras_export("keras.layers.StackedRNNCells")
class StackedRNNCells(Layer):
    """Wrapper allowing a stack of RNN cells to behave as a single cell.

    Used to implement efficient stacked RNNs.

    Args:
      cells: List of RNN cell instances.

    Example:

    ```python
    batch_size = 3
    sentence_length = 5
    num_features = 2
    new_shape = (batch_size, sentence_length, num_features)
    x = np.reshape(np.arange(30), new_shape)

    rnn_cells = [keras.layers.LSTMCell(128) for _ in range(2)]
    stacked_lstm = keras.layers.StackedRNNCells(rnn_cells)
    lstm_layer = keras.layers.RNN(stacked_lstm)

    result = lstm_layer(x)
    ```
    """

    def __init__(self, cells, **kwargs):
        super().__init__(**kwargs)
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

    @property
    def state_size(self):
        return [c.state_size for c in self.cells]

    @property
    def output_size(self):
        if getattr(self.cells[-1], "output_size", None) is not None:
            return self.cells[-1].output_size
        elif isinstance(self.cells[-1].state_size, (list, tuple)):
            return self.cells[-1].state_size[0]
        else:
            return self.cells[-1].state_size

    def get_initial_state(self, batch_size=None):
        initial_states = []
        for cell in self.cells:
            get_initial_state_fn = getattr(cell, "get_initial_state", None)
            if get_initial_state_fn:
                initial_states.append(
                    get_initial_state_fn(batch_size=batch_size)
                )
            else:
                if isinstance(cell.state_size, int):
                    initial_states.append(
                        ops.zeros(
                            (batch_size, cell.state_size),
                            dtype=self.compute_dtype,
                        )
                    )
                else:
                    initial_states.append(
                        [
                            ops.zeros((batch_size, d), dtype=self.compute_dtype)
                            for d in cell.state_size
                        ]
                    )
        return initial_states

    def call(self, inputs, states, training=False, **kwargs):
        # Call the cells in order and store the returned states.
        new_states = []
        for cell, states in zip(self.cells, states):
            state_is_list = tree.is_nested(states)
            states = list(states) if tree.is_nested(states) else [states]
            if isinstance(cell, Layer) and cell._call_has_training_arg:
                kwargs["training"] = training
            else:
                kwargs.pop("training", None)
            cell_call_fn = cell.__call__ if callable(cell) else cell.call
            inputs, states = cell_call_fn(inputs, states, **kwargs)
            if len(states) == 1 and not state_is_list:
                states = states[0]
            new_states.append(states)

        if len(new_states) == 1:
            new_states = new_states[0]
        return inputs, new_states

    def build(self, input_shape):
        for cell in self.cells:
            if isinstance(cell, Layer) and not cell.built:
                cell.build(input_shape)
                cell.built = True
            if getattr(cell, "output_size", None) is not None:
                output_dim = cell.output_size
            elif isinstance(cell.state_size, (list, tuple)):
                output_dim = cell.state_size[0]
            else:
                output_dim = cell.state_size
            batch_size = tree.flatten(input_shape)[0]
            input_shape = (batch_size, output_dim)
        self.built = True

    def get_config(self):
        cells = []
        for cell in self.cells:
            cells.append(serialization_lib.serialize_keras_object(cell))
        config = {"cells": cells}
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        cells = []
        for cell_config in config.pop("cells"):
            cells.append(
                serialization_lib.deserialize_keras_object(
                    cell_config, custom_objects=custom_objects
                )
            )
        return cls(cells, **config)
