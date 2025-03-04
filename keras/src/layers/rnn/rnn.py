from keras.src import backend
from keras.src import ops
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.layers.rnn.dropout_rnn_cell import DropoutRNNCell
from keras.src.layers.rnn.stacked_rnn_cells import StackedRNNCells
from keras.src.saving import serialization_lib
from keras.src.utils import tracking


@keras_export("keras.layers.RNN")
class RNN(Layer):
    """Base class for recurrent layers.

    Args:
        cell: A RNN cell instance or a list of RNN cell instances.
            A RNN cell is a class that has:
            - A `call(input_at_t, states_at_t)` method, returning
            `(output_at_t, states_at_t_plus_1)`. The call method of the
            cell can also take the optional argument `constants`, see
            section "Note on passing external constants" below.
            - A `state_size` attribute. This can be a single integer
            (single state) in which case it is the size of the recurrent
            state. This can also be a list/tuple of integers
            (one size per state).
            - A `output_size` attribute, a single integer.
            - A `get_initial_state(batch_size=None)`
            method that creates a tensor meant to be fed to `call()` as the
            initial state, if the user didn't specify any initial state
            via other means. The returned initial state should have
            shape `(batch_size, cell.state_size)`.
            The cell might choose to create a tensor full of zeros,
            or other values based on the cell's implementation.
            `inputs` is the input tensor to the RNN layer, with shape
            `(batch_size, timesteps, features)`.
            If this method is not implemented
            by the cell, the RNN layer will create a zero filled tensor
            with shape `(batch_size, cell.state_size)`.
            In the case that `cell` is a list of RNN cell instances, the cells
            will be stacked on top of each other in the RNN, resulting in an
            efficient stacked RNN.
        return_sequences: Boolean (default `False`). Whether to return the last
            output in the output sequence, or the full sequence.
        return_state: Boolean (default `False`).
            Whether to return the last state in addition to the output.
        go_backwards: Boolean (default `False`).
            If `True`, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default `False`). If True, the last state
            for each sample at index `i` in a batch will be used as initial
            state for the sample of index `i` in the following batch.
        unroll: Boolean (default `False`).
            If True, the network will be unrolled, else a symbolic loop will be
            used. Unrolling can speed-up a RNN, although it tends to be more
            memory-intensive. Unrolling is only suitable for short sequences.
        zero_output_for_mask: Boolean (default `False`).
            Whether the output should use zeros for the masked timesteps.
            Note that this field is only used when `return_sequences`
            is `True` and `mask` is provided.
            It can useful if you want to reuse the raw output sequence of
            the RNN without interference from the masked timesteps, e.g.,
            merging bidirectional RNNs.

    Call arguments:
        sequences: A 3-D tensor with shape `(batch_size, timesteps, features)`.
        initial_state: List of initial state tensors to be passed to the first
            call of the cell.
        mask: Binary tensor of shape `[batch_size, timesteps]`
            indicating whether a given timestep should be masked.
            An individual `True` entry indicates that the corresponding
            timestep should be utilized, while a `False` entry indicates
            that the corresponding timestep should be ignored.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. This argument is passed
            to the cell when calling it.
            This is for use with cells that use dropout.

    Output shape:

    - If `return_state`: a list of tensors. The first tensor is
    the output. The remaining tensors are the last states,
    each with shape `(batch_size, state_size)`, where `state_size` could
    be a high dimension tensor shape.
    - If `return_sequences`: 3D tensor with shape
    `(batch_size, timesteps, output_size)`.

    Masking:

    This layer supports masking for input data with a variable number
    of timesteps. To introduce masks to your data,
    use a `keras.layers.Embedding` layer with the `mask_zero` parameter
    set to `True`.

    Note on using statefulness in RNNs:

    You can set RNN layers to be 'stateful', which means that the states
    computed for the samples in one batch will be reused as initial states
    for the samples in the next batch. This assumes a one-to-one mapping
    between samples in different successive batches.

    To enable statefulness:

    - Specify `stateful=True` in the layer constructor.
    - Specify a fixed batch size for your model, by passing
        `batch_size=...` to the `Input` layer(s) of your model.
        Remember to also specify the same `batch_size=...` when
        calling `fit()`, or otherwise use a generator-like
        data source like a `keras.utils.PyDataset` or a
        `tf.data.Dataset`.
    - Specify `shuffle=False` when calling `fit()`, since your
        batches are expected to be temporally ordered.

    To reset the states of your model, call `.reset_state()` on either
    a specific layer, or on your entire model.

    Note on specifying the initial state of RNNs:

    You can specify the initial state of RNN layers symbolically by
    calling them with the keyword argument `initial_state`. The value of
    `initial_state` should be a tensor or list of tensors representing
    the initial state of the RNN layer.

    You can specify the initial state of RNN layers numerically by
    calling `reset_state()` with the keyword argument `states`. The value of
    `states` should be a numpy array or list of numpy arrays representing
    the initial state of the RNN layer.

    Examples:

    ```python
    from keras.layers import RNN
    from keras import ops

    # First, let's define a RNN Cell, as a layer subclass.
    class MinimalRNNCell(keras.Layer):

        def __init__(self, units, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.state_size = units

        def build(self, input_shape):
            self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                          initializer='uniform',
                                          name='kernel')
            self.recurrent_kernel = self.add_weight(
                shape=(self.units, self.units),
                initializer='uniform',
                name='recurrent_kernel')

        def call(self, inputs, states):
            prev_output = states[0]
            h = ops.matmul(inputs, self.kernel)
            output = h + ops.matmul(prev_output, self.recurrent_kernel)
            return output, [output]

    # Let's use this cell in a RNN layer:

    cell = MinimalRNNCell(32)
    x = keras.Input((None, 5))
    layer = RNN(cell)
    y = layer(x)

    # Here's how to use the cell to build a stacked RNN:

    cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
    x = keras.Input((None, 5))
    layer = RNN(cells)
    y = layer(x)
    ```
    """

    def __init__(
        self,
        cell,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        zero_output_for_mask=False,
        **kwargs,
    ):
        if isinstance(cell, (list, tuple)):
            cell = StackedRNNCells(cell)
        if "call" not in dir(cell):
            raise ValueError(
                "Argument `cell` should have a `call` method. "
                f"Received: cell={cell}"
            )
        if "state_size" not in dir(cell):
            raise ValueError(
                "The RNN cell should have a `state_size` attribute "
                "(single integer or list of integers, "
                "one integer per RNN state). "
                f"Received: cell={cell}"
            )
        super().__init__(**kwargs)

        # If True, the output for masked timestep will be zeros, whereas in the
        # False case, output from previous timestep is returned for masked
        # timestep.
        self.zero_output_for_mask = zero_output_for_mask
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll

        self.supports_masking = True
        self.input_spec = None
        self.states = None

        state_size = getattr(self.cell, "state_size", None)
        if state_size is None:
            raise ValueError(
                "state_size must be specified as property on the RNN cell."
            )
        if not isinstance(state_size, (list, tuple, int)):
            raise ValueError(
                "state_size must be an integer, or a list/tuple of integers "
                "(one for each state tensor)."
            )
        if isinstance(state_size, int):
            self.state_size = [state_size]
            self.single_state = True
        else:
            self.state_size = list(state_size)
            self.single_state = False

    def compute_output_shape(self, sequences_shape, initial_state_shape=None):
        batch_size = sequences_shape[0]
        length = sequences_shape[1]
        states_shape = []
        for state_size in self.state_size:
            if isinstance(state_size, int):
                states_shape.append((batch_size, state_size))
            elif isinstance(state_size, (list, tuple)):
                states_shape.append([(batch_size, s) for s in state_size])

        output_size = getattr(self.cell, "output_size", None)
        if output_size is None:
            output_size = self.state_size[0]
        if not isinstance(output_size, int):
            raise ValueError("output_size must be an integer.")
        if self.return_sequences:
            output_shape = (batch_size, length, output_size)
        else:
            output_shape = (batch_size, output_size)
        if self.return_state:
            return output_shape, *states_shape
        return output_shape

    def compute_mask(self, _, mask):
        # Time step masks must be the same for each input.
        # This is because the mask for an RNN is of size [batch, time_steps, 1],
        # and specifies which time steps should be skipped, and a time step
        # must be skipped for all inputs.
        mask = tree.flatten(mask)[0]
        output_mask = mask if self.return_sequences else None
        if self.return_state:
            state_mask = [None for _ in self.state_size]
            return [output_mask] + state_mask
        else:
            return output_mask

    def build(self, sequences_shape, initial_state_shape=None):
        # Build cell (if layer).
        step_input_shape = (sequences_shape[0],) + tuple(sequences_shape[2:])
        if isinstance(self.cell, Layer) and not self.cell.built:
            self.cell.build(step_input_shape)
            self.cell.built = True
        if self.stateful:
            if self.states is not None:
                self.reset_state()
            else:
                if sequences_shape[0] is None:
                    raise ValueError(
                        "When using `stateful=True` in a RNN, the "
                        "batch size must be static. Found dynamic "
                        f"batch size: sequence.shape={sequences_shape}"
                    )
                self._create_state_variables(sequences_shape[0])

    @tracking.no_automatic_dependency_tracking
    def _create_state_variables(self, batch_size):
        with backend.name_scope(self.name, caller=self):
            self.states = tree.map_structure(
                lambda value: backend.Variable(
                    value,
                    trainable=False,
                    dtype=self.variable_dtype,
                    name="rnn_state",
                ),
                self.get_initial_state(batch_size),
            )

    def get_initial_state(self, batch_size):
        get_initial_state_fn = getattr(self.cell, "get_initial_state", None)
        if get_initial_state_fn:
            init_state = get_initial_state_fn(batch_size=batch_size)
        else:
            return [
                ops.zeros((batch_size, d), dtype=self.cell.compute_dtype)
                for d in self.state_size
            ]

        # RNN expect the states in a list, even if single state.
        if not tree.is_nested(init_state):
            init_state = [init_state]
        # Force the state to be a list in case it is a namedtuple eg
        # LSTMStateTuple.
        return list(init_state)

    def reset_states(self):
        # Compatibility alias.
        self.reset_state()

    def reset_state(self):
        if self.states is not None:
            for v in self.states:
                v.assign(ops.zeros_like(v))

    def inner_loop(self, sequences, initial_state, mask, training=False):
        cell_kwargs = {}
        if isinstance(self.cell, Layer) and self.cell._call_has_training_arg:
            cell_kwargs["training"] = training

        def step(inputs, states):
            # Create new tensor copies when using PyTorch backend
            # with stateful=True. This prevents in-place modifications
            # that would otherwise break PyTorch's autograd functionality
            # by modifying tensors needed for gradient computation.
            if backend.backend() == "torch" and self.stateful:
                states = tree.map_structure(ops.copy, states)
            output, new_states = self.cell(inputs, states, **cell_kwargs)
            if not tree.is_nested(new_states):
                new_states = [new_states]
            return output, new_states

        if not tree.is_nested(initial_state):
            initial_state = [initial_state]

        return backend.rnn(
            step,
            sequences,
            initial_state,
            go_backwards=self.go_backwards,
            mask=mask,
            unroll=self.unroll,
            input_length=sequences.shape[1],
            zero_output_for_mask=self.zero_output_for_mask,
            return_all_outputs=self.return_sequences,
        )

    def call(
        self,
        sequences,
        initial_state=None,
        mask=None,
        training=False,
    ):
        timesteps = sequences.shape[1]
        if self.unroll and timesteps is None:
            raise ValueError(
                "Cannot unroll a RNN if the "
                "time dimension is undefined. \n"
                "- If using a Sequential model, "
                "specify the time dimension by passing "
                "an `Input()` as your first layer.\n"
                "- If using the functional API, specify "
                "the time dimension by passing a `shape` "
                "or `batch_shape` argument to your `Input()`."
            )

        if initial_state is None:
            if self.stateful:
                initial_state = self.states
            else:
                initial_state = self.get_initial_state(
                    batch_size=ops.shape(sequences)[0]
                )
        # RNN expect the states in a list, even if single state.
        if not tree.is_nested(initial_state):
            initial_state = [initial_state]
        initial_state = list(initial_state)

        # Cast states to compute dtype.
        # Note that states may be deeply nested
        # (e.g. in the stacked cells case).
        initial_state = tree.map_structure(
            lambda x: backend.convert_to_tensor(
                x, dtype=self.cell.compute_dtype
            ),
            initial_state,
        )

        # Prepopulate the dropout state so that the inner_loop is stateless
        # this is particularly important for JAX backend.
        self._maybe_config_dropout_masks(
            self.cell, sequences[:, 0, :], initial_state
        )

        last_output, outputs, states = self.inner_loop(
            sequences=sequences,
            initial_state=initial_state,
            mask=mask,
            training=training,
        )
        last_output = ops.cast(last_output, self.compute_dtype)
        outputs = ops.cast(outputs, self.compute_dtype)
        states = tree.map_structure(
            lambda x: ops.cast(x, dtype=self.compute_dtype), states
        )
        self._maybe_reset_dropout_masks(self.cell)

        if self.stateful:
            for self_state, state in zip(
                tree.flatten(self.states), tree.flatten(states)
            ):
                self_state.assign(state)

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        if self.return_state:
            return output, *states
        return output

    def _maybe_config_dropout_masks(self, cell, input_sequence, input_state):
        state = (
            input_state[0]
            if isinstance(input_state, (list, tuple))
            else input_state
        )
        if isinstance(cell, DropoutRNNCell):
            cell.get_dropout_mask(input_sequence)
            cell.get_recurrent_dropout_mask(state)
        if isinstance(cell, StackedRNNCells):
            for c, s in zip(cell.cells, input_state):
                self._maybe_config_dropout_masks(c, input_sequence, s)
                # Replicate the behavior of `StackedRNNCells.call` to compute
                # the inputs for the next cell.
                s = list(s) if tree.is_nested(s) else [s]
                cell_call_fn = c.__call__ if callable(c) else c.call
                input_sequence, _ = cell_call_fn(input_sequence, s)

    def _maybe_reset_dropout_masks(self, cell):
        if isinstance(cell, DropoutRNNCell):
            cell.reset_dropout_mask()
            cell.reset_recurrent_dropout_mask()
        if isinstance(cell, StackedRNNCells):
            for c in cell.cells:
                self._maybe_reset_dropout_masks(c)

    def get_config(self):
        config = {
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
            "go_backwards": self.go_backwards,
            "stateful": self.stateful,
            "unroll": self.unroll,
            "zero_output_for_mask": self.zero_output_for_mask,
        }
        config["cell"] = serialization_lib.serialize_keras_object(self.cell)
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        cell = serialization_lib.deserialize_keras_object(
            config.pop("cell"), custom_objects=custom_objects
        )
        layer = cls(cell, **config)
        return layer
