from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.layers.rnn.dropout_rnn_cell import DropoutRNNCell
from keras.src.layers.rnn.rnn import RNN


@keras_export("keras.layers.GRUCell")
class GRUCell(Layer, DropoutRNNCell):
    """Cell class for the GRU layer.

    This class processes one step within the whole time sequence input, whereas
    `keras.layer.GRU` processes the whole sequence.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use. Default: hyperbolic tangent
            (`tanh`). If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use for the recurrent step.
            Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
            applied (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer
            should use a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation
            of the recurrent state. Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Default: `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state. Default: 0.
        reset_after: GRU convention (whether to apply reset gate after or
            before matrix multiplication). False = "before",
            True = "after" (default and cuDNN compatible).
        seed: Random seed for dropout.

    Call arguments:
        inputs: A 2D tensor, with shape `(batch, features)`.
        states: A 2D tensor with shape `(batch, units)`, which is the state
            from the previous time step.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.

    Example:

    >>> inputs = np.random.random((32, 10, 8))
    >>> rnn = keras.layers.RNN(keras.layers.GRUCell(4))
    >>> output = rnn(inputs)
    >>> output.shape
    (32, 4)
    >>> rnn = keras.layers.RNN(
    ...    keras.layers.GRUCell(4),
    ...    return_sequences=True,
    ...    return_state=True)
    >>> whole_sequence_output, final_state = rnn(inputs)
    >>> whole_sequence_output.shape
    (32, 10, 4)
    >>> final_state.shape
    (32, 4)
    """

    def __init__(
        self,
        units,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        reset_after=True,
        seed=None,
        **kwargs,
    ):
        if units <= 0:
            raise ValueError(
                "Received an invalid value for argument `units`, "
                f"expected a positive integer, got {units}."
            )
        implementation = kwargs.pop("implementation", 2)
        super().__init__(**kwargs)
        self.implementation = implementation
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        if self.recurrent_dropout != 0.0:
            self.implementation = 1
        if self.implementation == 1:
            self.dropout_mask_count = 3
        self.seed = seed
        self.seed_generator = backend.random.SeedGenerator(seed=seed)

        self.reset_after = reset_after
        self.state_size = self.units
        self.output_size = self.units

    def build(self, input_shape):
        super().build(input_shape)
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 3),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )

        if self.use_bias:
            if not self.reset_after:
                bias_shape = (3 * self.units,)
            else:
                # separate biases for input and recurrent kernels
                # Note: the shape is intentionally different from CuDNNGRU
                # biases `(2 * 3 * self.units,)`, so that we can distinguish the
                # classes when loading and converting saved weights.
                bias_shape = (2, 3 * self.units)
            self.bias = self.add_weight(
                shape=bias_shape,
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

    def call(self, inputs, states, training=False):
        h_tm1 = (
            states[0] if tree.is_nested(states) else states
        )  # previous state

        if self.use_bias:
            if not self.reset_after:
                input_bias, recurrent_bias = self.bias, None
            else:
                input_bias, recurrent_bias = (
                    ops.squeeze(e, axis=0)
                    for e in ops.split(self.bias, self.bias.shape[0], axis=0)
                )

        if self.implementation == 1:
            if training and 0.0 < self.dropout < 1.0:
                dp_mask = self.get_dropout_mask(inputs)
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs

            x_z = ops.matmul(inputs_z, self.kernel[:, : self.units])
            x_r = ops.matmul(
                inputs_r, self.kernel[:, self.units : self.units * 2]
            )
            x_h = ops.matmul(inputs_h, self.kernel[:, self.units * 2 :])

            if self.use_bias:
                x_z += input_bias[: self.units]
                x_r += input_bias[self.units : self.units * 2]
                x_h += input_bias[self.units * 2 :]

            if training and 0.0 < self.recurrent_dropout < 1.0:
                rec_dp_mask = self.get_recurrent_dropout_mask(h_tm1)
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
            else:
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1

            recurrent_z = ops.matmul(
                h_tm1_z, self.recurrent_kernel[:, : self.units]
            )
            recurrent_r = ops.matmul(
                h_tm1_r, self.recurrent_kernel[:, self.units : self.units * 2]
            )
            if self.reset_after and self.use_bias:
                recurrent_z += recurrent_bias[: self.units]
                recurrent_r += recurrent_bias[self.units : self.units * 2]

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            # reset gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = ops.matmul(
                    h_tm1_h, self.recurrent_kernel[:, self.units * 2 :]
                )
                if self.use_bias:
                    recurrent_h += recurrent_bias[self.units * 2 :]
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = ops.matmul(
                    r * h_tm1_h, self.recurrent_kernel[:, self.units * 2 :]
                )

            hh = self.activation(x_h + recurrent_h)
        else:
            if training and 0.0 < self.dropout < 1.0:
                dp_mask = self.get_dropout_mask(inputs)
                inputs = inputs * dp_mask

            # inputs projected by all gate matrices at once
            matrix_x = ops.matmul(inputs, self.kernel)
            if self.use_bias:
                # biases: bias_z_i, bias_r_i, bias_h_i
                matrix_x += input_bias

            x_z, x_r, x_h = ops.split(matrix_x, 3, axis=-1)

            if self.reset_after:
                # hidden state projected by all gate matrices at once
                matrix_inner = ops.matmul(h_tm1, self.recurrent_kernel)
                if self.use_bias:
                    matrix_inner += recurrent_bias
            else:
                # hidden state projected separately for update/reset and new
                matrix_inner = ops.matmul(
                    h_tm1, self.recurrent_kernel[:, : 2 * self.units]
                )

            recurrent_z = matrix_inner[:, : self.units]
            recurrent_r = matrix_inner[:, self.units : self.units * 2]
            recurrent_h = matrix_inner[:, self.units * 2 :]

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            if self.reset_after:
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = ops.matmul(
                    r * h_tm1, self.recurrent_kernel[:, 2 * self.units :]
                )

            hh = self.activation(x_h + recurrent_h)

        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh
        new_state = [h] if tree.is_nested(states) else h
        return h, new_state

    def get_config(self):
        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(
                self.recurrent_activation
            ),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "reset_after": self.reset_after,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def get_initial_state(self, batch_size=None):
        return [
            ops.zeros((batch_size, self.state_size), dtype=self.compute_dtype)
        ]


@keras_export("keras.layers.GRU")
class GRU(RNN):
    """Gated Recurrent Unit - Cho et al. 2014.

    Based on available runtime hardware and constraints, this layer
    will choose different implementations (cuDNN-based or backend-native)
    to maximize the performance. If a GPU is available and all
    the arguments to the layer meet the requirement of the cuDNN kernel
    (see below for details), the layer will use a fast cuDNN implementation
    when using the TensorFlow backend.

    The requirements to use the cuDNN implementation are:

    1. `activation` == `tanh`
    2. `recurrent_activation` == `sigmoid`
    3. `recurrent_dropout` == 0
    4. `unroll` is `False`
    5. `use_bias` is `True`
    6. `reset_after` is `True`
    7. Inputs, if use masking, are strictly right-padded.
    8. Eager execution is enabled in the outermost context.

    There are two variants of the GRU implementation. The default one is based
    on [v3](https://arxiv.org/abs/1406.1078v3) and has reset gate applied to
    hidden state before matrix multiplication. The other one is based on
    [original](https://arxiv.org/abs/1406.1078v1) and has the order reversed.

    The second variant is compatible with CuDNNGRU (GPU-only) and allows
    inference on CPU. Thus it has separate biases for `kernel` and
    `recurrent_kernel`. To use this variant, set `reset_after=True` and
    `recurrent_activation='sigmoid'`.

    For example:

    >>> inputs = np.random.random((32, 10, 8))
    >>> gru = keras.layers.GRU(4)
    >>> output = gru(inputs)
    >>> output.shape
    (32, 4)
    >>> gru = keras.layers.GRU(4, return_sequences=True, return_state=True)
    >>> whole_sequence_output, final_state = gru(inputs)
    >>> whole_sequence_output.shape
    (32, 10, 4)
    >>> final_state.shape
    (32, 4)

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step.
            Default: sigmoid (`sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer
            should use a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation of the recurrent
            state. Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Default: `None`.
        activity_regularizer: Regularizer function applied to the output of the
            layer (its "activation"). Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Default: `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state. Default: 0.
        seed: Random seed for dropout.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence. Default: `False`.
        return_state: Boolean. Whether to return the last state in addition
            to the output. Default: `False`.
        go_backwards: Boolean (default `False`).
            If `True`, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default: `False`). If `True`, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default: `False`).
            If `True`, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        reset_after: GRU convention (whether to apply reset gate after or
            before matrix multiplication). `False` is `"before"`,
            `True` is `"after"` (default and cuDNN compatible).
        use_cudnn: Whether to use a cuDNN-backed implementation. `"auto"` will
            attempt to use cuDNN when feasible, and will fallback to the
            default implementation if not.

    Call arguments:
        inputs: A 3D tensor, with shape `(batch, timesteps, feature)`.
        mask: Binary tensor of shape `(samples, timesteps)` indicating whether
            a given timestep should be masked  (optional).
            An individual `True` entry indicates that the corresponding timestep
            should be utilized, while a `False` entry indicates that the
            corresponding timestep should be ignored. Defaults to `None`.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. This argument is passed to the
            cell when calling it. This is only relevant if `dropout` or
            `recurrent_dropout` is used  (optional). Defaults to `None`.
        initial_state: List of initial state tensors to be passed to the first
            call of the cell (optional, `None` causes creation
            of zero-filled initial state tensors). Defaults to `None`.
    """

    def __init__(
        self,
        units,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        reset_after=True,
        use_cudnn="auto",
        **kwargs,
    ):
        cell = GRUCell(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            reset_after=reset_after,
            dtype=kwargs.get("dtype", None),
            trainable=kwargs.get("trainable", True),
            name="gru_cell",
            seed=seed,
            implementation=kwargs.pop("implementation", 2),
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            activity_regularizer=activity_regularizer,
            **kwargs,
        )
        self.input_spec = InputSpec(ndim=3)
        if use_cudnn not in ("auto", True, False):
            raise ValueError(
                "Invalid valid received for argument `use_cudnn`. "
                "Expected one of {'auto', True, False}. "
                f"Received: use_cudnn={use_cudnn}"
            )
        self.use_cudnn = use_cudnn
        if (
            backend.backend() == "tensorflow"
            and backend.cudnn_ok(
                cell.activation,
                cell.recurrent_activation,
                self.unroll,
                cell.use_bias,
                reset_after=reset_after,
            )
            and use_cudnn in (True, "auto")
        ):
            self.supports_jit = False

    def inner_loop(self, sequences, initial_state, mask, training=False):
        if tree.is_nested(initial_state):
            initial_state = initial_state[0]
        if tree.is_nested(mask):
            mask = mask[0]
        if self.use_cudnn in ("auto", True):
            if not self.recurrent_dropout:
                try:
                    if training and self.dropout:
                        dp_mask = self.cell.get_dropout_mask(sequences[:, 0, :])
                        dp_mask = ops.expand_dims(dp_mask, axis=1)
                        dp_mask = ops.broadcast_to(
                            dp_mask, ops.shape(sequences)
                        )
                        dp_sequences = sequences * dp_mask
                    else:
                        dp_sequences = sequences
                    # Backends are allowed to specify (optionally) optimized
                    # implementation of the inner GRU loop. In the case of
                    # TF for instance, it will leverage cuDNN when feasible, and
                    # it will raise NotImplementedError otherwise.
                    out = backend.gru(
                        dp_sequences,
                        initial_state,
                        mask,
                        kernel=self.cell.kernel,
                        recurrent_kernel=self.cell.recurrent_kernel,
                        bias=self.cell.bias,
                        activation=self.cell.activation,
                        recurrent_activation=self.cell.recurrent_activation,
                        return_sequences=self.return_sequences,
                        go_backwards=self.go_backwards,
                        unroll=self.unroll,
                        reset_after=self.cell.reset_after,
                    )
                    # We disable jit_compile for the model in this case,
                    # since cuDNN ops aren't XLA compatible.
                    if backend.backend() == "tensorflow":
                        self.supports_jit = False
                    return out
                except NotImplementedError:
                    pass
        if self.use_cudnn is True:
            raise ValueError(
                "use_cudnn=True was specified, "
                "but cuDNN is not supported for this layer configuration "
                "with this backend. Pass use_cudnn='auto' to fallback "
                "to a non-cuDNN implementation."
            )
        return super().inner_loop(
            sequences, initial_state, mask=mask, training=training
        )

    def call(self, sequences, initial_state=None, mask=None, training=False):
        return super().call(
            sequences, mask=mask, training=training, initial_state=initial_state
        )

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def reset_after(self):
        return self.cell.reset_after

    def get_config(self):
        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(
                self.recurrent_activation
            ),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "reset_after": self.reset_after,
            "seed": self.cell.seed,
        }
        base_config = super().get_config()
        del base_config["cell"]
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
