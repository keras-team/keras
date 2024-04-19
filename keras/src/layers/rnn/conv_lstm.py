from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src import tree
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.layers.rnn.dropout_rnn_cell import DropoutRNNCell
from keras.src.layers.rnn.rnn import RNN
from keras.src.ops import operation_utils
from keras.src.utils import argument_validation


class ConvLSTMCell(Layer, DropoutRNNCell):
    """Cell class for the ConvLSTM layer.

    Args:
        rank: Integer, rank of the convolution, e.g. "2" for 2D convolutions.
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides: An integer or tuple/list of n integers, specifying the strides
            of the convolution. Specifying any stride value != 1
            is incompatible with specifying any `dilation_rate` value != 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly
            to the left/right or up/down of the input such that output
            has the same height/width dimension as the input.
        data_format: A string, one of `channels_last` (default) or
            `channels_first`. When unspecified, uses
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json` (if exists) else 'channels_last'.
            Defaults to `'channels_last'`.
        dilation_rate: An integer or tuple/list of n integers, specifying the
            dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function. If `None`, no activation is applied.
        recurrent_activation: Activation function to use for the recurrent step.
        use_bias: Boolean, (default `True`), whether the layer
            should use a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation of the recurrent
            state. Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        unit_forget_bias: Boolean (default `True`). If `True`,
            add 1 to the bias of the forget gate at initialization.
            Setting it to `True` will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](
            https://github.com/mlresearch/v37/blob/gh-pages/jozefowicz15.pdf)
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

    Call arguments:
        inputs: A (2+ `rank`)D tensor.
        states:  List of state tensors corresponding to the previous timestep.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.
    """

    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seed = seed
        self.seed_generator = backend.random.SeedGenerator(seed=seed)
        self.rank = rank
        if self.rank > 3:
            raise ValueError(
                f"Rank {rank} convolutions are not currently "
                f"implemented. Received: rank={rank}"
            )
        self.filters = filters
        self.kernel_size = argument_validation.standardize_tuple(
            kernel_size, self.rank, "kernel_size"
        )
        self.strides = argument_validation.standardize_tuple(
            strides, self.rank, "strides", allow_zero=True
        )
        self.padding = argument_validation.standardize_padding(padding)
        self.data_format = backend.standardize_data_format(data_format)
        self.dilation_rate = argument_validation.standardize_tuple(
            dilation_rate, self.rank, "dilation_rate"
        )
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        self.input_spec = InputSpec(ndim=rank + 2)
        self.state_size = -1  # Custom, defined in methods

    def build(self, inputs_shape, states_shape=None):
        if self.data_format == "channels_first":
            channel_axis = 1
            self.spatial_dims = inputs_shape[2:]
        else:
            channel_axis = -1
            self.spatial_dims = inputs_shape[1:-1]
        if None in self.spatial_dims:
            raise ValueError(
                "ConvLSTM layers only support static "
                "input shapes for the spatial dimension. "
                f"Received invalid input shape: input_shape={inputs_shape}"
            )
        if inputs_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs (last axis) should be "
                "defined. Found None. Full input shape received: "
                f"input_shape={inputs_shape}"
            )
        self.input_spec = InputSpec(
            ndim=self.rank + 3, shape=(None,) + inputs_shape[1:]
        )

        input_dim = inputs_shape[channel_axis]
        self.input_dim = input_dim
        self.kernel_shape = self.kernel_size + (input_dim, self.filters * 4)
        recurrent_kernel_shape = self.kernel_size + (
            self.filters,
            self.filters * 4,
        )

        self.kernel = self.add_weight(
            shape=self.kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )

        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return ops.concatenate(
                        [
                            self.bias_initializer(
                                (self.filters,), *args, **kwargs
                            ),
                            initializers.get("ones")(
                                (self.filters,), *args, **kwargs
                            ),
                            self.bias_initializer(
                                (self.filters * 2,), *args, **kwargs
                            ),
                        ]
                    )

            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.filters * 4,),
                name="bias",
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=False):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        # dp_mask = self.get_dropout_mask(inputs)
        # rec_dp_mask = self.get_recurrent_dropout_mask(h_tm1)

        # if training and 0.0 < self.dropout < 1.0:
        #     inputs *= dp_mask
        # if training and 0.0 < self.recurrent_dropout < 1.0:
        #     h_tm1 *= rec_dp_mask

        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs

        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1

        (kernel_i, kernel_f, kernel_c, kernel_o) = ops.split(
            self.kernel, 4, axis=self.rank + 1
        )
        (
            recurrent_kernel_i,
            recurrent_kernel_f,
            recurrent_kernel_c,
            recurrent_kernel_o,
        ) = ops.split(self.recurrent_kernel, 4, axis=self.rank + 1)

        if self.use_bias:
            bias_i, bias_f, bias_c, bias_o = ops.split(self.bias, 4)
        else:
            bias_i, bias_f, bias_c, bias_o = None, None, None, None

        x_i = self.input_conv(inputs_i, kernel_i, bias_i, padding=self.padding)
        x_f = self.input_conv(inputs_f, kernel_f, bias_f, padding=self.padding)
        x_c = self.input_conv(inputs_c, kernel_c, bias_c, padding=self.padding)
        x_o = self.input_conv(inputs_o, kernel_o, bias_o, padding=self.padding)

        h_i = self.recurrent_conv(h_tm1_i, recurrent_kernel_i)
        h_f = self.recurrent_conv(h_tm1_f, recurrent_kernel_f)
        h_c = self.recurrent_conv(h_tm1_c, recurrent_kernel_c)
        h_o = self.recurrent_conv(h_tm1_o, recurrent_kernel_o)

        i = self.recurrent_activation(x_i + h_i)
        f = self.recurrent_activation(x_f + h_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.recurrent_activation(x_o + h_o)
        h = o * self.activation(c)
        return h, [h, c]

    def compute_output_shape(self, inputs_shape, states_shape=None):
        conv_output_shape = operation_utils.compute_conv_output_shape(
            inputs_shape,
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        return conv_output_shape, [conv_output_shape, conv_output_shape]

    def get_initial_state(self, batch_size=None):
        if self.data_format == "channels_last":
            input_shape = (batch_size,) + self.spatial_dims + (self.input_dim,)
        else:
            input_shape = (batch_size, self.input_dim) + self.spatial_dims
        state_shape = self.compute_output_shape(input_shape)[0]
        return [
            ops.zeros(state_shape, dtype=self.compute_dtype),
            ops.zeros(state_shape, dtype=self.compute_dtype),
        ]

    def input_conv(self, x, w, b=None, padding="valid"):
        conv_out = ops.conv(
            x,
            w,
            strides=self.strides,
            padding=padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        if b is not None:
            if self.data_format == "channels_last":
                bias_shape = (1,) * (self.rank + 1) + (self.filters,)
            else:
                bias_shape = (1, self.filters) + (1,) * self.rank
            bias = ops.reshape(b, bias_shape)
            conv_out += bias
        return conv_out

    def recurrent_conv(self, x, w):
        strides = argument_validation.standardize_tuple(
            1, self.rank, "strides", allow_zero=True
        )
        conv_out = ops.conv(
            x, w, strides=strides, padding="same", data_format=self.data_format
        )
        return conv_out

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
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
            "unit_forget_bias": self.unit_forget_bias,
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
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class ConvLSTM(RNN):
    """Abstract N-D Convolutional LSTM layer (used as implementation base).

    Similar to an LSTM layer, but the input transformations
    and recurrent transformations are both convolutional.

    Args:
        rank: Integer, rank of the convolution, e.g. "2" for 2D convolutions.
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the strides of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input such that output has the same
            height/width dimension as the input.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, time, ..., channels)`
            while `channels_first` corresponds to
            inputs with shape `(batch, time, channels, ...)`.
            When unspecified, uses
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json` (if exists) else 'channels_last'.
            Defaults to `'channels_last'`.
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use.
            By default hyperbolic tangent activation function is applied
            (`tanh(x)`).
        recurrent_activation: Activation function to use
            for the recurrent step.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Use in combination with `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al., 2015](
            http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to.
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        seed: Random seed for dropout.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence. (default False)
        return_state: Boolean Whether to return the last state
            in addition to the output. (default False)
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
    """

    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
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
        **kwargs,
    ):
        cell = ConvLSTMCell(
            rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            seed=seed,
            name="conv_lstm_cell",
            dtype=kwargs.get("dtype"),
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            **kwargs,
        )
        self.input_spec = InputSpec(ndim=rank + 3)

    def call(self, sequences, initial_state=None, mask=None, training=False):
        return super().call(
            sequences, initial_state=initial_state, mask=mask, training=training
        )

    def compute_output_shape(self, sequences_shape, initial_state_shape=None):
        batch_size = sequences_shape[0]
        steps = sequences_shape[1]
        step_shape = (batch_size,) + sequences_shape[2:]
        state_shape = self.cell.compute_output_shape(step_shape)[0][1:]

        if self.return_sequences:
            output_shape = (
                batch_size,
                steps,
            ) + state_shape
        else:
            output_shape = (batch_size,) + state_shape

        if self.return_state:
            batched_state_shape = (batch_size,) + state_shape
            return output_shape, batched_state_shape, batched_state_shape
        return output_shape

    def compute_mask(self, _, mask):
        mask = tree.flatten(mask)[0]
        output_mask = mask if self.return_sequences else None
        if self.return_state:
            state_mask = [None, None]
            return [output_mask] + state_mask
        else:
            return output_mask

    @property
    def filters(self):
        return self.cell.filters

    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

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
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

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

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
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
            "unit_forget_bias": self.unit_forget_bias,
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
            "seed": self.cell.seed,
        }
        base_config = super().get_config()
        del base_config["cell"]
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
