"""Keras base class for transpose convolution layers."""

from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src.backend import standardize_data_format
from keras.src.backend.common.backend_utils import (
    compute_conv_transpose_output_shape,
)
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.utils.argument_validation import standardize_padding
from keras.src.utils.argument_validation import standardize_tuple


class BaseConvTranspose(Layer):
    """Abstract N-D transposed convolution layer.

    The need for transposed convolutions generally arises from the desire to use
    a transformation going in the opposite direction of a normal convolution,
    i.e., from something that has the shape of the output of some convolution to
    something that has the shape of its input while maintaining a connectivity
    pattern that is compatible with said convolution.

    Args:
        rank: int, the rank of the transposed convolution, e.g. 2 for 2D
            transposed convolution.
        filters: int, the dimension of the output space (the number of filters
            in the transposed convolution).
        kernel_size: int or tuple/list of `rank` integers, specifying the size
            of the transposed convolution window.
        strides: int or tuple/list of `rank` integers, specifying the stride
            length of the transposed convolution. If only one int is specified,
            the same stride size will be used for all dimensions.
            `strides > 1` is incompatible with `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input such that output has the same
            height/width dimension as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, steps, features)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, features, steps)`. It defaults to the `image_data_format`
            value found in your Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: int or tuple/list of `rank` integers, specifying the
            dilation rate to use for dilated convolution. If only one int is
            specified, the same dilation rate will be used for all dimensions.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        kernel_initializer: Initializer for the convolution kernel. If `None`,
            the default initializer (`"glorot_uniform"`) will be used.
        bias_initializer: Initializer for the bias vector. If `None`, the
            default initializer (`"zeros"`) will be used.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape). Constraints
            are not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.
    """

    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        output_padding=None,
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        **kwargs,
    ):
        super().__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=activity_regularizer,
            **kwargs,
        )
        self.rank = rank
        self.filters = filters
        self.kernel_size = standardize_tuple(kernel_size, rank, "kernel_size")
        self.strides = standardize_tuple(strides, rank, "strides")
        self.dilation_rate = standardize_tuple(
            dilation_rate, rank, "dilation_rate"
        )
        self.padding = standardize_padding(padding)
        if output_padding is None:
            self.output_padding = None
        else:
            self.output_padding = standardize_tuple(
                output_padding,
                rank,
                "output_padding",
                allow_zero=True,
            )
        self.data_format = standardize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=self.rank + 2)
        self.data_format = self.data_format

        if self.filters is not None and self.filters <= 0:
            raise ValueError(
                "Invalid value for argument `filters`. Expected a strictly "
                f"positive value. Received filters={self.filters}."
            )

        if not all(self.kernel_size):
            raise ValueError(
                "The argument `kernel_size` cannot contain 0. Received "
                f"kernel_size={self.kernel_size}."
            )

        if not all(self.strides):
            raise ValueError(
                "The argument `strides` cannot contains 0. Received "
                f"strides={self.strides}."
            )

        if max(self.strides) > 1 and max(self.dilation_rate) > 1:
            raise ValueError(
                "`strides > 1` not supported in conjunction with "
                f"`dilation_rate > 1`. Received: strides={self.strides} and "
                f"dilation_rate={self.dilation_rate}"
            )

    def build(self, input_shape):
        if self.data_format == "channels_last":
            channel_axis = -1
            input_channel = input_shape[-1]
        else:
            channel_axis = 1
            input_channel = input_shape[1]
        self.input_spec = InputSpec(
            min_ndim=self.rank + 2, axes={channel_axis: input_channel}
        )
        kernel_shape = self.kernel_size + (
            self.filters,
            input_channel,
        )

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None

    def call(self, inputs):
        outputs = ops.conv_transpose(
            inputs,
            self.kernel,
            strides=list(self.strides),
            padding=self.padding,
            output_padding=self.output_padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

        if self.use_bias:
            if self.data_format == "channels_last":
                bias_shape = (1,) * (self.rank + 1) + (self.filters,)
            else:
                bias_shape = (1, self.filters) + (1,) * self.rank
            bias = ops.reshape(self.bias, bias_shape)
            outputs = ops.add(outputs, bias)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return compute_conv_transpose_output_shape(
            input_shape,
            self.kernel_size,
            self.filters,
            strides=self.strides,
            padding=self.padding,
            output_padding=self.output_padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "data_format": self.data_format,
                "dilation_rate": self.dilation_rate,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
                "activity_regularizer": regularizers.serialize(
                    self.activity_regularizer
                ),
                "kernel_constraint": constraints.serialize(
                    self.kernel_constraint
                ),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        return config
