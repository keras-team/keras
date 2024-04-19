"""Keras abstract base layer for separable convolution."""

from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src.backend import standardize_data_format
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.ops.operation_utils import compute_conv_output_shape
from keras.src.utils.argument_validation import standardize_padding
from keras.src.utils.argument_validation import standardize_tuple


class BaseSeparableConv(Layer):
    """Abstract base layer for separable convolution.

    This layer performs a depthwise convolution that acts separately on
    channels, followed by a pointwise convolution that mixes channels. If
    `use_bias` is True and a bias initializer is provided, it adds a bias vector
    to the output.

    Args:
        rank: int, the rank of the convolution, e.g. 2 for 2D convolution.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel. The total number of depthwise convolution
            output channels will be equal to `input_channel * depth_multiplier`.
        filters: int, the dimensionality of the output space (i.e. the number
            of filters in the pointwise convolution).
        kernel_size: int or tuple/list of `rank` integers, specifying the size
            of the depthwise convolution window.
        strides: int or tuple/list of `rank` integers, specifying the stride
            length of the depthwise convolution. If only one int is specified,
            the same stride size will be used for all dimensions.
            `stride value != 1` is incompatible with `dilation_rate != 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input. When `padding="same"` and
            `strides=1`, the output has the same size as the input.
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
        depthwise_initializer: An initializer for the depthwise convolution
            kernel. If None, then the default initializer (`"glorot_uniform"`)
            will be used.
        pointwise_initializer: An initializer for the pointwise convolution
            kernel. If None, then the default initializer (`"glorot_uniform"`)
            will be used.
        bias_initializer: An initializer for the bias vector. If None, the
            default initializer ('"zeros"') will be used.
        depthwise_regularizer: Optional regularizer for the depthwise
            convolution kernel.
        pointwise_regularizer: Optional regularizer for the pointwise
            convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        depthwise_constraint: Optional projection function to be applied to the
            depthwise kernel after being updated by an `Optimizer` (e.g. used
            for norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape).
        pointwise_constraint: Optional projection function to be applied to the
            pointwise kernel after being updated by an `Optimizer`.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.
    """

    def __init__(
        self,
        rank,
        depth_multiplier,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        pointwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        pointwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        pointwise_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        **kwargs,
    ):
        super().__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs,
        )
        self.rank = rank
        self.depth_multiplier = depth_multiplier
        self.filters = filters
        self.kernel_size = standardize_tuple(kernel_size, rank, "kernel_size")
        self.strides = standardize_tuple(strides, rank, "strides")
        self.dilation_rate = standardize_tuple(
            dilation_rate, rank, "dilation_rate"
        )
        self.padding = standardize_padding(padding)
        self.data_format = standardize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.pointwise_initializer = initializers.get(pointwise_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.pointwise_regularizer = regularizers.get(pointwise_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.pointwise_constraint = constraints.get(pointwise_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.data_format = self.data_format

        self.input_spec = InputSpec(min_ndim=self.rank + 2)

        if self.depth_multiplier is not None and self.depth_multiplier <= 0:
            raise ValueError(
                "Invalid value for argument `depth_multiplier`. Expected a "
                "strictly positive value. Received "
                f"depth_multiplier={self.depth_multiplier}."
            )

        if self.filters is not None and self.filters <= 0:
            raise ValueError(
                "Invalid value for argument `filters`. Expected a strictly "
                f"positive value. Received filters={self.filters}."
            )

        if not all(self.kernel_size):
            raise ValueError(
                "The argument `kernel_size` cannot contain 0. Received: "
                f"kernel_size={self.kernel_size}."
            )

        if not all(self.strides):
            raise ValueError(
                "The argument `strides` cannot contains 0(s). Received: "
                f"strides={self.strides}"
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
        depthwise_kernel_shape = self.kernel_size + (
            input_channel,
            self.depth_multiplier,
        )
        pointwise_kernel_shape = (1,) * self.rank + (
            self.depth_multiplier * input_channel,
            self.filters,
        )

        self.depthwise_kernel = self.add_weight(
            name="depthwise_kernel",
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        self.pointwise_kernel = self.add_weight(
            name="pointwise_kernel",
            shape=pointwise_kernel_shape,
            initializer=self.pointwise_initializer,
            regularizer=self.pointwise_regularizer,
            constraint=self.pointwise_constraint,
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
        self.built = True

    def call(self, inputs):
        outputs = ops.separable_conv(
            inputs,
            self.depthwise_kernel,
            self.pointwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

        if self.use_bias:
            if self.data_format == "channels_last":
                bias_shape = (1,) * (self.rank + 1) + (self.filters,)
            else:
                bias_shape = (1, self.filters) + (1,) * self.rank
            bias = ops.reshape(self.bias, bias_shape)
            outputs += bias

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return compute_conv_output_shape(
            input_shape,
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depth_multiplier": self.depth_multiplier,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "data_format": self.data_format,
                "dilation_rate": self.dilation_rate,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "depthwise_initializer": initializers.serialize(
                    self.depthwise_initializer
                ),
                "pointwise_initializer": initializers.serialize(
                    self.pointwise_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "depthwise_regularizer": regularizers.serialize(
                    self.depthwise_regularizer
                ),
                "pointwise_regularizer": regularizers.serialize(
                    self.pointwise_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
                "activity_regularizer": regularizers.serialize(
                    self.activity_regularizer
                ),
                "depthwise_constraint": constraints.serialize(
                    self.depthwise_constraint
                ),
                "pointwise_constraint": constraints.serialize(
                    self.pointwise_constraint
                ),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        return config
