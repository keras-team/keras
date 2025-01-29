"""Keras base class for depthwise convolution layers."""

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


class BaseDepthwiseConv(Layer):
    """Abstract N-D depthwise convolution layer.

    Depthwise convolution is a type of convolution in which each input channel
    is convolved with a different kernel (called a depthwise kernel). You can
    understand depthwise convolution as the first step in a depthwise separable
    convolution.

    It is implemented via the following steps:

    - Split the input into individual channels.
    - Convolve each channel with an individual depthwise kernel with
      `depth_multiplier` output channels.
    - Concatenate the convolved outputs along the channels axis.

    Unlike a regular convolution, depthwise convolution does not mix information
    across different input channels.

    The `depth_multiplier` argument determines how many filter are applied to
    one input channel. As such, it controls the amount of output channels that
    are generated per input channel in the depthwise step.


    Args:
        rank: int, the rank of the convolution, e.g. 2 for 2D convolution.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel. The total number of depthwise convolution
            output channels will be equal to `input_channel * depth_multiplier`.
        kernel_size: int or tuple/list of `rank` integers, specifying the size
            of the depthwise convolution window.
        strides: int or tuple/list of `rank` integers, specifying the stride
            length of the depthwise convolution. If only one int is specified,
            the same stride size will be used for all dimensions.
            `strides > 1` is incompatible with `dilation_rate > 1`.
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
        depthwise_initializer: Initializer for the depthwsie convolution
            kernel. If `None`, the default initializer (`"glorot_uniform"`)
            will be used.
        bias_initializer: Initializer for the bias vector. If `None`, the
            default initializer (`"zeros"`) will be used.
        depthwise_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        depthwise_constraint: Optional projection function to be applied to the
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
        depth_multiplier,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
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
        self.bias_initializer = initializers.get(bias_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=self.rank + 2)
        self.data_format = self.data_format

        if self.depth_multiplier is not None and self.depth_multiplier <= 0:
            raise ValueError(
                "Invalid value for argument `depth_multiplier`. Expected a "
                "strictly positive value. Received "
                f"depth_multiplier={self.depth_multiplier}."
            )

        if not all(self.kernel_size):
            raise ValueError(
                "The argument `kernel_size` cannot contain 0. Received "
                f"kernel_size={self.kernel_size}."
            )

        if not all(self.strides):
            raise ValueError(
                "The argument `strides` cannot contains 0. Received "
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
        depthwise_shape = self.kernel_size + (
            input_channel,
            self.depth_multiplier,
        )
        self.kernel = self.add_weight(
            name="kernel",
            shape=depthwise_shape,
            initializer=self.depthwise_initializer,
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.depth_multiplier * input_channel,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        self.built = True

    def _get_input_channel(self, input_shape):
        if self.data_format == "channels_last":
            input_channel = input_shape[-1]
        else:
            input_channel = input_shape[1]
        return input_channel

    def call(self, inputs):
        input_channel = self._get_input_channel(inputs.shape)
        outputs = ops.depthwise_conv(
            inputs,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

        if self.use_bias:
            if self.data_format == "channels_last":
                bias_shape = (1,) * (self.rank + 1) + (
                    self.depth_multiplier * input_channel,
                )
            else:
                bias_shape = (1, self.depth_multiplier * input_channel) + (
                    1,
                ) * self.rank
            bias = ops.reshape(self.bias, bias_shape)
            outputs = ops.add(outputs, bias)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_channel = self._get_input_channel(input_shape)
        return compute_conv_output_shape(
            input_shape,
            self.depth_multiplier * input_channel,
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
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "depthwise_regularizer": regularizers.serialize(
                    self.depthwise_regularizer
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
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        return config
