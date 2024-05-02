from keras.src.api_export import keras_export
from keras.src.layers.convolutional.base_depthwise_conv import BaseDepthwiseConv


@keras_export("keras.layers.DepthwiseConv2D")
class DepthwiseConv2D(BaseDepthwiseConv):
    """2D depthwise convolution layer.

    Depthwise convolution is a type of convolution in which each input channel
    is convolved with a different kernel (called a depthwise kernel). You can
    understand depthwise convolution as the first step in a depthwise separable
    convolution.

    It is implemented via the following steps:

    - Split the input into individual channels.
    - Convolve each channel with an individual depthwise kernel with
      `depth_multiplier` output channels.
    - Concatenate the convolved outputs along the channels axis.

    Unlike a regular 2D convolution, depthwise convolution does not mix
    information across different input channels.

    The `depth_multiplier` argument determines how many filters are applied to
    one input channel. As such, it controls the amount of output channels that
    are generated per input channel in the depthwise step.

    Args:
        kernel_size: int or tuple/list of 2 integer, specifying the size of the
            depthwise convolution window.
        strides: int or tuple/list of 2 integer, specifying the stride length
            of the depthwise convolution. `strides > 1` is incompatible with
            `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input. When `padding="same"` and
            `strides=1`, the output has the same size as the input.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel. The total number of depthwise convolution
            output channels will be equal to `input_channel * depth_multiplier`.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file
            at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: int or tuple/list of 2 integers, specifying the dilation
            rate to use for dilated convolution.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        depthwise_initializer: Initializer for the convolution kernel.
            If `None`, the default initializer (`"glorot_uniform"`)
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

    Input shape:
    - If `data_format="channels_last"`:
        A 4D tensor with shape: `(batch_size, height, width, channels)`
    - If `data_format="channels_first"`:
        A 4D tensor with shape: `(batch_size, channels, height, width)`

    Output shape:
    - If `data_format="channels_last"`:
        A 4D tensor with shape:
        `(batch_size, new_height, new_width, channels * depth_multiplier)`
    - If `data_format="channels_first"`:
        A 4D tensor with shape:
        `(batch_size, channels * depth_multiplier, new_height, new_width)`

    Returns:
        A 4D tensor representing
        `activation(depthwise_conv2d(inputs, kernel) + bias)`.

    Raises:
        ValueError: when both `strides > 1` and `dilation_rate > 1`.

    Example:

    >>> x = np.random.rand(4, 10, 10, 12)
    >>> y = keras.layers.DepthwiseConv2D(3, 3, activation='relu')(x)
    >>> print(y.shape)
    (4, 8, 8, 36)
    """

    def __init__(
        self,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        depth_multiplier=1,
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            rank=2,
            depth_multiplier=depth_multiplier,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
