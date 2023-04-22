import tensorflow as tf


def relu(x):
    return tf.nn.relu(x)


def relu6(x):
    return tf.nn.relu6(x)


def sigmoid(x):
    return tf.nn.sigmoid(x)


def softplus(x):
    return tf.math.softplus(x)


def softsign(x):
    return tf.nn.softsign(x)


def silu(x, beta=1.0):
    return tf.nn.silu(x, beta=beta)


def swish(x):
    return x * sigmoid(x)


def log_sigmoid(x):
    return tf.math.log_sigmoid(x)


def leaky_relu(x, negative_slope=0.2):
    return tf.nn.leaky_relu(x, alpha=negative_slope)


def hard_sigmoid(x):
    x = x / 6.0 + 0.5
    return tf.clip_by_value(x, 0.0, 1.0)


def elu(x):
    return tf.nn.elu(x)


def selu(x):
    return tf.nn.selu(x)


def gelu(x, approximate=True):
    return tf.nn.gelu(x, approximate)


def softmax(x, axis=None):
    return tf.nn.softmax(x, axis=axis)


def log_softmax(x, axis=None):
    return tf.nn.log_softmax(x, axis=axis)


def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format="channels_last",
):
    strides = pool_size if strides is None else strides
    padding = padding.upper()
    data_format = _convert_data_format(data_format, len(inputs.shape))
    return tf.nn.max_pool(inputs, pool_size, strides, padding, data_format)


def average_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format="channels_last",
):
    strides = pool_size if strides is None else strides
    padding = padding.upper()
    data_format = _convert_data_format(data_format, len(inputs.shape))
    return tf.nn.avg_pool(inputs, pool_size, strides, padding, data_format)


def _convert_data_format(data_format, ndim):
    if data_format == "channels_last":
        if ndim == 3:
            return "NWC"
        elif ndim == 4:
            return "NHWC"
        elif ndim == 5:
            return "NDHWC"
        else:
            raise ValueError(
                f"Input rank not supported: {ndim}. "
                "Expected values are [3, 4, 5]"
            )
    elif data_format == "channels_first":
        if ndim == 3:
            return "NCW"
        elif ndim == 4:
            return "NCHW"
        elif ndim == 5:
            return "NCDHW"
        else:
            raise ValueError(
                f"Input rank not supported: {ndim}. "
                "Expected values are [3, 4, 5]"
            )
    else:
        raise ValueError(
            f"Invalid data_format: {data_format}. "
            'Expected values are ["channels_first", "channels_last"]'
        )


def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format="channel_last",
    dilation_rate=1,
):
    """General N-D convolution function.

    Arg:
    """

    data_format = _convert_data_format(data_format, len(inputs.shape))
    padding = padding.upper()

    return tf.nn.convolution(
        inputs,
        kernel,
        strides,
        padding,
        data_format=data_format,
        dilations=dilation_rate,
    )


def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
):
    num_spatial_dims = len(inputs.shape) - 2
    if num_spatial_dims > 2:
        raise ValueError(
            "`inputs` rank must be 3 (1D conv) or 4 (2D conv). Received: "
            "{inputs.ndim}."
        )
    tf_data_format = _convert_data_format(data_format, len(inputs.shape))
    padding = padding.upper()
    if isinstance(strides, int):
        strides = (strides,) * num_spatial_dims
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * num_spatial_dims
    if num_spatial_dims == 1:
        # 1D depthwise conv.
        if data_format == "channels_last":
            strides = (1,) + strides * 2 + (1,)
            spatial_start_dim = 1
        else:
            strides = (1, 1) + strides * 2
            spatial_start_dim = 2
        inputs = tf.expand_dims(inputs, spatial_start_dim)
        kernel = tf.expand_dims(kernel, axis=0)

        dilation_rate = None if dilation_rate is None else (1,) + dilation_rate

        outputs = tf.nn.depthwise_conv2d(
            inputs,
            kernel,
            strides,
            padding,
            data_format=tf_data_format,
            dilations=dilation_rate,
        )
        return tf.squeeze(outputs, [spatial_start_dim])

    if data_format == "channels_last":
        strides = (1,) + strides + (1,)
        spatial_start_dim = 1
    else:
        strides = (1, 1) + strides
        spatial_start_dim = 2
    return tf.nn.depthwise_conv2d(
        inputs,
        kernel,
        strides,
        padding,
        data_format=tf_data_format,
        dilations=dilation_rate,
    )


def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
):
    num_spatial_dims = len(inputs.shape) - 2
    if num_spatial_dims > 2:
        raise ValueError(
            "`num_spatial_dims` must be 1 or 2. Received: "
            f"num_spatial_dims={num_spatial_dims}."
        )
    tf_data_format = _convert_data_format(data_format, len(inputs.shape))
    padding = padding.upper()
    if isinstance(strides, int):
        strides = (strides,) * num_spatial_dims
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * num_spatial_dims
    if num_spatial_dims == 1:
        # 1D depthwise conv.
        if data_format == "channels_last":
            strides = (1,) + strides * 2 + (1,)
            spatial_start_dim = 1
        else:
            strides = (1, 1) + strides * 2
            spatial_start_dim = 2
        inputs = tf.expand_dims(inputs, spatial_start_dim)
        depthwise_kernel = tf.expand_dims(depthwise_kernel, axis=0)
        pointwise_kernel = tf.expand_dims(pointwise_kernel, axis=0)
        dilation_rate = None if dilation_rate is None else (1,) + dilation_rate

        outputs = tf.nn.separable_conv2d(
            inputs,
            depthwise_kernel,
            pointwise_kernel,
            strides,
            padding,
            data_format=tf_data_format,
            dilations=dilation_rate,
        )
        return tf.squeeze(outputs, [spatial_start_dim])

    if data_format == "channels_last":
        strides = (1,) + strides + (1,)
        spatial_start_dim = 1
    else:
        strides = (1, 1) + strides
        spatial_start_dim = 2
    return tf.nn.separable_conv2d(
        inputs,
        depthwise_kernel,
        pointwise_kernel,
        strides,
        padding,
        data_format=tf_data_format,
        dilations=dilation_rate,
    )


def _deconv_output_length(
    input_length,
    kernel_size,
    padding,
    output_padding=None,
    stride=1,
    dilation=1,
):
    """Determines output length of a transposed convolution given input length.

    Args:
        input_length: Integer.
        kernel_size: Integer.
        padding: one of `"same"` or `"valid"`.
        output_padding: Integer, amount of padding along the output dimension.
          Can be set to `None` in which case the output length is inferred.
        stride: Integer.
        dilation: Integer.

    Returns:
        The output length (integer).
    """
    assert padding in {"same", "valid"}
    if input_length is None:
        return None

    # Get the dilated kernel size
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)

    # Infer length if output padding is None, else compute the exact length
    if output_padding is None:
        if padding == "valid":
            length = input_length * stride + max(kernel_size - stride, 0)
        else:
            length = input_length * stride
    else:
        if padding == "same":
            pad = kernel_size // 2
        else:
            pad = 0

        length = (
            (input_length - 1) * stride + kernel_size - 2 * pad + output_padding
        )
    return length


def compute_output_shape_conv_transpose(
    inputs,
    kernel,
    strides,
    padding,
    output_padding=None,
    data_format="channels_last",
    dilation_rate=1,
):
    num_spatial_dims = len(inputs.shape) - 2
    kernel_spatial_shape = kernel.shape[:-2]

    if isinstance(output_padding, int):
        output_padding = (output_padding,) * len(kernel_spatial_shape)
    if isinstance(strides, int):
        strides = (strides,) * num_spatial_dims
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * num_spatial_dims

    if data_format == "channels_last":
        inputs_spatial_shape = inputs.shape[1:-1]
    else:
        inputs_spatial_shape = inputs.shape[2:]

    output_shape = []
    for i in range(num_spatial_dims):
        current_output_padding = (
            None if output_padding is None else output_padding[i]
        )
        output_shape.append(
            _deconv_output_length(
                inputs_spatial_shape[i],
                kernel_spatial_shape[i],
                padding=padding,
                output_padding=current_output_padding,
                stride=strides[i],
                dilation=dilation_rate[0],
            )
        )

    if data_format == "channels_last":
        output_shape = [inputs.shape[0]] + output_shape + [kernel.shape[-2]]
    else:
        output_shape = [inputs.shape[0], kernel.shape[-1]] + output_shape
    return output_shape


def conv_transpose(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    output_padding=None,
    data_format="channels_last",
    dilation_rate=1,
):
    tf_data_format = _convert_data_format(data_format, len(inputs.shape))
    output_shape = compute_output_shape_conv_transpose(
        inputs,
        kernel,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate,
    )

    return tf.nn.conv_transpose(
        inputs,
        kernel,
        output_shape,
        strides,
        padding=padding.upper(),
        data_format=tf_data_format,
        dilations=dilation_rate,
    )


def one_hot(x, num_classes, axis=-1):
    return tf.one_hot(x, num_classes, axis=axis)
