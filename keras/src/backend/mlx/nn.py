import mlx.core as mx
import mlx.nn as nn

from keras.src import tree
from keras.src.backend import standardize_data_format
from keras.src.backend import standardize_dtype
from keras.src.backend.config import epsilon
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.backend.mlx.core import to_mlx_dtype
from keras.src.backend.mlx.numpy import clip
from keras.src.utils.argument_validation import standardize_tuple


def relu(x):
    x = convert_to_tensor(x)
    return nn.relu(x)


def relu6(x):
    x = convert_to_tensor(x)
    return nn.relu6(x)


def sigmoid(x):
    x = convert_to_tensor(x)
    return mx.sigmoid(x)


def tanh(x):
    x = convert_to_tensor(x)
    return mx.tanh(x)


def softplus(x):
    x = convert_to_tensor(x)
    return nn.softplus(x)


def softsign(x):
    x = convert_to_tensor(x)
    return x / (1 + mx.abs(x))


def silu(x, beta=1.0):
    x = convert_to_tensor(x)
    if beta == 1:
        return nn.silu(x)
    else:
        return x * mx.sigmoid(beta * x)


def log_sigmoid(x):
    x = convert_to_tensor(x)
    return nn.log_sigmoid(x)


def leaky_relu(x, negative_slope=0.2):
    x = convert_to_tensor(x)
    return nn.leaky_relu(x, negative_slope=negative_slope)


def hard_sigmoid(x):
    x = convert_to_tensor(x)
    return mx.maximum(0, mx.minimum(1, x / 6 + 0.5))


def hard_silu(x):
    x = convert_to_tensor(x)
    xclipped = mx.minimum(mx.maximum(x, -3), 3)
    return x * (xclipped + 3) / 6


def elu(x, alpha=1.0):
    x = convert_to_tensor(x)
    xneg = mx.minimum(x, 0)
    xpos = mx.maximum(x, 0)
    if alpha != 1:
        xneg = alpha * (mx.exp(xneg) - 1)
    else:
        xneg = mx.exp(xneg) - 1
    return xneg + xpos


def selu(x):
    x = convert_to_tensor(x)
    a = 1.6732631921768188
    scale = 1.0507009873554805
    xneg = mx.minimum(x, 0)
    xpos = mx.maximum(x, 0)
    return scale * (xpos + a * (mx.exp(xneg) - 1))


def gelu(x, approximate=True):
    x = convert_to_tensor(x)

    def gelu_tanh_approx(x):
        return (
            0.5 * x * (1 + mx.tanh(0.7978846 * (x + 0.044715 * x * x.square())))
        )

    f = gelu_tanh_approx if approximate else nn.gelu
    return f(x)


def softmax(x, axis=-1):
    x = convert_to_tensor(x)
    return mx.softmax(x, axis=axis)


def log_softmax(x, axis=-1):
    x = convert_to_tensor(x)
    return x - mx.logsumexp(x, axis=axis, keepdims=True)


def _transpose_spatial_inputs(inputs, data_format="channels_last"):
    """Transposes spatial dimensions of input tensor based on data format.

    Args:
        inputs: Input tensor.
        data_format: Data format, either "channels_last" or "channels_first".

    Returns:
        Transposed input tensor with channels in the specified format.
    """
    print(f"Original input shape: {inputs.shape}")
    if data_format == "channels_first":
        if inputs.ndim == 4:  # (N, H, W, C) -> (N, C, H, W)
            inputs = mx.transpose(inputs, (0, 3, 1, 2))
        elif inputs.ndim == 3:  # (H, W, C) -> (C, H, W)
            inputs = mx.transpose(inputs, (2, 0, 1))
        print(f"Transposed inputs to channels_first: {inputs.shape}")
    else:
        print("No transposition needed for channels_last format.")
    return inputs


def _transpose_spatial_outputs(outputs):
    num_spatial_dims = outputs.ndim - 2
    if num_spatial_dims == 1:
        outputs = mx.transpose(outputs, (0, 2, 1))
    elif num_spatial_dims == 2:
        outputs = mx.transpose(outputs, (0, 2, 3, 1))
    elif num_spatial_dims == 3:
        outputs = mx.transpose(outputs, (0, 2, 3, 4, 1))
    return outputs


def _transpose_conv_kernel(kernel, data_format):
    """Transposes convolution kernel based on data format.

    Args:
        kernel: Convolution kernel tensor.
        data_format: Data format, either "channels_last" or "channels_first".

    Returns:
        Transposed kernel tensor with channels in the specified format.
    """
    print(f"Original kernel shape: {kernel.shape}, Data format: {data_format}")
    if data_format == "channels_first":
        # (kernel_height, kernel_width, input_channels, output_channels) -> (output_channels, input_channels, kernel_height, kernel_width)
        kernel = mx.transpose(kernel, (3, 2, 0, 1))
    print(f"Transposed kernel shape: {kernel.shape}")
    return kernel


def _compute_padding_length(
    input_length, kernel_length, stride, dilation_rate=1
):
    """Compute padding length along one dimension."""
    total_padding_length = (
        dilation_rate * (kernel_length - 1) - (input_length - 1) % stride
    )
    left_padding = total_padding_length // 2
    right_padding = (total_padding_length + 1) // 2
    return (left_padding, right_padding)


def _apply_same_padding(
    inputs, kernel_size, strides, operation_type, dilation_rate=1
):
    spatial_shape = inputs.shape[2:]
    num_spatial_dims = len(spatial_shape)
    padding = ()

    for i in range(num_spatial_dims):
        if operation_type == "pooling":
            padding_size = _compute_padding_length(
                spatial_shape[i], kernel_size[i], strides[i]
            )
        else:
            dilation_rate = standardize_tuple(
                dilation_rate, num_spatial_dims, "dilation_rate"
            )
            padding_size = _compute_padding_length(
                spatial_shape[i], kernel_size[i], strides[i], dilation_rate[i]
            )
        padding = (padding_size,) + padding

    if all([left == right for left, right in padding]):
        return inputs, [left for left, _ in padding]

    flattened_padding = tuple(
        value for left_and_right in padding for value in left_and_right
    )
    return mx.pad(inputs, pad_width=flattened_padding, constant_values=0), 0


def max_pool(
    inputs, pool_size, strides=None, padding="valid", data_format=None
):
    raise NotImplementedError("MLX backend doesn't support max pooling yet")


def average_pool(
    inputs, pool_size, strides=None, padding="valid", data_format=None
):
    raise NotImplementedError("MLX backend doesn't support average pooling yet")


def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
):
    """Performs convolution operation with data format handling and kernel transposition.

    Args:
        inputs: Input tensor.
        kernel: Convolution kernel tensor.
        strides: Convolution stride.
        padding: Padding mode, either "valid" or "same".
        data_format: Data format, either "channels_last" or "channels_first".
        dilation_rate: Dilation rate for dilated convolution.

    Returns:
        Output tensor after convolution.
    """
    print("Initial input shape:", inputs.shape)
    print("Initial kernel shape:", kernel.shape)

    # Convert inputs and kernel to tensors
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    print("After conversion - Input shape:", inputs.shape)
    print("After conversion - Kernel shape:", kernel.shape)

    # Print before standardizing data format
    print("Before standardizing data format - Data format:", data_format)
    data_format = standardize_data_format(data_format)
    # Print after standardizing data format
    print("After standardizing data format - Data format:", data_format)

    # Unpack and standardize strides and dilation_rate
    print(
        "Before standardization - Strides:",
        strides,
        "Dilation Rate:",
        dilation_rate,
    )
    strides = standardize_tuple(strides, inputs.ndim - 2, "strides")
    dilation_rate = standardize_tuple(
        dilation_rate, inputs.ndim - 2, "dilation_rate"
    )
    print(
        "After standardization - Strides:",
        strides,
        "Dilation Rate:",
        dilation_rate,
    )

    # Transpose input and kernel if necessary based on data format
    print(
        "Before transposition - Inputs and kernel transposition based on data format"
    )
    if data_format == "channels_first":
        inputs = _transpose_spatial_inputs(inputs, data_format)
        kernel = _transpose_conv_kernel(kernel, data_format)
        print("After transposition - Input shape:", inputs.shape)
        print("After transposition - Kernel shape:", kernel.shape)

    # Calculate padding if 'same' is required
    print("Before applying padding - Padding type:", padding)
    if padding == "same":
        inputs, padding = _apply_same_padding(
            inputs, kernel.shape[2:], strides, dilation_rate
        )
        print("After applying 'same' padding - Input shape:", inputs.shape)
        print("Padding applied:", padding)
    elif padding == "valid":
        padding = 0
        print("Using 'valid' padding - No padding applied")

    # Perform convolution based on dimensionality
    print("Before convolution - Input dimensions:", inputs.ndim)
    if inputs.ndim == 3:  # 1D Convolution
        print(f"Performing 1D convolution with inputs shape {inputs.shape}")
        print(f"Performing 1D convolution with kernel shape {kernel.shape}")
        print(f"Performing 1D convolution with strides {strides}")
        print(f"Performing 1D convolution with padding {padding}")
        print(f"Performing 1D convolution with dilation_rate {dilation_rate}")
        print("Performing 1D convolution with groups = 1")
        outputs = mx.conv1d(
            inputs,
            kernel,
            stride=strides,
            padding=padding,
            dilation=dilation_rate,
            groups=1,
        )
        print("After 1D convolution - Output shape:", outputs.shape)
    elif inputs.ndim == 4:  # 2D Convolution
        print(f"Performing 2D convolution with inputs shape {inputs.shape}")
        print(f"Performing 2D convolution with kernel shape {kernel.shape}")
        print(f"Performing 2D convolution with strides {strides}")
        print(f"Performing 2D convolution with padding {padding}")
        print(f"Performing 2D convolution with dilation_rate {dilation_rate}")
        print("Performing 2D convolution with groups = 1")
        outputs = mx.conv2d(
            inputs,
            kernel,
            stride=strides,
            padding=padding,
            dilation=dilation_rate,
            groups=1,
        )
        print("After 2D convolution - Output shape:", outputs.shape)
    else:
        raise ValueError("Unsupported number of dimensions for conv operation.")

    # Transpose output back if necessary
    print("Before final transposition - Data format:", data_format)
    if data_format == "channels_first":
        outputs = _transpose_spatial_outputs(outputs, data_format)
        print("After final transposition - Output shape:", outputs.shape)

    print("Final output shape:", outputs.shape)
    return outputs


def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    raise NotImplementedError("MLX backend doesn't support depthwise conv yet")


def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    raise NotImplementedError("MLX backend doesn't support separable conv yet")


def conv_transpose(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    output_padding=None,
    data_format=None,
    dilation_rate=1,
):
    raise NotImplementedError("MLX backend doesn't support conv transpose yet")


def one_hot(x, num_classes, axis=-1, dtype="float32", sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with mlx backend")
    x = convert_to_tensor(x, dtype=mx.int32)
    dtype = to_mlx_dtype(standardize_dtype(dtype))

    # TODO: Make this faster by instantiating 0s and using x as indices to
    #       write the 1s (basically using scatter)
    output = mx.eye(num_classes, dtype=dtype)[x]

    if axis != -1 and axis != output.ndim:
        output = mx.moveaxis(output, -1, axis)

    return output


def multi_hot(x, num_classes, axis=-1, dtype="float32", sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with mlx backend")
    x = convert_to_tensor(x)
    reduction_axis = 1 if x.ndim > 1 else 0
    return one_hot(x, num_classes, axis=axis, dtype=dtype).max(
        axis=reduction_axis
    )


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)

    if target.shape != output.shape:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    if len(target.shape) < 1:
        raise ValueError(
            "Arguments `target` and `output` must be at least rank 1. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        log_prob = output - mx.logsumexp(output, axis=axis)
    else:
        output = output / output.sum(axis=axis, keepdims=True)
        output = clip(output, epsilon(), 1 - epsilon())
        log_prob = mx.log(output)

    return -(target * log_prob).sum(axis=axis)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = convert_to_tensor(target, dtype=mx.int32)
    output = convert_to_tensor(output)

    if axis != -1:
        raise ValueError(
            "Only axis=-1 is supported in sparse_categorical_crossentropy"
        )

    if target.ndim == output.ndim and target.shape[-1] == 1:
        target = target.squeeze(-1)

    if len(output.shape) < 1:
        raise ValueError(
            "Argument `output` must be at least rank 1. "
            "Received: "
            f"output.shape={output.shape}"
        )
    if target.shape != output.shape[:-1]:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape "
            "up until the last dimension: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        log_prob = output - mx.logsumexp(output, axis=-1, keepdims=True)
    else:
        output = output / output.sum(axis=-1, keepdims=True)
        output = mx.minimum(mx.maximum(output, epsilon()), 1 - epsilon())
        log_prob = mx.log(output)

    return -mx.take_along_axis(log_prob, target[..., None], axis=-1).squeeze(-1)


def binary_crossentropy(target, output, from_logits=False):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)

    if target.shape != output.shape:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        return nn.losses.binary_cross_entropy(output, target, reduction="none")
    else:
        output = mx.minimum(mx.maximum(output, epsilon()), 1 - epsilon())
        return -target * mx.log(output) - (1 - target) * mx.log(1 - output)


def moments(x, axes, keepdims=False, synchronized=False):
    if synchronized:
        raise NotImplementedError(
            "Argument synchronized=True is not supported with MLX."
        )

    x = convert_to_tensor(x)

    # The dynamic range of float16 is too limited for statistics. As a
    # workaround, we simply perform the operations on float32 and convert back
    # to float16
    need_cast = False
    ori_dtype = x.dtype
    if ori_dtype == mx.float16:
        need_cast = True
        x = x.astype(mx.float32)

    mean = mx.mean(x, axis=axes, keepdims=True)
    variance = x.square().mean(axis=axes, keepdims=True) - mean.square()

    if not keepdims:
        mean = mean.squeeze(axes)
        variance = variance.squeeze(axes)

    if need_cast:
        # TODO: Clip as is done in the pytorch backend
        mean = mean.astype(ori_dtype)
        variance = variance.astype(ori_dtype)

    return mean, variance


def batch_normalization(
    x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3
):
    x = convert_to_tensor(x)
    mean = convert_to_tensor(x)
    shape = [1] * len(x.shape)
    shape[axis] = mean.shape[0]
    mean = mx.reshape(mean, shape)
    variance = mx.reshape(variance, shape)

    inv = mx.rsqrt(variance + epsilon)
    if scale is not None:
        scale = mx.reshape(scale, shape)
        inv = inv * scale

    res = -mean * inv
    if offset is not None:
        offset = mx.reshape(offset, shape)
        res = res + offset

    return mx.add(x * inv, res)


def ctc_loss(
    target,
    output,
    target_length,
    output_length,
    mask_index=0,
):
    raise NotImplementedError("MLX backend doesn't support the ctc loss yet.")
