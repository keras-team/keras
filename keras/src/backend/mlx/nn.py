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


def _transpose_spatial_inputs(inputs):
    num_spatial_dims = inputs.ndim - 2
    if num_spatial_dims == 1:
        inputs = mx.transpose(inputs, (0, 2, 1))
    elif num_spatial_dims == 2:
        inputs = mx.transpose(inputs, (0, 3, 1, 2))
    elif num_spatial_dims == 3:
        inputs = mx.transpose(inputs, (0, 4, 1, 2, 3))
    else:
        raise ValueError(
            "Inputs to conv transpose operation should have ndim=3, 4, or 5,"
            "corresponding to 1D, 2D and 3D inputs. Received input "
            f"shape: {inputs.shape}."
        )
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


def _transpose_conv_kernel(kernel):
    num_spatial_dims = len(kernel.shape) - 2
    if num_spatial_dims == 1:
        kernel = mx.transpose(kernel, (2, 1, 0))
    elif num_spatial_dims == 2:
        kernel = mx.transpose(kernel, (3, 2, 0, 1))
    else:
        raise ValueError(
            "Kernel for conv transpose operation should have ndim=3 or 4,"
            "corresponding to 1D, 2D kernels. Received kernel "
            f"shape: {kernel.shape}."
        )
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
    data_format=None,
    dilation_rate=1,
):
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    num_spatial_dims = inputs.ndim - 2
    strides = standardize_tuple(strides, num_spatial_dims, "strides")

    data_format = standardize_data_format(data_format)
    if data_format == "channels_last":
        inputs = _transpose_spatial_inputs(inputs)
    kernel = _transpose_conv_kernel(kernel)
    if padding == "same" and any(d != 1 for d in tree.flatten(strides)):
        inputs, padding = _apply_same_padding(
            inputs,
            kernel.shape[2:],
            strides,
            operation_type="conv",
            dilation_rate=dilation_rate,
        )
    elif padding == "valid":
        padding = 0

    channels = inputs.shape[1]
    kernel_in_channels = kernel.shape[1]
    if channels % kernel_in_channels > 0:
        raise ValueError(
            "The number of input channels must be evenly divisible by "
            f"kernel.shape[1]. Received: inputs.shape={inputs.shape}, "
            f"kernel.shape={kernel.shape}"
        )
    groups = channels // kernel_in_channels
    if groups != 1:
        raise ValueError(
            "MLX only supports groups=1 for convolution operations."
        )
    if num_spatial_dims == 1:
        outputs = mx.conv1d(
            inputs,
            kernel,
            stride=strides,
            padding=padding,
            dilation=dilation_rate,
        )
    elif num_spatial_dims == 2:
        outputs = mx.conv2d(
            inputs,
            kernel,
            stride=strides,
            padding=padding,
            dilation=dilation_rate,
        )
    else:
        raise ValueError(
            "Inputs to conv operation should have ndim=3 or 4,"
            "corresponding to 1D, 2D inputs. Received input "
            f"shape: {inputs.shape}."
        )

    if data_format == "channels_last":
        outputs = _transpose_spatial_outputs(outputs)
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
