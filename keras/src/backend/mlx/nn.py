import mlx.core as mx
import mlx.nn as nn

from keras.src.backend import standardize_dtype
from keras.src.backend.config import epsilon
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.backend.mlx.core import to_mlx_dtype
from keras.src.backend.mlx.numpy import clip


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
    # conv1d: input (array) – input array of shape (N, H, C_in)
    # conv2d: input (array) – input array of shape (N, H, W, C_in)
    kernel,
    # conv1d: weight (array) – weight array of shape (C_out, H, C_in)
    # conv2d: weight (array) – weight array of shape (C_out, H, W, C_in)
    strides=1,
    # conv1d: stride (int, optional) – kernel stride. Default: 1.
    # conv2d: stride (int or tuple(int), optional) – tuple of size 2 with kernel strides. All spatial dimensions get the same stride if only one number is specified. Default: 1.
    padding="valid",
    # conv1d: padding (int, optional) – input padding. Default: 0.
    # conv2d: padding (int or tuple(int), optional) – tuple of size 2 with symmetric input padding. All spatial dimensions get the same padding if only one number is specified. Default: 0.
    data_format="channels_last",
    dilation_rate=1,
):
    print("Initial input shape:", inputs.shape)
    print("Initial kernel shape:", kernel.shape)
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    print("After conversion - Input shape:", inputs.shape)
    print("After conversion - Kernel shape:", kernel.shape)

    print("Strides before conversion:", strides)
    if isinstance(strides, int):
        strides = (strides,) * (inputs.ndim - 2)
    print("Strides after conversion:", strides)

    print("Dilation rate before conversion:", dilation_rate)
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * (inputs.ndim - 2)
    print("Dilation rate after conversion:", dilation_rate)

    print("Padding before conversion:", padding)
    if padding == "valid":
        padding = 0
    elif padding == "same":
        # Calculate padding to maintain the same shape
        padding = ((strides - 1) + dilation_rate * (kernel - 1)) // 2
    else:
        raise ValueError(f"Unknown padding mode: {padding}")

    print(f"Input shape before conv: {inputs.shape}")
    print(f"Kernel shape before conv: {kernel.shape}")

    if data_format != "channels_last":
        raise NotImplementedError(
            "MLX backend only supports data_format='channels_last'"
        )
    # exclude in_channels and out_channels
    print(f"Kernel shape before transpose: {kernel.shape}")
    num_spatial_dims = len(kernel.shape) - 2
    print(f"Num spatial dims: {num_spatial_dims}")

    if num_spatial_dims == 1:
        print(f"1D kernel shape before transpose: {kernel.shape}")
        print("Transposing 1D kernel from (C_out, C_in, H) to (H, C_in, C_out)")
        # For 1D convolution kernels from (C_out, C_in, H) to (H, C_in, C_out)
        kernel = kernel.transpose((2, 0, 1))
        print(f"1D kernel shape after transpose: {kernel.shape}")
    elif num_spatial_dims == 2:
        print(f"2D kernel shape before transpose: {kernel.shape}")
        print(
            "Transposing 2D kernel from (C_out, H, W, C_in) to (C_out, C_in, H, W)"
        )
        # transpose for 2D convolution: from (H, W, C_in, C_out) to (C_out, H, W, C_in)
        kernel = kernel.transpose((3, 0, 1, 2))
        print(f"2D kernel shape after transpose: {kernel.shape}")

    if inputs.ndim == 3:
        outputs = mx.conv1d(
            inputs,
            kernel,
            stride=strides[0],
            padding=padding,
            dilation=dilation_rate[0],
            groups=1,
        )
    elif inputs.ndim == 4:
        outputs = mx.conv2d(
            inputs,
            kernel,
            stride=strides,
            padding=padding,
            dilation=dilation_rate,
            groups=1,
        )
    print(f"Output shape after conv: {outputs.shape}")
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
