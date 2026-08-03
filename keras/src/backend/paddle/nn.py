import functools

import numpy as np
import paddle
import paddle.nn.functional as F

from keras.src import backend
from keras.src.backend.common.backend_utils import (
    compute_conv_transpose_output_crops_for_torch,
)
from keras.src.backend.config import floatx
from keras.src.backend.paddle.core import convert_to_tensor
from keras.src.backend.paddle.core import needs_reduced_precision_upcast
from keras.src.backend.paddle.core import to_paddle_dtype


def _upcast_reduced_precision(fn):
    """Run an elementwise activation in float32 when on CPU.

    Paddle registers no CPU kernels for many float16/bfloat16 elementwise
    ops. Computing in float32 and casting the result back keeps the public
    dtype contract intact while staying a no-op on accelerators, where the
    reduced-precision kernels do exist.
    """

    @functools.wraps(fn)
    def wrapper(x, *args, **kwargs):
        x = convert_to_tensor(x)
        if not needs_reduced_precision_upcast(x):
            return fn(x, *args, **kwargs)
        orig_dtype = x.dtype
        return fn(x.cast("float32"), *args, **kwargs).cast(orig_dtype)

    return wrapper


@_upcast_reduced_precision
def relu(x):
    return F.relu(convert_to_tensor(x))


@_upcast_reduced_precision
def relu6(x):
    return F.relu6(convert_to_tensor(x))


@_upcast_reduced_precision
def sigmoid(x):
    return F.sigmoid(convert_to_tensor(x))


def softmax(x, axis=-1):
    x = convert_to_tensor(x)
    orig_dtype = x.dtype
    # Paddle has no CPU `softmax` kernel for float16/bfloat16.
    upcast = needs_reduced_precision_upcast(x)
    if upcast:
        x = x.cast("float32")
    if axis is None:
        shape = x.shape
        x = x.flatten()
        x = F.softmax(x, axis=0)
        x = x.reshape(shape)
    else:
        x = F.softmax(x, axis=axis)
    if upcast:
        x = x.cast(orig_dtype)
    return x


@_upcast_reduced_precision
def softplus(x):
    return F.softplus(convert_to_tensor(x))


@_upcast_reduced_precision
def softsign(x):
    return F.softsign(convert_to_tensor(x))


@_upcast_reduced_precision
def silu(x):
    return F.silu(convert_to_tensor(x))


@_upcast_reduced_precision
def log_sigmoid(x):
    x = convert_to_tensor(x)
    return paddle.log(F.sigmoid(x))


@_upcast_reduced_precision
def leaky_relu(x, negative_slope=0.2):
    return F.leaky_relu(convert_to_tensor(x), negative_slope=negative_slope)


def prelu(x, alpha):
    return F.prelu(convert_to_tensor(x), convert_to_tensor(alpha))


@_upcast_reduced_precision
def elu(x, alpha=1.0):
    return F.elu(convert_to_tensor(x), alpha=alpha)


@_upcast_reduced_precision
def selu(x):
    return F.selu(convert_to_tensor(x))


@_upcast_reduced_precision
def gelu(x, approximate=True):
    return F.gelu(convert_to_tensor(x), approximate=approximate)


@_upcast_reduced_precision
def celu(x, alpha=1.0):
    return F.celu(convert_to_tensor(x), alpha=alpha)


@_upcast_reduced_precision
def tanh(x):
    return paddle.tanh(convert_to_tensor(x))


@_upcast_reduced_precision
def hard_sigmoid(x):
    return F.hardsigmoid(convert_to_tensor(x))


@_upcast_reduced_precision
def hard_silu(x):
    return F.hardswish(convert_to_tensor(x))


@_upcast_reduced_precision
def hard_tanh(x):
    return F.hardtanh(convert_to_tensor(x))


def one_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with paddle backend")
    dtype = dtype or floatx()
    x = convert_to_tensor(x, dtype="int64")
    # Paddle's `F.one_hot` raises an error for negative or out-of-range
    # indices, so clamp the indices and zero out the invalid rows afterwards.
    valid = paddle.logical_and(x >= 0, x < num_classes)
    out = F.one_hot(paddle.clip(x, min=0, max=num_classes - 1), num_classes)
    out = paddle.where(valid.unsqueeze(-1), out, paddle.zeros_like(out))
    if axis != -1 and axis != out.ndim - 1:
        out = paddle.moveaxis(out, -1, axis)
    return paddle.cast(out, dtype)


@_upcast_reduced_precision
def log_softmax(x, axis=-1):
    x = convert_to_tensor(x)
    if axis is None:
        shape = x.shape
        return log_softmax(x.flatten(), axis=0).reshape(shape)
    # Paddle's CPU `log_softmax` kernel clamps the shifted logits to -64,
    # which loses far too much accuracy (e.g. `log_softmax([100, -100])`
    # returns -64 instead of -200), so shift and normalize explicitly.
    max_x = paddle.max(x, axis=axis, keepdim=True)
    max_x = paddle.where(
        paddle.isfinite(max_x), max_x, paddle.zeros_like(max_x)
    )
    shifted = x - max_x
    exp_shifted = paddle.exp(shifted)
    # Paddle's CPU `exp` kernel clamps subnormal results to the smallest
    # normal float instead of flushing them to zero.
    exp_shifted = paddle.where(
        shifted == float("-inf"),
        paddle.zeros_like(exp_shifted),
        exp_shifted,
    )
    log_sum_exp = paddle.log(paddle.sum(exp_shifted, axis=axis, keepdim=True))
    return shifted - log_sum_exp


@_upcast_reduced_precision
def soft_shrink(x, threshold=0.5):
    x = convert_to_tensor(x)
    return F.softshrink(x, threshold=threshold)


@_upcast_reduced_precision
def hard_shrink(x, threshold=0.5):
    x = convert_to_tensor(x)
    return paddle.where(paddle.abs(x) > threshold, x, paddle.zeros_like(x))


@_upcast_reduced_precision
def tanh_shrink(x):
    return convert_to_tensor(x) - paddle.tanh(convert_to_tensor(x))


@_upcast_reduced_precision
def sparsemax(x, axis=-1):
    logits = convert_to_tensor(x)
    logits_sorted = paddle.sort(logits, axis=axis, descending=True)
    logits_cumsum = paddle.cumsum(logits_sorted, axis=axis)
    r = paddle.arange(1, paddle.shape(logits)[axis] + 1, dtype=logits.dtype)
    r_shape = [1] * len(logits.shape)
    r_shape[axis] = -1
    r = paddle.reshape(r, r_shape)
    support = logits_sorted - (logits_cumsum - 1) / r > 0
    k = paddle.sum(support.cast("int32"), axis=axis, keepdim=True)
    sum_selected = paddle.take_along_axis(logits_cumsum, k - 1, axis=axis)
    tau = (sum_selected - 1) / k.cast(logits.dtype)
    output = paddle.clip(logits - tau, min=0.0)
    return output


@_upcast_reduced_precision
def squareplus(x, b=4):
    x = convert_to_tensor(x)
    return 0.5 * (x + paddle.sqrt(x * x + b))


@_upcast_reduced_precision
def sparse_plus(x):
    x = convert_to_tensor(x)
    return paddle.where(
        x < -1,
        paddle.zeros_like(x),
        paddle.where(x > 1, x, 0.25 * (x + 1) ** 2),
    )


@_upcast_reduced_precision
def sparse_sigmoid(x):
    x = convert_to_tensor(x)
    return paddle.where(
        x < -1,
        paddle.zeros_like(x),
        paddle.where(x > 1, paddle.ones_like(x), 0.5 * x + 0.5),
    )


@_upcast_reduced_precision
def glu(x, axis=-1):
    x = convert_to_tensor(x)
    a, b = paddle.chunk(x, 2, axis=axis)
    return a * F.sigmoid(b)


@_upcast_reduced_precision
def threshold(x, threshold, default_value):
    x = convert_to_tensor(x)
    return paddle.where(x > threshold, x, paddle.full_like(x, default_value))


def multi_hot(x, num_classes, axis=-1, dtype="float32", sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with paddle backend")
    x = convert_to_tensor(x)
    reduction_axis = 1 if len(x.shape) > 1 else 0
    outputs = one_hot(x.cast("int32"), num_classes, axis=axis, dtype=dtype)
    # `paddle.amax` has no bool CPU kernel; reduce in int32 instead.
    if outputs.dtype == paddle.bool:
        return paddle.amax(outputs.cast("int32"), axis=reduction_axis).cast(
            "bool"
        )
    return paddle.amax(outputs, axis=reduction_axis)


def _standardize_tuple(x, n, name):
    if isinstance(x, int):
        return (x,) * n
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            return (x[0],) * n
        if len(x) == n:
            return tuple(x)
        raise ValueError(f"`{name}` should have length 1 or {n}. Received: {x}")
    raise ValueError(f"`{name}` should be int or tuple. Received: {x}")


def _to_channels_first(x, data_format):
    """Convert channels_last to channels_first."""
    if data_format == "channels_last":
        ndim = x.ndim
        if ndim == 3:  # (B, L, C) -> (B, C, L)
            return paddle.transpose(x, [0, 2, 1])
        elif ndim == 4:  # (B, H, W, C) -> (B, C, H, W)
            return paddle.transpose(x, [0, 3, 1, 2])
        elif ndim == 5:  # (B, D, H, W, C) -> (B, C, D, H, W)
            return paddle.transpose(x, [0, 4, 1, 2, 3])
    return x


def _to_channels_last(x, data_format):
    """Convert channels_first to channels_last."""
    if data_format == "channels_last":
        ndim = x.ndim
        if ndim == 3:
            return paddle.transpose(x, [0, 2, 1])
        elif ndim == 4:
            return paddle.transpose(x, [0, 2, 3, 1])
        elif ndim == 5:
            return paddle.transpose(x, [0, 2, 3, 4, 1])
    return x


def _cast_conv_inputs(inputs, kernel):
    """Cast to float32 for the dtypes paddle has no CPU conv kernel for."""
    unsupported = {
        paddle.float16,
        paddle.bfloat16,
        paddle.int64,
        paddle.int32,
        paddle.int16,
        paddle.int8,
        paddle.uint8,
        paddle.bool,
    }
    if inputs.dtype in unsupported:
        return inputs.cast("float32"), kernel.cast("float32")
    return inputs, kernel


def _same_padding(input_size, kernel_size, strides, dilation_rate):
    """Return explicit `"same"` padding as a flat `[before, after, ...]` list.

    Paddle's built-in `padding="SAME"` ignores the dilation rate, so compute
    the (possibly asymmetric) padding that Keras expects here instead.
    """
    pads = []
    for size, k, stride, dilation in zip(
        input_size, kernel_size, strides, dilation_rate
    ):
        effective_k = (k - 1) * dilation + 1
        out_size = (size + stride - 1) // stride
        total = max((out_size - 1) * stride + effective_k - size, 0)
        pads.extend([total // 2, total - total // 2])
    return pads


def _conv_padding(padding, kernel_size, strides, dilation_rate):
    """Compute padding values for paddle conv."""
    if isinstance(padding, str):
        if padding == "valid":
            return (
                [0] * len(kernel_size)
                if isinstance(kernel_size, (list, tuple))
                else [0]
            )
        elif padding == "same":
            # Paddle supports "same" padding directly in some cases
            return "same"
    if isinstance(padding, int):
        return (
            [padding] * len(kernel_size)
            if isinstance(kernel_size, (list, tuple))
            else [padding]
        )
    return list(padding)


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
    orig_dtype = inputs.dtype
    inputs, kernel = _cast_conv_inputs(inputs, kernel)
    num_spatial = inputs.ndim - 2

    strides = _standardize_tuple(strides, num_spatial, "strides")
    dilation_rate = _standardize_tuple(
        dilation_rate, num_spatial, "dilation_rate"
    )

    from keras.src.backend.config import standardize_data_format

    data_format = standardize_data_format(data_format)

    # Convert to channels_first for paddle
    inputs = _to_channels_first(inputs, data_format)

    # Kernel: Keras [*kernel_size, in_channels, out_channels]
    # Paddle: [out_channels, in_channels, *kernel_size]
    perm = [num_spatial + 1, num_spatial] + list(range(num_spatial))
    kernel = paddle.transpose(kernel, perm)

    in_channels = inputs.shape[1]
    kernel_in_channels = kernel.shape[1]
    if in_channels % kernel_in_channels != 0:
        raise ValueError(
            f"Input channels ({in_channels}) must be divisible by "
            f"kernel input channels ({kernel_in_channels})"
        )
    groups = in_channels // kernel_in_channels

    if padding == "same":
        pad_mode = _same_padding(
            inputs.shape[2:], kernel.shape[2:], strides, dilation_rate
        )
    else:
        pad_mode = _conv_padding(
            padding, kernel.shape[2:], strides, dilation_rate
        )

    if num_spatial == 1:
        out = F.conv1d(
            inputs,
            kernel,
            stride=strides[0],
            padding=pad_mode,
            dilation=dilation_rate[0],
            groups=groups,
        )
    elif num_spatial == 2:
        out = F.conv2d(
            inputs,
            kernel,
            stride=strides,
            padding=pad_mode,
            dilation=dilation_rate,
            groups=groups,
        )
    elif num_spatial == 3:
        out = F.conv3d(
            inputs,
            kernel,
            stride=strides,
            padding=pad_mode,
            dilation=dilation_rate,
            groups=groups,
        )
    else:
        raise ValueError(f"Unsupported number of spatial dims: {num_spatial}")

    out = _to_channels_last(out, data_format)
    if out.dtype != orig_dtype:
        out = out.cast(orig_dtype)
    return out


def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    orig_dtype = inputs.dtype
    inputs, kernel = _cast_conv_inputs(inputs, kernel)
    num_spatial = inputs.ndim - 2

    strides = _standardize_tuple(strides, num_spatial, "strides")
    dilation_rate = _standardize_tuple(
        dilation_rate, num_spatial, "dilation_rate"
    )

    from keras.src.backend.config import standardize_data_format

    data_format = standardize_data_format(data_format)

    inputs = _to_channels_first(inputs, data_format)

    # Kernel: Keras [kernel_size..., in_channels, depth_multiplier]
    # Paddle depthwise: [in_channels * depth_multiplier, 1, *kernel_size]
    perm = [num_spatial, num_spatial + 1] + list(range(num_spatial))
    kernel = paddle.transpose(kernel, perm)
    kernel_shape = paddle.shape(kernel)
    new_kernel_shape = paddle.concat(
        [
            (kernel_shape[0] * kernel_shape[1]).reshape([1]),
            paddle.to_tensor([1], dtype=kernel_shape.dtype),
            kernel_shape[2:],
        ]
    )
    kernel = paddle.reshape(kernel, new_kernel_shape)

    in_channels = inputs.shape[1]
    groups = in_channels

    if padding == "same":
        pad_mode = _same_padding(
            inputs.shape[2:], kernel.shape[2:], strides, dilation_rate
        )
    else:
        pad_mode = _conv_padding(
            padding, kernel.shape[2:], strides, dilation_rate
        )

    if num_spatial == 1:
        out = F.conv1d(
            inputs,
            kernel,
            stride=strides[0],
            padding=pad_mode,
            dilation=dilation_rate[0],
            groups=groups,
        )
    elif num_spatial == 2:
        out = F.conv2d(
            inputs,
            kernel,
            stride=strides,
            padding=pad_mode,
            dilation=dilation_rate,
            groups=groups,
        )
    elif num_spatial == 3:
        out = F.conv3d(
            inputs,
            kernel,
            stride=strides,
            padding=pad_mode,
            dilation=dilation_rate,
            groups=groups,
        )
    else:
        raise ValueError(f"Unsupported number of spatial dims: {num_spatial}")

    out = _to_channels_last(out, data_format)
    if out.dtype != orig_dtype:
        out = out.cast(orig_dtype)
    return out


def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    # Depthwise convolution
    x = depthwise_conv(
        inputs,
        depthwise_kernel,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
    )
    # Pointwise convolution (1x1)
    x = conv(
        x,
        pointwise_kernel,
        strides=1,
        padding="valid",
        data_format=data_format,
        dilation_rate=1,
    )
    return x


def conv_transpose(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    output_padding=None,
    data_format=None,
    dilation_rate=1,
):
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    orig_dtype = inputs.dtype
    inputs, kernel = _cast_conv_inputs(inputs, kernel)
    num_spatial = inputs.ndim - 2

    strides = _standardize_tuple(strides, num_spatial, "strides")
    dilation_rate = _standardize_tuple(
        dilation_rate, num_spatial, "dilation_rate"
    )

    from keras.src.backend.config import standardize_data_format

    data_format = standardize_data_format(data_format)

    # Paddle's `conv*d_transpose`, like torch's, only supports a symmetric
    # `padding` plus a right-side `output_padding`, which cannot express the
    # asymmetric padding Keras' "same" mode requires when the stride is
    # greater than 1. Run the transposed convolution unpadded (the largest
    # "natural" output) and crop the result instead; negative crops mean we
    # extend with zeros.
    crops = compute_conv_transpose_output_crops_for_torch(
        input_shape=inputs.shape,
        kernel_shape=kernel.shape,
        strides=strides,
        padding=padding,
        output_padding=output_padding,
        dilation_rate=dilation_rate,
    )

    inputs = _to_channels_first(inputs, data_format)

    # Kernel: Keras [*kernel_size, out_channels, in_channels]
    # Paddle: [in_channels, out_channels, *kernel_size]
    perm = [num_spatial + 1, num_spatial] + list(range(num_spatial))
    kernel = paddle.transpose(kernel, perm)

    conv_fn = [F.conv1d_transpose, F.conv2d_transpose, F.conv3d_transpose][
        num_spatial - 1
    ]
    out = conv_fn(
        inputs,
        kernel,
        stride=strides,
        padding=0,
        output_padding=0,
        dilation=dilation_rate,
    )

    # `out` is channels-first here, so the spatial dims start at axis 2.
    slices = [slice(None), slice(None)]
    for crop_left, crop_right in crops:
        slices.append(
            slice(max(0, crop_left), -crop_right if crop_right > 0 else None)
        )
    out = out[tuple(slices)]
    if any(cl < 0 or cr < 0 for cl, cr in crops):
        pads = [0, 0, 0, 0]
        for crop_left, crop_right in crops:
            pads.extend(
                [
                    -crop_left if crop_left < 0 else 0,
                    -crop_right if crop_right < 0 else 0,
                ]
            )
        out = F.pad(out, pads, mode="constant", value=0.0)

    out = _to_channels_last(out, data_format)
    if out.dtype != orig_dtype:
        out = out.cast(orig_dtype)
    return out


def _pool(inputs, pool_size, strides, padding, data_format, pool_type):
    inputs = convert_to_tensor(inputs)
    num_spatial = inputs.ndim - 2

    pool_size = _standardize_tuple(pool_size, num_spatial, "pool_size")
    if strides is None:
        strides = pool_size
    strides = _standardize_tuple(strides, num_spatial, "strides")

    from keras.src.backend.config import standardize_data_format

    data_format = standardize_data_format(data_format)

    inputs = _to_channels_first(inputs, data_format)

    if padding == "same":
        pad_mode = "same"
    else:
        pad_mode = 0

    if pool_type == "max":
        pool_fn = [F.max_pool1d, F.max_pool2d, F.max_pool3d][num_spatial - 1]
    else:
        pool_fn = [F.avg_pool1d, F.avg_pool2d, F.avg_pool3d][num_spatial - 1]

    if num_spatial == 1:
        out = pool_fn(
            inputs,
            kernel_size=pool_size[0],
            stride=strides[0],
            padding=pad_mode,
        )
    else:
        out = pool_fn(
            inputs, kernel_size=pool_size, stride=strides, padding=pad_mode
        )

    return _to_channels_last(out, data_format)


def avg_pool(
    inputs, pool_size, strides=None, padding="valid", data_format=None
):
    return _pool(inputs, pool_size, strides, padding, data_format, "avg")


def max_pool(
    inputs, pool_size, strides=None, padding="valid", data_format=None
):
    return _pool(inputs, pool_size, strides, padding, data_format, "max")


def average_pool(
    inputs, pool_size, strides=None, padding="valid", data_format=None
):
    return avg_pool(inputs, pool_size, strides, padding, data_format)


def adaptive_pool(inputs, output_size, data_format, pool_type):
    inputs = convert_to_tensor(inputs)
    num_spatial = inputs.ndim - 2

    from keras.src.backend.config import standardize_data_format

    data_format = standardize_data_format(data_format)

    inputs = _to_channels_first(inputs, data_format)

    if isinstance(output_size, int):
        output_size = [output_size] * num_spatial

    if pool_type == "max":
        pool_fn = [
            F.adaptive_max_pool1d,
            F.adaptive_max_pool2d,
            F.adaptive_max_pool3d,
        ][num_spatial - 1]
    else:
        pool_fn = [
            F.adaptive_avg_pool1d,
            F.adaptive_avg_pool2d,
            F.adaptive_avg_pool3d,
        ][num_spatial - 1]

    if num_spatial == 1:
        out = pool_fn(inputs, output_size=output_size[0])
    else:
        out = pool_fn(inputs, output_size=output_size)

    return _to_channels_last(out, data_format)


def adaptive_avg_pool(inputs, output_size, data_format=None):
    return adaptive_pool(inputs, output_size, data_format, "avg")


def adaptive_average_pool(inputs, output_size, data_format=None):
    return adaptive_avg_pool(inputs, output_size, data_format)


def adaptive_max_pool(inputs, output_size, data_format=None):
    return adaptive_pool(inputs, output_size, data_format, "max")


def global_average_pool(inputs, data_format=None):
    inputs = convert_to_tensor(inputs)

    from keras.src.backend.config import standardize_data_format

    data_format = standardize_data_format(data_format)

    if data_format == "channels_last":
        # Reduce over all spatial dims (everything except batch and channel)
        axes = list(range(1, inputs.ndim - 1))
    else:
        # Reduce over spatial dims (everything except batch and channel)
        axes = list(range(2, inputs.ndim))

    return paddle.mean(inputs, axis=axes)


def global_max_pool(inputs, data_format=None):
    inputs = convert_to_tensor(inputs)

    from keras.src.backend.config import standardize_data_format

    data_format = standardize_data_format(data_format)

    if data_format == "channels_last":
        axes = list(range(1, inputs.ndim - 1))
    else:
        axes = list(range(2, inputs.ndim))

    return paddle.max(inputs, axis=axes)


def moments(x, axes, keepdims=False, synchronized=False):
    if synchronized:
        raise NotImplementedError(
            "Argument synchronized=True is not supported with Paddle."
        )
    x = convert_to_tensor(x)
    # The dynamic range of float16 is too limited for statistics (and
    # paddle has no float16/bfloat16 CPU kernel for `mean`/`var`), so
    # compute in float32 and clip before casting back.
    orig_dtype = backend.standardize_dtype(x.dtype)
    need_cast = orig_dtype in ("float16", "bfloat16")
    if need_cast:
        x = x.cast("float32")
    mean = paddle.mean(x, axis=axes, keepdim=keepdims)
    variance = paddle.var(x, axis=axes, keepdim=keepdims, unbiased=False)
    if need_cast:
        info = np.finfo(orig_dtype)
        mean = paddle.clip(mean, float(info.min), float(info.max))
        variance = paddle.clip(variance, float(info.min), float(info.max))
        mean = mean.cast(to_paddle_dtype(orig_dtype))
        variance = variance.cast(to_paddle_dtype(orig_dtype))
    return mean, variance


def batch_normalization(
    x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3
):
    x = convert_to_tensor(x)
    mean = convert_to_tensor(mean)
    variance = convert_to_tensor(variance)

    ndim = x.ndim
    axis = axis + ndim if axis < 0 else axis
    shape = [1] * ndim
    shape[axis] = -1

    mean = paddle.reshape(mean, shape)
    variance = paddle.reshape(variance, shape)
    x_norm = (x - mean) / paddle.sqrt(variance + epsilon)
    if scale is not None:
        x_norm = x_norm * paddle.reshape(convert_to_tensor(scale), shape)
    if offset is not None:
        x_norm = x_norm + paddle.reshape(convert_to_tensor(offset), shape)
    return x_norm


def ctc_decode(
    inputs,
    sequence_lengths,
    strategy="greedy",
    beam_width=100,
    top_paths=1,
    merge_repeated=True,
    mask_index=0,
):
    if strategy not in ("greedy", "beam_search"):
        raise ValueError(
            f"Invalid strategy {strategy}. Supported values are "
            "'greedy' and 'beam_search'."
        )
    inputs = convert_to_tensor(inputs)
    sequence_lengths = convert_to_tensor(sequence_lengths, dtype="int32")
    # `argmax`, `max` and `where` have no float16/bfloat16 CPU kernels.
    # The scores are float32 or wider anyway, per the op contract.
    if needs_reduced_precision_upcast(inputs):
        inputs = inputs.cast("float32")
    inputs_shape = paddle.shape(inputs)
    batch_size = inputs_shape[0]
    max_length = inputs_shape[1]
    num_classes = inputs.shape[2]

    if strategy == "greedy":
        indices = paddle.argmax(inputs, axis=-1).cast("int32")
        scores = paddle.max(inputs, axis=-1)

        seqlen_mask = paddle.arange(max_length, dtype="int32")
        seqlen_mask = seqlen_mask.unsqueeze(0)
        seqlen_mask = seqlen_mask >= sequence_lengths.unsqueeze(1)

        blank_idx = num_classes - 1 if mask_index == -1 else mask_index
        indices = paddle.where(
            seqlen_mask, paddle.to_tensor(blank_idx, dtype="int32"), indices
        )
        scores = paddle.where(seqlen_mask, paddle.zeros_like(scores), scores)

        if merge_repeated:
            repeat = indices[:, 1:] == indices[:, :-1]
            zeros = paddle.zeros([batch_size, 1], dtype="bool")
            repeat = paddle.concat([zeros, repeat], axis=1)
            indices = paddle.where(
                repeat, paddle.to_tensor(blank_idx, dtype="int32"), indices
            )

        invalid_mask = indices == blank_idx
        indices = paddle.where(
            invalid_mask, paddle.to_tensor(-1, dtype="int32"), indices
        )

        order = paddle.arange(max_length, dtype="int32").unsqueeze(0)
        order = paddle.broadcast_to(order, [batch_size, max_length])
        order = paddle.where(
            invalid_mask, paddle.to_tensor(max_length, dtype="int32"), order
        )
        order = paddle.argsort(order, axis=-1)
        indices = paddle.take_along_axis(indices, order, axis=-1)

        scores = -paddle.sum(scores, axis=1).unsqueeze(1)
        indices = indices.unsqueeze(0)
        return indices, scores
    raise NotImplementedError(
        "CTC decode strategy 'beam_search' is not supported with the "
        "paddle backend."
    )


def psnr(x1, x2, max_val):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    if x1.shape != x2.shape:
        raise ValueError(
            f"Input shapes {x1.shape} and {x2.shape} must "
            "match for PSNR calculation. "
        )
    max_val = convert_to_tensor(max_val, dtype=x2.dtype)
    mse = paddle.mean(paddle.square(x1 - x2))
    return 20 * paddle.log10(max_val) - 10 * paddle.log10(mse)


def _get_large_negative(dtype):
    dtype = backend.standardize_dtype(dtype)
    val = 65500.0 if dtype == "float16" else 3.38953e38
    return paddle.to_tensor(val * -0.7, dtype=to_paddle_dtype(dtype))


def _apply_masks(logits, mask, is_causal):
    if mask is None and not is_causal:
        return logits

    combined_mask = paddle.ones_like(logits).cast("bool")
    if mask is not None:
        mask = convert_to_tensor(mask).cast("bool")
        combined_mask = paddle.logical_and(combined_mask, mask)

    if is_causal:
        T, S = logits.shape[2], logits.shape[3]
        causal_mask = paddle.tril(paddle.ones([T, S], dtype="int32")).cast(
            "bool"
        )
        causal_mask = causal_mask[None, None, :, :]
        combined_mask = paddle.logical_and(combined_mask, causal_mask)

    return paddle.where(
        combined_mask, logits, _get_large_negative(logits.dtype)
    )


def _dot_product_attention_xla(query, key, value, bias, mask, is_causal, scale):
    original_dtype = key.dtype
    logits_dtype = backend.result_type(
        backend.standardize_dtype(query.dtype), "float32"
    )
    # `einsum` lacks CPU kernels for reduced precision, so accumulate the
    # logits in a wider dtype.
    if backend.standardize_dtype(key.dtype) in ("float16", "bfloat16"):
        query = query.cast("float32")
        key = key.cast("float32")
        value = value.cast("float32")
    logits = paddle.einsum("BTNH,BSNH->BNTS", query, key)
    logits = logits.cast(to_paddle_dtype(logits_dtype))
    logits = logits * paddle.to_tensor(scale, dtype=logits.dtype)

    if bias is not None:
        logits = (logits + convert_to_tensor(bias)).cast(logits.dtype)

    padded_logits = _apply_masks(logits, mask, is_causal)

    # Softmax is always carried out in fp32.
    padded_logits = padded_logits.cast("float32")
    probs = F.softmax(padded_logits, axis=-1).cast(original_dtype)
    if backend.standardize_dtype(probs.dtype) in ("float16", "bfloat16"):
        probs = probs.cast("float32")
        value = value.cast("float32")
    encoded = paddle.einsum("BNTS,BSNH->BTNH", probs, value)
    return encoded.cast(original_dtype)


def dot_product_attention(
    query,
    key,
    value,
    bias=None,
    mask=None,
    scale=None,
    is_causal=False,
    flash_attention=None,
    attn_logits_soft_cap=None,
):
    if flash_attention:
        raise ValueError(
            "Flash attention is not supported in the paddle backend."
        )
    query = convert_to_tensor(query)
    key = convert_to_tensor(key)
    value = convert_to_tensor(value)
    if len(query.shape) != 4:
        raise ValueError(
            "`dot_product_attention` only supports 4D inputs. "
            f"Received: query.shape={query.shape}, key.shape={key.shape}, "
            f"value.shape={value.shape}."
        )
    compute_dtype = backend.result_type(
        backend.standardize_dtype(query.dtype),
        backend.standardize_dtype(key.dtype),
        backend.standardize_dtype(value.dtype),
    )
    paddle_compute_dtype = to_paddle_dtype(compute_dtype)
    query = query.cast(paddle_compute_dtype)
    key = key.cast(paddle_compute_dtype)
    value = value.cast(paddle_compute_dtype)

    if attn_logits_soft_cap is not None:
        raise NotImplementedError(
            "`attn_logits_soft_cap` is not supported with the paddle backend."
        )

    H = key.shape[-1]
    scale = (1.0 / (H**0.5)) if scale is None else scale
    return _dot_product_attention_xla(
        query, key, value, bias, mask, is_causal, scale
    )


def binary_crossentropy(target, output, from_logits=False):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)

    if tuple(target.shape) != tuple(output.shape):
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    if target.dtype != output.dtype:
        target = paddle.cast(target, output.dtype)
    if from_logits:
        return F.binary_cross_entropy_with_logits(
            output, target, reduction="none"
        )
    epsilon = backend.epsilon()
    output = paddle.clip(output, min=epsilon, max=1.0 - epsilon)
    return F.binary_cross_entropy(output, target, reduction="none")


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)

    if tuple(target.shape) != tuple(output.shape):
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
    if target.dtype != output.dtype:
        target = paddle.cast(target, output.dtype)

    if from_logits:
        log_prob = log_softmax(output, axis=axis)
    else:
        # Normalize so that the values form a proper probability
        # distribution, matching the other Keras backends.
        output = output / paddle.sum(output, axis=axis, keepdim=True)
        epsilon = backend.epsilon()
        output = paddle.clip(output, min=epsilon, max=1.0 - epsilon)
        log_prob = paddle.log(output)
    return -paddle.sum(target * log_prob, axis=axis)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = convert_to_tensor(target, dtype="int64")
    output = convert_to_tensor(output)
    if len(target.shape) == len(output.shape) and target.shape[-1] == 1:
        target = paddle.squeeze(target, axis=-1)

    if len(output.shape) < 1:
        raise ValueError(
            "Argument `output` must be at least rank 1. "
            "Received: "
            f"output.shape={output.shape}"
        )
    if tuple(target.shape) != tuple(output.shape[:-1]):
        raise ValueError(
            "Arguments `target` and `output` must have the same shape "
            "up until the last dimension: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        log_prob = log_softmax(output, axis=axis)
    else:
        output = output / paddle.sum(output, axis=axis, keepdim=True)
        epsilon = backend.epsilon()
        output = paddle.clip(output, min=epsilon, max=1.0 - epsilon)
        log_prob = paddle.log(output)
    target_one_hot = one_hot(
        target, output.shape[axis], axis=axis, dtype=log_prob.dtype
    )
    return -paddle.sum(target_one_hot * log_prob, axis=axis)


def ctc_loss(target, output, target_length, output_length, mask_index=0):
    target = convert_to_tensor(target, dtype="int32")
    output = convert_to_tensor(output, dtype="float32")
    target_length = convert_to_tensor(target_length, dtype="int64")
    output_length = convert_to_tensor(output_length, dtype="int64")

    # Paddle expects logits of shape (max_length, batch_size, num_classes).
    output = paddle.transpose(output, [1, 0, 2])
    logits = F.log_softmax(output, axis=-1)
    return F.ctc_loss(
        logits,
        target,
        output_length,
        target_length,
        blank=mask_index,
        reduction="none",
    )


def fold(x, output_size, kernel_size, dilation=1, padding=0, stride=1):
    """Paddle implementation of Fold.
    Combine an array of sliding local blocks into a large tensor (col2im).

    Args:
        x: 3-D tensor, shape (N, C*kH*kW, L)
        output_size: int or (oH, oW)
        kernel_size: int or (kH, kW)
        dilation: int or (dH, dW), default 1
        padding: int or (pH, pW), default 0
        stride: int or (sH, sW), default 1

    Returns:
        4-D tensor, shape (N, C, oH, oW)
    """
    x = convert_to_tensor(x)
    return F.fold(
        x,
        output_sizes=output_size,
        kernel_sizes=kernel_size,
        dilations=dilation,
        paddings=padding,
        strides=stride,
    )


def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    """Paddle implementation of Unfold.
    Extract sliding local blocks from a **NCHW** batched image tensor.

    Args:
        input: 4-D tensor, shape (N, C, H, W)
        kernel_size: int or (kH, kW)
        dilation: int or (dH, dW), default 1
        padding: int or (pH, pW), default 0
        stride: int or (sH, sW), default 1

    Returns:
        3-D tensor, shape (N, C*kH*kW, L)
    """
    return F.unfold(
        input,
        kernel_sizes=kernel_size,
        dilations=dilation,
        paddings=padding,
        strides=stride,
    )


def depth_to_space(x, block_size, data_format="channels_last"):
    x = convert_to_tensor(x)
    from keras.src.backend.config import standardize_data_format

    data_format = standardize_data_format(data_format)

    if data_format == "channels_last":
        n, h, w, c = x.shape
        new_c = c // (block_size**2)
        x = paddle.reshape(x, [n, h, w, block_size, block_size, new_c])
        x = paddle.transpose(x, [0, 1, 3, 2, 4, 5])
        return paddle.reshape(x, [n, h * block_size, w * block_size, new_c])

    n, c, h, w = x.shape
    new_c = c // (block_size**2)
    x = paddle.reshape(x, [n, new_c, block_size, block_size, h, w])
    x = paddle.transpose(x, [0, 1, 4, 2, 5, 3])
    return paddle.reshape(x, [n, new_c, h * block_size, w * block_size])


def space_to_depth(x, block_size, data_format="channels_last"):
    x = convert_to_tensor(x)
    from keras.src.backend.config import standardize_data_format

    data_format = standardize_data_format(data_format)

    if data_format == "channels_last":
        n, h, w, c = x.shape
        new_h = h // block_size
        new_w = w // block_size
        x = paddle.reshape(x, [n, new_h, block_size, new_w, block_size, c])
        x = paddle.transpose(x, [0, 1, 3, 2, 4, 5])
        return paddle.reshape(x, [n, new_h, new_w, c * block_size**2])

    n, c, h, w = x.shape
    new_h = h // block_size
    new_w = w // block_size
    x = paddle.reshape(x, [n, c, new_h, block_size, new_w, block_size])
    x = paddle.transpose(x, [0, 1, 3, 5, 2, 4])
    return paddle.reshape(x, [n, c * block_size**2, new_h, new_w])
