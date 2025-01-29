import builtins
import math
import operator
from itertools import accumulate

import mlx.core as mx
import mlx.nn as nn

from keras.src.backend import result_type
from keras.src.backend import standardize_data_format
from keras.src.backend import standardize_dtype
from keras.src.backend.common.backend_utils import (
    compute_conv_transpose_padding_args_for_mlx,
)
from keras.src.backend.common.backend_utils import (
    compute_transpose_padding_args_for_mlx,
)
from keras.src.backend.config import epsilon
from keras.src.backend.mlx.core import cast
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.backend.mlx.core import scan
from keras.src.backend.mlx.core import to_mlx_dtype
from keras.src.backend.mlx.numpy import flip
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


def celu(x, alpha=1.0):
    x = convert_to_tensor(x)
    return nn.celu(x, alpha=alpha)


def softmax(x, axis=-1):
    x = convert_to_tensor(x)
    return mx.softmax(x, axis=axis)


def log_softmax(x, axis=-1):
    x = convert_to_tensor(x)
    return x - mx.logsumexp(x, axis=axis, keepdims=True)


def _calculate_padding(input_shape, pool_size, strides):
    ndim = len(input_shape)

    padding = ()
    for d in range(ndim):
        pad = max(0, (pool_size[d] - 1) - ((input_shape[d] - 1) % strides[d]))
        padding = padding + (pad,)

    return [(p // 2, (p + 1) // 2) for p in padding]


def _non_overlapping_sliding_windows(x, shape, window_shape):
    # Compute the intermediate shape
    new_shape = [shape[0]]
    for s, w in zip(shape[1:], window_shape):
        new_shape.append(s // w)
        new_shape.append(w)
    new_shape.append(shape[-1])

    last_axis = len(new_shape) - 1
    axis_order = [
        0,
        *range(1, last_axis, 2),
        *range(2, last_axis, 2),
        last_axis,
    ]

    x = x.reshape(new_shape)
    x = x.transpose(axis_order)
    return x


def _sliding_windows(x, window_shape, window_strides):
    if x.ndim < 3:
        raise ValueError(
            "To extract sliding windows at least 1 spatial dimension "
            f"(3 total) is needed but the input only has {x.ndim} dimension(s)."
        )

    spatial_dims = x.shape[1:-1]
    if not (len(spatial_dims) == len(window_shape) == len(window_strides)):
        raise ValueError(
            "To extract sliding windows, the lengths of window_shape and "
            "window_strides must be equal to the signal's spatial dimensions. "
            f"However, the signal has spatial_dims={spatial_dims} while "
            f"window_shape={window_shape} and window_strides={window_strides}."
        )

    shape = x.shape
    if all(
        window == stride and size % window == 0
        for size, window, stride in zip(
            spatial_dims, window_shape, window_strides
        )
    ):
        return _non_overlapping_sliding_windows(x, shape, window_shape)

    strides = list(
        reversed(list(accumulate(reversed(shape + (1,)), operator.mul)))
    )[1:]

    # Compute the output shape
    final_shape = [shape[0]]
    final_shape += [
        (size - window) // stride + 1
        for size, window, stride in zip(
            spatial_dims, window_shape, window_strides
        )
    ]
    final_shape += window_shape
    final_shape += [shape[-1]]

    # Compute the output strides
    final_strides = strides[:1]
    final_strides += [
        og_stride * stride
        for og_stride, stride in zip(strides[1:-1], window_strides)
    ]
    final_strides += strides[1:-1]
    final_strides += strides[-1:]  # should always be [1]

    return mx.as_strided(x, final_shape, final_strides)


def _pool(
    inputs, pool_size, strides, padding, padding_value, data_format, pooling_fn
):
    if padding not in ("same", "valid"):
        raise ValueError(
            f"Invalid padding '{padding}', must be 'same' or 'valid'."
        )

    if data_format == "channels_first":
        # mlx expects channels_last
        inputs = inputs.transpose(0, *range(2, inputs.ndim), 1)

    if padding == "same":
        pads = _calculate_padding(inputs.shape[1:-1], pool_size, strides)

        if any(p[1] > 0 for p in pads):
            inputs = mx.pad(
                inputs,
                [(0, 0)] + pads + [(0, 0)],
                constant_values=padding_value,
            )

    inputs = _sliding_windows(inputs, pool_size, strides)

    axes = tuple(range(-len(pool_size) - 1, -1, 1))
    result = pooling_fn(inputs, axes)

    if data_format == "channels_first":
        result = result.transpose(0, -1, *range(1, result.ndim - 1))
    return result


def max_pool(
    inputs, pool_size, strides=None, padding="valid", data_format=None
):
    inputs = convert_to_tensor(inputs)
    data_format = standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2
    pool_size = standardize_tuple(pool_size, num_spatial_dims, "pool_size")
    strides = pool_size if strides is None else strides
    strides = standardize_tuple(strides, num_spatial_dims, "strides")

    return _pool(
        inputs, pool_size, strides, padding, -mx.inf, data_format, mx.max
    )


def average_pool(
    inputs, pool_size, strides=None, padding="valid", data_format=None
):
    inputs = convert_to_tensor(inputs)
    data_format = standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2
    pool_size = standardize_tuple(pool_size, num_spatial_dims, "pool_size")
    strides = pool_size if strides is None else strides
    strides = standardize_tuple(strides, num_spatial_dims, "strides")

    # Create a pool by applying the sum function in each window
    pooled = _pool(
        inputs, pool_size, strides, padding, 0.0, data_format, mx.sum
    )
    if padding == "valid":
        # No padding needed. Divide by the size of the pool which gives
        # the average
        return pooled / math.prod(pool_size)
    else:
        # Create a tensor of ones of the same shape of inputs.
        # Then create a pool, padding by zero and using sum as function.
        # This will create a tensor of the smae dimensions as pooled tensor
        # with values being the sum.
        # By dividing pooled by windows_counts, we get the average while
        # skipping the padded values.
        window_counts = _pool(
            mx.ones(inputs.shape, inputs.dtype),
            pool_size,
            strides,
            padding,
            0.0,
            data_format,
            mx.sum,
        )
        return pooled / window_counts


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
    data_format = standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2

    strides = standardize_tuple(strides, num_spatial_dims, "strides")
    dilation_rate = standardize_tuple(
        dilation_rate, num_spatial_dims, "dilation_rate"
    )

    if data_format == "channels_first":
        # mlx expects channels_last
        inputs = inputs.transpose(0, *range(2, inputs.ndim), 1)

    # mlx expects kernel with (out_channels, spatial..., in_channels)
    kernel = kernel.transpose(-1, *range(kernel.ndim - 2), -2)

    kernel_spatial_shape = kernel.shape[1:-1]
    input_spatial_shape = inputs.shape[1:-1]
    mlx_padding = compute_transpose_padding_args_for_mlx(
        padding,
        input_spatial_shape,
        kernel_spatial_shape,
        dilation_rate,
        strides,
    )

    channels = inputs.shape[-1]
    kernel_in_channels = kernel.shape[-1]
    if channels % kernel_in_channels > 0:
        raise ValueError(
            "The number of input channels must be evenly divisible by "
            f"kernel's in_channels. Received input channels {channels} and "
            f"kernel in_channels {kernel_in_channels}. "
        )
    groups = channels // kernel_in_channels

    result = mx.conv_general(
        inputs,
        kernel,
        stride=strides,
        padding=mlx_padding,
        kernel_dilation=dilation_rate,
        input_dilation=1,
        groups=groups,
        flip=False,
    )
    if data_format == "channels_first":
        result = result.transpose(0, -1, *range(1, result.ndim - 1))

    return result


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
    data_format = standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2

    strides = standardize_tuple(strides, num_spatial_dims, "strides")
    dilation_rate = standardize_tuple(
        dilation_rate, num_spatial_dims, "dilation_rate"
    )

    if data_format == "channels_first":
        # mlx expects channels_last
        inputs = inputs.transpose(0, *range(2, inputs.ndim), 1)

    feature_group_count = inputs.shape[-1]

    # reshape first for depthwise conv, then transpose to expected mlx format
    kernel = kernel.reshape(
        *iter(kernel.shape[:-2]), 1, feature_group_count * kernel.shape[-1]
    )
    # mlx expects kernel with (out_channels, spatial..., in_channels)
    kernel = kernel.transpose(-1, *range(kernel.ndim - 2), -2)

    kernel_spatial_shape = kernel.shape[1:-1]
    input_spatial_shape = inputs.shape[1:-1]
    mlx_padding = compute_transpose_padding_args_for_mlx(
        padding,
        input_spatial_shape,
        kernel_spatial_shape,
        dilation_rate,
        strides,
    )

    result = mx.conv_general(
        inputs,
        kernel,
        stride=strides,
        padding=mlx_padding,
        kernel_dilation=dilation_rate,
        input_dilation=1,
        groups=feature_group_count,
        flip=False,
    )
    if data_format == "channels_first":
        result = result.transpose(0, -1, *range(1, result.ndim - 1))

    return result


def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    data_format = standardize_data_format(data_format)
    depthwise_conv_output = depthwise_conv(
        inputs,
        depthwise_kernel,
        strides,
        padding,
        data_format,
        dilation_rate,
    )
    return conv(
        depthwise_conv_output,
        pointwise_kernel,
        strides=1,
        padding="valid",
        data_format=data_format,
        dilation_rate=dilation_rate,
    )


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
    data_format = standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2

    strides = standardize_tuple(strides, num_spatial_dims, "strides")
    dilation_rate = standardize_tuple(
        dilation_rate, num_spatial_dims, "dilation_rate"
    )
    if output_padding is not None:
        output_padding = standardize_tuple(
            output_padding, num_spatial_dims, "output_padding"
        )

    if data_format == "channels_first":
        # mlx expects channels_last
        inputs = inputs.transpose(0, *range(2, inputs.ndim), 1)

    # mlx expects kernel with (out_channels, spatial..., in_channels)
    kernel = kernel.transpose(-2, *range(kernel.ndim - 2), -1)
    kernel_spatial_shape = kernel.shape[1:-1]

    mlx_padding = compute_conv_transpose_padding_args_for_mlx(
        padding,
        num_spatial_dims,
        kernel_spatial_shape,
        dilation_rate,
        strides,
        output_padding,
    )

    channels = inputs.shape[-1]
    kernel_in_channels = kernel.shape[-1]
    if channels % kernel_in_channels > 0:
        raise ValueError(
            "The number of input channels must be evenly divisible by "
            f"kernel's in_channels. Received input channels {channels} and "
            f"kernel in_channels {kernel_in_channels}. "
        )
    groups = channels // kernel_in_channels

    result = mx.conv_general(
        inputs,
        kernel,
        stride=1,  # stride is handled by input_dilation
        padding=mlx_padding,
        kernel_dilation=dilation_rate,
        input_dilation=strides,
        groups=groups,
        flip=True,
    )

    if data_format == "channels_first":
        result = result.transpose(0, -1, *range(1, result.ndim - 1))

    return result


def one_hot(x, num_classes, axis=-1, dtype="float32", sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with mlx backend")
    x = convert_to_tensor(x, dtype=mx.int32)
    dtype = to_mlx_dtype(standardize_dtype(dtype))

    r = mx.arange(num_classes, dtype=mx.int32)

    x_expanded = mx.expand_dims(x, -1)
    output = x_expanded == r
    output = output.astype(dtype)

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
        log_prob = log_softmax(output)
    else:
        output = output / mx.sum(output, axis=axis, keepdims=True)
        output = mx.clip(output, epsilon(), 1 - epsilon())
        log_prob = mx.log(output)

    return -mx.sum(target * log_prob, axis=axis)


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
        log_prob = log_softmax(output)
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
    variance = mx.var(x, axis=axes, keepdims=True)

    if not keepdims:
        mean = mean.squeeze(axes)
        variance = variance.squeeze(axes)

    if need_cast:
        # clip values to avoid overflow/underflow when casting back to float16
        mean = mx.clip(mean, mx.finfo(mx.float16).min, mx.finfo(mx.float16).max)
        variance = mx.clip(
            variance, mx.finfo(mx.float16).min, mx.finfo(mx.float16).max
        )
        mean = mean.astype(ori_dtype)
        variance = variance.astype(ori_dtype)

    return mean, variance


def batch_normalization(
    x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3
):
    x = convert_to_tensor(x)
    mean = convert_to_tensor(mean)
    variance = convert_to_tensor(variance)
    shape = [1] * len(x.shape)
    shape[axis] = mean.shape[0]
    mean = mx.reshape(mean, shape)
    variance = mx.reshape(variance, shape)

    inv = mx.rsqrt(variance + epsilon)
    if scale is not None:
        scale = convert_to_tensor(scale)
        scale = mx.reshape(scale, shape)
        inv = inv * scale

    res = -mean * inv
    if offset is not None:
        offset = convert_to_tensor(offset)
        offset = mx.reshape(offset, shape)
        res = res + offset

    return mx.add(x * inv, res)


def ctc_loss(target, output, target_length, output_length, mask_index=0):
    # Ref: https://github.com/google-deepmind/optax
    # optax.ctc_loss_with_forward_probs
    target = convert_to_tensor(target, dtype="int32")
    output = convert_to_tensor(output)
    target_length = convert_to_tensor(target_length, "int32")
    output_length = convert_to_tensor(output_length, "int32")
    batch_size, max_input_length, num_classes = output.shape
    batch_size, max_label_length = target.shape
    log_epsilon = -1e5

    # Ensure that the dtype promotion behavior matchs that of `tf.nn.ctc_loss`
    dtype = result_type(output.dtype, "float32")
    dtype = to_mlx_dtype(standardize_dtype(dtype))
    output = cast(output, dtype)

    def _lengths_to_paddings(lengths, max_length):
        indices = mx.arange(max_length).reshape(
            (1,) * lengths.ndim + (max_length,)
        )
        lengths = mx.expand_dims(lengths, axis=-1)
        elem_valid = indices < lengths
        return mx.logical_not(elem_valid)

    target_paddings = _lengths_to_paddings(target_length, max_label_length)
    output_paddings = _lengths_to_paddings(output_length, max_input_length)
    target_paddings = target_paddings.astype(output.dtype)
    output_paddings = output_paddings.astype(output.dtype)

    logprobs = log_softmax(output)
    label_lengths = max_label_length - mx.sum(target_paddings, axis=1).astype(
        mx.int32
    )

    # repeat[b, n] == 1.0 when label[b, n] == label[b, n+1].
    repeat = (target[:, :-1] == target[:, 1:]).astype(mx.float32)
    repeat = mx.pad(repeat, ((0, 0), (0, 1)))

    logprobs_phi = logprobs[:, :, mask_index : mask_index + 1]  # [B, T, 1]
    logprobs_phi = mx.transpose(logprobs_phi, (1, 0, 2))  # [T, B, 1]

    _one_hot = one_hot(target, num_classes=num_classes)  # [B, N, K]
    logprobs_emit = mx.einsum("btk,bnk->btn", logprobs, _one_hot)
    logprobs_emit = mx.transpose(logprobs_emit, (1, 0, 2))  # [T, B, N]

    # [B, N]
    logalpha_phi_init = (
        mx.ones((batch_size, max_label_length + 1), dtype=output.dtype)
        * log_epsilon
    )
    logalpha_phi_init[:, 0] = 0.0
    logalpha_emit_init = (
        mx.ones((batch_size, max_label_length), dtype=output.dtype)
        * log_epsilon
    )

    def update_phi_score(phi, added_score):
        # Update `phi[:, 1:]`` with adding `added_score` in log space.
        return mx.concatenate(
            [phi[:, :1], mx.logaddexp(phi[:, 1:], added_score)], axis=-1
        )

    def loop_body(prev, x):
        prev_phi, prev_emit = prev
        # emit-to-phi epsilon transition, except if the next label is repetition
        prev_phi_orig = prev_phi
        prev_phi = update_phi_score(prev_phi, prev_emit + log_epsilon * repeat)

        logprob_emit, logprob_phi, pad = x

        # phi-to-emit transition
        next_emit = mx.logaddexp(
            prev_phi[:, :-1] + logprob_emit, prev_emit + logprob_emit
        )
        # self-loop transition
        next_phi = prev_phi + logprob_phi
        # emit-to-phi blank transition only when the next label is repetition
        next_phi = update_phi_score(
            next_phi, prev_emit + logprob_phi + log_epsilon * (1.0 - repeat)
        )

        pad = pad.reshape((batch_size, 1))
        next_emit = pad * prev_emit + (1.0 - pad) * next_emit
        next_phi = pad * prev_phi_orig + (1.0 - pad) * next_phi

        return (next_phi, next_emit), (next_phi, next_emit)

    xs = (logprobs_emit, logprobs_phi, output_paddings.transpose((1, 0)))

    _, (logalpha_phi, logalpha_emit) = scan(
        loop_body, (logalpha_phi_init, logalpha_emit_init), xs
    )

    # last row needs to be updated with the last epsilon transition
    logalpha_phi_last = update_phi_score(logalpha_phi[-1], logalpha_emit[-1])
    logalpha_phi[-1] = logalpha_phi_last

    # extract per_seq_loss
    # [B, N+1]
    _one_hot = one_hot(label_lengths, num_classes=max_label_length + 1)
    per_seq_loss = -mx.einsum("bn,bn->b", logalpha_phi_last, _one_hot)
    return per_seq_loss


def _ctc_greedy_decode(
    inputs,
    sequence_lengths,
    merge_repeated=True,
    mask_index=None,
):
    inputs = convert_to_tensor(inputs)
    sequence_lengths = convert_to_tensor(sequence_lengths, dtype="int32")
    batch_size, max_length, num_classes = inputs.shape

    if mask_index is None:
        mask_index = num_classes - 1

    indices = mx.argmax(inputs, axis=-1).astype(
        mx.int32
    )  # mlx argmax outputs uint32
    scores = mx.max(inputs, axis=-1)

    seqlen_mask = mx.arange(max_length)[None, :]
    seqlen_mask = seqlen_mask >= sequence_lengths[:, None]

    indices = mx.where(seqlen_mask, mask_index, indices)
    scores = mx.where(seqlen_mask, 0.0, scores)

    if merge_repeated:
        repeat_mask = indices[:, 1:] == indices[:, :-1]
        repeat_mask = mx.pad(repeat_mask, ((0, 0), (1, 0)))
        indices = mx.where(repeat_mask, mask_index, indices)

    # We set to -1 for blank labels
    invalid_mask = indices == mask_index
    indices = mx.where(invalid_mask, -1, indices)

    # We rearrange the indices by moving `mask_index` to the end of the array
    order = mx.expand_dims(mx.arange(max_length), axis=0)  # [1, N]
    order = mx.tile(order, (batch_size, 1))  # [B, N]
    order = mx.where(invalid_mask, max_length, order)
    order = mx.argsort(order, axis=-1)
    indices = mx.take_along_axis(indices, order, axis=-1)

    scores = -mx.sum(scores, axis=1)[:, None]
    indices = mx.expand_dims(indices, axis=0)
    return indices, scores


def _unique_2d(arr, size=None, fill_value=0, return_inverse=True):
    if arr.ndim != 2:
        raise ValueError(
            f"Invalid dimension: {arr.ndim}"
            "unique_2d only supports 2 dimensional arrays"
        )

    unique_set = set()
    indices = []

    for i, row in enumerate(arr):
        row_tuple = tuple(row.tolist())
        if row_tuple not in unique_set:
            unique_set.add(row_tuple)
            indices.append(i)

    unique_vals = mx.array([list(t) for t in sorted(unique_set)])

    if size is not None:
        pad_rows = size - len(unique_vals)
        if pad_rows > 0:
            padding = mx.full((pad_rows, arr.shape[1]), fill_value)
            unique_vals = mx.concatenate([unique_vals, padding])

    unique_dict = {tuple(row.tolist()): i for i, row in enumerate(unique_vals)}
    inverse = mx.array([unique_dict[tuple(row.tolist())] for row in arr])
    if return_inverse:
        return unique_vals, inverse
    else:
        return unique_vals


def _ctc_beam_search_decode(
    inputs,
    sequence_lengths,
    beam_width=100,
    top_paths=1,
    mask_index=None,
):
    inputs = convert_to_tensor(inputs)
    sequence_lengths = convert_to_tensor(sequence_lengths)

    batch_size, max_seq_len, num_classes = inputs.shape
    inputs = log_softmax(inputs)
    seqlen_mask = mx.arange(max_seq_len)[None, :] >= sequence_lengths[:, None]

    if mask_index is None:
        mask_index = num_classes - 1

    # This is a workaround for the fact that mlx.core.argsort does not support
    # the order parameter which is used to break ties when scores are equal.
    # For compatibility with the tensorflow implementation, we flip the inputs
    # and the mask_index, and then flip the classes back to the correct indices
    inputs = flip(inputs, axis=2)
    mask_index = num_classes - mask_index - 1

    _pad = -1

    init_paths = mx.full(
        (batch_size, 2 * beam_width, max_seq_len), _pad, dtype=mx.int32
    )

    num_init_paths = builtins.min(num_classes, beam_width)
    max_classes = mx.argsort(inputs[:, 0], axis=1)[:, -num_init_paths:].astype(
        mx.int32
    )
    init_classes = mx.where(max_classes == mask_index, _pad, max_classes)
    init_paths[:, :num_init_paths, 0] = init_classes

    init_scores = mx.full(
        (batch_size, 2 * beam_width), -mx.inf, dtype=inputs.dtype
    )
    init_scores[:, :num_init_paths] = mx.take_along_axis(
        inputs[:, 0], max_classes, axis=1
    )
    init_masked = init_paths[:, :, 0] == _pad

    def _extend_paths(paths, scores, masked, x):
        paths = mx.repeat(paths, num_classes, axis=0)
        scores = mx.repeat(scores, num_classes)
        masked = mx.repeat(masked, num_classes)

        path_tail_index = mx.argmax(paths == _pad, axis=1).astype(mx.int32)
        paths_arange = mx.arange(2 * beam_width * num_classes)
        path_tails = paths[paths_arange, path_tail_index - 1]
        path_tails = mx.where(path_tail_index == 0, _pad, path_tails)

        classes = mx.arange(num_classes)
        classes[mask_index] = _pad
        classes = mx.tile(classes, 2 * beam_width)

        prev_masked = masked
        masked = classes == _pad

        masked_repeat = ~prev_masked & (path_tails == classes)
        classes = mx.where(masked_repeat, _pad, classes)
        paths = (
            paths.at[paths_arange, path_tail_index]
            .multiply(0)
            .at[paths_arange, path_tail_index]
            .add(classes)
        )

        x = mx.tile(x, 2 * beam_width)
        scores = scores + x

        return paths, scores, masked

    def _merge_scores(unique_inverse, scores):
        scores_max = mx.max(scores)
        scores_exp = mx.exp(scores - scores_max)
        scores = mx.zeros_like(scores).at[unique_inverse].add(scores_exp)
        scores = mx.log(scores) + scores_max
        return scores

    def _prune_paths(paths, scores, masked):
        paths, unique_inverse = _unique_2d(
            paths,
            return_inverse=True,
            size=2 * num_classes * beam_width,
            fill_value=_pad,
        )
        if len(unique_inverse.shape) >= 2:
            unique_inverse = mx.squeeze(unique_inverse, axis=1)

        emit_scores = mx.where(masked, -mx.inf, scores)
        mask_scores = mx.where(masked, scores, -mx.inf)

        emit_scores = _merge_scores(unique_inverse, emit_scores)
        mask_scores = _merge_scores(unique_inverse, mask_scores)

        total_scores = mx.logaddexp(emit_scores, mask_scores)
        top_indices = mx.argsort(total_scores)[-beam_width:]

        paths = paths[top_indices]
        emit_scores = emit_scores[top_indices]
        mask_scores = mask_scores[top_indices]

        paths = mx.tile(paths, (2, 1))
        scores = mx.concatenate([emit_scores, mask_scores])
        masked = mx.concatenate(
            [
                mx.zeros(beam_width, dtype=mx.bool_),
                mx.ones(beam_width, dtype=mx.bool_),
            ]
        )

        return paths, scores, masked

    def _decode_step(paths, scores, masked, x):
        paths, scores, masked = _extend_paths(paths, scores, masked, x)
        paths, scores, masked = _prune_paths(paths, scores, masked)
        return paths, scores, masked

    def _step(prev, x):
        paths, scores, masked = prev
        x, seqlen_mask = x

        new_paths, new_scores, new_masked = _decode_step(
            paths, scores, masked, x
        )

        # Keep old values where seqlen_mask is True
        mask_expanded = (
            seqlen_mask[..., None]
            if seqlen_mask.ndim < paths.ndim
            else seqlen_mask
        )
        paths = mx.where(mask_expanded, paths, new_paths)
        scores = mx.where(mask_expanded, scores, new_scores)
        masked = mx.where(mask_expanded, masked, new_masked)
        return (paths, scores, masked), None

    def _decode_batch(
        init_paths, init_scores, init_masked, inputs, seqlen_mask
    ):
        paths, scores, masked = (init_paths, init_scores, init_masked)
        for i in range(len(inputs) - 1):
            (paths, scores, masked), _ = _step(
                (paths, scores, masked), (inputs[i + 1], seqlen_mask[i + 1])
            )

        paths, unique_inverse = _unique_2d(
            paths,
            return_inverse=True,
            size=2 * num_classes * beam_width,
            fill_value=_pad,
        )
        if len(unique_inverse.shape) >= 2:
            unique_inverse = mx.squeeze(unique_inverse, axis=1)
        scores = _merge_scores(unique_inverse, scores)

        top_indices = mx.argsort(scores)[-top_paths:][::-1]
        paths = paths[top_indices]
        scores = scores[top_indices]

        return paths, scores

    def _decode_batch_loop(
        init_paths, init_scores, init_masked, inputs, seqlen_mask
    ):
        batch_size = init_paths.shape[0]
        all_paths = []
        all_scores = []

        for b in range(batch_size):
            paths, scores = _decode_batch(
                init_paths[b],
                init_scores[b],
                init_masked[b],
                inputs[b],
                seqlen_mask[b],
            )
            all_paths.append(paths)
            all_scores.append(scores)

        return mx.stack(all_paths), mx.stack(all_scores)

    paths, scores = _decode_batch_loop(
        init_paths, init_scores, init_masked, inputs, seqlen_mask
    )

    # convert classes back to the correct indices
    paths = mx.where(paths == _pad, _pad, num_classes - paths - 1)
    paths = mx.transpose(paths, [1, 0, 2])
    return paths, scores


def ctc_decode(
    inputs,
    sequence_lengths,
    strategy="greedy",
    beam_width=100,
    top_paths=1,
    merge_repeated=True,
    mask_index=0,
):
    inputs = convert_to_tensor(inputs)
    dtype = result_type(inputs.dtype, "float32")
    inputs = cast(inputs, dtype)

    if strategy == "greedy":
        return _ctc_greedy_decode(
            inputs,
            sequence_lengths,
            merge_repeated=merge_repeated,
            mask_index=mask_index,
        )
    elif strategy == "beam_search":
        return _ctc_beam_search_decode(
            inputs,
            sequence_lengths,
            beam_width=beam_width,
            top_paths=top_paths,
            mask_index=mask_index,
        )
    else:
        raise ValueError(
            f"Invalid strategy {strategy}. Supported values are "
            "'greedy' and 'beam_search'."
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
    mse = mx.mean(mx.square(x1 - x2))
    psnr = 20 * mx.log10(max_val) - 10 * mx.log10(mse)
    return psnr
