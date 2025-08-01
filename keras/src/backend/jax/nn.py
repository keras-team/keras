import builtins
import inspect
import math

import jax
import jax.experimental.sparse as jax_sparse
import jax.numpy as jnp
from jax import lax
from jax import nn as jnn
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask,
)

from keras.src import backend
from keras.src.backend.common.backend_utils import (
    compute_conv_transpose_padding_args_for_jax,
)
from keras.src.backend.jax.core import cast
from keras.src.backend.jax.core import convert_to_tensor


def relu(x):
    x = convert_to_tensor(x)
    return jnn.relu(x)


def relu6(x):
    x = convert_to_tensor(x)
    return jnn.relu6(x)


def sigmoid(x):
    x = convert_to_tensor(x)
    return jnn.sigmoid(x)


def sparse_sigmoid(x):
    x = convert_to_tensor(x)
    return jnn.sparse_sigmoid(x)


def tanh(x):
    x = convert_to_tensor(x)
    return jnn.tanh(x)


def tanh_shrink(x):
    x = convert_to_tensor(x)
    return x - jnp.tanh(x)


def softplus(x):
    x = convert_to_tensor(x)
    return jnn.softplus(x)


def softsign(x):
    x = convert_to_tensor(x)
    return jnn.soft_sign(x)


def soft_shrink(x, threshold=0.5):
    x = convert_to_tensor(x)
    return jnp.where(
        x > threshold,
        x - threshold,
        jnp.where(x < -threshold, x + threshold, 0.0),
    )


def sparse_plus(x):
    x = convert_to_tensor(x)
    return jnn.sparse_plus(x)


def silu(x):
    x = convert_to_tensor(x)
    return jnn.silu(x)


def squareplus(x, b=4):
    x = convert_to_tensor(x)
    return jnn.squareplus(x, b=b)


def log_sigmoid(x):
    x = convert_to_tensor(x)
    return jnn.log_sigmoid(x)


def leaky_relu(x, negative_slope=0.2):
    x = convert_to_tensor(x)
    return jnn.leaky_relu(x, negative_slope=negative_slope)


def hard_sigmoid(x):
    x = convert_to_tensor(x)
    return jnn.hard_sigmoid(x)


def hard_silu(x):
    x = convert_to_tensor(x)
    return jnn.hard_silu(x)


def elu(x, alpha=1.0):
    x = convert_to_tensor(x)
    return jnn.elu(x, alpha=alpha)


def selu(x):
    x = convert_to_tensor(x)
    return jnn.selu(x)


def gelu(x, approximate=True):
    x = convert_to_tensor(x)
    return jnn.gelu(x, approximate)


def celu(x, alpha=1.0):
    x = convert_to_tensor(x)
    return jnn.celu(x, alpha=alpha)


def glu(x, axis=-1):
    x = convert_to_tensor(x)
    return jnn.glu(x, axis=axis)


def hard_tanh(x):
    x = convert_to_tensor(x)
    return jnn.hard_tanh(x)


def hard_shrink(x, threshold=0.5):
    x = convert_to_tensor(x)
    return jnp.where(jnp.abs(x) > threshold, x, 0.0)


def threshold(x, threshold, default_value):
    x = convert_to_tensor(x)
    return jnp.where(x > threshold, x, default_value)


def softmax(x, axis=-1):
    x = convert_to_tensor(x)
    return jnn.softmax(x, axis=axis)


def log_softmax(x, axis=-1):
    x = convert_to_tensor(x)
    return jnn.log_softmax(x, axis=axis)


def sparsemax(x, axis=-1):
    # Sort logits along the specified axis in descending order
    logits = convert_to_tensor(x)
    logits_sorted = -1.0 * jnp.sort(logits * -1.0, axis=axis)
    logits_cumsum = jnp.cumsum(logits_sorted, axis=axis)  # find cumulative sum
    r = jnp.arange(1, logits.shape[axis] + 1)  # Determine the sparsity
    r_shape = [1] * logits.ndim
    r_shape[axis] = -1  # Broadcast to match the target axis
    r = r.reshape(r_shape)
    support = logits_sorted - (logits_cumsum - 1) / r > 0
    # Find the threshold
    k = jnp.sum(support, axis=axis, keepdims=True)
    logits_cumsum_safe = jnp.where(support, logits_cumsum, 0.0)
    tau = (jnp.sum(logits_cumsum_safe, axis=axis, keepdims=True) - 1) / k
    output = jnp.maximum(logits - tau, 0.0)
    return output


def _convert_to_spatial_operand(
    x,
    num_spatial_dims,
    data_format="channels_last",
    include_batch_and_channels=True,
):
    # Helper function that converts an operand to a spatial operand.
    x = (x,) * num_spatial_dims if isinstance(x, int) else x
    if not include_batch_and_channels:
        return x
    if data_format == "channels_last":
        x = (1,) + x + (1,)
    else:
        x = (1,) + (1,) + x
    return x


def _pool(
    inputs,
    initial_value,
    reduce_fn,
    pool_size,
    strides=None,
    padding="valid",
):
    """Helper function to define pooling functions.

    Args:
        inputs: input data of shape `N+2`.
        initial_value: the initial value for the reduction.
        reduce_fn: a reduce function of the form `(T, T) -> T`.
        pool_size: a sequence of `N` integers, representing the window size to
            reduce over.
        strides: a sequence of `N` integers, representing the inter-window
            strides (default: `(1, ..., 1)`).
        padding: either the string `same` or `valid`.

    Returns:
        The output of the reduction for each window slice.
    """
    if padding not in ("same", "valid"):
        raise ValueError(
            f"Invalid padding '{padding}', must be 'same' or 'valid'."
        )
    padding = padding.upper()
    return lax.reduce_window(
        inputs,
        initial_value,
        reduce_fn,
        pool_size,
        strides,
        padding,
    )


def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2
    pool_size = _convert_to_spatial_operand(
        pool_size, num_spatial_dims, data_format
    )
    strides = pool_size if strides is None else strides
    strides = _convert_to_spatial_operand(
        strides, num_spatial_dims, data_format
    )
    return _pool(inputs, -jnp.inf, lax.max, pool_size, strides, padding)


def average_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2
    pool_size = _convert_to_spatial_operand(
        pool_size, num_spatial_dims, data_format
    )
    strides = pool_size if strides is None else strides
    strides = _convert_to_spatial_operand(
        strides, num_spatial_dims, data_format
    )

    pooled = _pool(inputs, 0.0, lax.add, pool_size, strides, padding)
    if padding == "valid":
        # Avoid the extra reduce_window.
        return pooled / math.prod(pool_size)
    else:
        # Count the number of valid entries at each input point, then use that
        # for computing average. Assumes that any two arrays of same shape will
        # be padded the same. Avoid broadcasting on axis where pooling is
        # skipped.
        shape = [
            (a if b != 1 else 1) for (a, b) in zip(inputs.shape, pool_size)
        ]
        window_counts = _pool(
            jnp.ones(shape, inputs.dtype),
            0.0,
            lax.add,
            pool_size,
            strides,
            padding,
        )
        return pooled / window_counts


def _convert_to_lax_conv_dimension_numbers(
    num_spatial_dims,
    data_format="channels_last",
    transpose=False,
):
    """Create a `lax.ConvDimensionNumbers` for the given inputs."""
    num_dims = num_spatial_dims + 2

    if data_format == "channels_last":
        spatial_dims = tuple(range(1, num_dims - 1))
        inputs_dn = (0, num_dims - 1) + spatial_dims
    else:
        spatial_dims = tuple(range(2, num_dims))
        inputs_dn = (0, 1) + spatial_dims

    if transpose:
        kernel_dn = (num_dims - 2, num_dims - 1) + tuple(range(num_dims - 2))
    else:
        kernel_dn = (num_dims - 1, num_dims - 2) + tuple(range(num_dims - 2))

    return lax.ConvDimensionNumbers(
        lhs_spec=inputs_dn, rhs_spec=kernel_dn, out_spec=inputs_dn
    )


def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2
    dimension_numbers = _convert_to_lax_conv_dimension_numbers(
        num_spatial_dims,
        data_format,
        transpose=False,
    )
    strides = _convert_to_spatial_operand(
        strides,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )
    dilation_rate = _convert_to_spatial_operand(
        dilation_rate,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )
    if data_format == "channels_last":
        channels = inputs.shape[-1]
    else:
        channels = inputs.shape[1]
    kernel_in_channels = kernel.shape[-2]
    if channels % kernel_in_channels > 0:
        raise ValueError(
            "The number of input channels must be evenly divisible by "
            f"kernel's in_channels. Received input channels {channels} and "
            f"kernel in_channels {kernel_in_channels}. "
        )
    feature_group_count = channels // kernel_in_channels
    kernel = convert_to_tensor(kernel)
    inputs = convert_to_tensor(inputs, dtype=kernel.dtype)
    return jax.lax.conv_general_dilated(
        inputs,
        kernel,
        strides,
        padding,
        rhs_dilation=dilation_rate,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
    )


def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2
    dimension_numbers = _convert_to_lax_conv_dimension_numbers(
        num_spatial_dims,
        data_format,
        transpose=False,
    )
    strides = _convert_to_spatial_operand(
        strides,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )
    dilation_rate = _convert_to_spatial_operand(
        dilation_rate,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )
    feature_group_count = (
        inputs.shape[-1] if data_format == "channels_last" else inputs.shape[1]
    )
    kernel = jnp.reshape(
        kernel,
        kernel.shape[:-2] + (1, feature_group_count * kernel.shape[-1]),
    )
    return jax.lax.conv_general_dilated(
        inputs,
        kernel,
        strides,
        padding,
        rhs_dilation=dilation_rate,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
    )


def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    data_format = backend.standardize_data_format(data_format)
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
    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2
    padding_values = compute_conv_transpose_padding_args_for_jax(
        input_shape=inputs.shape,
        kernel_shape=kernel.shape,
        strides=strides,
        padding=padding,
        output_padding=output_padding,
        dilation_rate=dilation_rate,
    )
    dimension_numbers = _convert_to_lax_conv_dimension_numbers(
        num_spatial_dims,
        data_format,
        transpose=False,
    )
    strides = _convert_to_spatial_operand(
        strides,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )
    dilation_rate = _convert_to_spatial_operand(
        dilation_rate,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )

    return jax.lax.conv_transpose(
        inputs,
        kernel,
        strides,
        padding=padding_values,
        rhs_dilation=dilation_rate,
        dimension_numbers=dimension_numbers,
        transpose_kernel=True,
    )


def one_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    x = convert_to_tensor(x)
    if sparse:
        if axis < 0:
            axis = axis + len(x.shape) + 1
        if dtype is None:
            dtype = "float32"
        # We deal with negative inputs by having zeros in the output although
        # it's useless. It makes shapes static.
        values = jnp.greater_equal(jnp.ravel(x), 0).astype(dtype)
        values_count = values.shape[0]
        indices = [jnp.arange(dim) for dim in x.shape]
        indices = jnp.meshgrid(*indices, indexing="ij")
        indices.insert(axis, jnp.maximum(x, 0))  # Deal with negative indices
        indices = [a.reshape(values_count, 1).astype("int32") for a in indices]
        indices = jnp.concatenate(indices, axis=1)
        shape = list(x.shape)
        shape.insert(axis, num_classes)
        shape = tuple(shape)
        return jax_sparse.BCOO(
            (values, indices),
            shape=shape,
            indices_sorted=True,
            unique_indices=True,
        )
    return jnn.one_hot(x, num_classes, axis=axis, dtype=dtype)


def multi_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    x = convert_to_tensor(x)
    reduction_axis = 1 if len(x.shape) > 1 else 0
    if sparse:
        result = one_hot(
            x, num_classes, axis=axis, dtype="int32", sparse=sparse
        )
        # JAX's BCOO does not support max reduction, use sum and compare with 0.
        result = jax_sparse.bcoo_reduce_sum(result, axes=(reduction_axis,))
        result = jax_sparse.bcoo_sum_duplicates(result)
        values = jnp.greater_equal(result.data, 0).astype(dtype)
        return jax_sparse.BCOO(
            (values, result.indices),
            shape=result.shape,
            indices_sorted=True,
            unique_indices=True,
        )
    return jnp.max(
        one_hot(cast(x, "int32"), num_classes, axis=axis, dtype=dtype),
        axis=reduction_axis,
    )


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = jnp.array(target)
    output = jnp.array(output)

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
        log_prob = jax.nn.log_softmax(output, axis=axis)
    else:
        output = output / jnp.sum(output, axis, keepdims=True)
        output = jnp.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
        log_prob = jnp.log(output)
    return -jnp.sum(target * log_prob, axis=axis)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = jnp.array(target, dtype="int32")
    output = jnp.array(output)
    if len(target.shape) == len(output.shape) and target.shape[-1] == 1:
        target = jnp.squeeze(target, axis=-1)

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
        log_prob = jax.nn.log_softmax(output, axis=axis)
    else:
        output = output / jnp.sum(output, axis, keepdims=True)
        output = jnp.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
        log_prob = jnp.log(output)
    target = jnn.one_hot(target, output.shape[axis], axis=axis)
    return -jnp.sum(target * log_prob, axis=axis)


def binary_crossentropy(target, output, from_logits=False):
    target = jnp.array(target)
    output = jnp.array(output)

    if target.shape != output.shape:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        log_logits = jax.nn.log_sigmoid(output)
        log_neg_logits = jax.nn.log_sigmoid(-output)
        return -1.0 * target * log_logits - (1.0 - target) * log_neg_logits

    output = jnp.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
    bce = target * jnp.log(output)
    bce += (1.0 - target) * jnp.log(1.0 - output)
    return -bce


def moments(x, axes, keepdims=False, synchronized=False):
    if synchronized:
        raise NotImplementedError(
            "Argument synchronized=True is not supported with JAX."
        )
    # The dynamic range of float16 is too limited for statistics. As a
    # workaround, we simply perform the operations on float32 and convert back
    # to float16
    need_cast = False
    ori_dtype = backend.standardize_dtype(x.dtype)
    if ori_dtype in ("float16", "bfloat16"):
        need_cast = True
        x = cast(x, "float32")

    mean = jnp.mean(x, axes, keepdims=True)
    variance = jnp.var(x, axis=axes, keepdims=True)

    if not keepdims:
        mean = jnp.squeeze(mean, axes)
        variance = jnp.squeeze(variance, axes)
    if need_cast:
        # avoid overflow and underflow when casting from float16 to float32
        mean = jnp.clip(
            mean, jnp.finfo(jnp.float16).min, jnp.finfo(jnp.float16).max
        )
        variance = jnp.clip(
            variance, jnp.finfo(jnp.float16).min, jnp.finfo(jnp.float16).max
        )
        mean = cast(mean, ori_dtype)
        variance = cast(variance, ori_dtype)
    return mean, variance


def batch_normalization(
    x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3
):
    shape = [1] * len(x.shape)
    shape[axis] = mean.shape[0]
    mean = jnp.reshape(mean, shape)
    variance = jnp.reshape(variance, shape)

    inv = jax.lax.rsqrt(variance + epsilon)
    if scale is not None:
        scale = jnp.reshape(scale, shape)
        inv = inv * scale

    res = -mean * inv
    if offset is not None:
        offset = jnp.reshape(offset, shape)
        res = res + offset

    return jnp.add(x * inv, res)


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

    # Ensure that the dtype promotion behavior matches that of `tf.nn.ctc_loss`
    dtype = backend.result_type(output.dtype, "float32")
    output = cast(output, dtype)

    def _lengths_to_paddings(lengths, max_length):
        indices = jnp.arange(max_length).reshape(
            (1,) * lengths.ndim + (max_length,)
        )
        lengths = jnp.expand_dims(lengths, axis=-1)
        elem_valid = indices < lengths
        return jnp.logical_not(elem_valid)

    target_paddings = _lengths_to_paddings(target_length, max_label_length)
    output_paddings = _lengths_to_paddings(output_length, max_input_length)
    target_paddings = target_paddings.astype(output.dtype)
    output_paddings = output_paddings.astype(output.dtype)

    logprobs = jnn.log_softmax(output)
    label_lengths = max_label_length - jnp.sum(target_paddings, axis=1).astype(
        jnp.int32
    )

    # repeat[b, n] == 1.0 when label[b, n] == label[b, n+1].
    repeat = (target[:, :-1] == target[:, 1:]).astype(jnp.float32)
    repeat = jnp.pad(repeat, ((0, 0), (0, 1)))

    logprobs_phi = logprobs[:, :, mask_index : mask_index + 1]  # [B, T, 1]
    logprobs_phi = jnp.transpose(logprobs_phi, (1, 0, 2))  # [T, B, 1]

    _one_hot = jax.nn.one_hot(target, num_classes=num_classes)  # [B, N, K]
    logprobs_emit = jnp.einsum("btk,bnk->btn", logprobs, _one_hot)
    logprobs_emit = jnp.transpose(logprobs_emit, (1, 0, 2))  # [T, B, N]

    # [B, N]
    logalpha_phi_init = (
        jnp.ones((batch_size, max_label_length + 1), dtype=output.dtype)
        * log_epsilon
    )
    logalpha_phi_init = logalpha_phi_init.at[:, 0].set(0.0)
    logalpha_emit_init = (
        jnp.ones((batch_size, max_label_length), dtype=output.dtype)
        * log_epsilon
    )

    def update_phi_score(phi, added_score):
        # Update `phi[:, 1:]`` with adding `added_score` in log space.
        return jnp.concatenate(
            [phi[:, :1], jnp.logaddexp(phi[:, 1:], added_score)], axis=-1
        )

    def loop_body(prev, x):
        prev_phi, prev_emit = prev
        # emit-to-phi epsilon transition, except if the next label is repetition
        prev_phi_orig = prev_phi
        prev_phi = update_phi_score(prev_phi, prev_emit + log_epsilon * repeat)

        logprob_emit, logprob_phi, pad = x

        # phi-to-emit transition
        next_emit = jnp.logaddexp(
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
    _, (logalpha_phi, logalpha_emit) = jax.lax.scan(
        loop_body, (logalpha_phi_init, logalpha_emit_init), xs
    )

    # last row needs to be updated with the last epsilon transition
    logalpha_phi_last = update_phi_score(logalpha_phi[-1], logalpha_emit[-1])
    logalpha_phi = logalpha_phi.at[-1].set(logalpha_phi_last)

    # extract per_seq_loss
    # [B, N+1]
    _one_hot = jax.nn.one_hot(label_lengths, num_classes=max_label_length + 1)
    per_seq_loss = -jnp.einsum("bn,bn->b", logalpha_phi_last, _one_hot)
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

    indices = jnp.argmax(inputs, axis=-1)
    scores = jnp.max(inputs, axis=-1)

    seqlen_mask = jnp.arange(max_length)[None, :]
    seqlen_mask = seqlen_mask >= sequence_lengths[:, None]

    indices = jnp.where(seqlen_mask, mask_index, indices)
    scores = jnp.where(seqlen_mask, 0.0, scores)

    if merge_repeated:
        repeat_mask = indices[:, 1:] == indices[:, :-1]
        repeat_mask = jnp.pad(repeat_mask, ((0, 0), (1, 0)))
        indices = jnp.where(repeat_mask, mask_index, indices)

    # We set to -1 for blank labels
    invalid_mask = indices == mask_index
    indices = jnp.where(invalid_mask, -1, indices)

    # We rearrange the indices by moving `mask_index` to the end of the array
    order = jnp.expand_dims(jnp.arange(max_length), axis=0)  # [1, N]
    order = jnp.tile(order, (batch_size, 1))  # [B, N]
    order = jnp.where(invalid_mask, max_length, order)
    order = jnp.argsort(order, axis=-1)
    indices = jnp.take_along_axis(indices, order, axis=-1)

    scores = -jnp.sum(scores, axis=1)[:, None]
    indices = jnp.expand_dims(indices, axis=0)
    return indices, scores


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
    inputs = jnn.log_softmax(inputs)
    seqlen_mask = jnp.arange(max_seq_len)[None, :] >= sequence_lengths[:, None]

    if mask_index is None:
        mask_index = num_classes - 1

    # This is a workaround for the fact that jnp.argsort does not support
    # the order parameter which is used to break ties when scores are equal.
    # For compatibility with the tensorflow implementation, we flip the inputs
    # and the mask_index, and then flip the classes back to the correct indices
    inputs = jnp.flip(inputs, axis=2)
    mask_index = num_classes - mask_index - 1

    _pad = -1

    init_paths = jnp.full(
        (batch_size, 2 * beam_width, max_seq_len), _pad, dtype=jnp.int32
    )

    num_init_paths = builtins.min(num_classes, beam_width)
    max_classes = jnp.argsort(inputs[:, 0], axis=1)[:, -num_init_paths:]
    init_classes = jnp.where(max_classes == mask_index, _pad, max_classes)
    init_paths = init_paths.at[:, :num_init_paths, 0].set(init_classes)

    init_scores = (
        jnp.full((batch_size, 2 * beam_width), -jnp.inf, dtype=inputs.dtype)
        .at[:, :num_init_paths]
        .set(jnp.take_along_axis(inputs[:, 0], max_classes, axis=1))
    )
    init_masked = init_paths[:, :, 0] == _pad

    def _extend_paths(paths, scores, masked, x):
        paths = jnp.repeat(paths, num_classes, axis=0)
        scores = jnp.repeat(scores, num_classes)
        masked = jnp.repeat(masked, num_classes)

        path_tail_index = jnp.argmax(paths == _pad, axis=1)
        paths_arange = jnp.arange(2 * beam_width * num_classes)
        path_tails = paths[paths_arange, path_tail_index - 1]
        path_tails = jnp.where(path_tail_index == 0, _pad, path_tails)

        classes = jnp.arange(num_classes).at[mask_index].set(_pad)
        classes = jnp.tile(classes, 2 * beam_width)

        prev_masked = masked
        masked = classes == _pad

        masked_repeat = ~prev_masked & (path_tails == classes)
        classes = jnp.where(masked_repeat, _pad, classes)
        paths = paths.at[paths_arange, path_tail_index].set(classes)

        x = jnp.tile(x, 2 * beam_width)
        scores = scores + x

        return paths, scores, masked

    def _merge_scores(unique_inverse, scores):
        scores_max = jnp.max(scores)
        scores_exp = jnp.exp(scores - scores_max)
        scores = jnp.zeros_like(scores).at[unique_inverse].add(scores_exp)
        scores = jnp.log(scores) + scores_max
        return scores

    def _prune_paths(paths, scores, masked):
        paths, unique_inverse = jnp.unique(
            paths,
            return_inverse=True,
            size=2 * num_classes * beam_width,
            axis=0,
            fill_value=_pad,
        )
        if len(unique_inverse.shape) >= 2:
            unique_inverse = jnp.squeeze(unique_inverse, axis=1)

        emit_scores = jnp.where(masked, -jnp.inf, scores)
        mask_scores = jnp.where(masked, scores, -jnp.inf)

        emit_scores = _merge_scores(unique_inverse, emit_scores)
        mask_scores = _merge_scores(unique_inverse, mask_scores)

        total_scores = jnp.logaddexp(emit_scores, mask_scores)
        top_indices = jnp.argsort(total_scores)[-beam_width:]

        paths = paths[top_indices]
        emit_scores = emit_scores[top_indices]
        mask_scores = mask_scores[top_indices]

        paths = jnp.tile(paths, (2, 1))
        scores = jnp.concatenate([emit_scores, mask_scores])
        masked = jnp.concatenate(
            [jnp.zeros(beam_width, bool), jnp.ones(beam_width, bool)]
        )

        return paths, scores, masked

    def _decode_step(paths, scores, masked, x):
        paths, scores, masked = _extend_paths(paths, scores, masked, x)
        paths, scores, masked = _prune_paths(paths, scores, masked)
        return paths, scores, masked

    def _step(prev, x):
        paths, scores, masked = prev
        x, seqlen_mask = x

        paths, scores, masked = lax.cond(
            seqlen_mask,
            lambda paths, scores, masked, x: (paths, scores, masked),
            _decode_step,
            paths,
            scores,
            masked,
            x,
        )

        return (paths, scores, masked), None

    def _decode_batch(
        init_paths, init_scores, init_masked, inputs, seqlen_mask
    ):
        (paths, scores, masked), _ = lax.scan(
            _step,
            (init_paths, init_scores, init_masked),
            (inputs[1:], seqlen_mask[1:]),
        )

        paths, unique_inverse = jnp.unique(
            paths,
            return_inverse=True,
            size=2 * num_classes * beam_width,
            axis=0,
            fill_value=_pad,
        )
        if len(unique_inverse.shape) >= 2:
            unique_inverse = jnp.squeeze(unique_inverse, axis=1)
        scores = _merge_scores(unique_inverse, scores)

        top_indices = jnp.argsort(scores)[-top_paths:][::-1]
        paths = paths[top_indices]
        scores = scores[top_indices]

        return paths, scores

    paths, scores = jax.vmap(_decode_batch)(
        init_paths, init_scores, init_masked, inputs, seqlen_mask
    )

    # convert classes back to the correct indices
    paths = jnp.where(paths == _pad, _pad, num_classes - paths - 1)
    paths = jnp.transpose(paths, [1, 0, 2])
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
    dtype = backend.result_type(inputs.dtype, "float32")
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
    if x1.shape != x2.shape:
        raise ValueError(
            f"Input shapes {x1.shape} and {x2.shape} must "
            "match for PSNR calculation. "
        )

    max_val = convert_to_tensor(max_val, dtype=x2.dtype)
    mse = jnp.mean(jnp.square(x1 - x2))
    psnr = 20 * jnp.log10(max_val) - 10 * jnp.log10(mse)
    return psnr


def _can_use_flash_attention(query, key, value, bias, raise_error=False):
    """Verify the availability of flash attention."""
    try:
        from jax._src.cudnn.fused_attention_stablehlo import _normalize_layout
        from jax._src.cudnn.fused_attention_stablehlo import (
            check_compute_capability,
        )
        from jax._src.cudnn.fused_attention_stablehlo import check_cudnn_version
        from jax._src.cudnn.fused_attention_stablehlo import (
            check_is_flash_attention,
        )
        from jax._src.cudnn.fused_attention_stablehlo import check_layout
        from jax.nn import dot_product_attention as dot_product_attention
    except ImportError:
        if raise_error:
            raise ImportError(
                "Flash attention is not supported in your current JAX version. "
                "Please update it by following the official guide: "
                "https://jax.readthedocs.io/en/latest/installation.html"
            )
        return False

    if jax.devices()[0].platform == "tpu":
        return True
    try:
        # Check if cuDNN is installed and raise RuntimeError if cuDNN is not
        # detected
        cudnn_version = check_cudnn_version()
        # Only support at least Ampere
        if not check_compute_capability("8.0"):
            raise RuntimeError("Require at least Ampere arch to run")
        # Check inputs layout
        check_layout_params = list(
            inspect.signature(check_layout).parameters.keys()
        )
        for known_param in ("query", "key", "value", "bias", "layout"):
            check_layout_params.remove(known_param)
        # Defaults to `None` when not specified.
        kwargs = {key: None for key in check_layout_params}
        check_layout(
            query, key, value, bias, layout=_normalize_layout("BTNH"), **kwargs
        )
        check_is_flash_attention(
            query,
            key,
            _normalize_layout("BTNH"),
            cudnn_version,
            bias is not None,
            is_training=False,
        )
        return True
    except:
        if raise_error:
            raise
        return False


def _apply_masks(logits, mask, is_causal):
    if mask is None and not is_causal:
        return logits

    combined_mask = jnp.ones_like(logits, dtype="bool")
    if mask is not None:
        combined_mask = jnp.logical_and(combined_mask, mask)

    if is_causal:
        T, S = logits.shape[2], logits.shape[3]
        mask = jnp.tril(jnp.ones((T, S), dtype="bool"))
        mask = mask[None, None, :, :]
        combined_mask = jnp.logical_and(combined_mask, mask)

    large_negative_number = jnp.asarray(
        -0.7 * jnp.finfo(logits.dtype).max, dtype=logits.dtype
    )
    padded_logits = jnp.where(combined_mask, logits, large_negative_number)
    return padded_logits


def _dot_product_attention_core(
    query, key, value, bias, mask, is_causal, scale
):
    logits_dtype = jnp.promote_types(query.dtype, jnp.float32)
    logits = jnp.einsum(
        "BTNH,BSNH->BNTS", query, key, preferred_element_type=logits_dtype
    )
    logits *= jnp.array(scale, dtype=logits.dtype)

    if bias is not None:
        logits = (logits + bias).astype(logits.dtype)

    padded_logits = _apply_masks(logits, mask, is_causal)

    # Softmax and it is always carried out in fp32.
    padded_logits = padded_logits.astype(jnp.float32)
    probs = jax.nn.softmax(padded_logits, axis=-1).astype(key.dtype)
    return jnp.einsum("BNTS,BSNH->BTNH", probs, value)


def wrap_flash_attention(
    query,
    key,
    value,
    decoder_segment_ids,
    custom_mask=None,
    attn_logits_soft_cap=None,
    head_shards=1,
    q_seq_shards=1,
):
    """Applies a wrapped flash attention mechanism using the Splash kernel.
    This function prepares the appropriate attention mask (causal or custom),
    constructs a multi-head mask, and applies the Splash multi-head attention
    kernel to the provided query, key, and value tensors. It supports optional
    sharding and soft capping of attention logits.
    Args:
        query: jax.Array. The query tensor of shape
            (batch, num_heads, seq_len, head_dim).
        key: jax.Array. The key tensor of shape
            (batch, num_heads, seq_len, head_dim).
        value: jax.Array. The value tensor of shape
            (batch, num_heads, seq_len, head_dim).
        decoder_segment_ids: Optional. Segment IDs for the decoder, used for
            sharding or masking.
        custom_mask: Optional[jax.Array]. A custom attention mask to apply. If
            None, a causal mask is used.
        attn_logits_soft_cap: Optional[float]. If provided, applies a soft cap
            to the attention logits.
        head_shards: int, default=1. Number of shards for the attention heads.
        q_seq_shards: int, default=1. Number of shards for the query sequence
            dimension.
    Returns:
        jax.Array: The result of applying the Splash multi-head attention
            kernel to the inputs.
    Raises:
        AssertionError: If sharding along the sequence dimension is attempted
            with decoder_segment_ids.
    """
    if decoder_segment_ids is not None:
        assert query.shape[2] == decoder_segment_ids.q.shape[1], (
            "Sharding along sequence dimension not allowed"
            " in TPU kernel attention"
        )

    if custom_mask is not None:
        mask = splash_attention_mask.NumpyMask(array=custom_mask)
    else:
        mask = splash_attention_mask.CausalMask(
            shape=(query.shape[2], query.shape[2])
        )

    # Create multi-head mask
    multi_head_mask = splash_attention_mask.MultiHeadMask(
        masks=(mask,) * query.shape[1]
    )
    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=multi_head_mask,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )

    return jax.vmap(splash_kernel)(
        query, key, value, segment_ids=decoder_segment_ids
    )


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
    """Computes dot-product attention given query, key, and value.

    This is the core computation of attention that is used in transformers.
    For TPU platforms, flash attention optimizations are automatically applied
    when possible, and sharding parameters are inferred from the layout map
    in the current distribution context.

    Args:
        query: Queries with shape `[batch, time, heads,
            depth_k]`.
        key: Keys with shape `[batch, time, heads,
            depth_k]`.
        value: Values with shape `[batch, time, heads,
            depth_v]`.
        bias: Optional bias with shape broadcastable to
            `[batch, heads, dest_time, source_time]`.
        mask: Optional mask with shape broadcastable to
            `[batch, heads, dest_time, source_time]`.
        scale: Float. Optional scale that is applied to the attention
            computation.
        is_causal: Boolean. Specifying whether causal masking is applied.
        flash_attention: Boolean. Whether to use flash attention optimization
            for increased performance. Default to None, which means it will
            be auto-determined based on the platform, input shapes and
            compatibility.
        attn_logits_soft_cap: Float. Optional float to softly cap attention
            logits to avoid numerical stability issues. Applied as:
            `logits = logits / (1.0 + abs(logits) / attn_logits_soft_cap)`.

    Returns:
        JAX Array of shape `[batch, time, heads, depth_v]`.
    """
    query = convert_to_tensor(query)
    key = convert_to_tensor(key)
    value = convert_to_tensor(value)
    if len(query.shape) != 4 or len(key.shape) != 4 or len(value.shape) != 4:
        raise ValueError(
            "`dot_product_attention` only supports 4D inputs. "
            f"Received: query.shape={query.shape}, key.shape={key.shape}, "
            f"value.shape={value.shape}."
        )

    # Check platform
    platform = jax.devices()[0].platform
    is_tpu = platform == "tpu"

    # Determine flash attention compatibility
    if flash_attention is None:
        flash_attention = _can_use_flash_attention(query, key, value, bias)
    elif flash_attention is True:
        # Use `raise_error=True` to provide more details if the inputs failed to
        # use flash attention
        _can_use_flash_attention(query, key, value, bias, raise_error=True)

    # TPU-specific flash attention path
    if is_tpu and flash_attention:
        # Get sharding parameters from distribution context
        try:
            from keras.src.distribution.distribution_lib import ModelParallel
            from keras.src.distribution.distribution_lib import (
                distribution as get_dist,
            )

            # Get current distribution if available
            dist = get_dist()
            if dist and isinstance(dist, ModelParallel):
                mesh = dist.device_mesh
                if "model" in mesh.axis_names:
                    model_dim_index = mesh.axis_names.index("model")
                    # Set head_shards based on the model dimension of the mesh
                    head_shards = mesh.shape[model_dim_index]
                    # Typically keep q_seq_shards=1 for best performance
                    q_seq_shards = 1
        except (ImportError, ValueError, AttributeError):
            # Use default values if detection fails
            head_shards = 1
            q_seq_shards = 1
        # Transpose to ('batch', 'heads', 'length', 'head_dim')
        query_tpu_layout = jnp.transpose(query, axes=(0, 2, 1, 3))
        key_tpu_layout = jnp.transpose(key, axes=(0, 2, 1, 3))
        value_tpu_layout = jnp.transpose(value, axes=(0, 2, 1, 3))

        bs, num_heads, q_len, head_dim = query_tpu_layout.shape

        # Apply scale to query if provided
        if scale is not None:
            # TPU kernel applies 1/sqrt(head_dim) internally, to achieve
            # overall QK^T * scale, scale query by (scale * sqrt(head_dim))
            query_tpu_layout = query_tpu_layout * (scale * math.sqrt(head_dim))

        # Create segment IDs for Splash Attention (for packing/batching)
        segment_ids = jnp.zeros([bs, q_len], dtype=jnp.int32)
        decoder_segment_ids = splash_attention_kernel.SegmentIds(
            q=segment_ids, kv=segment_ids
        )

        # Process mask for Splash Attention
        custom_mask = None
        if mask is not None:
            mask_bool = mask.astype("bool") if mask.dtype != jnp.bool_ else mask

            if mask_bool.ndim == 3 and mask_bool.shape[0] == bs:
                custom_mask = mask_bool[0]
            elif mask_bool.ndim == 4 and mask_bool.shape[0] == bs:
                custom_mask = mask_bool[0, 0]

            if is_causal and custom_mask is not None:
                causal_mask = jnp.tril(
                    jnp.ones((q_len, q_len), dtype=jnp.bool_)
                )
                custom_mask = jnp.logical_and(custom_mask, causal_mask)

        if custom_mask is None and is_causal:
            custom_mask = jnp.tril(jnp.ones((q_len, q_len), dtype=jnp.bool_))

        try:
            output = wrap_flash_attention(
                query_tpu_layout,
                key_tpu_layout,
                value_tpu_layout,
                decoder_segment_ids=decoder_segment_ids,
                custom_mask=custom_mask,
                attn_logits_soft_cap=attn_logits_soft_cap,
                head_shards=head_shards,
                q_seq_shards=q_seq_shards,
            )
            # Transpose output back to Keras layout
            return jnp.transpose(output, axes=(0, 2, 1, 3))
        except Exception:
            flash_attention = False

    # JAX native dot_product_attention for GPU or fallback for TPU
    if hasattr(jax.nn, "dot_product_attention"):
        try:
            return jax.nn.dot_product_attention(
                query,
                key,
                value,
                bias=bias,
                mask=mask,
                scale=scale,
                is_causal=is_causal,
                implementation="cudnn" if flash_attention else "xla",
            )
        except Exception:
            # If flash attention fails, fall back to XLA implementation
            if flash_attention:
                return jax.nn.dot_product_attention(
                    query,
                    key,
                    value,
                    bias=bias,
                    mask=mask,
                    scale=scale,
                    is_causal=is_causal,
                    implementation="xla",
                )
            raise

    if flash_attention:
        raise RuntimeError(
            "Flash attention is not supported in your current JAX version. "
            "Please update it by following the official guide: "
            "https://jax.readthedocs.io/en/latest/installation.html"
        )
    # Ref: jax.nn.dot_product_attention
    # https://github.com/jax-ml/jax/blob/jax-v0.4.33/jax/_src/nn/functions.py#L886
    # Not support `query_seq_lengths` and `key_value_seq_lengths` args

    # Fallback to custom XLA implementation
    # This is the reference implementation from jax.nn.dot_product_attention
    output_shape = query.shape
    _, _, K, H = key.shape
    scale = (1.0 / jnp.sqrt(H)) if scale is None else scale

    # _dot_product_attention_xla
    B, T, N, H = query.shape
    G = N // K
    query = jnp.reshape(query, (B, T, K, G, H))

    def _reshape_to_grouped(t):
        if t is not None:
            tB, tN, tT, tS = t.shape
            if tN == 1:
                t = jnp.broadcast_to(t[:, :, None, :, :], (tB, tN, G, tT, tS))
            else:
                assert tN == N
                t = jnp.reshape(t, (tB, K, G, tT, tS))
        return t

    bias = _reshape_to_grouped(bias)
    mask = _reshape_to_grouped(mask)
    vmapped_fn = jax.vmap(
        _dot_product_attention_core,
        in_axes=(3, None, None, 2, 2, None, None),
        out_axes=3,
    )
    encoded = vmapped_fn(query, key, value, bias, mask, is_causal, scale)
    return jnp.reshape(encoded, output_shape)
