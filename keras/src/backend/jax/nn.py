import jax
import jax.experimental.sparse as jax_sparse
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax import nn as jnn

from keras.src.backend import standardize_data_format
from keras.src.backend import standardize_dtype
from keras.src.backend.common.backend_utils import (
    compute_conv_transpose_padding_args_for_jax,
)
from keras.src.backend.config import epsilon
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


def tanh(x):
    x = convert_to_tensor(x)
    return jnn.tanh(x)


def softplus(x):
    x = convert_to_tensor(x)
    return jnn.softplus(x)


def softsign(x):
    x = convert_to_tensor(x)
    return jnn.soft_sign(x)


def silu(x):
    x = convert_to_tensor(x)
    return jnn.silu(x)


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


def softmax(x, axis=-1):
    x = convert_to_tensor(x)
    return jnn.softmax(x, axis=axis)


def log_softmax(x, axis=-1):
    x = convert_to_tensor(x)
    return jnn.log_softmax(x, axis=axis)


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
    data_format = standardize_data_format(data_format)
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
    strides,
    padding,
    data_format=None,
):
    data_format = standardize_data_format(data_format)
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
        return pooled / np.prod(pool_size)
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
    data_format = standardize_data_format(data_format)
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
    return jax.lax.conv_general_dilated(
        convert_to_tensor(inputs),
        convert_to_tensor(kernel),
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
    data_format = standardize_data_format(data_format)
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
    data_format = standardize_data_format(data_format)
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


def one_hot(x, num_classes, axis=-1, dtype="float32", sparse=False):
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


def multi_hot(x, num_classes, axis=-1, dtype="float32", sparse=False):
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
        output = jnp.clip(output, epsilon(), 1.0 - epsilon())
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
        output = jnp.clip(output, epsilon(), 1.0 - epsilon())
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

    output = jnp.clip(output, epsilon(), 1.0 - epsilon())
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
    ori_dtype = standardize_dtype(x.dtype)
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


def ctc_loss(
    target,
    output,
    target_length,
    output_length,
    mask_index=0,
):
    batch_size, _, _ = output.shape
    batch_size, max_target_length = target.shape

    output = output.transpose((1, 0, 2))
    target = target.transpose((1, 0)).astype("int32")

    logits = jnn.log_softmax(output)
    mgrid_t, mgrid_b = jnp.meshgrid(
        jnp.arange(max_target_length), jnp.arange(batch_size)
    )
    logprobs_emit = logits[mgrid_t, mgrid_b, target[:, :, None]]
    logprobs_mask = logits[:, :, mask_index]

    logit_paddings = jnp.array(
        jnp.arange(max_target_length) < output_length[:, None],
        dtype=jnp.float32,
    )

    repeat = jnp.array(target[1:] == target[:-1])
    repeat = jnp.pad(repeat, ((0, 1), (0, 0))).transpose((1, 0))

    _logepsilon = -100000.0

    def _iterate(prev, x):
        prev_mask, prev_emit = prev
        logprob_mask, logprob_emit, pad = x

        prev_mask_orig = prev_mask
        prev_mask = prev_mask.at[:, 1:].set(
            jnp.logaddexp(prev_mask[:, 1:], prev_emit + _logepsilon * repeat),
        )
        emit = jnp.logaddexp(
            prev_mask[:, :-1] + logprob_emit, prev_emit + logprob_emit
        )

        mask = prev_mask + logprob_mask[:, None]
        mask = mask.at[:, 1:].set(
            jnp.logaddexp(
                mask[:, 1:],
                prev_emit + logprob_mask[:, None] + _logepsilon * (1 - repeat),
            )
        )

        pad = pad[:, None]
        emit = emit * pad + prev_emit * (1 - pad)
        mask = mask * pad + prev_mask_orig * (1 - pad)

        return (mask, emit), (mask, emit)

    mask_init = jnp.full((batch_size, max_target_length + 1), _logepsilon)
    mask_init = mask_init.at[:, 0].set(0.0)
    emit_init = jnp.full((batch_size, max_target_length), _logepsilon)

    _, (alphas_mask, alphas_emit) = lax.scan(
        _iterate,
        (mask_init, emit_init),
        (logprobs_mask, logprobs_emit, logit_paddings.transpose()),
    )

    last_alpha_mask = (
        alphas_mask[-1]
        .at[:, 1:]
        .set(jnp.logaddexp(alphas_mask[-1, :, 1:], alphas_emit[-1]))
    )

    return -last_alpha_mask[jnp.arange(batch_size), target_length]


def ctc_greedy_decode(
    inputs,
    sequence_length,
    merge_repeated=True,
    mask_index=None,
):
    inputs = jnp.array(inputs)
    sequence_length = jnp.array(sequence_length, dtype=jnp.int32)

    if mask_index is None:
        mask_index = inputs.shape[-1] - 1

    indices = jnp.argmax(inputs, axis=-1)
    scores = jnp.max(inputs, axis=-1)

    seqlen_mask = jnp.arange(inputs.shape[1])[None, :]
    seqlen_mask = seqlen_mask >= sequence_length[:, None]

    if merge_repeated:
        repeat = indices[:, 1:] == indices[:, :-1]
        repeat = jnp.pad(repeat, ((0, 0), (1, 0)))

        indices = jnp.where(repeat, mask_index, indices)
    else:
        repeat = jnp.zeros_like(indices, dtype=bool)

    indices = jnp.where(seqlen_mask, mask_index, indices)
    indices = [batch[batch != mask_index] for batch in indices]
    max_len = max(len(batch) for batch in indices)
    indices = jnp.array(
        [jnp.pad(batch, (0, max_len - len(batch))) for batch in indices]
    )

    scores = jnp.where(seqlen_mask, 0.0, scores)
    scores = -jnp.sum(scores, axis=1)[:, None]

    return [indices], scores


def ctc_beam_search_decode(
    inputs,
    sequence_length,
    beam_width=100,
    top_paths=1,
    mask_index=None,
):
    inputs = jnp.array(inputs)
    sequence_length = jnp.array(sequence_length)

    batch_size, max_seq_len, num_classes = inputs.shape
    inputs = jnn.log_softmax(inputs)
    seqlen_mask = jnp.arange(max_seq_len)[None, :] >= sequence_length[:, None]

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

    num_init_paths = jnp.min(jnp.array([num_classes, beam_width]))
    max_classes = jnp.argsort(inputs[:, 0], axis=1)[:, -num_init_paths:]
    init_classes = jnp.where(max_classes == mask_index, _pad, max_classes)
    init_paths = init_paths.at[:, :num_init_paths, 0].set(init_classes)

    init_scores = (
        jnp.full((batch_size, 2 * beam_width), -jnp.inf)
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

    lengths = jnp.argmax(paths == _pad, axis=2)
    lengths = jnp.max(lengths, axis=0)
    paths = jnp.where(paths == _pad, 0, paths)

    paths = paths.transpose((1, 0, 2))
    paths = [path[:, :length] for path, length in zip(paths, lengths)]

    return paths, scores


def ctc_decode(
    inputs,
    sequence_length,
    strategy,
    beam_width=100,
    top_paths=1,
    merge_repeated=True,
    mask_index=None,
):
    if strategy == "greedy":
        return ctc_greedy_decode(
            inputs,
            sequence_length,
            merge_repeated=merge_repeated,
            mask_index=mask_index,
        )
    else:
        return ctc_beam_search_decode(
            inputs,
            sequence_length,
            beam_width=beam_width,
            top_paths=top_paths,
            mask_index=mask_index,
        )
