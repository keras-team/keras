import jax
import numpy as np
from jax import lax

from keras.src import backend
from keras.src.backend.common.backend_utils import (
    compute_conv_transpose_padding_args_for_jax,
)
from keras.src.backend.numpy.core import cast
from keras.src.backend.numpy.core import convert_to_tensor
from keras.src.backend.numpy.core import is_tensor
from keras.src.utils.module_utils import scipy


def relu(x):
    x = convert_to_tensor(x)
    return np.maximum(x, np.array(0.0, x.dtype))


def relu6(x):
    x = convert_to_tensor(x)
    # np.clip incorrectly promote bfloat16 to float32, so we replace it with
    # np.minimum and np.maximum here
    return np.minimum(
        np.maximum(x, np.array(0.0, x.dtype)), np.array(6.0, x.dtype)
    )


def sigmoid(x):
    x = convert_to_tensor(x)
    return np.array(1.0, x.dtype) / (np.array(1.0, x.dtype) + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def softplus(x):
    x = convert_to_tensor(x)
    return np.logaddexp(x, np.array(0.0, x.dtype))


def softsign(x):
    x = convert_to_tensor(x)
    return x / (np.array(1.0, x.dtype) + np.abs(x))


def silu(x):
    x = convert_to_tensor(x)
    return x * sigmoid(x)


def log_sigmoid(x):
    x = convert_to_tensor(x)
    return -softplus(-x)


def leaky_relu(x, negative_slope=0.2):
    x = convert_to_tensor(x)
    return np.maximum(x, np.array(negative_slope, x.dtype) * x)


def hard_sigmoid(x):
    # python numbers will be promoted to float64 by np, so it's necessary to
    # first convert the python numbers to np scalars
    x = x / np.array(6.0, x.dtype) + np.array(0.5, x.dtype)
    return np.where(
        x <= 0.0,
        np.array(0.0, x.dtype),
        np.where(x >= 1.0, np.array(1.0, x.dtype), x),
    )


def hard_silu(x):
    return x * hard_sigmoid(x)


def elu(x, alpha=1.0):
    x = convert_to_tensor(x)
    return np.where(
        x >= np.array(0.0, x.dtype), x, np.array(alpha, x.dtype) * np.expm1(x)
    )


def selu(
    x,
    alpha=1.6732632423543772848170429916717,
    scale=1.0507009873554804934193349852946,
):
    x = convert_to_tensor(x)
    return np.array(scale, x.dtype) * elu(x, alpha)


def gelu(x, approximate=True):
    x = convert_to_tensor(x)
    # followed by JAX's implementation
    if approximate:
        sqrt_2_over_pi = np.sqrt(2 / np.pi).astype(x.dtype)
        cdf = np.array(0.5, x.dtype) * (
            np.array(1.0, x.dtype)
            + np.tanh(
                sqrt_2_over_pi
                * (x + np.array(0.044715, x.dtype) * (x**3).astype(x.dtype))
            )
        )
        return x * cdf
    else:
        sqrt_2 = np.sqrt(2).astype(x.dtype)
        return (
            x
            * (scipy.special.erf(x / sqrt_2) + 1).astype(x.dtype)
            / np.array(2, x.dtype)
        )


def softmax(x, axis=None):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x, axis=None):
    max_x = np.max(x, axis=axis, keepdims=True)
    logsumexp = np.log(np.exp(x - max_x).sum(axis=axis, keepdims=True))
    return x - max_x - logsumexp


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
    return np.array(
        lax.reduce_window(
            inputs,
            initial_value,
            reduce_fn,
            pool_size,
            strides,
            padding,
        )
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
    return _pool(inputs, -np.inf, lax.max, pool_size, strides, padding)


def average_pool(
    inputs,
    pool_size,
    strides,
    padding,
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
            np.ones(shape, inputs.dtype),
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
    return np.array(
        jax.lax.conv_general_dilated(
            inputs,
            kernel if is_tensor(kernel) else kernel.numpy(),
            strides,
            padding,
            rhs_dilation=dilation_rate,
            dimension_numbers=dimension_numbers,
            feature_group_count=feature_group_count,
        )
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
    kernel = np.reshape(
        kernel if is_tensor(kernel) else kernel.numpy(),
        kernel.shape[:-2] + (1, feature_group_count * kernel.shape[-1]),
    )
    return np.array(
        jax.lax.conv_general_dilated(
            inputs,
            kernel,
            strides,
            padding,
            rhs_dilation=dilation_rate,
            dimension_numbers=dimension_numbers,
            feature_group_count=feature_group_count,
        )
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

    return np.array(
        jax.lax.conv_transpose(
            inputs,
            kernel if is_tensor(kernel) else kernel.numpy(),
            strides,
            padding=padding_values,
            rhs_dilation=dilation_rate,
            dimension_numbers=dimension_numbers,
            transpose_kernel=True,
        )
    )


def one_hot(x, num_classes, axis=-1, dtype="float32", sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with numpy backend")
    x = convert_to_tensor(x)
    input_shape = x.shape

    x = x.reshape(-1)
    if not num_classes:
        num_classes = np.max(x) + 1

    batch_size = x.shape[0]
    categorical = np.zeros((batch_size, num_classes), dtype=dtype)
    valid_indices = x >= 0
    categorical[np.arange(batch_size)[valid_indices], x[valid_indices]] = 1

    # First, reshape the array with the extra dimension at the end
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)

    # Then, move this new dimension to the right place (according to axis)
    if axis != -1:
        categorical = np.moveaxis(categorical, -1, axis)

    return categorical


def multi_hot(x, num_classes, axis=-1, dtype="float32", sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with numpy backend")
    x = convert_to_tensor(x)
    reduction_axis = 1 if len(x.shape) > 1 else 0
    outputs = np.max(
        one_hot(cast(x, "int32"), num_classes, axis=axis, dtype=dtype),
        axis=reduction_axis,
    )
    return outputs


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = np.array(target)
    output = np.array(output)

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
        log_prob = log_softmax(output, axis=axis)
    else:
        output = output / np.sum(output, axis, keepdims=True)
        output = np.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
        log_prob = np.log(output)
    return -np.sum(target * log_prob, axis=axis)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = np.array(target, dtype="int32")
    output = np.array(output)
    if len(target.shape) == len(output.shape) and target.shape[-1] == 1:
        target = np.squeeze(target, axis=-1)

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
        log_prob = log_softmax(output, axis=axis)
    else:
        output = output / np.sum(output, axis, keepdims=True)
        output = np.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
        log_prob = np.log(output)
    target = one_hot(target, output.shape[axis], axis=axis)
    return -np.sum(target * log_prob, axis=axis)


def binary_crossentropy(target, output, from_logits=False):
    target = np.array(target)
    output = np.array(output)

    if target.shape != output.shape:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        output = sigmoid(output)

    output = np.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
    bce = target * np.log(output)
    bce += (1.0 - target) * np.log(1.0 - output)
    return -bce


def moments(x, axes, keepdims=False, synchronized=False):
    if synchronized:
        raise NotImplementedError(
            "Argument synchronized=True is not supported with NumPy."
        )
    axes = tuple(axes) if isinstance(axes, list) else axes
    # The dynamic range of float16 is too limited for statistics. As a
    # workaround, we simply perform the operations on float32 and convert back
    # to float16
    need_cast = False
    ori_dtype = backend.standardize_dtype(x.dtype)
    if ori_dtype == "float16":
        need_cast = True
        x = cast(x, "float32")

    mean = np.mean(x, axes, keepdims=True)

    # The variance is computed using $Var = E[|x|^2] - |E[x]|^2$, It is faster
    # but less numerically stable.
    variance = np.mean(np.square(x), axis=axes, keepdims=True) - np.square(mean)

    if not keepdims:
        mean = np.squeeze(mean, axes)
        variance = np.squeeze(variance, axes)
    if need_cast:
        # avoid overflow and underflow when casting from float16 to float32
        mean = np.clip(mean, np.finfo(np.float16).min, np.finfo(np.float16).max)
        variance = np.clip(
            variance, np.finfo(np.float16).min, np.finfo(np.float16).max
        )
        mean = cast(mean, ori_dtype)
        variance = cast(variance, ori_dtype)
    return mean, variance


def batch_normalization(
    x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3
):
    shape = [1] * len(x.shape)
    shape[axis] = mean.shape[0]
    mean = np.reshape(mean, shape)
    variance = np.reshape(variance, shape)

    inv = 1.0 / np.sqrt(variance + epsilon)
    if scale is not None:
        scale = np.reshape(scale, shape)
        inv = inv * scale

    res = -mean * inv
    if offset is not None:
        offset = np.reshape(offset, shape)
        res = res + offset

    return x * inv + res


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
    dtype = backend.result_type(output.dtype, "float32")
    output = output.astype(dtype)

    def _lengths_to_paddings(lengths, max_length):
        indices = np.arange(max_length).reshape(
            (1,) * lengths.ndim + (max_length,)
        )
        lengths = np.expand_dims(lengths, axis=-1)
        elem_valid = indices < lengths
        return np.logical_not(elem_valid)

    target_paddings = _lengths_to_paddings(target_length, max_label_length)
    output_paddings = _lengths_to_paddings(output_length, max_input_length)
    target_paddings = target_paddings.astype(output.dtype)
    output_paddings = output_paddings.astype(output.dtype)

    logprobs = log_softmax(output, axis=-1)
    label_lengths = max_label_length - np.sum(target_paddings, axis=1).astype(
        np.int32
    )

    # repeat[b, n] == 1.0 when label[b, n] == label[b, n+1].
    repeat = (target[:, :-1] == target[:, 1:]).astype(np.float32)
    repeat = np.pad(repeat, ((0, 0), (0, 1)))

    logprobs_phi = logprobs[:, :, mask_index : mask_index + 1]  # [B, T, 1]
    logprobs_phi = np.transpose(logprobs_phi, (1, 0, 2))  # [T, B, 1]

    _one_hot = one_hot(target, num_classes=num_classes)  # [B, N, K]
    logprobs_emit = np.einsum("btk,bnk->btn", logprobs, _one_hot)
    logprobs_emit = np.transpose(logprobs_emit, (1, 0, 2))  # [T, B, N]

    # [B, N]
    logalpha_phi_init = (
        np.ones((batch_size, max_label_length + 1), dtype=output.dtype)
        * log_epsilon
    )
    logalpha_phi_init[:, 0] = 0.0
    logalpha_emit_init = (
        np.ones((batch_size, max_label_length), dtype=output.dtype)
        * log_epsilon
    )

    def update_phi_score(phi, added_score):
        # Update `phi[:, 1:]`` with adding `added_score` in log space.
        return np.concatenate(
            [phi[:, :1], np.logaddexp(phi[:, 1:], added_score)], axis=-1
        )

    def loop_body(prev, x):
        prev_phi, prev_emit = prev
        # emit-to-phi epsilon transition, except if the next label is repetition
        prev_phi_orig = prev_phi
        prev_phi = update_phi_score(prev_phi, prev_emit + log_epsilon * repeat)

        logprob_emit, logprob_phi, pad = x

        # phi-to-emit transition
        next_emit = np.logaddexp(
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

    def np_scan(f, init, xs):
        carry = init
        ys = []
        for x in zip(*xs):
            carry, y = f(carry, x)
            ys.append(y)
        result = []
        for i in range(len(ys[0])):
            result.append(np.stack([y[i] for y in ys]))
        return carry, result

    xs = (logprobs_emit, logprobs_phi, output_paddings.transpose((1, 0)))
    _, (logalpha_phi, logalpha_emit) = np_scan(
        loop_body, (logalpha_phi_init, logalpha_emit_init), xs
    )

    # last row needs to be updated with the last epsilon transition
    logalpha_phi_last = update_phi_score(logalpha_phi[-1], logalpha_emit[-1])
    logalpha_phi[-1] = logalpha_phi_last

    # extract per_seq_loss
    # [B, N+1]
    _one_hot = one_hot(label_lengths, num_classes=max_label_length + 1)
    per_seq_loss = -np.einsum("bn,bn->b", logalpha_phi_last, _one_hot)
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

    indices = np.argmax(inputs, axis=-1).astype("int32")
    scores = np.max(inputs, axis=-1)

    seqlen_mask = np.arange(max_length)[None, :]
    seqlen_mask = seqlen_mask >= sequence_lengths[:, None]

    indices = np.where(seqlen_mask, mask_index, indices)
    scores = np.where(seqlen_mask, 0.0, scores)

    if merge_repeated:
        repeat_mask = indices[:, 1:] == indices[:, :-1]
        repeat_mask = np.pad(repeat_mask, ((0, 0), (1, 0)))
        indices = np.where(repeat_mask, mask_index, indices)

    # We set to -1 for blank labels
    invalid_mask = indices == mask_index
    indices = np.where(invalid_mask, -1, indices)

    # We rearrange the indices by moving `mask_index` to the end of the array
    order = np.expand_dims(np.arange(max_length), axis=0)  # [1, N]
    order = np.tile(order, (batch_size, 1))  # [B, N]
    order = np.where(invalid_mask, max_length, order)
    order = np.argsort(order, axis=-1)
    indices = np.take_along_axis(indices, order, axis=-1)

    scores = -np.sum(scores, axis=1)[:, None]
    indices = np.expand_dims(indices, axis=0)
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
    inputs = log_softmax(inputs, axis=-1)
    seqlen_mask = np.arange(max_seq_len)[None, :] >= sequence_lengths[:, None]

    if mask_index is None:
        mask_index = num_classes - 1

    # This is a workaround for the fact that np.argsort does not support
    # the order parameter which is used to break ties when scores are equal.
    # For compatibility with the tensorflow implementation, we flip the inputs
    # and the mask_index, and then flip the classes back to the correct indices
    inputs = np.flip(inputs, axis=2)
    mask_index = num_classes - mask_index - 1

    _pad = -1

    init_paths = np.full(
        (batch_size, 2 * beam_width, max_seq_len), _pad, dtype=np.int32
    )

    num_init_paths = np.min(np.array([num_classes, beam_width]))
    max_classes = np.argsort(inputs[:, 0], axis=1)[:, -num_init_paths:]
    init_classes = np.where(max_classes == mask_index, _pad, max_classes)
    init_paths[:, :num_init_paths, 0] = init_classes

    init_scores = np.full(
        (batch_size, 2 * beam_width), -np.inf, dtype=inputs.dtype
    )
    init_scores[:, :num_init_paths] = np.take_along_axis(
        inputs[:, 0], max_classes, axis=1
    )
    init_masked = init_paths[:, :, 0] == _pad

    def _extend_paths(paths, scores, masked, x):
        paths = np.repeat(paths, num_classes, axis=0)
        scores = np.repeat(scores, num_classes)
        masked = np.repeat(masked, num_classes)

        path_tail_index = np.argmax(paths == _pad, axis=1)
        paths_arange = np.arange(2 * beam_width * num_classes)
        path_tails = paths[paths_arange, path_tail_index - 1]
        path_tails = np.where(path_tail_index == 0, _pad, path_tails)

        classes = np.arange(num_classes)
        classes[mask_index] = _pad
        classes = np.tile(classes, 2 * beam_width)

        prev_masked = masked
        masked = classes == _pad

        masked_repeat = ~prev_masked & (path_tails == classes)
        classes = np.where(masked_repeat, _pad, classes)
        paths[paths_arange, path_tail_index] = classes

        x = np.tile(x, 2 * beam_width)
        scores = scores + x

        return paths, scores, masked

    def _merge_scores(unique_inverse, scores):
        scores_max = np.max(scores)
        scores_exp = np.exp(scores - scores_max)
        scores = np.zeros_like(scores)
        for i, u in enumerate(unique_inverse):
            scores[u] += scores_exp[i]
        scores = np.log(scores) + scores_max
        return scores

    def _prune_paths(paths, scores, masked):
        paths, unique_inverse = np.unique(paths, return_inverse=True, axis=0)
        pad_size = (2 * num_classes * beam_width) - len(paths)
        if pad_size > 0:
            paths = np.pad(paths, [[0, pad_size], [0, 0]], constant_values=_pad)
        paths = paths[: 2 * num_classes * beam_width]
        if len(unique_inverse.shape) >= 2:
            unique_inverse = np.squeeze(unique_inverse, axis=1)

        emit_scores = np.where(masked, -np.inf, scores)
        mask_scores = np.where(masked, scores, -np.inf)

        emit_scores = _merge_scores(unique_inverse, emit_scores)
        mask_scores = _merge_scores(unique_inverse, mask_scores)

        total_scores = np.logaddexp(emit_scores, mask_scores)
        top_indices = np.argsort(total_scores, kind="stable")[-beam_width:]

        paths = paths[top_indices]
        emit_scores = emit_scores[top_indices]
        mask_scores = mask_scores[top_indices]

        paths = np.tile(paths, (2, 1))
        scores = np.concatenate([emit_scores, mask_scores])
        masked = np.concatenate(
            [np.zeros(beam_width, bool), np.ones(beam_width, bool)]
        )

        return paths, scores, masked

    def _decode_step(paths, scores, masked, x):
        paths, scores, masked = _extend_paths(paths, scores, masked, x)
        paths, scores, masked = _prune_paths(paths, scores, masked)
        return paths, scores, masked

    def _step(prev, x):
        paths, scores, masked = prev
        x, seqlen_mask = x
        if not seqlen_mask:
            paths, scores, masked = _decode_step(paths, scores, masked, x)
        return (paths, scores, masked), None

    def _decode_batch(
        init_paths, init_scores, init_masked, inputs, seqlen_mask
    ):
        def np_scan_only_carry(f, init, xs):
            carry = init
            for x in zip(*xs):
                carry, y = f(carry, x)
            return carry, None

        (paths, scores, masked), _ = np_scan_only_carry(
            _step,
            (init_paths, init_scores, init_masked),
            (inputs[1:], seqlen_mask[1:]),
        )

        paths, unique_inverse = np.unique(paths, return_inverse=True, axis=0)
        pad_size = (2 * num_classes * beam_width) - len(paths)
        if pad_size > 0:
            paths = np.pad(paths, [[0, pad_size], [0, 0]], constant_values=_pad)
        paths = paths[: 2 * num_classes * beam_width]
        if len(unique_inverse.shape) >= 2:
            unique_inverse = np.squeeze(unique_inverse, axis=1)
        scores = _merge_scores(unique_inverse, scores)

        top_indices = np.argsort(scores)[-top_paths:][::-1]
        paths = paths[top_indices]
        scores = scores[top_indices]

        return paths, scores

    results = [
        _decode_batch(p, s, m, i, sm)
        for p, s, m, i, sm in zip(
            init_paths, init_scores, init_masked, inputs, seqlen_mask
        )
    ]
    paths = np.stack([r[0] for r in results])
    scores = np.stack([r[1] for r in results])

    # convert classes back to the correct indices
    paths = np.where(paths == _pad, _pad, num_classes - paths - 1)
    paths = np.transpose(paths, [1, 0, 2])
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
    mse = np.mean(np.square(x1 - x2))
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr
