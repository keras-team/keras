import jax
import numpy as np
from jax import lax
from jax import numpy as jnp

from keras.src.backend import standardize_data_format
from keras.src.backend import standardize_dtype
from keras.src.backend.common.backend_utils import (
    compute_conv_transpose_padding_args_for_jax,
)
from keras.src.backend.config import epsilon
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

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

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
        output = np.clip(output, epsilon(), 1.0 - epsilon())
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
        output = np.clip(output, epsilon(), 1.0 - epsilon())
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

    output = np.clip(output, epsilon(), 1.0 - epsilon())
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
    ori_dtype = standardize_dtype(x.dtype)
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


def ctc_decode(
    inputs,
    sequence_length,
    strategy,
    beam_width=100,
    top_paths=1,
    merge_repeated=True,
    mask_index=None,
):
    raise NotImplementedError(
        "NumPy backend does not yet support CTC decoding."
    )
