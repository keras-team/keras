import math
import warnings

import tensorflow as tf

from keras.src import backend
from keras.src.backend.common.backend_utils import (
    compute_adaptive_pooling_window_sizes,
)
from keras.src.backend.common.backend_utils import (
    compute_conv_transpose_output_shape,
)
from keras.src.backend.tensorflow.core import cast
from keras.src.backend.tensorflow.core import convert_to_tensor


def relu(x):
    return tf.nn.relu(x)


def relu6(x):
    return tf.nn.relu6(x)


def sigmoid(x):
    logits = x
    output = tf.nn.sigmoid(x)
    output._keras_logits = logits
    return output


def sparse_sigmoid(x):
    x = convert_to_tensor(x)
    return tf.where(
        x <= -1,
        tf.constant(0.0, dtype=x.dtype),
        tf.where(x >= 1, tf.constant(1.0, dtype=x.dtype), 0.5 * (x + 1)),
    )


def tanh(x):
    return tf.nn.tanh(x)


def tanh_shrink(x):
    return x - tf.math.tanh(x)


def softplus(x):
    return tf.math.softplus(x)


def softsign(x):
    return tf.nn.softsign(x)


def soft_shrink(x, threshold=0.5):
    return tf.where(
        x > threshold,
        x - threshold,
        tf.where(x < -threshold, x + threshold, tf.zeros_like(x)),
    )


def sparse_plus(x):
    return tf.where(
        x <= -1,
        tf.zeros_like(x),
        tf.where(x < 1, (1 / 4) * tf.pow(x + 1, 2), x),
    )


def silu(x):
    return tf.nn.silu(x)


def squareplus(x, b=4):
    x = convert_to_tensor(x)
    b = convert_to_tensor(b, dtype=x.dtype)
    y = x + tf.sqrt(tf.square(x) + b)
    return y / 2


def log_sigmoid(x):
    return tf.math.log_sigmoid(x)


def leaky_relu(x, negative_slope=0.2):
    return tf.nn.leaky_relu(x, alpha=negative_slope)


def hard_sigmoid(x):
    x = convert_to_tensor(x)
    return relu6(x + tf.constant(3.0, x.dtype)) / tf.constant(6.0, x.dtype)


def hard_silu(x):
    return x * hard_sigmoid(x)


def elu(x, alpha=1.0):
    res = tf.nn.elu(x)
    if alpha == 1:
        return res
    else:
        return tf.where(x > 0, res, alpha * res)


def selu(x):
    return tf.nn.selu(x)


def gelu(x, approximate=True):
    x = convert_to_tensor(x)
    return tf.nn.gelu(x, approximate=approximate)


def celu(x, alpha=1.0):
    return tf.maximum(x, 0.0) + alpha * tf.math.expm1(
        tf.minimum(x, 0.0) / alpha
    )


def glu(x, axis=-1):
    if x.shape[axis] % 2 != 0:
        raise ValueError(
            "axis size must be divisible by 2. "
            f"Received: x.shape={x.shape} with axis={axis}"
        )
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=axis)
    return x1 * tf.sigmoid(x2)


def hard_tanh(x):
    return tf.clip_by_value(x, clip_value_min=-1.0, clip_value_max=1.0)


def hard_shrink(x, threshold=0.5):
    return tf.where(tf.abs(x) > threshold, x, tf.zeros_like(x))


def threshold(x, threshold, default_value):
    return tf.where(x > threshold, x, default_value)


def softmax(x, axis=-1):
    logits = x
    if axis is None:
        # Unlike numpy, tf will handle axis=None as axis=-1.
        # We need this workaround for the reduction on every dim.
        output = tf.reshape(x, [-1])
        output = tf.nn.softmax(output, axis=-1)
        output = tf.reshape(output, tf.shape(x))
    else:
        output = tf.nn.softmax(x, axis=axis)
    output._keras_logits = logits
    return output


def log_softmax(x, axis=-1):
    if axis is None:
        # Unlike numpy, tf will handle axis=None as axis=-1.
        # We need this workaround for the reduction on every dim.
        output = tf.reshape(x, [-1])
        output = tf.nn.log_softmax(output, axis=-1)
        return tf.reshape(output, tf.shape(x))
    return tf.nn.log_softmax(x, axis=axis)


def sparsemax(x, axis=-1):
    # Sort logits along the specified axis in descending order
    logits = convert_to_tensor(x)
    logits_sorted = tf.sort(logits, direction="DESCENDING", axis=axis)
    logits_cumsum = tf.cumsum(logits_sorted, axis=axis)
    r = tf.range(1, tf.shape(logits)[axis] + 1, dtype=logits.dtype)
    r_shape = [1] * len(logits.shape)
    r_shape[axis] = -1  # Broadcast to match the target axis
    r = tf.reshape(r, r_shape)  # Reshape for broadcasting
    support = logits_sorted - (logits_cumsum - 1) / r > 0
    # Find the threshold
    logits_cumsum_safe = tf.where(support, logits_cumsum, 0.0)
    k = tf.reduce_sum(tf.cast(support, logits.dtype), axis=axis, keepdims=True)
    tau = (tf.reduce_sum(logits_cumsum_safe, axis=axis, keepdims=True) - 1) / k
    output = tf.maximum(logits - tau, 0.0)
    return output


def _transpose_spatial_inputs(inputs):
    num_spatial_dims = len(inputs.shape) - 2
    # Tensorflow pooling does not support `channels_first` format, so
    # we need to transpose to `channels_last` format.
    if num_spatial_dims == 1:
        inputs = tf.transpose(inputs, (0, 2, 1))
    elif num_spatial_dims == 2:
        inputs = tf.transpose(inputs, (0, 2, 3, 1))
    elif num_spatial_dims == 3:
        inputs = tf.transpose(inputs, (0, 2, 3, 4, 1))
    else:
        raise ValueError(
            "Pooling inputs's shape must be 3, 4 or 5, corresponding to 1D, 2D "
            f"and 3D inputs. But received shape: {inputs.shape}."
        )
    return inputs


def _transpose_spatial_outputs(outputs):
    # Undo the transpose in `_transpose_spatial_inputs`.
    num_spatial_dims = len(outputs.shape) - 2
    if num_spatial_dims == 1:
        outputs = tf.transpose(outputs, (0, 2, 1))
    elif num_spatial_dims == 2:
        outputs = tf.transpose(outputs, (0, 3, 1, 2))
    elif num_spatial_dims == 3:
        outputs = tf.transpose(outputs, (0, 4, 1, 2, 3))
    return outputs


def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    strides = pool_size if strides is None else strides
    padding = padding.upper()
    tf_data_format = _convert_data_format("channels_last", len(inputs.shape))
    if data_format == "channels_first":
        # Tensorflow pooling does not support `channels_first` format, so
        # we need to transpose to `channels_last` format.
        inputs = _transpose_spatial_inputs(inputs)

    outputs = tf.nn.max_pool(
        inputs,
        pool_size,
        strides,
        padding,
        tf_data_format,
    )
    if data_format == "channels_first":
        outputs = _transpose_spatial_outputs(outputs)
    return outputs


def average_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    strides = pool_size if strides is None else strides
    padding = padding.upper()
    tf_data_format = _convert_data_format("channels_last", len(inputs.shape))
    if data_format == "channels_first":
        # Tensorflow pooling does not support `channels_first` format, so
        # we need to transpose to `channels_last` format.
        inputs = _transpose_spatial_inputs(inputs)

    outputs = tf.nn.avg_pool(
        inputs,
        pool_size,
        strides,
        padding,
        tf_data_format,
    )
    if data_format == "channels_first":
        outputs = _transpose_spatial_outputs(outputs)
    return outputs


def _compute_static_gather_indices(
    input_dim, output_size, small_window, big_window
):
    """Compute gather indices for Two-Pool Gather method (corrected)."""
    window_starts = tf.cast(
        tf.floor(
            tf.cast(tf.range(output_size), tf.float32)
            * tf.cast(input_dim, tf.float32)
            / tf.cast(output_size, tf.float32)
        ),
        tf.int32,
    )
    window_ends = tf.cast(
        tf.math.ceil(
            tf.cast(tf.range(1, output_size + 1), tf.float32)
            * tf.cast(input_dim, tf.float32)
            / tf.cast(output_size, tf.float32)
        ),
        tf.int32,
    )

    window_ends = tf.minimum(window_ends, input_dim)
    window_starts = tf.minimum(window_starts, input_dim - 1)

    window_sizes = window_ends - window_starts
    is_big_window = tf.equal(window_sizes, big_window)

    small_pool_len = max(1, input_dim - small_window + 1)

    small_indices = window_starts
    big_indices = window_starts + small_pool_len

    gather_indices = tf.where(is_big_window, big_indices, small_indices)
    return tf.cast(gather_indices, tf.int32)


def _adaptive_average_pool1d(inputs, output_size, data_format="channels_first"):
    if isinstance(output_size, int):
        output_size = (output_size,)
    if data_format == "channels_first":
        inputs = tf.transpose(inputs, (0, 2, 1))

    static_shape = inputs.shape.as_list()
    l_static = static_shape[1]
    out_l = output_size[0]

    if l_static is None:
        raise ValueError(
            "Input length must be statically known for adaptive pooling"
        )

    small_l, big_l = compute_adaptive_pooling_window_sizes(l_static, out_l)
    gather_l = _compute_static_gather_indices(l_static, out_l, small_l, big_l)

    small_pool_l = tf.nn.pool(
        inputs,
        window_shape=(small_l,),
        pooling_type="AVG",
        strides=(1,),
        padding="VALID",
        data_format="NWC",
    )
    big_pool_l = tf.nn.pool(
        inputs,
        window_shape=(big_l,),
        pooling_type="AVG",
        strides=(1,),
        padding="VALID",
        data_format="NWC",
    )

    combined_l = tf.concat([small_pool_l, big_pool_l], axis=1)
    pooled_l = tf.gather(combined_l, gather_l, axis=1)

    if data_format == "channels_first":
        pooled_l = tf.transpose(pooled_l, (0, 2, 1))
    return pooled_l


def _adaptive_max_pool1d(inputs, output_size, data_format="channels_first"):
    if isinstance(output_size, int):
        output_size = (output_size,)
    if data_format == "channels_first":
        inputs = tf.transpose(inputs, (0, 2, 1))

    static_shape = inputs.shape.as_list()
    l_static = static_shape[1]
    out_l = output_size[0]

    if l_static is None:
        raise ValueError(
            "Input length must be statically known for adaptive pooling"
        )

    small_l, big_l = compute_adaptive_pooling_window_sizes(l_static, out_l)
    gather_l = _compute_static_gather_indices(l_static, out_l, small_l, big_l)

    small_pool_l = tf.nn.pool(
        inputs,
        window_shape=(small_l,),
        pooling_type="MAX",
        strides=(1,),
        padding="VALID",
        data_format="NWC",
    )
    big_pool_l = tf.nn.pool(
        inputs,
        window_shape=(big_l,),
        pooling_type="MAX",
        strides=(1,),
        padding="VALID",
        data_format="NWC",
    )

    combined_l = tf.concat([small_pool_l, big_pool_l], axis=1)
    pooled_l = tf.gather(combined_l, gather_l, axis=1)

    if data_format == "channels_first":
        pooled_l = tf.transpose(pooled_l, (0, 2, 1))
    return pooled_l


def _adaptive_average_pool2d(inputs, output_size, data_format="channels_first"):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    if data_format == "channels_first":
        inputs = tf.transpose(inputs, (0, 2, 3, 1))

    static_shape = inputs.shape.as_list()
    h_static = static_shape[1]
    w_static = static_shape[2]
    out_h, out_w = output_size

    if h_static is None or w_static is None:
        raise ValueError(
            "Input spatial dimensions must be "
            "statically known for adaptive pooling"
        )

    small_h, big_h = compute_adaptive_pooling_window_sizes(h_static, out_h)
    small_w, big_w = compute_adaptive_pooling_window_sizes(w_static, out_w)

    gather_h = _compute_static_gather_indices(h_static, out_h, small_h, big_h)
    gather_w = _compute_static_gather_indices(w_static, out_w, small_w, big_w)

    small_pool_h = tf.nn.pool(
        inputs,
        window_shape=(small_h, 1),
        pooling_type="AVG",
        strides=(1, 1),
        padding="VALID",
        data_format="NHWC",
    )
    big_pool_h = tf.nn.pool(
        inputs,
        window_shape=(big_h, 1),
        pooling_type="AVG",
        strides=(1, 1),
        padding="VALID",
        data_format="NHWC",
    )

    combined_h = tf.concat([small_pool_h, big_pool_h], axis=1)
    pooled_h = tf.gather(combined_h, gather_h, axis=1)

    small_pool_w = tf.nn.pool(
        pooled_h,
        window_shape=(1, small_w),
        pooling_type="AVG",
        strides=(1, 1),
        padding="VALID",
        data_format="NHWC",
    )
    big_pool_w = tf.nn.pool(
        pooled_h,
        window_shape=(1, big_w),
        pooling_type="AVG",
        strides=(1, 1),
        padding="VALID",
        data_format="NHWC",
    )

    combined_w = tf.concat([small_pool_w, big_pool_w], axis=2)
    pooled_w = tf.gather(combined_w, gather_w, axis=2)

    if data_format == "channels_first":
        pooled_w = tf.transpose(pooled_w, (0, 3, 1, 2))

    return pooled_w


def _adaptive_max_pool2d(inputs, output_size, data_format="channels_first"):
    """Adaptive Max Pooling 2D using Two-Pool Gather method."""
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    if data_format == "channels_first":
        inputs = tf.transpose(inputs, (0, 2, 3, 1))

    static_shape = inputs.shape.as_list()
    h_static = static_shape[1]
    w_static = static_shape[2]
    out_h, out_w = output_size

    if h_static is None or w_static is None:
        raise ValueError(
            "Input spatial dimensions must be "
            "statically known for adaptive pooling"
        )

    small_h, big_h = compute_adaptive_pooling_window_sizes(h_static, out_h)
    small_w, big_w = compute_adaptive_pooling_window_sizes(w_static, out_w)

    gather_h = _compute_static_gather_indices(h_static, out_h, small_h, big_h)
    gather_w = _compute_static_gather_indices(w_static, out_w, small_w, big_w)

    small_pool_h = tf.nn.pool(
        inputs,
        window_shape=(small_h, 1),
        pooling_type="MAX",
        strides=(1, 1),
        padding="VALID",
        data_format="NHWC",
    )
    big_pool_h = tf.nn.pool(
        inputs,
        window_shape=(big_h, 1),
        pooling_type="MAX",
        strides=(1, 1),
        padding="VALID",
        data_format="NHWC",
    )

    combined_h = tf.concat([small_pool_h, big_pool_h], axis=1)
    pooled_h = tf.gather(combined_h, gather_h, axis=1)

    small_pool_w = tf.nn.pool(
        pooled_h,
        window_shape=(1, small_w),
        pooling_type="MAX",
        strides=(1, 1),
        padding="VALID",
        data_format="NHWC",
    )
    big_pool_w = tf.nn.pool(
        pooled_h,
        window_shape=(1, big_w),
        pooling_type="MAX",
        strides=(1, 1),
        padding="VALID",
        data_format="NHWC",
    )

    combined_w = tf.concat([small_pool_w, big_pool_w], axis=2)
    pooled_w = tf.gather(combined_w, gather_w, axis=2)

    if data_format == "channels_first":
        pooled_w = tf.transpose(pooled_w, (0, 3, 1, 2))

    return pooled_w


def _adaptive_average_pool3d(inputs, output_size, data_format="channels_first"):
    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)

    if data_format == "channels_first":
        inputs = tf.transpose(inputs, (0, 2, 3, 4, 1))

    static_shape = inputs.shape.as_list()
    d_static = static_shape[1]
    h_static = static_shape[2]
    w_static = static_shape[3]
    out_d, out_h, out_w = output_size

    if d_static is None or h_static is None or w_static is None:
        raise ValueError(
            "Input spatial dimensions must be "
            "statically known for adaptive pooling"
        )

    small_d, big_d = compute_adaptive_pooling_window_sizes(d_static, out_d)
    small_h, big_h = compute_adaptive_pooling_window_sizes(h_static, out_h)
    small_w, big_w = compute_adaptive_pooling_window_sizes(w_static, out_w)

    gather_d = _compute_static_gather_indices(d_static, out_d, small_d, big_d)
    gather_h = _compute_static_gather_indices(h_static, out_h, small_h, big_h)
    gather_w = _compute_static_gather_indices(w_static, out_w, small_w, big_w)

    small_pool_d = tf.nn.pool(
        inputs,
        window_shape=(small_d, 1, 1),
        pooling_type="AVG",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )
    big_pool_d = tf.nn.pool(
        inputs,
        window_shape=(big_d, 1, 1),
        pooling_type="AVG",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )

    combined_d = tf.concat([small_pool_d, big_pool_d], axis=1)
    pooled_d = tf.gather(combined_d, gather_d, axis=1)

    small_pool_h = tf.nn.pool(
        pooled_d,
        window_shape=(1, small_h, 1),
        pooling_type="AVG",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )
    big_pool_h = tf.nn.pool(
        pooled_d,
        window_shape=(1, big_h, 1),
        pooling_type="AVG",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )

    combined_h = tf.concat([small_pool_h, big_pool_h], axis=2)
    pooled_h = tf.gather(combined_h, gather_h, axis=2)

    small_pool_w = tf.nn.pool(
        pooled_h,
        window_shape=(1, 1, small_w),
        pooling_type="AVG",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )
    big_pool_w = tf.nn.pool(
        pooled_h,
        window_shape=(1, 1, big_w),
        pooling_type="AVG",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )

    combined_w = tf.concat([small_pool_w, big_pool_w], axis=3)
    pooled_w = tf.gather(combined_w, gather_w, axis=3)

    if data_format == "channels_first":
        pooled_w = tf.transpose(pooled_w, (0, 4, 1, 2, 3))

    return pooled_w


def _adaptive_max_pool3d(inputs, output_size, data_format="channels_first"):
    """Adaptive Max Pooling 3D using Two-Pool Gather method."""
    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)

    if data_format == "channels_first":
        inputs = tf.transpose(inputs, (0, 2, 3, 4, 1))

    static_shape = inputs.shape.as_list()
    d_static = static_shape[1]
    h_static = static_shape[2]
    w_static = static_shape[3]
    out_d, out_h, out_w = output_size

    if d_static is None or h_static is None or w_static is None:
        raise ValueError(
            "Input spatial dimensions must be "
            "statically known for adaptive pooling"
        )

    small_d, big_d = compute_adaptive_pooling_window_sizes(d_static, out_d)
    small_h, big_h = compute_adaptive_pooling_window_sizes(h_static, out_h)
    small_w, big_w = compute_adaptive_pooling_window_sizes(w_static, out_w)

    gather_d = _compute_static_gather_indices(d_static, out_d, small_d, big_d)
    gather_h = _compute_static_gather_indices(h_static, out_h, small_h, big_h)
    gather_w = _compute_static_gather_indices(w_static, out_w, small_w, big_w)

    small_pool_d = tf.nn.pool(
        inputs,
        window_shape=(small_d, 1, 1),
        pooling_type="MAX",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )
    big_pool_d = tf.nn.pool(
        inputs,
        window_shape=(big_d, 1, 1),
        pooling_type="MAX",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )

    combined_d = tf.concat([small_pool_d, big_pool_d], axis=1)
    pooled_d = tf.gather(combined_d, gather_d, axis=1)

    small_pool_h = tf.nn.pool(
        pooled_d,
        window_shape=(1, small_h, 1),
        pooling_type="MAX",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )
    big_pool_h = tf.nn.pool(
        pooled_d,
        window_shape=(1, big_h, 1),
        pooling_type="MAX",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )

    combined_h = tf.concat([small_pool_h, big_pool_h], axis=2)
    pooled_h = tf.gather(combined_h, gather_h, axis=2)

    small_pool_w = tf.nn.pool(
        pooled_h,
        window_shape=(1, 1, small_w),
        pooling_type="MAX",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )
    big_pool_w = tf.nn.pool(
        pooled_h,
        window_shape=(1, 1, big_w),
        pooling_type="MAX",
        strides=(1, 1, 1),
        padding="VALID",
        data_format="NDHWC",
    )

    combined_w = tf.concat([small_pool_w, big_pool_w], axis=3)
    pooled_w = tf.gather(combined_w, gather_w, axis=3)

    if data_format == "channels_first":
        pooled_w = tf.transpose(pooled_w, (0, 4, 1, 2, 3))

    return pooled_w


def adaptive_average_pool(inputs, output_size, data_format=None):
    data_format = backend.standardize_data_format(data_format)
    ndims = len(inputs.shape) - 2
    if ndims == 1:
        return _adaptive_average_pool1d(inputs, output_size, data_format)
    elif ndims == 2:
        return _adaptive_average_pool2d(inputs, output_size, data_format)
    elif ndims == 3:
        return _adaptive_average_pool3d(inputs, output_size, data_format)
    else:
        raise ValueError(
            "adaptive_average_pool supports 1D, 2D, or 3D inputs only."
        )


def adaptive_max_pool(inputs, output_size, data_format=None):
    data_format = backend.standardize_data_format(data_format)
    ndims = len(inputs.shape) - 2
    if ndims == 1:
        return _adaptive_max_pool1d(inputs, output_size, data_format)
    elif ndims == 2:
        return _adaptive_max_pool2d(inputs, output_size, data_format)
    elif ndims == 3:
        return _adaptive_max_pool3d(inputs, output_size, data_format)
    else:
        raise ValueError(
            "adaptive_max_pool supports 1D, 2D, or 3D inputs only."
        )


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
    data_format=None,
    dilation_rate=1,
):
    def _conv():
        tf_data_format = _convert_data_format(data_format, len(inputs.shape))
        result = tf.nn.convolution(
            inputs,
            kernel,
            strides,
            padding.upper(),
            data_format=tf_data_format,
            dilations=dilation_rate,
        )
        result_shape = result.shape
        if (
            result_shape.is_fully_defined()
            and math.prod(result_shape.as_list()) == 0
        ):
            raise ValueError(
                "The convolution operation resulted in an empty output. "
                "Output shape:"
                f" {result_shape}. This can happen if the input is too small "
                "for the given kernel size, strides, dilation rate, and "
                "padding mode. Please check the input shape and convolution "
                "parameters."
            )
        return result

    # Certain ops are are broken in Tensorflow on CPU only.
    # We can work around by compiling the op with XLA.
    @tf.function(jit_compile=True)
    def _conv_xla():
        return _conv()

    # Channels first "NCDHW" (3d convolutions) are broken on CPU without XLA.
    needs_xla = data_format == "channels_first" and len(inputs.shape) == 5
    # grouped convolutions are broken on CPU without XLA.
    data_format = backend.standardize_data_format(data_format)
    if data_format == "channels_last":
        channels = inputs.shape[-1]
    else:
        channels = inputs.shape[1]
    needs_xla = needs_xla or channels != kernel.shape[-2]
    if needs_xla:
        return _conv_xla()
    else:
        return _conv()


def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = len(inputs.shape) - 2
    if num_spatial_dims > 2:
        raise ValueError(
            "`inputs` rank must be 3 (1D conv) or 4 (2D conv). Received: "
            f"{inputs.ndim}."
        )
    # Because we use `tf.nn.depthwise_conv2d` for both 1D and 2D convs, we set
    # `tf_data_format` using 2D conv format.
    tf_data_format = _convert_data_format(data_format, 4)
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
    data_format=None,
    dilation_rate=1,
):
    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = len(inputs.shape) - 2
    if num_spatial_dims > 2:
        raise ValueError(
            "`num_spatial_dims` must be 1 or 2. Received: "
            f"num_spatial_dims={num_spatial_dims}."
        )
    # Because we use `tf.nn.separable_conv2d` for both 1D and 2D convs, we set
    # `tf_data_format` using 2D conv format.
    tf_data_format = _convert_data_format(data_format, 4)
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
    else:
        strides = (1, 1) + strides
    return tf.nn.separable_conv2d(
        inputs,
        depthwise_kernel,
        pointwise_kernel,
        strides,
        padding,
        data_format=tf_data_format,
        dilations=dilation_rate,
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
    tf_data_format = _convert_data_format(data_format, len(inputs.shape))
    kernel_size = kernel.shape[:-2]
    filters = kernel.shape[-2]
    input_shape = list(inputs.shape)
    symbolic_shape = tf.shape(inputs)
    for i, e in enumerate(input_shape):
        if e is None:
            input_shape[i] = symbolic_shape[i]
    output_shape = compute_conv_transpose_output_shape(
        input_shape,
        kernel_size,
        filters,
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


def one_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    x = convert_to_tensor(x, dtype="int64")
    if dtype is None:
        dtype = "float32"
    else:
        dtype = backend.standardize_dtype(dtype)
    if sparse:
        # We don't use `tf.sparse.bincount`, it doesn't handle negative indices
        # and only support rank 1 and 2 tensors (`one_hot` adds a dimension).
        if axis < 0:
            axis = axis + len(x.shape) + 1
        values_count = math.prod(x.shape)
        values = tf.reshape(x, (values_count,))
        # We deal with negative inputs by having zeros in the output although
        # it's useless. It makes shapes static.
        values = tf.cast(tf.greater_equal(values, 0), dtype=dtype)
        indices = [tf.range(dim) for dim in x.shape]
        indices = tf.meshgrid(*indices, indexing="ij")
        indices.insert(axis, tf.maximum(x, 0))  # Deal with negative indices
        indices = [tf.reshape(a, (values_count, 1)) for a in indices]
        indices = [tf.cast(a, tf.int64) for a in indices]
        indices = tf.concat(indices, axis=1)
        shape = list(x.shape)
        shape.insert(axis, num_classes)
        return tf.SparseTensor(indices, values, shape)
    on_value, off_value = (True, False) if dtype == "bool" else (None, None)
    return tf.one_hot(
        x,
        num_classes,
        on_value=on_value,
        off_value=off_value,
        axis=axis,
        dtype=dtype,
    )


def multi_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    reduction_axis = 1 if len(x.shape) > 1 else 0
    if backend.standardize_dtype(dtype) == "bool":
        if sparse:
            # `tf.sparse.reduce_max` doesn't work on bool and there is no
            # `tf.sparse.reduce_any`.
            outputs = one_hot(
                x, num_classes, axis=axis, dtype="int8", sparse=True
            )
            outputs = tf.sparse.reduce_max(
                outputs, axis=reduction_axis, output_is_sparse=True
            )
            outputs_shape = outputs.shape
            outputs = tf.cast(outputs, dtype)
            outputs.set_shape(outputs_shape)
            return outputs
        else:
            outputs = one_hot(x, num_classes, axis=axis, dtype=dtype)
            return tf.reduce_any(outputs, axis=reduction_axis)
    else:
        if sparse:
            # We don't use `tf.sparse.bincount`, it doesn't handle negative
            # indices and has a rank limitation.
            outputs = one_hot(
                x, num_classes, axis=axis, dtype=dtype, sparse=True
            )
            return tf.sparse.reduce_max(
                outputs, axis=reduction_axis, output_is_sparse=True
            )
        else:
            outputs = one_hot(x, num_classes, axis=axis, dtype=dtype)
            return tf.reduce_max(outputs, axis=reduction_axis)


def _get_logits(output, from_logits, op_type, fn_name):
    """Retrieves logits tensor from maybe-softmax or maybe-sigmoid tensor."""
    output_ = output
    from_logits_ = from_logits

    has_keras_logits = hasattr(output, "_keras_logits")
    if has_keras_logits:
        output_ = output._keras_logits
        from_logits_ = True

    from_expected_op_type = (
        hasattr(output, "op")
        and not isinstance(output, (tf.__internal__.EagerTensor, tf.Variable))
        and output.op.type == op_type
    ) and not has_keras_logits

    if from_expected_op_type:
        # When softmax activation function is used for output operation, we
        # use logits from the softmax function directly to compute loss in order
        # to prevent collapsing zero when training.
        assert len(output.op.inputs) == 1
        output_ = output.op.inputs[0]
        from_logits_ = True

    if from_logits and (has_keras_logits or from_expected_op_type):
        warnings.warn(
            f'"`{fn_name}` received `from_logits=True`, but '
            f"the `output` argument was produced by a {op_type} "
            "activation and thus does not represent logits. "
            "Was this intended?",
            stacklevel=2,
        )
    return output_, from_logits_


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy between an output tensor and a target tensor.

    Args:
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is `True`, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1` corresponds to data
            format `channels_last`, and `axis=1` corresponds to data format
            `channels_first`.

    Returns:
        Output tensor.

    Example:

    >>> a = tf.constant([1., 0., 0., 0., 1., 0., 0., 0., 1.], shape=[3,3])
    >>> print(a)
    tf.Tensor(
      [[1. 0. 0.]
       [0. 1. 0.]
       [0. 0. 1.]], shape=(3, 3), dtype=float32)
    >>> b = tf.constant([.9, .05, .05, .05, .89, .06, .05, .01, .94],
    ...                 shape=[3, 3])
    >>> print(b)
    tf.Tensor(
      [[0.9  0.05 0.05]
       [0.05 0.89 0.06]
       [0.05 0.01 0.94]], shape=(3, 3), dtype=float32)
    >>> loss = categorical_crossentropy(a, b)
    >>> print(np.around(loss, 5))
    [0.10536 0.11653 0.06188]
    >>> loss = categorical_crossentropy(a, a)
    >>> print(np.around(loss, 5))
    [0. 0. 0.]
    """
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)

    if len(target.shape) < 1:
        raise ValueError(
            "Arguments `target` and `output` must be at least rank 1. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    if len(target.shape) != len(output.shape):
        raise ValueError(
            "Arguments `target` and `output` must have the same rank "
            "(ndim). Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    for e1, e2 in zip(target.shape, output.shape):
        if e1 is not None and e2 is not None and e1 != e2:
            raise ValueError(
                "Arguments `target` and `output` must have the same shape. "
                "Received: "
                f"target.shape={target.shape}, output.shape={output.shape}"
            )

    output, from_logits = _get_logits(
        output, from_logits, "Softmax", "categorical_crossentropy"
    )
    if from_logits:
        return tf.nn.softmax_cross_entropy_with_logits(
            labels=target, logits=output, axis=axis
        )

    # Adjust the predictions so that the probability of
    # each class for every sample adds up to 1
    # This is needed to ensure that the cross entropy is
    # computed correctly.
    output = output / tf.reduce_sum(output, axis, keepdims=True)

    # Compute cross entropy from probabilities.
    output = tf.clip_by_value(
        output, backend.epsilon(), 1.0 - backend.epsilon()
    )
    return -tf.reduce_sum(target * tf.math.log(output), axis)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy with integer targets.

    Args:
        target: An integer tensor.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1` corresponds to data
            format `channels_last`, and `axis=1` corresponds to data format
            `channels_first`.

    Returns:
        Output tensor.
    """
    if axis != -1 and axis != len(output.shape) - 1:
        raise ValueError(
            f"Only axis=-1 is currently supported. Received: axis={axis}"
        )
    output, from_logits = _get_logits(
        output, from_logits, "Softmax", "sparse_categorical_crossentropy"
    )

    target = tf.convert_to_tensor(target)
    target = tf.cast(target, dtype="int64")
    output = tf.convert_to_tensor(output)
    if len(target.shape) == len(output.shape) and target.shape[-1] == 1:
        target = tf.squeeze(target, axis=-1)

    if len(output.shape) < 1:
        raise ValueError(
            "Argument `output` must be at least rank 1. "
            "Received: "
            f"output.shape={output.shape}"
        )
    if len(target.shape) != len(output.shape[:-1]):
        raise ValueError(
            "Argument `output` must have rank (ndim) `target.ndim - 1`. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    for e1, e2 in zip(target.shape, output.shape[:-1]):
        if e1 is not None and e2 is not None and e1 != e2:
            raise ValueError(
                "Arguments `target` and `output` must have the same shape "
                "up until the last dimension: "
                f"target.shape={target.shape}, output.shape={output.shape}"
            )

    if not from_logits:
        output = tf.clip_by_value(
            output, backend.epsilon(), 1 - backend.epsilon()
        )
        output = tf.math.log(output)

    result = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target, logits=output
    )
    return result


def binary_crossentropy(target, output, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.

    Args:
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.

    Returns:
        A tensor.
    """
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)

    if len(target.shape) != len(output.shape):
        raise ValueError(
            "Arguments `target` and `output` must have the same rank "
            "(ndim). Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    for e1, e2 in zip(target.shape, output.shape):
        if e1 is not None and e2 is not None and e1 != e2:
            raise ValueError(
                "Arguments `target` and `output` must have the same shape. "
                "Received: "
                f"target.shape={target.shape}, output.shape={output.shape}"
            )

    output, from_logits = _get_logits(
        output, from_logits, "Sigmoid", "binary_crossentropy"
    )

    if from_logits:
        return tf.nn.sigmoid_cross_entropy_with_logits(
            labels=target, logits=output
        )

    # Compute cross entropy from probabilities.
    output = tf.clip_by_value(
        output, backend.epsilon(), 1.0 - backend.epsilon()
    )
    bce = target * tf.math.log(output)
    bce += (1 - target) * tf.math.log(1 - output)
    return -bce


def moments(x, axes, keepdims=False, synchronized=False):
    # The dynamic range of float16 is too limited for statistics. As a
    # workaround, we simply perform the operations on float32 and convert back
    # to float16
    need_cast = False
    ori_dtype = backend.standardize_dtype(x.dtype)
    if ori_dtype in ("float16", "bfloat16"):
        need_cast = True
        x = cast(x, "float32")

    if synchronized:
        mean, variance = _compute_moments_sync(x, axes, keepdims)
    else:
        mean, variance = _compute_moments(x, axes, keepdims)
    if need_cast:
        # avoid overflow and underflow when casting from float16 to float32
        mean = tf.clip_by_value(mean, tf.float16.min, tf.float16.max)
        variance = tf.clip_by_value(variance, tf.float16.min, tf.float16.max)
        mean = cast(mean, ori_dtype)
        variance = cast(variance, ori_dtype)
    return mean, variance


def _compute_moments_sync(x, axes, keepdims):
    replica_ctx = tf.distribute.get_replica_context()
    if not replica_ctx:
        return _compute_moments(x, axes, keepdims)

    local_count = tf.ones_like(x, name="count")

    local_sum = tf.reduce_sum(x, axis=axes, keepdims=True)
    local_squared_sum = tf.reduce_sum(tf.square(x), axis=axes, keepdims=True)
    local_count = tf.reduce_sum(local_count, axis=axes, keepdims=True)

    # TODO(b/163099951): batch the all-reduces once we sort out the
    # ordering issue for NCCL. We don't have a mechanism to launch
    # NCCL in the same order in each replica nowadays, so we limit
    # NCCL to batch all-reduces.
    y_sum = replica_ctx.all_reduce(tf.distribute.ReduceOp.SUM, local_sum)
    y_squared_sum = replica_ctx.all_reduce(
        tf.distribute.ReduceOp.SUM, local_squared_sum
    )
    count_sum = replica_ctx.all_reduce(tf.distribute.ReduceOp.SUM, local_count)

    mean = tf.math.divide_no_nan(y_sum, count_sum)
    y_squared_mean = tf.math.divide_no_nan(y_squared_sum, count_sum)
    # var = E(x^2) - E(x)^2
    variance = tf.maximum(y_squared_mean - tf.square(mean), 0.0)
    if not keepdims:
        mean = tf.squeeze(mean, axes)
        variance = tf.squeeze(variance, axes)

    return mean, variance


def _compute_moments(x, axes, keepdims):
    return tf.nn.moments(x, axes, keepdims=keepdims)


def batch_normalization(
    x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3
):
    if axis != -1:
        shape = [1] * len(x.shape)
        shape[axis] = mean.shape[0]
        mean = tf.reshape(mean, shape)
        variance = tf.reshape(variance, shape)
        if offset is not None:
            offset = tf.reshape(offset, shape)
        if scale is not None:
            scale = tf.reshape(scale, shape)

    return tf.nn.batch_normalization(
        x=x,
        mean=mean,
        variance=variance,
        offset=offset,
        scale=scale,
        variance_epsilon=epsilon,
    )


def ctc_loss(target, output, target_length, output_length, mask_index=0):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)
    target = tf.cast(target, dtype="int32")

    # `tf.nn.ctc_loss` will internally cast to float32 when the input is float16
    # or bfloat16. Additionally, it will raise an error when the input is
    # float64. As a result, we perform the casting externally and add support
    # for float64.
    result_dtype = backend.result_type(output.dtype, "float32")
    compute_dtype = "float32" if result_dtype == "float64" else result_dtype
    output = tf.cast(output, compute_dtype)
    loss = tf.nn.ctc_loss(
        labels=target,
        logits=output,
        label_length=target_length,
        logit_length=output_length,
        blank_index=mask_index,
        logits_time_major=False,
    )
    return tf.cast(loss, result_dtype)


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
    input_shape = tf.shape(inputs)
    num_samples, num_steps = input_shape[0], input_shape[1]
    inputs = tf.transpose(inputs, (1, 0, 2))

    dtype = backend.result_type(inputs.dtype, "float32")
    inputs = tf.cast(inputs, dtype)

    sequence_lengths = convert_to_tensor(sequence_lengths, dtype="int32")
    if strategy == "greedy":
        (decoded, scores) = tf.nn.ctc_greedy_decoder(
            inputs=inputs,
            sequence_length=sequence_lengths,
            merge_repeated=merge_repeated,
            blank_index=mask_index,
        )
    elif strategy == "beam_search":
        # Move `mask_index` column to the last position since this is the
        # default for `tf.nn.ctc_beam_search_decoder`
        if mask_index is not None:
            inputs_before = inputs[..., :mask_index]
            inputs_mask = inputs[..., mask_index : mask_index + 1]
            inputs_after = inputs[..., mask_index + 1 :]
            inputs = tf.concat(
                [inputs_before, inputs_after, inputs_mask], axis=-1
            )
        (decoded, scores) = tf.nn.ctc_beam_search_decoder(
            inputs=inputs,
            sequence_length=sequence_lengths,
            beam_width=beam_width,
            top_paths=top_paths,
        )
    else:
        raise ValueError(
            f"Invalid strategy {strategy}. Supported values are "
            "'greedy' and 'beam_search'."
        )

    # Postprocess sparse tensor
    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
    decoded_dense = tf.stack(decoded_dense, axis=0)
    decoded_dense = tf.cast(decoded_dense, "int32")

    # We need to recover the labels because we swapped the indices earlier
    if strategy == "beam_search" and mask_index is not None:
        if mask_index < 0:
            mask_index = mask_index + input_shape[-1]
        decoded_dense = tf.where(
            decoded_dense >= mask_index, decoded_dense + 1, decoded_dense
        )
    return decoded_dense, scores


def psnr(x1, x2, max_val):
    from keras.src.backend.tensorflow.numpy import log10

    if x1.shape != x2.shape:
        raise ValueError(
            f"Input shapes {x1.shape} and {x2.shape} must "
            "match for PSNR calculation. "
        )

    max_val = convert_to_tensor(max_val, dtype=x2.dtype)
    mse = tf.reduce_mean(tf.square(x1 - x2))
    psnr = 20 * log10(max_val) - 10 * log10(mse)
    return psnr


def _get_large_negative(dtype):
    dtype = backend.standardize_dtype(dtype)
    val = 65500.0 if dtype == "float16" else 3.38953e38
    return tf.constant(val * -0.7, dtype=dtype)


def _apply_masks(logits, mask, is_causal):
    if mask is None and not is_causal:
        return logits

    combined_mask = tf.ones_like(logits, dtype="bool")
    if mask is not None:
        combined_mask = tf.logical_and(combined_mask, mask)

    if is_causal:
        logits_shape = tf.shape(logits)
        T, S = logits_shape[2], logits_shape[3]
        mask = tf.linalg.band_part(tf.ones((T, S), "bool"), -1, 0)
        mask = mask[None, None, :, :]
        combined_mask = tf.logical_and(combined_mask, mask)

    padded_logits = tf.where(
        combined_mask, logits, _get_large_negative(logits.dtype)
    )
    return padded_logits


def _dot_product_attention_xla(query, key, value, bias, mask, is_causal, scale):
    logits_dtype = backend.result_type(query.dtype, "float32")
    logits = tf.einsum("BTNH,BSNH->BNTS", query, key, optimize="optimal")
    logits = tf.cast(logits, logits_dtype)
    logits = tf.multiply(logits, tf.cast(scale, logits.dtype))

    if bias is not None:
        logits = tf.add(logits, tf.cast(bias, logits.dtype))

    padded_logits = _apply_masks(logits, mask, is_causal)

    # Softmax is always carried out in high precision.
    probs_dtype = backend.result_type(padded_logits.dtype, "float32")
    probs = tf.cast(
        tf.nn.softmax(tf.cast(padded_logits, probs_dtype), axis=-1), key.dtype
    )
    return tf.einsum("BNTS,BSNH->BTNH", probs, value, optimize="optimal")


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
    if flash_attention is None:
        flash_attention = False
    if flash_attention:
        raise ValueError(
            "Flash attention is not supported in tensorflow backend."
        )

    # Ref: jax.nn.dot_product_attention
    # https://github.com/jax-ml/jax/blob/jax-v0.4.32/jax/_src/nn/functions.py#L828
    # Not support `query_seq_lengths` and `key_value_seq_lengths` args
    query = convert_to_tensor(query)
    key = convert_to_tensor(key)
    value = convert_to_tensor(value)
    if len(query.shape) != 4:
        raise ValueError(
            "`dot_product_attention` only supports 4D inputs. "
            f"Received: query.shape={query.shape}, key.shape={key.shape}, "
            f"value.shape={value.shape}."
        )
    compute_dtype = backend.result_type(query.dtype, key.dtype, value.dtype)
    query = cast(query, compute_dtype)
    key = cast(key, compute_dtype)
    value = cast(value, compute_dtype)
    if bias is not None:
        bias = convert_to_tensor(bias, dtype=compute_dtype)

    H = tf.shape(key)[-1]
    scale = (1.0 / tf.sqrt(tf.cast(H, "float32"))) if scale is None else scale
    return _dot_product_attention_xla(
        query, key, value, bias, mask, is_causal, scale
    )


def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    """Tensorflow implementation of Unfold.
    Extract sliding local blocks from a **NCHW** batched image tensor.

    Args:
        input: 4-D tensor, shape (N, C, H, W)  **required**.
        kernel_size: int or (kH, kW)
        dilation: int or (dH, dW), default 1
        padding: int or (pH, pW), default 0
        stride: int or (sH, sW), default 1

    Returns:
        3-D tensor, shape (N, C*kH*kW, L)
    """
    k = (
        (kernel_size, kernel_size)
        if isinstance(kernel_size, int)
        else kernel_size
    )
    d = (dilation, dilation) if isinstance(dilation, int) else dilation
    p = (padding, padding) if isinstance(padding, int) else padding
    s = (stride, stride) if isinstance(stride, int) else stride
    N, C, H, W = input.shape

    # ---- padding ----
    if any(_ > 0 for _ in p):
        input = tf.pad(input, [[0, 0], [0, 0], [p[0], p[0]], [p[1], p[1]]])
    x = tf.transpose(input, [0, 2, 3, 1])  # (N, H, W, C)
    patches = tf.image.extract_patches(
        images=x,
        sizes=[1, k[0], k[1], 1],
        strides=[1, s[0], s[1], 1],
        rates=[1, d[0], d[1], 1],
        padding="VALID",
    )  # (N, nH, nW, kH*kW*C)

    N, nH, nW, D = patches.shape
    patches = tf.reshape(
        patches, [N, nH, nW, k[0], k[1], C]
    )  # (N, nH, nW, kH, kW, C)
    patches = tf.transpose(
        patches, [0, 5, 3, 4, 1, 2]
    )  # (N, C, kH, kW, nH, nW)
    patches = tf.reshape(patches, [N, C * k[0] * k[1], nH * nW])
    return patches
