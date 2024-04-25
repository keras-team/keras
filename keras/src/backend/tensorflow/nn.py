import math
import warnings

import tensorflow as tf

from keras.src import backend
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


def tanh(x):
    return tf.nn.tanh(x)


def softplus(x):
    return tf.math.softplus(x)


def softsign(x):
    return tf.nn.softsign(x)


def silu(x):
    return tf.nn.silu(x)


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
        return tf.nn.convolution(
            inputs,
            kernel,
            strides,
            padding.upper(),
            data_format=tf_data_format,
            dilations=dilation_rate,
        )

    # Reason for making this function is in Tensorflow, `groups > 1` does not
    # work on CPU for `tf.nn.convolution`, but wrapping it by XLA works.
    @tf.function(jit_compile=True)
    def _conv_xla():
        return _conv()

    data_format = backend.standardize_data_format(data_format)
    if data_format == "channels_last":
        channels = inputs.shape[-1]
    else:
        channels = inputs.shape[1]
    if channels != kernel.shape[-2]:
        # If kernel's in_channel does not match input's channels,  it indicates
        # convolution is broken down into groups.
        return _conv_xla()
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
            "{inputs.ndim}."
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


def one_hot(x, num_classes, axis=-1, dtype="float32", sparse=False):
    x = convert_to_tensor(x, dtype="int64")
    if dtype is None:
        dtype = "float32"
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
    return tf.one_hot(x, num_classes, axis=axis, dtype=dtype)


def multi_hot(x, num_classes, axis=-1, dtype="float32", sparse=False):
    x = convert_to_tensor(x)
    reduction_axis = 1 if len(x.shape) > 1 else 0
    one_hot_outputs = one_hot(
        cast(x, "int32"), num_classes, axis=axis, dtype=dtype, sparse=sparse
    )
    if sparse:
        # We don't use `tf.sparse.bincount`, it doesn't handle negative indices.
        return tf.sparse.reduce_max(
            one_hot_outputs, axis=reduction_axis, output_is_sparse=True
        )
    return tf.reduce_max(one_hot_outputs, axis=reduction_axis)


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


def ctc_loss(
    target,
    output,
    target_length,
    output_length,
    mask_index=0,
):
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
    sequence_length,
    strategy="greedy",
    beam_width=100,
    top_paths=1,
    merge_repeated=True,
    mask_index=None,
):
    inputs = convert_to_tensor(inputs)
    input_shape = tf.shape(inputs)
    num_samples, num_steps = input_shape[0], input_shape[1]
    inputs = tf.transpose(inputs, (1, 0, 2))

    dtype = backend.result_type(inputs.dtype, "float32")
    inputs = tf.cast(inputs, dtype)

    sequence_length = convert_to_tensor(sequence_length, dtype="int32")
    if strategy == "greedy":
        (decoded, scores) = tf.nn.ctc_greedy_decoder(
            inputs=inputs,
            sequence_length=sequence_length,
            merge_repeated=merge_repeated,
            blank_index=mask_index,
        )
    elif strategy == "beam_search":
        (decoded, scores) = tf.nn.ctc_beam_search_decoder(
            inputs=inputs,
            sequence_length=sequence_length,
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
