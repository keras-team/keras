import openvino.opset15 as ov_opset
from openvino import Type

from keras.src import backend
from keras.src.backend.openvino.core import OPENVINO_DTYPES
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import get_ov_output


def relu(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.relu(x).output(0))


def relu6(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.clamp(x, 0.0, 6.0).output(0))


def celu(x, alpha=1.0):
    x = get_ov_output(x)
    const_zero = get_ov_output(0.0, x.get_element_type())
    const_alpha = get_ov_output(alpha, x.get_element_type())
    const_one = get_ov_output(1.0, x.get_element_type())
    exp_x_div_alpha = ov_opset.exp(ov_opset.divide(x, const_alpha)).output(0)
    negative_branch = ov_opset.multiply(
        const_alpha, ov_opset.subtract(exp_x_div_alpha, const_one)
    )

    celu_x = ov_opset.add(
        ov_opset.maximum(x, const_zero).output(0),
        ov_opset.minimum(negative_branch, const_zero).output(0),
    )
    return OpenVINOKerasTensor(celu_x.output(0))


def sigmoid(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.sigmoid(x).output(0))


def tanh(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.tanh(x).output(0))


def tanh_shrink(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.subtract(x, ov_opset.tanh(x)).output(0))


def hard_tanh(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.clamp(x, -1.0, 1.0).output(0))


def soft_shrink(x, threshold=0.5):
    x = get_ov_output(x)
    et = x.get_element_type()
    thr = get_ov_output(threshold, et)
    zero = get_ov_output(0.0, et)
    abs_x = ov_opset.abs(x)
    sub = ov_opset.subtract(abs_x, thr)
    shrunk = ov_opset.maximum(sub, zero)
    sign = ov_opset.sign(x)
    out = ov_opset.multiply(sign, shrunk)
    return OpenVINOKerasTensor(out.output(0))


def hard_shrink(x, threshold=0.5):
    x = get_ov_output(x)
    et = x.get_element_type()
    thr = get_ov_output(threshold, et)
    zero = get_ov_output(0.0, et)
    cond = ov_opset.greater(ov_opset.abs(x), thr)
    out = ov_opset.select(cond, x, zero)
    return OpenVINOKerasTensor(out.output(0))


def softplus(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.softplus(x).output(0))


def softsign(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.softsign(x).output(0))


def silu(x):
    x = get_ov_output(x)
    beta = get_ov_output(1.0, x.get_element_type())
    return OpenVINOKerasTensor(ov_opset.swish(x, beta=beta).output(0))


def log_sigmoid(x):
    x = get_ov_output(x)
    neg_x = ov_opset.negative(x)
    return OpenVINOKerasTensor(
        ov_opset.negative(ov_opset.softplus(neg_x)).output(0)
    )


def leaky_relu(x, negative_slope=0.2):
    x = get_ov_output(x)
    slope_const = ov_opset.constant(
        negative_slope, x.get_element_type()
    ).output(0)
    leaky_relu = ov_opset.prelu(x, slope_const).output(0)
    return OpenVINOKerasTensor(leaky_relu)


def sparse_sigmoid(x):
    x = get_ov_output(x)
    et = x.get_element_type()
    one = get_ov_output(1.0, et)
    neg_one = get_ov_output(-1.0, et)
    half = get_ov_output(0.5, et)
    y = ov_opset.minimum(ov_opset.maximum(x, neg_one), one)
    out = ov_opset.multiply(half, ov_opset.add(y, one))
    return OpenVINOKerasTensor(out.output(0))


def hard_sigmoid(x):
    x = get_ov_output(x)
    alpha = get_ov_output(1.0 / 6.0, x.get_element_type())
    beta = get_ov_output(0.5, x.get_element_type())
    return OpenVINOKerasTensor(ov_opset.hard_sigmoid(x, alpha, beta).output(0))


def hard_silu(x):
    hard_sigmoid_output = get_ov_output(hard_sigmoid(x))
    x = get_ov_output(x)
    return OpenVINOKerasTensor(
        ov_opset.multiply(x, hard_sigmoid_output).output(0)
    )


def elu(x, alpha=1.0):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.elu(x, alpha).output(0))


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    x = get_ov_output(x)
    alpha = get_ov_output(alpha, x.get_element_type())
    scale = get_ov_output(scale, x.get_element_type())
    return OpenVINOKerasTensor(ov_opset.selu(x, alpha, scale).output(0))


def gelu(x, approximate=True):
    x = get_ov_output(x)
    approximate_mode = "erf"
    if approximate:
        approximate_mode = "tanh"
    return OpenVINOKerasTensor(ov_opset.gelu(x, approximate_mode).output(0))


def softmax(x, axis=-1):
    x = get_ov_output(x)
    if axis is None:
        x_shape = ov_opset.shape_of(x)
        flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
        flatten_x = ov_opset.reshape(x, flatten_shape, False).output(0)
        softmax_x = ov_opset.softmax(flatten_x, 0).output(0)
        return OpenVINOKerasTensor(
            ov_opset.reshape(softmax_x, x_shape, False).output(0)
        )
    return OpenVINOKerasTensor(ov_opset.softmax(x, axis).output(0))


def log_softmax(x, axis=-1):
    x = get_ov_output(x)
    if axis is None:
        x_shape = ov_opset.shape_of(x)
        flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
        flatten_x = ov_opset.reshape(x, flatten_shape, False).output(0)
        log_softmax_x = ov_opset.log_softmax(flatten_x, 0).output(0)
        return OpenVINOKerasTensor(
            ov_opset.reshape(log_softmax_x, x_shape, False).output(0)
        )
    return OpenVINOKerasTensor(ov_opset.log_softmax(x, axis).output(0))


def squareplus(x, b=4):
    x = get_ov_output(x)
    et = x.get_element_type()
    b = get_ov_output(b, et)
    two = get_ov_output(2.0, et)
    x_squared = ov_opset.multiply(x, x)
    inside = ov_opset.add(x_squared, b)
    root = ov_opset.sqrt(inside)
    summed = ov_opset.add(x, root)
    out = ov_opset.divide(summed, two)
    return OpenVINOKerasTensor(out.output(0))


def sparse_plus(x):
    x = get_ov_output(x)
    et = x.get_element_type()
    one = get_ov_output(1.0, et)
    neg_one = get_ov_output(-1.0, et)
    zero = get_ov_output(0.0, et)
    quarter = get_ov_output(0.25, et)
    x_plus_1 = ov_opset.add(x, one)
    quad = ov_opset.multiply(quarter, ov_opset.multiply(x_plus_1, x_plus_1))
    leq_than_neg_one = ov_opset.less_equal(x, neg_one)
    less_than_one = ov_opset.less(x, one)
    out = ov_opset.select(
        leq_than_neg_one,
        zero,
        ov_opset.select(less_than_one, quad, x),
    )
    return OpenVINOKerasTensor(out.output(0))


def threshold(x, threshold, default_value):
    x = get_ov_output(x)
    et = x.get_element_type()
    thr = get_ov_output(threshold, et)
    dv = get_ov_output(default_value, et)
    cond = ov_opset.greater(x, thr)
    out = ov_opset.select(cond, x, dv)
    return OpenVINOKerasTensor(out.output(0))


def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    num_spatial_dims = (
        get_ov_output(inputs).get_partial_shape().rank.get_length() - 2
    )
    kwargs = {"dilations": [1] * num_spatial_dims}  # required for ov max_pool
    return _pool(
        inputs,
        pool_size,
        ov_opset.max_pool,
        strides,
        padding,
        data_format,
        **kwargs,
    )


def average_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    return _pool(
        inputs,
        pool_size,
        ov_opset.avg_pool,
        strides,
        padding,
        data_format,
        exclude_pad=True,
    )


def adaptive_average_pool(inputs, output_size, data_format=None):
    """Adaptive average pooling - OpenVINO backend not yet supported."""
    raise NotImplementedError("Adaptive pooling not implemented for OpenVINO.")


def adaptive_max_pool(inputs, output_size, data_format=None):
    """Adaptive max pooling - OpenVINO backend not yet supported."""
    raise NotImplementedError("Adaptive pooling not implemented for OpenVINO.")


def _pool(
    inputs,
    pool_size,
    pooling_func,
    strides=None,
    padding="valid",
    data_format=None,
    **kwargs,
):
    data_format = backend.standardize_data_format(data_format)
    inputs = get_ov_output(inputs)

    num_spatial_dims = inputs.get_partial_shape().rank.get_length() - 2
    if isinstance(pool_size, int):
        pool_size = [pool_size] * num_spatial_dims

    if strides is None:
        strides = pool_size

    strides = _adjust_strides_dilation(strides, num_spatial_dims)
    pad_mode, pads_begin, pads_end = _adjust_padding(padding)
    inputs = _adjust_input(inputs, num_spatial_dims, data_format)
    pool_kwargs = {
        "kernel_shape": pool_size,
        "strides": strides,
        "auto_pad": pad_mode,
        "pads_begin": pads_begin,
        "pads_end": pads_end,
        **kwargs,
    }
    pooled = pooling_func(inputs, **pool_kwargs).output(0)
    adjusted_pooled = _adjust_outputs(pooled, num_spatial_dims, data_format)
    return OpenVINOKerasTensor(adjusted_pooled)


def _adjust_strides_dilation(
    x,
    num_spatial_dims,
):
    # Helper function that converts an operand to a spatial operand.
    x = (x,) * num_spatial_dims if isinstance(x, int) else x
    # OpenVINO expects input in NCHW layout
    # x = [1, 1] + list(x)
    x = list(x)
    return x


def _adjust_padding(
    padding,
):
    padding = padding.lower() if isinstance(padding, str) else padding
    if padding == "same":
        return "SAME_UPPER", [], []
    elif padding == "same_lower":
        return "SAME_LOWER", [], []
    elif padding == "valid":
        return "VALID", [], []
    pads_begin = []
    pads_end = []
    for padding_pair in padding:
        pads_begin.append(padding_pair[0])
        pads_end.append(padding_pair[1])
    return "EXPLICIT", pads_begin, pads_end


def _adjust_input(inputs, num_spatial_dims, data_format):
    if data_format == "channels_first":
        return inputs
    if num_spatial_dims == 1:
        permutation = [0, 2, 1]
    elif num_spatial_dims == 2:
        permutation = [0, 3, 1, 2]
    else:
        permutation = [0, 4, 1, 2, 3]
    permutation = ov_opset.constant(permutation, Type.i32)
    return ov_opset.transpose(inputs, permutation).output(0)


def _adjust_kernel(kernel, num_spatial_dims):
    if num_spatial_dims == 1:
        permutation = [2, 1, 0]
    elif num_spatial_dims == 2:
        permutation = [3, 2, 0, 1]
    else:
        permutation = [4, 3, 0, 1, 2]
    permutation = ov_opset.constant(permutation, Type.i32)
    return ov_opset.transpose(kernel, permutation).output(0)


def _adjust_depthwise_kernel(kernel, num_spatial_dims):
    # kernel layout: filter_H, filter_W, C_IN, Ch_mul
    if num_spatial_dims == 1:
        # kernel layout: filter_H, C_IN, Ch_mul
        permutation = [1, 2, 0]
    elif num_spatial_dims == 2:
        # kernel layout: filter_H, filter_W, C_IN, Ch_mul
        permutation = [2, 3, 0, 1]
    else:
        # kernel layout: filter_H, filter_W, filter_Z, C_IN, Ch_mul
        permutation = [3, 4, 0, 1, 2]
    permutation = ov_opset.constant(permutation, Type.i32)
    return ov_opset.transpose(kernel, permutation).output(0)


def _adjust_outputs(outputs, num_spatial_dims, data_format):
    if data_format == "channels_first":
        return outputs
    # convert a tensor from NCHW to NHWC layout
    if num_spatial_dims == 1:
        permutation = [0, 2, 1]
    elif num_spatial_dims == 2:
        permutation = [0, 2, 3, 1]
    else:
        permutation = [0, 2, 3, 4, 1]
    permutation = ov_opset.constant(permutation, Type.i32)
    return ov_opset.transpose(outputs, permutation).output(0)


def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    inputs = get_ov_output(inputs)
    kernel = get_ov_output(kernel)

    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = inputs.get_partial_shape().rank.get_length() - 2

    if data_format == "channels_last":
        inputs_in_channels = inputs.get_partial_shape()[
            2 + num_spatial_dims - 1
        ]
    else:
        inputs_in_channels = inputs.get_partial_shape()[1]
    kernel_in_channels = kernel.get_partial_shape()[-2]

    strides = _adjust_strides_dilation(strides, num_spatial_dims)
    dilation_rate = _adjust_strides_dilation(dilation_rate, num_spatial_dims)
    pad_mode, pads_begin, pads_end = _adjust_padding(padding)
    inputs = _adjust_input(inputs, num_spatial_dims, data_format)
    kernel = _adjust_kernel(kernel, num_spatial_dims)

    num_groups = (
        inputs_in_channels.get_length() // kernel_in_channels.get_length()
    )
    if num_groups == 1:
        conv = ov_opset.convolution(
            inputs,
            kernel,
            strides,
            pads_begin,
            pads_end,
            dilation_rate,
            pad_mode,
        )
    else:
        input_shape = ov_opset.shape_of(inputs).output(0)
        filter_shape = ov_opset.shape_of(kernel).output(0)
        zero_const = ov_opset.constant([0], Type.i32).output(0)
        one_const = ov_opset.constant([1], Type.i32).output(0)
        two_const = ov_opset.constant([2], Type.i32).output(0)
        input_cin = ov_opset.slice(
            input_shape, one_const, two_const, one_const
        ).output(0)
        filter_cin = ov_opset.slice(
            filter_shape, one_const, two_const, one_const
        ).output(0)
        num_groups = ov_opset.divide(input_cin, filter_cin).output(0)

        # reshape the filter based on the number of groups information
        int_max_const = ov_opset.constant([2**31 - 1], Type.i32).output(0)
        filter_cout = ov_opset.slice(
            filter_shape, zero_const, one_const, one_const
        ).output(0)
        filter_new_cout = ov_opset.divide(filter_cout, num_groups).output(0)
        shape_cin_xy = ov_opset.slice(
            filter_shape, one_const, int_max_const, one_const
        ).output(0)
        filter_new_shape = ov_opset.concat(
            [num_groups, filter_new_cout, shape_cin_xy], 0
        ).output(0)
        new_filter = ov_opset.reshape(kernel, filter_new_shape, False).output(0)
        conv = ov_opset.group_convolution(
            inputs,
            new_filter,
            strides,
            pads_begin,
            pads_end,
            dilation_rate,
            pad_mode,
        )
    conv = _adjust_outputs(conv.output(0), num_spatial_dims, data_format)
    return OpenVINOKerasTensor(conv)


def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    inputs = get_ov_output(inputs)
    kernel = get_ov_output(kernel)

    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = inputs.get_partial_shape().rank.get_length() - 2

    assert data_format == "channels_last", (
        "`depthwise_conv` is supported only for channels_last data_format"
    )

    strides = _adjust_strides_dilation(strides, num_spatial_dims)
    dilation_rate = _adjust_strides_dilation(dilation_rate, num_spatial_dims)
    pad_mode, pads_begin, pads_end = _adjust_padding(padding)

    inputs = _adjust_input(inputs, num_spatial_dims, data_format)
    kernel = _adjust_depthwise_kernel(kernel, num_spatial_dims)
    unsqueeze_dim = ov_opset.constant([2], Type.i32)
    kernel = ov_opset.unsqueeze(kernel, unsqueeze_dim)

    group_conv = ov_opset.group_convolution(
        inputs, kernel, strides, pads_begin, pads_end, dilation_rate, pad_mode
    )
    group_conv = _adjust_outputs(
        group_conv.output(0), num_spatial_dims, data_format
    )
    return OpenVINOKerasTensor(group_conv)


def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    raise NotImplementedError(
        "`separable_conv` is not supported with openvino backend"
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
    raise NotImplementedError(
        "`conv_transpose` is not supported with openvino backend"
    )


def one_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    if sparse:
        raise ValueError("`sparse=True` is not supported with openvino backend")
    x = get_ov_output(x)
    if dtype is None:
        dtype = backend.floatx()
    ov_dtype = OPENVINO_DTYPES[dtype]
    on_value = get_ov_output(1, ov_dtype)
    off_value = get_ov_output(0, ov_dtype)
    one_hot_encoded = ov_opset.one_hot(
        x,
        depth=num_classes,
        axis=axis,
        on_value=on_value,
        off_value=off_value,
    ).output(0)
    return OpenVINOKerasTensor(one_hot_encoded)


def multi_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    reduction_axis = 1 if len(x.shape) > 1 else 0
    if backend.standardize_dtype(dtype) == "bool":
        outputs = one_hot(x, num_classes, axis=axis, dtype=dtype, sparse=sparse)
        result = ov_opset.reduce_logical_or(outputs, reduction_axis)
    else:
        outputs = one_hot(x, num_classes, axis=axis, dtype=dtype)
        result = ov_opset.reduce_max(outputs, reduction_axis)
    return OpenVINOKerasTensor(result.output(0))


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = get_ov_output(target)
    output = get_ov_output(output)

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
        log_prob = ov_opset.log_softmax(output, axis).output(0)
    else:
        sum_result = ov_opset.reduce_sum(output, axis, keep_dims=True).output(0)
        output = ov_opset.divide(output, sum_result).output(0)
        output = ov_opset.clamp(
            output, min_value=backend.epsilon(), max_value=1 - backend.epsilon()
        ).output(0)
        log_prob = ov_opset.log(output).output(0)
    result = ov_opset.multiply(target, log_prob).output(0)
    loss = ov_opset.reduce_sum(result, axis).output(0)
    loss = ov_opset.negative(loss).output(0)
    return OpenVINOKerasTensor(loss)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = get_ov_output(target)
    output = get_ov_output(output)

    if len(target.shape) == len(output.shape) and target.shape[-1] == 1:
        target = ov_opset.squeeze(target, -1).output(0)

    if len(output.shape) < 1:
        raise ValueError(
            "Argument `output` must be at least rank 1. "
            "Received: "
            f"output.shape={output.shape}"
        )

    output_shape_without_class_dim = list(output.shape)
    del output_shape_without_class_dim[axis]

    if list(target.shape) != output_shape_without_class_dim:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape "
            "up until the last dimension: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        log_prob = ov_opset.log_softmax(output, axis).output(0)
    else:
        sum = ov_opset.reduce_sum(output, axis, keep_dims=True).output(0)
        output = ov_opset.divide(output, sum).output(0)
        output = ov_opset.clamp(
            output, min_value=backend.epsilon(), max_value=1 - backend.epsilon()
        ).output(0)
        log_prob = ov_opset.log(output).output(0)

    output_type = output.get_element_type()
    on_val = ov_opset.constant(1, output_type).output(0)
    off_val = ov_opset.constant(0, output_type).output(0)
    one_hot_target = ov_opset.one_hot(
        target,
        depth=output.shape[axis],
        on_value=on_val,
        off_value=off_val,
        axis=axis,
    ).output(0)
    result = ov_opset.multiply(one_hot_target, log_prob).output(0)
    loss = ov_opset.reduce_sum(result, axis).output(0)
    loss = ov_opset.negative(loss).output(0)
    return OpenVINOKerasTensor(loss)


def binary_crossentropy(target, output, from_logits=False):
    target = get_ov_output(target)
    output = get_ov_output(output)

    if target.shape != output.shape:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        output = ov_opset.sigmoid(output).output(0)

    output = ov_opset.clamp(
        output, min_value=backend.epsilon(), max_value=1 - backend.epsilon()
    ).output(0)
    one = ov_opset.constant(1, target.get_element_type()).output(0)

    minus_output = ov_opset.subtract(one, output).output(0)
    minus_target = ov_opset.subtract(one, target).output(0)

    log_prob = ov_opset.log(output).output(0)
    minus_log_prob = ov_opset.log(minus_output).output(0)
    result = ov_opset.multiply(target, log_prob).output(0)
    minus_result = ov_opset.multiply(minus_target, minus_log_prob).output(0)
    bce = ov_opset.add(result, minus_result).output(0)
    bce = ov_opset.negative(bce).output(0)
    return OpenVINOKerasTensor(bce)


def moments(x, axes, keepdims=False, synchronized=False):
    x = get_ov_output(x)
    axes = ov_opset.constant(axes, Type.i32).output(0)
    # The variance is computed using $Var = E[|x|^2] - |E[x]|^2$, It is faster
    # but less numerically stable.
    mean = ov_opset.reduce_mean(x, axes, keepdims).output(0)
    const_two = ov_opset.constant(2, x.get_element_type()).output(0)
    squared_x = ov_opset.power(x, const_two).output(0)
    squared_mean = ov_opset.power(mean, const_two).output(0)
    squared_x_mean = ov_opset.reduce_mean(squared_x, axes, keepdims)
    mean = OpenVINOKerasTensor(mean)
    variance = OpenVINOKerasTensor(
        ov_opset.subtract(squared_x_mean, squared_mean).output(0)
    )
    return mean, variance


def batch_normalization(
    x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3
):
    x = get_ov_output(x)
    mean = get_ov_output(mean)
    variance = get_ov_output(variance)
    if offset is not None:
        offset = get_ov_output(offset)
    else:
        mean_shape = ov_opset.shape_of(mean)
        mean_type = mean.get_element_type()
        zero_const = ov_opset.constant([0], mean_type)
        offset = ov_opset.broadcast(zero_const, mean_shape)
    if scale is not None:
        scale = get_ov_output(scale)
    else:
        mean_shape = ov_opset.shape_of(mean)
        mean_type = mean.get_element_type()
        one_const = ov_opset.constant([1], mean_type)
        scale = ov_opset.broadcast(one_const, mean_shape)

    # adjust x input to have the second dimension representing the channel axis
    x_rank = x.get_partial_shape().rank.get_length()
    if axis < 0:
        axis += x_rank
    if axis != 1:
        perm_vector = list(range(0, x_rank))
        perm_vector[1] = axis
        perm_vector[axis] = 1
        perm_vector = ov_opset.constant(perm_vector, Type.i32).output(0)
        x = ov_opset.transpose(x, perm_vector).output(0)
    batch_norm = ov_opset.batch_norm_inference(
        x, scale, offset, mean, variance, epsilon
    ).output(0)
    if axis != 1:
        perm_vector = list(range(0, x_rank))
        perm_vector[1] = axis
        perm_vector[axis] = 1
        perm_vector = ov_opset.constant(perm_vector, Type.i32).output(0)
        batch_norm = ov_opset.transpose(batch_norm, perm_vector).output(0)
    return OpenVINOKerasTensor(batch_norm)


def ctc_loss(target, output, target_length, output_length, mask_index=0):
    target = get_ov_output(target)
    output = get_ov_output(output)
    target_length = get_ov_output(target_length)
    output_length = get_ov_output(output_length)
    ctc_loss_ = ov_opset.ctc_loss(
        output, output_length, target, target_length, blank_index=mask_index
    )
    ctc_loss_ = ov_opset.convert(ctc_loss_, OPENVINO_DTYPES[backend.floatx()])
    return OpenVINOKerasTensor(ctc_loss_.output(0))


def ctc_decode(
    inputs,
    sequence_lengths,
    strategy="greedy",
    beam_width=100,
    top_paths=1,
    merge_repeated=True,
    mask_index=0,
):
    raise NotImplementedError(
        "`ctc_decode` is not supported with openvino backend"
    )


def psnr(x1, x2, max_val):
    from keras.src.backend.openvino.numpy import log10

    x1 = get_ov_output(x1)
    x2 = get_ov_output(x2)
    max_val = get_ov_output(max_val, x1.get_element_type())
    diff = ov_opset.subtract(x1, x2)
    squared_diff = ov_opset.multiply(diff, diff)
    reduction_axes = list(range(0, x1.get_partial_shape().rank.get_length()))
    mse = ov_opset.reduce_mean(squared_diff, reduction_axes).output(0)
    log_max_val = get_ov_output(log10(OpenVINOKerasTensor(max_val)))
    log_mse = get_ov_output(log10(OpenVINOKerasTensor(mse)))

    psnr = ov_opset.subtract(
        ov_opset.multiply(
            ov_opset.constant(20, log_max_val.get_element_type()), log_max_val
        ),
        ov_opset.multiply(
            ov_opset.constant(10, log_mse.get_element_type()), log_mse
        ),
    ).output(0)
    return OpenVINOKerasTensor(psnr)


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
    if bias is not None:
        raise NotImplementedError(
            "`dot_product_attention` with `bias` is not supported "
            "with openvino backend"
        )
    if flash_attention:
        raise NotImplementedError(
            "`dot_product_attention` with `flash_attention` is not supported "
            "with openvino backend"
        )
    if attn_logits_soft_cap is not None:
        raise NotImplementedError(
            "`dot_product_attention` with `attn_logits_soft_cap` is not "
            "supported with openvino backend"
        )
    query = get_ov_output(query)
    key = get_ov_output(key)
    value = get_ov_output(value)
    if query.get_element_type() != key.get_element_type():
        ov_type = OPENVINO_DTYPES[backend.floatx()]
        query = ov_opset.convert(query, ov_type).output(0)
        key = ov_opset.convert(key, ov_type).output(0)
    if value.get_element_type() != query.get_element_type():
        value = ov_opset.convert(value, query.get_element_type()).output(0)
    axes_const = ov_opset.constant([0, 2, 1, 3], Type.i32).output(0)

    query = ov_opset.transpose(query, axes_const)
    key = ov_opset.transpose(key, axes_const)
    value = ov_opset.transpose(value, axes_const)
    mask = get_ov_output(mask) if mask is not None else None
    scale = (
        get_ov_output(scale, query.get_element_type())
        if scale is not None
        else None
    )
    dpa = ov_opset.scaled_dot_product_attention(
        query, key, value, attention_mask=mask, scale=scale, causal=is_causal
    )
    dpa = ov_opset.transpose(dpa, axes_const)
    return OpenVINOKerasTensor(dpa.output(0))


def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    raise NotImplementedError("`unfold` is not supported with openvino backend")


def depth_to_space(x, block_size, data_format="channels_last"):
    """OpenVINO implementation of depth_to_space (pixel shuffle).

    Rearranges data from depth into blocks of spatial data.

    Args:
        x: 4-D tensor with shape (N, H, W, C) for channels_last or
            (N, C, H, W) for channels_first.
        block_size: An integer specifying the block size.
        data_format: "channels_last" or "channels_first".

    Returns:
        A tensor with shape (N, H*block_size, W*block_size, C/block_size**2)
        for channels_last or (N, C/block_size**2, H*block_size, W*block_size)
        for channels_first.
    """
    x = get_ov_output(x)
    # OpenVINO depth_to_space uses "blocks_first" mode by default
    # and expects NCHW format
    if data_format == "channels_last":
        # Convert NHWC to NCHW
        axes = ov_opset.constant([0, 3, 1, 2], Type.i32).output(0)
        x = ov_opset.transpose(x, axes).output(0)
        result = ov_opset.depth_to_space(x, "blocks_first", block_size).output(
            0
        )
        # Convert back to NHWC
        axes_back = ov_opset.constant([0, 2, 3, 1], Type.i32).output(0)
        result = ov_opset.transpose(result, axes_back).output(0)
    else:
        result = ov_opset.depth_to_space(x, "blocks_first", block_size).output(
            0
        )
    return OpenVINOKerasTensor(result)


def space_to_depth(x, block_size, data_format="channels_last"):
    """OpenVINO implementation of space_to_depth (pixel unshuffle).

    Rearranges blocks of spatial data into depth.

    Args:
        x: 4-D tensor with shape (N, H, W, C) for channels_last or
            (N, C, H, W) for channels_first.
        block_size: An integer specifying the block size.
        data_format: "channels_last" or "channels_first".

    Returns:
        A tensor with shape (N, H/block_size, W/block_size, C*block_size**2)
        for channels_last or (N, C*block_size**2, H/block_size, W/block_size)
        for channels_first.
    """
    x = get_ov_output(x)
    # OpenVINO space_to_depth uses "blocks_first" mode by default
    # and expects NCHW format
    if data_format == "channels_last":
        # Convert NHWC to NCHW
        axes = ov_opset.constant([0, 3, 1, 2], Type.i32).output(0)
        x = ov_opset.transpose(x, axes).output(0)
        result = ov_opset.space_to_depth(x, "blocks_first", block_size).output(
            0
        )
        # Convert back to NHWC
        axes_back = ov_opset.constant([0, 2, 3, 1], Type.i32).output(0)
        result = ov_opset.transpose(result, axes_back).output(0)
    else:
        result = ov_opset.space_to_depth(x, "blocks_first", block_size).output(
            0
        )
    return OpenVINOKerasTensor(result)
