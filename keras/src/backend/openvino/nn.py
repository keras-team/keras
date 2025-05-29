import openvino.runtime.opset14 as ov_opset
from openvino import Type

from keras.src import backend
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import get_ov_output


def relu(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.relu(x).output(0))


def relu6(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.clamp(x, 0.0, 6.0).output(0))


def sigmoid(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.sigmoid(x).output(0))


def tanh(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.tanh(x).output(0))


def softplus(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.softplus(x).output(0))


def softsign(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.softsign(x).output(0))


def silu(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(
        ov_opset.multiply(x, ov_opset.sigmoid(x)).output(0)
    )


def log_sigmoid(x):
    raise NotImplementedError(
        "`log_sigmoid` is not supported with openvino backend"
    )


def leaky_relu(x, negative_slope=0.2):
    x = get_ov_output(x)
    slope_const = ov_opset.constant(
        negative_slope, x.get_element_type()
    ).output(0)
    leaky_relu = ov_opset.prelu(x, slope_const).output(0)
    return OpenVINOKerasTensor(leaky_relu)


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


def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    raise NotImplementedError(
        "`max_pool` is not supported with openvino backend"
    )


def average_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    raise NotImplementedError(
        "`average_pool` is not supported with openvino backend"
    )


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
    raise NotImplementedError(
        "`one_hot` is not supported with openvino backend"
    )


def multi_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    raise NotImplementedError(
        "`multi_hot` is not supported with openvino backend"
    )


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    raise NotImplementedError(
        "`categorical_crossentropy` is not supported with openvino backend"
    )


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    raise NotImplementedError(
        "`sparse_categorical_crossentropy` is not supported "
        "with openvino backend"
    )


def binary_crossentropy(target, output, from_logits=False):
    raise NotImplementedError(
        "`binary_crossentropy` is not supported with openvino backend"
    )


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
    raise NotImplementedError(
        "`ctc_loss` is not supported with openvino backend"
    )


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
    raise NotImplementedError("`psnr` is not supported with openvino backend")
