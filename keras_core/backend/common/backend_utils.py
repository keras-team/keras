import warnings


def _convert_conv_tranpose_padding_args_from_keras_to_jax(
    kernel_size, stride, dilation_rate, padding, output_padding
):
    """Convert the padding arguments from Keras to the ones used by JAX.
    JAX starts with an shape of size `(input-1) * stride - kernel_size + 2`,
    then adds `left_pad` on the left, and `right_pad` on the right.
    In Keras, the `padding` argument determines a base shape, to which
    `output_padding` is added on the right. If `output_padding` is None, it will
    be given a default value.
    """

    assert padding.lower() in {"valid", "same"}
    kernel_size = (kernel_size - 1) * dilation_rate + 1

    if padding.lower() == "valid":
        # If output_padding is None, we fill it so that the shape of the ouput
        # is `(input-1)*s + max(kernel_size, stride)`
        output_padding = (
            max(kernel_size, stride) - kernel_size
            if output_padding is None
            else output_padding
        )
        left_pad = kernel_size - 1
        right_pad = kernel_size - 1 + output_padding

    else:
        if output_padding is None:
            # When output_padding is None, we want the shape of the ouput to
            # be `input * s`, therefore a total padding of
            # `stride + kernel_size - 2`
            pad_len = stride + kernel_size - 2
        else:
            # When output_padding is filled, we want the shape of the ouput to
            # be `(input-1)*stride + kernel_size%2 + output_padding`
            pad_len = kernel_size + kernel_size % 2 - 2 + output_padding
        left_pad = min(pad_len // 2 + pad_len % 2, kernel_size - 1)
        right_pad = pad_len - left_pad

    return left_pad, right_pad


def _convert_conv_tranpose_padding_args_from_keras_to_torch(
    kernel_size, stride, dilation_rate, padding, output_padding
):
    """Convert the padding arguments from Keras to the ones used by Torch.
    Torch starts with an output shape of `(input-1) * stride + kernel_size`,
    then removes `torch_padding` from both sides, and adds
    `torch_output_padding` on the right.
    Because in Torch the output_padding can only be added to the right,
    consistency with Tensorflow is not always possible. In particular this is
    the case when both the Torch padding and output_padding values are stricly
    positive.
    """
    assert padding.lower() in {"valid", "same"}
    original_kernel_size = kernel_size
    kernel_size = (kernel_size - 1) * dilation_rate + 1

    if padding.lower() == "valid":
        # If output_padding is None, we fill it so that the shape of the ouput
        # is `(i-1)*s + max(k, s)`
        output_padding = (
            max(kernel_size, stride) - kernel_size
            if output_padding is None
            else output_padding
        )
        torch_padding = 0
        torch_output_padding = output_padding

    else:
        # When output_padding is None, we want the shape of the ouput to be
        # `input * s`, otherwise we use the value provided.
        output_padding = (
            stride - kernel_size % 2
            if output_padding is None
            else output_padding
        )
        torch_padding = max(
            -((kernel_size % 2 - kernel_size + output_padding) // 2), 0
        )
        torch_output_padding = (
            2 * torch_padding + kernel_size % 2 - kernel_size + output_padding
        )

    if torch_padding > 0 and torch_output_padding > 0:
        warnings.warn(
            f"You might experience inconsistencies accross backends when "
            f"calling conv transpose with kernel_size={original_kernel_size}, "
            f"stride={stride}, dilation_rate={dilation_rate}, "
            f"padding={padding}, output_padding={output_padding}."
        )

    if torch_output_padding >= stride:
        raise ValueError(
            f"The padding arguments (padding={padding}) and "
            f"output_padding={output_padding}) lead to a Torch "
            f"output_padding ({torch_output_padding}) that is greater than "
            f"strides ({stride}). This is not supported. You can change the "
            f"padding arguments, kernel or stride, or run on another backend. "
        )

    return torch_padding, torch_output_padding


def compute_conv_transpose_padding_args_for_jax(
    input_shape,
    kernel_shape,
    strides,
    padding,
    output_padding,
    dilation_rate,
):
    num_spatial_dims = len(input_shape) - 2
    kernel_spatial_shape = kernel_shape[:-2]

    jax_padding = []
    for i in range(num_spatial_dims):
        output_padding_i = (
            output_padding
            if output_padding is None or isinstance(output_padding, int)
            else output_padding[i]
        )
        strides_i = strides if isinstance(strides, int) else strides[i]
        dilation_rate_i = (
            dilation_rate
            if isinstance(dilation_rate, int)
            else dilation_rate[i]
        )
        (
            pad_left,
            pad_right,
        ) = _convert_conv_tranpose_padding_args_from_keras_to_jax(
            kernel_size=kernel_spatial_shape[i],
            stride=strides_i,
            dilation_rate=dilation_rate_i,
            padding=padding,
            output_padding=output_padding_i,
        )
        jax_padding.append((pad_left, pad_right))

    return jax_padding


def compute_conv_transpose_padding_args_for_torch(
    input_shape,
    kernel_shape,
    strides,
    padding,
    output_padding,
    dilation_rate,
):
    num_spatial_dims = len(input_shape) - 2
    kernel_spatial_shape = kernel_shape[:-2]

    torch_paddings = []
    torch_output_paddings = []
    for i in range(num_spatial_dims):
        output_padding_i = (
            output_padding
            if output_padding is None or isinstance(output_padding, int)
            else output_padding[i]
        )
        strides_i = strides if isinstance(strides, int) else strides[i]
        dilation_rate_i = (
            dilation_rate
            if isinstance(dilation_rate, int)
            else dilation_rate[i]
        )
        (
            torch_padding,
            torch_output_padding,
        ) = _convert_conv_tranpose_padding_args_from_keras_to_torch(
            kernel_size=kernel_spatial_shape[i],
            stride=strides_i,
            dilation_rate=dilation_rate_i,
            padding=padding,
            output_padding=output_padding_i,
        )
        torch_paddings.append(torch_padding)
        torch_output_paddings.append(torch_output_padding)

    return torch_paddings, torch_output_paddings


def _get_output_shape_given_tf_padding(
    input_size, kernel_size, strides, padding, output_padding, dilation_rate
):
    if input_size is None:
        return None

    assert padding.lower() in {"valid", "same"}

    kernel_size = (kernel_size - 1) * dilation_rate + 1

    if padding.lower() == "valid":
        output_padding = (
            max(kernel_size, strides) - kernel_size
            if output_padding is None
            else output_padding
        )
        return (input_size - 1) * strides + kernel_size + output_padding

    else:
        if output_padding is None:
            return input_size * strides
        else:
            return (input_size - 1) * strides + kernel_size % 2 + output_padding


def compute_conv_transpose_output_shape(
    input_shape,
    kernel_size,
    filters,
    strides,
    padding,
    output_padding=None,
    data_format="channels_last",
    dilation_rate=1,
):
    num_spatial_dims = len(input_shape) - 2
    kernel_spatial_shape = kernel_size

    if isinstance(output_padding, int):
        output_padding = (output_padding,) * len(kernel_spatial_shape)
    if isinstance(strides, int):
        strides = (strides,) * num_spatial_dims
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * num_spatial_dims

    if data_format == "channels_last":
        input_spatial_shape = input_shape[1:-1]
    else:
        input_spatial_shape = input_shape[2:]

    output_shape = []
    for i in range(num_spatial_dims):
        current_output_padding = (
            None if output_padding is None else output_padding[i]
        )

        shape_i = _get_output_shape_given_tf_padding(
            input_size=input_spatial_shape[i],
            kernel_size=kernel_spatial_shape[i],
            strides=strides[i],
            padding=padding,
            output_padding=current_output_padding,
            dilation_rate=dilation_rate[i],
        )
        output_shape.append(shape_i)

    if data_format == "channels_last":
        output_shape = [input_shape[0]] + output_shape + [filters]
    else:
        output_shape = [input_shape[0], filters] + output_shape
    return output_shape
