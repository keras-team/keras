def compute_conv_transpose_output_length(
    input_length,
    kernel_size,
    padding,
    output_padding=None,
    stride=1,
    dilation=1,
):
    """Computes output size of a transposed convolution given input size."""
    assert padding in {"same", "valid"}
    if input_length is None:
        return None

    # Get the dilated kernel size
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)

    # Infer length if output padding is None, else compute the exact length
    if output_padding is None:
        if padding == "valid":
            length = input_length * stride + max(kernel_size - stride, 0)
        else:
            length = input_length * stride
    else:
        if padding == "same":
            pad = kernel_size // 2
        else:
            pad = 0

        length = (
            (input_length - 1) * stride + kernel_size - 2 * pad + output_padding
        )
    return length


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
        output_shape.append(
            compute_conv_transpose_output_length(
                input_spatial_shape[i],
                kernel_spatial_shape[i],
                padding=padding,
                output_padding=current_output_padding,
                stride=strides[i],
                dilation=dilation_rate[i],
            )
        )

    if data_format == "channels_last":
        output_shape = [input_shape[0]] + output_shape + [filters]
    else:
        output_shape = [input_shape[0], filters] + output_shape
    return output_shape


def _compute_conv_transpose_padding_one_dim(
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    dilation_rate,
):
    """Computes adjusted padding for `conv_transpose` in one dim."""
    kernel_size = (kernel_size - 1) * dilation_rate + 1
    if padding == "valid":
        padding_before = 0
    else:
        # padding == "same".
        padding_needed = max(
            0, (input_length - 1) * stride + kernel_size - output_length
        )
        padding_before = padding_needed // 2

    expanded_input_length = (input_length - 1) * stride + 1
    padded_out_length = output_length + kernel_size - 1
    pad_before = kernel_size - 1 - padding_before
    pad_after = padded_out_length - expanded_input_length - pad_before
    return (pad_before, pad_after)


def compute_conv_transpose_padding(
    input_shape,
    kernel_shape,
    strides=1,
    padding="valid",
    output_padding=None,
    data_format="channels_last",
    dilation_rate=1,
):
    """Computes adjusted padding for `conv_transpose`."""
    num_spatial_dims = len(input_shape) - 2
    if isinstance(output_padding, int):
        output_padding = (output_padding,) * num_spatial_dims
    if isinstance(strides, int):
        strides = (strides,) * num_spatial_dims
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * num_spatial_dims

    kernel_spatial_shape = kernel_shape[:-2]
    if data_format == "channels_last":
        input_spatial_shape = input_shape[1:-1]
    else:
        input_spatial_shape = input_shape[2:]
    padding_values = []
    for i in range(num_spatial_dims):
        input_length = input_spatial_shape[i]
        current_output_padding = (
            None if output_padding is None else output_padding[i]
        )
        output_length = compute_conv_transpose_output_length(
            input_spatial_shape[i],
            kernel_spatial_shape[i],
            padding=padding,
            output_padding=current_output_padding,
            stride=strides[i],
            dilation=dilation_rate[i],
        )
        padding_value = _compute_conv_transpose_padding_one_dim(
            input_length,
            output_length,
            kernel_spatial_shape[i],
            strides[i],
            padding=padding,
            dilation_rate=dilation_rate[i],
        )
        padding_values.append(padding_value)
    return padding_values
