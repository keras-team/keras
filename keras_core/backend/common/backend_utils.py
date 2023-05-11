def _compute_conv_transpose_output_length(
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
    inputs,
    kernel,
    strides,
    padding,
    output_padding=None,
    data_format="channels_last",
    dilation_rate=1,
):
    num_spatial_dims = len(inputs.shape) - 2
    kernel_spatial_shape = kernel.shape[:-2]

    if isinstance(output_padding, int):
        output_padding = (output_padding,) * len(kernel_spatial_shape)
    if isinstance(strides, int):
        strides = (strides,) * num_spatial_dims
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * num_spatial_dims

    if data_format == "channels_last":
        inputs_spatial_shape = inputs.shape[1:-1]
    else:
        inputs_spatial_shape = inputs.shape[2:]

    output_shape = []
    for i in range(num_spatial_dims):
        current_output_padding = (
            None if output_padding is None else output_padding[i]
        )
        output_shape.append(
            _compute_conv_transpose_output_length(
                inputs_spatial_shape[i],
                kernel_spatial_shape[i],
                padding=padding,
                output_padding=current_output_padding,
                stride=strides[i],
                dilation=dilation_rate[0],
            )
        )

    if data_format == "channels_last":
        output_shape = [inputs.shape[0]] + output_shape + [kernel.shape[-2]]
    else:
        output_shape = [inputs.shape[0], kernel.shape[-1]] + output_shape
    return output_shape
