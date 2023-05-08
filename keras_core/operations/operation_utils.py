import numpy as np


def compute_pooling_output_shape(
    input_shape,
    pool_size,
    strides,
    padding="valid",
    data_format="channels_last",
):
    """Compute the output shape of pooling ops."""
    strides = pool_size if strides is None else strides
    input_shape_origin = list(input_shape)
    input_shape = np.array(input_shape)
    if data_format == "channels_last":
        spatial_shape = input_shape[1:-1]
    else:
        spatial_shape = input_shape[2:]
    pool_size = np.array(pool_size)
    if padding == "valid":
        output_spatial_shape = (
            np.floor((spatial_shape - pool_size) / strides) + 1
        )
        negative_in_shape = np.all(output_spatial_shape < 0)
        if negative_in_shape:
            raise ValueError(
                "Computed output size would be negative. Received: "
                f"`inputs.shape={input_shape}` and `pool_size={pool_size}`."
            )
    elif padding == "same":
        output_spatial_shape = np.floor((spatial_shape - 1) / strides) + 1
    else:
        raise ValueError(
            "`padding` must be either `'valid'` or `'same'`. Received "
            f"{padding}."
        )
    output_spatial_shape = tuple([int(i) for i in output_spatial_shape])
    if data_format == "channels_last":
        output_shape = (
            (input_shape_origin[0],)
            + output_spatial_shape
            + (input_shape_origin[-1],)
        )
    else:
        output_shape = (
            input_shape_origin[0],
            input_shape_origin[1],
        ) + output_spatial_shape
    return output_shape


def compute_conv_output_shape(
    input_shape,
    filters,
    kernel_size,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
):
    """Compute the output shape of conv ops."""
    if data_format == "channels_last":
        spatial_shape = input_shape[1:-1]
        kernel_shape = kernel_size + (input_shape[-1], filters)
    else:
        spatial_shape = input_shape[2:]
        kernel_shape = kernel_size + (input_shape[1], filters)
    if len(kernel_shape) != len(input_shape):
        raise ValueError(
            "Kernel shape must have the same length as input, but received "
            f"kernel of shape {kernel_shape} and "
            f"input of shape {input_shape}."
        )
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * len(spatial_shape)
    if isinstance(strides, int):
        strides = (strides,) * len(spatial_shape)
    if len(dilation_rate) != len(spatial_shape):
        raise ValueError(
            "Dilation must be None, scalar or tuple/list of length of "
            "inputs' spatial shape, but received "
            f"`dilation_rate={dilation_rate}` and "
            f"input of shape {input_shape}."
        )
    spatial_shape = np.array(spatial_shape)
    kernel_spatial_shape = np.array(kernel_shape[:-2])
    dilation_rate = np.array(dilation_rate)
    if padding == "valid":
        output_spatial_shape = (
            np.floor(
                (spatial_shape - dilation_rate * (kernel_spatial_shape - 1) - 1)
                / strides
            )
            + 1
        )
        negative_in_shape = np.all(output_spatial_shape < 0)
        if negative_in_shape:
            raise ValueError(
                "Computed output size would be negative. Received "
                f"`inputs shape={input_shape}`, "
                f"`kernel shape={kernel_shape}`, "
                f"`dilation_rate={dilation_rate}`."
            )
    elif padding == "same":
        output_spatial_shape = np.floor((spatial_shape - 1) / strides) + 1
    output_spatial_shape = tuple([int(i) for i in output_spatial_shape])
    if data_format == "channels_last":
        output_shape = (
            (input_shape[0],) + output_spatial_shape + (kernel_shape[-1],)
        )
    else:
        output_shape = (input_shape[0], kernel_shape[-1]) + output_spatial_shape
    return output_shape
