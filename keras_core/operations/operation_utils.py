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
