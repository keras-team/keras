import math

import numpy as np
from tensorflow import nest


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
    elif padding == "same" or padding == "causal":
        output_spatial_shape = np.floor((spatial_shape - 1) / strides) + 1
    output_spatial_shape = tuple([int(i) for i in output_spatial_shape])
    if data_format == "channels_last":
        output_shape = (
            (input_shape[0],) + output_spatial_shape + (kernel_shape[-1],)
        )
    else:
        output_shape = (input_shape[0], kernel_shape[-1]) + output_spatial_shape
    return output_shape


def compute_reshape_output_shape(input_shape, new_shape, new_shape_arg_name):
    """Converts `-1` in `new_shape` to either an actual dimension or `None`.

    This utility does not special case the 0th dimension (batch size).
    """
    unknown_dim_count = new_shape.count(-1)
    if unknown_dim_count > 1:
        raise ValueError(
            "There must be at most one unknown dimension (-1) in "
            f"{new_shape_arg_name}. Received: {new_shape_arg_name}={new_shape}."
        )

    # If there is a None in input_shape, we can't infer what the -1 is
    if None in input_shape:
        return tuple(dim if dim != -1 else None for dim in new_shape)

    input_size = math.prod(input_shape)
    # If the new_shape fully defined, return it
    if unknown_dim_count == 0:
        if input_size != math.prod(new_shape):
            raise ValueError(
                "The total size of the tensor must be unchanged. Received: "
                f"input_shape={input_shape}, {new_shape_arg_name}={new_shape}"
            )
        return new_shape

    # We have one -1 in new_shape, compute the actual value
    known_output_size = 1
    unknown_dim_index = None
    for index, dim in enumerate(new_shape):
        if dim == -1:
            unknown_dim_index = index
        else:
            known_output_size *= dim

    if known_output_size == 0 or input_size % known_output_size != 0:
        raise ValueError(
            "The total size of the tensor must be unchanged, however, the "
            "input size cannot by divided by the specified dimensions in "
            f"{new_shape_arg_name}. Received: input_shape={input_shape}, "
            f"{new_shape_arg_name}={new_shape}"
        )

    output_shape = list(new_shape)
    output_shape[unknown_dim_index] = input_size // known_output_size
    return tuple(output_shape)


def reduce_shape(shape, axis=None, keepdims=False):
    shape = list(shape)
    if axis is None:
        if keepdims:
            output_shape = [1 for _ in range(shape)]
        else:
            output_shape = []
        return output_shape

    if keepdims:
        for ax in axis:
            shape[ax] = 1
        return shape
    else:
        for ax in axis:
            shape[ax] = -1
        output_shape = list(filter((-1).__ne__, shape))
        return output_shape


def get_source_inputs(tensor):
    """Returns the list of input tensors necessary to compute `tensor`.

    Output will always be a list of tensors
    (potentially with 1 element).

    Args:
        tensor: The tensor to start from.

    Returns:
        List of input tensors.
    """
    if not hasattr(tensor, "_keras_history"):
        return tensor

    operation, node_index, _ = tensor._keras_history
    if not operation or not operation._inbound_nodes:
        return [tensor]
    else:
        node = operation._inbound_nodes[node_index]
        if node.is_input:
            # Reached input node, stop recursion.
            return nest.flatten(node.input_tensors)
        else:
            source_tensors = []
            for tensor in node.input_tensors:
                previous_sources = get_source_inputs(tensor)
                # Avoid input redundancy.
                for x in previous_sources:
                    if all(x is not t for t in source_tensors):
                        source_tensors.append(x)
            return source_tensors
