import math

import numpy as np

from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.backend.common.backend_utils import to_tuple_or_list


def broadcast_shapes(shape1, shape2):
    """Broadcast input shapes to a unified shape.

    Convert to list for mutability.

    Args:
        shape1: A tuple or list of integers.
        shape2: A tuple or list of integers.

    Returns:
        output_shape (list of integers or `None`): The broadcasted shape.

    Example:
    >>> broadcast_shapes((5, 3), (1, 3))
    [5, 3]
    """
    shape1 = list(shape1)
    shape2 = list(shape2)
    origin_shape1 = shape1
    origin_shape2 = shape2

    if len(shape1) > len(shape2):
        shape2 = [1] * (len(shape1) - len(shape2)) + shape2
    if len(shape1) < len(shape2):
        shape1 = [1] * (len(shape2) - len(shape1)) + shape1
    output_shape = list(shape1)
    for i in range(len(shape1)):
        if shape1[i] == 1:
            output_shape[i] = shape2[i]
        elif shape1[i] is None:
            output_shape[i] = None if shape2[i] == 1 else shape2[i]
        else:
            if shape2[i] == 1 or shape2[i] is None or shape2[i] == shape1[i]:
                output_shape[i] = shape1[i]
            else:
                raise ValueError(
                    "Cannot broadcast shape, the failure dim has value "
                    f"{shape1[i]}, which cannot be broadcasted to {shape2[i]}. "
                    f"Input shapes are: {origin_shape1} and {origin_shape2}."
                )

    return output_shape


def compute_expand_dims_output_shape(input_shape, axis):
    """Compute the output shape for the `expand_dims` operation.

    Args:
        input_shape: Input shape.
        axis: int or sequence of ints for the axis to expand.

    Returns:
        Tuple of ints: The output shape after the `expand_dims` operation.
    """
    input_shape = list(input_shape)
    if axis is None:
        axis = len(input_shape)
    axis = to_tuple_or_list(axis)
    out_ndim = len(axis) + len(input_shape)
    axis = [canonicalize_axis(a, out_ndim) for a in axis]
    shape_iter = iter(input_shape)
    new_shape = [
        1 if ax in axis else next(shape_iter) for ax in range(out_ndim)
    ]
    return tuple(new_shape)


def compute_pooling_output_shape(
    input_shape,
    pool_size,
    strides,
    padding="valid",
    data_format="channels_last",
):
    """Computes the output shape of pooling operations.

    Args:
        input_shape: Input shape. Must be a tuple of integers.
        pool_size: Size of the pooling operation. Must be a tuple of integers.
        strides: Stride of the pooling operation. Must be a tuple of integers.
            Defaults to `pool_size`.
        padding: Padding method. Available methods are `"valid"` or `"same"`.
            Defaults to `"valid"`.
        data_format: String, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, weight)`. Defaults to `"channels_last"`.

    Returns:
        Tuple of ints: The output shape of the pooling operation.

    Examples:

    # Basic usage with square pooling on a single image
    >>> compute_pooling_output_shape((1, 4, 4, 1), (2, 2))
    (1, 2, 2, 1)

    # Strided pooling on a single image with strides different from pool_size
    >>> compute_pooling_output_shape((1, 4, 4, 1), (2, 2), strides=(1, 1))
    (1, 3, 3, 1)

    # Pooling on a batch of images
    >>> compute_pooling_output_shape((32, 4, 4, 3), (2, 2))
    (32, 2, 2, 3)
    """
    strides = pool_size if strides is None else strides
    input_shape_origin = list(input_shape)
    input_shape = np.array(input_shape)
    if data_format == "channels_last":
        spatial_shape = input_shape[1:-1]
    else:
        spatial_shape = input_shape[2:]
    none_dims = []
    for i in range(len(spatial_shape)):
        if spatial_shape[i] is None:
            # Set `None` shape to a manual value so that we can run numpy
            # computation on `spatial_shape`.
            spatial_shape[i] = -1
            none_dims.append(i)
    pool_size = np.array(pool_size)
    if padding == "valid":
        output_spatial_shape = (
            np.floor((spatial_shape - pool_size) / strides) + 1
        )
        for i in range(len(output_spatial_shape)):
            if i not in none_dims and output_spatial_shape[i] < 0:
                raise ValueError(
                    "Computed output size would be negative. Received: "
                    f"`inputs.shape={input_shape}` and `pool_size={pool_size}`."
                )
    elif padding == "same":
        output_spatial_shape = np.floor((spatial_shape - 1) / strides) + 1
    else:
        raise ValueError(
            "Argument `padding` must be either 'valid' or 'same'. Received: "
            f"padding={padding}"
        )
    output_spatial_shape = [int(i) for i in output_spatial_shape]
    for i in none_dims:
        output_spatial_shape[i] = None
    output_spatial_shape = tuple(output_spatial_shape)
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
    none_dims = []
    spatial_shape = np.array(spatial_shape)
    for i in range(len(spatial_shape)):
        if spatial_shape[i] is None:
            # Set `None` shape to a manual value so that we can run numpy
            # computation on `spatial_shape`.
            spatial_shape[i] = -1
            none_dims.append(i)

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
        for i in range(len(output_spatial_shape)):
            if i not in none_dims and output_spatial_shape[i] < 0:
                raise ValueError(
                    "Computed output size would be negative. Received "
                    f"`inputs shape={input_shape}`, "
                    f"`kernel shape={kernel_shape}`, "
                    f"`dilation_rate={dilation_rate}`."
                )
    elif padding == "same" or padding == "causal":
        output_spatial_shape = np.floor((spatial_shape - 1) / strides) + 1
    else:
        raise ValueError(
            "`padding` must be either `'valid'` or `'same'`. Received "
            f"{padding}."
        )
    output_spatial_shape = [int(i) for i in output_spatial_shape]
    for i in none_dims:
        output_spatial_shape[i] = None
    output_spatial_shape = tuple(output_spatial_shape)
    if data_format == "channels_last":
        output_shape = (
            (input_shape[0],) + output_spatial_shape + (kernel_shape[-1],)
        )
    else:
        output_shape = (input_shape[0], kernel_shape[-1]) + output_spatial_shape
    return output_shape


def compute_matmul_output_shape(shape1, shape2):
    """Compute the output shape of a `matmul` operation.

    Args:
        shape1: Shape of the left operand.
        shape2: Shape of the right operand.

    Returns:
        Tuple of ints: The output shape for the `matmul` operation.
    """
    if len(shape1) == 1:
        shape1 = (1, shape1[0])
    if len(shape2) == 1:
        shape2 = (shape2[0], 1)
    if (
        shape1[-1] is not None
        and shape2[-2] is not None
        and shape1[-1] != shape2[-2]
    ):
        raise ValueError(
            "Inner dimensions (`x1.shape[-1]` and `x2.shape[-2]`) must be "
            f"equal, but received `x1.shape={shape1}` and "
            f"`x2.shape={shape2}`."
        )

    leading_shape = broadcast_shapes(shape1[:-2], shape2[:-2])
    last_2_dims_shape = [shape1[-2], shape2[-1]]
    output_shape = leading_shape + last_2_dims_shape
    if len(shape1) == 1:
        del output_shape[-2]
    if len(shape2) == 1:
        del output_shape[-1]
    return tuple(output_shape)


def compute_reshape_output_shape(input_shape, newshape, newshape_arg_name):
    """Converts `-1` in `newshape` to either an actual dimension or `None`.

    This utility does not special case the 0th dimension (batch size).
    """
    unknown_dim_count = newshape.count(-1)
    if unknown_dim_count > 1:
        raise ValueError(
            "There must be at most one unknown dimension (-1) in "
            f"{newshape_arg_name}. Received: {newshape_arg_name}={newshape}."
        )

    # If there is a None in input_shape, we can't infer what the -1 is
    if None in input_shape:
        return tuple(dim if dim != -1 else None for dim in newshape)

    input_size = math.prod(input_shape)
    # If the `newshape` is fully defined, return it
    if unknown_dim_count == 0:
        if input_size != math.prod(newshape):
            raise ValueError(
                "The total size of the tensor must be unchanged. Received: "
                f"input_shape={input_shape}, {newshape_arg_name}={newshape}"
            )
        return newshape

    # We have one -1 in `newshape`, compute the actual value
    known_output_size = 1
    unknown_dim_index = None
    for index, dim in enumerate(newshape):
        if dim == -1:
            unknown_dim_index = index
        else:
            known_output_size *= dim

    if known_output_size == 0 or input_size % known_output_size != 0:
        raise ValueError(
            "The total size of the tensor must be unchanged, however, the "
            "input size cannot by divided by the specified dimensions in "
            f"{newshape_arg_name}. Received: input_shape={input_shape}, "
            f"{newshape_arg_name}={newshape}"
        )

    output_shape = list(newshape)
    output_shape[unknown_dim_index] = input_size // known_output_size
    return tuple(output_shape)


def compute_transpose_output_shape(input_shape, axes):
    """Compute the output shape for the `transpose` operation.

    Args:
        input_shape: Input shape.
        axes: Permutation of the dimensions for the `transpose` operation.

    Returns:
        Tuple of ints: The output shape after the `transpose` operation.
    """
    input_shape = list(input_shape)
    if axes is None:
        return tuple(input_shape[::-1])

    if len(axes) != len(input_shape):
        raise ValueError(
            "axis must be a list of the same length as the input shape, "
            f"expected {len(input_shape)}, but received {len(axes)}."
        )
    return tuple(input_shape[ax] for ax in axes)


def reduce_shape(shape, axis=None, keepdims=False):
    shape = list(shape)
    if axis is None:
        if keepdims:
            return tuple([1 for _ in shape])
        else:
            return tuple([])

    if keepdims:
        for ax in axis:
            shape[ax] = 1
        return tuple(shape)
    else:
        for ax in sorted(axis, reverse=True):
            del shape[ax]
        return tuple(shape)


@keras_export("keras.utils.get_source_inputs")
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
            return tree.flatten(node.output_tensors)
        else:
            source_tensors = []
            for tensor in node.input_tensors:
                previous_sources = get_source_inputs(tensor)
                # Avoid input redundancy.
                for x in previous_sources:
                    if all(x is not t for t in source_tensors):
                        source_tensors.append(x)
            return source_tensors
