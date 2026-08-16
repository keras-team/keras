import re

from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.core import shape
from keras.src.ops.numpy import prod
from keras.src.ops.numpy import reshape
from keras.src.ops.numpy import transpose
from keras.src.ops.operation import Operation


def _create_axes_map(axes, input_shape, axes_lengths):
    axes_map = {}

    for axis, dim in zip(axes, input_shape):
        # Check for grouped axes pattern, e.g., "(h1 h)"
        grouped_axes = re.match(r"\(([\w\s]+)\)", axis)

        if grouped_axes:
            inner_axes = grouped_axes.group(1).split()
            known_axes = [a for a in inner_axes if a in axes_lengths]
            inferred_axes = [a for a in inner_axes if a not in axes_lengths]

            if inferred_axes:
                inferred_axis = inferred_axes[0]
                known_product = prod([axes_lengths[a] for a in known_axes])
                axes_lengths[inferred_axis] = dim // known_product

            axes_map.update({a: axes_lengths[a] for a in inner_axes})
        else:
            axes_map[axis] = dim

    return axes_map


def _create_grouped_axes(axes):
    grouped_output_axes = []
    for axis in axes:
        grouped_axes = re.match(r"\(([\w\s]+)\)", axis)

        if grouped_axes:
            inner_axes = grouped_axes.group(1).split()
            grouped_output_axes.append(inner_axes)
        else:
            grouped_output_axes.append([axis])

    return grouped_output_axes


def _flatten_group(axes):
    return [x for xs in axes for x in xs]


def _get_transpose_order(from_shape, to_shape):
    flattened_from_shape = _flatten_group(_create_grouped_axes(from_shape))

    return [flattened_from_shape.index(dim) for dim in to_shape]


def _compute_output_shape(axes_map, grouped_axes):
    output_shape = []
    for group in grouped_axes:
        size = 1
        for axis in group:
            size *= axes_map[axis]
        output_shape.append(size)

    return tuple(output_shape)


def _compute_decomposed_shape(input_axes, axes_lengths, axes_map):
    reshaped_input_axes = []
    reshaped_sizes = []

    for axis in input_axes:
        if "(" in axis:  # Decomposed axis
            inner_axes = re.findall(r"\w+", axis)
            sizes = [axes_lengths[a] for a in inner_axes]
            reshaped_input_axes.extend(inner_axes)
            reshaped_sizes.extend(sizes)
        else:
            reshaped_input_axes.append(axis)
            reshaped_sizes.append(axes_map[axis])

    return reshaped_sizes


class Rearrange(Operation):
    def call(self, tensor, pattern, **axes_lengths):
        return rearrange(tensor, pattern, **axes_lengths)

    def compute_output_spec(self, tensor, pattern, **axes_lengths):
        input_pattern, output_pattern = re.split(r"\s*->\s*", pattern)
        input_axes = re.findall(r"\w+|\(.*?\)", input_pattern)
        output_axes = re.findall(r"\w+|\(.*?\)", output_pattern)
        input_shape = shape(tensor)

        axes_map = _create_axes_map(input_axes, input_shape, axes_lengths)
        grouped_output_axes = _create_grouped_axes(output_axes)
        output_shape = _compute_output_shape(axes_map, grouped_output_axes)

        return KerasTensor(shape=output_shape, dtype=tensor.dtype)


@keras_export("keras.ops.rearrange")
def rearrange(tensor, pattern, **axes_lengths):
    """Rearranges the axes of a Keras tensor according to a specified pattern,
    einops-style.

    Args:
        tensor: Input Keras tensor.
        pattern: String describing the rearrangement in einops notation.
        **axes_lengths: Keyword arguments specifying lengths of axes
            when axes decomposition is used.

    Returns:
        Tensor: A Keras tensor with rearranged axes.

    Follows the logic of:

    1. If decomposition is needed, reshape to match decomposed dimensions.
    2. Permute known and inferred axes to match the form of the output.
    3. Reshape to match the desired output shape.


    Example Usage:

    ```
    >>> import numpy as np
    >>> from keras.ops import rearrange
    >>> images = np.random.rand(32, 30, 40, 3) # BHWC format

    # Reordering to BCHW
    >>> rearrange(images, 'b h w c -> b c h w').shape
    TensorShape([32, 3, 30, 40])

    # "Merge" along first axis - concat images from a batch
    >>> rearrange(images, 'b h w c -> (b h) w c').shape
    TensorShape([960, 40, 3])

    # "Merge" along second axis - concat images horizontally
    >>> rearrange(images, 'b h w c -> h (b w) c').shape
    TensorShape([30, 1280, 3])

    # Flatten images into a CHW vector
    >>> rearrange(images, 'b h w c -> b (c h w)').shape
    TensorShape([32, 3600])

    # Decompose H and W axes into 4 smaller patches
    >>> rearrange(images, 'b (h1 h) (w1 w) c -> (b h1 w1) h w c', h1=2, w1=2).shape
    TensorShape([128, 15, 20, 3])

    # Space-to-depth decomposition of input axes
    >>> rearrange(images, 'b (h h1) (w w1) c -> b h w (c h1 w1)', h1=2, w1=2).shape
    TensorShape([32, 15, 20, 12])
    ```
    """  # noqa: E501

    if any_symbolic_tensors((tensor,)):
        return Rearrange().symbolic_call(tensor, pattern, **axes_lengths)

    # Split the input and output patterns
    input_pattern, output_pattern = re.split(r"\s*->\s*", pattern)
    input_axes = re.findall(r"\w+|\(.*?\)", input_pattern)
    output_axes = re.findall(r"\w+|\(.*?\)", output_pattern)
    input_shape = shape(tensor)

    # Create axes map, and flattened output group
    axes_map = _create_axes_map(input_axes, input_shape, axes_lengths)
    grouped_output_axes = _create_grouped_axes(output_axes)
    flattened_output_axes = _flatten_group(grouped_output_axes)

    # 1. Axes decomposition
    decomposed_shapes = _compute_decomposed_shape(
        input_axes, axes_lengths, axes_map
    )
    if decomposed_shapes != tensor.shape:
        tensor = reshape(tensor, decomposed_shapes)

    # 2. Transpose to match target shape
    permute_order = _get_transpose_order(input_axes, flattened_output_axes)
    tensor = transpose(tensor, permute_order)

    # 3. Reshape to final target shape
    output_shape = _compute_output_shape(axes_map, grouped_output_axes)
    tensor = reshape(tensor, output_shape)

    return tensor
