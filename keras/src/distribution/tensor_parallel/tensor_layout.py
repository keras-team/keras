import collections

from keras.src import ops


def split_tensor_for_parallelism(tensor, index, device_count, dim):
    """Calculates a slice of a tensor along a specified dimension for a
    given index.

    This utility is used in tensor parallelism API to distribute a
    tensor across multiple devices.

    Args:
        tensor: The full tensor to be sharded.
        index: The index of the device/shard to return (e.g., 0, 1, 2...).
        device_count: The total number of parallel devices or splits.
        dim: The dimension along which to split the tensor. Supports negative
            indexing.

    Returns:
        A tensor slice corresponding to the given `index`.
    """
    if dim < 0:
        split_dim = ops.ndim(tensor) + dim
    else:
        split_dim = dim

    splits = ops.array_split(
        tensor, indices_or_sections=device_count, axis=split_dim
    )
    return splits[index]


LayoutMap = collections.namedtuple("LayoutMap", ["state_rules", "output_rules"])