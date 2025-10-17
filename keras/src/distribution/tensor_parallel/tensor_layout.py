import collections

import keras


class Split:
    """Splits a tensor into shards along a specified dimension.

    This is an internal utility used by a higher-level distribution API.
    It implements sharding by slicing a tensor along one of its axes.
    It handles cases where the dimension size is not perfectly divisible by the
    number of workers by distributing the remainder elements one by one to the
    first few workers.
    """

    def __init__(self, device_count, dim, sharding_type="auto"):
        """Initializes the Split action.

        Args:
            device_count: The total number of workers/shards.
            dim: The dimension along which to split the tensor. If -1, the
                last dimension is used.
            sharding_type: If `dim` is -1, this can be 'row' (dim=0) or
                'column' (dim=1) to infer the split axis for 2D tensors.
                Defaults to "auto".
        """
        self.device_count = device_count
        self.dim = dim
        self.sharding_type = sharding_type

        if dim == -1 and sharding_type != "auto":
            if sharding_type == "row":
                self.dim = 0
            elif sharding_type == "column":
                self.dim = 1

    def __call__(self, tensor, rank):
        """Splits the tensor and returns the shard corresponding to the rank.

        This method calculates the correct slice of the tensor for a given
        worker rank, handling uneven distributions gracefully.

        Args:
            tensor: The full tensor to be sharded.
            rank: The rank of the worker for which to get the shard.

        Returns:
            A tensor shard corresponding to the given rank.
        """
        if self.dim == -1:
            dim = keras.ops.ndim(tensor) - 1
        else:
            dim = self.dim

        total_size = tensor.shape[dim]
        split_size = total_size // self.device_count
        remainder = total_size % self.device_count

        start_idx = rank * split_size + min(rank, remainder)
        end_idx = start_idx + split_size + (1 if rank < remainder else 0)

        slices = [slice(None)] * keras.ops.ndim(tensor)
        slices[dim] = slice(start_idx, end_idx)
        return tensor[tuple(slices)]


LayoutMap = collections.namedtuple("LayoutMap", ["state_rules", "output_rules"])
