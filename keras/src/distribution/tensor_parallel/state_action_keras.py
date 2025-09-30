from typing import Any
from typing import Sequence

import keras


class StateActionKeras:
    """
    Abstract base class for actions that transform tensors for distribution.

    An action defines how a tensor should be processed for a specific worker
    (rank) and how to reverse that action to reconstruct the original tensor.
    """

    def __call__(self, tensor: Any, rank: int) -> Any:
        """
        Apply the state action to a tensor for a given worker rank.

        Args:
            tensor: The input tensor to transform.
            rank: The rank of the worker process.

        Returns:
            The transformed tensor shard for the specified rank.
        """
        raise NotImplementedError

    def undo(self, tensors: Sequence[Any]) -> Any:
        """
        Reverse the action to reconstruct the original tensor from its parts.

        Args:
            tensors: A sequence of tensor shards from all worker processes.

        Returns:
            The reconstructed, original tensor.
        """
        raise NotImplementedError


class _ConcatenateMixin:
    """A mixin class that provides a common `undo` method via concatenation."""

    def undo(self, tensors: Sequence[Any]) -> Any:
        """Concatenate a sequence of tensors along the specified dimension."""
        if self.dim == -1:
            # Resolve dim=-1 to the last dimension of the input tensors
            dim = keras.ops.ndim(tensors[0]) - 1
        else:
            dim = self.dim
        return keras.ops.concatenate(tensors, axis=dim)


class SplitKeras(StateActionKeras, _ConcatenateMixin):
    """
    Splits a tensor into shards along a specified dimension for each worker.

    Args:
        world_size: The total number of workers/shards.
        dim: The dimension along which to split the tensor. If -1, the last
             dimension is used.
        sharding_type: If `dim` is -1, this can be 'row' (dim=0) or 'column'
                       (dim=1) to infer the split axis.
    """

    def __init__(self, world_size: int, dim: int, sharding_type: str = "auto"):
        self.world_size = world_size
        self.dim = dim
        self.sharding_type = sharding_type

        if dim == -1 and sharding_type != "auto":
            if sharding_type == "row":
                self.dim = 0
            elif sharding_type == "column":
                self.dim = 1

    def __call__(self, tensor: Any, rank: int) -> Any:
        """Splits the tensor and returns the shard corresponding to the rank."""
        if self.dim == -1:
            dim = keras.ops.ndim(tensor) - 1
        else:
            dim = self.dim

        total_size = tensor.shape[dim]
        split_size = total_size // self.world_size
        remainder = total_size % self.world_size

        start_idx = rank * split_size + min(rank, remainder)
        end_idx = start_idx + split_size + (1 if rank < remainder else 0)

        slices = [slice(None)] * keras.ops.ndim(tensor)
        slices[dim] = slice(start_idx, end_idx)
        return tensor[tuple(slices)]


class GatherKeras(StateActionKeras, _ConcatenateMixin):
    """
    Represents a gather operation, where tensors are collected from all ranks.

    The actual collective communication is handled by a different layer; this
    class primarily serves as a placeholder to trigger that communication and
    define how to undo it.

    Args:
        world_size: The total number of workers.
        dim: The dimension along which tensors will be concatenated in the
             `undo` operation.
    """

    def __init__(self, world_size: int, dim: int):
        self.world_size = world_size
        self.dim = dim

    def __call__(self, tensor: Any, rank: int) -> Any:
        """
        Returns the tensor as-is.

        The actual gathering is performed by the communication backend.
        """
        return tensor


class SumKeras(StateActionKeras):
    """
    Represents a sum operation, where tensors are summed across all ranks.

    The actual collective communication (AllReduce) is handled by a different
    layer. This class triggers that operation and defines the `undo` logic.

    Args:
        world_size: The total number of workers.
    """

    def __init__(self, world_size: int):
        self.world_size = world_size

    def __call__(self, tensor: Any, rank: int) -> Any:
        """
        Returns the tensor as-is.

        The actual summing is performed by the communication backend.
        """
        return tensor

    def undo(self, tensors: Sequence[Any]) -> Any:
        """Sums the collected tensors from all workers."""
        return sum(tensors)
