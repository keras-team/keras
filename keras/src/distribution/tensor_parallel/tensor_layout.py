import keras


class LayoutAction:
    """Abstract base class for actions that transform tensors for distribution.

    A LayoutAction defines a rule for how a single tensor should be physically
    represented across multiple devices. It includes forward operation
    (`__call__`) to shard the tensor and a reverse operation (`undo`)
    to reconstruct it."""

    def __call__(self, tensor, rank):
        """Applies the distribution action to a tensor for a specific worker.

        Args:
            tensor: The input tensor to be distributed.
            rank: The integer rank of the current worker/device.

        Raises:
            NotImplementedError: This is an abstract method and must be
                implemented by subclasses.

        Returns:
            A shard or transformation of the input tensor specific to the given
            rank.
        """
        raise NotImplementedError

    def undo(self, tensors):
        """Reverses the distribution action, reconstructing the original tensor.

        Args:
            tensors: A sequence of tensor shards from all workers.

        Raises:
            NotImplementedError: This is an abstract method and must be
                implemented by subclasses.

        Returns:
            The reconstructed, single tensor.
        """
        raise NotImplementedError


class _ConcatenateMixin:
    """A mixin class providing a common `undo` method via concatenation.

    This class is intended to be used as a mixin for `LayoutAction` subclasses
    that can be undone by simple concatenation.
    """

    def undo(self, tensors):
        """Concatenates sequence of tensors to reconstruct the original tensor.

        Args:
            tensors: A sequence of tensor shards, one from each worker.

        Returns:
            The single tensor reconstructed by concatenating the shards.
        """
        if self.dim == -1:
            dim = keras.ops.ndim(tensors[0]) - 1
        else:
            dim = self.dim
        return keras.ops.concatenate(tensors, axis=dim)


class Split(_ConcatenateMixin, LayoutAction):
    """Splits a tensor into shards along a specified dimension for each worker.

    This action implements sharding by slicing a tensor along one of its axes.
    It handles cases where the dimension size is not perfectly divisible by the
    number of workers by distributing the remainder elements one by one to the
    first few workers.

    The `undo` operation is handled by the `_ConcatenateMixin`, which
    concatenates the shards back together.

    Args:
        world_size (int): The total number of workers/shards.
        dim (int): The dimension along which to split the tensor. If -1, the
            last dimension is used.
        sharding_type (str): If `dim` is -1, this can be 'row' (dim=0) or
            'column' (dim=1) to infer the split axis for 2D tensors.
            Defaults to "auto".
    """

    def __init__(self, world_size, dim, sharding_type="auto"):
        """Initializes the Split action.

        Args:
            world_size (int): The total number of workers/shards.
            dim (int): The dimension along which to split the tensor.
            sharding_type (str): A hint for inferring the dimension if `dim`
                is -1.
        """
        super().__init__()
        self.world_size = world_size
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
            rank (int): The rank of the worker for which to get the shard.

        Returns:
            A tensor shard corresponding to the given rank.
        """
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


class LayoutMap:
    """A mapping that defines layout rules for model states and outputs.

    This class acts as a configuration object that holds dictionaries of
    `LayoutAction` instances. These rules specify how model variables (states)
    and layer outputs should be distributed across a set of devices.

    Attributes:
        state_rules (dict): A dictionary mapping variable names or patterns to
            `LayoutAction` instances.
        output_rules (dict): A dictionary mapping layer output names or
            patterns to `LayoutAction` instances.
    """

    def __init__(self, state_rules, output_rules):
        """Initializes the LayoutMap.

        Args:
            state_rules (dict): A dictionary of rules for model states.
            output_rules (dict): A dictionary of rules for model outputs.
        """
        self.state_rules = state_rules
        self.output_rules = output_rules

    def create_collective_ops(self, devices):
        """Creates the necessary collective communication operations.

        Args:
            devices: A sequence of device identifiers.

        Returns:
            The `LayoutMap` instance itself.
        """
        return self
