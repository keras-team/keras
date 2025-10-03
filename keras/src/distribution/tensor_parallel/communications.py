from typing import Any
from typing import List
from typing import Tuple

from keras.src.distribution import distributed_backend


class CollectiveOpKeras:
    """Base class for Keras collective communication operations.

    This class provides a common interface for various collective communication
    primitives like AllReduce, AllGather, and Broadcast. Subclasses must
    implement the `__call__` method.

    Args:
        world_size (int): The total number of participating processes or devices
            in the communication group.
        rank (int, optional): The rank of the current process. Defaults to 0.
    """

    def __init__(self, world_size: int, rank: int = 0):
        """Initializes the collective operation.

        Args:
            world_size (int): The total number of participating processes or
                devices in the communication group.
            rank (int, optional): The rank of the current process. Defaults
                to 0.
        """
        self.world_size = world_size
        self.rank = rank

    def __call__(self, *args, **kwargs):
        """Executes the collective operation."""
        raise NotImplementedError


class AllReduceKeras(CollectiveOpKeras):
    """Performs an AllReduce collective operation.

    AllReduce reduces the input tensor across all devices and distributes the
    final result back to all devices.

    Args:
        world_size (int): The total number of participating processes.
        op (str, optional): The reduction operation. Supported values are
            "sum" and "mean". Defaults to "sum".
        rank (int, optional): The rank of the current process. Defaults to 0.

    Raises:
        NotImplementedError: If the current backend does not support the
            AllReduce operation.
    """

    def __init__(self, world_size: int, op: str = "sum", rank: int = 0):
        """Initializes the AllReduce operation.

        Args:
            world_size (int): The total number of participating processes.
            op (str, optional): The reduction operation. Supported values are
                "sum" and "mean". Defaults to "sum".
            rank (int, optional): The rank of current process. Defaults to 0.
        """
        super().__init__(world_size, rank)
        self.op = op
        self.all_reduce_fn = distributed_backend.get_communication_ops().get(
            "all_reduce"
        )
        if self.all_reduce_fn is None:
            raise NotImplementedError(
                "AllReduce is not supported by the current backend."
            )

    def __call__(self, local_tensor: Any, axis_name: str) -> Any:
        """Executes the AllReduce operation.

        Args:
            local_tensor (Any): The tensor on the local device to be reduced.
            axis_name (str): The name of the axis to reduce over, used by the
                backend for identifying the device group.

        Returns:
            Any: The reduced tensor, which is identical on all devices.
        """
        return self.all_reduce_fn(local_tensor, op=self.op, axis_name=axis_name)


class AllGatherKeras(CollectiveOpKeras):
    """Performs an AllGather collective operation.

    AllGather gathers tensors from all devices and concatenates them along a
    specified dimension. The final concatenated tensor is available on all
    devices.

    Args:
        world_size (int): The total number of participating processes.
        dim (int, optional): The dimension along which to concatenate the
            gathered tensors. Defaults to -1.
        rank (int, optional): The rank of the current process. Defaults to 0.

    Raises:
        NotImplementedError: If the current backend does not support the
            AllGather operation.
    """

    def __init__(self, world_size: int, dim: int = -1, rank: int = 0):
        """Initializes the AllGather operation.

        Args:
            world_size (int): The total number of participating processes.
            dim (int, optional): The dimension along which to concatenate the
                gathered tensors. Defaults to -1.
            rank (int, optional): The rank of current process. Defaults to 0.
        """
        super().__init__(world_size, rank)
        self.dim = dim
        self.all_gather_fn = distributed_backend.get_communication_ops().get(
            "all_gather"
        )
        if self.all_gather_fn is None:
            raise NotImplementedError(
                "AllGather is not supported by the current backend."
            )

    def __call__(self, local_tensor: Any, axis_name: str) -> Any:
        """Executes the AllGather operation.

        Args:
            local_tensor (Any): The tensor on the local device to be gathered.
            axis_name (str): The name of the axis for the device group, used by
                the backend for communication.

        Returns:
            Any: The concatenated tensor, containing data from all devices.
        """
        return self.all_gather_fn(
            local_tensor, axis=self.dim, axis_name=axis_name
        )


class BroadcastKeras(CollectiveOpKeras):
    """Performs a Broadcast collective operation.

    Broadcast sends a tensor from a single source device to all other devices
    in the group.

    Args:
        world_size (int): The total number of participating processes.
        src_rank (int, optional): The rank of the source process that is
            broadcasting the tensor. Defaults to 0.
        rank (int, optional): The rank of the current process. Defaults to 0.

    Raises:
        NotImplementedError: If the current backend does not support the
            Broadcast operation.
    """

    def __init__(self, world_size: int, src_rank: int = 0, rank: int = 0):
        """Initializes the Broadcast operation.

        Args:
            world_size (int): The total number of participating processes.
            src_rank (int, optional): The rank of the source process that is
                broadcasting the tensor. Defaults to 0.
            rank (int, optional): The rank of current process. Defaults to 0.
        """
        super().__init__(world_size, rank)
        self.src_rank = src_rank
        self.broadcast_fn = distributed_backend.get_communication_ops().get(
            "broadcast"
        )
        if self.broadcast_fn is None:
            raise NotImplementedError(
                "Broadcast is not supported by the current backend."
            )

    def __call__(self, tensor: Any, axis_name: str) -> Any:
        """Executes the Broadcast operation.

        Args:
            tensor (Any): The tensor to be broadcasted (on the source device) or
                received (on other devices).
            axis_name (str): The name of the axis for the device group, used by
                the backend for communication.

        Returns:
            Any: The broadcasted tensor from the source device.
        """
        return self.broadcast_fn(
            tensor, root=self.src_rank, axis_name=axis_name
        )


class TensorParallelCommunicator:
    """Manages communication operations for tensor parallelism.

    This class abstracts the collective communication logic required for
    implementing tensor-parallel models, providing specific methods for
    column-parallel and row-parallel layers.

    Args:
        world_size (int): The total number of devices in the group.
        rank (int, optional): The rank of the current device. Defaults to 0.
    """

    def __init__(self, world_size: int, rank: int = 0):
        """Initializes the communicator.

        Args:
            world_size (int): The total number of devices in the group.
            rank (int, optional): The rank of the current device. Defaults to 0.
        """
        self.world_size = world_size
        self.rank = rank
        self.allreduce = AllReduceKeras(world_size, rank=rank)
        self.allgather = AllGatherKeras(world_size, rank=rank)
        self.broadcast = BroadcastKeras(world_size, rank=rank)

    def forward_column_parallel(
        self, partial_outputs: List, dim: int = -1, axis_name: str = "batch"
    ):
        """Gathers output shards in a column-parallel forward pass.

        In a column-parallel layer, the output activations are sharded across
        devices. This function collects all shards using an AllGather operation
        to form the full output tensor.

        Args:
            partial_outputs (List): A list of output shards, with one tensor
                from each device in the communication group.
            dim (int, optional): The dimension along which to concatenate the
                gathered tensors. Defaults to -1.
            axis_name (str, optional): The name of the communication axis used
                by the backend. Defaults to "batch".

        Returns:
            Any: The full, gathered output tensor, which is identical on all
            devices.
        """
        self.allgather.dim = dim
        return self.allgather(partial_outputs[self.rank], axis_name=axis_name)

    def backward_column_parallel(
        self,
        partial_gradients: List,
        op: str = "sum",
        axis_name: str = "batch",
    ) -> List:
        """Reduces weight gradients in a column-parallel backward pass.

        This is the conjugate operation to `forward_column_parallel`. It uses an
        AllReduce operation to sum the gradients computed on each device for
        the weight matrix.

        Args:
            partial_gradients (List): A list of local weight gradients, with
                one tensor from each device.
            op (str, optional): The reduction operation, either "sum" or "mean".
                Defaults to "sum".
            axis_name (str, optional): The name of the communication axis.
                Defaults to "batch".

        Returns:
            Any: The reduced gradient tensor, identical on all devices.
        """
        self.allreduce.op = op
        return self.allreduce(partial_gradients[self.rank], axis_name=axis_name)

    def forward_row_parallel(
        self, partial_outputs: List, op: str = "sum", axis_name: str = "batch"
    ) -> List:
        """Reduces output shards in a row-parallel forward pass.

        In a row-parallel layer, each device computes a partial output. This
        function uses an AllReduce operation to sum these partial outputs into
        the final, correct output tensor.

        Args:
            partial_outputs (List): A list of partial outputs, one from each
                device.
            op (str, optional): The reduction operation, either "sum" or "mean".
                Defaults to "sum".
            axis_name (str, optional): The name of the communication axis.
                Defaults to "batch".

        Returns:
            Any: The final, reduced output tensor.
        """
        self.allreduce.op = op
        return self.allreduce(partial_outputs[self.rank], axis_name=axis_name)

    def backward_row_parallel(
        self, partial_gradients: List, dim: int = -1, axis_name: str = "batch"
    ):
        """Gathers input gradients in a row-parallel backward pass.

        This is the conjugate operation to `forward_row_parallel`. It uses an
        AllGather operation to collect the sharded input gradients from all
        devices to reconstruct the full gradient tensor.

        Args:
            partial_gradients (List): A list of local input gradients, one
                from each device.
            dim (int, optional): The dimension along which to concatenate the
                gradients. Defaults to -1.
            axis_name (str, optional): The name of the communication axis.
                Defaults to "batch".

        Returns:
            Any: The full, gathered gradient tensor.
        """
        self.allgather.dim = dim
        return self.allgather(partial_gradients[self.rank], axis_name=axis_name)

    def handle_mlp_handshake(
        self, up_projection_outputs: List, down_projection_inputs: List
    ) -> Tuple:
        """Manages communication between two MLP layers for tensor parallelism.

        This is a specialized function for a common pattern where a
        column-parallel layer (`up_projection`) is followed by a row-parallel
        layer (`down_projection`). It combines their forward communication.

        Args:
            up_projection_outputs (List): A list of local output tensors from
                the `up_projection` layer on each device.
            down_projection_inputs (List): A list of local input tensors for
                the `down_projection` layer on each device.

        Returns:
            tuple: A tuple with the gathered output from `up_projection` and
                the reduced input for `down_projection`.
        """
        up_output = self.forward_column_parallel(
            up_projection_outputs[self.rank], dim=-1
        )
        down_inputs = self.forward_row_parallel(
            down_projection_inputs[self.rank], op="sum"
        )
        return up_output, down_inputs

    def slice_upstream_gradient_for_column_parallel(
        self, full_gradient: Any, rank: int, world_size: int, dim: int = -1
    ) -> Any:
        """Slices the gradient for a column-parallel layer's backward pass.

        Before the backward pass of a column-parallel layer, the full upstream
        gradient must be sliced so that each device receives the portion
        corresponding to its output shard. It handles uneven sharding.

        Args:
            full_gradient (Any): The complete upstream gradient tensor.
            rank (int): The rank of the current device.
            world_size (int): The total number of devices.
            dim (int, optional): The dimension to slice along. Defaults to -1.

        Returns:
            Any: The sliced portion of the gradient for the current device.
        """
        shape = getattr(full_gradient, "shape", None)
        if shape is None or not (-len(shape) <= dim < len(shape)):
            return full_gradient

        total_size = shape[dim]
        slice_size = total_size // world_size
        remainder = total_size % world_size
        start_idx = rank * slice_size + min(rank, remainder)
        end_idx = start_idx + slice_size + (1 if rank < remainder else 0)
        slices = [slice(None)] * len(shape)
        slices[dim] = slice(start_idx, end_idx)
        return full_gradient[tuple(slices)]

    def slice_upstream_gradient_for_row_parallel(
        self, full_gradient: Any, rank: int, world_size: int, dim: int = 0
    ) -> Any:
        """Slices the gradient for a row-parallel layer's backward pass.

        Before the backward pass of a row-parallel layer, the full upstream
        gradient must be sliced so each device gets the part
        corresponding to its input shard.

        Args:
            full_gradient (Any): The complete upstream gradient tensor.
            rank (int): The rank of the current device.
            world_size (int): The total number of devices.
            dim (int, optional): The dimension to slice along. Defaults to 0.

        Returns:
            Any: The sliced portion of the gradient for the current device.
        """
        shape = getattr(full_gradient, "shape", None)
        if shape is None or not (-len(shape) <= dim < len(shape)):
            return full_gradient

        total_size = shape[dim]
        slice_size = total_size // world_size
        start_idx = rank * slice_size
        end_idx = (rank + 1) * slice_size
        if rank == world_size - 1:
            end_idx = total_size
        slices = [slice(None)] * len(shape)
        slices[dim] = slice(start_idx, end_idx)
        return full_gradient[tuple(slices)]


def allreduce_gradients(gradients: Any, world_size: int) -> Any:
    """Utility function to perform a mean AllReduce operation on gradients.

    This is commonly used in data parallelism to average gradients across all
    devices before applying the optimizer step.

    Args:
        gradients (Any): A tensor or list of tensors representing the gradients
            on the local device.
        world_size (int): The total number of devices.

    Returns:
        Any: The averaged gradient tensor.
    """
    allreduce_op = AllReduceKeras(world_size, op="mean")
    local_gradient = gradients[0] if isinstance(gradients, list) else gradients
    return allreduce_op(local_gradient, axis_name="batch")


def allgather_outputs(outputs: Any, world_size: int, dim: int = -1) -> Any:
    """Utility function to perform an AllGather operation on model outputs.

    This can be used to collect the final outputs from all devices when running
    inference in a distributed manner.

    Args:
        outputs (Any): A tensor or list of tensors representing the model's
            output on the local device.
        world_size (int): The total number of devices.
        dim (int, optional): The dimension along which to concatenate the
            outputs. Defaults to -1.

    Returns:
        Any: The gathered, full output tensor.
    """
    allgather_op = AllGatherKeras(world_size, dim=dim)
    local_output = outputs[0] if isinstance(outputs, list) else outputs
    return allgather_op(local_output, axis_name="batch")


def broadcast_parameters(
    parameters: List[Any], world_size: int, src_rank: int = 0
) -> Any:
    """Utility function to broadcast model parameters from a source device.

    This is typically used at the beginning of training to ensure all devices
    start with the same initial model weights.

    Args:
        parameters (List[Any]): A list of model parameters, where each element
            corresponds to the parameters on a device.
        world_size (int): The total number of devices.
        src_rank (int, optional): The rank of the source device to broadcast
            from. Defaults to 0.

    Returns:
        Any: The broadcasted parameters.
    """
    broadcast_op = BroadcastKeras(world_size, src_rank=src_rank)
    return broadcast_op(parameters[src_rank], axis_name="batch")
