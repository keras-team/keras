from typing import Any
from typing import List
from typing import Tuple

from keras.src.backend.distributed import backend_resolver
from keras.src.backend.distributed.base import DistributedBackend


class CollectiveOpKeras:
    """Base class for Keras collective communication operations.

    This class provides a common interface for distributed communication
    primitives like AllReduce, AllGather, and Broadcast. It is not meant
    to be used directly but rather subclassed to implement specific
    collective operations.

    Args:
        world_size (int): The total number of participating processes or devices
            in the distributed job.
        rank (int, optional): The unique identifier for the current process.
            Defaults to 0.
    """

    def __init__(self, world_size: int, rank: int = 0):
        self.world_size = world_size
        self.rank = rank

    def __call__(self, *args, **kwargs):
        """Executes the collective operation."""
        raise NotImplementedError


class AllReduceKeras(CollectiveOpKeras):
    """
    Performs an AllReduce collective operation.

    AllReduce combines a tensor from each process and distributes the result
    back to all processes. For example, it can be used to sum or average

    gradients across all workers.

    Args:
        world_size (int): The total number of participating processes.
        backend (DistributedBackend): The distributed backend implementation
            (e.g., for JAX, TensorFlow).
        op (str, optional): The reduction operation to perform. Common values
            are "sum" and "mean". Defaults to "sum".
        rank (int, optional): The rank of the current process. Defaults to 0.

    Raises:
        NotImplementedError: If the 'all_reduce' operation is not supported
            by the provided backend.
    """

    def __init__(
        self,
        world_size: int,
        backend: DistributedBackend,
        op: str = "sum",
        rank: int = 0,
    ):
        super().__init__(world_size, rank)
        self.op = op
        self.backend = backend
        self.all_reduce_fn = self.backend.get_communication_ops().get(
            "all_reduce"
        )
        if self.all_reduce_fn is None:
            raise NotImplementedError(
                "AllReduce is not supported by the current backend."
            )

    def __call__(self, local_tensor: Any, axis_name: str) -> Any:
        """
        Executes the AllReduce operation on a local tensor.

        Args:
            local_tensor (Any): The tensor on the current device to be reduced.
            axis_name (str): The name of the axis to reduce over, used by
                distributed backends like JAX to identify the group of devices.

        Returns:
            Any: The reduced tensor, which is identical on all participating
                devices.
        """
        return self.all_reduce_fn(local_tensor, op=self.op, axis_name=axis_name)


class AllGatherKeras(CollectiveOpKeras):
    """
    Performs an AllGather collective operation.

    AllGather collects a tensor from each process and concatenates them along
    a specified dimension on all processes.

    Args:
        world_size (int): The total number of participating processes.
        backend (DistributedBackend): The distributed backend implementation.
        dim (int, optional): The dimension along which to concatenate the
            tensors. Defaults to -1.
        rank (int, optional): The rank of the current process. Defaults to 0.

    Raises:
        NotImplementedError: If the 'all_gather' operation is not supported
            by the provided backend.
    """

    def __init__(
        self,
        world_size: int,
        backend: DistributedBackend,
        dim: int = -1,
        rank: int = 0,
    ):
        super().__init__(world_size, rank)
        self.dim = dim
        self.backend = backend
        self.all_gather_fn = self.backend.get_communication_ops().get(
            "all_gather"
        )
        if self.all_gather_fn is None:
            raise NotImplementedError(
                "AllGather is not supported by the current backend."
            )

    def __call__(self, local_tensor: Any, axis_name: str) -> Any:
        """
        Executes the AllGather operation on a local tensor.

        Args:
            local_tensor (Any): The tensor on the current device to be gathered.
            axis_name (str): The name of the axis to gather along, used by
                distributed backends to identify the device group.

        Returns:
            Any: The gathered tensor, containing concatenated data from all
                devices. This tensor is identical on all participating devices.
        """
        return self.all_gather_fn(
            local_tensor, axis=self.dim, axis_name=axis_name
        )


class BroadcastKeras(CollectiveOpKeras):
    """
    Performs a Broadcast collective operation.

    Broadcast sends a tensor from a single source process (src_rank) to all
    other processes.

    Args:
        world_size (int): The total number of participating processes.
        backend (DistributedBackend): The distributed backend implementation.
        src_rank (int, optional): The rank of the process that sends the
            tensor. Defaults to 0.
        rank (int, optional): The rank of the current process. Defaults to 0.

    Raises:
        NotImplementedError: If the 'broadcast' operation is not supported
            by the provided backend.
    """

    def __init__(
        self,
        world_size: int,
        backend: DistributedBackend,
        src_rank: int = 0,
        rank: int = 0,
    ):
        super().__init__(world_size, rank)
        self.src_rank = src_rank
        self.backend = backend
        self.broadcast_fn = self.backend.get_communication_ops().get(
            "broadcast"
        )
        if self.broadcast_fn is None:
            raise NotImplementedError(
                "Broadcast is not supported by the current backend."
            )

    def __call__(self, tensor: Any, axis_name: str) -> Any:
        """
        Executes the Broadcast operation.

        Args:
            tensor (Any): The tensor to be broadcasted. On the `src_rank` device
                this is the data to be sent. On other devices, it can be a
                placeholder with the correct shape and dtype.
            axis_name (str): The name of the axis, used by distributed backends
                to identify the device group.

        Returns:
            Any: The broadcasted tensor received from the source rank.
        """
        return self.broadcast_fn(
            tensor, root=self.src_rank, axis_name=axis_name
        )


class TensorParallelCommunicator:
    """
    Manages communication operations for tensor parallelism.

    This class provides a high-level interface for the specific communication
    patterns required in tensor-parallel models, such as column-parallel and
    row-parallel linear layers.

    Args:
        world_size (int): The total number of devices in the tensor-parallel
            group.
        rank (int, optional): The rank of the current device. Defaults to 0.
    """

    def __init__(self, world_size: int, rank: int = 0):
        self.world_size = world_size
        self.rank = rank
        self.backend = backend_resolver.get_distributed_backend()
        self.allreduce = AllReduceKeras(
            world_size, backend=self.backend, rank=rank
        )
        self.allgather = AllGatherKeras(
            world_size, backend=self.backend, rank=rank
        )
        self.broadcast = BroadcastKeras(
            world_size, backend=self.backend, rank=rank
        )

    def forward_column_parallel(
        self, local_tensor: Any, dim: int = -1, axis_name: str = "i"
    ) -> Any:
        """
        Communication for the forward pass of a column-parallel layer.

        In a column-parallel linear layer, each device computes a part of the
        output. This function gathers these parts from all devices to form the
        full output tensor. This is an AllGather operation.

        Args:
            local_tensor (Any): The partial output tensor from the local device.
            dim (int, optional): The dimension to gather along. Defaults to -1.
            axis_name (str, optional): The axis name for the backend.
                Defaults to "i".

        Returns:
            Any: The full output tensor, gathered from all devices.
        """
        self.allgather.dim = dim
        return self.allgather(local_tensor, axis_name=axis_name)

    def backward_column_parallel(
        self, local_gradient: Any, op: str = "sum", axis_name: str = "i"
    ) -> Any:
        """
        Communication for the backward pass of a column-parallel layer.

        The gradient with respect to the input is computed locally. Since the
        forward pass was an identity operation on the input, the backward pass
        requires an AllReduce to sum the gradients from all devices.

        Args:
            local_gradient (Any): The local gradient computed on the device.
            op (str, optional): The reduction operation. Defaults to "sum".
            axis_name (str, optional): The axis name for the backend.
                Defaults to "i".

        Returns:
            Any: The reduced gradient.
        """
        self.allreduce.op = op
        return self.allreduce(local_gradient, axis_name=axis_name)

    def forward_row_parallel(
        self, local_output: Any, op: str = "sum", axis_name: str = "i"
    ) -> Any:
        """
        Communication for the forward pass of a row-parallel layer.

        In a row-parallel linear layer, the input is sharded, and each device
        computes a partial output. These partial outputs must be summed via
        AllReduce to get the final correct output.

        Args:
            local_output (Any): The partial output from the local device.
            op (str, optional): The reduction operation. Defaults to "sum".
            axis_name (str, optional): The axis name for the backend.
                Defaults to "i".

        Returns:
            Any: The final output tensor after reduction.
        """
        self.allreduce.op = op
        return self.allreduce(local_output, axis_name=axis_name)

    def backward_row_parallel(
        self, local_gradient: Any, dim: int = -1, axis_name: str = "i"
    ) -> Any:
        """
        Communication for the backward pass of a row-parallel layer.

        The gradient with respect to the input needs to be gathered from all
        devices, as the forward pass was an AllReduce. This is an identity
        operation on the gradient (no communication needed for the input grad),
        but if the gradient itself needs to be passed to another parallel layer,
        it may need to be gathered.

        Note: Typically, the gradient with respect to the input of a
        row-parallel layer is an identity operation from the perspective of
        communication, as the upstream gradient is already the correct value.
        This AllGather is for cases where subsequent layers need the full
        gradient tensor.

        Args:
            local_gradient (Any): The local gradient on the device.
            dim (int, optional): The dimension to gather along. Defaults to -1.
            axis_name (str, optional): The axis name for the backend.
                Defaults to "i".

        Returns:
            Any: The gathered gradient.
        """
        self.allgather.dim = dim
        return self.allgather(local_gradient, axis_name=axis_name)

    def handle_mlp_handshake(
        self, up_projection_outputs: List, down_projection_inputs: List
    ) -> Tuple:
        """
        Manages the communication between two MLP layers for tensor parallelism.

        This handles the typical pattern where a column-parallel layer (`up`)
        is followed by a row-parallel layer (`down`). It gathers the output
        of the first layer and reduces the input to the second layer.

        Args:
            up_projection_outputs (List): A list of partial outputs from the
                column-parallel layer across all devices.
            down_projection_inputs (List): A list of partial inputs for the
                row-parallel layer across all devices.

        Returns:
            Tuple: A tuple containing full gathered output of the up-projection
                and the fully reduced input for the down-projection.
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
        """
        Slices the upstream gradient for column-parallel layer's backward pass.

        Since forward pass involved gathering tensors, backward pass
        requires slicing gradient before it's passed to the local computation.
        This function handles both even and uneven splits of the tensor.

        Args:
            full_gradient (Any): The full gradient tensor to be sliced.
            rank (int): The rank of the current device.
            world_size (int): The total number of devices.
            dim (int, optional): The dimension along which to slice.
                Defaults to -1.

        Returns:
            Any: The sliced portion of the gradient for the current device.
                 Returns the original gradient if slicing fails.
        """
        try:
            total_size = full_gradient.shape[dim]
            slice_size = total_size // world_size
            remainder = total_size % world_size
            start_idx = rank * slice_size + min(rank, remainder)
            end_idx = start_idx + slice_size + (1 if rank < remainder else 0)
            slices = [slice(None)] * len(full_gradient.shape)
            slices[dim] = slice(start_idx, end_idx)
            return full_gradient[tuple(slices)]
        except Exception:
            # Fallback if slicing is not possible (e.g., shape is unknown)
            return full_gradient

    def slice_upstream_gradient_for_row_parallel(
        self, full_gradient: Any, rank: int, world_size: int, dim: int = 0
    ) -> Any:
        """
        Slices the upstream gradient for a row-parallel layer's backward pass.

        Since the input to the row-parallel layer was sharded, the gradient
        w.r.t the input must also be sharded in the same way.

        Args:
            full_gradient (Any): The full gradient tensor to be sliced.
            rank (int): The rank of the current device.
            world_size (int): The total number of devices.
            dim (int, optional): The dimension along which to slice.
                Defaults to 0.

        Returns:
            Any: The sliced portion of the gradient for the current device.
                 Returns the original gradient if slicing fails.
        """
        try:
            total_size = full_gradient.shape[dim]
            slice_size = total_size // world_size
            start_idx = rank * slice_size
            end_idx = (rank + 1) * slice_size
            # Ensure the last rank gets the remainder
            if rank == world_size - 1:
                end_idx = total_size
            slices = [slice(None)] * len(full_gradient.shape)
            slices[dim] = slice(start_idx, end_idx)
            return full_gradient[tuple(slices)]
        except Exception:
            # Fallback if slicing is not possible (e.g., shape is unknown)
            return full_gradient


def allreduce_gradients(
    gradients: Any, world_size: int, backend: DistributedBackend
) -> Any:
    """
    Utility function to perform a mean AllReduce operation on gradients.

    This is commonly used in data parallelism to average gradients across all
    workers before applying the optimizer step.

    Args:
        gradients (Any): A tensor or list of tensors representing gradients.
            If a list, the first element is used.
        world_size (int): The total number of participating processes.
        backend (DistributedBackend): The distributed backend instance.

    Returns:
        Any: The averaged gradient tensor.
    """
    allreduce_op = AllReduceKeras(world_size, backend=backend, op="mean")
    # Handle cases where gradients might be passed as a single-element list
    local_gradient = gradients[0] if isinstance(gradients, list) else gradients
    return allreduce_op(local_gradient, axis_name="batch")


def allgather_outputs(
    outputs: Any,
    world_size: int,
    backend: DistributedBackend,
    dim: int = -1,
) -> Any:
    """
    Utility function to perform an AllGather operation on model outputs.

    This can be used to collect outputs from all devices to form a complete
    batch of predictions.

    Args:
        outputs (Any): A tensor or list of tensors representing local outputs.
            If a list, the first element is used.
        world_size (int): The total number of participating processes.
        backend (DistributedBackend): The distributed backend instance.
        dim (int, optional): The dimension to concatenate along. Defaults to -1.

    Returns:
        Any: The gathered output tensor from all devices.
    """
    allgather_op = AllGatherKeras(world_size, backend=backend, dim=dim)
    local_output = outputs[0] if isinstance(outputs, list) else outputs
    return allgather_op(local_output, axis_name="batch")


def broadcast_parameters(
    parameters: List[Any],
    world_size: int,
    backend: DistributedBackend,
    src_rank: int = 0,
) -> Any:
    """
    Utility function to broadcast model parameters from a source device.

    This ensures that all devices start with the exact same model weights at the
    beginning of training.

    Args:
        parameters (List[Any]): A list of parameters from all devices. The
            parameter from `src_rank` will be broadcast.
        world_size (int): The total number of participating processes.
        backend (DistributedBackend): The distributed backend instance.
        src_rank (int, optional): The rank of the source device. Defaults to 0.

    Returns:
        Any: The broadcasted parameters, which will be identical on all devices.
    """
    broadcast_op = BroadcastKeras(
        world_size, backend=backend, src_rank=src_rank
    )
    # The tensor from the source rank is the one to be broadcast
    return broadcast_op(parameters[src_rank], axis_name="batch")
