from typing import Any
from typing import List
from typing import Tuple

from keras.src.backend.distributed import backend_resolver
from keras.src.backend.distributed.base import DistributedBackend


class CollectiveOpKeras:
    def __init__(self, world_size: int, rank: int = 0):
        self.world_size = world_size
        self.rank = rank

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class AllReduceKeras(CollectiveOpKeras):
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
        return self.all_reduce_fn(local_tensor, op=self.op, axis_name=axis_name)


class AllGatherKeras(CollectiveOpKeras):
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
        return self.all_gather_fn(
            local_tensor, axis=self.dim, axis_name=axis_name
        )


class BroadcastKeras(CollectiveOpKeras):
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
        return self.broadcast_fn(
            tensor, root=self.src_rank, axis_name=axis_name
        )


class TensorParallelCommunicator:
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
    ):
        self.allgather.dim = dim
        return self.allgather(local_tensor, axis_name=axis_name)

    def backward_column_parallel(
        self, local_gradient: Any, op: str = "sum", axis_name: str = "i"
    ):
        self.allreduce.op = op
        return self.allreduce(local_gradient, axis_name=axis_name)

    def forward_row_parallel(
        self, local_output: Any, op: str = "sum", axis_name: str = "i"
    ):
        self.allreduce.op = op
        return self.allreduce(local_output, axis_name=axis_name)

    def backward_row_parallel(
        self, local_gradient: Any, dim: int = -1, axis_name: str = "i"
    ):
        self.allgather.dim = dim
        return self.allgather(local_gradient, axis_name=axis_name)

    def handle_mlp_handshake(
        self, up_projection_outputs: List, down_projection_inputs: List
    ) -> Tuple:
        up_output = self.forward_column_parallel(
            up_projection_outputs[self.rank], dim=-1
        )
        down_inputs = self.forward_row_parallel(
            down_projection_inputs[self.rank], op="sum"
        )
        return up_output, down_inputs

    def slice_upstream_gradient_for_column_parallel(
        self, full_gradient, rank: int, world_size: int, dim: int = -1
    ):
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
            return full_gradient

    def slice_upstream_gradient_for_row_parallel(
        self, full_gradient, rank: int, world_size: int, dim: int = 0
    ):
        try:
            total_size = full_gradient.shape[dim]
            slice_size = total_size // world_size
            start_idx = rank * slice_size
            end_idx = (rank + 1) * slice_size
            if rank == world_size - 1:
                end_idx = total_size
            slices = [slice(None)] * len(full_gradient.shape)
            slices[dim] = slice(start_idx, end_idx)
            return full_gradient[tuple(slices)]
        except Exception:
            return full_gradient


def allreduce_gradients(
    gradients: List, world_size: int, backend: DistributedBackend
) -> List:
    allreduce_op = AllReduceKeras(world_size, backend=backend, op="mean")
    local_gradient = gradients[0] if isinstance(gradients, list) else gradients
    return allreduce_op(local_gradient)


def allgather_outputs(
    outputs: List,
    world_size: int,
    backend: DistributedBackend,
    dim: int = -1,
):
    allgather_op = AllGatherKeras(world_size, backend=backend, dim=dim)
    local_output = outputs[0] if isinstance(outputs, list) else outputs
    return allgather_op(local_output)


def broadcast_parameters(
    parameters: List,
    world_size: int,
    backend: DistributedBackend,
    src_rank: int = 0,
) -> List:
    broadcast_op = BroadcastKeras(
        world_size, backend=backend, src_rank=src_rank
    )
    return broadcast_op(parameters[src_rank])
