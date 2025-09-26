import logging
from typing import Any
from typing import List

import torch
import torch.distributed as dist

from keras.src.backend.distributed.base import BaseDistributedBackend

logger = logging.getLogger(__name__)


class PytorchDistributedBackend(BaseDistributedBackend):
    """PyTorch-specific implementation of distributed operations."""

    def get_tensor_lib(self):
        return torch

    def convert_to_backend_tensor(self, tensor: Any) -> Any:
        return torch.as_tensor(tensor)

    def compute_gradients(
        self, loss: Any, trainable_vars: List[Any]
    ) -> List[Any]:
        logger.warning(
            "PyTorch gradient computation is handled by `loss.backward()` in "
            "the Keras model's `train_step`. This is a placeholder."
        )
        return [torch.zeros_like(var) for var in trainable_vars]

    def apply_gradients(
        self,
        gradients: List[Any],
        trainable_vars: List[Any],
        learning_rate: float = 0.001,
    ) -> None:
        for grad, var in zip(gradients, trainable_vars):
            if grad is not None:
                with torch.no_grad():
                    var.sub_(grad * learning_rate)

    def create_optimizer(self, optimizer_class: str, **kwargs):
        if optimizer_class.lower() == "adam":
            return torch.optim.Adam(**kwargs)
        elif optimizer_class.lower() == "sgd":
            return torch.optim.SGD(**kwargs)
        else:
            return torch.optim.Adam(lr=0.001)

    def get_device_info(self) -> dict:
        info = {"backend": "pytorch", "devices": [], "device_count": 0}
        try:
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                info["devices"] = [f"cuda:{i}" for i in range(count)]
                info["device_count"] = count
            else:
                info["devices"] = ["cpu"]
                info["device_count"] = 1
        except Exception as e:
            logger.warning(f"Could not get device info for PyTorch: {e}")
            info["devices"] = ["cpu"]
            info["device_count"] = 1
        return info

    def is_multi_device_capable(self) -> bool:
        return self.get_device_info()["device_count"] > 1

    def get_communication_ops(self) -> dict:
        def all_reduce_torch(x, op="sum"):
            if op == "sum":
                dist.all_reduce(x, op=dist.ReduceOp.SUM)
            elif op == "mean":
                dist.all_reduce(x, op=dist.ReduceOp.SUM)
                x /= dist.get_world_size()
            else:
                raise ValueError(f"Unsupported all_reduce op: {op}")
            return x

        def all_gather_torch(x, axis=0):
            world_size = dist.get_world_size()
            tensor_list = [torch.empty_like(x) for _ in range(world_size)]
            dist.all_gather(tensor_list, x)
            return torch.cat(tensor_list, dim=axis)

        def broadcast_torch(x, root=0):
            dist.broadcast(x, src=root)
            return x

        def scatter_torch(x, root=0):
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            if rank == root:
                if x.shape[0] % world_size != 0:
                    raise ValueError(
                        "The first dimension of the tensor must be divisible "
                        "by world size."
                    )
                scatter_list = list(torch.chunk(x, world_size, dim=0))
            else:
                scatter_list = None
            chunk_shape = (x.shape[0] // world_size,) + x.shape[1:]
            output_tensor = torch.empty(
                chunk_shape, dtype=x.dtype, device=x.device
            )
            dist.scatter(output_tensor, scatter_list, src=root)
            return output_tensor

        try:
            if not (dist.is_available() and dist.is_initialized()):
                raise RuntimeError(
                    "torch.distributed is not available or not initialized."
                )
            logger.info("Using real torch.distributed communication ops.")
            return {
                "all_reduce": all_reduce_torch,
                "all_gather": all_gather_torch,
                "broadcast": broadcast_torch,
                "scatter": scatter_torch,
            }
        except (ImportError, RuntimeError) as e:
            logger.warning(
                f"torch.distributed not available: {e}. Using SIMULATED ops."
            )

            def all_reduce_simulated(x, op="sum"):
                return x

            def all_gather_simulated(x, axis=0):
                return torch.cat([x, x], dim=axis)

            def broadcast_simulated(x, root=0):
                return x

            def scatter_simulated(x, root=0):
                return x

            return {
                "all_reduce": all_reduce_simulated,
                "all_gather": all_gather_simulated,
                "broadcast": broadcast_simulated,
                "scatter": scatter_simulated,
            }
