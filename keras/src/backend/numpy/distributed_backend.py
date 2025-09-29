import logging
from typing import Any
from typing import List

import numpy as np

import keras
from keras.src.backend.distributed.base import BaseDistributedBackend

logger = logging.getLogger(__name__)


class NumpyDistributedBackend(BaseDistributedBackend):
    """NumPy-based fallback implementation of distributed operations."""

    def get_tensor_lib(self):
        return np

    def convert_to_backend_tensor(self, tensor: Any) -> Any:
        return keras.ops.convert_to_numpy(tensor)

    def compute_gradients(
        self, loss: Any, trainable_vars: List[Any]
    ) -> List[Any]:
        """
        NumPy backend does not support automatic differentiation.

        This method returns zero gradients as a fallback. In a real workflow,
        gradients would need to be computed manually or by a different backend.
        """
        logger.warning(
            "NumPy backend does not support automatic differentiation. "
            "Returning zero gradients as a fallback."
        )
        return [np.zeros_like(var) for var in trainable_vars]

    def apply_gradients(
        self,
        gradients: List[Any],
        trainable_vars: List[Any],
        learning_rate: float = 0.001,
    ) -> None:
        for grad, var in zip(gradients, trainable_vars):
            if grad is not None:
                new_value = var - (learning_rate * grad)
                if hasattr(var, "assign"):
                    var.assign(new_value)
                else:
                    var[:] = new_value

    def create_optimizer(self, optimizer_class: str, **kwargs):
        class NumpyOptimizer:
            def __init__(self, learning_rate=0.001):
                self.learning_rate = learning_rate

            def apply_gradients(self, grads_and_vars):
                for grad, var in grads_and_vars:
                    if grad is not None:
                        if isinstance(var, np.ndarray):
                            var -= self.learning_rate * grad
                        else:
                            var.assign(var.value - self.learning_rate * grad)

        return NumpyOptimizer(**kwargs)

    def get_device_info(self) -> dict:
        return {"backend": "numpy", "devices": ["cpu"], "device_count": 1}

    def is_multi_device_capable(self) -> bool:
        return False

    def get_communication_ops(self) -> dict:
        device_info = self.get_device_info()
        world_size = device_info.get("device_count", 1)
        if world_size == 0:
            world_size = 1

        logger.info(
            "Using SIMULATED NumPy communication ops. "
            f"Simulating with world_size={world_size} "
            "based on available devices."
        )

        def all_reduce_np(x, op="sum"):
            if op == "sum":
                return keras.ops.sum(x, axis=0)
            elif op == "mean":
                return keras.ops.mean(x, axis=0)
            else:
                raise ValueError(f"Unsupported all_reduce op: {op}")

        def all_gather_np(x, axis=0):
            if world_size <= 1:
                return x
            return keras.ops.concatenate([x] * world_size, axis=axis)

        def broadcast_np(x, root=0):
            return x

        def scatter_np(x, root=0):
            if world_size <= 1:
                return x
            if keras.ops.shape(x)[0] % world_size != 0:
                raise ValueError(
                    "For simulation, the first dimension of the tensor must "
                    f"be divisible by the simulated world size ({world_size})."
                )
            chunks = keras.ops.split(x, world_size, axis=0)
            return chunks[0]

        return {
            "all_reduce": all_reduce_np,
            "all_gather": all_gather_np,
            "broadcast": broadcast_np,
            "scatter": scatter_np,
        }
