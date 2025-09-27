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
        epsilon = 1e-7
        gradients = []
        for var in trainable_vars:
            if hasattr(var, "shape"):
                grad = np.zeros_like(var)
                it = np.nditer(
                    var, flags=["multi_index"], op_flags=["readwrite"]
                )
                while not it.finished:
                    idx = it.multi_index
                    original_value = var[idx]
                    var[idx] = original_value + epsilon
                    # This part is flawed as loss is a scalar.
                    # Numerical differentiation needs a function to re-evaluate.
                    # This is a placeholder for a no-op.
                    loss_plus = loss
                    var[idx] = original_value - epsilon
                    loss_minus = loss
                    grad[idx] = (loss_plus - loss_minus) / (
                        2 * epsilon
                    )  # Will be 0
                    var[idx] = original_value  # Restore
                    it.iternext()
                gradients.append(grad)
            else:
                gradients.append(0.0)
        return gradients

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
                        var -= self.learning_rate * grad

        return NumpyOptimizer(**kwargs)

    def get_device_info(self) -> dict:
        return {"backend": "numpy", "devices": ["cpu"], "device_count": 1}

    def is_multi_device_capable(self) -> bool:
        return False

    def get_communication_ops(self) -> dict:
        logger.info("Using SIMULATED NumPy communication ops.")

        def all_reduce_np(x, op="sum"):
            return keras.ops.sum(x, axis=0)

        def all_gather_np(x, axis=0):
            return keras.ops.concatenate([x, x], axis=axis)

        def broadcast_np(x):
            return x

        def scatter_np(x, num_devices):
            return keras.ops.split(x, num_devices, axis=0)

        return {
            "all_reduce": all_reduce_np,
            "all_gather": all_gather_np,
            "broadcast": broadcast_np,
            "scatter": scatter_np,
        }
