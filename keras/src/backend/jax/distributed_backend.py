import logging
from typing import Any
from typing import List

import jax
import jax.lax as lax
import jax.numpy as jnp
import optax

from keras.src.backend.distributed.base import BaseDistributedBackend

logger = logging.getLogger(__name__)


class JaxDistributedBackend(BaseDistributedBackend):
    """JAX-specific implementation of distributed operations."""

    def get_tensor_lib(self):
        return jnp

    def convert_to_backend_tensor(self, tensor: Any) -> Any:
        if hasattr(tensor, "numpy"):
            return jnp.array(tensor.numpy())
        else:
            return jnp.array(tensor)

    def compute_gradients(
        self, loss: Any, trainable_vars: List[Any]
    ) -> List[Any]:
        logger.warning(
            "JAX `compute_gradients` is a placeholder. Gradient computation "
            "should be handled in the model's `train_step` using `jax.grad`."
        )
        params_jax = [self.convert_to_backend_tensor(v) for v in trainable_vars]
        return [jnp.zeros_like(p) for p in params_jax]

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

    def create_optimizer(self, optimizer_class: str, **kwargs):
        if optimizer_class.lower() == "adam":
            return optax.adam(**kwargs)
        elif optimizer_class.lower() == "sgd":
            return optax.sgd(**kwargs)
        else:
            return optax.adam(learning_rate=0.001)

    def get_device_info(self) -> dict:
        info = {"backend": "jax", "devices": [], "device_count": 0}
        try:
            info["devices"] = [str(d) for d in jax.devices()]
            info["device_count"] = jax.local_device_count()
        except Exception as e:
            logger.warning(f"Could not get device info for JAX: {e}")
            info["devices"] = ["cpu"]
            info["device_count"] = 1
        return info

    def is_multi_device_capable(self) -> bool:
        return self.get_device_info()["device_count"] > 1

    def get_communication_ops(self) -> dict:
        def all_reduce_jax(x, op="sum", axis_name="data"):
            if op == "sum":
                return lax.psum(x, axis_name=axis_name)
            elif op == "mean":
                return lax.pmean(x, axis_name=axis_name)
            raise ValueError(f"Unsupported all_reduce op: {op}")

        def all_gather_jax(x, axis=0, axis_name="model"):
            return lax.all_gather(x, axis_name=axis_name, axis=axis)

        def broadcast_jax(x, root=0, axis_name="data"):
            """Broadcasts the tensor from the root device to all others."""
            return lax.all_gather(x, axis_name=axis_name)[root]

        def scatter_jax(x, root=0):
            logger.warning("Scatter is not a native op in JAX pmap.")
            return x

        def no_op_simulated(x, **kwargs):
            return x

        def scatter_simulated(x, **kwargs):
            return x

        try:
            if jax.device_count() > 1:
                logger.info("Using real JAX collective communication ops.")
                return {
                    "all_reduce": all_reduce_jax,
                    "all_gather": all_gather_jax,
                    "broadcast": broadcast_jax,
                    "scatter": scatter_jax,
                }
            else:
                raise RuntimeError("Not running on multiple JAX devices.")
        except (ImportError, RuntimeError) as e:
            logger.warning(
                "JAX collective ops not available or multiple devices not "
                f"configured: {e}. Using SIMULATED ops."
            )
            return {
                "all_reduce": no_op_simulated,
                "all_gather": no_op_simulated,
                "broadcast": no_op_simulated,
                "scatter": scatter_simulated,
            }
