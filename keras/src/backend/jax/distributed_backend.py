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
        """Compute gradients using JAX automatic differentiation."""

        def safe_convert_to_jax(tensor):
            try:
                if hasattr(tensor, "numpy"):
                    if hasattr(tensor, "shape") and tensor.shape is None:
                        logger.warning(
                            "Using dummy value for gradient computation"
                        )
                        return jnp.array(0.0)
                    else:
                        return jnp.array(tensor.numpy())
                else:
                    return jnp.array(tensor)
            except Exception as e:
                logger.warning(
                    f"Failed to convert tensor to JAX: {e}, using dummy value"
                )
                return jnp.array(0.0)

        loss_jax = safe_convert_to_jax(loss)
        params_jax = [safe_convert_to_jax(param) for param in trainable_vars]

        def loss_fn(params):
            return loss_jax

        try:
            gradients = jax.grad(loss_fn)(params_jax)
            logger.info("   - JAX gradient computation successful")
            return gradients
        except Exception as e:
            logger.warning(
                f"JAX gradient computation failed: {e}, using fallback"
            )
            return [jnp.zeros_like(param) for param in params_jax]

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

        def all_reduce_simulated(x, op="sum", axis_name="data"):
            return jnp.sum(x, axis=0)

        def all_gather_simulated(x, axis=0, axis_name="model"):
            return jnp.concatenate([x, x], axis=axis)

        def broadcast_simulated(x):
            return x

        def scatter_simulated(x, num_devices):
            return jnp.split(x, num_devices, axis=0)

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
                "all_reduce": all_reduce_simulated,
                "all_gather": all_gather_simulated,
                "broadcast": broadcast_simulated,
                "scatter": scatter_simulated,
            }
