import logging
from typing import Any
from typing import List

import jax
import jax.lax as lax
import jax.numpy as jnp
import optax

import keras
from keras.src.backend.distributed.base import BaseDistributedBackend

logger = logging.getLogger(__name__)


class JaxDistributedBackend(BaseDistributedBackend):
    """JAX-specific implementation of distributed operations."""

    def get_tensor_lib(self):
        return jnp

    def convert_to_backend_tensor(self, tensor: Any) -> Any:
        if isinstance(tensor, jax.Array):
            return tensor
        return jnp.array(tensor)

    def compute_gradients(
        self, loss: Any, trainable_vars: List[Any]
    ) -> List[Any]:
        """
        JAX backend doesn't support gradient computation with pre-computed loss.

        This method returns zero gradients as a fallback. For JAX, gradient
        computation must be done via `jax.grad` on a function that computes
        the loss from the parameters, which requires a different architecture.
        """
        logger.warning(
            "JAX backend `compute_gradients` is a fallback and returns "
            "zero gradients. A functional `jax.grad` approach should be used "
            "for training."
        )
        return [jnp.zeros_like(var) for var in trainable_vars]

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
                    logger.warning(
                        "Applying gradients to a standard JAX array has no "
                        "effect as JAX arrays are immutable. This operation "
                        "only works for mutable objects with an `.assign()` "
                        "method."
                    )

    def create_optimizer(self, optimizer_class: str, **kwargs):
        if optimizer_class.lower() == "adam":
            return optax.adam(**kwargs)
        elif optimizer_class.lower() == "sgd":
            return optax.sgd(**kwargs)
        else:
            kwargs.setdefault("learning_rate", 0.001)
            return optax.adam(**kwargs)

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
        try:
            if not self.is_multi_device_capable():
                raise RuntimeError("JAX is not running on multiple devices.")

            logger.info("Using real JAX collective communication ops.")

            def all_reduce_jax(x, op="sum", axis_name="data"):
                if op == "sum":
                    return lax.psum(x, axis_name=axis_name)
                elif op == "mean":
                    return lax.pmean(x, axis_name=axis_name)
                raise ValueError(f"Unsupported all_reduce op: {op}")

            def all_gather_jax(x, axis=0, axis_name="model"):
                return lax.all_gather(x, axis_name=axis_name, axis=axis)

            def broadcast_jax(x, root=0, axis_name="data"):
                return lax.all_gather(x, axis_name=axis_name, axis=0)[root]

            def scatter_jax(x, root=0):
                logger.warning(
                    "Scatter is not a native op in JAX pmap; returning the "
                    "input tensor as a fallback."
                )
                return x

            return {
                "all_reduce": all_reduce_jax,
                "all_gather": all_gather_jax,
                "broadcast": broadcast_jax,
                "scatter": scatter_jax,
            }
        except (ImportError, RuntimeError) as e:
            logger.warning(
                "JAX collective ops not available or multiple devices not "
                f"configured: {e}. Using SIMULATED ops."
            )

            device_info = self.get_device_info()
            simulated_world_size = device_info.get("device_count", 1)
            if simulated_world_size == 0:
                simulated_world_size = 1

            logger.info(
                f"Simulating with world_size={simulated_world_size} "
                "based on available devices."
            )

            def all_reduce_simulated(x, op="sum"):
                if simulated_world_size <= 1:
                    return x
                if op == "sum":
                    return keras.ops.multiply(x, simulated_world_size)
                elif op == "mean":
                    return x
                else:
                    raise ValueError(f"Unsupported all_reduce op: {op}")

            def all_gather_simulated(x, axis=0):
                if simulated_world_size <= 1:
                    return x
                return keras.ops.concatenate(
                    [x] * simulated_world_size, axis=axis
                )

            def broadcast_simulated(x, root=0):
                return x

            def scatter_simulated(x, root=0):
                if simulated_world_size <= 1:
                    return x
                if keras.ops.shape(x)[0] % simulated_world_size != 0:
                    raise ValueError(
                        "For simulation, the first dimension of tensor must "
                        f"be divisible by the simulated world size "
                        f"({simulated_world_size})."
                    )
                chunks = keras.ops.split(x, simulated_world_size, axis=0)
                return chunks[0]

            return {
                "all_reduce": all_reduce_simulated,
                "all_gather": all_gather_simulated,
                "broadcast": broadcast_simulated,
                "scatter": scatter_simulated,
            }