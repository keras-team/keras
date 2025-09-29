from typing import Any
from typing import List

import jax
import jax.lax as lax
import jax.numpy as jnp
import optax

import keras
from keras.src.backend.distributed.base import DistributedBackend


class JaxDistributedBackend(DistributedBackend):
    """JAX-specific implementation of distributed operations."""

    def get_tensor_lib(self):
        return jnp

    def compute_gradients(
        self, loss: Any, trainable_vars: List[Any]
    ) -> List[Any]:
        """
        JAX backend doesn't support gradient computation with pre-computed loss.

        This method returns zero gradients as a fallback. For JAX, gradient
        computation must be done via `jax.grad` on a function that computes
        the loss from the parameters, which requires a different architecture.
        """
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
        except Exception:
            info["devices"] = ["cpu"]
            info["device_count"] = 1
        return info

    def is_multi_device_capable(self) -> bool:
        return self.get_device_info()["device_count"] > 1

    def get_communication_ops(self) -> dict:
        """
        Provides robust JAX communication ops that work both inside and
        outside a pmap context.
        """

        def all_reduce(x, op="sum", axis_name="data"):
            try:
                jax.lax.axis_index(axis_name)
                if op == "sum":
                    return lax.psum(x, axis_name=axis_name)
                elif op == "mean":
                    return lax.pmean(x, axis_name=axis_name)
                raise ValueError(f"Unsupported all_reduce op: {op}")
            except NameError:
                world_size = self.get_device_info()["device_count"]
                if world_size <= 1:
                    return x
                if op == "sum":
                    return keras.ops.multiply(x, world_size)
                elif op == "mean":
                    return x
                raise ValueError(f"Unsupported all_reduce op: {op}")

        def all_gather(x, axis=0, axis_name="data"):
            try:
                jax.lax.axis_index(axis_name)
                return lax.all_gather(x, axis_name=axis_name, axis=axis)
            except NameError:
                world_size = self.get_device_info()["device_count"]
                if world_size <= 1:
                    return x
                return keras.ops.concatenate([x] * world_size, axis=axis)

        def broadcast(x, root=0, axis_name="data"):
            try:
                jax.lax.axis_index(axis_name)
                return lax.all_gather(x, axis_name=axis_name, axis=0)[root]
            except NameError:
                return x

        def scatter(x, root=0, axis=0, axis_name="data"):
            try:
                jax.lax.axis_index(axis_name)
                full_tensor = lax.all_gather(x, axis_name=axis_name, axis=0)[
                    root
                ]
                device_id = lax.axis_index(axis_name=axis_name)
                num_devices = lax.psum(1, axis_name=axis_name)
                chunk_size = full_tensor.shape[axis] // num_devices
                start_index = device_id * chunk_size
                return lax.dynamic_slice_in_dim(
                    operand=full_tensor,
                    start_index=start_index,
                    slice_size=chunk_size,
                    axis=axis,
                )
            except NameError:
                world_size = self.get_device_info()["device_count"]
                if world_size <= 1:
                    return x
                chunks = keras.ops.split(x, world_size, axis=axis)
                return chunks[root]

        return {
            "all_reduce": all_reduce,
            "all_gather": all_gather,
            "broadcast": broadcast,
            "scatter": scatter,
        }
