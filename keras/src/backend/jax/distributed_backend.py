from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal

import jax
import jax.lax as lax
import jax.numpy as jnp

import keras


def compute_gradients(
    _loss: jnp.ndarray, trainable_vars: List[jnp.ndarray]
) -> List[jnp.ndarray]:
    """Computes gradients of the loss with respect to trainable variables.

    Note: This is a placeholder implementation that returns zeros. A real
    implementation would use `jax.grad`.

    Args:
        _loss (jnp.ndarray): The loss value for which to compute gradients.
        trainable_vars (List[jnp.ndarray]): A list of variables to compute
            gradients with respect to.

    Returns:
        List[jnp.ndarray]: A list of gradients corresponding to the
        trainable variables.
    """
    return [jnp.zeros_like(var) for var in trainable_vars]


def apply_gradients(
    gradients: List[jnp.ndarray],
    trainable_vars: List[jnp.ndarray],
    learning_rate: float = 0.001,
) -> None:
    """Applies gradients to trainable variables using basic SGD.

    Args:
        gradients (List[jnp.ndarray]): A list of gradients.
        trainable_vars (List[jnp.ndarray]): A list of variables to be updated.
        learning_rate (float, optional): The learning rate for the update step.
            Defaults to 0.001.
    """
    for grad, var in zip(gradients, trainable_vars):
        if grad is not None:
            new_value = var - (learning_rate * grad)
            if hasattr(var, "assign"):
                var.assign(new_value)


def create_optimizer(optimizer_class: str, **kwargs) -> Dict[str, Any]:
    """Creates a configuration dictionary for an optimizer.

    This function returns a dictionary containing the optimizer's configuration,
    removing the need for a specific optimizer library like Optax.

    Args:
        optimizer_class (str): The name of the optimizer to create (e.g.,
            `"adam"`, `"sgd"`).
        **kwargs: Keyword arguments to be passed to the optimizer's
            constructor (e.g., `learning_rate`).

    Returns:
        Dict[str, Any]: A dictionary representing the optimizer configuration.
    """
    config = kwargs.copy()
    config["name"] = optimizer_class.lower()
    config.setdefault("learning_rate", 0.001)
    return config


def get_device_info() -> Dict[str, Any]:
    """Retrieves information about the available JAX devices.

    Returns:
        Dict[str, Any]: A dictionary containing the backend name, a list of
        available device strings, and the total device count.
    """
    available_devices = jax.devices()
    return {
        "backend": "jax",
        "devices": [str(d) for d in available_devices],
        "device_count": len(available_devices),
    }


def is_multi_device_capable() -> bool:
    """Checks if more than one JAX device is available.

    Returns:
        bool: `True` if JAX reports more than one local device, `False`
        otherwise.
    """
    return jax.local_device_count() > 1


def get_communication_ops() -> Dict[str, Callable]:
    """Provides a dictionary of JAX collective communication operations.

    These operations are designed to work within a `jax.pmap` context for
    multi-device computation. If not in a `pmap` context, they generally
    behave as no-ops or simulate the operation on the single local device.

    Returns:
        Dict[str, Callable]: A dictionary mapping operation names to their
        JAX implementations.
    """

    def _is_in_pmap(axis_name: str = "data") -> bool:
        """Checks if currently inside a pmap by probing the axis name."""
        try:
            lax.axis_index(axis_name)
            return True
        except NameError:
            return False

    def all_reduce(
        x: jnp.ndarray,
        op: Literal["sum", "mean"] = "sum",
        axis_name: str = "data",
    ) -> jnp.ndarray:
        """Reduces a tensor across all devices in a `pmap`.

        Args:
            x (jnp.ndarray): The tensor to reduce.
            op (Literal["sum", "mean"], optional): The reduction operation.
                Defaults to "sum".
            axis_name (str, optional): The name of the `pmap` axis.
                Defaults to "data".

        Returns:
            jnp.ndarray: The reduced tensor. Returns the input tensor `x` if
            not in a `pmap` context.
        """
        if _is_in_pmap(axis_name):
            if op == "sum":
                return lax.psum(x, axis_name=axis_name)
            elif op == "mean":
                return lax.pmean(x, axis_name=axis_name)
            raise ValueError(f"Unsupported all_reduce op: {op}")
        else:
            return x

    def all_gather(
        x: jnp.ndarray, axis: int = 0, axis_name: str = "data"
    ) -> jnp.ndarray:
        """Gathers tensors from all devices and concatenates them.

        Args:
            x (jnp.ndarray): The local tensor to gather.
            axis (int, optional): The axis along which to concatenate the
                gathered tensors. Defaults to 0.
            axis_name (str, optional): The name of the `pmap` axis.
                Defaults to "data".

        Returns:
            jnp.ndarray: The concatenated tensor from all devices.
        """
        if _is_in_pmap(axis_name):
            return lax.all_gather(x, axis_name=axis_name, axis=axis)
        else:
            world_size = jax.local_device_count()
            if world_size <= 1:
                return x
            return keras.ops.concatenate([x] * world_size, axis=axis)

    def broadcast(
        x: jnp.ndarray, root: int = 0, axis_name: str = "data"
    ) -> jnp.ndarray:
        """Broadcasts a tensor from a root device to all other devices.

        Args:
            x (jnp.ndarray): The tensor to broadcast. On the root device, this
                is the tensor to be sent.
            root (int, optional): The rank of the device from which to
                broadcast. Defaults to 0.
            axis_name (str, optional): The name of the `pmap` axis.
                Defaults to "data".

        Returns:
            jnp.ndarray: The tensor received from the root device.
        """
        if _is_in_pmap(axis_name):
            return lax.all_gather(x, axis_name=axis_name, axis=0)[root]
        else:
            return x

    def scatter(
        x: jnp.ndarray,
        root: int = 0,
        axis: int = 0,
        axis_name: str = "data",
    ) -> jnp.ndarray:
        """Scatters a tensor from a root device to all devices.

        Args:
            x (jnp.ndarray): The tensor on the root device to be scattered.
            root (int, optional): The rank of the device that holds the full
                tensor. Defaults to 0.
            axis (int, optional): The axis along which to split the tensor.
                Defaults to 0.
            axis_name (str, optional): The name of the `pmap` axis.
                Defaults to "data".

        Returns:
            jnp.ndarray: The chunk of the tensor for the local device.
        """
        if _is_in_pmap(axis_name):
            full_tensor = lax.all_gather(x, axis_name=axis_name, axis=0)[root]
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
        else:
            world_size = jax.local_device_count()
            if world_size <= 1:
                return x
            if x.shape[axis] % world_size != 0:
                raise ValueError(
                    f"Tensor with shape {x.shape} cannot be scattered along "
                    f"axis {axis} across {world_size} devices."
                )
            chunks = keras.ops.split(x, world_size, axis=axis)
            return chunks[0]

    return {
        "all_reduce": all_reduce,
        "all_gather": all_gather,
        "broadcast": broadcast,
        "scatter": scatter,
    }
