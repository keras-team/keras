from typing import Any, Callable, Dict, Literal

import jax
import jax.lax as lax
import jax.numpy as jnp

def get_device_info() -> Dict[str, Any]:
    """Retrieves information about the available JAX devices."""
    available_devices = jax.devices()
    return {
        "backend": "jax",
        "devices": [str(d) for d in available_devices],
        "device_count": len(available_devices),
    }

def is_multi_device_capable() -> bool:
    """Checks if more than one JAX device is available."""
    return jax.local_device_count() > 1


def get_communication_ops() -> Dict[str, Callable]:
    """
    Provides a dictionary of JAX collective communication operations.

    Note: These operations are thin wrappers around `jax.lax` primitives
    and are intended to be used exclusively within a `jax.pmap` context.
    Calling them outside of `pmap` will result in an error.

    Returns:
        Dict[str, Callable]: A dictionary mapping operation names to their
        JAX implementations.
    """

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
            jnp.ndarray: The reduced tensor.
        """
        reduce_ops = {
            "sum": lax.psum,
            "mean": lax.pmean,
        }
        reduce_fn = reduce_ops.get(op)

        if reduce_fn is None:
            raise ValueError(f"Unsupported all_reduce op: {op}")
        return reduce_fn(x, axis_name=axis_name)

    def all_gather(
        x: jnp.ndarray, axis: int = 0, axis_name: str = "data"
    ) -> jnp.ndarray:
        """Gathers tensors from all devices and concatenates them.

        Args:
            x (jnp.ndarray): The local tensor to gather.
            axis (int, optional): The axis to concatenate along. Defaults to 0.
            axis_name (str, optional): The name of the `pmap` axis.
                Defaults to "data".

        Returns:
            jnp.ndarray: The concatenated tensor from all devices.
        """
        return lax.all_gather(x, axis_name=axis_name, axis=axis)

    def broadcast(
        x: jnp.ndarray, root: int = 0, axis_name: str = "data"
    ) -> jnp.ndarray:
        """Broadcasts a tensor from a root device to all other devices.

        This is implemented by gathering the tensor from all devices and then
        having each device select the tensor from the `root` device. It assumes
        the value of `x` on the `root` device is the one to be broadcast.

        Args:
            x (jnp.ndarray): The tensor to broadcast.
            root (int, optional): The rank of the source device. Defaults to 0.
            axis_name (str, optional): The name of the `pmap` axis.
                Defaults to "data".

        Returns:
            jnp.ndarray: The tensor received from the root device.
        """
        # A common JAX pattern for broadcast is to all-gather and then index.
        return lax.all_gather(x, axis_name=axis_name, axis=0)[root]

    def scatter(
        x: jnp.ndarray,
        root: int = 0,
        axis: int = 0,
        axis_name: str = "data",
    ) -> jnp.ndarray:
        """Scatters a tensor from a root device to all devices.

        Args:
            x (jnp.ndarray): On the root device, the full tensor to scatter.
            root (int, optional): The rank of the source device. Defaults to 0.
            axis (int, optional): The axis along which to split the tensor.
                Defaults to 0.
            axis_name (str, optional): The name of the `pmap` axis.
                Defaults to "data".

        Returns:
            jnp.ndarray: The chunk of the tensor for the local device.
        """
        # First, ensure all devices have the full tensor from the root.
        full_tensor = broadcast(x, root=root, axis_name=axis_name)

        # Then, each device calculates its own slice.
        device_id = lax.axis_index(axis_name=axis_name)
        num_devices = lax.psum(1, axis_name=axis_name)

        if full_tensor.shape[axis] % num_devices != 0:
            raise ValueError(
                f"Tensor with shape {x.shape} cannot be scattered along "
                f"axis {axis} across {num_devices} devices."
            )

        chunk_size = full_tensor.shape[axis] // num_devices
        start_index = device_id * chunk_size
        return lax.dynamic_slice_in_dim(
            operand=full_tensor,
            start_index=start_index,
            slice_size=chunk_size,
            axis=axis,
        )

    return {
        "all_reduce": all_reduce,
        "all_gather": all_gather,
        "broadcast": broadcast,
        "scatter": scatter,
    }