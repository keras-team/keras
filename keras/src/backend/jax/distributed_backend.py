import jax
import jax.lax as lax


def get_device_info():
    """Retrieves information about the available JAX devices.

    This function queries the JAX backend to identify the type and number
    of available computational devices (e.g., CPU, GPU, TPU).

    Returns:
        dict: A dictionary containing the backend name ('jax'), a list of
        device string representations, and the total count of devices.
    """
    available_devices = jax.devices()
    return {
        "backend": "jax",
        "devices": [str(d) for d in available_devices],
        "device_count": len(available_devices),
    }


def is_multi_device_capable():
    """Checks if more than one JAX device is available for computation.

    Returns:
        bool: True if the local JAX environment has more than one device,
        False otherwise.
    """
    return jax.local_device_count() > 1


def get_communication_ops():
    """Provides a dictionary of JAX collective communication operations.

    Returns:
        dict: A dictionary mapping operation names (e.g., 'all_reduce') to their
        corresponding JAX implementation functions.
    """

    def all_reduce(x, op="sum", axis_name="model"):
        """Reduces a tensor across a device mesh axis using a collective.

        This function assumes it is called within a `pjit` context that has a
        device mesh with the specified `axis_name`. It performs a collective
        reduction operation (like sum or mean) across all devices mapped to
        that axis.

        Args:
            x (jax.Array): The input JAX array (tensor) on the local device.
            op (str, optional): The reduction operation to perform. Supported
                values are 'sum' and 'mean'. Defaults to 'sum'.
            axis_name (str, optional): The name of the mapped axis in the device
                mesh over which to communicate. Defaults to 'model'.

        Returns:
            jax.Array: The reduced JAX array, which is identical across all
            devices participating in the reduction.
        """
        if op == "sum":
            return lax.psum(x, axis_name=axis_name)
        elif op == "mean":
            return lax.pmean(x, axis_name=axis_name)
        else:
            raise ValueError(
                f"Unsupported reduction operation: {op}. "
                "Supported options are 'sum' and 'mean'."
            )

    def all_gather(x, axis, axis_name="model"):
        """Gathers and concatenates tensors from all devices across a mesh axis.

        This function assumes it is called within a `pjit` context. It takes
        the local shard `x` from each device along the `axis_name` of the mesh
        and concatenates them along the specified tensor `axis` to form a
        single, larger tensor that is then replicated on participating devices.

        Args:
            x (jax.Array): The input JAX array (tensor) shard on local device.
            axis (int): The tensor axis along which to concatenate the gathered
                shards.
            axis_name (str, optional): The name of the mesh axis to gather
                from. Defaults to 'model'.

        Returns:
            jax.Array: The full, gathered JAX array, which is identical across
            all devices participating in the gather.
        """
        return lax.all_gather(x, axis_name=axis_name, axis=axis, tiled=True)

    return {
        "all_reduce": all_reduce,
        "all_gather": all_gather,
    }
