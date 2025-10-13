import jax
import jax.lax as lax


def get_device_info():
    """Retrieves information about the available JAX devices.

    This function queries the JAX backend to identify the type and number
    of available computational devices (e.g., CPU, GPU, TPU).

    Returns:
        A dictionary containing the backend name ('jax'), a list of
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

    This is useful for determining if parallel computation strategies like
    `pmap` can be utilized.

    Returns:
        True if the local JAX environment has more than one device,
        False otherwise.
    """
    return jax.local_device_count() > 1


def get_communication_ops():
    """Provides a dictionary of JAX collective communication operations.

    These functions wrap JAX's low-level collective primitives (`lax`)
    and are designed to be called from within a parallel context, such as
    one created by `jax.pmap` or `jax.pjit`. They enable communication
    and data transfer between different devices.

    Returns:
        A dictionary mapping operation names (e.g., 'all_reduce') to their
        corresponding JAX implementation functions.
    """

    def all_reduce(x, op="sum", axis_name="data"):
        """Reduces a tensor across all devices along a mapped axis.

        For example, `all_reduce(t, op="sum")` will compute the element-wise
        sum of the tensor `t` from all devices and distribute the result
        back to every device.

        Args:
            x: The input JAX array (tensor) on the local device.
            op: The reduction operation to perform. Supported values are
                'sum' and 'mean'. Defaults to 'sum'.
            axis_name: The name of the mapped axis in the `pmap` context
                over which to communicate. Defaults to 'data'.

        Returns:
            The reduced JAX array, which is identical across all devices.
        """
        reduce_ops = {
            "sum": lax.psum,
            "mean": lax.pmean,
        }
        reduce_fn = reduce_ops.get(op)
        return reduce_fn(x, axis_name=axis_name)

    def all_gather(x, axis=0, axis_name="data"):
        """Gathers and concatenates tensors from all devices.

        Each device contributes its local tensor `x`. These tensors are
        concatenated along the specified `axis`, and the resulting larger
        tensor is distributed to all devices.

        Args:
            x: The input JAX array (tensor) on the local device.
            axis: The axis along which to concatenate the gathered tensors.
                Defaults to 0.
            axis_name: The name of the mapped axis in the `pmap` context
                over which to communicate. Defaults to 'data'.

        Returns:
            The gathered JAX array, which is identical across all devices.
        """
        return lax.all_gather(x, axis_name=axis_name, axis=axis)

    def broadcast(x, root=0, axis_name="data"):
        """Broadcasts a tensor from a single root device to all other devices.

        This operation is implemented by first gathering the tensor from all
        devices and then selecting the tensor from the specified `root` device.

        Args:
            x: The input JAX array (tensor) on the local device. The value from
                the `root` device will be used.
            root: The integer index of the device that holds the data to be
                broadcast. Defaults to 0.
            axis_name: The name of the mapped axis in the `pmap` context
                over which to communicate. Defaults to 'data'.

        Returns:
            The JAX array from the `root` device, now present on all devices.
        """
        return lax.all_gather(x, axis_name=axis_name, axis=0)[root]

    def scatter(x, root=0, axis=0, axis_name="data"):
        """Scatters a tensor from a root device to all devices.

        The tensor on the `root` device is split into chunks along the specified
        `axis`. Each device then receives one chunk. This assumes the tensor
        dimension is evenly divisible by the number of devices.

        Args:
            x: The input JAX array (tensor) on the `root` device.
            root: The integer index of the device holding the full tensor.
                Defaults to 0.
            axis: The axis along which to split the tensor for scattering.
                Defaults to 0.
            axis_name: The name of the mapped axis in the `pmap` context
                over which to communicate. Defaults to 'data'.

        Returns:
            A chunk of the original tensor on each respective device.
        """
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

    return {
        "all_reduce": all_reduce,
        "all_gather": all_gather,
        "broadcast": broadcast,
        "scatter": scatter,
    }
