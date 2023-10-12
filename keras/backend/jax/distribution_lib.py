"""!!!DO NOT USE!!!

Distribution related class for JAX backend.

This is just a prototype and we might want to unify it
with other backends in the future.
"""
import jax
import numpy as np

from keras.utils import jax_utils


def list_devices(device_type=None):
    """Return all the available devices based on the device type.

    Note that this should return the global devices in a distributed setting.

    Args:
        device_type: string of `"cpu"`, `"gpu"` or `"tpu"`. Defaults to `"gpu"`
            or `"tpu"` if available when device_type is not provided. Otherwise
            will return the `"cpu"` devices.

    Return:
        List of devices that are available for distribute computation.
    """
    device_type = device_type.lower() if device_type else None
    jax_devices = jax.devices(backend=device_type)
    return [f"{device.device_kind}:{device.id}" for device in jax_devices]


def distribute_variable(value, layout):
    """Create a distributed variable for JAX.

    Since JAX doesn't have a variable class, this will just return a `jax.Array`
    with the corresponding layout/sharding specified.

    Note that this function should be used in eager context, not in jitted
    function.

    Args:
        value: the initial value of the variable.
        layout: `TensorLayout` for the created variable, or a
            `jax.sharding.Sharding` instance.

    Returns:
        jax.Array which is the distributed variable.
    """
    if not isinstance(layout, jax.sharding.Sharding):
        layout = _to_jax_layout(layout)
    return jax.device_put(value, layout)


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout.

    Note that this function can be used both in eager context, or within a
    jitted function.

    Args:
        tensor: `jax.Array` that need to be distributed.
        layout: `TensorLayout` for the distribution information, or a
            `jax.sharding.Sharding` instance.

    Returns:
        Distributed value.
    """
    if not isinstance(layout, jax.sharding.Sharding):
        layout = _to_jax_layout(layout)
    # TODO(scottzhu): This might not be a cheap check, we should consider
    # have some proper JAX API for doing this check.
    if jax_utils.is_in_jax_tracing_scope():
        return jax.lax.with_sharding_constraint(tensor, layout)

    if layout.is_fully_addressable:
        return jax.device_put(tensor, layout)
    else:
        # Need to only distribute the value to local addressible devices, and
        # repack them back into global format.
        mapping = layout.addressable_devices_indices_map(tensor.shape)
        local_values = jax.device_put(
            [tensor[i] for i in mapping.values()], list(mapping.keys())
        )
        global_value = jax.make_array_from_single_device_arrays(
            tensor.shape, layout, local_values
        )
        return global_value


def initialize(job_addresses, num_processes, process_id):
    if job_addresses and "," in job_addresses:
        # When user provide all the job addresses, we will split and get the
        # first one, which is the coordinator.
        job_addresses = job_addresses.split(",")
        # Do a sanity check to make sure the number of addresses also match
        # the num_processes.
        if num_processes is not None and num_processes != len(job_addresses):
            raise ValueError(
                f"The provided job_addresses {job_addresses} has "
                f"{len(job_addresses)} jobs, but num_processes is "
                f"{num_processes}"
            )
        corrdinator_address = job_addresses[0]
    else:
        corrdinator_address = job_addresses

    jax.distributed.initialize(
        corrdinator_address=corrdinator_address,
        num_processes=num_processes,
        process_id=process_id,
    )


def _to_jax_device(device_id):
    device_type, index = device_id.split(":")
    index = int(index)
    devices = jax.devices(backend=device_type)
    if index >= len(devices):
        raise ValueError(f"Unknown device: {device_id}")
    return devices[index]


def _to_jax_mesh(device_mesh):
    """Convert the DeviceMesh to JAX backend specific Mesh.

    Args:
        device_mesh: DeviceMesh instance to convert.

    Returns:
        A `jax.sharding.Mesh` instance.
    """
    shape = device_mesh.devices.shape
    devices = [_to_jax_device(d) for d in device_mesh.devices.flatten()]
    devices = np.array(devices).reshape(shape)
    return jax.sharding.Mesh(devices, device_mesh.axis_names)


def _to_jax_layout(tensor_layout):
    """Convert the TensorLayout to JAX backend specific Sharding.

    Args:
        tensor_layout: TensorLayout instance to convert.

    Returns:
        A `jax.sharding.NamedSharding` instance.
    """
    if tensor_layout.device_mesh is None:
        raise ValueError(
            "Cannot create sharding when device mesh is not set "
            "for TensorLayout."
        )
    partition_spec = jax.sharding.PartitionSpec(*tensor_layout.axes)
    jax_mesh = _to_jax_mesh(tensor_layout.device_mesh)
    return jax.sharding.NamedSharding(jax_mesh, partition_spec)
