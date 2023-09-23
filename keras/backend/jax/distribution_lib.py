"""!!!DO NOT USE!!!

Distribution related class for JAX backend.

This is just a prototype and we might want to unify it
with other backends in the future.
"""
import jax
import numpy as np


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


def distribute_value(value, tensor_layout):
    """Distribute the value based on the layout.

    Args:
        value: `jax.Array` that need to be distributed.
        tensor_layout: `TensorLayout` for the distribution information, or a
            `jax.sharding.Sharding` instance.

    Returns:
        Distributed value.
    """
    if not isinstance(tensor_layout, jax.sharding.Sharding):
        tensor_layout = _to_jax_layout(tensor_layout)
    return jax.device_put(value, tensor_layout)


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
