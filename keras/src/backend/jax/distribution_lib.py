"""Utilities for distribution strategy with JAX backend.

This file contains the core JAX distribution primitives from Keras,
along with higher-level device management and auto-configuration utilities.
This version does not use try-except blocks for error handling.
"""

import logging
from typing import Dict
from typing import List
from typing import Optional

import jax
import numpy as np

from keras.src.backend.common import global_state
from keras.src.random import seed_generator
from keras.src.utils import jax_utils
from keras.src.utils import rng_utils

logger = logging.getLogger(__name__)


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
    return [f"{device.platform}:{device.id}" for device in jax_devices]


def get_device_info(device_id: str) -> Dict[str, any]:
    """
    Get detailed information about a specific device.

    Args:
        device_id: Device identifier (e.g., 'gpu:0', 'tpu:0', 'cpu:0')

    Returns:
        Dictionary containing device information
    """
    device_info = {
        "id": device_id,
        "type": None,
        "index": None,
        "memory": None,
        "capabilities": None,
    }

    device_type, device_index = device_id.split(":")
    device_info["type"] = device_type.upper()
    device_info["index"] = int(device_index)

    return device_info


def get_best_devices(count: int = 1) -> List[str]:
    """
    Get the best available devices for tensor parallelism.

    Args:
        count: Number of devices needed

    Returns:
        List of best device identifiers
    """
    all_devices = list_devices()

    if count <= 0:
        return []

    if count > len(all_devices):
        logger.warning(
            f"Requested {count} devices but only {len(all_devices)} available"
        )
        count = len(all_devices)

    return all_devices[:count]


def get_device_backend(device_type: str) -> str:
    """
    Get the recommended backend for a device type.

    Args:
        device_type: Device type ('tpu', 'gpu', 'cpu')

    Returns:
        Recommended backend name
    """
    backend_mapping = {"tpu": "jax", "gpu": "jax", "cpu": "jax"}

    return backend_mapping.get(device_type.lower(), "jax")


def validate_device_placement(device_id: str) -> bool:
    """
    Validate if a device can be used for tensor operations.

    Args:
        device_id: Device identifier

    Returns:
        True if device is valid and available
    """
    all_devices = list_devices()
    return device_id in all_devices


def get_device_memory_info(device_id: str) -> Optional[Dict[str, any]]:
    """
    Get memory information for a device (if available).

    Args:
        device_id: Device identifier

    Returns:
        Memory information dictionary or None if not available
    """
    if device_id.startswith("gpu:"):
        return {
            "type": "GPU",
            "index": int(device_id.split(":")[1]),
            "memory": "Available",
        }
    elif device_id.startswith("tpu:"):
        return {
            "type": "TPU",
            "index": int(device_id.split(":")[1]),
            "memory": "TPU Memory",
        }
    elif device_id.startswith("cpu:"):
        return {
            "type": "CPU",
            "index": int(device_id.split(":")[1]),
            "memory": "System RAM",
        }

    return None


def auto_configure_tensor_parallel(
    world_size: int = None, backend: str = None
) -> Dict[str, any]:
    """
    Automatically configure tensor parallelism with the best available devices.

    Args:
        world_size: Number of devices to use (if None, uses all available)
        backend: Backend to use (if None, will be set to 'jax')

    Returns:
        Configuration dictionary with devices, backend, and other settings
    """
    all_devices = list_devices()

    if not all_devices:
        raise RuntimeError("No devices available for tensor parallelism")

    if world_size is None:
        world_size = len(all_devices)
    else:
        world_size = min(world_size, len(all_devices))

    selected_devices = all_devices[:world_size]

    recommended_backend = "jax"

    config = {
        "devices": selected_devices,
        "world_size": world_size,
        "backend": recommended_backend,
    }

    logger.info(f"Auto-configured tensor parallelism: {config}")
    return config


def distribute_variable(value, layout):
    """Create a distributed variable for JAX.

    Since JAX doesn't have a variable class, this will just return a `jax.Array`
    with the corresponding layout/sharding specified.

    Note that this function should be used in eager context, not in jitted
    function.

    Args:
        value: the initial value of the variable.
        layout: `TensorLayout` for the created variable, or a
            JAX-supported layout instance (e.g. `jax.sharding.Sharding`).

    Returns:
        jax.Array which is the distributed variable.
    """
    return distribute_tensor(value, layout)


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout.

    Note that this function can be used both in eager context, or within a
    jitted function.

    Args:
        tensor: `jax.Array` that need to be distributed.
        layout: `TensorLayout` for the created variable, or a
            JAX-supported layout instance (e.g. `jax.sharding.Sharding`).

    Returns:
        Distributed value.
    """
    # Avoid circular imports.
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = layout.backend_layout

    # TODO(scottzhu): This might not be a cheap check, we should consider
    # have some proper JAX API for doing this check.
    if jax_utils.is_in_jax_tracing_scope():
        return jax.lax.with_sharding_constraint(tensor, layout)

    # Skip relayout if unnecessary.
    if isinstance(tensor, jax.Array):
        if isinstance(
            layout, jax.sharding.Sharding
        ) and tensor.sharding.is_equivalent_to(layout, ndim=len(tensor.shape)):
            return tensor
        # JAX explicit "layout" support.
        elif hasattr(layout, "layout"):
            current_layout = getattr(tensor, "layout", None)
            if current_layout == layout:
                return tensor
        # JAX explicit "format" support.
        elif hasattr(layout, "format"):
            current_layout = getattr(tensor, "format", None)
            if current_layout == layout:
                return tensor

    return jax.device_put(tensor, layout)


def distribute_data_input(per_process_batch, layout, batch_dim_name):
    """Distribute the input data with the corresponding layout.

    Note that the inputs here is a local worker batch. Within the local worker,
    the data need to be further partitioned to map to each of the devices.

    Args:
        inputs: `jax.Array` that is already sharded to a local process size.
        layout: `TensorLayout` for the distribution information, or a
            `jax.sharding.Sharding` instance.

    Returns:
        A global batch distributed according to `layout`.
    """
    # Avoid circular imports.
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = layout.backend_layout

    return jax.make_array_from_process_local_data(layout, per_process_batch)


def initialize_rng():
    """Initializes the global random number generator across processes.

    This is required for consistent initialization in multi-host settings.
    """
    global_seed = rng_utils.get_random_seed()
    # Only set a random seed if not already set
    # via keras.config.set_random_seed()
    if global_seed is None:
        # Generate a random seed on each CPU host and psum them to get a single
        # consistent seed across all processes.
        cpu_devices = jax.devices("cpu")
        num_local_cpu_devices = jax.local_device_count("cpu")
        # Seed must be in range [0, 2^32 - 1], so to ensure proper range and
        # avoid signed integer overflow, we use uint32.
        local_seed = jax.numpy.asarray(
            [seed_generator.make_default_seed()] * num_local_cpu_devices,
            dtype=jax.numpy.uint32,
        )
        # Sum across processes and pull out the first item.
        global_seed = jax.pmap(
            lambda x: jax.lax.psum(x, "all"),
            axis_name="all",
            devices=cpu_devices,
        )(local_seed).item(0)
        # Set the global seed.
        rng_utils.set_random_seed(global_seed)

    # Check if the global seed generator is set and ensure it has an initialized
    # seed.  Otherwise, reset the seed to the global seed.
    global_seed_generator = global_state.get_global_attribute(
        "global_seed_generator"
    )
    if global_seed_generator is not None:
        seed = global_seed_generator.get_config()["seed"]
        if seed is None:
            global_state.set_global_attribute(
                "global_seed_generator",
                seed_generator.SeedGenerator(
                    seed=global_seed,
                    name=global_seed_generator.name,
                    backend=global_seed_generator.backend,
                ),
            )


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
        coordinator_address = job_addresses[0]
    else:
        coordinator_address = job_addresses

    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
    )

    # Ensure the random number generator is initialized across processes.
    initialize_rng()


def num_processes():
    """Return the number of processes for the current distribution setting."""
    return jax.process_count()


def process_id():
    """Return the current process ID for the distribution setting."""
    return jax.process_index()


def _to_backend_device(device_name):
    if isinstance(device_name, jax.Device):
        return device_name
    device_name = str(device_name)
    if ":" not in device_name:
        device_type, device_id = device_name, 0
    else:
        device_type, device_id = device_name.split(":")

    devices = jax.devices(backend=device_type)
    for device in devices:
        if device.platform == device_type and device.id == int(device_id):
            return device
    raise ValueError(f"Device not found: {device_name}")


def _to_backend_mesh(device_mesh):
    """Convert the DeviceMesh to JAX backend specific Mesh.

    Args:
        device_mesh: DeviceMesh instance to convert.

    Returns:
        A `jax.sharding.Mesh` instance.
    """
    shape = device_mesh.devices.shape
    devices = [_to_backend_device(d) for d in device_mesh.devices.flatten()]
    devices = np.array(devices).reshape(shape)
    return jax.sharding.Mesh(devices, device_mesh.axis_names)


def _to_backend_layout(tensor_layout):
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
    jax_mesh = tensor_layout.device_mesh.backend_mesh
    return jax.sharding.NamedSharding(jax_mesh, partition_spec)
