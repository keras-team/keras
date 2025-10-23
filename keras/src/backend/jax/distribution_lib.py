"""Utilities for distribution strategy with JAX backend."""

import jax
import numpy as np

from keras.src.backend.common import global_state
from keras.src.random import seed_generator
from keras.src.utils import jax_utils
from keras.src.utils import rng_utils


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


def get_device_count(device_type=None):
    """Returns the number of available JAX devices.
    Args:
        device_type: Optional device type to count (e.g., "cpu", "gpu", "tpu").
            If `None`, it defaults to counting "gpu" or "tpu" devices if
            available, otherwise it counts "cpu" devices. It does not
            return the sum of all device types.
    Returns:
        int: The total number of JAX devices for the specified type.
    """
    device_type = device_type.lower() if device_type else None
    return jax.device_count(device_type)


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


def _distribute_initializer(
    init_func=None, mean=0.0, stddev=1.0, seed=None, layout=None
):
    """
    Distribution-aware token embedding initializer for JAX backend.

    This function will create a Jax random array and
    distribute it according to the current token embedding layout.

    Args:
        init_func: A functools.partial-wrapped object that takes the seed
            as argument and returns a jax.Array. Must have shape and dtype
            already bound via partial.
        mean: Mean of distribution (applied to normal/truncated_normal).
        stddev: Standard deviation of the distribution.
        seed: Random seed for initialization.
        layout: TensorLayout for the distributed tensor.

    Returns:
        A distributed jax array.

    Raises:
        ValueError: If init_func or seed is None.
                    If init_func.func is not a supported random function.
                    Supported jax.random func: normal, truncated_normal, uniform
        TypeError: If init_func is not a functools.partial object.
    """
    import warnings
    from functools import partial

    # Create SeedGenerator to ensure backend variable exists
    # For future state tracking for distributed keys, add
    # attributes for base/split keys and number of devices sharded.
    if isinstance(seed, jax.Array):
        seed_gen = seed_generator.SeedGenerator(seed=int(seed[0]))
    elif isinstance(seed, int):
        seed_gen = seed_generator.SeedGenerator(seed=seed)
    elif isinstance(seed, seed_generator.SeedGenerator):
        seed_gen = seed
    else:
        raise ValueError(
            f"seed must be int, JAX array, or SeedGenerator, got {type(seed)}"
        )

    # Extract the state value as JAX array
    jax_seed = seed_gen.state.value

    # Convert to JAX PRNG key format (swap counter and seed value)
    jax_compatible_seed = jax.numpy.array(
        [jax_seed[1], jax_seed[0]], dtype=jax.numpy.uint32
    )

    # Validate all required arguments
    if init_func is None or init_func.func.__name__ not in [
        "normal",
        "truncated_normal",
        "uniform",
    ]:
        raise ValueError(
            "init_func cannot be None or "
            "Unsupported initializer: {init_func.func.__name__}."
            "only JAX-compatible random initializers are supported. "
            "Supported jax.random funcs: normal, truncated_normal, uniform"
        )

    # Ensure init_func is a partial
    if not isinstance(init_func, partial):
        raise TypeError(
            f"init_func must be functools.partial object, got {type(init_func)}"
            "init_func is a jax.random.* function with shape and "
            "dtype bound via partial"
        )

    # Shard based on tensor layout
    if layout is None:
        warnings.warn(
            f"The layout is {layout}, sharding will default to single device"
        )
        sharding = None
    else:
        sharding = _to_backend_layout(layout)

    # JAX PRNG key handling within JIT:
    # The key is passed directly to jax.random.* functions which are
    # JIT-compatible and functional. JAX automatically ensures different
    # random values per shard when out_shardings is specified.
    try:
        compiled_init = jax.jit(
            lambda jax_compatible_seed: init_func(jax_compatible_seed),
            out_shardings=sharding,
        )
        sample = compiled_init(jax_compatible_seed)
    except RuntimeError as e:
        warnings.warn(
            f"Sharding failed due to: {e}, falling back to single device"
        )
        compiled_init = jax.jit(
            lambda jax_compatible_seed: init_func(jax_compatible_seed),
            out_shardings=None,
        )
        sample = compiled_init(jax_compatible_seed)

    # Store the SeedGenerator for state tracking
    seed = seed_gen.next()

    # Apply mean/stddev only for distributions where it makes sense
    if init_func.func in (jax.random.normal, jax.random.truncated_normal):
        return sample * stddev + mean
    elif init_func.func == jax.random.uniform:
        # Uniform doesn't use mean/stddev - warn
        if mean != 0.0 or stddev != 1.0:
            warnings.warn(
                "mean and stddev are ignored for uniform distribution"
            )
        return sample
