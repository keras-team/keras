"""Utilities for distribution strategy with PyTorch backend.

This module provides PyTorch-specific implementations of distribution
primitives that mirror the JAX distribution API. This enables Keras
distribution strategies (DataParallel, ModelParallel) to work with
PyTorch backend.

Note: PyTorch doesn't have JAX's automatic sharding, so we use explicit
tensor partitioning and torch.distributed primitives.
"""

import os
import sys
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# Debugging utilities for torch distribution
_DEBUG_ENABLED = None
_DEBUG_PREFIX = "[TORCH-DISTRIBUTION-DEBUG]"


def _is_debug_enabled():
    """Check if debug mode is enabled via environment variable."""
    global _DEBUG_ENABLED
    if _DEBUG_ENABLED is None:
        _DEBUG_ENABLED = os.environ.get("KERAS_TORCH_DISTRIBUTION_DEBUG", "0") == "1"
    return _DEBUG_ENABLED


def _get_debug_prefix():
    """Get prefix for debug messages with timestamp and rank info."""
    if dist.is_initialized():
        rank = dist.get_rank()
        return f"{_DEBUG_PREFIX} [Rank {rank}] [{datetime.now().strftime('%H:%M:%S.%f')[:-3]}]"
    else:
        return f"{_DEBUG_PREFIX} [{datetime.now().strftime('%H:%M:%S.%f')[:-3]}]"


def _debug_log(message):
    """Log debug message if debug mode is enabled."""
    if _is_debug_enabled():
        prefix = _get_debug_prefix()
        print(f"{prefix} {message}", flush=True)
        sys.stdout.flush()


def _debug_function_entry(func_name, args=None, kwargs=None):
    """Log function entry with arguments."""
    if _is_debug_enabled():
        args_str = str(args) if args else ""
        kwargs_str = str(kwargs) if kwargs else ""
        _debug_log(f"ENTER {func_name}({args_str}, {kwargs_str})")


def _debug_function_exit(func_name, result=None):
    """Log function exit with result."""
    if _is_debug_enabled():
        result_str = str(result) if result is not None else "None"
        # Truncate long results
        if len(result_str) > 200:
            result_str = result_str[:200] + "..."
        _debug_log(f"EXIT {func_name} -> {result_str}")


def _debug_tensor_info(tensor, name="tensor"):
    """Log tensor information for debugging."""
    if _is_debug_enabled() and tensor is not None:
        if hasattr(tensor, 'shape') and hasattr(tensor, 'dtype'):
            _debug_log(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
                      f"requires_grad={tensor.requires_grad if hasattr(tensor, 'requires_grad') else 'N/A'}")
        else:
            _debug_log(f"{name}: {type(tensor)}")


def list_devices(device_type=None):
    """Return all the available devices based on the device type.

    Note that this should return the global devices in a distributed setting.

    Args:
        device_type: string of `"cpu"`, `"gpu"` or `"tpu"`. Defaults to `"gpu"`
            or `"cuda"` if available when device_type is not provided. Otherwise
            will return the `"cpu"` devices.

    Return:
        List of devices that are available for distribute computation.
    """
    device_type = device_type.lower() if device_type else None

    if device_type is None or device_type in ("gpu", "cuda"):
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            return [f"cuda:{i}" for i in range(num_devices)]
        elif hasattr(torch, "mps") and torch.mps.is_available():
            # macOS MPS support
            return ["mps:0"]
        elif device_type in ("gpu", "cuda"):
            raise RuntimeError("No CUDA devices available")

    if device_type == "cpu":
        return ["cpu:0"]

    raise ValueError(f"Unsupported device_type: {device_type}")


def get_device_count(device_type=None):
    """Returns the number of available devices.

    Args:
        device_type: Optional device type to count (e.g., "cpu", "gpu", "cuda").
            If `None`, it defaults to counting "gpu" or "cuda" devices if
            available, otherwise it counts "cpu" devices.

    Returns:
        int: The total number of devices for the specified type.
    """
    device_type = device_type.lower() if device_type else None

    if device_type is None or device_type in ("gpu", "cuda"):
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        elif hasattr(torch, "mps") and torch.mps.is_available():
            return 1  # MPS typically uses single device
        elif device_type in ("gpu", "cuda"):
            return 0

    if device_type == "cpu":
        return 1  # CPU is treated as single device

    return 0


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout.

    For PyTorch, this simulates sharding by:
    - For data parallelism: replicating tensors or slicing along batch dim
    - For model parallelism: partitioning tensors according to layout axes

    Args:
        tensor: `torch.Tensor` that needs to be distributed.
        layout: `TensorLayout` for the distribution information.

    Returns:
        Distributed tensor (or tuple of shards for model parallel).
    """
    _debug_function_entry("distribute_tensor", kwargs={"layout": layout})
    _debug_tensor_info(tensor, "input_tensor")

    # Avoid circular imports.
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        # Convert to backend layout if needed
        backend_layout = layout.backend_layout if hasattr(layout, 'backend_layout') else layout
        _debug_log(f"Using TensorLayout: axes={backend_layout.axes}, device_mesh={backend_layout.device_mesh}")
    else:
        backend_layout = layout
        _debug_log(f"Using raw layout: {backend_layout}")

    # Handle None layout (no distribution needed)
    if backend_layout is None:
        _debug_log("Layout is None, returning tensor as-is (replicated)")
        _debug_tensor_info(tensor, "output_tensor")
        _debug_function_exit("distribute_tensor", result=tensor)
        return tensor

    # Get axes from layout
    if hasattr(backend_layout, 'axes'):
        axes = backend_layout.axes
    else:
        axes = backend_layout

    _debug_log(f"Layout axes: {axes}")

    # Get device mesh
    device_mesh = None
    if hasattr(backend_layout, 'device_mesh') and backend_layout.device_mesh is not None:
        device_mesh = backend_layout.device_mesh
        _debug_log(f"Device mesh: shape={device_mesh.shape}, axis_names={device_mesh.axis_names}")

    # Check if this is a replicated tensor (no sharding axes)
    sharding_axes = [ax for ax in axes if ax is not None]
    if not sharding_axes:
        # Replicated - return tensor as-is
        _debug_log("No sharding axes found, returning tensor as-is (replicated)")
        _debug_tensor_info(tensor, "output_tensor")
        _debug_function_exit("distribute_tensor", result=tensor)
        return tensor

    # For now, support basic sharding on first sharding axis
    # Model parallelism will require more sophisticated handling
    first_sharding_axis = sharding_axes[0]
    _debug_log(f"First sharding axis: {first_sharding_axis}")

    if device_mesh is not None:
        # Get the axis index in the mesh
        axis_names = device_mesh.axis_names
        if first_sharding_axis in axis_names:
            axis_idx = axis_names.index(first_sharding_axis)
            mesh_dim = device_mesh.shape[axis_idx]

            _debug_log(f"Sharding on mesh axis {axis_idx} (dimension size: {mesh_dim})")
            _debug_log(f"Tensor dimension at axis {axis_idx}: {tensor.shape[axis_idx]}")

            # Shard along the specified axis
            if tensor.shape[axis_idx] >= mesh_dim:
                if tensor.shape[axis_idx] % mesh_dim == 0:
                    shard_size = tensor.shape[axis_idx] // mesh_dim
                    _debug_log(f"Even sharding: shard_size={shard_size}, splitting tensor")
                    result = list(torch.split(tensor, shard_size, dim=axis_idx))
                    _debug_log(f"Split into {len(result)} shards")
                    _debug_tensor_info(result[0], "first_shard")
                    _debug_function_exit("distribute_tensor", result=result)
                    return result
                else:
                    # Cannot evenly divide, return as list
                    _debug_log("Uneven sharding: cannot evenly divide, using torch.chunk")
                    result = list(torch.chunk(tensor, mesh_dim, dim=axis_idx))
                    _debug_log(f"Chunked into {len(result)} shards")
                    _debug_tensor_info(result[0], "first_chunk")
                    _debug_function_exit("distribute_tensor", result=result)
                    return result
            else:
                _debug_log(f"Tensor dimension ({tensor.shape[axis_idx]}) < mesh dimension ({mesh_dim}), cannot shard")
        else:
            _debug_log(f"Axis '{first_sharding_axis}' not found in device mesh axes {axis_names}")

    # Fallback: return original tensor if sharding not applicable
    _debug_log("Falling back to original tensor (no sharding applied)")
    _debug_tensor_info(tensor, "output_tensor")
    _debug_function_exit("distribute_tensor", result=tensor)
    return tensor


def distribute_variable(value, layout):
    """Create a distributed variable for PyTorch.

    This creates a representation of a variable that respects the given
    layout/sharding specification.

    Args:
        value: the initial value of the variable (torch.Tensor).
        layout: `TensorLayout` for the created variable.

    Returns:
        Distributed variable representation.
    """
    _debug_function_entry("distribute_variable", kwargs={"layout": layout})
    _debug_tensor_info(value, "variable_value")

    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        backend_layout = layout.backend_layout if hasattr(layout, 'backend_layout') else layout
        _debug_log(f"Using TensorLayout: axes={backend_layout.axes}")
    else:
        backend_layout = layout
        _debug_log(f"Using raw layout: {backend_layout}")

    # For PyTorch, we wrap the tensor with distribution metadata
    # The actual sharding is applied during operations
    if hasattr(value, '_distributed_layout'):
        _debug_log("Variable already has distributed layout, updating it")
        value._distributed_layout = backend_layout
    else:
        # Store layout as attribute for later use
        _debug_log("Setting distributed layout attribute on variable")
        value._distributed_layout = backend_layout

    _debug_log(f"Variable layout set to: {backend_layout}")
    _debug_function_exit("distribute_variable", result=value)
    return value


def distribute_data_input(per_process_batch, layout, batch_dim_name):
    """Distribute the input data with the corresponding layout.

    Note that the inputs here is a local worker batch. Within the local worker,
    the data need to be further partitioned to map to each of the devices.

    Args:
        per_process_batch: `torch.Tensor` that is already sharded to local
            process size.
        layout: `TensorLayout` for the distribution information.
        batch_dim_name: string name of the batch dimension.

    Returns:
        A global batch distributed according to `layout`.
    """
    _debug_function_entry("distribute_data_input", 
                         kwargs={"layout": layout, "batch_dim_name": batch_dim_name})
    
    if isinstance(per_process_batch, (tuple, list)):
        _debug_log(f"Input is {type(per_process_batch)} with {len(per_process_batch)} elements")
        result = type(per_process_batch)(
            distribute_data_input(x, layout, batch_dim_name) for x in per_process_batch
        )
        _debug_function_exit("distribute_data_input", result=result)
        return result
    
    _debug_tensor_info(per_process_batch, "per_process_batch")
    _debug_log(f"Batch dimension name: {batch_dim_name}")
    
    # For PyTorch, we just need to ensure the data is on the right device
    # and properly shaped
    result = per_process_batch
    _debug_tensor_info(result, "distributed_batch")
    _debug_function_exit("distribute_data_input", result=result)
    return result


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the distribution system for multi-process setting.

    Calling `initialize` will prepare PyTorch for distributed execution
    on multiple processes/GPUs.

    Args:
        job_addresses: string. Comma separated IP addresses for all the jobs
            that will form the whole computation cluster.
        num_processes: int. The number of worker/processes that will form the
            whole computation cluster.
        process_id: int. The ID number of the current worker/process. The value
            should be ranged from `0` to `num_processes - 1`.

    Environment Variables:
        Can also be configured via environment variables:
        - KERAS_DISTRIBUTION_JOB_ADDRESSES
        - KERAS_DISTRIBUTION_NUM_PROCESSES
        - KERAS_DISTRIBUTION_PROCESS_ID
    """
    _debug_function_entry("initialize", 
                         args=(job_addresses, num_processes, process_id))
    
    # Check environment variables first
    if job_addresses is None and "KERAS_DISTRIBUTION_JOB_ADDRESSES" in os.environ:
        job_addresses = os.environ["KERAS_DISTRIBUTION_JOB_ADDRESSES"]
        _debug_log(f"Using job_addresses from env: {job_addresses}")
    if num_processes is None and "KERAS_DISTRIBUTION_NUM_PROCESSES" in os.environ:
        num_processes = int(os.environ["KERAS_DISTRIBUTION_NUM_PROCESSES"])
        _debug_log(f"Using num_processes from env: {num_processes}")
    if process_id is None and "KERAS_DISTRIBUTION_PROCESS_ID" in os.environ:
        process_id = int(os.environ["KERAS_DISTRIBUTION_PROCESS_ID"])
        _debug_log(f"Using process_id from env: {process_id}")

    if num_processes is None or num_processes <= 1:
        # No multi-process setup needed
        _debug_log("No multi-process setup needed (num_processes <= 1)")
        _debug_function_exit("initialize")
        return

    # Parse job addresses
    if job_addresses and "," in job_addresses:
        job_addresses = job_addresses.split(",")
        coordinator_address = job_addresses[0]
    else:
        coordinator_address = job_addresses

    _debug_log(f"Coordinator address: {coordinator_address}")
    _debug_log(f"Number of processes: {num_processes}")
    _debug_log(f"Current process ID: {process_id}")

    # Set environment variables for torch.distributed
    os.environ["MASTER_ADDR"] = coordinator_address.split(":")[0] if ":" in coordinator_address else coordinator_address
    os.environ["MASTER_PORT"] = "29500"  # Default port for distributed

    if "RANK" not in os.environ:
        os.environ["RANK"] = str(process_id if process_id is not None else 0)
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(process_id if process_id is not None else 0)
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = str(num_processes)

    # Initialize process group
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        _debug_log(f"Initializing torch.distributed with backend: {backend}")
        _debug_log(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        _debug_log(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
        
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=process_id if process_id is not None else 0,
            world_size=num_processes,
        )
        
        _debug_log("Successfully initialized torch.distributed process group")
        _debug_log(f"World size: {dist.get_world_size()}")
        _debug_log(f"Local rank: {dist.get_rank()}")
    else:
        _debug_log("torch.distributed already initialized")
        _debug_log(f"World size: {dist.get_world_size()}")
        _debug_log(f"Local rank: {dist.get_rank()}")
    
    _debug_function_exit("initialize")


def num_processes():
    """Return the number of processes for the current distribution setting."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def process_id():
    """Return the current process ID for the distribution setting."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def _to_backend_device(device_name):
    """Convert a device name to PyTorch device.

    Args:
        device_name: string device name like "cuda:0" or "cpu".

    Returns:
        torch.device instance.
    """
    device_name = str(device_name).lower()
    if device_name.startswith("cuda") or device_name.startswith("gpu"):
        if ":" in device_name:
            idx = int(device_name.split(":")[-1])
            return torch.device(f"cuda:{idx}")
        return torch.device("cuda")
    elif device_name.startswith("mps"):
        return torch.device("mps")
    elif device_name.startswith("cpu"):
        return torch.device("cpu")
    else:
        return torch.device(device_name)


def _to_backend_mesh(device_mesh):
    """Convert the DeviceMesh to PyTorch backend specific representation.

    For PyTorch, we create a representation that can be used for
    distributed operations.

    Args:
        device_mesh: DeviceMesh instance to convert.

    Returns:
        A PyTorch mesh representation (dict with device list and axis names).
    """
    # For PyTorch, we return a simple representation
    # The actual distributed operations use torch.distributed primitives
    return {
        "devices": device_mesh.devices,
        "axis_names": device_mesh.axis_names,
        "shape": device_mesh.shape,
    }


def _to_backend_layout(tensor_layout):
    """Convert the TensorLayout to PyTorch backend specific representation.

    Args:
        tensor_layout: TensorLayout instance to convert.

    Returns:
        A PyTorch layout representation.
    """
    if tensor_layout.device_mesh is None:
        raise ValueError(
            "Cannot create sharding when device mesh is not set "
            "for TensorLayout."
        )

    # For PyTorch, we store the sharding specification
    return {
        "axes": tensor_layout.axes,
        "mesh": _to_backend_mesh(tensor_layout.device_mesh),
    }


def all_reduce(tensor, reduce_op="sum"):
    """Perform all-reduce operation on a tensor across all processes.

    Args:
        tensor: torch.Tensor to reduce.
        reduce_op: reduction operation - "sum", "product", "min", "max", etc.

    Returns:
        Reduced tensor (available on all processes).
    """
    _debug_function_entry("all_reduce", kwargs={"reduce_op": reduce_op})
    _debug_tensor_info(tensor, "input_tensor")

    if not dist.is_initialized():
        _debug_log("torch.distributed not initialized, skipping all_reduce")
        _debug_function_exit("all_reduce", result=tensor)
        return tensor

    # Convert string to torch distributed op
    if reduce_op == "sum":
        op = dist.ReduceOp.SUM
    elif reduce_op == "product":
        op = dist.ReduceOp.PRODUCT
    elif reduce_op == "min":
        op = dist.ReduceOp.MIN
    elif reduce_op == "max":
        op = dist.ReduceOp.MAX
    else:
        op = dist.ReduceOp.SUM

    _debug_log(f"Performing all_reduce with operation: {reduce_op}")
    _debug_log(f"World size: {dist.get_world_size()}, Rank: {dist.get_rank()}")

    # All-reduce requires same tensor on all processes
    dist.all_reduce(tensor, op)
    
    _debug_tensor_info(tensor, "reduced_tensor")
    _debug_function_exit("all_reduce", result=tensor)
    return tensor


def all_gather(tensor):
    """Gather tensors from all processes.

    Args:
        tensor: torch.Tensor to gather (should be same shape on all processes).

    Returns:
        Concatenated tensor from all processes.
    """
    _debug_function_entry("all_gather")
    _debug_tensor_info(tensor, "input_tensor")

    if not dist.is_initialized():
        _debug_log("torch.distributed not initialized, skipping all_gather")
        _debug_function_exit("all_gather", result=tensor)
        return tensor

    world_size = dist.get_world_size()
    _debug_log(f"World size: {world_size}")

    if world_size == 1:
        _debug_log("Single process, no gathering needed")
        _debug_function_exit("all_gather", result=tensor)
        return tensor

    # Gather tensor shapes to handle potentially different sizes
    _debug_log("Gathering tensors from all processes")
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)

    result = torch.cat(tensor_list, dim=0)
    _debug_tensor_info(result, "gathered_tensor")
    _debug_log(f"Gathered tensor shape: {result.shape}")
    _debug_function_exit("all_gather", result=result)
    return result


def broadcast(tensor, src=0):
    """Broadcast tensor from source process to all others.

    Args:
        tensor: torch.Tensor to broadcast.
        src: source process rank.

    Returns:
        Broadcasted tensor.
    """
    _debug_function_entry("broadcast", kwargs={"src": src})
    _debug_tensor_info(tensor, "input_tensor")

    if not dist.is_initialized():
        _debug_log("torch.distributed not initialized, skipping broadcast")
        _debug_function_exit("broadcast", result=tensor)
        return tensor

    rank = dist.get_rank()
    _debug_log(f"Current rank: {rank}, Source rank: {src}")

    if rank != src:
        _debug_log(f"Rank {rank} is not source, zeroing tensor before broadcast")
        tensor = torch.zeros_like(tensor)

    _debug_log(f"Broadcasting tensor from source rank {src}")
    dist.broadcast(tensor, src=src)
    
    _debug_tensor_info(tensor, "broadcasted_tensor")
    _debug_function_exit("broadcast", result=tensor)
    return tensor

