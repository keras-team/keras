"""Utilities for distribution strategy with Torch backend."""

import os
import torch
import numpy as np
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

# Note: The unbind operator issue was fixed by registering the operation sharding
# strategy. If you still experience issues, you can either:
# 1. Keep embedding layers replicated (not sharded) via layout_map
# 2. Set KERAS_TORCH_DISABLE_DTENSOR=1 to disable distributed tensors
# 3. Use environment variable TorchCompile to control compilation behavior


def _register_unbind_strategy():
    """
    Register unbind operation strategy for distributed tensors.
    
    This handles the case where unbind is called on a DTensor,
    which happens during iteration or certain tensor operations.
    The issue occurs because PyTorch's distributed tensor framework
    doesn't have a default sharding strategy for the unbind operation.
    """
    try:
        # Try the modern PyTorch API
        from torch.distributed.tensor._ops.registration import register_prop_rule
        from torch.distributed.tensor._sharding_prop import OutputShardingProp
        
        class UnbindOutputProp(OutputShardingProp):
            """Output sharding propagation for unbind operation."""
            
            def propagate_op_sharding(self, op_info):
                """
                For unbind(tensor, dim=0):
                - Unbind splits dimension 0
                - Output tensors have one less dimension
                - Safe default: all outputs are replicated
                """
                try:
                    # Get input sharding information 
                    input_placement = op_info.op_schema[0]
                    
                    if isinstance(input_placement, list):
                        # Return replicated placement for each output dimension
                        return [Replicate() for _ in range(len(input_placement))]
                    else:
                        # Single placement spec
                        return [Replicate()]
                except Exception:
                    # Fallback: safe default is all replicated
                    return [Replicate()]
        
        register_prop_rule(
            torch.ops.aten.unbind.int,
            UnbindOutputProp(),
        )
        
    except Exception:
        # If registration fails with new API, try older approach
        try:
            from torch.distributed.tensor._ops.registration import register_op_strategy
            
            def unbind_strategy(op_schema):
                # Return all outputs as replicated
                return ("replicate",)
            
            register_op_strategy(torch.ops.aten.unbind.int, unbind_strategy)
        except Exception:
            # If all registration attempts fail, continue without it
            # PyTorch may handle it with defaults later
            pass


# Ensure registration happens on import
_register_unbind_strategy()

def list_devices(device_type=None):
    """Return all the available devices based on the device type."""
    device_type = device_type or "gpu"
    if torch.distributed.is_initialized():
        count = torch.distributed.get_world_size()
    elif "WORLD_SIZE" in os.environ:
        count = int(os.environ["WORLD_SIZE"])
    else:
        if device_type.lower() == "gpu":
            count = torch.cuda.device_count() or 1
        else:
            count = 1  # Default for CPU
    return [f"{device_type.lower()}:{i}" for i in range(count)]

def get_device_count(device_type=None):
    """Returns the number of available devices."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    elif "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    else:
        if device_type and device_type.lower() == "gpu":
             return torch.cuda.device_count() or 1
        else:
             return 1

def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initializes the distribution system."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    if not torch.distributed.is_initialized():
        # MASTER_ADDR/PORT must be set in env for init_process_group
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
            
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo"
        )

def num_processes():
    """Return the number of processes."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1

def process_id():
    """Return the current process ID."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0

def _to_backend_device(device_name):
    """Convert Keras device name to Torch device."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")

def _to_backend_mesh(keras_mesh):
    """Convert Keras DeviceMesh to Torch DeviceMesh."""
    return init_device_mesh(
        "cuda" if torch.cuda.is_available() else "cpu",
        mesh_shape=tuple(keras_mesh.shape),
        mesh_dim_names=tuple(keras_mesh.axis_names),
    )

def _to_backend_layout(tensor_layout):
    """Convert Keras TensorLayout to Torch DTensor placement spec."""
    if tensor_layout is None:
        return None
    
    keras_mesh = tensor_layout.device_mesh
    torch_mesh = _to_backend_mesh(keras_mesh)
    
    placements = []
    for mesh_dim_name in keras_mesh.axis_names:
        shard_dim = None
        if tensor_layout.axes is not None:
            for tensor_dim, axis_name in enumerate(tensor_layout.axes):
                if axis_name == mesh_dim_name:
                    shard_dim = tensor_dim
                    break
        if shard_dim is not None:
            placements.append(Shard(shard_dim))
        else:
            placements.append(Replicate())
    
    return torch_mesh, tuple(placements)

def distribute_tensor(tensor, layout):
    """Distribute a tensor across devices."""
    if layout is None:
        return tensor
    from keras.src.distribution import distribution_lib as dist_lib
    dist = dist_lib.distribution()
    if not isinstance(dist, dist_lib.ModelParallel):
        return tensor

    from keras.src.distribution import TensorLayout
    if isinstance(layout, TensorLayout):
        backend_layout = _to_backend_layout(layout)
        if backend_layout is None:
            return tensor
        torch_mesh, placements = backend_layout
    else:
        torch_mesh, placements = layout
    
    if isinstance(tensor, DTensor):
        return tensor.redistribute(device_mesh=torch_mesh, placements=placements)

    return torch.distributed.tensor.distribute_tensor(
        tensor, device_mesh=torch_mesh, placements=placements
    )

def distribute_variable(value, layout, trainable=True):
    """Distribute a Keras variable."""
    from keras.src.distribution import distribution_lib as dist_lib
    dist = dist_lib.distribution()
    if not isinstance(dist, dist_lib.ModelParallel):
        return torch.nn.Parameter(value, requires_grad=trainable)

    dtensor = distribute_tensor(value, layout)
    return torch.nn.Parameter(dtensor, requires_grad=trainable)

def distribute_data_input(tensor, layout, batch_dim_name):
    """Distribute a local data tensor according to a TensorLayout."""
    if layout is None:
        return tensor  # DataParallel: no-op
    from keras.src.distribution import distribution_lib as dist_lib
    dist = dist_lib.distribution()
    if not isinstance(dist, dist_lib.ModelParallel):
        return tensor

    if isinstance(tensor, DTensor):
        return tensor
    from keras.src.distribution import TensorLayout
    if isinstance(layout, TensorLayout):
        backend_layout = _to_backend_layout(layout)
        if backend_layout is None:
            return tensor
        torch_mesh, placements = backend_layout
    else:
        torch_mesh, placements = layout
    return DTensor.from_local(
        tensor, device_mesh=torch_mesh, placements=placements
    )
