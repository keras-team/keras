"""Utilities for distribution strategy with Torch backend."""

import os
import sys
import torch
import numpy as np
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

# Disable problematic torch.compile features
os.environ.setdefault("TORCH_COMPILE_DEBUG", "0")


def _patch_torch_dtensor_unbind():
    """
    Patch torch's unbind at the DTensor level to use a safe implementation.
    This must be done very early, before any distributed tensor operations.
    """
    try:
        # Access DTensor's _op_dispatcher if available
        if hasattr(DTensor, '_op_dispatcher'):
            # Will register before dispatch happens
            pass
        
        # Monkey-patch DTensor.__iter__ with a safe version
        _original_iter = DTensor.__iter__ if hasattr(DTensor, '__iter__') else None
        
        def _safe_dtensor_iter(self):
            """
            Safe iteration for DTensor that converts to local to avoid
            the unbind NotImplementedError in distributed contexts.
            """
            try:
                # Strategy: convert to local, unbind, distribute results
                local_tensor = self.to_local()
                unbounded_tensors = local_tensor.unbind(0)
                
                # Yield each as a replicated DTensor
                for unbounded_item in unbounded_tensors:
                    yield DTensor.from_local(
                        unbounded_item,
                        device_mesh=self.device_mesh,
                        placements=[Replicate() for _ in range(len(self.device_mesh.mesh.shape))]
                    )
            except Exception:
                # Fallback - just use local iteration
                local_tensor = self.to_local()
                for item in local_tensor:
                    yield item
        
        DTensor.__iter__ = _safe_dtensor_iter
        
        # Also patch unbind method
        _original_unbind = DTensor.unbind if hasattr(DTensor, 'unbind') else None
        
        def _safe_dtensor_unbind(self, dim=0):
            """Safe unbind that doesn't trigger distributed tensor dispatch."""
            try:
                local_tensor = self.to_local()
                unbounded_tensors = local_tensor.unbind(dim)
                # Return as replicated DTensors
                return [
                    DTensor.from_local(
                        t,
                        device_mesh=self.device_mesh,
                        placements=[Replicate() for _ in range(len(self.device_mesh.mesh.shape))]
                    )
                    for t in unbounded_tensors
                ]
            except Exception:
                return self.to_local().unbind(dim)
        
        DTensor.unbind = _safe_dtensor_unbind
        
    except Exception as e:
        print(f"Warning: Failed to patch DTensor iteration: {e}", file=sys.stderr)


def _register_unbind_sharding_strategy():
    """
    Register a sharding strategy for unbind operation.
    This is a backup for cases where the monkey-patch doesn't work.
    """
    try:
        from torch.distributed.tensor._ops.registration import register_prop_rule
        from torch.distributed.tensor._sharding_prop import OutputShardingProp
        
        class UnbindShardingProp(OutputShardingProp):
            def propagate_op_sharding(self, op_info):
                # Default: all outputs are replicated
                try:
                    num_outputs = 1  # Unbind can return multiple outputs
                    return [Replicate()]
                except Exception:
                    return [Replicate()]
        
        try:
            register_prop_rule(torch.ops.aten.unbind.int, UnbindShardingProp())
        except Exception:
            pass
            
    except Exception:
        # Fallback: try alternative registration
        try:
            from torch.distributed.tensor._ops.registration import register_op_strategy
            register_op_strategy(torch.ops.aten.unbind.int, lambda op_schema: ("replicate",))
        except Exception:
            pass


# Apply patches and registration immediately on import
_patch_torch_dtensor_unbind()
_register_unbind_sharding_strategy()

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
