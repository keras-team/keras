import os

import torch


def get_device_count(device_type=None):
    """Returns total device count across all hosts."""
    if device_type is None:
        if torch.cuda.is_available():
            device_type = "gpu"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device_type = "xpu"
        else:
            from keras.src.utils.module_utils import torch_xla

            if torch_xla.available:
                device_type = "tpu"
            else:
                device_type = "cpu"

    device_type = device_type.lower()
    if device_type in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            return 0
    elif device_type == "mps":
        if not torch.backends.mps.is_available():
            return 0
    elif device_type == "xpu":
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return 0
    elif device_type == "tpu":
        from keras.src.utils.module_utils import torch_xla

        if not torch_xla.available:
            return 0
    elif device_type == "cpu":
        pass
    else:
        return 0

    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])

    if device_type in ("gpu", "cuda"):
        return torch.cuda.device_count()
    if device_type == "xpu":
        return torch.xpu.device_count()
    if device_type == "tpu":
        from keras.src.utils.module_utils import torch_xla

        return torch_xla.runtime.global_device_count()
    return 1


def list_devices(device_type=None):
    """Returns Keras device strings representing global indices."""
    if device_type is None:
        if torch.cuda.is_available():
            device_type = "gpu"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device_type = "xpu"
        else:
            from keras.src.utils.module_utils import torch_xla

            if torch_xla.available:
                device_type = "tpu"
            else:
                device_type = "cpu"

    device_type = device_type.lower()
    if device_type == "cuda":
        device_type = "gpu"

    count = get_device_count(device_type)
    return [f"{device_type}:{i}" for i in range(count)]


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the current process for distributed training."""
    if not torch.distributed.is_initialized():
        if num_processes is not None:
            world_size = int(num_processes)
        elif "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            world_size = 1

        if process_id is not None:
            rank = int(process_id)
        elif "RANK" in os.environ:
            rank = int(os.environ["RANK"])
        else:
            rank = 0

        if world_size > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            torch.distributed.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
            )


def num_processes():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def process_id():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def to_backend_device(device_name):
    """Returns the local device for the current process."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device_name is not None:
        device_name_lower = device_name.lower()
        if "meta" in device_name_lower:
            return torch.device("meta")
        if "cpu" in device_name_lower:
            return torch.device("cpu")
        if "gpu" in device_name_lower or "cuda" in device_name_lower:
            if ":" in device_name_lower:
                device_idx = int(device_name_lower.split(":")[1])
                return torch.device(f"cuda:{device_idx}")
            if torch.cuda.is_available():
                return torch.device(f"cuda:{local_rank}")
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _to_backend_mesh(keras_mesh):
    """Maps a Keras DeviceMesh to a Torch DeviceMesh."""
    from torch.distributed.device_mesh import DeviceMesh as TorchDeviceMesh
    from torch.distributed.device_mesh import init_device_mesh

    if isinstance(keras_mesh, TorchDeviceMesh):
        return keras_mesh
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return init_device_mesh(
        device_type,
        mesh_shape=tuple(keras_mesh.shape),
        mesh_dim_names=tuple(keras_mesh.axis_names),
    )


class DTensorLayout:
    """Wraps a torch DeviceMesh + placements for use as a backend layout."""

    def __init__(self, device_mesh, placements):
        self.device_mesh = device_mesh
        self.placements = placements


def _to_backend_layout(tensor_layout):
    """Converts Keras TensorLayout to PyTorch (DeviceMesh, placements)."""
    if tensor_layout is None:
        return None

    keras_mesh = tensor_layout.device_mesh
    torch_mesh = _to_backend_mesh(keras_mesh)

    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor import Shard

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

    return DTensorLayout(torch_mesh, tuple(placements))


def set_distribution(value):
    """Set the distribution as the global distribution setting."""
    from keras.src.distribution import distribution_lib as dist_lib

    if isinstance(value, dist_lib.ModelParallel):
        _register_distributed_strategies()


def distribute_tensor(tensor, layout):
    """Scatters or replicates a tensor according to the layout."""
    if layout is None:
        return tensor

    if isinstance(layout, (str, torch.device)):
        device = (
            to_backend_device(layout) if isinstance(layout, str) else layout
        )
        if tensor.device != device:
            return tensor.to(device)
        return tensor

    from keras.src.distribution import distribution_lib as dist_lib

    dist = dist_lib.distribution()
    if not isinstance(dist, dist_lib.ModelParallel):
        return tensor

    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = _to_backend_layout(layout)

    from torch.distributed.tensor import DTensor

    if isinstance(tensor, DTensor):
        return tensor.redistribute(
            device_mesh=layout.device_mesh, placements=layout.placements
        )

    return torch.distributed.tensor.distribute_tensor(
        tensor, device_mesh=layout.device_mesh, placements=layout.placements
    )


def distribute_data_input(tensor, layout, batch_dim_name):
    """Wraps per-process data as a DTensor."""
    if layout is None:
        return tensor

    if isinstance(layout, (str, torch.device)):
        device = (
            to_backend_device(layout) if isinstance(layout, str) else layout
        )
        if tensor.device != device:
            return tensor.to(device)
        return tensor

    from keras.src.distribution import distribution_lib as dist_lib

    dist = dist_lib.distribution()
    if not isinstance(dist, dist_lib.ModelParallel):
        return tensor

    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = _to_backend_layout(layout)

    from torch.distributed.tensor import DTensor

    if isinstance(tensor, DTensor):
        return tensor

    if not isinstance(tensor, torch.Tensor):
        from keras.src.backend.torch import core as torch_core

        tensor = torch_core.convert_to_tensor(tensor, layout=None)

    if tensor.device.type == "meta":
        return tensor

    return DTensor.from_local(
        tensor, device_mesh=layout.device_mesh, placements=layout.placements
    )


_STRATEGIES_REGISTERED = False


def _unbind_op_strategy(op_schema):
    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor import Shard
    from torch.distributed.tensor._dtensor_spec import DTensorSpec
    from torch.distributed.tensor._op_schema import OpSpec
    from torch.distributed.tensor._op_schema import OpStrategy

    input_strategy = op_schema.args_schema[0]
    mesh = input_strategy.mesh
    new_strategy = OpStrategy([])

    for arg_strategy in input_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        dim = op_schema.args_schema[1] if len(op_schema.args_schema) > 1 else 0
        dim = dim if dim >= 0 else dim + arg_spec.ndim

        is_sharded_on_dim = any(
            isinstance(p, Shard) and p.dim == dim for p in arg_spec.placements
        )
        if is_sharded_on_dim:
            rep_placements = tuple(Replicate() for _ in arg_spec.placements)
            rep_spec = DTensorSpec(
                mesh=mesh,
                placements=rep_placements,
                tensor_meta=arg_spec.tensor_meta,
            )
            out_spec = DTensorSpec(mesh=mesh, placements=rep_placements)
            new_strategy.strategies.append(
                OpSpec(
                    output_specs=(out_spec,) * arg_spec.shape[dim],
                    input_specs=(rep_spec,),
                )
            )
        else:
            out_placements = [
                Shard(p.dim - 1) if isinstance(p, Shard) and p.dim > dim else p
                for p in arg_spec.placements
            ]
            out_spec = DTensorSpec(mesh=mesh, placements=tuple(out_placements))
            new_strategy.strategies.append(
                OpSpec(
                    output_specs=(out_spec,) * arg_spec.shape[dim],
                    input_specs=(arg_spec,),
                )
            )
    return new_strategy


def _register_distributed_strategies():
    """Register sharding propagation for ops.

    No-ops if already registered or if the PyTorch internal API is unavailable.
    """
    global _STRATEGIES_REGISTERED
    if _STRATEGIES_REGISTERED:
        return

    try:
        from torch.distributed.tensor._op_schema import RuntimeSchemaInfo
        from torch.distributed.tensor._ops import register_op_strategy

        register_op_strategy(
            torch.ops.aten.unbind.int, schema_info=RuntimeSchemaInfo(1)
        )(_unbind_op_strategy)
        _STRATEGIES_REGISTERED = True
    except (ImportError, AttributeError):
        # PyTorch version does not expose these internal APIs yet
        # unbind on sharded DTensors will fall back to PyTorch's default.
        pass
