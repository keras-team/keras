"""Utilities for distribution strategy with Torch backend."""

import os

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import Shard
from torch.distributed.tensor import (
    distribute_tensor as torch_distribute_tensor,
)
from torch.distributed.tensor.parallel import ColwiseParallel
from torch.distributed.tensor.parallel import RowwiseParallel
from torch.distributed.tensor.parallel import parallelize_module

from keras.src.backend.common import global_state


def list_devices(device_type=None):
    """List all available devices for the given type.

    Args:
        device_type: String, either "cpu", "gpu"/"cuda", or "xla".
            If None, the default device type is used.

    Returns:
        A list of device strings (e.g., ["cuda:0", "cuda:1"]).
    """
    if device_type is None:
        from keras.src.backend.torch.core import get_device

        device_type = str(get_device()).split(":")[0]
    else:
        device_type = device_type.lower()
        if device_type == "gpu":
            device_type = "cuda"

    if device_type == "cuda":
        num_devices = torch.cuda.device_count()
    elif device_type == "xla":
        from keras.src.utils.module_utils import torch_xla

        if torch_xla.available:
            import torch_xla.core.xla_model as xm

            num_devices = len(xm.get_xla_supported_devices())
        else:
            num_devices = 0
    elif device_type == "cpu":
        num_devices = 1
    else:
        num_devices = 0

    return [f"{device_type}:{i}" for i in range(num_devices)]


def get_device_count(device_type=None):
    """Get the number of available devices for the given type.

    Args:
        device_type: String, either "cpu", "gpu"/"cuda", or "xla".

    Returns:
        Integer count of available devices.
    """
    return len(list_devices(device_type))


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the Torch distributed process group.

    Args:
        job_addresses: Optional string of comma-separated master addresses.
        num_processes: Optional integer, total number of processes.
        process_id: Optional integer, rank of the current process.
    """
    if not torch.distributed.is_initialized():
        if job_addresses:
            master_addr, master_port = job_addresses.split(",")[0].split(":")
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port

        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"

        if num_processes is not None:
            os.environ["WORLD_SIZE"] = str(num_processes)
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "1"

        if process_id is not None:
            os.environ["RANK"] = str(process_id)
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"

        from keras.src.backend.torch.core import get_device

        device_type = str(get_device()).split(":")[0]

        if device_type == "xla":
            backend = "xla"
        elif device_type == "cuda":
            backend = "nccl"
        else:
            backend = "gloo"
        torch.distributed.init_process_group(backend=backend)


def num_processes():
    """Get the number of processes in the distributed group.

    Returns:
        Integer number of processes.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def process_id():
    """Get the rank of the current process.

    Returns:
        Integer rank of the process.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _to_backend_mesh(device_mesh):
    """Convert Keras DeviceMesh to Torch DeviceMesh.

    Args:
        device_mesh: The Keras DeviceMesh instance.

    Returns:
        A Torch DeviceMesh instance.
    """
    from keras.src.backend.torch.core import get_device

    device_type = str(get_device()).split(":")[0]
    mesh_shape = device_mesh.shape
    return init_device_mesh(
        device_type, mesh_shape, mesh_dim_names=device_mesh.axis_names
    )


def _to_backend_layout(tensor_layout):
    """Convert Keras TensorLayout to Torch placements.

    Args:
        tensor_layout: The Keras TensorLayout instance.

    Returns:
        A tuple of (Torch DeviceMesh, list of placements).
    """
    if tensor_layout.device_mesh is None:
        raise ValueError(
            "Cannot create sharding when device mesh is not set for "
            "TensorLayout."
        )

    device_mesh = tensor_layout.device_mesh
    torch_mesh = device_mesh.backend_mesh

    placements = []
    for mesh_dim_name in device_mesh.axis_names:
        shard_dim = None
        for i, axis in enumerate(tensor_layout.axes):
            if axis == mesh_dim_name:
                shard_dim = i
                break
        if shard_dim is not None:
            placements.append(Shard(shard_dim))
        else:
            placements.append(Replicate())

    return (torch_mesh, placements)


class DDPModelWrapper(torch.nn.Module):
    """A wrapper for Keras models to be used with PyTorch DDP.

    This wrapper avoids DDP's recursive traversal of Keras layer attributes,
    which can lead to infinite recursion due to the way Keras tracks variables
    and layers.

    Args:
        keras_model: The Keras Model instance to wrap.
    """

    def __init__(self, keras_model):
        super().__init__()
        self._keras_model = [keras_model]

    def parameters(self, recurse=True):
        """Yield the parameters of the wrapped Keras model."""
        for var in self._keras_model[0].variables:
            yield var.value

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        """Yield named parameters of the wrapped Keras model."""
        for var in self._keras_model[0].variables:
            yield prefix + var.path, var.value

    def forward(self, *args, **kwargs):
        """Forward pass of the wrapped Keras model."""
        return self._keras_model[0](*args, **kwargs)


def distribute_variable(value, layout):
    """Distribute a Torch variable based on the given layout.

    Args:
        value: The Torch tensor or Parameter to distribute.
        layout: The layout to apply (Torch mesh and placements).

    Returns:
        A distributed Torch tensor or Parameter.
    """
    is_parameter = isinstance(value, torch.nn.Parameter)
    requires_grad = value.requires_grad if is_parameter else False

    sharded_tensor = distribute_tensor(value, layout)

    if is_parameter:
        res = torch.nn.Parameter(sharded_tensor, requires_grad=requires_grad)
        if hasattr(value, "constraint"):
            res.constraint = value.constraint
        else:
            res.constraint = None
        return res
    return sharded_tensor


def distribute_tensor(tensor, layout):
    """Distribute a Torch tensor based on the given layout.

    Args:
        tensor: The Torch tensor to distribute.
        layout: The layout to apply. Can be a Keras TensorLayout or a backend
            layout tuple.

    Returns:
        A distributed Torch tensor (DTensor).
    """
    from keras.src.distribution import DataParallel
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = layout.backend_layout

    mesh, placements = layout
    mesh_device_type = mesh.device_type

    distribution = global_state.get_global_attribute("distribution")
    if (
        isinstance(distribution, DataParallel)
        and distribution._is_multi_process
    ):
        return tensor

    if hasattr(tensor, "device_mesh"):
        return tensor.redistribute(mesh, placements)

    if str(tensor.device).split(":")[0] != mesh_device_type:
        tensor = tensor.to(mesh_device_type)

    if not tensor.is_leaf:
        res = DTensor.from_local(tensor, mesh, placements, run_check=False)
    else:
        res = torch_distribute_tensor(tensor, mesh, placements)

    return res


def distribute_data_input(per_process_batch, layout, batch_dim_name):
    """Distribute the input data based on the given layout.

    Args:
        per_process_batch: The local Torch tensor for the current process.
        layout: The layout to apply.
        batch_dim_name: Name of the batch dimension (unused).

    Returns:
        A distributed Torch tensor.
    """
    from keras.src.distribution import ModelParallel

    distribution = global_state.get_global_attribute("distribution")
    if isinstance(distribution, ModelParallel):
        return distribute_tensor(per_process_batch, layout)

    return distribute_tensor(per_process_batch, layout)


def parallelize_layer(layer, distribution):
    """Parallelize a layer based on the given distribution.

    Args:
        layer: The Keras Layer or Model instance to parallelize.
        distribution: The Keras Distribution instance.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    from keras.src.backend.torch.core import Variable
    from keras.src.distribution import DataParallel
    from keras.src.distribution import ModelParallel

    if not isinstance(distribution, (ModelParallel, DataParallel)):
        return

    if getattr(layer, "_is_parallelized", False):
        return

    mesh = distribution.device_mesh.backend_mesh

    if isinstance(distribution, ModelParallel):
        layout_map = distribution._layout_map
        variable_to_attr = {}
        param_id_to_var = {id(var.value): var for var in layer.variables}

        def find_variables(obj):
            if isinstance(obj, Variable):
                return

            for _, child in obj.named_children():
                find_variables(child)

            for name, param in obj.named_parameters(recurse=False):
                var = param_id_to_var.get(id(param))
                if var is not None:
                    style = _infer_parallel_style(var, layout_map, name)
                    if style is not None:
                        variable_to_attr[var.path] = (var, obj, name, style)

        find_variables(layer)

        module_plans = {}
        for var_path, (
            var,
            module,
            attr_name,
            style,
        ) in variable_to_attr.items():
            if module not in module_plans:
                module_plans[module] = {}
            module_plans[module][attr_name] = style
            setattr(module, attr_name, var.value)

        tp_mesh = mesh
        if "model" in distribution.device_mesh.axis_names:
            tp_mesh = mesh["model"]

        for module, sub_plan in module_plans.items():
            if isinstance(module, torch.nn.ParameterDict):
                continue
            parallelize_module(module, tp_mesh, sub_plan)

        for var_path, (
            var,
            module,
            attr_name,
            style,
        ) in variable_to_attr.items():
            sharded_param = getattr(module, attr_name)
            if not hasattr(sharded_param, "placements"):
                layout = layout_map[var.path]
                sharded_param = distribute_variable(var.value, layout)
                setattr(module, attr_name, sharded_param)

            if not isinstance(sharded_param, Variable):
                var._value = sharded_param
                if not hasattr(sharded_param, "constraint"):
                    sharded_param.constraint = var.constraint

        if hasattr(layer, "_torch_params"):
            for var in layer.variables:
                if var.path in layer.torch_params:
                    layer.torch_params[var.path] = var.value

    if (
        isinstance(distribution, DataParallel)
        and distribution._is_multi_process
    ):
        from keras.src.models import Model

        if isinstance(layer, Model):
            from keras.src.backend.torch.core import get_device

            device = get_device()

            wrapper_module = DDPModelWrapper(layer)
            if "cuda" in str(device):
                device_ids = [torch.cuda.current_device()]
                layer._ddp_wrapper = DDP(wrapper_module, device_ids=device_ids)
            else:
                layer._ddp_wrapper = DDP(wrapper_module)

    layer._is_parallelized = True


def _infer_parallel_style(variable, layout_map, attr_name):
    """Infer PyTorch ParallelStyle from Keras LayoutMap.

    Args:
        variable: The Keras Variable instance.
        layout_map: The LayoutMap for the current distribution.
        attr_name: Name of the attribute in the PyTorch module.

    Returns:
        A Torch ParallelStyle instance (ColwiseParallel or RowwiseParallel),
        or None if no parallel style is applicable.
    """
    layout = layout_map[variable.path]
    if layout is None or not any(axis is not None for axis in layout.axes):
        return None

    model_dim = "model"
    if model_dim in layout.axes:
        shard_idx = layout.axes.index(model_dim)
        if (
            "kernel" in attr_name
            or "embeddings" in attr_name
            or "weight" in attr_name
        ):
            if shard_idx == 1:
                return ColwiseParallel()
            elif shard_idx == 0:
                return RowwiseParallel()
        elif "bias" in attr_name:
            if shard_idx == 0:
                return ColwiseParallel()
    return None
