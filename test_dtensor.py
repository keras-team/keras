import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import distribute_tensor

# Setup a dummy mesh
mesh = init_device_mesh("cpu", (1,))
layout = (mesh, [Replicate()])

t = torch.randn(4, 4)
dt = distribute_tensor(t, mesh, [Replicate()])

print(f"Is DTensor: {isinstance(dt, DTensor)}")

# Check zeros_like without monkeypatch
z = torch.zeros_like(dt)
print(f"zeros_like(dt) type: {type(z)}")
print(f"zeros_like(dt) is DTensor: {isinstance(z, DTensor)}")

# Check zeros with mesh? Usually factory functions need device_mesh context or explicit mesh
try:
    # Factory functions in newer Torch can be mesh-aware via context manager
    from torch.distributed.tensor import Distribute

    with mesh:
        z2 = torch.zeros(4, 4)
        print(f"torch.zeros(4, 4) inside mesh type: {type(z2)}")
except ImportError:
    print("Distribute context manager not available")
except Exception as e:
    print(f"Error inside mesh context: {e}")
