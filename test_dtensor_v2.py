import os

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import distribute_tensor

# Mock initialization like Keras does
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29505"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
torch.distributed.init_process_group(backend="gloo")

# Setup a dummy mesh
mesh = init_device_mesh("cpu", (1,))

t = torch.randn(4, 4)
dt = distribute_tensor(t, mesh, [Replicate()])

print(f"Is DTensor: {isinstance(dt, DTensor)}")

# Check zeros_like without monkeypatch
z = torch.zeros_like(dt)
print(f"zeros_like(dt) type: {type(z)}")
print(f"zeros_like(dt) is DTensor: {isinstance(z, DTensor)}")

# Check zeros with mesh
try:
    with mesh:
        z2 = torch.zeros(4, 4)
        print(f"torch.zeros(4, 4) inside mesh type: {type(z2)}")
        print(
            f"torch.zeros(4, 4) inside mesh is DTensor: {isinstance(z2, DTensor)}"
        )
except Exception as e:
    print(f"Error inside mesh context: {e}")

torch.distributed.destroy_process_group()
