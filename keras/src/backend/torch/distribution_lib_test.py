"""Tests for distribution_lib.py."""

import os
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from keras.src import backend
from keras.src import testing
from keras.src.backend.torch import distribution_lib as backend_dlib
from keras.src.distribution import distribution_lib


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="Backend specific test",
)
class TorchDistributionLibTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        if not dist.is_initialized():
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            # Using gloo backend for CPU testing
            dist.init_process_group(backend="gloo", rank=0, world_size=1)

    def test_list_devices(self):
        devices = backend_dlib.list_devices()
        self.assertGreater(len(devices), 0)
        # Verify the format of device strings
        self.assertTrue(any(":" in d for d in devices))

        cpu_devices = backend_dlib.list_devices("cpu")
        self.assertGreater(len(cpu_devices), 0)
        self.assertIn("cpu:", cpu_devices[0])

        if torch.cuda.is_available():
            cuda_devices = backend_dlib.list_devices("gpu")
            self.assertGreater(len(cuda_devices), 0)
            self.assertIn("cuda:", cuda_devices[0])

    def test_get_device_count(self):
        count = backend_dlib.get_device_count()
        self.assertGreaterEqual(count, 1)

    def test_to_backend_mesh(self):
        devices = ["cpu:0"]
        shape = (1,)
        axis_names = ["data"]

        mesh = distribution_lib.DeviceMesh(shape, axis_names, devices)
        torch_mesh = backend_dlib._to_backend_mesh(mesh)

        from torch.distributed.device_mesh import DeviceMesh

        self.assertIsInstance(torch_mesh, DeviceMesh)
        self.assertEqual(torch_mesh.mesh.shape, shape)
        self.assertEqual(torch_mesh.mesh_dim_names, tuple(axis_names))

    def test_to_backend_layout(self):
        axes = ["data"]
        mesh = distribution_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
        layout = distribution_lib.TensorLayout(axes, mesh)
        placements = backend_dlib._to_backend_layout(layout)

        from torch.distributed.tensor import Shard

        self.assertIsInstance(placements[0], Shard)
        self.assertEqual(placements[0].dim, 0)

    def test_distribute_tensor(self):
        mesh = distribution_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
        layout = distribution_lib.TensorLayout(["data"], mesh)

        tensor = torch.arange(4, dtype=torch.float32)
        distributed_tensor = backend_dlib.distribute_tensor(tensor, layout)

        from torch.distributed.tensor import DTensor

        self.assertIsInstance(distributed_tensor, DTensor)
        self.assertEqual(tuple(distributed_tensor.shape), (4,))

    def test_sync_tensors(self):
        mesh = distribution_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
        layout = distribution_lib.TensorLayout([None], mesh)

        from torch.distributed.tensor import DTensor

        torch_mesh = mesh.backend_mesh

        t1 = torch.ones((4,))
        # Create a DTensor
        d1 = backend_dlib.distribute_tensor(t1, layout)

        t2 = torch.zeros((4,))

        # Sync mixed list
        synced = backend_dlib._sync_tensors(d1, t2)

        self.assertIsInstance(synced[0], DTensor)
        self.assertIsInstance(synced[1], DTensor)
        self.assertEqual(synced[0].device_mesh, torch_mesh)
        self.assertEqual(synced[1].device_mesh, torch_mesh)

    @mock.patch("torch.distributed.init_process_group")
    def test_initialize(self, mock_init):
        # Reset initialized state for test if possible or just test mock
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            backend_dlib.initialize("localhost:12345", 1, 0)
            mock_init.assert_called_once()
            self.assertEqual(os.environ["MASTER_ADDR"], "localhost")
            self.assertEqual(os.environ["MASTER_PORT"], "12345")
            self.assertEqual(os.environ["WORLD_SIZE"], "1")
            self.assertEqual(os.environ["RANK"], "0")

    def test_initialize_rng(self):
        from keras.src.backend.common import global_state
        from keras.src.utils import rng_utils

        # Ensure seed is None
        old_seed = rng_utils.get_random_seed()
        global_state.set_global_attribute(rng_utils.GLOBAL_RANDOM_SEED, None)

        try:
            # Test rank 0 case (seed generation)
            with mock.patch("torch.distributed.get_rank", return_value=0):
                with mock.patch(
                    "torch.distributed.broadcast"
                ) as mock_broadcast:
                    backend_dlib.initialize_rng()
                    mock_broadcast.assert_called_once()
                    self.assertIsNotNone(rng_utils.get_random_seed())
        finally:
            # Restore seed
            global_state.set_global_attribute(
                rng_utils.GLOBAL_RANDOM_SEED, old_seed
            )

    def test_distribute_data_input(self):
        mesh = distribution_lib.DeviceMesh((1,), ["batch"], ["cpu:0"])
        layout = distribution_lib.TensorLayout(["batch"], mesh)

        data = torch.arange(8).reshape(2, 4)
        distributed_data = backend_dlib.distribute_data_input(
            data, layout, "batch"
        )

        from torch.distributed.tensor import DTensor

        self.assertIsInstance(distributed_data, DTensor)
        self.assertEqual(tuple(distributed_data.shape), (2, 4))
