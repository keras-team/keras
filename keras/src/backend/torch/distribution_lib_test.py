import os

import pytest
import torch

from keras.src import backend
from keras.src import testing
from keras.src.backend.torch import distribution_lib as backend_dlib
from keras.src.distribution import distribution_lib


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="Backend specific test",
)
class TorchDistributionLibTest(testing.TestCase):
    def tearDown(self):
        super().tearDown()
        if "WORLD_SIZE" in os.environ:
            del os.environ["WORLD_SIZE"]
        if "RANK" in os.environ:
            del os.environ["RANK"]
        if "LOCAL_RANK" in os.environ:
            del os.environ["LOCAL_RANK"]

    def test_list_devices(self):
        devices = backend_dlib.list_devices()
        self.assertGreater(len(devices), 0)

        os.environ["WORLD_SIZE"] = "4"
        devices = backend_dlib.list_devices("cpu")
        self.assertEqual(len(devices), 4)
        self.assertEqual(devices[0], "cpu:0")
        self.assertEqual(devices[3], "cpu:3")

    def test_get_device_count(self):
        count = backend_dlib.get_device_count()
        self.assertGreaterEqual(count, 1)

        os.environ["WORLD_SIZE"] = "2"
        self.assertEqual(backend_dlib.get_device_count(), 2)

    def test_num_processes(self):
        self.assertEqual(backend_dlib.num_processes(), 1)

    def test_process_id(self):
        self.assertEqual(backend_dlib.process_id(), 0)

    def test_to_backend_device(self):
        device = backend_dlib._to_backend_device("cpu")
        self.assertIsInstance(device, torch.device)
        if torch.cuda.is_available():
            self.assertEqual(device.type, "cuda")
        else:
            self.assertEqual(device.type, "cpu")

    def test_sharding_scope(self):
        from keras.src.backend.common import global_state

        initial_state = global_state.get_global_attribute(
            "enable_torch_sharding", False
        )
        with backend_dlib.sharding_scope():
            self.assertTrue(
                global_state.get_global_attribute("enable_torch_sharding")
            )
        self.assertEqual(
            global_state.get_global_attribute("enable_torch_sharding", False),
            initial_state,
        )

    def test_all_reduce_no_dist(self):
        tensor = torch.tensor([1.0, 2.0])
        result = backend_dlib.all_reduce(tensor, op="sum")
        self.assertTrue(torch.equal(tensor, result))

    def test_DTensorLayout(self):
        layout = backend_dlib.DTensorLayout("mesh", "placements")
        self.assertEqual(layout.device_mesh, "mesh")
        self.assertEqual(layout.placements, "placements")

    def test_to_backend_layout_none(self):
        self.assertIsNone(backend_dlib._to_backend_layout(None))

    def test_to_backend_layout(self):
        mesh = distribution_lib.DeviceMesh(
            (2, 2), ["batch", "model"], ["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
        )
        mesh._backend_mesh = "mock_torch_mesh"
        layout = distribution_lib.TensorLayout(["batch", None], mesh)

        backend_layout = backend_dlib._to_backend_layout(layout)
        self.assertIsInstance(backend_layout, backend_dlib.DTensorLayout)
        self.assertEqual(backend_layout.device_mesh, "mock_torch_mesh")
        self.assertEqual(len(backend_layout.placements), 2)

        from torch.distributed.tensor import Replicate
        from torch.distributed.tensor import Shard

        self.assertIsInstance(backend_layout.placements[0], Shard)
        self.assertEqual(backend_layout.placements[0].dim, 0)
        self.assertIsInstance(backend_layout.placements[1], Replicate)

    def test_distribute_tensor_none_layout(self):
        tensor = torch.tensor([1.0, 2.0])
        self.assertIs(backend_dlib.distribute_tensor(tensor, None), tensor)

    def test_distribute_variable_none_layout(self):
        variable = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        result = backend_dlib.distribute_variable(variable, None)
        self.assertIsInstance(result, torch.nn.Parameter)
        self.assertTrue(torch.equal(result, variable))

    def test_distribute_data_input_none_layout(self):
        tensor = torch.tensor([1.0, 2.0])
        self.assertIs(backend_dlib.distribute_data_input(tensor, None), tensor)
