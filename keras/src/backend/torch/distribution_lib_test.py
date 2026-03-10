"""Tests for distribution_lib.py."""

from unittest import mock

import pytest
import torch

from keras.src import backend
from keras.src import testing
from keras.src.backend.torch import distribution_lib
from keras.src.distribution import distribution_lib as keras_distribution_lib


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="Backend specific test",
)
class TorchDistributionLibTest(testing.TestCase):
    def test_list_devices(self):
        devices = distribution_lib.list_devices("cpu")
        self.assertEqual(devices, ["cpu:0"])

    def test_get_device_count(self):
        count = distribution_lib.get_device_count("cpu")
        self.assertEqual(count, 1)

    def test_num_processes(self):
        self.assertEqual(distribution_lib.num_processes(), 1)

    def test_process_id(self):
        self.assertEqual(distribution_lib.process_id(), 0)

    @mock.patch("keras.src.backend.torch.distribution_lib.init_device_mesh")
    def test_to_backend_mesh(self, mock_init_device_mesh):
        mesh = keras_distribution_lib.DeviceMesh(
            shape=(1,), axis_names=["data"], devices=["cpu:0"]
        )
        distribution_lib._to_backend_mesh(mesh)
        mock_init_device_mesh.assert_called_once()

    def test_to_backend_layout(self):
        mock_backend_mesh = mock.MagicMock()
        mock_backend_mesh.device_type = "cpu"
        
        mesh = keras_distribution_lib.DeviceMesh(
            shape=(2,), axis_names=["model"], devices=["cpu:0", "cpu:0"]
        )
        mesh._backend_mesh = mock_backend_mesh
        
        layout = keras_distribution_lib.TensorLayout(axes=["model", None], device_mesh=mesh)
        
        torch_mesh, placements = distribution_lib._to_backend_layout(layout)
        
        self.assertEqual(torch_mesh, mock_backend_mesh)
        self.assertEqual(len(placements), 1)
        from torch.distributed.tensor import Shard
        self.assertIsInstance(placements[0], Shard)
        self.assertEqual(placements[0].dim, 0)

    @mock.patch("keras.src.backend.torch.distribution_lib.distribute_tensor")
    def test_distribute_variable(self, mock_distribute_tensor):
        value = torch.nn.Parameter(torch.randn(4, 4))
        layout = mock.MagicMock()
        mock_distribute_tensor.return_value = value.data
        
        res = distribution_lib.distribute_variable(value, layout)
        self.assertIsInstance(res, torch.nn.Parameter)
        mock_distribute_tensor.assert_called_once_with(value, layout)

    def test_infer_parallel_style(self):
        from keras.src.backend.torch.core import Variable
        from keras.src.distribution import LayoutMap, TensorLayout
        
        mesh = keras_distribution_lib.DeviceMesh(
            shape=(2,), axis_names=["model"], devices=["cpu:0", "cpu:0"]
        )
        
        # Test ColwiseParallel
        layout_map = LayoutMap(mesh)
        layout_map["kernel_col"] = TensorLayout([None, "model"])
        variable = mock.MagicMock(spec=Variable)
        variable.path = "kernel_col"
        from torch.distributed.tensor.parallel import ColwiseParallel
        style = distribution_lib._infer_parallel_style(variable, layout_map, "kernel")
        self.assertIsInstance(style, ColwiseParallel)

        # Test RowwiseParallel
        layout_map = LayoutMap(mesh)
        layout_map["kernel_row"] = TensorLayout(["model", None])
        variable.path = "kernel_row"
        from torch.distributed.tensor.parallel import RowwiseParallel
        style = distribution_lib._infer_parallel_style(variable, layout_map, "kernel")
        self.assertIsInstance(style, RowwiseParallel)
