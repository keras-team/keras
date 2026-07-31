"""Tests for PyTorch backend core utilities."""

import os

import numpy as np
import pytest
import torch

from keras.src import backend
from keras.src import testing
from keras.src.backend.torch import distribution_lib
from keras.src.backend.torch.core import Variable
from keras.src.backend.torch.core import convert_to_tensor
from keras.src.backend.torch.core import slice as torch_slice
from keras.src.distribution import distribution_lib as dist_lib


def _get_backed_symint(hint=2):
    """Create a backed SymInt via torch.export dynamic shapes."""

    class _M(torch.nn.Module):
        def forward(self, x):
            return x + x.shape[0]

    ep = torch.export.export(
        _M(),
        (torch.randn(hint, 3),),
        dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
    )
    for node in ep.graph.nodes:
        if node.op == "placeholder":
            return node.meta["val"].shape[0]
    raise RuntimeError("Could not extract SymInt from exported program")


def _get_backed_symfloat(hint=2):
    """Create a backed SymFloat from a backed SymInt."""
    return torch.sym_float(_get_backed_symint(hint))


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Requires torch backend"
)
class TorchCoreTest(testing.TestCase):
    def _assert_sym_convert(
        self,
        value,
        expected_dtype,
        expected_item=None,
        expected_shape=None,
        expected_values=None,
        dtype=None,
    ):
        result = convert_to_tensor(value, dtype=dtype)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, expected_dtype)
        if expected_item is not None:
            self.assertEqual(result.item(), expected_item)
        if expected_shape is not None:
            self.assertEqual(tuple(result.shape), expected_shape)
        if expected_values is not None:
            self.assertListEqual(result.tolist(), expected_values)

    def test_convert_to_tensor_symint_scalar(self):
        self._assert_sym_convert(
            _get_backed_symint(5), torch.int64, expected_item=5
        )

    def test_convert_to_tensor_symfloat_scalar(self):
        self._assert_sym_convert(
            _get_backed_symfloat(5), torch.float32, expected_item=5.0
        )

    def test_convert_to_tensor_list_of_symint(self):
        self._assert_sym_convert(
            [_get_backed_symint(3), _get_backed_symint(4)],
            torch.int64,
            expected_shape=(2,),
            expected_values=[3, 4],
        )

    def test_convert_to_tensor_tuple_of_symfloat(self):
        self._assert_sym_convert(
            (_get_backed_symfloat(3), _get_backed_symfloat(4)),
            torch.float32,
            expected_shape=(2,),
            expected_values=[3.0, 4.0],
        )

    def test_convert_to_tensor_nested_list_of_symint(self):
        self._assert_sym_convert(
            [[_get_backed_symint(3), _get_backed_symint(4)]],
            torch.int64,
            expected_shape=(1, 2),
            expected_values=[[3, 4]],
        )

    def test_convert_to_tensor_explicit_dtype_for_symint(self):
        self._assert_sym_convert(
            _get_backed_symint(5),
            torch.float32,
            dtype="float32",
        )

    def test_slice_fast_path_accepts_symint(self):
        """slice fast path should accept SymInt without crashing."""
        x = torch.arange(24).reshape(2, 3, 4)
        batch = _get_backed_symint(2)
        start_indices = [0, 0, 0]
        shape = [batch, 2, 2]
        result = torch_slice(x, start_indices, shape)
        self.assertEqual(tuple(result.shape), (2, 2, 2))

    def test_variable_with_layout(self):
        if not torch.distributed.is_initialized():
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29516"
            distribution_lib.initialize(num_processes=1, process_id=0)
            self.addCleanup(
                lambda: (
                    torch.distributed.destroy_process_group()
                    if torch.distributed.is_initialized()
                    else None
                )
            )

        mesh = dist_lib.DeviceMesh(
            shape=(1,), axis_names=["x"], devices=np.array(["cpu:0"])
        )
        # Replicated layout just to avoid sharding complexity in single process
        layout = dist_lib.TensorLayout(axes=(None, None), device_mesh=mesh)

        # Test initialization with layout
        v = Variable(
            initializer=np.ones((2, 2), dtype="float32"), layout=layout
        )
        self.assertTrue(hasattr(v, "_layout"))
        self.assertEqual(v._layout, layout)

        # Test initialization with layout from Parameter
        param = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float32))
        v2 = Variable(initializer=param, layout=layout)
        self.assertTrue(hasattr(v2, "_layout"))
        self.assertEqual(v2._layout, layout)

        # Test direct assignment with layout
        v.assign(np.zeros((2, 2), dtype="float32"))
        self.assertEqual(v.numpy().sum(), 0)

    def test_variable_model_parallel_initialization(self):
        if not torch.distributed.is_initialized():
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29517"
            distribution_lib.initialize(num_processes=1, process_id=0)
            self.addCleanup(
                lambda: (
                    torch.distributed.destroy_process_group()
                    if torch.distributed.is_initialized()
                    else None
                )
            )

        mesh = dist_lib.DeviceMesh(
            shape=(1,), axis_names=["x"], devices=np.array(["cpu:0"])
        )
        layout = dist_lib.TensorLayout(axes=("x", None), device_mesh=mesh)

        layout_map = dist_lib.LayoutMap(mesh)
        layout_map[".*"] = layout

        dist = dist_lib.ModelParallel(
            device_mesh=mesh, layout_map=layout_map, batch_dim_name="batch"
        )

        dist_lib.set_distribution(dist)
        self.addCleanup(lambda: dist_lib.set_distribution(None))

        # 1. Test initialization with numpy array (hits the 'else' branch)
        v1 = Variable(initializer=np.ones((2, 2), dtype="float32"))
        self.assertTrue(hasattr(v1.value, "placements"))
        self.assertEqual(v1._layout, layout)

        # 2. Test initialization with torch.nn.Parameter
        param = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float32))
        v2 = Variable(initializer=param)
        self.assertTrue(hasattr(v2.value, "placements"))
        self.assertEqual(v2._layout, layout)

        # 3. Test _initialize_with_initializer (callable)
        def my_initializer(shape, dtype=None):
            return torch.ones(shape, dtype=torch.float32)

        v3 = Variable(initializer=my_initializer, shape=(4, 4), dtype="float32")
        self.assertTrue(hasattr(v3.value, "placements"))
        self.assertEqual(v3.value.shape, (4, 4))
        self.assertEqual(v3.numpy().sum(), 16)

        # 4. Test _initialize with meta tensor
        meta_tensor = torch.empty((4, 4), device="meta")
        v4 = Variable(initializer=meta_tensor, dtype="float32")
        self.assertTrue(hasattr(v4.value, "placements"))
        self.assertEqual(v4.value.shape, (4, 4))
        # It should have been moved to real device (CPU in this test)
        self.assertEqual(v4.value.to_local().device.type, "cpu")

        # 5. Test assign with numpy array
        v1.assign(np.zeros((2, 2), dtype="float32"))
        self.assertTrue(hasattr(v1.value, "placements"))
        self.assertEqual(v1.numpy().sum(), 0)

        # 6. Test assign with torch tensor requiring grad
        new_value = torch.ones((2, 2), requires_grad=True)
        v1.assign(new_value)
        self.assertTrue(hasattr(v1.value, "placements"))
        self.assertEqual(v1.numpy().sum(), 4)
