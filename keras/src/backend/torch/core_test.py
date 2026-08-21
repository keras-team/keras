"""Tests for PyTorch backend core utilities."""

import os

import numpy as np
import pytest
import torch
from absl.testing import parameterized
from torch.distributed.device_mesh import DeviceMesh as TorchDeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate

from keras.src import backend
from keras.src import testing
from keras.src.backend.torch import core as torch_core
from keras.src.backend.torch import distribution_lib
from keras.src.backend.torch.core import KerasDTensorPromotionMode
from keras.src.backend.torch.core import Variable
from keras.src.backend.torch.core import convert_to_tensor
from keras.src.backend.torch.core import slice as torch_slice
from keras.src.distribution.distribution_lib import DeviceMesh
from keras.src.distribution.distribution_lib import LayoutMap
from keras.src.distribution.distribution_lib import ModelParallel
from keras.src.distribution.distribution_lib import set_distribution


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


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Requires torch backend"
)
class TorchCoreDistributedTest(testing.TestCase):
    def set_env(self, key, value):
        old = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
        self.addCleanup(
            lambda: (
                os.environ.update({key: old})
                if old is not None
                else os.environ.pop(key, None)
            )
        )

    def tearDown(self):
        super().tearDown()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        from keras.src.backend.torch import core as torch_core

        if (
            getattr(torch_core, "_GLOBAL_DTENSOR_PROMOTION_MODE", None)
            is not None
        ):
            torch_core._GLOBAL_DTENSOR_PROMOTION_MODE.__exit__(None, None, None)
            torch_core._GLOBAL_DTENSOR_PROMOTION_MODE = None

    def _ensure_distributed_initialized(self, port="29500"):
        if not torch.distributed.is_initialized():
            self.set_env("MASTER_ADDR", "localhost")
            self.set_env("MASTER_PORT", port)
            distribution_lib.initialize(num_processes=1, process_id=0)

    def test_keras_dtensor_promotion_mode(self):
        self._ensure_distributed_initialized(port="29501")

        device_type = torch_core.get_device().split(":")[0]
        mesh = TorchDeviceMesh(device_type, np.array([0]))
        local_tensor = torch.ones((2, 2), device=device_type)
        dtensor = DTensor.from_local(
            local_tensor, device_mesh=mesh, placements=[Replicate()]
        )

        plain_tensor = torch.zeros((2, 2), device=device_type)

        with KerasDTensorPromotionMode():
            # Test simple addition
            result = dtensor + plain_tensor
            self.assertIsInstance(result, DTensor)
            self.assertTrue(
                torch.allclose(
                    result.to_local(),
                    torch.ones((2, 2), device=device_type),
                )
            )

            # Test in-place
            dtensor += plain_tensor
            self.assertIsInstance(dtensor, DTensor)

            # Test nested structures
            result = torch.addcmul(dtensor, plain_tensor, dtensor, value=0.5)
            self.assertIsInstance(result, DTensor)

    def test_convert_to_tensor_pushes_dtensor_mode(self):
        self._ensure_distributed_initialized(port="29502")

        device_type = torch_core.get_device().split(":")[0]
        mesh = TorchDeviceMesh(device_type, np.array([0]))
        local_tensor = torch.ones((2, 2), device=device_type)
        dtensor = DTensor.from_local(
            local_tensor, device_mesh=mesh, placements=[Replicate()]
        )

        if torch_core._GLOBAL_DTENSOR_PROMOTION_MODE is not None:
            torch_core._GLOBAL_DTENSOR_PROMOTION_MODE.__exit__(None, None, None)
            torch_core._GLOBAL_DTENSOR_PROMOTION_MODE = None

        convert_to_tensor(dtensor)

        self.assertIsNotNone(torch_core._GLOBAL_DTENSOR_PROMOTION_MODE)

        plain_tensor = torch.zeros((2, 2), device=device_type)
        result = dtensor + plain_tensor
        self.assertIsInstance(result, DTensor)
        self.assertTrue(
            torch.allclose(
                result.to_local(), torch.ones((2, 2), device=device_type)
            )
        )

        if torch_core._GLOBAL_DTENSOR_PROMOTION_MODE is not None:
            torch_core._GLOBAL_DTENSOR_PROMOTION_MODE.__exit__(None, None, None)
            torch_core._GLOBAL_DTENSOR_PROMOTION_MODE = None

    def test_convert_to_numpy_dtensor(self):
        self._ensure_distributed_initialized(port="29508")
        device_type = torch_core.get_device().split(":")[0]
        mesh = TorchDeviceMesh(device_type, np.array([0]))
        dtensor = DTensor.from_local(
            torch.ones((2, 2), device=device_type),
            device_mesh=mesh,
            placements=[Replicate()],
        )

        nv = torch_core.convert_to_numpy(dtensor)
        self.assertIsInstance(nv, np.ndarray)
        self.assertEqual(nv.shape, (2, 2))
        self.assertTrue(np.allclose(nv, 1.0))

    @parameterized.parameters(
        ("callable",),
        ("parameter",),
        ("tensor_with_grad",),
    )
    def test_variable_initialize_distributed(self, init_type):
        self._ensure_distributed_initialized()

        mesh = DeviceMesh(
            shape=(1,),
            axis_names=["x"],
            devices=np.array([distribution_lib.list_devices()[0]]),
        )

        layout_map = LayoutMap(mesh)
        layout_map[".*"] = ("x", None)
        dist = ModelParallel(layout_map=layout_map)

        set_distribution(dist)

        if init_type == "callable":

            def initializer(shape, dtype):
                return torch.ones(shape)

            v = Variable(initializer, shape=(2, 2), dtype="float32")
        elif init_type == "parameter":
            v = Variable(torch.nn.Parameter(torch.ones((2, 2))))
        elif init_type == "tensor_with_grad":
            v = Variable(torch.ones((2, 2), requires_grad=True))

        self.assertIsInstance(v.value, torch.nn.Parameter)
        self.assertIsInstance(v.value.data, DTensor)
        self.assertEqual(v.value.device.type, mesh.backend_mesh.device_type)

    def test_variable_direct_assign(self):
        self._ensure_distributed_initialized(port="29505")

        mesh = DeviceMesh(
            shape=(1,),
            axis_names=["x"],
            devices=np.array([distribution_lib.list_devices()[0]]),
        )

        layout_map = LayoutMap(mesh)
        layout_map[".*"] = ("x", None)
        dist = ModelParallel(layout_map=layout_map)

        set_distribution(dist)

        v = Variable(torch.ones((2, 2)))

        self.assertIsInstance(v.value, torch.nn.Parameter)
        self.assertIsInstance(v.value.data, DTensor)

        new_val = torch.zeros((2, 2))
        v._direct_assign(new_val)

        self.assertIsInstance(v.value.data, DTensor)
        self.assertTrue(
            torch.allclose(
                v.value.data.to_local(),
                torch.zeros((2, 2), device=v.value.data.to_local().device),
            )
        )
