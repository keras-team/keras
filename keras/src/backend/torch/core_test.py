"""Tests for PyTorch backend core utilities."""

import numpy as np
import pytest
import torch

from keras.src import backend
from keras.src import testing
from keras.src.backend.torch.core import DEFAULT_DEVICE
from keras.src.backend.torch.core import Variable
from keras.src.backend.torch.core import cast
from keras.src.backend.torch.core import convert_to_tensor
from keras.src.backend.torch.core import device_scope
from keras.src.backend.torch.core import get_device
from keras.src.backend.torch.core import slice as torch_slice
from keras.src.backend.torch.core import slice_update


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

    def test_get_device_reflects_device_scope(self):
        """get_device() returns the default device outside any scope and the
        scoped device inside device_scope."""
        self.assertEqual(get_device(), DEFAULT_DEVICE)
        with device_scope("cpu:0"):
            self.assertEqual(get_device(), "cpu:0")
        self.assertEqual(get_device(), DEFAULT_DEVICE)

    # ---------------------------------------------------------------
    # RC10/T2: fast-path tests for convert_to_tensor and cast
    # ---------------------------------------------------------------

    def test_convert_to_tensor_tensor_no_op_is_identity(self):
        """Tensor on default device, dtype=None → same object returned."""
        t = torch.tensor([1.0, 2.0], dtype=torch.float32, device=get_device())
        r = convert_to_tensor(t, dtype=None)
        self.assertIs(r, t)

    def test_convert_to_tensor_parameter_no_op_is_identity(self):
        """nn.Parameter (subclass of Tensor) on default device, dtype=None
        → same object."""
        p = torch.nn.Parameter(
            torch.tensor([1.0, 2.0], dtype=torch.float32, device=get_device())
        )
        r = convert_to_tensor(p, dtype=None)
        self.assertIs(r, p)

    def test_convert_to_tensor_dtype_change_is_not_identity(self):
        """Tensor requiring a dtype cast → new object with correct dtype."""
        t = torch.tensor([1.0, 2.0], dtype=torch.float32)
        r = convert_to_tensor(t, dtype="float64")
        self.assertIsNot(r, t)
        self.assertEqual(r.dtype, torch.float64)

    def test_convert_to_tensor_variable_no_op_is_identity(self):
        """Variable with dtype=None → same tensor object as var.value."""
        var = Variable(
            torch.tensor([1.0, 2.0], dtype=torch.float32), trainable=False
        )
        r = convert_to_tensor(var, dtype=None)
        self.assertIs(r, var.value)

    def test_convert_to_tensor_numpy_float32_dtype(self):
        """numpy float32 array → torch.float32 tensor on the default device."""
        arr = np.array([1.0, 2.0], dtype=np.float32)
        r = convert_to_tensor(arr)
        self.assertEqual(r.dtype, torch.float32)
        self.assertEqual(r.device.type, torch.device(get_device()).type)

    def test_convert_to_tensor_numpy_int32_dtype(self):
        """numpy int32 array → torch.int32 tensor."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        r = convert_to_tensor(arr)
        self.assertEqual(r.dtype, torch.int32)

    def test_convert_to_tensor_python_int_small(self):
        """Python int within int32 range → torch.int32."""
        r = convert_to_tensor(42)
        self.assertEqual(r.dtype, torch.int32)

    def test_convert_to_tensor_python_int_large(self):
        """Python int at/above 2^31 → torch.int64."""
        r = convert_to_tensor(2**31)
        self.assertEqual(r.dtype, torch.int64)

    def test_convert_to_tensor_python_float(self):
        """Python float → floatx() dtype (float32 by default)."""
        r = convert_to_tensor(3.14)
        self.assertEqual(r.dtype, torch.float32)

    def test_convert_to_tensor_python_bool(self):
        """Python bool → torch.bool."""
        r = convert_to_tensor(True)
        self.assertEqual(r.dtype, torch.bool)

    def test_convert_to_tensor_list(self):
        """list of floats → torch.float32."""
        r = convert_to_tensor([1.0, 2.0])
        self.assertEqual(r.dtype, torch.float32)

    def test_convert_to_tensor_device_mismatch_moved(self):
        """Tensor on a different-named device → moved to target device."""

        t = torch.tensor([1.0], device="cpu")
        with device_scope("meta"):
            r = convert_to_tensor(t)
        self.assertEqual(r.device.type, "meta")

    def test_cast_same_dtype_is_identity(self):
        """cast with matching dtype → same object."""
        t = torch.tensor([1.0, 2.0], dtype=torch.float32)
        r = cast(t, "float32")
        self.assertIs(r, t)

    def test_cast_dtype_change(self):
        """cast with different dtype → new tensor, correct dtype."""
        t = torch.tensor([1.0, 2.0], dtype=torch.float32)
        r = cast(t, "float64")
        self.assertIsNot(r, t)
        self.assertEqual(r.dtype, torch.float64)

    def test_cast_variable_same_dtype_is_identity(self):
        """cast(Variable, same dtype) → same tensor object after unwrap."""
        var = Variable(
            torch.tensor([1.0, 2.0], dtype=torch.float32), trainable=False
        )
        r = cast(var, "float32")
        self.assertIs(r, var.value)

    def test_cast_parameter_same_dtype_is_identity(self):
        """cast(nn.Parameter, same dtype) → same object (hits Tensor branch)."""
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
        r = cast(p, "float32")
        self.assertIs(r, p)

    def test_compile_numpy_input_does_not_raise(self):
        """torch.compile with numpy input must not raise
        (dynamo-disable guard)."""
        import numpy as np

        def fn(np_arr):
            return convert_to_tensor(np_arr) * 2

        compiled = torch.compile(fn)
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = compiled(arr)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.tolist(), [2.0, 4.0, 6.0])

    def test_convert_to_tensor_sparse_raises(self):
        """convert_to_tensor(tensor, sparse=True) must raise ValueError."""
        t = torch.tensor([1.0])
        with self.assertRaises(ValueError):
            convert_to_tensor(t, sparse=True)

    def test_convert_to_tensor_ragged_raises(self):
        """convert_to_tensor(tensor, ragged=True) must raise ValueError."""
        t = torch.tensor([1.0])
        with self.assertRaises(ValueError):
            convert_to_tensor(t, ragged=True)

    def test_slice_with_zero_d_tensor_start_indices(self):
        """slice fast path must accept 0-d integer tensors as start indices."""
        device = get_device()
        x = torch.arange(10, dtype=torch.float32).reshape(2, 5).to(device)
        out = torch_slice(
            x,
            [torch.tensor(0).to(device), torch.tensor(1).to(device)],
            [2, 3],
        )
        expected = x[0:2, 1:4]
        self.assertAllClose(out.cpu().numpy(), expected.cpu().numpy())

    def test_slice_export_preserves_dynamic_dim(self):
        """slice() must keep a dynamic dim symbolic under torch.export.

        Passing a dynamic dimension through slice() as a length must not
        specialize it. Calling int() on the SymInt would pin it to a constant
        (#22998), which fails export with a constraints-violated error and
        stops the exported program from accepting other batch sizes.
        """

        class _SliceModule(torch.nn.Module):
            def forward(self, x):
                n = x.shape[0]
                return torch_slice(x, [0, 0], [n, 2])

        ep = torch.export.export(
            _SliceModule(),
            (torch.arange(8).reshape(2, 4),),
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=2)}},
        )
        # Re-running with a different batch size must work; a specialized dim
        # would have failed export above or fixed the first output dim to 2.
        out = ep.module()(torch.arange(12).reshape(3, 4))
        self.assertEqual(tuple(out.shape), (3, 2))

    def test_slice_update_1d_correctness(self):
        """slice_update on a 1-D tensor produces the correct values."""
        device = get_device()
        x = torch.zeros(8, dtype=torch.float32).to(device)
        updates = torch.tensor([9.0, 10.0, 11.0, 12.0]).to(device)
        out = slice_update(x, [1], updates)
        expected = torch.tensor([0.0, 9.0, 10.0, 11.0, 12.0, 0.0, 0.0, 0.0])
        self.assertAllClose(out.cpu().numpy(), expected.numpy())

    def test_slice_update_2d_correctness(self):
        """slice_update on a 2-D tensor at a non-zero start position."""
        device = get_device()
        x = torch.ones(4, 3, dtype=torch.float32).to(device)
        updates = torch.full((2, 3), 5.0).to(device)
        out = slice_update(x, [1, 0], updates)
        expected = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [5.0, 5.0, 5.0],
                [5.0, 5.0, 5.0],
                [1.0, 1.0, 1.0],
            ]
        )
        self.assertAllClose(out.cpu().numpy(), expected.numpy())

    def test_slice_update_nd_offset_correctness(self):
        """slice_update with an interior start position on an N-D tensor."""
        device = get_device()
        x = torch.zeros(3, 6, dtype=torch.float32).to(device)
        updates = torch.ones(2, 3).to(device)
        out = slice_update(x, [1, 2], updates)
        self.assertAllClose(
            out[1:3, 2:5].cpu().numpy(), torch.ones(2, 3).numpy()
        )
        # Untouched region must remain zero.
        self.assertAllClose(out[0, :].cpu().numpy(), torch.zeros(6).numpy())
