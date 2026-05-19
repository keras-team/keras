"""Tests for PyTorch backend core utilities."""

import pytest
import torch

from keras.src import backend
from keras.src import testing
from keras.src.backend.torch.core import convert_to_tensor
from keras.src.backend.torch.core import slice as torch_slice


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
    def test_convert_to_tensor_symint_scalar(self):
        s = _get_backed_symint(5)
        result = convert_to_tensor(s)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.dtype, torch.int64)
        self.assertEqual(result.item(), 5)

    def test_convert_to_tensor_symfloat_scalar(self):
        s = _get_backed_symfloat(5)
        result = convert_to_tensor(s)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.item(), 5.0)

    def test_convert_to_tensor_list_of_symint(self):
        s1 = _get_backed_symint(3)
        s2 = _get_backed_symint(4)
        result = convert_to_tensor([s1, s2])
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.dtype, torch.int64)
        self.assertListEqual(result.tolist(), [3, 4])

    def test_convert_to_tensor_tuple_of_symfloat(self):
        s1 = _get_backed_symfloat(3)
        s2 = _get_backed_symfloat(4)
        result = convert_to_tensor((s1, s2))
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.dtype, torch.float32)
        self.assertListEqual(result.tolist(), [3.0, 4.0])

    def test_convert_to_tensor_explicit_dtype_for_symint(self):
        s = _get_backed_symint(5)
        result = convert_to_tensor(s, dtype="float32")
        self.assertEqual(result.dtype, torch.float32)

    def test_slice_fast_path_accepts_symint(self):
        """slice fast path should accept SymInt without crashing."""
        x = torch.arange(24).reshape(2, 3, 4)
        batch = _get_backed_symint(2)
        start_indices = [0, 0, 0]
        shape = [batch, 2, 2]
        result = torch_slice(x, start_indices, shape)
        self.assertEqual(tuple(result.shape), (2, 2, 2))
