"""Tests for PyTorch backend nn utilities."""

import numpy as np
import pytest
import torch
import torch.nn.functional as tnn
from absl.testing import parameterized

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.backend.torch.core import get_device
from keras.src.backend.torch.nn import conv


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="This test is only applicable to the PyTorch backend.",
)
class DotProductAttentionCompileTest(testing.TestCase):
    def _qkv(self, batch=2, seq=16, heads=4, head_dim=8):
        rng = np.random.default_rng(0)
        shape = (batch, seq, heads, head_dim)
        return tuple(
            ops.convert_to_tensor(rng.standard_normal(shape).astype("float32"))
            for _ in range(3)
        )

    @parameterized.named_parameters(
        ("no_options", False, False, False, None),
        ("is_causal", True, False, False, None),
        ("mask", False, True, False, None),
        ("mask_and_is_causal", True, True, False, None),
        ("bias", False, False, True, None),
        ("scale", False, False, False, 0.125),
    )
    def test_compiled_matches_eager(self, is_causal, use_mask, use_bias, scale):
        query, key, value = self._qkv()
        kwargs = {"is_causal": is_causal}
        if scale is not None:
            kwargs["scale"] = scale
        rng = np.random.default_rng(1)
        # `mask` and `bias` must broadcast to (batch, heads, q_len, kv_len).
        if use_mask:
            kwargs["mask"] = ops.convert_to_tensor(
                rng.integers(0, 2, (2, 1, 16, 16)).astype("bool")
            )
        if use_bias:
            kwargs["bias"] = ops.convert_to_tensor(
                rng.standard_normal((2, 1, 16, 16)).astype("float32")
            )

        # `fullgraph=True` turns a graph break into an error, so this also
        # covers the untraceable flash attention probe regressing back in.
        compiled = torch.compile(ops.dot_product_attention, fullgraph=True)
        self.assertAllClose(
            compiled(query, key, value, **kwargs),
            ops.dot_product_attention(query, key, value, **kwargs),
        )


def _reference_channels_last_conv(inputs, kernel):
    # Manual torch reference using a plain channels_first standard
    # contiguous conv, then permute the output back to channels_last.
    # channels_last is only a memory layout, so this must match the
    # backend conv numerically.
    ndim = inputs.ndim - 2
    if ndim == 2:
        x = torch.permute(inputs, (0, 3, 1, 2)).contiguous()
        w = torch.permute(kernel, (3, 2, 0, 1)).contiguous()
        out = tnn.conv2d(x, w, stride=1, padding=0, dilation=1)
        return torch.permute(out, (0, 2, 3, 1))
    x = torch.permute(inputs, (0, 4, 1, 2, 3)).contiguous()
    w = torch.permute(kernel, (4, 3, 0, 1, 2)).contiguous()
    out = tnn.conv3d(x, w, stride=1, padding=0, dilation=1)
    return torch.permute(out, (0, 2, 3, 4, 1))


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="This test is only applicable to the PyTorch backend.",
)
class TorchConvChannelsLastTest(testing.TestCase):
    def test_conv2d_channels_last_parity(self):
        device = get_device()
        inputs = torch.from_numpy(
            np.random.RandomState(0).randn(2, 8, 9, 3).astype("float32")
        ).to(device)
        kernel = torch.from_numpy(
            np.random.RandomState(1).randn(3, 3, 3, 5).astype("float32")
        ).to(device)

        outputs = conv(
            inputs,
            kernel,
            strides=1,
            padding="valid",
            data_format="channels_last",
        )
        reference = _reference_channels_last_conv(inputs, kernel)

        self.assertEqual(tuple(outputs.shape), (2, 6, 7, 5))
        self.assertTrue(torch.isfinite(outputs).all().cpu().numpy())
        self.assertAllClose(
            outputs.detach().cpu().numpy(),
            reference.detach().cpu().numpy(),
            atol=1e-5,
        )

    def test_conv3d_channels_last_parity(self):
        device = get_device()
        inputs = torch.from_numpy(
            np.random.RandomState(2).randn(2, 6, 7, 8, 3).astype("float32")
        ).to(device)
        kernel = torch.from_numpy(
            np.random.RandomState(3).randn(2, 2, 2, 3, 4).astype("float32")
        ).to(device)

        outputs = conv(
            inputs,
            kernel,
            strides=1,
            padding="valid",
            data_format="channels_last",
        )
        reference = _reference_channels_last_conv(inputs, kernel)

        self.assertEqual(tuple(outputs.shape), (2, 5, 6, 7, 4))
        self.assertTrue(torch.isfinite(outputs).all().cpu().numpy())
        self.assertAllClose(
            outputs.detach().cpu().numpy(),
            reference.detach().cpu().numpy(),
            atol=1e-5,
        )

    def test_conv2d_channels_last_noncontiguous_input(self):
        """A sliced/permuted (non-contiguous) input must still produce a
        correct result: the single-copy path must not skip a copy that
        was actually necessary to reach channels_last-contiguity."""
        device = get_device()
        base = torch.from_numpy(
            np.random.RandomState(6).randn(2, 8, 18, 3).astype("float32")
        ).to(device)
        # A strided slice along a spatial axis is not contiguous in any
        # memory format.
        inputs = base[:, :, ::2, :]
        self.assertFalse(inputs.is_contiguous())
        kernel = torch.from_numpy(
            np.random.RandomState(7).randn(3, 3, 3, 5).astype("float32")
        ).to(device)

        outputs = conv(
            inputs,
            kernel,
            strides=1,
            padding="valid",
            data_format="channels_last",
        )
        reference = _reference_channels_last_conv(inputs, kernel)

        self.assertAllClose(
            outputs.detach().cpu().numpy(),
            reference.detach().cpu().numpy(),
            atol=1e-5,
        )

    def test_conv2d_channels_first_unaffected(self):
        """channels_first must not be routed through the channels_last
        memory-format optimization."""
        device = get_device()
        inputs = torch.from_numpy(
            np.random.RandomState(8).randn(2, 3, 8, 9).astype("float32")
        ).to(device)
        kernel = torch.from_numpy(
            np.random.RandomState(9).randn(3, 3, 3, 5).astype("float32")
        ).to(device)

        outputs = conv(
            inputs,
            kernel,
            strides=1,
            padding="valid",
            data_format="channels_first",
        )
        reference = tnn.conv2d(
            inputs.contiguous(), kernel.permute(3, 2, 0, 1).contiguous()
        )

        self.assertAllClose(
            outputs.detach().cpu().numpy(),
            reference.detach().cpu().numpy(),
            atol=1e-5,
        )

    def test_conv1d_channels_last_format_is_noop(self):
        """1D conv has no torch memory-format equivalent to channels_last;
        the optimization must be a no-op there, not misapplied."""
        device = get_device()
        inputs = torch.from_numpy(
            np.random.RandomState(10).randn(2, 9, 3).astype("float32")
        ).to(device)
        kernel = torch.from_numpy(
            np.random.RandomState(11).randn(3, 3, 5).astype("float32")
        ).to(device)

        outputs = conv(
            inputs,
            kernel,
            strides=1,
            padding="valid",
            data_format="channels_last",
        )
        reference = torch.permute(
            tnn.conv1d(
                torch.permute(inputs, (0, 2, 1)).contiguous(),
                torch.permute(kernel, (2, 1, 0)).contiguous(),
            ),
            (0, 2, 1),
        )

        self.assertAllClose(
            outputs.detach().cpu().numpy(),
            reference.detach().cpu().numpy(),
            atol=1e-5,
        )

    def test_conv2d_channels_last_gradients_flow(self):
        device = get_device()
        inputs = torch.from_numpy(
            np.random.RandomState(4).randn(1, 5, 5, 2).astype("float32")
        ).to(device)
        inputs.requires_grad_(True)
        kernel = torch.from_numpy(
            np.random.RandomState(5).randn(3, 3, 2, 4).astype("float32")
        ).to(device)

        outputs = conv(
            inputs,
            kernel,
            strides=1,
            padding="valid",
            data_format="channels_last",
        )
        outputs.sum().backward()

        self.assertIsNotNone(inputs.grad)
        self.assertEqual(tuple(inputs.grad.shape), tuple(inputs.shape))
        self.assertTrue(torch.isfinite(inputs.grad).all().cpu().numpy())
