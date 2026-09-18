"""Tests for PyTorch backend nn utilities."""

import numpy as np
import pytest
import torch
from absl.testing import parameterized

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.backend.torch.core import get_device
from keras.src.backend.torch.nn import _mask_invalid_class_indices

# Every accelerator present is tested alongside CPU: the bounds checking of
# `nll_loss` / `cross_entropy` is device-dependent.
DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")
if torch.cuda.is_available():
    DEVICES.append("cuda")


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


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="This test is only applicable to the PyTorch backend.",
)
class SparseCategoricalCrossentropyBoundsTest(testing.TestCase):
    """Out-of-range class indices must never pass silently.

    `nll_loss` / `cross_entropy` only validate class indices on CPU, where
    they raise. The MPS kernels performed no check and returned a garbage
    loss, and the CUDA ones fired a device-side assert that poisons the
    context. Off CPU the offending indices are now replaced with 0 and their
    losses marked NaN, which avoids a host synchronization on every step and
    matches the contract the TensorFlow backend already exposes.
    """

    # 3 classes.
    PROBS = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]], dtype="float32")
    LOGITS = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0]], dtype="float32")

    def assertInvalidIndex(self, device, fn, index):
        """CPU raises with the offending index, accelerators return NaN."""
        if device == "cpu":
            with self.assertRaisesRegex(
                IndexError, f"Target {index} is out of bounds."
            ):
                fn()
            return None
        result = ops.convert_to_numpy(fn())
        self.assertTrue(np.isnan(result).any())
        return result

    @parameterized.named_parameters(
        (
            f"{device}_{name}_{str(index).replace('-', 'neg')}",
            device,
            from_logits,
            index,
        )
        for device in DEVICES
        for name, from_logits in (("probs", False), ("logits", True))
        for index in (3, 5, -1)
    )
    def test_out_of_range_target(self, device, from_logits, index):
        output = self.LOGITS if from_logits else self.PROBS
        with backend.device(device):
            result = self.assertInvalidIndex(
                device,
                lambda: ops.sparse_categorical_crossentropy(
                    np.array([0, index]), output, from_logits=from_logits
                ),
                index,
            )
        if result is not None:
            # Only the invalid entry is poisoned; the valid one is intact.
            self.assertFalse(np.isnan(result[0]))
            self.assertTrue(np.isnan(result[1]))

    @parameterized.named_parameters((device, device) for device in DEVICES)
    def test_out_of_range_target_with_axis(self, device):
        target = np.array([[0, 1], [7, 2]])
        # Class dim already at axis 1: the `movedim` branch is skipped.
        channels_first = np.tile(self.PROBS[:, :, None], (1, 1, 2))
        # Class dim at the end: it is moved to axis 1 before the loss, so
        # this also checks the class count is read after the transpose.
        channels_last = np.tile(self.PROBS[:, None, :], (1, 2, 1))
        with backend.device(device):
            self.assertInvalidIndex(
                device,
                lambda: ops.sparse_categorical_crossentropy(
                    target, channels_first, axis=1
                ),
                7,
            )
            self.assertInvalidIndex(
                device,
                lambda: ops.sparse_categorical_crossentropy(
                    target, channels_last, axis=-1
                ),
                7,
            )

    @parameterized.named_parameters((device, device) for device in DEVICES)
    def test_in_range_target_is_unaffected(self, device):
        logits = self.LOGITS[0]
        expected_1d = np.log(np.sum(np.exp(logits))) - logits[2]
        with backend.device(device):
            self.assertAllClose(
                ops.sparse_categorical_crossentropy(
                    np.array([0, 1]), self.PROBS
                ),
                -np.log([0.9, 0.8]),
            )
            # A 1D `output` goes through a separate code path.
            self.assertAllClose(
                ops.sparse_categorical_crossentropy(
                    np.array(2), logits, from_logits=True
                ),
                expected_1d,
            )

    @parameterized.named_parameters((device, device) for device in DEVICES)
    def test_ignore_index_is_not_out_of_bounds(self, device):
        # `-100` is the default `ignore_index` of the underlying torch ops, so
        # the CPU kernels skip those entries instead of erroring out. This
        # asserts the accelerators stay consistent with CPU; it is not a Keras
        # API, which expresses this through `losses(ignore_class=...)`.
        with backend.device(device):
            self.assertAllClose(
                ops.sparse_categorical_crossentropy(
                    np.array([0, -100]), self.PROBS
                ),
                [-np.log(0.9), 0.0],
            )

    @parameterized.named_parameters((device, device) for device in DEVICES)
    def test_empty_target_is_allowed(self, device):
        with backend.device(device):
            result = ops.sparse_categorical_crossentropy(
                np.array([], dtype="int32"), np.zeros((0, 3), dtype="float32")
            )
        self.assertEqual(tuple(result.shape), (0,))

    def test_mask_invalid_class_indices(self):
        # Keras CI is CPU-only, where every assertion above collapses onto the
        # `IndexError` path. The helper is therefore exercised directly, on a
        # `meta` tensor so it runs on any machine and takes the non-CPU branch.
        target = torch.tensor([0, 5, -1, -100, 2], device="meta")
        masked, invalid = _mask_invalid_class_indices(target, 3)
        self.assertIsNotNone(invalid)
        self.assertEqual(tuple(invalid.shape), (5,))
        self.assertEqual(tuple(masked.shape), (5,))
        # Same computation on CPU data, to assert on actual values.
        target = torch.tensor([0, 5, -1, -100, 2])
        invalid = ~((target == -100) | ((target >= 0) & (target < 3)))
        self.assertAllClose(invalid.numpy(), [False, True, True, False, False])
        # Out-of-range entries are replaced with 0, not clamped to
        # `num_classes - 1`, and `-100` is left untouched for `ignore_index`.
        self.assertAllClose(
            target.masked_fill(invalid, 0).numpy(), [0, 0, 0, -100, 2]
        )

    def test_invalid_target_contributes_no_gradient(self):
        # `masked_fill` passes no gradient through the poisoned entries, so
        # an invalid label can neither pull the weights toward the substituted
        # class nor turn them into NaN: only the reported loss is NaN.
        device = DEVICES[-1]
        if device == "cpu":
            self.skipTest("Requires an accelerator; CPU raises instead.")
        with backend.device(device):
            logits = ops.convert_to_tensor(self.LOGITS)
            logits.requires_grad_(True)
            loss = ops.sparse_categorical_crossentropy(
                np.array([0, 5]), logits, from_logits=True
            )
            loss.mean().backward()
            grad = ops.convert_to_numpy(logits.grad)
        self.assertTrue(np.isfinite(grad).all())
        self.assertAllClose(grad[1], np.zeros(3))
        self.assertGreater(np.abs(grad[0]).sum(), 0.0)

    def test_meta_device_shape_inference(self):
        # Meta tensors hold no data, so the masking must stay free of any
        # host read for shape inference to work.
        with backend.device("meta"):
            result = ops.sparse_categorical_crossentropy(
                np.array([0, 1]), self.PROBS
            )
        self.assertEqual(tuple(result.shape), (2,))

    def test_compiled_matches_eager(self):
        # The masking has no data-dependent branch, so it is traceable and
        # compiled models get the same protection. `fullgraph=True` turns a
        # graph break into an error.
        compiled = torch.compile(
            ops.sparse_categorical_crossentropy, fullgraph=True
        )
        output = ops.convert_to_tensor(self.PROBS)
        self.assertAllClose(
            compiled(ops.convert_to_tensor(np.array([0, 1])), output),
            -np.log([0.9, 0.8]),
        )
        if get_device() != "cpu":
            result = ops.convert_to_numpy(
                compiled(ops.convert_to_tensor(np.array([0, 5])), output)
            )
            self.assertFalse(np.isnan(result[0]))
            self.assertTrue(np.isnan(result[1]))
