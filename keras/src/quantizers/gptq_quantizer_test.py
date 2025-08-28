# tests/test_gptq_quant_ops.py
import numpy as np
from absl.testing import parameterized

from keras import ops
from keras.src import testing
from keras.src.quantizers.gptq_quantizer import compute_scale_zero
from keras.src.quantizers.gptq_quantizer import dequantize
from keras.src.quantizers.gptq_quantizer import quantize


class QuantizersTest(testing.TestCase):
    @parameterized.named_parameters(
        ("bits_2_sym_False", 2, False),
        ("bits_4_sym_False", 4, False),
        ("bits_8_sym_False", 8, False),
        ("bits_2_sym_True", 2, True),
        ("bits_4_sym_True", 4, True),
        ("bits_8_sym_True", 8, True),
    )
    def test_quantize_dequantize_roundtrip_error_bound_per_tensor(
        self, bits, symmetric
    ):
        """
        For finite inputs and positive scales, the reconstruction error
        |x_hat - clip(x)| is bounded by 0.5 * scale elementwise.
        """
        rng = np.random.default_rng(0)
        x = ops.array(rng.standard_normal((64, 32)), "float32")
        scale = ops.array(0.05)  # per-tensor scale
        maxq = ops.array(ops.subtract(ops.power(2, bits), 1), "float32")
        zero = ops.array(maxq / 2.0 if symmetric else 3.0, "float32")

        quantized = quantize(x, scale, zero, maxq)
        dequantized = dequantize(quantized, scale, zero)

        # Representable dequantization range:
        # [scale*(0 - zero), scale*(maxq - zero)]
        lo = ops.multiply(scale, ops.subtract(ops.array(0.0), zero))
        hi = ops.multiply(scale, ops.subtract(maxq, zero))
        x_clipped = ops.clip(x, lo, hi)

        err = ops.abs(dequantized - x_clipped)
        self.assertTrue(
            ops.all(err <= (ops.add(ops.multiply(0.5, scale), 1e-7)))
        )

    def test_quantize_clipping_behavior_extremes(self):
        """
        Very negative q == 0 ; very positive q == maxq.
        """
        maxq = ops.array(15.0)
        scale = ops.array(0.1)
        zero = ops.array(7.0)

        x = ops.array([[-1e6, 1e6]], "float32")
        quantized = quantize(x, scale, zero, maxq)

        self.assertEqual(quantized.shape, (1, 2))
        self.assertEqual(quantized[0, 0], 0.0)
        self.assertEqual(quantized[0, 1], maxq)

    def test_zero_scale_guard_no_nans_for_finite_inputs(self):
        """
        If scale == 0, quantize should not produce NaNs (uses epsilon
        replacement).
        """
        x = ops.array([[0.0, 1.0, -2.0]])
        scale = ops.array(0.0)  # triggers epsilon path
        zero = ops.array(5.0)
        maxq = ops.array(15.0)

        q = quantize(x, scale, zero, maxq)
        self.assertFalse(ops.any(ops.isnan(q)))

        # Dequantize should also be finite
        x_hat = dequantize(q, scale, zero)
        self.assertTrue(ops.all(ops.isfinite(x_hat)))

    @parameterized.parameters(4, 8)
    def test_idempotent_quantize_when_input_is_already_levels(self, bits):
        """
        If input is already exactly on representable dequantized grid,
        quantize→dequantize should return the same values (within float eps).
        """
        scale = ops.array(0.125)
        maxq = ops.array(ops.subtract(ops.power(2, bits), 1), "float32")
        zero = ops.array(ops.divide(maxq, 2.0))

        # Build dequantized grid points: x = scale * (k - zero), k in [0..maxq]
        ks = ops.arange(0, ops.add(maxq, 1))
        x_vals = ops.multiply(scale, ops.subtract(ks, zero))
        x = ops.reshape(x_vals, (1, -1))

        q = quantize(x, scale, zero, maxq)
        x_hat = dequantize(q, scale, zero)
        self.assertAllClose(x_hat, x, rtol=0, atol=1e-6)


class ComputeScaleZeroTest(testing.TestCase):
    def test_error_when_x_is_none(self):
        with self.assertRaisesRegex(ValueError, "cannot be None"):
            compute_scale_zero(None, bits=4)

    def test_error_when_x_is_empty(self):
        x = ops.array([], "float32")
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            compute_scale_zero(x, bits=4)

    def test_error_when_weight_rank_too_low(self):
        x = ops.array([1.0, 2.0], "float32")  # rank-1
        with self.assertRaisesRegex(ValueError, "rank of at least 2"):
            compute_scale_zero(x, bits=4, weight=True)

    @parameterized.named_parameters(
        ("bits2_asym", 2, False),
        ("bits4_asym", 4, False),
        ("bits8_asym", 8, False),
        ("bits2_sym", 2, True),
        ("bits4_sym", 4, True),
        ("bits8_sym", 8, True),
    )
    def test_per_tensor_shapes_and_basic_invariants(self, bits, symmetric):
        """Test per-tensor shapes and basic invariants."""
        x = ops.array(
            np.random.default_rng(0).standard_normal((7, 5), dtype="float32")
        )
        scale, zero, maxq = compute_scale_zero(
            x, bits=bits, symmetric=symmetric, per_channel=False, weight=False
        )

        # Shapes (per-tensor): (1,) for scale/zero
        self.assertEqual(scale.shape, (1,))
        self.assertEqual(zero.shape, (1,))

        # Scale must be strictly positive
        self.assertTrue(ops.all(scale > 0.0))

        if symmetric:
            # zero should be (maxq + 1)/2 for symmetric
            expected_zero = ops.divide(ops.add(maxq, 1.0), 2.0)
            self.assertAllClose(zero, expected_zero)
        else:
            # Asymmetric: zero ~ round(-min/scale) on the flattened input
            flat = ops.reshape(x, (1, -1))
            min_val = ops.min(flat, axis=1)
            expected_zero = ops.round(ops.divide(ops.negative(min_val), scale))
            self.assertAllClose(zero, expected_zero)

    def test_per_tensor_symmetric_on_constant_input_uses_safe_range(self):
        """Ensures safe range adjustment if entries are equal"""
        x = ops.array(np.full((3, 4), 0.0, dtype=np.float32))
        scale, zero, maxq = compute_scale_zero(
            x, bits=4, symmetric=True, per_channel=False, weight=False
        )
        # With symmetric=True and constant input, zero = (maxq+1)/2
        self.assertAllClose(zero, ops.array((float(maxq) + 1.0) / 2.0))
        self.assertTrue(ops.all(scale > 0.0))

    def test_weight_per_tensor_tiles_rows(self):
        """Tests that scales/zeros tensors are properly tiled when
        per-channel quantization is not used."""
        x = ops.array(
            np.random.default_rng(1).standard_normal((8, 16)), "float32"
        )
        scale, zero, _ = compute_scale_zero(
            x, bits=4, symmetric=False, per_channel=False, weight=True
        )
        # When weight=True and per_channel=False, shapes are (rows, 1)
        self.assertEqual(scale.shape, (8, 1))
        self.assertEqual(zero.shape, (8, 1))

        # All elements in the scale and zero tensors must be equal due to
        # tiling.
        self.assertTrue(ops.all(scale == scale[0, 0]))
        self.assertTrue(ops.all(zero == zero[0, 0]))

    def test_weight_per_channel_ungrouped_shapes(self):
        """Tests that scales/zeros tensors have the correct shape when
        per-channel quantization is used without grouping."""
        x = ops.array(
            np.random.default_rng(2).standard_normal((6, 10)), "float32"
        )
        scale, zero, _ = compute_scale_zero(
            x,
            bits=4,
            symmetric=False,
            per_channel=True,
            group_size=-1,
            weight=True,
        )
        # Per-channel (ungrouped): one scale per output row -> (rows, 1)
        self.assertEqual(scale.shape, (6, 1))
        self.assertEqual(zero.shape, (6, 1))
        self.assertTrue(ops.all(scale > 0.0))

        # Each channel should have roughly unique scales and zeros
        self.assertFalse(ops.all(scale == scale[0, 0]))
        self.assertFalse(ops.all(zero == zero[0, 0]))

    def test_weight_per_channel_grouped_shapes_and_count(self):
        """Tests that scales/zeros have the correct shape and count when
        per-channel quantization is used with grouping."""
        rows, cols, groups = 8, 16, 4
        x = ops.array(
            np.random.default_rng(3).standard_normal((rows, cols)), "float32"
        )
        scale, zero, _ = compute_scale_zero(
            x,
            bits=4,
            symmetric=False,
            per_channel=True,
            group_size=groups,
            weight=True,
        )
        # Grouped path reshapes to [-1, group_size]
        # number of groups = rows*cols / groups
        num_groups = (rows * cols) // groups
        self.assertEqual(scale.shape, (num_groups, 1))
        self.assertEqual(zero.shape, (num_groups, 1))
        self.assertTrue(ops.all(scale > 0.0))

    @parameterized.named_parameters(
        ("sym_true", True),
        ("sym_false", False),
    )
    def test_dtype_and_finiteness(self, symmetric):
        x = ops.array(
            np.random.default_rng(4).standard_normal((5, 7)).astype("float32")
        )
        scale, zero, maxq = compute_scale_zero(
            x,
            bits=8,
            symmetric=symmetric,
            per_channel=True,
            group_size=-1,
            weight=True,
        )
        # Dtypes should be float-like and all finite
        self.assertEqual(scale.dtype, "float32")
        self.assertEqual(zero.dtype, "float32")
        self.assertEqual(maxq.dtype, "float32")
        self.assertTrue(ops.all(ops.isfinite(scale)))
        self.assertTrue(ops.all(ops.isfinite(zero)))
