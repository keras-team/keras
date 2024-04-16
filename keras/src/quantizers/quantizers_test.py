from keras.src import ops
from keras.src import quantizers
from keras.src import random
from keras.src import testing


class QuantizersTest(testing.TestCase):
    def test_get_method(self):
        quantizer = quantizers.get("abs_max_quantizer", axis=-1)
        self.assertTrue(quantizer, quantizers.AbsMaxQuantizer)

        quantizer = quantizers.get(None)
        self.assertEqual(quantizer, None)

        with self.assertRaises(ValueError):
            quantizers.get("typo")

    def test_abs_max_quantizer(self):
        values = random.uniform([3, 4, 5], minval=-1, maxval=1)
        quantizer = quantizers.AbsMaxQuantizer(axis=-1)

        # Test quantizing
        quantized_values, scale = quantizer(values)
        self.assertEqual(tuple(quantized_values.shape), (3, 4, 5))
        self.assertEqual(tuple(scale.shape), (3, 4, 1))
        self.assertLessEqual(ops.max(quantized_values), 127)
        self.assertGreaterEqual(ops.min(quantized_values), -127)

        # Test dequantizing
        dequantized_values = ops.divide(quantized_values, scale)
        rmse = ops.sqrt(
            ops.mean(ops.square(ops.subtract(values, dequantized_values)))
        )
        self.assertLess(rmse, 1e-1)  # loose assertion

        # Test serialization
        self.run_class_serialization_test(quantizer)

    def test_compute_float8_scale(self):
        amax = 3.0
        scale = 4.0
        dtype_max = 448.0  # float8_e4m3fn
        # The algorithm for computing the new scale is sourced from
        # https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/jax.html#transformer_engine.jax.update_fp8_metas
        expected_scale = 1.0 / (dtype_max / amax) / (2**0)

        computed_scale = quantizers.compute_float8_scale(amax, scale, dtype_max)
        self.assertAllClose(computed_scale, expected_scale)

    def test_compute_float8_amax_history(self):
        values = random.uniform([3, 4, 5], minval=-1, maxval=1)
        amax_history = random.uniform([123])
        amax_from_values = ops.max(ops.abs(values))

        computed_amax_history = quantizers.compute_float8_amax_history(
            values, amax_history
        )
        self.assertAllClose(computed_amax_history[0], amax_from_values)
        # Shift to left with 1 step
        self.assertAllClose(
            computed_amax_history[1:], ops.roll(amax_history, -1)[1:]
        )

    def test_quantize_and_dequantize(self):
        scale = 1.0 / 100.0
        values = random.uniform([3, 4, 5], minval=-1, maxval=1)
        qdq_values = quantizers.quantize_and_dequantize(
            values, scale, "float8_e4m3fn", "float32"
        )
        # A loose assertion due to an expected quantization error
        self.assertAllClose(qdq_values, values, atol=1e-1)

        qdq_values = quantizers.quantize_and_dequantize(
            values, scale, "float8_e5m2", "float32"
        )
        # A loose assertion due to an expected quantization error
        self.assertAllClose(qdq_values, values, atol=5e-1)
