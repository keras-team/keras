from keras import ops
from keras import quantizers
from keras import random
from keras import testing


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
