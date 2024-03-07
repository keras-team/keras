from keras import ops
from keras import quantizers
from keras import random
from keras import testing


class QuantizersTest(testing.TestCase):
    def test_abs_max_quantizer(self):
        values = random.uniform([3, 4, 5], minval=-1, maxval=1)
        quantizer = quantizers.AbsMaxQuantizer(axis=-1)

        # Test quantize
        quantized_values, scale = quantizer(values)
        self.assertEqual(quantized_values.shape, [3, 4, 5])
        self.assertEqual(scale.shape, [3, 4, 1])
        self.assertLessEqual(ops.max(quantized_values), 127)
        self.assertGreaterEqual(ops.min(quantized_values), -127)

        # Test dequantize
        dequantized_values = ops.divide(quantized_values, scale)
        self.assertAllClose(values, dequantized_values, atol=1)

        # Test serialization
        self.run_class_serialization_test(quantizer)
