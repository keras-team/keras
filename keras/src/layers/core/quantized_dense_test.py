import numpy as np
import pytest

from keras.src import layers
from keras.src import testing


class QuantizedDenseTest(testing.TestCase):
    def test_quantized_dense_basics(self):
        self.run_layer_test(
            layers.QuantizedDense,
            init_kwargs={
                "units": 4,
                "bits": 8,
                "activation": "relu",
            },
            input_shape=(2, 3),
            input_dtype="float32",
            expected_output_shape=(2, 4),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )

    def test_quantized_dense_4bit(self):
        self.run_layer_test(
            layers.QuantizedDense,
            init_kwargs={
                "units": 4,
                "bits": 4,
            },
            input_shape=(3, 5),
            input_dtype="float32",
            expected_output_shape=(3, 4),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )

    def test_invalid_bits(self):
        with self.assertRaisesRegex(ValueError, "Only 4-bit and 8-bit"):
            layers.QuantizedDense(units=4, bits=6)

    def test_quantized_dense_no_bias(self):
        self.run_layer_test(
            layers.QuantizedDense,
            init_kwargs={
                "units": 4,
                "bits": 8,
                "use_bias": False,
            },
            input_shape=(2, 3),
            input_dtype="float32",
            expected_output_shape=(2, 4),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )

    def test_missing_input_dim(self):
        layer = layers.QuantizedDense(units=4)
        with self.assertRaisesRegex(ValueError, "last dimension"):
            layer.build((None, None))
            
    def test_fake_quant_with_ste_narrow_range_false(self):
        from keras.src.layers.core.quantized_dense import _fake_quant_with_ste
        from keras.src import ops
        inputs = ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0])
        quantized = _fake_quant_with_ste(inputs, 0.0, 3.0, 8, False)
        self.assertEqual(quantized.shape, (4,))
