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
