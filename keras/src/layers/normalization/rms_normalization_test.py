import numpy as np
import pytest

from keras.src import layers
from keras.src import ops
from keras.src import testing


class RMSNormalizationTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_ln_basics(self):
        self.run_layer_test(
            layers.RMSNormalization,
            init_kwargs={},
            input_shape=(4, 2),
            expected_output_shape=(4, 2),
            expected_num_trainable_weights=1,
            expected_num_seed_generators=0,
        )
        self.run_layer_test(
            layers.RMSNormalization,
            init_kwargs={
                "axis": -1,
            },
            input_shape=(4, 2),
            expected_output_shape=(4, 2),
            expected_num_trainable_weights=1,
            expected_num_seed_generators=0,
        )

    def test_correctness(self):
        layer = layers.RMSNormalization()
        layer.build(input_shape=(2, 2, 2))
        inputs = np.random.normal(
            loc=5.0, scale=10.0, size=(1000, 2, 2, 2)
        ).astype("float32")

        inputs = ops.convert_to_tensor(inputs)

        out = layer(inputs)
        expected = ops.multiply(
            ops.multiply(
                inputs,
                ops.rsqrt(ops.mean(ops.square(inputs), axis=-1, keepdims=True)),
            ),
            layer.scale,
        )

        self.assertAllClose(out, expected, atol=1e-1)

    def test_output(self):
        layer = layers.RMSNormalization()
        inputs = np.arange(10).astype("float32")[None, :]
        out = layer(inputs)
        self.assertAllClose(
            out,
            [
                [
                    0.0,
                    0.18731716,
                    0.37463433,
                    0.5619515,
                    0.74926865,
                    0.9365858,
                    1.123903,
                    1.3112202,
                    1.4985373,
                    1.6858544,
                ]
            ],
        )
