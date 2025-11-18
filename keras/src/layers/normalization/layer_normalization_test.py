import numpy as np
import pytest

from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import regularizers
from keras.src import testing


class LayerNormalizationTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_ln_basics(self):
        self.run_layer_test(
            layers.LayerNormalization,
            init_kwargs={
                "gamma_regularizer": regularizers.L2(0.01),
                "beta_regularizer": regularizers.L2(0.01),
            },
            input_shape=(3, 4, 2),
            expected_output_shape=(3, 4, 2),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=2,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.LayerNormalization,
            init_kwargs={
                "gamma_initializer": "ones",
                "beta_initializer": "ones",
            },
            input_shape=(3, 4, 2),
            expected_output_shape=(3, 4, 2),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.LayerNormalization,
            init_kwargs={"scale": False, "center": False},
            input_shape=(3, 3),
            expected_output_shape=(3, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.LayerNormalization,
            init_kwargs={"rms_scaling": True},
            input_shape=(3, 3),
            expected_output_shape=(3, 3),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.LayerNormalization,
            init_kwargs={"axis": (-3, -2, -1)},
            input_shape=(2, 8, 8, 3),
            expected_output_shape=(2, 8, 8, 3),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.LayerNormalization,
            init_kwargs={},
            input_shape=(1, 0, 10),
            expected_output_shape=(1, 0, 10),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    def test_invalid_axis(self):
        with self.assertRaisesRegex(
            TypeError,
            ("Expected an int or a list/tuple of ints for the argument 'axis'"),
        ):
            layers.LayerNormalization(axis={"axis": -1})

    def test_correctness(self):
        layer = layers.LayerNormalization(dtype="float32")
        layer.build(input_shape=(2, 2, 2))
        inputs = np.random.normal(
            loc=5.0, scale=10.0, size=(1000, 2, 2, 2)
        ).astype("float32")

        out = layer(inputs)
        out = ops.subtract(out, layer.beta)
        out = ops.divide(out, layer.gamma)

        self.assertAllClose(ops.mean(out), 0.0, atol=1e-1)
        self.assertAllClose(ops.std(out), 1.0, atol=1e-1)

    def test_output(self):
        layer = layers.LayerNormalization(
            dtype="float32",
            beta_initializer="ones",
            gamma_initializer="ones",
        )
        inputs = np.arange(5).astype("float32")[None, :]
        out = layer(inputs)
        self.assertAllClose(out, [[-0.41386, 0.29307, 1.0, 1.70693, 2.41386]])

    def test_output_with_rms_scaling(self):
        layer = layers.LayerNormalization(
            dtype="float32",
            rms_scaling=True,
            gamma_initializer="ones",
        )
        inputs = np.arange(5).astype("float32")[None, :]
        out = layer(inputs)
        self.assertAllClose(out, [[0.0, 0.70693, 1.41386, 2.12079, 2.82772]])

    def test_large_value_within_autocast_scope(self):
        layer = layers.LayerNormalization()
        layer.build((1, 4, 4, 3))
        # Use 70000 to trigger overflow for float16
        large_value = ops.full(layer.gamma.shape, 70000)
        with backend.AutocastScope("float16"):
            layer.gamma.assign(large_value)
            self.assertAllClose(layer.gamma.value, large_value)
