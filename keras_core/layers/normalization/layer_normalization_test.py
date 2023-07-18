import numpy as np
import pytest

from keras_core import layers
from keras_core import ops
from keras_core import regularizers
from keras_core import testing


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

    def test_correctness(self):
        layer = layers.LayerNormalization(dtype="float32")
        layer.build(input_shape=(2, 2, 2))
        inputs = np.random.normal(
            loc=5.0, scale=10.0, size=(1000, 2, 2, 2)
        ).astype("float32")

        out = layer(inputs)
        out -= layer.beta
        out /= layer.gamma

        self.assertAllClose(ops.mean(out), 0.0, atol=1e-1)
        self.assertAllClose(ops.std(out), 1.0, atol=1e-1)
