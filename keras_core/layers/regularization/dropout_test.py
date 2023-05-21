import numpy as np
import pytest

from keras_core import backend
from keras_core import layers
from keras_core import testing


class DropoutTest(testing.TestCase):
    def test_dropout_basics(self):
        self.run_layer_test(
            layers.Dropout,
            init_kwargs={
                "rate": 0.2,
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=1,
            expected_num_losses=0,
            supports_masking=True,
        )

    def test_dropout_rescaling(self):
        inputs = np.ones((20, 500))
        layer = layers.Dropout(0.5, seed=1337)
        outputs = layer(inputs, training=True)
        self.assertAllClose(np.mean(outputs), 1.0, atol=0.02)
        self.assertAllClose(np.max(outputs), 2.0)

    @pytest.mark.skipif(
        backend.backend() == "jax",
        reason="JAX does not support dynamic shapes",
    )
    def test_dropout_partial_noise_shape_dynamic(self):
        inputs = np.ones((20, 5, 10))
        layer = layers.Dropout(0.5, noise_shape=(None, 1, None))
        outputs = layer(inputs, training=True)
        self.assertAllClose(outputs[:, 0, :], outputs[:, 1, :])

    def test_dropout_partial_noise_shape_static(self):
        inputs = np.ones((20, 5, 10))
        layer = layers.Dropout(0.5, noise_shape=(20, 1, 10))
        outputs = layer(inputs, training=True)
        self.assertAllClose(outputs[:, 0, :], outputs[:, 1, :])
