import numpy as np
import pytest

from keras.src import backend
from keras.src import layers
from keras.src import testing


class AlphaDropoutTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_alpha_dropout_basics(self):
        self.run_layer_test(
            layers.AlphaDropout,
            init_kwargs={
                "rate": 0.2,
            },
            input_shape=(2, 3),
            call_kwargs={"training": True},
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=1,
            expected_num_losses=0,
            supports_masking=True,
            assert_built_after_instantiation=True,
        )

    def test_alpha_dropout_correctness(self):
        inputs = np.ones((20, 500)).astype("float32")
        layer = layers.AlphaDropout(0.3, seed=1337)
        outputs = layer(inputs, training=True)
        self.assertAllClose(
            np.std(backend.convert_to_numpy(outputs)), 1.0, atol=1e-1
        )

    def test_alpha_dropout_partial_noise_shape_dynamic(self):
        inputs = np.ones((20, 5, 10))
        layer = layers.AlphaDropout(0.5, noise_shape=(None, 1, None))
        outputs = layer(inputs, training=True)
        self.assertAllClose(outputs[:, 0, :], outputs[:, 1, :])

    def test_alpha_dropout_partial_noise_shape_static(self):
        inputs = np.ones((20, 5, 10))
        layer = layers.AlphaDropout(0.5, noise_shape=(20, 1, 10))
        outputs = layer(inputs, training=True)
        self.assertAllClose(outputs[:, 0, :], outputs[:, 1, :])

    def test_alpha_dropout_negative_rate(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid value received for argument `rate`. "
            "Expected a float value between 0 and 1.",
        ):
            _ = layers.AlphaDropout(rate=-0.5)

    def test_alpha_dropout_rate_greater_than_one(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid value received for argument `rate`. "
            "Expected a float value between 0 and 1.",
        ):
            _ = layers.AlphaDropout(rate=1.5)
