import numpy as np
import pytest

from keras.src import backend
from keras.src import layers
from keras.src import testing


class DropoutTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_dropout_basics(self):
        self.run_layer_test(
            layers.Dropout,
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

    def test_dropout_rescaling(self):
        inputs = np.ones((20, 500))
        layer = layers.Dropout(0.5, seed=1337)
        outputs = layer(inputs, training=True)
        outputs = backend.convert_to_numpy(outputs)
        self.assertAllClose(np.mean(outputs), 1.0, atol=0.02)
        self.assertAllClose(np.max(outputs), 2.0)

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

    def test_dropout_negative_rate(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid value received for argument `rate`. "
            "Expected a float value between 0 and 1.",
        ):
            _ = layers.Dropout(rate=-0.5)

    def test_dropout_rate_greater_than_one(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid value received for argument `rate`. "
            "Expected a float value between 0 and 1.",
        ):
            _ = layers.Dropout(rate=1.5)

    def test_validate_noise_shape_none(self):
        layer = layers.Dropout(0.5, noise_shape=None)
        self.assertIsNone(layer.noise_shape)

    def test_validate_noise_shape_integer_tuple(self):
        layer = layers.Dropout(0.5, noise_shape=(20, 1, 10))
        self.assertEqual(layer.noise_shape, (20, 1, 10))

    def test_validate_noise_shape_none_values(self):
        layer = layers.Dropout(0.5, noise_shape=(None, 1, None))
        self.assertEqual(layer.noise_shape, (None, 1, None))

    def test_validate_noise_shape_cast_to_a_tuple(self):
        layer = layers.Dropout(0.5, noise_shape=[20, 1, 10])
        self.assertEqual(layer.noise_shape, (20, 1, 10))
        self.assertIsInstance(layer.noise_shape, tuple)

    def test_validate_noise_shape_non_iterable(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid value received for argument `noise_shape`. "
            "Expected a tuple or list of integers.",
        ):
            layers.Dropout(0.5, noise_shape="Invalid")

    def test_validate_noise_shape_invalid_type(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid value received for argument `noise_shape`. "
            "Expected all elements to be integers or None.",
        ):
            layers.Dropout(0.5, noise_shape=(20, 1.5, 10))

    def test_validate_noise_shape_negative_value(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid value received for argument `noise_shape`. "
            "Expected all dimensions to be positive integers or None.",
        ):
            layers.Dropout(0.5, noise_shape=(20, -1, 10))

    def test_validate_noise_shape_zero_value(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid value received for argument `noise_shape`. "
            "Expected all dimensions to be positive integers or None.",
        ):
            layers.Dropout(0.5, noise_shape=(20, 0, 10))
