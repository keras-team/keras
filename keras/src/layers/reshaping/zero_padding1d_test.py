import numpy as np
from absl.testing import parameterized

from keras.src import dtype_policies
from keras.src import layers
from keras.src import testing


class ZeroPadding1DTest(testing.TestCase, parameterized.TestCase):
    def test_zero_padding_1d(self):
        inputs = np.random.rand(1, 2, 3)
        outputs = layers.ZeroPadding1D(padding=(1, 2))(inputs)

        for index in [0, -1, -2]:
            self.assertAllClose(outputs[:, index, :], 0.0)
        self.assertAllClose(outputs[:, 1:-2, :], inputs)

    @parameterized.named_parameters(("one_tuple", (2, 2)), ("one_int", 2))
    def test_zero_padding_1d_with_same_padding(self, padding):
        inputs = np.random.rand(1, 2, 3)
        outputs = layers.ZeroPadding1D(padding=padding)(inputs)

        for index in [0, 1, -1, -2]:
            self.assertAllClose(outputs[:, index, :], 0.0)
        self.assertAllClose(outputs[:, 2:-2, :], inputs)

    def test_zero_padding_1d_with_dynamic_spatial_dim(self):
        input_layer = layers.Input(batch_shape=(1, None, 3))
        padded = layers.ZeroPadding1D((1, 2))(input_layer)
        self.assertEqual(padded.shape, (1, None, 3))

        input_layer = layers.Input(batch_shape=(1, 2, 3))
        padded = layers.ZeroPadding1D((1, 2))(input_layer)
        self.assertEqual(padded.shape, (1, 5, 3))

    @parameterized.parameters(
        {"padding": (1,)},
        {"padding": (1, 2, 3)},
        {"padding": "1"},
    )
    def test_zero_padding_1d_errors_if_padding_argument_invalid(self, padding):
        with self.assertRaises(ValueError):
            layers.ZeroPadding1D(padding)

    def test_zero_padding_1d_get_config(self):
        layer = layers.ZeroPadding1D(padding=(1, 2))
        expected_config = {
            "dtype": dtype_policies.serialize(layer.dtype_policy),
            "name": layer.name,
            "padding": (1, 2),
            "trainable": layer.trainable,
        }
        self.assertEqual(layer.get_config(), expected_config)
