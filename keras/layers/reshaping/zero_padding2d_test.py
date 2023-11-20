import numpy as np
from absl.testing import parameterized

from keras import backend
from keras import layers
from keras import testing


class ZeroPadding2DTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("channels_first", "channels_first"), ("channels_last", "channels_last")
    )
    def test_zero_padding_2d(self, data_format):
        inputs = np.random.rand(1, 2, 3, 4)
        outputs = layers.ZeroPadding2D(
            padding=((1, 2), (3, 4)), data_format=data_format
        )(inputs)

        if data_format == "channels_first":
            for index in [0, -1, -2]:
                self.assertAllClose(outputs[:, :, index, :], 0.0)
            for index in [0, 1, 2, -1, -2, -3, -4]:
                self.assertAllClose(outputs[:, :, :, index], 0.0)
            self.assertAllClose(outputs[:, :, 1:-2, 3:-4], inputs)
        else:
            for index in [0, -1, -2]:
                self.assertAllClose(outputs[:, index, :, :], 0.0)
            for index in [0, 1, 2, -1, -2, -3, -4]:
                self.assertAllClose(outputs[:, :, index, :], 0.0)
            self.assertAllClose(outputs[:, 1:-2, 3:-4, :], inputs)

    @parameterized.product(
        (
            {"padding": ((2, 2), (2, 2))},  # 2 tuples
            {"padding": (2, 2)},  # 1 tuple
            {"padding": 2},  # 1 int
        ),
        (
            {"data_format": "channels_first"},
            {"data_format": "channels_last"},
        ),
    )
    def test_zero_padding_2d_with_same_padding(self, padding, data_format):
        inputs = np.random.rand(1, 2, 3, 4)
        outputs = layers.ZeroPadding2D(
            padding=padding, data_format=data_format
        )(inputs)

        if data_format == "channels_first":
            for index in [0, 1, -1, -2]:
                self.assertAllClose(outputs[:, :, index, :], 0.0)
                self.assertAllClose(outputs[:, :, :, index], 0.0)
            self.assertAllClose(outputs[:, :, 2:-2, 2:-2], inputs)
        else:
            for index in [0, 1, -1, -2]:
                self.assertAllClose(outputs[:, index, :, :], 0.0)
                self.assertAllClose(outputs[:, :, index, :], 0.0)
            self.assertAllClose(outputs[:, 2:-2, 2:-2, :], inputs)

    def test_zero_padding_2d_with_dynamic_spatial_dim(self):
        if backend.config.image_data_format() == "channels_last":
            input_layer = layers.Input(batch_shape=(1, 2, None, 4))
        else:
            input_layer = layers.Input(batch_shape=(1, 4, 2, None))
        padded = layers.ZeroPadding2D(((1, 2), (3, 4)))(input_layer)
        if backend.config.image_data_format() == "channels_last":
            self.assertEqual(padded.shape, (1, 5, None, 4))
        else:
            self.assertEqual(padded.shape, (1, 4, 5, None))

    def test_zero_padding_2d_errors_if_padding_argument_invalid(self):
        with self.assertRaises(ValueError):
            layers.ZeroPadding2D(padding=(1,))
        with self.assertRaises(ValueError):
            layers.ZeroPadding2D(padding=(1, 2, 3))
        with self.assertRaises(ValueError):
            layers.ZeroPadding2D(padding="1")
        with self.assertRaises(ValueError):
            layers.ZeroPadding2D(padding=((1, 2), (3, 4, 5)))
        with self.assertRaises(ValueError):
            layers.ZeroPadding2D(padding=((1, 2), (3, -4)))
        with self.assertRaises(ValueError):
            layers.ZeroPadding2D(padding=((1, 2), "3"))
