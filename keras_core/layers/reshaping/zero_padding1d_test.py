import numpy as np
import pytest
from absl.testing import parameterized

from keras_core import backend
from keras_core import layers
from keras_core import testing


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

    @pytest.mark.skipif(
        not backend.DYNAMIC_SHAPES_OK,
        reason="Backend does not support dynamic shapes",
    )
    def test_zero_padding_1d_with_dynamic_spatial_dim(self):
        input_layer = layers.Input(batch_shape=(1, None, 3))
        padded = layers.ZeroPadding1D((1, 2))(input_layer)
        self.assertEqual(padded.shape, (1, None, 3))

    def test_zero_padding_1d_errors_if_padding_argument_invalid(self):
        with self.assertRaises(ValueError):
            layers.ZeroPadding1D(padding=(1,))
        with self.assertRaises(ValueError):
            layers.ZeroPadding1D(padding=(1, 2, 3))
        with self.assertRaises(ValueError):
            layers.ZeroPadding1D(padding="1")
