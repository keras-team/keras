import os

import numpy as np
import pytest

os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.src import testing


class PoolingConsistencyTest(testing.TestCase):
    def test_average_pooling_1d_same_asymmetric(self):
        # Input size 2, kernel 3, stride 2, padding "same" -> padding (0, 1)
        x = np.array([[[10.0], [20.0]]], dtype="float32")

        # Expected behavior (matches JAX/TF):
        # Padded with zero: [10, 20, 0]
        # Average excluding padding: (10+20)/2 = 15.0
        expected_output = np.array([[[15.0]]], dtype="float32")

        layer = keras.layers.AveragePooling1D(
            pool_size=3, strides=2, padding="same"
        )
        output = layer(x)
        self.assertAllClose(output, expected_output)

    def test_average_pooling_2d_same_asymmetric(self):
        x = np.ones((1, 2, 2, 1), dtype="float32")
        x[0, :, :, 0] = [[10.0, 20.0], [30.0, 40.0]]

        # pool_size=(3, 3), strides=(2, 2), padding="same"
        # Padding should be (0, 1) for both height and width
        # Padded:
        # [[10, 20, 0],
        #  [30, 40, 0],
        #  [ 0,  0, 0]]
        # Top-left 3x3 window: [[10, 20, 0], [30, 40, 0], [0, 0, 0]]
        # Valid elements: 10, 20, 30, 40. Sum = 100. Count = 4. Avg = 25.0.

        expected_output = np.array([[[[25.0]]]], dtype="float32")
        layer = keras.layers.AveragePooling2D(
            pool_size=3, strides=2, padding="same"
        )
        output = layer(x)
        self.assertAllClose(output, expected_output)


if __name__ == "__main__":
    pytest.main([__file__])
