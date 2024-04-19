import numpy as np
import pytest

from keras.src import testing
from keras.src.layers.activations import softmax


class SoftmaxTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_softmax(self):
        self.run_layer_test(
            softmax.Softmax,
            init_kwargs={},
            input_shape=(2, 3, 4),
            supports_masking=True,
        )

    def test_softmax_correctness(self):
        softmax_layer = softmax.Softmax()
        input = np.array([[1.0, 2.0, 1.0], [1.0, 2.0, 1.0]])
        expected_output = np.array(
            [
                [0.21194157, 0.5761169, 0.21194157],
                [0.21194157, 0.5761169, 0.21194157],
            ]
        )
        result = softmax_layer(input)
        self.assertAllClose(result, expected_output)

    def test_softmax_correctness_with_mask(self):
        softmax_layer = softmax.Softmax(axis=(1, 0))
        input = np.array([[1.0, 2.0, 1.0], [1.0, 2.0, 1.0]])
        mask = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        expected_output = np.array(
            [[0.21194154, 0.0, 0.21194154], [0.0, 0.57611686, 0.0]]
        )
        result = softmax_layer(input, mask=mask)
        self.assertAllClose(result, expected_output)

    def test_softmax_correctness_with_axis(self):
        softmax_layer = softmax.Softmax(axis=(1))
        input = np.array([[1.0, 2.0, 1.0], [1.0, 2.0, 1.0]])
        expected_output = np.array(
            [
                [0.21194157, 0.5761169, 0.21194157],
                [0.21194157, 0.5761169, 0.21194157],
            ]
        )
        result = softmax_layer(input)
        self.assertAllClose(result, expected_output)
