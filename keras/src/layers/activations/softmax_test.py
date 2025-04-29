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
            assert_built_after_instantiation=True,
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

    def test_softmax_masked_values_are_zero_including_fully_masked(self):
        """
        Tests softmax with mask on default axis (-1).
        Ensures output is 0 where mask is False.
        Includes a row where all elements are masked.
        """
        softmax_layer = softmax.Softmax()  # Default axis = -1

        input = np.array(
            [
                [1.0, 2.0, 5.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [3.0, 1.0, 2.0, 4.0],
            ],
            dtype=np.float32,
        )
        mask = np.array(
            [
                [True, True, False, False],  # Partially masked
                [False, False, False, False],  # Fully masked
                [True, True, True, True],  # Not masked
            ],
            dtype=bool,
        )

        expected_output = np.array(
            [
                [0.268941, 0.731059, 0.0, 0.0],  # last two masked
                [0.0, 0.0, 0.0, 0.0],  # Fully masked row should be all zeros
                [0.236883, 0.032059, 0.087144, 0.643914],
            ]
        )

        result = softmax_layer(input, mask=mask)

        self.assertAllClose(result, expected_output)
