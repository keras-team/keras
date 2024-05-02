import numpy as np
import pytest

from keras.src import testing
from keras.src.layers.activations import leaky_relu


class LeakyReLUTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_leaky_relu(self):
        self.run_layer_test(
            leaky_relu.LeakyReLU,
            init_kwargs={
                "negative_slope": 1,
            },
            input_shape=(2, 3, 4),
            supports_masking=True,
        )

    def test_leaky_relu_correctness(self):
        leaky_relu_layer = leaky_relu.LeakyReLU(negative_slope=0.5)
        input = np.array([-10, -5, 0.0, 5, 10])
        expected_output = np.array([-5.0, -2.5, 0.0, 5.0, 10.0])
        result = leaky_relu_layer(input)
        self.assertAllClose(result, expected_output)

    def test_invalid_usage(self):
        with self.assertRaisesRegex(
            ValueError,
            "The negative_slope value of a Leaky ReLU layer cannot be None",
        ):
            self.run_layer_test(
                leaky_relu.LeakyReLU,
                init_kwargs={"negative_slope": None},
                input_shape=(2, 3, 4),
                supports_masking=True,
            )
