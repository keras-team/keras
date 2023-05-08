import numpy as np

from keras_core import testing
from keras_core.layers.activations import prelu


class PReLUTest(testing.TestCase):
    def test_prelu(self):
        self.run_layer_test(
            prelu.PReLU,
            init_kwargs={
                "negative_slope_initializer": "zeros",
                "negative_slope_regularizer": "L1",
                "negative_slope_constraint": "MaxNorm",
                "shared_axes": 1,
            },
            input_shape=(2, 3, 4),
            supports_masking=True,
        )

    def test_prelu_correctness(self):
        prelu_layer = prelu.PReLU(
            negative_slope_initializer="glorot_uniform",
            negative_slope_regularizer="l1",
            negative_slope_constraint="non_neg",
            shared_axes=None,
        )
        test_input = np.random.randn(10, 5)
        result = prelu_layer(test_input)
        expected_output = np.maximum(
            0, test_input
        ) + prelu_layer.negative_slope.numpy() * np.minimum(0, test_input)
        self.assertAllClose(result, expected_output)
