import numpy as np
import pytest

from keras import testing
from keras.layers.activations import prelu


class PReLUTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_prelu(self):
        self.run_layer_test(
            prelu.PReLU,
            init_kwargs={
                "alpha_initializer": "zeros",
                "alpha_regularizer": "L1",
                "alpha_constraint": "MaxNorm",
                "shared_axes": 1,
            },
            input_shape=(2, 3, 4),
            supports_masking=True,
        )

    def test_prelu_correctness(self):
        def np_prelu(x, alpha):
            return (x > 0) * x + (x <= 0) * alpha * x

        inputs = np.random.randn(2, 10, 5, 3)
        prelu_layer = prelu.PReLU(
            alpha_initializer="glorot_uniform",
            alpha_regularizer="l1",
            alpha_constraint="non_neg",
            shared_axes=(1, 2),
        )
        prelu_layer.build(inputs.shape)

        weights = np.random.random((1, 1, 3))
        prelu_layer.alpha.assign(weights)
        ref_out = np_prelu(inputs, weights)
        self.assertAllClose(prelu_layer(inputs), ref_out)
