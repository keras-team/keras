import numpy as np

from keras_core import testing
from keras_core.layers.activations import prelu
import tensorflow as tf


class PReLUTest(testing.TestCase):
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
        inputs = np.random.randn(2, 10, 5, 3)
        prelu_layer = prelu.PReLU(
            alpha_initializer="glorot_uniform",
            alpha_regularizer="l1",
            alpha_constraint="non_neg",
            shared_axes=(1, 2),
        )
        tf_prelu_layer = tf.keras.layers.PReLU(
            alpha_initializer="glorot_uniform",
            alpha_regularizer="l1",
            alpha_constraint="non_neg",
            shared_axes=(1, 2),
        )

        prelu_layer.build(inputs.shape)
        tf_prelu_layer.build(inputs.shape)

        weights = np.random.random((1, 1, 3))
        prelu_layer.alpha.assign(weights)
        tf_prelu_layer.alpha.assign(weights)

        self.assertAllClose(prelu_layer(inputs), tf_prelu_layer(inputs))
