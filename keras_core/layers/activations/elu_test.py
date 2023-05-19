import numpy as np
import tensorflow as tf

from keras_core import testing
from keras_core.layers.activations import elu


class ELUTest(testing.TestCase):
    def test_config(self):
        elu_layer = elu.ELU()
        self.run_class_serialization_test(elu_layer)

    def test_elu(self):
        self.run_layer_test(
            elu.ELU,
            init_kwargs={},
            input_shape=(2, 3, 4),
            supports_masking=True,
        )

    def test_correctness(self):
        x = np.random.random((2, 2, 5))
        elu_layer = elu.ELU()
        tf_elu_layer = tf.keras.layers.ELU()
        self.assertAllClose(elu_layer(x), tf_elu_layer(x))

        elu_layer = elu.ELU(alpha=0.7)
        tf_elu_layer = tf.keras.layers.ELU(alpha=0.7)
        self.assertAllClose(elu_layer(x), tf_elu_layer(x))
