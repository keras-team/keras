import tensorflow as tf
from absl.testing import parameterized

import keras
from keras.src.testing import test_case


class MyDenseLayer(keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=[int(input_shape[-1]), self.num_outputs],
        )

    def call(self, inputs):
        kernel = tf.cast(self.kernel, tf.complex64)
        return tf.matmul(inputs, kernel)


# Custom layer test with complex input
class TestDenseLayer(test_case.TestCase, parameterized.TestCase):
    def test_layer_output_shape(self):
        input = tf.zeros([10, 5], dtype=tf.complex64)
        layer = MyDenseLayer(10)
        output = layer(input)
        self.assertAllEqual(output.shape, (10, 10))
