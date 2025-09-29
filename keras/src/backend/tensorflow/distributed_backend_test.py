import logging
import unittest

import numpy as np
import pytest
import tensorflow as tf

from keras.src import backend
from keras.src.backend.tensorflow.distributed_backend import (
    TensorflowDistributedBackend,
)

logging.disable(logging.WARNING)


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="TensorFlow-specific distributed backend tests",
)
class TestTensorflowDistributedBackend(unittest.TestCase):
    """Unit tests for the TensorflowDistributedBackend class."""

    def setUp(self):
        self.backend = TensorflowDistributedBackend()

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_get_tensor_lib(self):
        self.assertIs(self.backend.get_tensor_lib(), tf)

    def test_convert_to_backend_tensor(self):
        py_list = [1.0, 2.0, 3.0]
        tf_tensor = self.backend.convert_to_backend_tensor(py_list)
        self.assertIsInstance(tf_tensor, tf.Tensor)
        np.testing.assert_array_equal(
            tf_tensor.numpy(), np.array([1.0, 2.0, 3.0])
        )

    def test_compute_gradients_returns_nones(self):
        trainable_vars = [tf.Variable(3.0), tf.Variable(5.0)]
        loss = tf.constant(10.0)
        gradients = self.backend.compute_gradients(loss, trainable_vars)

        self.assertEqual(gradients, [None, None])

    def test_apply_gradients(self):
        """Test applying gradients to tf.Variable objects."""
        var1 = tf.Variable(10.0)
        var2 = tf.Variable(20.0)
        trainable_vars = [var1, var2]

        grad1 = tf.constant(0.5)
        grad2 = tf.constant(1.5)
        gradients = [grad1, grad2]

        self.backend.apply_gradients(
            gradients, trainable_vars, learning_rate=0.1
        )

        np.testing.assert_allclose(var1.numpy(), 10.0 - 0.1 * 0.5)
        np.testing.assert_allclose(var2.numpy(), 20.0 - 0.1 * 1.5)

    def test_create_optimizer(self):
        """Test the creation of TensorFlow Keras optimizers."""
        adam = self.backend.create_optimizer("adam")
        self.assertIsInstance(adam, tf.keras.optimizers.Adam)

        sgd = self.backend.create_optimizer("sgd")
        self.assertIsInstance(sgd, tf.keras.optimizers.SGD)

        default = self.backend.create_optimizer("unknown")
        self.assertIsInstance(default, tf.keras.optimizers.Adam)

    def test_get_device_info(self):
        info = self.backend.get_device_info()
        self.assertEqual(info["backend"], "tensorflow")
        self.assertIsInstance(info["devices"], list)
        self.assertIsInstance(info["device_count"], int)
        self.assertGreater(info["device_count"], 0)

    def test_is_multi_device_capable(self):
        self.assertIsInstance(self.backend.is_multi_device_capable(), bool)

    def test_get_communication_ops_simulated(self):
        ops = self.backend.get_communication_ops()

        x_reduce = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        reduced = ops["all_reduce"](x_reduce)
        np.testing.assert_allclose(reduced.numpy(), np.array([4.0, 6.0]))

        x_gather = tf.constant([[1.0, 2.0]])
        gathered = ops["all_gather"](x_gather, axis=0)
        np.testing.assert_allclose(
            gathered.numpy(), np.array([[1.0, 2.0], [1.0, 2.0]])
        )

        x_broadcast = tf.constant([5.0, 6.0])
        broadcasted = ops["broadcast"](x_broadcast)
        np.testing.assert_allclose(broadcasted.numpy(), x_broadcast.numpy())

        x_scatter = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]])
        scattered = ops["scatter"](x_scatter, num_devices=2)
        self.assertEqual(len(scattered), 2)
        np.testing.assert_allclose(
            scattered[0].numpy(), np.array([[1, 2], [3, 4]])
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
