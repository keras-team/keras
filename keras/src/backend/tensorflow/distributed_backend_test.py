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
        """
        Test the simulated communication ops for a non-distributed context.
        """
        ops = self.backend.get_communication_ops()

        device_info = self.backend.get_device_info()
        world_size = device_info.get("device_count", 1)
        if world_size == 0:
            world_size = 1

        x_reduce = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        reduced = ops["all_reduce"](x_reduce, op="sum")
        expected_reduce = x_reduce * world_size
        self.assertEqual(reduced.shape, x_reduce.shape)
        tf.debugging.assert_near(reduced, expected_reduce, rtol=1e-6)

        x_gather = tf.constant([[1.0, 2.0]])
        gathered = ops["all_gather"](x_gather, axis=0)
        expected_gather = tf.concat([x_gather] * world_size, axis=0)
        self.assertEqual(gathered.shape, (world_size, 2))
        tf.debugging.assert_near(gathered, expected_gather, rtol=1e-6)

        scatter_data = list(range(world_size * 2))
        x_scatter = tf.constant(scatter_data, dtype=tf.float32)
        scattered = ops["scatter"](x_scatter)
        expected_scatter = tf.constant(scatter_data[:2], dtype=tf.float32)
        self.assertEqual(scattered.shape, (2,))
        tf.debugging.assert_near(scattered, expected_scatter, rtol=1e-6)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
