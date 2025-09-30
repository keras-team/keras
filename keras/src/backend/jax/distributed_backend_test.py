import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import optax
import pytest

import keras
from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.backend.jax.distributed_backend import JaxDistributedBackend


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="Jax Backend specific test",
)
class TestJaxDistributedBackend(testing.TestCase):
    """Unit tests for the JaxDistributedBackend class."""

    def setUp(self):
        """Set up the test case by instantiating the backend."""
        super().setUp()
        self.backend = JaxDistributedBackend()

    def test_compute_gradients_returns_zeros(self):
        """Test that compute_gradients returns correctly shaped zero tensors."""
        loss = ops.array(10.0)
        trainable_vars = [ops.array([1.0, 2.0]), ops.array(3.0)]

        gradients = self.backend.compute_gradients(loss, trainable_vars)

        self.assertEqual(len(gradients), 2)
        self.assertAllClose(gradients[0], ops.zeros_like(trainable_vars[0]))
        self.assertAllClose(gradients[1], ops.zeros_like(trainable_vars[1]))

    def test_apply_gradients(self):
        """Test the application of gradients to Keras variables."""
        var1 = keras.Variable([1.0, 2.0])
        var2 = keras.Variable(5.0)
        trainable_vars = [var1, var2]

        grad1 = ops.array([0.1, 0.2])
        grad2 = ops.array(0.5)
        gradients = [grad1, grad2]
        learning_rate = 0.1
        self.backend.apply_gradients(gradients, trainable_vars, learning_rate)

        expected_var1 = ops.array([1.0, 2.0]) - ops.multiply(
            ops.array([0.1, 0.2]), learning_rate
        )
        expected_var2 = 5.0 - (0.5 * learning_rate)

        self.assertAllClose(var1.value, expected_var1)
        self.assertAllClose(var2.value, expected_var2)

    def test_create_optimizer(self):
        """Test optimizer creation for Adam, SGD, and a default case."""
        adam_optimizer = self.backend.create_optimizer(
            "adam", learning_rate=0.01
        )
        self.assertIsInstance(adam_optimizer, optax.GradientTransformation)

        sgd_optimizer = self.backend.create_optimizer("sgd", learning_rate=0.01)
        self.assertIsInstance(sgd_optimizer, optax.GradientTransformation)

        default_optimizer = self.backend.create_optimizer(
            "some_unknown_optimizer"
        )
        self.assertIsInstance(default_optimizer, optax.GradientTransformation)

    def test_get_device_info(self):
        """Test retrieving device information from the JAX backend."""
        info = self.backend.get_device_info()
        self.assertEqual(info["backend"], "jax")
        self.assertIsInstance(info["devices"], list)
        self.assertIsInstance(info["device_count"], int)
        self.assertGreater(info["device_count"], 0)
        self.assertEqual(len(info["devices"]), info["device_count"])

    def test_is_multi_device_capable(self):
        """Test the boolean check for multi-device capability."""
        self.assertIsInstance(self.backend.is_multi_device_capable(), bool)

    def test_get_communication_ops_simulated(self):
        """Test the simulated communication ops in a single-device context."""
        comm_ops = self.backend.get_communication_ops()

        device_info = self.backend.get_device_info()
        simulated_world_size = device_info.get("device_count", 1)
        if simulated_world_size == 0:
            simulated_world_size = 1

        # Test all_reduce
        x_reduce = ops.array([[1.0, 2.0], [3.0, 4.0]])
        reduced = comm_ops["all_reduce"](x_reduce, op="sum")
        self.assertAllClose(
            reduced, ops.multiply(x_reduce, simulated_world_size)
        )

        # Test all_gather
        x_gather = ops.array([[1.0, 2.0]])
        gathered = comm_ops["all_gather"](x_gather, axis=0)
        expected_gather = ops.concatenate(
            [x_gather] * simulated_world_size, axis=0
        )
        self.assertAllClose(gathered, expected_gather)

        # Test broadcast
        x_broadcast = ops.array([5.0, 6.0])
        broadcasted = comm_ops["broadcast"](x_broadcast)
        self.assertAllClose(broadcasted, x_broadcast)

        # Test scatter
        scatter_data = ops.arange(simulated_world_size * 2)
        scatter_data = ops.reshape(scatter_data, (simulated_world_size, 2))
        x_scatter = ops.cast(scatter_data, dtype="float32")
        scattered = comm_ops["scatter"](x_scatter)

        expected_scatter = ops.split(x_scatter, simulated_world_size, axis=0)[0]
        self.assertAllClose(scattered, expected_scatter)
