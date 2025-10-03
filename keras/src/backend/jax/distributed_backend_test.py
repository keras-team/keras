import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import pytest

import keras
from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.backend import distributed_backend


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="Jax Backend specific test",
)
class TestJaxDistributedFunctions(testing.TestCase):
    """Unit tests for the JAX distributed backend standalone functions."""

    def test_compute_gradients_returns_zeros(self):
        """Test that compute_gradients returns correctly shaped zero tensors."""
        loss = ops.array(10.0)
        trainable_vars = [ops.array([1.0, 2.0]), ops.array(3.0)]
        gradients = distributed_backend.compute_gradients(loss, trainable_vars)
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

        updated_vars = distributed_backend.apply_gradients(
            gradients, trainable_vars, learning_rate
        )
        expected_var1 = ops.array([1.0, 2.0]) - ops.multiply(
            ops.array([0.1, 0.2]), learning_rate
        )
        expected_var2 = 5.0 - (0.5 * learning_rate)
        self.assertAllClose(updated_vars[0], expected_var1)
        self.assertAllClose(updated_vars[1], expected_var2)

    def test_create_optimizer(self):
        """Test optimizer configuration creation."""
        adam_config = distributed_backend.create_optimizer(
            "adam", learning_rate=0.01
        )
        self.assertIsInstance(adam_config, dict)
        self.assertEqual(adam_config["name"], "adam")
        self.assertEqual(adam_config["learning_rate"], 0.01)

        sgd_config = distributed_backend.create_optimizer(
            "sgd", learning_rate=0.1, momentum=0.9
        )
        self.assertIsInstance(sgd_config, dict)
        self.assertEqual(sgd_config["name"], "sgd")
        self.assertEqual(sgd_config["learning_rate"], 0.1)
        self.assertEqual(sgd_config["momentum"], 0.9)

        unknown_config = distributed_backend.create_optimizer(
            "some_unknown_optimizer"
        )
        self.assertIsInstance(unknown_config, dict)
        self.assertEqual(unknown_config["name"], "some_unknown_optimizer")
        self.assertEqual(unknown_config["learning_rate"], 0.001)

    def test_get_device_info(self):
        """Test retrieving device information from the JAX backend."""
        info = distributed_backend.get_device_info()
        self.assertEqual(info["backend"], "jax")
        self.assertIsInstance(info["devices"], list)
        self.assertIsInstance(info["device_count"], int)
        self.assertGreater(info["device_count"], 0)
        self.assertEqual(len(info["devices"]), info["device_count"])

    def test_is_multi_device_capable(self):
        """Test the boolean check for multi-device capability."""
        self.assertIsInstance(
            distributed_backend.is_multi_device_capable(), bool
        )

    def test_get_communication_ops_simulated(self):
        """Test the simulated communication ops in a single-device context."""
        comm_ops = distributed_backend.get_communication_ops()
        device_info = distributed_backend.get_device_info()
        simulated_world_size = device_info.get("device_count", 1)

        # Test all_reduce
        x_reduce = ops.array([[1.0, 2.0], [3.0, 4.0]])
        reduced = comm_ops["all_reduce"](x_reduce, op="sum")
        self.assertAllClose(reduced, x_reduce)

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
        if simulated_world_size > 0:
            scatter_data = ops.arange(simulated_world_size * 2)
            scatter_data = ops.reshape(scatter_data, (simulated_world_size, 2))
            x_scatter = ops.cast(scatter_data, dtype="float32")
            scattered = comm_ops["scatter"](x_scatter)
            expected_scatter = ops.split(
                x_scatter, simulated_world_size, axis=0
            )[0]
            self.assertAllClose(scattered, expected_scatter)
