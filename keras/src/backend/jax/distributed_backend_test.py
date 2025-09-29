import logging
import os
import unittest
from unittest.mock import patch

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import numpy as np
import optax
import pytest

import keras
from keras.src import backend
from keras.src.backend.jax.distributed_backend import JaxDistributedBackend

logging.disable(logging.WARNING)


class MockVariable:
    """A mock stateful variable with an `assign` method."""

    def __init__(self, value):
        self.value = jnp.array(value, dtype=jnp.float32)

    def assign(self, new_value):
        self.value = jnp.array(new_value)

    def __sub__(self, other):
        return self.value - other

    @property
    def __array_interface__(self):
        return self.value.__array_interface__


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="Backend specific test",
)
class TestJaxDistributedBackend(unittest.TestCase):
    """Unit tests for the JaxDistributedBackend class."""

    def setUp(self):
        """Set up the test case by instantiating the backend."""
        self.backend = JaxDistributedBackend()

    def tearDown(self):
        """Re-enable logging after tests are done."""
        logging.disable(logging.NOTSET)

    def test_get_tensor_lib(self):
        """Test if the correct tensor library (jnp) is returned."""
        self.assertIs(self.backend.get_tensor_lib(), jnp)

    def test_convert_to_backend_tensor(self):
        """Test tensor conversion from various types to JAX arrays."""
        py_list = [1.0, 2.0, 3.0]
        jax_tensor = self.backend.convert_to_backend_tensor(py_list)
        self.assertIsInstance(jax_tensor, jnp.ndarray)
        np.testing.assert_array_equal(jax_tensor, jnp.array([1.0, 2.0, 3.0]))

        np_array = np.array([4.0, 5.0, 6.0])
        jax_tensor = self.backend.convert_to_backend_tensor(np_array)
        self.assertIsInstance(jax_tensor, jnp.ndarray)
        np.testing.assert_array_equal(jax_tensor, jnp.array([4.0, 5.0, 6.0]))

    def test_compute_gradients_returns_zeros(self):
        loss = jnp.array(10.0)
        trainable_vars = [jnp.array([1.0, 2.0]), jnp.array(3.0)]

        gradients = self.backend.compute_gradients(loss, trainable_vars)

        self.assertEqual(len(gradients), 2)
        np.testing.assert_array_equal(
            gradients[0], jnp.zeros_like(trainable_vars[0])
        )
        np.testing.assert_array_equal(
            gradients[1], jnp.zeros_like(trainable_vars[1])
        )

    def test_apply_gradients(self):
        var1 = MockVariable([1.0, 2.0])
        var2 = MockVariable(5.0)
        trainable_vars = [var1, var2]

        grad1 = jnp.array([0.1, 0.2])
        grad2 = jnp.array(0.5)
        gradients = [grad1, grad2]
        learning_rate = 0.1
        self.backend.apply_gradients(gradients, trainable_vars, learning_rate)

        expected_var1 = np.array([1.0 - 0.1 * 0.1, 2.0 - 0.1 * 0.2])
        expected_var2 = 5.0 - 0.1 * 0.5

        np.testing.assert_allclose(var1.value, expected_var1, atol=1e-6)
        np.testing.assert_allclose(var2.value, expected_var2, atol=1e-6)

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
        with patch.object(
            self.backend,
            "get_device_info",
            return_value={
                "backend": "jax",
                "devices": ["cpu:0", "cpu:1"],
                "device_count": 2,
            },
        ):
            with patch.object(
                self.backend, "is_multi_device_capable", return_value=False
            ):
                ops = self.backend.get_communication_ops()
                simulated_world_size = 2

                x_reduce = jnp.array([[1.0, 2.0], [3.0, 4.0]])
                reduced = ops["all_reduce"](x_reduce, op="sum")
                np.testing.assert_allclose(
                    reduced, x_reduce * simulated_world_size
                )

                x_gather = jnp.array([[1.0, 2.0]])
                gathered = ops["all_gather"](x_gather, axis=0)
                expected_gather = keras.ops.concatenate(
                    [x_gather] * simulated_world_size, axis=0
                )
                np.testing.assert_allclose(gathered, expected_gather)

                x_broadcast = jnp.array([5.0, 6.0])
                broadcasted = ops["broadcast"](x_broadcast)
                np.testing.assert_allclose(broadcasted, x_broadcast)

                x_scatter = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
                scattered = ops["scatter"](x_scatter)
                expected_scatter = keras.ops.split(
                    x_scatter, simulated_world_size, axis=0
                )[0]
                np.testing.assert_allclose(scattered, expected_scatter)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
