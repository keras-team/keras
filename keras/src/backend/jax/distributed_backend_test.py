import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import pytest

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.backend import distributed_backend


@pytest.mark.skipif(
    backend.backend() != "jax" or jax.device_count() < 2,
    reason="Test requires JAX backend and at least 2 devices",
)
class TestJaxDistributedFunctions(testing.TestCase):
    """Unit tests for the JAX distributed backend standalone functions."""

    def test_get_device_info(self):
        """Test retrieving device information from the JAX backend."""
        info = distributed_backend.get_device_info()
        self.assertEqual(info["backend"], "jax")
        self.assertIsInstance(info["devices"], list)
        self.assertEqual(info["device_count"], 8)

    def test_is_multi_device_capable(self):
        """Test the boolean check for multi-device capability."""
        self.assertTrue(distributed_backend.is_multi_device_capable())

    def test_ops_raise_error_outside_pmap(self):
        """Verify that communication ops fail when not in pmap."""
        comm_ops = distributed_backend.get_communication_ops()
        x = ops.array([1.0, 2.0])
        with self.assertRaisesRegex(NameError, "unbound axis name: data"):
            comm_ops["all_reduce"](x)

    def test_communication_ops_in_pmap(self):
        """Test the communication ops work correctly inside jax.pmap context."""
        comm_ops = distributed_backend.get_communication_ops()
        world_size = distributed_backend.get_device_info()["device_count"]

        x_reduce = ops.array([[1.0, 2.0], [3.0, 4.0]])
        sharded_reduce_input = jnp.stack([x_reduce] * world_size)
        pmapped_reduce = jax.pmap(
            lambda x: comm_ops["all_reduce"](x, op="sum"), axis_name="data"
        )
        reduced_result = pmapped_reduce(sharded_reduce_input)
        expected_reduce = ops.multiply(x_reduce, float(world_size))
        self.assertAllClose(reduced_result[0], expected_reduce)

        x_gather = jnp.arange(world_size * 2, dtype="float32").reshape(
            (world_size, 2)
        )
        pmapped_gather = jax.pmap(
            lambda x: comm_ops["all_gather"](x, axis=0), axis_name="data"
        )
        gathered_result = pmapped_gather(x_gather)
        self.assertAllClose(gathered_result[0], x_gather)

        x_broadcast = ops.array([5.0, 6.0])
        sharded_broadcast_input = jnp.stack(
            [x_broadcast] + [jnp.zeros_like(x_broadcast)] * (world_size - 1)
        )
        pmapped_broadcast = jax.pmap(
            lambda x: comm_ops["broadcast"](x, root=0), axis_name="data"
        )
        broadcasted_result = pmapped_broadcast(sharded_broadcast_input)
        self.assertAllClose(broadcasted_result[0], x_broadcast)

        x_scatter = jnp.arange(world_size * 2, dtype="float32").reshape(
            (world_size, 2)
        )
        sharded_scatter_input = jnp.stack(
            [x_scatter] + [jnp.zeros_like(x_scatter)] * (world_size - 1)
        )
        pmapped_scatter = jax.pmap(
            lambda x: comm_ops["scatter"](x, root=0, axis=0), axis_name="data"
        )
        scattered_result = pmapped_scatter(sharded_scatter_input)

        fixed_scattered_result = jnp.squeeze(scattered_result, axis=1)
        self.assertAllClose(fixed_scattered_result, x_scatter)
