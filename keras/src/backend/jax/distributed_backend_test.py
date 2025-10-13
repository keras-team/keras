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
    """Unit tests for the JAX distributed backend functions."""

    def setUp(self):
        """Set up common variables for the tests."""
        super().setUp()
        self.comm_ops = distributed_backend.get_communication_ops()
        self.devices = jax.devices()
        self.world_size = len(self.devices)

    def test_get_device_info(self):
        """Test retrieving device information from the JAX backend."""
        info = distributed_backend.get_device_info()
        self.assertEqual(info["backend"], "jax")
        self.assertIsInstance(info["devices"], list)
        self.assertEqual(info["device_count"], self.world_size)
        self.assertEqual(self.world_size, 8)

    def test_is_multi_device_capable(self):
        """Test the boolean check for multi-device capability."""
        self.assertTrue(distributed_backend.is_multi_device_capable())

    def test_ops_raise_error_outside_parallel_context(self):
        """Verify that communication ops fail when not in pmap/pjit context."""
        x = ops.array([1.0, 2.0])
        with self.assertRaisesRegex(NameError, "unbound axis name: model"):
            self.comm_ops["all_reduce"](x)

    def test_all_reduce_sums_inputs_in_pmap(self):
        """Tests that all_reduce with sum works correctly in pmap context."""
        x_reduce = ops.array([[1.0, 2.0], [3.0, 4.0]])
        sharded_reduce_input = jnp.stack([x_reduce] * self.world_size)

        pmapped_reduce = jax.pmap(
            lambda x: self.comm_ops["all_reduce"](
                x, op="sum", axis_name="data"
            ),
            axis_name="data",
        )
        reduced_result = pmapped_reduce(sharded_reduce_input)

        expected_reduce = ops.multiply(x_reduce, float(self.world_size))
        self.assertAllClose(reduced_result[0], expected_reduce)

    def test_all_reduce_averages_inputs_in_pmap(self):
        """Tests that all_reduce with mean works correctly in pmap context."""
        x_reduce = ops.array([[1.0, 2.0], [3.0, 4.0]])
        sharded_reduce_input = jnp.stack(
            [x_reduce + i for i in range(self.world_size)]
        )

        pmapped_reduce = jax.pmap(
            lambda x: self.comm_ops["all_reduce"](
                x, op="mean", axis_name="data"
            ),
            axis_name="data",
        )
        reduced_result = pmapped_reduce(sharded_reduce_input)

        expected_reduce = jnp.mean(sharded_reduce_input, axis=0)
        self.assertAllClose(reduced_result[0], expected_reduce)

    def test_all_gather_collects_inputs_in_pmap(self):
        """Tests that all_gather correctly collects inputs from all devices."""
        x_gather = jnp.arange(self.world_size * 2, dtype="float32").reshape(
            (self.world_size, 2)
        )

        pmapped_gather = jax.pmap(
            lambda x: self.comm_ops["all_gather"](x, axis=0, axis_name="data"),
            axis_name="data",
        )
        gathered_result = pmapped_gather(x_gather)

        self.assertAllClose(
            gathered_result[0].reshape(x_gather.shape), x_gather
        )
