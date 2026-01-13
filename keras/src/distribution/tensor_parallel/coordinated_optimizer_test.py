import numpy as np
import pytest

import keras
from keras import ops
from keras import optimizers
from keras.src import backend
from keras.src import testing
from keras.src.distribution.tensor_parallel.coordinated_optimizer import (
    TensorParallelOptimizer,
)


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="This test is for the JAX backend only.",
)
class TensorParallelOptimizerTest(testing.TestCase):
    """Tests for the TensorParallelOptimizer class."""

    def _get_simple_model(self):
        """Creates a simple, uncompiled Keras model for testing."""
        inputs = keras.Input(shape=(10,))
        x = keras.layers.Dense(20, name="dense_1")(inputs)
        outputs = keras.layers.Dense(5, name="dense_2")(x)
        return keras.Model(inputs, outputs)

    def _get_mock_gradients_and_vars(self, model, device_count):
        """Generates mock gradients and variables for N shards."""
        model.build(input_shape=(None, 10))
        variables = model.trainable_variables
        grads_and_vars_per_shard = []
        for i in range(device_count):
            multiplier = float(i + 1)
            gradients = [
                ops.convert_to_tensor(
                    np.ones_like(v.numpy()) * multiplier, dtype="float32"
                )
                for v in variables
            ]
            grads_and_vars_per_shard.append(list(zip(gradients, variables)))
        return grads_and_vars_per_shard

    def test_initialization(self):
        """Verifies optimizer initializes with correct base optimizer."""
        base_optimizer = optimizers.Adam()
        optimizer = TensorParallelOptimizer(base_optimizer, device_count=4)
        self.assertEqual(optimizer.base_optimizer, base_optimizer)
        self.assertTrue(optimizer.shard_optimizer_states)
        self.assertEqual(optimizer._sharded_states, {})

    def test_init_from_string(self):
        """Verifies optimizer correctly fetches base optimizer."""
        optimizer = TensorParallelOptimizer("adam", device_count=4)
        self.assertIsInstance(optimizer.base_optimizer, optimizers.Adam)

    def test_build_and_state_sharding(self):
        """Verifies building optimizer partitions state variables correctly."""
        device_count = 4
        optimizer = TensorParallelOptimizer(
            optimizers.Adam(), device_count=device_count
        )
        model = self._get_simple_model()
        model.build(input_shape=(None, 10))

        optimizer.build(model.trainable_variables)
        self.assertTrue(optimizer.built)

        sharded_states = optimizer._sharded_states
        self.assertIn("iterations", sharded_states)

        self.assertIn("momentum", sharded_states)
        self.assertIn("velocity", sharded_states)

        dense_1_kernel_path = model.get_layer("dense_1").kernel.path
        self.assertIn(dense_1_kernel_path, sharded_states["momentum"])
        self.assertEqual(
            len(sharded_states["momentum"][dense_1_kernel_path]), device_count
        )

    def test_apply_gradients_fallback(self):
        """Checks fallback logic when grads are not sharded."""
        device_count = 2
        base_opt = optimizers.Adam()
        optimizer = TensorParallelOptimizer(base_opt, device_count=device_count)
        model = self._get_simple_model()
        model.build((None, 10))

        grads = [ops.zeros_like(v) for v in model.trainable_variables]
        grads_and_vars = list(zip(grads, model.trainable_variables))

        optimizer.apply_gradients(grads_and_vars)
        self.assertEqual(int(optimizer.iterations), 1)

    def test_synchronize_gradients_logic(self):
        """Verifies that non-sharded variables undergo gradient averaging."""
        device_count = 2
        model = self._get_simple_model()
        optimizer = TensorParallelOptimizer(
            optimizers.SGD(), device_count=device_count
        )

        mock_grads = self._get_mock_gradients_and_vars(model, device_count)
        synced = optimizer._synchronize_gradients(mock_grads)

        for shard_idx in range(device_count):
            grad_val = ops.convert_to_numpy(synced[shard_idx][0][0])
            self.assertAllClose(grad_val, np.ones_like(grad_val) * 1.5)

    def test_serialization(self):
        """Verifies optimizer config serialization and reconstruction."""
        device_count = 8
        base_opt = optimizers.Adam(learning_rate=0.01)
        optimizer = TensorParallelOptimizer(
            base_opt, device_count=device_count, shard_optimizer_states=False
        )

        config = optimizer.get_config()
        recreated = TensorParallelOptimizer.from_config(config)

        self.assertEqual(recreated.device_count, device_count)
        self.assertFalse(recreated.shard_optimizer_states)
        self.assertIsInstance(recreated.base_optimizer, optimizers.Adam)
        self.assertAllClose(recreated.base_optimizer.learning_rate, 0.01)
