import numpy as np
import pytest

import keras
from keras import ops
from keras.src import optimizers
from keras.src import testing

if keras.backend.backend() == "jax":
    from keras.src.distribution.tensor_parallel.coordinated_optimizer import (
        CoordinatedOptimizer,
    )
    from keras.src.distribution.tensor_parallel.coordinated_optimizer import (
        TensorParallelOptimizer,
    )


@pytest.mark.skipif(
    keras.backend.backend() != "jax",
    reason="This test is JAX-specific.",
)
class CoordinatedOptimizerTest(testing.TestCase):
    def _get_simple_model(self):
        """Creates a simple, uncompiled Keras model."""
        inputs = keras.Input(shape=(10,))
        x = keras.layers.Dense(20, name="dense_1")(inputs)
        outputs = keras.layers.Dense(5, name="dense_2")(x)
        return keras.Model(inputs, outputs)

    def _get_mock_gradients_and_vars(self, model, world_size):
        """Generates mock gradients and variables for N shards."""
        model.build(input_shape=(None, 10))
        variables = model.trainable_variables
        grads_and_vars_per_shard = []
        for i in range(world_size):
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
        """Tests that the optimizer initializes with the correct defaults."""
        base_optimizer = optimizers.Adam()
        coord = CoordinatedOptimizer(base_optimizer, world_size=4)
        self.assertEqual(coord.base_optimizer, base_optimizer)
        self.assertTrue(coord.shard_optimizer_states)
        self.assertEqual(coord.sharded_states, {})

    def test_apply_gradients_with_replicated_states(self):
        """Tests that replicated gradients are averaged and applied once."""

        class AdamWithCallCounter(optimizers.Adam):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.apply_gradients_call_count = 0
                self.received_grads = []

            def apply_gradients(self, grads_and_vars, *args, **kwargs):
                self.apply_gradients_call_count += 1
                self.received_grads = [g for g, v in grads_and_vars]
                super().apply_gradients(grads_and_vars, *args, **kwargs)

        world_size = 4
        model = self._get_simple_model()
        optimizer = AdamWithCallCounter()
        model.build((None, 10))
        mock_grads = self._get_mock_gradients_and_vars(model, world_size)

        coord = CoordinatedOptimizer(
            optimizer,
            world_size,
            shard_optimizer_states=False,
        )
        coord.apply_gradients(mock_grads, [])

        self.assertEqual(optimizer.apply_gradients_call_count, 1)
        self.assertAllClose(
            optimizer.received_grads[0],
            np.ones_like(optimizer.received_grads[0]) * 2.5,
        )

    def test_init_from_string(self):
        optimizer = TensorParallelOptimizer("adam", world_size=4)
        self.assertIsInstance(optimizer.base_optimizer, optimizers.Adam)

    def test_apply_gradients_delegation(self):
        """Tests that apply_gradients correctly delegates."""
        world_size = 4
        base_opt = optimizers.Adam()
        optimizer = TensorParallelOptimizer(base_opt, world_size)
        model = self._get_simple_model()
        mock_grads = self._get_mock_gradients_and_vars(model, world_size)

        coord_apply_tracker = {"called": False}

        def coord_apply_mock(*args, **kwargs):
            coord_apply_tracker["called"] = True

        optimizer.coordinated_optimizer.apply_gradients = coord_apply_mock

        base_apply_tracker = {"called": False}

        def base_apply_mock(*args, **kwargs):
            base_apply_tracker["called"] = True

        optimizer.base_optimizer.apply_gradients = base_apply_mock

        optimizer.apply_gradients(mock_grads, shard_models=[])
        self.assertTrue(coord_apply_tracker["called"])
        self.assertFalse(base_apply_tracker["called"])

        coord_apply_tracker["called"] = False
        unsharded_grads = mock_grads[0]
        optimizer.apply_gradients(unsharded_grads)
        self.assertTrue(base_apply_tracker["called"])
        self.assertFalse(coord_apply_tracker["called"])

    def test_build_and_state_sharding(self):
        """Tests that the build method correctly initializes sharded states."""
        optimizer = TensorParallelOptimizer(optimizers.Adam(), world_size=4)
        model = self._get_simple_model()
        model.build(input_shape=(None, 10))

        self.assertEqual(optimizer.coordinated_optimizer.sharded_states, {})
        optimizer.build(model.trainable_variables)
        self.assertTrue(optimizer.built)

        sharded_states = optimizer.coordinated_optimizer.sharded_states
        self.assertIn("momentum", sharded_states)
        self.assertIn("velocity", sharded_states)
        self.assertIn("iterations", sharded_states)

        dense_1_kernel_path = model.get_layer("dense_1").kernel.path
        self.assertIn(dense_1_kernel_path, sharded_states["momentum"])
        self.assertEqual(
            len(sharded_states["momentum"][dense_1_kernel_path]), 4
        )

    def test_serialization(self):
        world_size = 4
        base_opt = optimizers.Adam(learning_rate=0.1)
        optimizer = TensorParallelOptimizer(
            base_opt, world_size, distributed_backend=None
        )

        config = optimizer.get_config()
        recreated = TensorParallelOptimizer.from_config(config)

        self.assertEqual(recreated.world_size, world_size)
        self.assertIsInstance(recreated.base_optimizer, optimizers.Adam)
        self.assertIsNone(recreated.distributed_backend)
        self.assertAllClose(recreated.base_optimizer.learning_rate, 0.1)

    def test_sharding_with_prefixed_variable_names(self):
        """Tests that state is correctly mapped with prefixed variable names."""
        inputs = keras.Input(shape=(10,))
        x = keras.layers.Dense(4, name="dense")(inputs)
        outputs = keras.layers.Dense(2, name="dense_output")(x)
        model = keras.Model(inputs, outputs)
        model.build(input_shape=(None, 10))

        optimizer = TensorParallelOptimizer(optimizers.Adam(), world_size=2)
        optimizer.build(model.trainable_variables)

        state_to_param = (
            optimizer.coordinated_optimizer._state_variable_to_parameter
        )
        self.assertGreater(len(state_to_param), 0)

        dense_output_kernel = model.get_layer("dense_output").kernel
        optimizer_name = optimizer.base_optimizer.name
        kernel_path = dense_output_kernel.path.replace("/", "_")
        momentum_path = f"{optimizer_name}/{kernel_path}_momentum"

        self.assertIs(state_to_param[momentum_path], dense_output_kernel)
