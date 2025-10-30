import numpy as np
import pytest

import keras
from keras import ops
from keras.src import backend
from keras.src import optimizers
from keras.src import testing
from keras.src.distribution.tensor_parallel.coordinated_optimizer import (
    CoordinatedOptimizer,
)
from keras.src.distribution.tensor_parallel.coordinated_optimizer import (
    TensorParallelOptimizer,
)


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="This test is for the JAX backend only.",
)
class CoordinatedOptimizerTest(testing.TestCase):
    def _get_simple_model(self):
        """Creates a simple, uncompiled Keras model."""
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
        """Tests that the optimizer initializes with the correct defaults."""
        base_optimizer = optimizers.Adam()
        coord = CoordinatedOptimizer(base_optimizer, device_count=4)
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

        device_count = 4
        model = self._get_simple_model()
        optimizer = AdamWithCallCounter()
        model.build((None, 10))
        mock_grads = self._get_mock_gradients_and_vars(model, device_count)

        coord = CoordinatedOptimizer(
            optimizer,
            device_count,
            shard_optimizer_states=False,
        )
        coord.apply_gradients(mock_grads, [])

        self.assertEqual(optimizer.apply_gradients_call_count, 1)
        grad_numpy = ops.convert_to_numpy(optimizer.received_grads[0])
        self.assertAllClose(
            grad_numpy,
            np.ones_like(grad_numpy) * 2.5,
        )

    def test_init_from_string(self):
        optimizer = TensorParallelOptimizer("adam", device_count=4)
        self.assertIsInstance(optimizer.base_optimizer, optimizers.Adam)

    def test_apply_gradients_delegation(self):
        """Tests that apply_gradients correctly delegates."""
        device_count = 4
        base_opt = optimizers.Adam()
        optimizer = TensorParallelOptimizer(base_opt, device_count)
        model = self._get_simple_model()
        mock_grads = self._get_mock_gradients_and_vars(model, device_count)

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
        optimizer = TensorParallelOptimizer(optimizers.Adam(), device_count=4)
        model = self._get_simple_model()
        model.build(input_shape=(None, 10))

        self.assertEqual(optimizer.coordinated_optimizer.sharded_states, {})
        optimizer.build(model.trainable_variables)
        self.assertTrue(optimizer.built)

        sharded_states = optimizer.coordinated_optimizer.sharded_states

        self.assertTrue(
            any("momentum" in key for key in sharded_states),
            msg="Momentum slot not found in sharded_states keys.",
        )
        self.assertTrue(
            any("velocity" in key for key in sharded_states),
            msg="Velocity slot not found in sharded_states keys.",
        )
        self.assertIn("iterations", sharded_states)

        dense_1_kernel_path = model.get_layer("dense_1").kernel.path

        momentum_slot_key = "dense_1_kernel_momentum"

        self.assertIn(dense_1_kernel_path, sharded_states[momentum_slot_key])
        self.assertEqual(
            len(sharded_states[momentum_slot_key][dense_1_kernel_path]), 4
        )

    def test_serialization(self):
        device_count = 4
        base_opt = optimizers.Adam(learning_rate=0.1)

        optimizer = TensorParallelOptimizer(base_opt, device_count)

        config = optimizer.get_config()
        recreated = TensorParallelOptimizer.from_config(config)

        self.assertEqual(recreated.device_count, device_count)
        self.assertIsInstance(recreated.base_optimizer, optimizers.Adam)
        self.assertAllClose(recreated.base_optimizer.learning_rate, 0.1)

    def test_sharding_with_prefixed_variable_names(self):
        """Tests that state is correctly mapped with prefixed variable names."""
        inputs = keras.Input(shape=(10,))
        x = keras.layers.Dense(4, name="dense")(inputs)
        outputs = keras.layers.Dense(2, name="dense_output")(x)
        model = keras.Model(inputs, outputs)
        model.build(input_shape=(None, 10))

        optimizer = TensorParallelOptimizer(optimizers.Adam(), device_count=2)
        optimizer.build(model.trainable_variables)

        state_to_param = optimizer.coordinated_optimizer._slot_path_to_parameter
        self.assertGreater(len(state_to_param), 0)

        dense_output_kernel = model.get_layer("dense_output").kernel

        found_slot_path = None
        for slot_path, param in state_to_param.items():
            if param is dense_output_kernel:
                found_slot_path = slot_path
                break

        self.assertIsNotNone(found_slot_path)
        self.assertIs(state_to_param[found_slot_path], dense_output_kernel)
