import numpy as np
import pytest

from keras.src import Model
from keras.src import backend
from keras.src import layers
from keras.src import ops
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
        inputs = layers.Input(shape=(10,), dtype="float32")
        x = layers.Dense(20, name="dense_1")(inputs)
        outputs = layers.Dense(5, name="dense_2")(x)
        return Model(inputs, outputs)

    def _get_mock_gradients_and_vars(self, model, device_count):
        """Generates mock gradients and variables for N shards."""
        model.build(input_shape=(None, 10))
        variables = model.trainable_variables
        grads_and_vars_per_shard = []
        for i in range(device_count):
            multiplier = float(i + 1)
            gradients = []
            for v in variables:
                v_dtype = backend.standardize_dtype(v.dtype)

                grad = ops.convert_to_tensor(
                    np.ones_like(v.numpy()) * multiplier, dtype=v_dtype
                )
                gradients.append(grad)

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

    def test_serialization(self):
        """Tests manual reconstruction via from_config."""
        device_count = 4
        base_opt = optimizers.Adam(learning_rate=0.1)

        optimizer = TensorParallelOptimizer(base_opt, device_count)

        self.assertEqual(optimizer.device_count, device_count)
        self.assertIsInstance(optimizer.base_optimizer, optimizers.Adam)
        self.assertAllClose(optimizer.base_optimizer.learning_rate, 0.1)
