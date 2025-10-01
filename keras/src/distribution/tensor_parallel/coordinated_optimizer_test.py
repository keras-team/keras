import numpy as np
from coordinated_optimizer import CoordinatedOptimizer
from coordinated_optimizer import TensorParallelOptimizer

import keras
from keras.src import optimizers
from keras.src import testing


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
                keras.ops.convert_to_tensor(
                    np.ones_like(v.numpy()) * multiplier, dtype="float32"
                )
                for v in variables
            ]
            grads_and_vars_per_shard.append(list(zip(gradients, variables)))
        return grads_and_vars_per_shard

    def test_initialization(self):
        base_optimizer = optimizers.Adam()
        coord = CoordinatedOptimizer(base_optimizer, world_size=4)
        self.assertEqual(coord.base_optimizer, base_optimizer)
        self.assertFalse(coord.shard_optimizer_states)

    def test_parse_variable_name(self):
        coord = CoordinatedOptimizer(optimizers.Adam(), world_size=4)

        self.assertEqual(
            coord._parse_variable_name("dense/kernel/_momentum"),
            ("momentum", "dense/kernel/"),
        )

        self.assertEqual(
            coord._parse_variable_name("dense/bias/_velocity"),
            (
                "velocity",
                "dense/bias/",
            ),
        )
        self.assertEqual(
            coord._parse_variable_name("dense/bias/_v"),
            (
                "v",
                "dense/bias/",
            ),
        )
        self.assertEqual(
            coord._parse_variable_name("dense/bias/_m"),
            (
                "m",
                "dense/bias/",
            ),
        )

    def test_initialize_sharded_states(self):
        model = self._get_simple_model()
        optimizer = optimizers.Adam()
        model.build((None, 10))
        optimizer.build(model.trainable_variables)

        coord = CoordinatedOptimizer(optimizer, world_size=4)
        self.assertTrue(coord.shard_optimizer_states)
        self.assertIn("momentum", coord.sharded_states)
        self.assertIn("velocity", coord.sharded_states)

    def test_apply_gradients_with_replicated_states(self):
        class AdamWithCallCounter(optimizers.Adam):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.apply_gradients_call_count = 0

            def apply_gradients(self, *args, **kwargs):
                self.apply_gradients_call_count += 1

        world_size = 4
        model = self._get_simple_model()
        optimizer = AdamWithCallCounter()
        model.build((None, 10))
        optimizer.build(model.trainable_variables)
        mock_grads = self._get_mock_gradients_and_vars(model, world_size)

        coord = CoordinatedOptimizer(
            optimizer, world_size, shard_optimizer_states=False
        )
        coord._apply_gradients_with_replicated_states(mock_grads, [])
        self.assertEqual(optimizer.apply_gradients_call_count, 1)

    def test_init_from_string(self):
        optimizer = TensorParallelOptimizer("adam", world_size=4)
        self.assertIsInstance(optimizer.base_optimizer, optimizers.Adam)

    def test_apply_gradients_delegation(self):
        world_size = 4
        base_opt = optimizers.Adam()
        optimizer = TensorParallelOptimizer(base_opt, world_size)
        model = self._get_simple_model()
        mock_grads = self._get_mock_gradients_and_vars(model, world_size)

        coord_apply_tracker = {"called": False}
        optimizer.coordinated_optimizer.apply_gradients = (
            lambda *a, **kw: coord_apply_tracker.update({"called": True})
        )
        base_apply_tracker = {"called": False}
        optimizer.base_optimizer.apply_gradients = (
            lambda *a, **kw: base_apply_tracker.update({"called": True})
        )

        optimizer.apply_gradients(mock_grads, shard_models=[])
        self.assertTrue(coord_apply_tracker["called"])
        self.assertFalse(base_apply_tracker["called"])

        coord_apply_tracker["called"] = False
        unsharded_grads = mock_grads[0]
        optimizer.apply_gradients(unsharded_grads)
        self.assertTrue(base_apply_tracker["called"])
        self.assertFalse(coord_apply_tracker["called"])

    def test_build(self):
        optimizer = TensorParallelOptimizer(optimizers.Adam(), world_size=4)
        model = self._get_simple_model()

        optimizer.build(model.trainable_variables)

        self.assertTrue(optimizer.built)
        self.assertTrue(optimizer.coordinated_optimizer.shard_optimizer_states)

    def test_serialization(self):
        world_size = 4
        base_opt = optimizers.Adam(learning_rate=0.1)
        optimizer = TensorParallelOptimizer(base_opt, world_size)

        config = optimizer.get_config()
        recreated = TensorParallelOptimizer.from_config(config)

        self.assertEqual(recreated.world_size, world_size)
        self.assertIsInstance(recreated.base_optimizer, optimizers.Adam)
        self.assertAllClose(recreated.base_optimizer.learning_rate.numpy(), 0.1)
