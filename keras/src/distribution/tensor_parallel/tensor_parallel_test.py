import numpy as np
import pytest

import keras
from keras import layers
from keras.src import backend
from keras.src.distribution.tensor_parallel.tensor_parallel import (
    TensorParallelKeras,
)
from keras.src.testing import TestCase


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="This test is for the JAX backend only.",
)
class TensorParallelKerasTest(TestCase):
    """
    Test suite for the TensorParallelKeras class running on the JAX backend.
    """

    def setUp(self):
        """Set up a reusable model and data for all tests."""
        super().setUp()

        inputs = keras.Input(shape=(64,), name="input_layer")
        x = layers.Dense(128, activation="relu", name="dense_column_sharded")(
            inputs
        )
        outputs = layers.Dense(10, name="dense_row_sharded")(x)
        self.original_model = keras.Model(
            inputs=inputs, outputs=outputs, name="test_mlp"
        )

        self.input_data = np.random.rand(32, 64).astype("float32")
        self.target_data = np.random.rand(32, 10).astype("float32")

        self.world_size = 2
        self.device_ids = [f"cpu:{i}" for i in range(self.world_size)]

    def test_initialization_and_sharding_verification(self):
        """
        Tests if model is correctly initialized and parameter sharding occurs.
        """
        tp_model = TensorParallelKeras(
            self.original_model,
            world_size=self.world_size,
            device_ids=self.device_ids,
        )

        self.assertTrue(tp_model.distributed)
        self.assertEqual(tp_model.world_size, self.world_size)
        self.assertEqual(len(tp_model.model_shards), self.world_size)

        original_params = self.original_model.count_params()
        shard_0_params = tp_model.model_shards[0].count_params()

        self.assertLess(shard_0_params, original_params)

        tp_model_total_params = sum(np.prod(v.shape) for v in tp_model.weights)
        self.assertEqual(tp_model_total_params, original_params)

    def test_non_distributed_case_world_size_one(self):
        """
        Tests the behavior when world_size is 1, ensuring it gracefully degrades
        to a standard, non-distributed model.
        """
        tp_model = TensorParallelKeras(self.original_model, world_size=1)

        self.assertFalse(tp_model.distributed)
        self.assertEqual(tp_model.world_size, 1)
        self.assertEqual(len(tp_model.model_shards), 1)
        self.assertIs(tp_model.assembled_model, self.original_model)

        output = tp_model.predict(self.input_data, verbose=0)
        self.assertEqual(output.shape, (32, 10))

    def test_forward_pass_correctness(self):
        """
        Tests if the output of the sharded model is numerically identical
        to the original model.
        """
        inputs = keras.Input(shape=(64,), name="input_layer")
        x = layers.Dense(
            128, activation="relu", kernel_initializer="glorot_uniform"
        )(inputs)
        outputs = layers.Dense(10, kernel_initializer="glorot_uniform")(x)
        original_model = keras.Model(inputs=inputs, outputs=outputs)

        input_data = np.random.rand(32, 64).astype("float32")

        original_output = original_model(input_data, training=False)

        tp_model = TensorParallelKeras(
            original_model,
            world_size=self.world_size,
            device_ids=self.device_ids,
        )

        tp_output = tp_model(input_data, training=False)

        self.assertAllClose(original_output, tp_output, atol=1e-5, rtol=1e-5)

    def test_distributed_training_workflow(self):
        """
        Tests if model can be compiled and trained for one step without errors.
        """
        tp_model = TensorParallelKeras(
            self.original_model,
            world_size=self.world_size,
            device_ids=self.device_ids,
        )

        tp_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss="mse",
        )

        self.assertTrue(hasattr(tp_model, "coordinated_optimizer"))

        history = tp_model.fit(
            self.input_data,
            self.target_data,
            epochs=1,
            batch_size=16,
            verbose=0,
        )

        self.assertIn("loss", history.history)
        self.assertIsNotNone(history.history["loss"][0])
