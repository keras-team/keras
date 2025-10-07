import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
import re

import numpy as np
import pytest

import keras
from keras import distribution
from keras.src.distribution.tensor_parallel.config import ConfigKeras
from keras.src.distribution.tensor_parallel.parameter_sharding import (
    ShardedWeight,
)
from keras.src.distribution.tensor_parallel.parameter_sharding import (
    make_parameter_sharded_model,
)
from keras.src.distribution.tensor_parallel.state_action_keras import SplitKeras
from keras.src.testing import TestCase


@pytest.mark.skipif(
    keras.backend.backend() != "jax",
    reason="This test is JAX-specific.",
)
def _create_simple_mlp():
    """Creates a simple, unsharded Keras MLP model for testing."""
    inputs = keras.Input(shape=(16,), name="input")
    x = keras.layers.Dense(32, use_bias=True, name="up_proj")(inputs)
    x = keras.layers.Activation("relu")(x)
    outputs = keras.layers.Dense(8, use_bias=False, name="down_proj")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="simple_mlp")


class ParameterShardingTest(TestCase):
    def setUp(self):
        super().setUp()
        import logging

        logging.getLogger().setLevel(logging.ERROR)

        self.world_size = 2
        all_devices = distribution.list_devices()
        self.devices = all_devices[: self.world_size]
        if len(self.devices) < self.world_size:
            self.skipTest(
                f"""Not enough devices to run TP test. 
                Found {len(self.devices)}, need {self.world_size}"""
            )

        self.original_model = _create_simple_mlp()
        self.original_model.build(input_shape=(None, 16))

        self.tp_config = ConfigKeras(
            state_rules={
                re.escape("simple_mlp.up_proj.kernel"): SplitKeras(
                    self.world_size, dim=1
                ),
                re.escape("simple_mlp.down_proj.kernel"): SplitKeras(
                    self.world_size, dim=0
                ),
            },
            output_rules={},
        )
        self.input_data = np.random.rand(4, 16).astype("float32")
        self.labels = np.random.rand(4, 8).astype("float32")

    def test_model_sharding_creation_and_weight_counts(self):
        sharded_models = []
        for rank in range(self.world_size):
            with keras.device(self.devices[rank]):
                sharded_model, modified_params = make_parameter_sharded_model(
                    self.original_model,
                    self.tp_config,
                    rank=rank,
                    world_size=self.world_size,
                    device_id=self.devices[rank],
                )
                self.assertIsInstance(sharded_model, keras.Model)
                self.assertIn("simple_mlp.up_proj.kernel", modified_params)
                self.assertIn("simple_mlp.down_proj.kernel", modified_params)
                sharded_models.append(sharded_model)
        self.assertEqual(
            len(self.original_model.weights), len(sharded_models[0].weights)
        )

    def test_sharded_weight_shapes(self):
        rank = 0
        with keras.device(self.devices[rank]):
            sharded_model, _ = make_parameter_sharded_model(
                self.original_model,
                self.tp_config,
                rank=rank,
                world_size=self.world_size,
                device_id=self.devices[rank],
            )
        original_weights_dict = {w.path: w for w in self.original_model.weights}
        sharded_weights_dict = {
            w.name if isinstance(w, ShardedWeight) else w.path: w
            for w in sharded_model.weights
        }
        orig_up_kernel = original_weights_dict["up_proj/kernel"]
        shard_up_kernel = sharded_weights_dict["simple_mlp.up_proj.kernel"]
        self.assertEqual(shard_up_kernel.shape[0], orig_up_kernel.shape[0])
        self.assertEqual(
            shard_up_kernel.shape[1],
            orig_up_kernel.shape[1] // self.world_size,
        )
        orig_down_kernel = original_weights_dict["down_proj/kernel"]
        shard_down_kernel = sharded_weights_dict["simple_mlp.down_proj.kernel"]
        self.assertEqual(
            shard_down_kernel.shape[0],
            orig_down_kernel.shape[0] // self.world_size,
        )
        self.assertEqual(shard_down_kernel.shape[1], orig_down_kernel.shape[1])

    def test_forward_pass_correctness(self):
        expected_output = self.original_model(self.input_data)
        sharded_outputs = []
        original_weights = self.original_model.get_weights()
        for rank in range(self.world_size):
            with keras.device(self.devices[rank]):
                cloned_original = keras.models.clone_model(self.original_model)
                cloned_original.set_weights(original_weights)
                sharded_model, _ = make_parameter_sharded_model(
                    cloned_original,
                    self.tp_config,
                    rank=rank,
                    world_size=self.world_size,
                    device_id=self.devices[rank],
                )
                output = sharded_model(self.input_data)
                sharded_outputs.append(output)
        reconstructed_output = (
            keras.ops.sum(keras.ops.stack(sharded_outputs), axis=0)
            / self.world_size
        )

        self.assertAllClose(
            expected_output, reconstructed_output, atol=1e-5, rtol=1e-5
        )
