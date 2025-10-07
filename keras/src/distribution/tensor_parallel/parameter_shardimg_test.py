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
    """Creates a simple, unsharded Keras MLP model for testing.

    This model serves as the baseline for sharding tests. It consists of
    an input layer, a hidden dense layer with a ReLU activation, and an
    output dense layer.

    Returns:
        A `keras.Model` instance.
    """
    inputs = keras.Input(shape=(16,), name="input")
    x = keras.layers.Dense(32, use_bias=True, name="up_proj")(inputs)
    x = keras.layers.Activation("relu")(x)
    outputs = keras.layers.Dense(8, use_bias=False, name="down_proj")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="simple_mlp")


class ParameterShardingTest(TestCase):
    """Test suite for parameter sharding functionality.

    This class tests the creation of sharded models, the correctness of
    sharded weight shapes, and the numerical accuracy of the forward pass
    of a sharded model compared to its original, unsharded counterpart.
    """

    def setUp(self):
        """Sets up the testing environment before each test case."""
        super().setUp()

        self.world_size = 2
        all_devices = distribution.list_devices()
        self.devices = all_devices[: self.world_size]
        if len(self.devices) < self.world_size:
            self.skipTest(
                f"""Not enough devices to run TP test.
                Found {len(self.devices)}, need {self.world_size}"""
            )

        # Create the original model and the sharding configuration.
        self.original_model = _create_simple_mlp()
        self.original_model.build(input_shape=(None, 16))

        self.tp_config = ConfigKeras(
            state_rules={
                # Rule to split the first dense layer's kernel along the output
                # dimension (column-wise).
                re.escape("simple_mlp.up_proj.kernel"): SplitKeras(
                    self.world_size, dim=1
                ),
                # Rule to split the second dense layer's kernel along the input
                # dimension (row-wise).
                re.escape("simple_mlp.down_proj.kernel"): SplitKeras(
                    self.world_size, dim=0
                ),
            },
            output_rules={},
        )
        # Generate dummy data for testing forward passes.
        self.input_data = np.random.rand(4, 16).astype("float32")
        self.labels = np.random.rand(4, 8).astype("float32")

    def test_model_sharding_creation_and_weight_counts(self):
        """Tests if sharded models are created correctly.

        Verifies that:
        1. `make_parameter_sharded_model` returns a valid Keras model.
        2. The set of modified parameters correctly identifies sharded layers.
        3. The total number of weights in the sharded model matches the
           original model, ensuring no weights are lost or added.
        """
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

        # The sharded model should have the same number of weight objects.
        self.assertEqual(
            len(self.original_model.weights), len(sharded_models[0].weights)
        )

    def test_sharded_weight_shapes(self):
        """Validates the shapes of the weights after sharding.

        This test ensures that the dimensions specified in the sharding rules
        are correctly split by the world size, while other dimensions remain
        unchanged.
        """
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

        # Check the shape of the column-split kernel.
        orig_up_kernel = original_weights_dict["up_proj/kernel"]
        shard_up_kernel = sharded_weights_dict["simple_mlp.up_proj.kernel"]
        self.assertEqual(shard_up_kernel.shape[0], orig_up_kernel.shape[0])
        self.assertEqual(
            shard_up_kernel.shape[1],
            orig_up_kernel.shape[1] // self.world_size,
        )

        # Check the shape of the row-split kernel.
        orig_down_kernel = original_weights_dict["down_proj/kernel"]
        shard_down_kernel = sharded_weights_dict["simple_mlp.down_proj.kernel"]
        self.assertEqual(
            shard_down_kernel.shape[0],
            orig_down_kernel.shape[0] // self.world_size,
        )
        self.assertEqual(shard_down_kernel.shape[1], orig_down_kernel.shape[1])

    def test_forward_pass_correctness(self):
        """Checks if the sharded model's output matches the original.

        This test performs a forward pass on both the original model and the
        sharded models. It then reconstructs the output from the sharded
        models and asserts that it is numerically close to the original
        model's output. This serves as an end-to-end correctness check.
        """
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
