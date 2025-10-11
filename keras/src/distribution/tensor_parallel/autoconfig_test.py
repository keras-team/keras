import os

import pytest

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

from keras import Input
from keras import Model
from keras import layers
from keras.src import backend
from keras.src import testing
from keras.src.distribution import distributed_backend
from keras.src.distribution.tensor_parallel.autoconfig import (
    analyze_dense_layer_directly,
)
from keras.src.distribution.tensor_parallel.autoconfig import (
    get_default_config_keras,
)
from keras.src.distribution.tensor_parallel.state_action_keras import SplitKeras


@pytest.mark.skipif(
    backend.backend() not in ("torch", "jax")
    or distributed_backend.get_device_info()["device_count"] <= 1,
    reason="This test is for JAX/PyTorch backends and requires > 1 device.",
)
class TestAutoConfigKeras(testing.TestCase):
    def setUp(self):
        """Set up the test case and common variables."""
        super().setUp()
        device_info = distributed_backend.get_device_info()
        self.world_size = device_info["device_count"]
        self.device_ids = [f"cpu:{i}" for i in range(self.world_size)]

        self.assertGreater(
            self.world_size, 1, "Distribution tests require more than 1 device."
        )

    def _assert_split_keras_equal(self, rule1, rule2):
        """Helper to compare two SplitKeras objects by their attributes."""
        self.assertIsInstance(rule1, SplitKeras)
        self.assertIsInstance(rule2, SplitKeras)
        self.assertDictEqual(vars(rule1), vars(rule2))

    def _assert_rules_equal(self, actual_rules, expected_rules):
        """Helper to compare two dictionaries of sharding rules."""
        self.assertSetEqual(
            set(actual_rules.keys()), set(expected_rules.keys())
        )
        for key in expected_rules:
            actual_val = actual_rules[key]
            expected_val = expected_rules[key]
            if isinstance(expected_val, SplitKeras):
                self._assert_split_keras_equal(actual_val, expected_val)
            else:
                self.assertEqual(actual_val, expected_val)

    def test_analyze_dense_layer(self):
        """Tests the direct analysis and classification of Dense layers."""
        up_proj_layer = layers.Dense(32)
        up_proj_layer.build(input_shape=(None, 16))
        self.assertEqual(
            analyze_dense_layer_directly(up_proj_layer, None, ""),
            "up_projection",
        )

        down_proj_layer = layers.Dense(16)
        down_proj_layer.build(input_shape=(None, 32))
        self.assertEqual(
            analyze_dense_layer_directly(down_proj_layer, None, ""),
            "down_projection",
        )

        generic_layer = layers.Dense(20)
        generic_layer.build(input_shape=(None, 16))
        self.assertEqual(
            analyze_dense_layer_directly(generic_layer, None, ""),
            "generic_dense",
        )

    def test_simple_mlp_sharding(self):
        """Tests a simple MLP with up and down projection layers."""
        inputs = Input(shape=(64,))
        x = layers.Dense(256, name="up_projection_layer", use_bias=True)(inputs)
        outputs = layers.Dense(64, name="down_projection_layer", use_bias=True)(
            x
        )
        model = Model(inputs=inputs, outputs=outputs, name="simple_mlp")

        config = get_default_config_keras(model, self.device_ids)

        expected_state_rules = {
            r"^simple_mlp.up_projection_layer.kernel$": SplitKeras(
                self.world_size, 1, "column"
            ),
            r"^simple_mlp.up_projection_layer.bias$": SplitKeras(
                self.world_size, 0, "column"
            ),
            r"^simple_mlp.down_projection_layer.kernel$": SplitKeras(
                self.world_size, 0, "row"
            ),
            # Bias for down-projection is not sharded according to the new logic
        }
        expected_output_rules = {
            r"^simple_mlp.up_projection_layer$": {0: "gather"},
            r"^simple_mlp.down_projection_layer$": {0: "allreduce"},
        }

        self._assert_rules_equal(config.state_rules, expected_state_rules)
        self._assert_rules_equal(config.output_rules, expected_output_rules)

    def test_generic_dense_sharding(self):
        """Tests a generic Dense layer that isn't an up/down projection."""
        inputs = Input(shape=(64,))
        outputs = layers.Dense(80, name="generic_layer", use_bias=True)(inputs)
        model = Model(inputs=inputs, outputs=outputs, name="generic_model")

        config = get_default_config_keras(model, self.device_ids)

        expected_state_rules = {
            r"^generic_model.generic_layer.kernel$": SplitKeras(
                self.world_size, 1, "column"
            ),
            r"^generic_model.generic_layer.bias$": SplitKeras(
                self.world_size, 0, "column"
            ),
        }
        expected_output_rules = {
            r"^generic_model.generic_layer$": {0: "gather -1"}
        }

        self._assert_rules_equal(config.state_rules, expected_state_rules)
        self._assert_rules_equal(config.output_rules, expected_output_rules)

    def test_embedding_sharding(self):
        """Tests an Embedding layer for vocabulary parallelism."""
        inputs = Input(shape=(10,), dtype="int32")
        outputs = layers.Embedding(
            input_dim=1000, output_dim=128, name="token_embedding"
        )(inputs)
        model = Model(inputs=inputs, outputs=outputs, name="embed_model")

        config = get_default_config_keras(model, self.device_ids)

        expected_state_rules = {
            # FIX: Removed the incorrect backslash before ".token_embedding"
            r"^embed_model.token_embedding\..*embeddings$": SplitKeras(
                self.world_size, 1, "column"
            )
        }
        expected_output_rules = {
            r"^embed_model.token_embedding$": {0: "no_comm"}
        }

        self._assert_rules_equal(config.state_rules, expected_state_rules)
        self._assert_rules_equal(config.output_rules, expected_output_rules)

    def test_einsum_dense_sharding(self):
        """Tests the special handling for EinsumDense layers."""
        inputs = Input(shape=(64,))
        x = layers.EinsumDense(
            "bh,hd->bd", output_shape=128, name="query_proj"
        )(inputs)
        outputs = layers.EinsumDense(
            "bd,dh->bh", output_shape=64, name="attention_output"
        )(x)
        model = Model(inputs=inputs, outputs=outputs, name="einsum_model")

        config = get_default_config_keras(model, self.device_ids)

        expected_state_rules = {
            r"^einsum_model.query_proj.kernel$": SplitKeras(
                self.world_size, 1, "column"
            ),
            r"^einsum_model.attention_output.kernel$": SplitKeras(
                self.world_size, 0, "row"
            ),
        }
        expected_output_rules = {
            r"^einsum_model.query_proj$": {0: "gather -1"},
            r"^einsum_model.attention_output$": {0: "allreduce"},
        }

        self._assert_rules_equal(config.state_rules, expected_state_rules)
        self._assert_rules_equal(config.output_rules, expected_output_rules)

    def test_normalization_layers_ignored(self):
        """Tests that normalization layers are correctly ignored."""
        inputs = Input(shape=(64,))
        x = layers.Dense(64, name="dense1", use_bias=True)(inputs)
        x = layers.LayerNormalization(name="layernorm")(x)
        outputs = layers.Dense(64, name="dense2", use_bias=True)(x)
        model = Model(inputs=inputs, outputs=outputs, name="norm_model")

        config = get_default_config_keras(model, self.device_ids)

        for key in config.state_rules:
            self.assertNotIn("layernorm", key)
        for key in config.output_rules:
            self.assertNotIn("layernorm", key)

        self.assertIn(r"^norm_model.dense1.kernel$", config.state_rules)
        self.assertIn(r"^norm_model.dense2.kernel$", config.state_rules)
        self.assertEqual(len(config.state_rules), 4)
        self.assertEqual(len(config.output_rules), 2)

    def test_nested_model_sharding(self):
        """Tests that the traversal logic correctly handles nested models."""
        inner_inputs = Input(shape=(32,))
        inner_outputs = layers.Dense(128, name="inner_dense", use_bias=True)(
            inner_inputs
        )
        inner_model = Model(
            inputs=inner_inputs, outputs=inner_outputs, name="inner_block"
        )

        outer_inputs = Input(shape=(32,))
        x = inner_model(outer_inputs)
        outer_outputs = layers.Dense(32, name="outer_dense", use_bias=True)(x)
        outer_model = Model(
            inputs=outer_inputs, outputs=outer_outputs, name="outer_model"
        )

        config = get_default_config_keras(outer_model, self.device_ids)

        expected_state_rules = {
            r"^outer_model.inner_block.inner_dense.kernel$": SplitKeras(
                self.world_size, 1, "column"
            ),
            r"^outer_model.inner_block.inner_dense.bias$": SplitKeras(
                self.world_size, 0, "column"
            ),
            r"^outer_model.outer_dense.kernel$": SplitKeras(
                self.world_size, 0, "row"
            ),
            # Bias for down-projection is not sharded according to the new logic
        }
        expected_output_rules = {
            r"^outer_model.inner_block.inner_dense$": {0: "gather"},
            r"^outer_model.outer_dense$": {0: "allreduce"},
        }

        self.maxDiff = None
        self._assert_rules_equal(config.state_rules, expected_state_rules)
        self._assert_rules_equal(config.output_rules, expected_output_rules)
