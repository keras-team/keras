import os

if "WORLD_SIZE" not in os.environ:
    os.environ["WORLD_SIZE"] = "4"

from keras import Input
from keras import Model
from keras import layers
from keras.src import testing
from keras.src.distribution.tensor_parallel.autoconfig import (
    analyze_dense_layer_directly,
)
from keras.src.distribution.tensor_parallel.autoconfig import (
    get_default_config_keras,
)
from keras.src.distribution.tensor_parallel.state_action_keras import SplitKeras


class TestAutoConfigKeras(testing.TestCase):
    def setUp(self):
        """Set up the test case and common variables."""
        super().setUp()
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.device_ids = [f"device:{i}" for i in range(self.world_size)]

    def _assert_split_keras_equal(self, rule1, rule2):
        """
        Helper to compare two SplitKeras objects by their attributes.
        MODIFIED: Use vars() for robust comparison without knowing attr names.
        """
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

    def test_simple_mlp_sharding(self):
        """Tests a simple MLP with up and down projection layers."""
        inputs = Input(shape=(64,))
        x = layers.Dense(256, name="up_projection_layer", use_bias=True)(inputs)
        outputs = layers.Dense(
            64, name="down_projection_layer", use_bias=False
        )(x)
        model = Model(inputs=inputs, outputs=outputs, name="simple_mlp")

        config = get_default_config_keras(model, self.device_ids)

        expected_state_rules = {
            r"^up_projection_layer.kernel$": SplitKeras(
                self.world_size, 1, "column"
            ),
            r"^up_projection_layer.bias$": SplitKeras(
                self.world_size, 0, "column"
            ),
            r"^down_projection_layer.kernel$": SplitKeras(
                self.world_size, 0, "row"
            ),
        }
        expected_output_rules = {
            r"^up_projection_layer$": {0: "no_comm"},
            r"^down_projection_layer$": {0: "allreduce"},
        }

        self._assert_rules_equal(config.state_rules, expected_state_rules)
        self._assert_rules_equal(config.output_rules, expected_output_rules)

    def test_embedding_sharding(self):
        """Tests an Embedding layer."""
        inputs = Input(shape=(10,), dtype="int32")
        outputs = layers.Embedding(
            input_dim=1000, output_dim=128, name="token_embedding"
        )(inputs)
        model = Model(inputs=inputs, outputs=outputs, name="embed_model")

        config = get_default_config_keras(model, self.device_ids)

        expected_state_rules = {
            r"^token_embedding\.embeddings$": SplitKeras(
                self.world_size, 1, "column"
            )
        }
        expected_output_rules = {r"^token_embedding$": {0: "no_comm"}}

        self._assert_rules_equal(config.state_rules, expected_state_rules)
        self._assert_rules_equal(config.output_rules, expected_output_rules)

    def test_nested_model_sharding(self):
        """Tests that the traversal logic correctly handles nested models."""
        inner_inputs = Input(shape=(32,))
        inner_outputs = layers.Dense(128, name="inner_dense")(inner_inputs)
        inner_model = Model(
            inputs=inner_inputs, outputs=inner_outputs, name="inner_block"
        )

        outer_inputs = Input(shape=(32,))
        x = inner_model(outer_inputs)
        outer_outputs = layers.Dense(32, name="outer_dense")(x)
        outer_model = Model(
            inputs=outer_inputs, outputs=outer_outputs, name="outer_model"
        )

        config = get_default_config_keras(outer_model, self.device_ids)

        expected_state_rules = {
            r"^inner_block.inner_dense.kernel$": SplitKeras(
                self.world_size, 1, "column"
            ),
            r"^inner_block.inner_dense.bias$": SplitKeras(
                self.world_size, 0, "column"
            ),
            r"^outer_dense.kernel$": SplitKeras(self.world_size, 0, "row"),
        }
        expected_output_rules = {
            r"^inner_block.inner_dense$": {0: "no_comm"},
            r"^outer_dense$": {0: "allreduce"},
        }

        self._assert_rules_equal(config.state_rules, expected_state_rules)
        self._assert_rules_equal(config.output_rules, expected_output_rules)
