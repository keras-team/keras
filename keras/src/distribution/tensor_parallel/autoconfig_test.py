import keras
from keras.src import layers
from keras.src import testing

from autoconfig import analyze_dense_layer_directly, get_default_config_keras

class AutoConfigTest(testing.TestCase):
    def test_analyze_dense_layer_directly(self):
        """Tests the heuristic for classifying Dense layers."""
        up_proj_layer = layers.Dense(64, name="up")
        up_proj_layer.build(input_shape=(None, 16))
        self.assertEqual(
            analyze_dense_layer_directly(up_proj_layer, None, ""), "up_projection"
        )
        down_proj_layer = layers.Dense(16, name="down")
        down_proj_layer.build(input_shape=(None, 64))
        self.assertEqual(
            analyze_dense_layer_directly(down_proj_layer, None, ""),
            "down_projection",
        )
        generic_layer = layers.Dense(32, name="generic")
        generic_layer.build(input_shape=(None, 28))
        self.assertEqual(
            analyze_dense_layer_directly(generic_layer, None, ""), "generic_dense"
        )
        non_dense_layer = layers.LayerNormalization()
        self.assertEqual(
            analyze_dense_layer_directly(non_dense_layer, None, ""), "generic_dense"
        )

    def test_simple_mlp_model(self):
        """Tests rule generation for a standard MLP block (like in a Transformer)."""
        world_size = 2
        devices = [f"gpu:{i}" for i in range(world_size)]

        model = keras.Sequential(
            [
                keras.Input(shape=(32,)),
                layers.Dense(128, name="mlp_up"),  # Up-projection
                layers.Dense(32, name="mlp_down"),  # Down-projection
            ],
            name="mlp_block",
        )

        layout_map = get_default_config_keras(model, devices)
        state_rules = layout_map.state_rules
        output_rules = layout_map.output_rules

        # Assertions for State (Weight) Sharding Rules
        up_kernel_rule = state_rules["^mlp_block.mlp_up.kernel$"]
        self.assertEqual(up_kernel_rule.world_size, world_size)
        self.assertEqual(up_kernel_rule.dim, 1)

        down_kernel_rule = state_rules["^mlp_block.mlp_down.kernel$"]
        self.assertEqual(down_kernel_rule.world_size, world_size)
        self.assertEqual(down_kernel_rule.dim, 0)

        # Assertions for Output Communication Rules
        # --- FIX: Removed trailing space. The source code generates "{0: 'gather'}" ---
        self.assertEqual(output_rules["^mlp_block.mlp_up$"], {0: "gather"})
        self.assertEqual(output_rules["^mlp_block.mlp_down$"], {0: "allreduce"})

    def test_model_with_embedding_and_einsumdense(self):
        """Tests rule generation for Embedding and EinsumDense layers."""
        world_size = 4
        devices = [f"gpu:{i}" for i in range(world_size)]

        class SimpleTransformer(layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                # --- FIX: Add explicit `name` arguments to ensure layer names are predictable ---
                self.embedding = layers.Embedding(
                    input_dim=1000, output_dim=64, name="embedding"
                )
                self.qkv_proj = layers.EinsumDense(
                    "abc,cde->abde",
                    output_shape=(None, 3, 128),
                    bias_axes="de",
                    name="qkv_proj",
                )
                self.attention_output = layers.EinsumDense(
                    "abde,cde->abc",
                    output_shape=(None, 64),
                    bias_axes="c",
                    name="attention_output",
                )
            
            def call(self, inputs):
                x = self.embedding(inputs)
                x = self.qkv_proj(x)
                x = self.attention_output(x)
                return x

        model = SimpleTransformer(name="transformer")
        model(keras.ops.zeros((1, 10)))

        layout_map = get_default_config_keras(model, devices)
        state_rules = layout_map.state_rules

        # --- Assertions ---
        # --- FIX: The regex key must match what the provided autoconfig.py generates ---
        expected_key = "^transformer.embedding\\..*embeddings$"
        self.assertIn(expected_key, state_rules)
        emb_rule = state_rules[expected_key]
        self.assertEqual(emb_rule.world_size, world_size)
        self.assertEqual(emb_rule.dim, 1)

        # These assertions are now correct because the layers are explicitly named
        qkv_rule = state_rules["^transformer.qkv_proj.kernel$"]
        self.assertEqual(qkv_rule.world_size, world_size)
        self.assertEqual(qkv_rule.dim, 1)

        attn_out_rule = state_rules["^transformer.attention_output.kernel$"]
        self.assertEqual(attn_out_rule.world_size, world_size)
        self.assertEqual(attn_out_rule.dim, 0)

    def test_nested_model(self):
        """Tests that the recursive traversal finds layers in nested models."""
        # This test is correct and requires no changes.
        world_size = 2
        devices = [f"gpu:{i}" for i in range(world_size)]
        inner_model = keras.Sequential(
            [layers.Dense(64, name="inner_dense")], name="inner_block"
        )
        outer_model = keras.Sequential(
            [
                keras.Input(shape=(32,)),
                layers.Dense(32, name="outer_dense_1"),
                inner_model,
            ],
            name="outer_block",
        )
        layout_map = get_default_config_keras(outer_model, devices)
        state_rules = layout_map.state_rules
        expected_key = "^outer_block.inner_block.inner_dense.kernel$"
        self.assertIn(expected_key, state_rules)
        inner_rule = state_rules[expected_key]
        self.assertEqual(inner_rule.world_size, world_size)
        self.assertEqual(inner_rule.dim, 1)