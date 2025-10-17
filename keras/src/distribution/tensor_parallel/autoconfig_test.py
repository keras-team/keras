from autoconfig import analyze_dense_layer
from autoconfig import get_default_config_keras

import keras
from keras.src import layers
from keras.src import testing


class AutoConfigTest(testing.TestCase):
    def test_analyze_dense_layer_directly(self):
        """Tests the heuristic for classifying Dense layers."""

        up_proj_layer = layers.Dense(64, name="up")
        up_proj_layer.build(input_shape=(None, 16))
        self.assertEqual(analyze_dense_layer(up_proj_layer), "up_projection")
        down_proj_layer = layers.Dense(16, name="down")
        down_proj_layer.build(input_shape=(None, 64))
        self.assertEqual(
            analyze_dense_layer(down_proj_layer),
            "down_projection",
        )
        generic_layer = layers.Dense(32, name="generic")
        generic_layer.build(input_shape=(None, 28))
        self.assertEqual(analyze_dense_layer(generic_layer), "dense")
        non_dense_layer = layers.LayerNormalization()
        self.assertEqual(analyze_dense_layer(non_dense_layer), "dense")

    def test_simple_mlp_model(self):
        """Tests rule generation for a standard MLP block."""
        device_count = 2
        devices = [f"gpu:{i}" for i in range(device_count)]

        model = keras.Sequential(
            [
                keras.Input(shape=(32,)),
                layers.Dense(128, name="mlp_up"),
                layers.Dense(32, name="mlp_down"),
            ],
            name="mlp_block",
        )

        layout_map = get_default_config_keras(model, devices)
        state_rules = layout_map.state_rules
        output_rules = layout_map.output_rules

        up_kernel_key = "mlp_block.mlp_up.kernel"
        up_kernel_rule = state_rules[up_kernel_key]
        self.assertEqual(up_kernel_rule.device_count, device_count)
        self.assertEqual(up_kernel_rule.dim, 1)

        down_kernel_key = "mlp_block.mlp_down.kernel"
        down_kernel_rule = state_rules[down_kernel_key]
        self.assertEqual(down_kernel_rule.device_count, device_count)
        self.assertEqual(down_kernel_rule.dim, 0)

        self.assertEqual(output_rules["mlp_block.mlp_up"], {0: "gather"})
        self.assertEqual(output_rules["mlp_block.mlp_down"], {0: "allreduce"})

    def test_model_with_embedding_and_einsumdense(self):
        """Tests rule generation for Embedding and EinsumDense layers."""
        device_count = 4
        devices = [f"gpu:{i}" for i in range(device_count)]

        class SimpleTransformer(layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
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

        expected_key = "transformer.embedding.embeddings"
        self.assertIn(expected_key, state_rules)
        emb_rule = state_rules[expected_key]
        self.assertEqual(emb_rule.device_count, device_count)
        self.assertEqual(emb_rule.dim, 1)

        qkv_key = "transformer.qkv_proj.kernel"
        qkv_rule = state_rules[qkv_key]
        self.assertEqual(qkv_rule.device_count, device_count)
        self.assertEqual(qkv_rule.dim, 1)

        attn_out_key = "transformer.attention_output.kernel"
        attn_out_rule = state_rules[attn_out_key]
        self.assertEqual(attn_out_rule.device_count, device_count)
        self.assertEqual(attn_out_rule.dim, 0)

    def test_nested_model(self):
        """Tests that the recursive traversal finds layers in nested models."""
        device_count = 2
        devices = [f"gpu:{i}" for i in range(device_count)]
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

        expected_key = "outer_block.inner_block.inner_dense.kernel"
        self.assertIn(expected_key, state_rules)
        inner_rule = state_rules[expected_key]
        self.assertEqual(inner_rule.device_count, device_count)
        self.assertEqual(inner_rule.dim, 1)
