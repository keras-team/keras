import functools

from keras.src import layers
from keras.src import ops
from keras.src import testing
from keras.src.models import Model, Sequential
from keras.src.layers import Input
from keras.src.distribution.tensor_parallel.autoconfig import _gather
from keras.src.distribution.tensor_parallel.autoconfig import _reduce_sum
from keras.src.distribution.tensor_parallel.autoconfig import (
    analyze_dense_layer,
)
from keras.src.distribution.tensor_parallel.autoconfig import get_default_config
from keras.src.distribution.tensor_parallel.tensor_layout import (
    split_tensor_for_parallelism,
)


class AutoConfigTest(testing.TestCase):
    def check_rule(self, rule, expected_device_count, expected_dim):
        """Helper to verify a rule."""
        self.assertIsInstance(rule, functools.partial)
        self.assertEqual(rule.func, split_tensor_for_parallelism)
        self.assertEqual(rule.keywords["device_count"], expected_device_count)
        self.assertEqual(rule.keywords["dim"], expected_dim)

    def test_analyze_dense_layer_directly(self):
        """Tests the heuristic for classifying Dense layers."""
        up_proj_layer = layers.Dense(64, name="up")
        up_proj_layer.build(input_shape=(None, 16))
        self.assertEqual(analyze_dense_layer(up_proj_layer), "up_projection")
        
        down_proj_layer = layers.Dense(16, name="down")
        down_proj_layer.build(input_shape=(None, 64))
        self.assertEqual(analyze_dense_layer(down_proj_layer), "down_projection")
        
        generic_layer = layers.Dense(32, name="generic")
        generic_layer.build(input_shape=(None, 28))
        self.assertEqual(analyze_dense_layer(generic_layer), "dense")
        
        non_dense_layer = layers.LayerNormalization()
        self.assertEqual(analyze_dense_layer(non_dense_layer), "dense")

    def test_simple_mlp_model(self):
        """Tests rule generation for a standard MLP block."""
        device_count = 2
        devices = [f"gpu:{i}" for i in range(device_count)]

        model = Sequential(
            [
                Input(shape=(32,)),
                layers.Dense(128, name="mlp_up"),
                layers.Dense(32, name="mlp_down"),
            ],
            name="mlp_block",
        )

        layout_map = get_default_config(model, devices)
        state_rules = layout_map.state_rules
        output_rules = layout_map.output_rules

        up_kernel_key = "mlp_block/mlp_up/kernel"
        self.assertIn(up_kernel_key, state_rules)
        self.check_rule(state_rules[up_kernel_key], device_count, 1)

        down_kernel_key = "mlp_block/mlp_down/kernel"
        self.assertIn(down_kernel_key, state_rules)
        self.check_rule(state_rules[down_kernel_key], device_count, 0)

        # Access rule directly (fixed structure)
        self.assertIn("mlp_block/mlp_up", output_rules)
        up_output_rule = output_rules["mlp_block/mlp_up"]
        self.assertIsInstance(up_output_rule, functools.partial)
        self.assertEqual(up_output_rule.func, _gather)
        self.assertEqual(up_output_rule.keywords["axis"], -1)

        self.assertIn("mlp_block/mlp_down", output_rules)
        self.assertEqual(output_rules["mlp_block/mlp_down"], _reduce_sum)

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
        model(ops.zeros((1, 10)))

        layout_map = get_default_config(model, devices)
        state_rules = layout_map.state_rules

        emb_key = "transformer/embedding/embeddings"
        self.assertIn(emb_key, state_rules)
        self.check_rule(state_rules[emb_key], device_count, 1)

        qkv_key = "transformer/qkv_proj/kernel"
        self.assertIn(qkv_key, state_rules)
        self.check_rule(state_rules[qkv_key], device_count, 1)

        attn_key = "transformer/attention_output/kernel"
        self.assertIn(attn_key, state_rules)
        self.check_rule(state_rules[attn_key], device_count, 0)

    def test_nested_model(self):
        """Tests that recursive traversal finds layers in nested models."""
        device_count = 2
        devices = [f"gpu:{i}" for i in range(device_count)]
        
        inner_model = Sequential(
            [layers.Dense(64, name="inner_dense")], name="inner_block"
        )
        outer_model = Sequential(
            [
                Input(shape=(32,)),
                layers.Dense(32, name="outer_dense_1"),
                inner_model,
            ],
            name="outer_block",
        )
        layout_map = get_default_config(outer_model, devices)
        state_rules = layout_map.state_rules

        inner_key = "outer_block/inner_block/inner_dense/kernel"
        self.assertIn(inner_key, state_rules)
        self.check_rule(state_rules[inner_key], device_count, 1)