from keras.src import layers
from keras.src import testing
from keras.src.quantizers.gptq_config import GPTQConfig


class TestGPTQConfig(testing.TestCase):
    def test_invalid_weight_bits(self):
        with self.assertRaisesRegex(ValueError, "Unsupported weight_bits"):
            GPTQConfig(dataset=None, tokenizer=None, weight_bits=1)
        with self.assertRaisesRegex(ValueError, "Unsupported weight_bits"):
            GPTQConfig(dataset=None, tokenizer=None, weight_bits=5)

    def test_invalid_num_samples(self):
        with self.assertRaisesRegex(
            ValueError, "num_samples must be a positive"
        ):
            GPTQConfig(dataset=None, tokenizer=None, num_samples=0)
        with self.assertRaisesRegex(
            ValueError, "num_samples must be a positive"
        ):
            GPTQConfig(dataset=None, tokenizer=None, num_samples=-1)

    def test_invalid_sequence_length(self):
        with self.assertRaisesRegex(
            ValueError, "sequence_length must be a positive"
        ):
            GPTQConfig(dataset=None, tokenizer=None, sequence_length=0)
        with self.assertRaisesRegex(
            ValueError, "sequence_length must be a positive"
        ):
            GPTQConfig(dataset=None, tokenizer=None, sequence_length=-10)

    def test_invalid_hessian_damping(self):
        with self.assertRaisesRegex(
            ValueError, "hessian_damping must be between"
        ):
            GPTQConfig(dataset=None, tokenizer=None, hessian_damping=-0.1)
        with self.assertRaisesRegex(
            ValueError, "hessian_damping must be between"
        ):
            GPTQConfig(dataset=None, tokenizer=None, hessian_damping=1.1)

    def test_invalid_group_size(self):
        with self.assertRaisesRegex(ValueError, "Invalid group_size"):
            GPTQConfig(dataset=None, tokenizer=None, group_size=0)
        with self.assertRaisesRegex(ValueError, "Invalid group_size"):
            GPTQConfig(dataset=None, tokenizer=None, group_size=-2)

    def test_dtype_policy_string(self):
        config = GPTQConfig(
            dataset=None, tokenizer=None, weight_bits=4, group_size=64
        )
        self.assertEqual(config.dtype_policy_string(), "gptq/4/64")

    def test_gptq_config_serialization(self):
        config = GPTQConfig(
            dataset=None, tokenizer=None, weight_bits=4, group_size=64
        )
        serialized_config = config.get_config()
        deserialized_config = GPTQConfig.from_config(serialized_config)
        self.assertDictEqual(config.__dict__, deserialized_config.__dict__)

    def test_quantization_layer_structure_not_serialized(self):
        # The layer structure may hold live layer objects; like the
        # dataset/tokenizer it is calibration-only state and must not be
        # serialized.
        config = GPTQConfig(
            dataset=None,
            tokenizer=None,
            weight_bits=4,
            group_size=64,
            quantization_layer_structure={
                "pre_block_layers": [],
                "sequential_blocks": [],
            },
        )
        cfg = config.get_config()
        self.assertIsNone(cfg["quantization_layer_structure"])

        restored = GPTQConfig.from_config(cfg)
        self.assertIsNone(restored.quantization_layer_structure)

    def test_live_layer_structure_not_serialized(self):
        """The live layer structure must be dropped on serialization.

        It references live model layers; serializing it would create a
        reference cycle (layer -> config -> layer) and recurse infinitely.
        """
        layer = layers.Dense(4)
        layer.build((None, 4))
        config = GPTQConfig(
            dataset=None,
            tokenizer=None,
            quantization_layer_structure={
                "pre_block_layers": [layer],
                "sequential_blocks": [layer],
            },
        )
        # This must not recurse.
        self.assertIsNone(config.get_config()["quantization_layer_structure"])
        restored = GPTQConfig.from_config(config.get_config())
        self.assertIsNone(restored.quantization_layer_structure)
