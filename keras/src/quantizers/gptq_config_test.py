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
        assert config.dtype_policy_string() == "gptq/4/64"
