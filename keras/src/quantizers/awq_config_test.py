import pytest

from keras.src import testing
from keras.src.quantizers.awq_config import AWQConfig


@pytest.mark.requires_trainable_backend
class AWQConfigTest(testing.TestCase):
    """Test AWQConfig validation and serialization."""

    class MockTokenizer:
        """Mock tokenizer for testing purposes."""

        def __init__(self):
            pass

    def test_config_defaults(self):
        """Test default configuration values."""
        config = AWQConfig(dataset=["test"], tokenizer=self.MockTokenizer())
        self.assertEqual(config.weight_bits, 4)
        self.assertEqual(config.num_samples, 128)
        self.assertEqual(config.sequence_length, 512)
        self.assertEqual(config.group_size, 128)
        self.assertEqual(config.num_grid_points, 20)
        self.assertEqual(config.mode, "awq")

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = AWQConfig(
            dataset=["test"],
            tokenizer=self.MockTokenizer(),
            num_samples=64,
            sequence_length=256,
            group_size=64,
            num_grid_points=30,
        )
        self.assertEqual(config.num_samples, 64)
        self.assertEqual(config.sequence_length, 256)
        self.assertEqual(config.group_size, 64)
        self.assertEqual(config.num_grid_points, 30)

    def test_config_only_4bit(self):
        """Test that AWQ only supports 4-bit quantization."""
        with self.assertRaisesRegex(ValueError, "only supports 4-bit"):
            AWQConfig(
                dataset=["test"], tokenizer=self.MockTokenizer(), weight_bits=8
            )

    def test_config_invalid_num_samples(self):
        """Test invalid num_samples validation."""
        with self.assertRaisesRegex(ValueError, "num_samples must be"):
            AWQConfig(
                dataset=["test"], tokenizer=self.MockTokenizer(), num_samples=0
            )

    def test_config_invalid_sequence_length(self):
        """Test invalid sequence_length validation."""
        with self.assertRaisesRegex(ValueError, "sequence_length must be"):
            AWQConfig(
                dataset=["test"],
                tokenizer=self.MockTokenizer(),
                sequence_length=-1,
            )

    def test_config_invalid_group_size(self):
        """Test invalid group_size validation."""
        with self.assertRaisesRegex(ValueError, "Invalid group_size"):
            AWQConfig(
                dataset=["test"], tokenizer=self.MockTokenizer(), group_size=0
            )

    def test_config_invalid_num_grid_points(self):
        """Test invalid num_grid_points validation."""
        with self.assertRaisesRegex(ValueError, "num_grid_points must be"):
            AWQConfig(
                dataset=["test"],
                tokenizer=self.MockTokenizer(),
                num_grid_points=0,
            )

    def test_config_per_channel_group_size(self):
        """Test that -1 group_size is valid (per-channel)."""
        config = AWQConfig(
            dataset=["test"], tokenizer=self.MockTokenizer(), group_size=-1
        )
        self.assertEqual(config.group_size, -1)

    def test_config_serialization(self):
        """Test configuration serialization."""
        config = AWQConfig(
            dataset=["test"],
            tokenizer=self.MockTokenizer(),
            group_size=64,
            num_grid_points=30,
        )
        cfg = config.get_config()
        self.assertEqual(cfg["weight_bits"], 4)
        self.assertEqual(cfg["group_size"], 64)
        self.assertEqual(cfg["num_grid_points"], 30)
        # Dataset and tokenizer should not be serialized
        self.assertIsNone(cfg["dataset"])
        self.assertIsNone(cfg["tokenizer"])

    def test_dtype_policy_string(self):
        """Test dtype policy string generation."""
        config = AWQConfig(
            dataset=["test"], tokenizer=self.MockTokenizer(), group_size=128
        )
        self.assertEqual(config.dtype_policy_string(), "awq/4/128")

        config2 = AWQConfig(
            dataset=["test"], tokenizer=self.MockTokenizer(), group_size=-1
        )
        self.assertEqual(config2.dtype_policy_string(), "awq/4/-1")

    def test_awq_config_serialization(self):
        """Test AWQConfig serialization and deserialization round-trip."""
        config = AWQConfig(
            dataset=["test"],
            tokenizer=self.MockTokenizer(),
            weight_bits=4,
            num_samples=64,
            sequence_length=256,
            group_size=64,
            num_grid_points=30,
        )
        serialized_config = config.get_config()
        deserialized_config = AWQConfig.from_config(serialized_config)
        # Compare the serializable fields (dataset/tokenizer are not serialized)
        self.assertEqual(config.weight_bits, deserialized_config.weight_bits)
        self.assertEqual(config.num_samples, deserialized_config.num_samples)
        self.assertEqual(
            config.sequence_length, deserialized_config.sequence_length
        )
        self.assertEqual(config.group_size, deserialized_config.group_size)
        self.assertEqual(
            config.num_grid_points, deserialized_config.num_grid_points
        )
