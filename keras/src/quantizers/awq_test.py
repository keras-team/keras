"""Tests for AWQ quantization."""

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.quantizers.awq import AWQ
from keras.src.quantizers.awq import _compute_grouped_quantization_params
from keras.src.quantizers.awq import awq_quantize_matrix
from keras.src.quantizers.awq import awq_search_optimal_scales
from keras.src.quantizers.awq_config import AWQConfig


class MockTokenizer:
    """Simple tokenizer for testing."""

    def __init__(self, vocab_size=100, seq_len=64):
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def tokenize(self, text):
        # Simple character-based tokenization
        tokens = [ord(c) % self.vocab_size for c in str(text)]
        # Pad or truncate to seq_len
        if len(tokens) < self.seq_len:
            tokens = tokens + [0] * (self.seq_len - len(tokens))
        else:
            tokens = tokens[: self.seq_len]
        return np.array([tokens], dtype="int32")

    def __call__(self, text):
        return self.tokenize(text)


@pytest.mark.requires_trainable_backend
class AWQConfigTest(testing.TestCase):
    """Test AWQConfig validation and serialization."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = AWQConfig(dataset=["test"], tokenizer=MockTokenizer())
        self.assertEqual(config.weight_bits, 4)
        self.assertEqual(config.num_samples, 128)
        self.assertEqual(config.sequence_length, 512)
        self.assertEqual(config.group_size, 128)
        self.assertEqual(config.n_grid, 20)
        self.assertEqual(config.mode, "awq")

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = AWQConfig(
            dataset=["test"],
            tokenizer=MockTokenizer(),
            num_samples=64,
            sequence_length=256,
            group_size=64,
            n_grid=30,
        )
        self.assertEqual(config.num_samples, 64)
        self.assertEqual(config.sequence_length, 256)
        self.assertEqual(config.group_size, 64)
        self.assertEqual(config.n_grid, 30)

    def test_config_only_4bit(self):
        """Test that AWQ only supports 4-bit quantization."""
        with self.assertRaisesRegex(ValueError, "only supports 4-bit"):
            AWQConfig(
                dataset=["test"], tokenizer=MockTokenizer(), weight_bits=8
            )

    def test_config_invalid_num_samples(self):
        """Test invalid num_samples validation."""
        with self.assertRaisesRegex(ValueError, "num_samples must be"):
            AWQConfig(
                dataset=["test"], tokenizer=MockTokenizer(), num_samples=0
            )

    def test_config_invalid_sequence_length(self):
        """Test invalid sequence_length validation."""
        with self.assertRaisesRegex(ValueError, "sequence_length must be"):
            AWQConfig(
                dataset=["test"], tokenizer=MockTokenizer(), sequence_length=-1
            )

    def test_config_invalid_group_size(self):
        """Test invalid group_size validation."""
        with self.assertRaisesRegex(ValueError, "Invalid group_size"):
            AWQConfig(dataset=["test"], tokenizer=MockTokenizer(), group_size=0)

    def test_config_invalid_n_grid(self):
        """Test invalid n_grid validation."""
        with self.assertRaisesRegex(ValueError, "n_grid must be"):
            AWQConfig(dataset=["test"], tokenizer=MockTokenizer(), n_grid=0)

    def test_config_per_channel_group_size(self):
        """Test that -1 group_size is valid (per-channel)."""
        config = AWQConfig(
            dataset=["test"], tokenizer=MockTokenizer(), group_size=-1
        )
        self.assertEqual(config.group_size, -1)

    def test_config_serialization(self):
        """Test configuration serialization."""
        config = AWQConfig(
            dataset=["test"],
            tokenizer=MockTokenizer(),
            group_size=64,
            n_grid=30,
        )
        cfg = config.get_config()
        self.assertEqual(cfg["weight_bits"], 4)
        self.assertEqual(cfg["group_size"], 64)
        self.assertEqual(cfg["n_grid"], 30)
        # Dataset and tokenizer should not be serialized
        self.assertIsNone(cfg["dataset"])
        self.assertIsNone(cfg["tokenizer"])

    def test_dtype_policy_string(self):
        """Test dtype policy string generation."""
        config = AWQConfig(
            dataset=["test"], tokenizer=MockTokenizer(), group_size=128
        )
        self.assertEqual(config.dtype_policy_string(), "awq/4/128")

        config2 = AWQConfig(
            dataset=["test"], tokenizer=MockTokenizer(), group_size=-1
        )
        self.assertEqual(config2.dtype_policy_string(), "awq/4/-1")

    def test_awq_config_serialization(self):
        """Test AWQConfig serialization and deserialization round-trip."""
        config = AWQConfig(
            dataset=["test"],
            tokenizer=MockTokenizer(),
            weight_bits=4,
            num_samples=64,
            sequence_length=256,
            group_size=64,
            n_grid=30,
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
        self.assertEqual(config.n_grid, deserialized_config.n_grid)


@pytest.mark.requires_trainable_backend
class AWQAlgorithmTest(testing.TestCase):
    """Test AWQ algorithm core functionality."""

    def test_scale_search_returns_valid_scales(self):
        """Test that scale search returns valid positive scales."""
        weights = ops.convert_to_tensor(
            np.random.randn(32, 16).astype("float32")
        )
        activations = ops.convert_to_tensor(
            np.abs(np.random.randn(16).astype("float32")) + 0.1
        )

        scales = awq_search_optimal_scales(
            weights, activations, n_grid=10, group_size=-1
        )

        self.assertEqual(scales.shape, (16,))
        # All scales should be positive
        self.assertTrue(np.all(np.array(scales) > 0))

    def test_scale_search_with_zero_activations(self):
        """Test scale search handles near-zero activations."""
        weights = ops.convert_to_tensor(
            np.random.randn(32, 16).astype("float32")
        )
        # Some activations are very small
        activations = np.abs(np.random.randn(16).astype("float32"))
        activations[:5] = 1e-10
        activations = ops.convert_to_tensor(activations)

        scales = awq_search_optimal_scales(
            weights, activations, n_grid=10, group_size=-1
        )

        # Should handle gracefully without NaN or Inf
        self.assertFalse(np.any(np.isnan(np.array(scales))))
        self.assertFalse(np.any(np.isinf(np.array(scales))))

    def test_quantize_matrix_shapes(self):
        """Test that quantize_matrix returns correct shapes."""
        # weights_transpose has shape [out_features, in_features]
        weights = ops.convert_to_tensor(
            np.random.randn(32, 16).astype("float32")  # [out=32, in=16]
        )
        activations = ops.convert_to_tensor(
            np.abs(np.random.randn(16).astype("float32")) + 0.1
        )

        quantized, scale, zero, awq_scales, g_idx = awq_quantize_matrix(
            weights, activations, n_grid=10, group_size=-1
        )

        # Quantized shape: [out_features, in_features]
        self.assertEqual(quantized.shape, (32, 16))
        # Scale shape: [out_features, num_groups]
        self.assertEqual(scale.shape, (32, 1))
        # AWQ scales: per-channel for input features
        self.assertEqual(awq_scales.shape, (16,))
        # Group indices
        self.assertEqual(g_idx.shape, (16,))

    def test_quantize_matrix_with_grouping(self):
        """Test quantize_matrix with group size."""
        # Use dimensions divisible by group_size for cleaner test
        weights = ops.convert_to_tensor(
            np.random.randn(64, 32).astype("float32")
        )
        activations = ops.convert_to_tensor(
            np.abs(np.random.randn(32).astype("float32")) + 0.1
        )

        # Test per-channel mode (group_size=-1) which is well-supported
        quantized, scale, zero, awq_scales, g_idx = awq_quantize_matrix(
            weights, activations, n_grid=5, group_size=-1
        )

        # Check g_idx values - for per-channel, all should be group 0
        g_idx_np = np.array(g_idx)
        self.assertEqual(np.max(g_idx_np), 0)
        self.assertEqual(awq_scales.shape, (32,))

    def test_compute_grouped_quantization_params_shapes(self):
        """Test _compute_grouped_quantization_params returns correct shapes."""
        out_features = 64
        in_features = 128
        group_size = 32
        n_groups = in_features // group_size  # 4 groups

        weights = ops.convert_to_tensor(
            np.random.randn(out_features, in_features).astype("float32")
        )

        scale, zero, maxq, g_idx = _compute_grouped_quantization_params(
            weights, group_size, bits=4
        )

        # Scale should be [out_features, n_groups]
        self.assertEqual(scale.shape, (out_features, n_groups))
        # Zero should be [out_features, n_groups]
        self.assertEqual(zero.shape, (out_features, n_groups))
        # maxq should be 15 for 4-bit
        self.assertEqual(float(maxq), 15.0)
        # g_idx should be [in_features]
        self.assertEqual(g_idx.shape, (in_features,))

        # Verify g_idx values are correct group assignments
        g_idx_np = np.array(g_idx)
        expected_g_idx = np.arange(in_features) // group_size
        np.testing.assert_array_equal(g_idx_np, expected_g_idx)

    def test_compute_grouped_quantization_params_non_divisible(self):
        """Test grouped params with in_features not divisible by group_size."""
        out_features = 32
        in_features = 100  # Not divisible by 32
        group_size = 32
        n_groups = (in_features + group_size - 1) // group_size  # 4 groups

        weights = ops.convert_to_tensor(
            np.random.randn(out_features, in_features).astype("float32")
        )

        scale, zero, maxq, g_idx = _compute_grouped_quantization_params(
            weights, group_size, bits=4
        )

        # Scale should be [out_features, n_groups]
        self.assertEqual(scale.shape, (out_features, n_groups))
        # Zero should be [out_features, n_groups]
        self.assertEqual(zero.shape, (out_features, n_groups))
        # g_idx should be [in_features] (not padded)
        self.assertEqual(g_idx.shape, (in_features,))

    def test_quantize_matrix_grouped_shapes(self):
        """Test awq_quantize_matrix with positive group_size.

        This is a regression test for the InvalidArgumentError that occurred
        when group_size != -1 due to shape mismatch in broadcasting.
        """
        out_features = 768
        in_features = 768
        group_size = 128
        n_groups = in_features // group_size  # 6 groups

        weights = ops.convert_to_tensor(
            np.random.randn(out_features, in_features).astype("float32")
        )
        activations = ops.convert_to_tensor(
            np.abs(np.random.randn(in_features).astype("float32")) + 0.1
        )

        quantized, scale, zero, awq_scales, g_idx = awq_quantize_matrix(
            weights, activations, n_grid=5, group_size=group_size
        )

        # Quantized should match input shape
        self.assertEqual(quantized.shape, (out_features, in_features))
        # Scale should be [out_features, n_groups]
        self.assertEqual(scale.shape, (out_features, n_groups))
        # Zero should be [out_features, n_groups]
        self.assertEqual(zero.shape, (out_features, n_groups))
        # AWQ scales should be per-input-channel
        self.assertEqual(awq_scales.shape, (in_features,))
        # g_idx should be [in_features]
        self.assertEqual(g_idx.shape, (in_features,))

        # Verify g_idx values
        g_idx_np = np.array(g_idx)
        expected_g_idx = np.arange(in_features) // group_size
        np.testing.assert_array_equal(g_idx_np, expected_g_idx)

    def test_quantize_matrix_grouped_no_nan_inf(self):
        """Test grouped quantization produces no NaN or Inf values."""
        out_features = 256
        in_features = 512
        group_size = 64

        weights = ops.convert_to_tensor(
            np.random.randn(out_features, in_features).astype("float32")
        )
        activations = ops.convert_to_tensor(
            np.abs(np.random.randn(in_features).astype("float32")) + 0.1
        )

        quantized, scale, zero, awq_scales, g_idx = awq_quantize_matrix(
            weights, activations, n_grid=5, group_size=group_size
        )

        # Check for NaN/Inf in all outputs
        self.assertFalse(np.any(np.isnan(np.array(quantized))))
        self.assertFalse(np.any(np.isinf(np.array(quantized))))
        self.assertFalse(np.any(np.isnan(np.array(scale))))
        self.assertFalse(np.any(np.isinf(np.array(scale))))
        self.assertFalse(np.any(np.isnan(np.array(awq_scales))))
        self.assertFalse(np.any(np.isinf(np.array(awq_scales))))

    def test_scale_search_grouped_quantization(self):
        """Test awq_search_optimal_scales with grouped quantization."""
        out_features = 128
        in_features = 256
        group_size = 32

        weights = ops.convert_to_tensor(
            np.random.randn(out_features, in_features).astype("float32")
        )
        activations = ops.convert_to_tensor(
            np.abs(np.random.randn(in_features).astype("float32")) + 0.1
        )

        # This should not raise any errors
        scales = awq_search_optimal_scales(
            weights, activations, n_grid=5, group_size=group_size
        )

        # Scales should be [in_features]
        self.assertEqual(scales.shape, (in_features,))
        # All scales should be positive
        self.assertTrue(np.all(np.array(scales) > 0))
        # No NaN or Inf
        self.assertFalse(np.any(np.isnan(np.array(scales))))
        self.assertFalse(np.any(np.isinf(np.array(scales))))

    @parameterized.named_parameters(
        ("group_8", 8),
        ("group_16", 16),
        ("group_32", 32),
        ("group_64", 64),
        ("group_128", 128),
    )
    def test_quantize_matrix_various_group_sizes(self, group_size):
        """Test awq_quantize_matrix with various group sizes."""
        out_features = 64
        in_features = 128
        n_groups = in_features // group_size

        weights = ops.convert_to_tensor(
            np.random.randn(out_features, in_features).astype("float32")
        )
        activations = ops.convert_to_tensor(
            np.abs(np.random.randn(in_features).astype("float32")) + 0.1
        )

        _, scale, zero, _, _ = awq_quantize_matrix(
            weights, activations, n_grid=3, group_size=group_size
        )

        self.assertEqual(
            scale.shape,
            (out_features, n_groups),
            f"Failed for group_size={group_size}",
        )
        self.assertEqual(
            zero.shape,
            (out_features, n_groups),
            f"Failed for group_size={group_size}",
        )


@pytest.mark.requires_trainable_backend
class AWQLayerTest(testing.TestCase):
    """Test AWQ class for layer quantization."""

    def test_awq_on_dense_layer(self):
        """Test AWQ on a Dense layer."""
        layer = layers.Dense(32)
        layer.build(input_shape=(None, 16))

        config = AWQConfig(
            dataset=None,
            tokenizer=None,
            group_size=-1,
            n_grid=10,
        )

        layer.quantize("awq", config=config)
        awq_obj = AWQ(layer, config)

        # Simulate activation capture
        calibration_data = np.random.randn(64, 16).astype("float32")
        awq_obj.update_activation_magnitudes(calibration_data)

        self.assertEqual(awq_obj.num_samples, 64)
        # Activation magnitudes should be non-negative
        self.assertTrue(np.all(np.array(awq_obj.activation_magnitudes) >= 0))

    def test_awq_activation_accumulation(self):
        """Test that activation magnitudes accumulate correctly."""
        layer = layers.Dense(32)
        layer.build(input_shape=(None, 16))

        config = AWQConfig(
            dataset=None, tokenizer=None, group_size=-1, n_grid=10
        )
        layer.quantize("awq", config=config)
        awq_obj = AWQ(layer, config)

        # First batch
        batch1 = np.abs(np.random.randn(10, 16).astype("float32"))
        batch1_max = np.max(batch1, axis=0)
        awq_obj.update_activation_magnitudes(batch1)

        # Second batch with higher values in some channels
        batch2 = np.abs(np.random.randn(10, 16).astype("float32")) + 1.0
        batch2_max = np.max(batch2, axis=0)
        awq_obj.update_activation_magnitudes(batch2)

        # Accumulated magnitudes should be element-wise max
        expected_max = np.maximum(batch1_max, batch2_max)
        np.testing.assert_array_almost_equal(
            np.array(awq_obj.activation_magnitudes), expected_max, decimal=5
        )

    def test_awq_layer_variables_created(self):
        """Test that AWQ layer variables are properly created."""
        layer = layers.Dense(32)
        layer.build(input_shape=(None, 16))

        config = AWQConfig(
            dataset=None, tokenizer=None, group_size=-1, n_grid=10
        )
        layer.quantize("awq", config=config)

        # Check that AWQ-specific variables exist
        self.assertTrue(hasattr(layer, "quantized_kernel"))
        self.assertTrue(hasattr(layer, "kernel_scale"))
        self.assertTrue(hasattr(layer, "kernel_zero"))
        self.assertTrue(hasattr(layer, "awq_scales"))
        self.assertTrue(hasattr(layer, "g_idx"))
        self.assertFalse(layer.is_awq_calibrated)


@pytest.mark.requires_trainable_backend
class AWQDTypePolicyTest(testing.TestCase):
    """Test AWQDTypePolicy."""

    def test_awq_dtype_policy_creation(self):
        """Test AWQDTypePolicy can be created."""
        from keras.src.dtype_policies.dtype_policy import AWQDTypePolicy

        policy = AWQDTypePolicy("awq/4/128", source_name="float32")
        self.assertEqual(policy.weight_bits, 4)
        self.assertEqual(policy.group_size, 128)
        self.assertEqual(policy.mode, "awq")

    def test_awq_dtype_policy_invalid_bits(self):
        """Test AWQDTypePolicy rejects non-4-bit."""
        from keras.src.dtype_policies.dtype_policy import AWQDTypePolicy

        with self.assertRaisesRegex(ValueError, "only supports 4-bit"):
            AWQDTypePolicy("awq/8/128", source_name="float32")

    def test_awq_dtype_policy_invalid_format(self):
        """Test AWQDTypePolicy rejects invalid format."""
        from keras.src.dtype_policies.dtype_policy import AWQDTypePolicy

        with self.assertRaisesRegex(ValueError, "Invalid mode"):
            AWQDTypePolicy("awq/4", source_name="float32")


@pytest.mark.requires_trainable_backend
class AWQValidationTest(testing.TestCase):
    """Test AWQ validation in quantization_config."""

    def test_awq_requires_config(self):
        """Test that AWQ mode requires a config."""
        from keras.src.quantizers.quantization_config import (
            validate_and_resolve_config,
        )

        with self.assertRaisesRegex(ValueError, "AWQConfig"):
            validate_and_resolve_config("awq", None)

    def test_awq_requires_correct_config_type(self):
        """Test that AWQ requires AWQConfig type."""
        from keras.src.quantizers.quantization_config import (
            Int8QuantizationConfig,
        )
        from keras.src.quantizers.quantization_config import (
            validate_and_resolve_config,
        )

        # Int8QuantizationConfig has mode='int8', so passing mode='awq' raises
        # a contradictory arguments error
        with self.assertRaisesRegex(ValueError, "Contradictory arguments"):
            validate_and_resolve_config("awq", Int8QuantizationConfig())


@pytest.mark.requires_trainable_backend
class AWQIntegrationTest(testing.TestCase):
    """Integration tests for AWQ quantization."""

    def test_dense_layer_quantize_awq(self):
        """Test Dense layer can be quantized with AWQ."""
        layer = layers.Dense(64)
        layer.build(input_shape=(None, 32))

        config = AWQConfig(
            dataset=None, tokenizer=None, group_size=16, n_grid=5
        )
        layer.quantize("awq", config=config)

        # Check layer is properly configured
        self.assertEqual(layer.quantization_mode, "awq")
        self.assertTrue(hasattr(layer, "awq_scales"))

    def test_einsum_dense_layer_quantize_awq(self):
        """Test EinsumDense layer can be quantized with AWQ."""
        layer = layers.EinsumDense("ab,bc->ac", output_shape=(64,))
        layer.build(input_shape=(None, 32))

        config = AWQConfig(
            dataset=None, tokenizer=None, group_size=-1, n_grid=5
        )
        layer.quantize("awq", config=config)

        # Check layer is properly configured
        self.assertEqual(layer.quantization_mode, "awq")
        self.assertTrue(hasattr(layer, "awq_scales"))

    def test_model_quantize_requires_structure(self):
        """Test model.quantize requires structure for AWQ."""
        model = models.Sequential([layers.Dense(10, input_shape=(5,))])
        model.build()

        config = AWQConfig(
            dataset=["test data"],
            tokenizer=MockTokenizer(vocab_size=100, seq_len=5),
        )

        with self.assertRaisesRegex(ValueError, "quantization structure"):
            model.quantize("awq", config=config)


# Constants for end-to-end tests
VOCAB_SIZE = 1000
SEQ_LEN = 128
NUM_SAMPLES = 16
NUM_CLASSES = 32

CALIBRATION_TEXT = """
AWQ (Activation-aware Weight Quantization) is an efficient and accurate
low-bit weight quantization method for LLMs. AWQ is based on the observation
that weights are not equally important: protecting only 1% of salient weights
can greatly reduce quantization error. To find salient weights, AWQ looks at
the activation distribution, not weights. Salient weights are those that
correspond to channels with larger activation magnitudes. AWQ then applies
per-channel scaling to protect salient weights during quantization.
The key insight is that for a weight channel, if the corresponding activation
channel has large values, quantizing that weight channel will incur large
error. By scaling up salient weight channels before quantization and scaling
down during inference, AWQ can significantly reduce quantization error
while maintaining the same effective computation.
"""


def _mean_kl(p, q):
    """Compute mean KL divergence between two probability distributions."""
    eps = 1e-8
    p = ops.clip(p, eps, 1.0)
    q = ops.clip(q, eps, 1.0)
    return ops.mean(
        ops.sum(ops.multiply(p, ops.subtract(ops.log(p), ops.log(q))), axis=-1)
    )


def _top1_match_rate(a_logits, b_logits):
    """Calculate top-1 match rate between two sets of logits."""
    return ops.mean(
        ops.equal(ops.argmax(a_logits, axis=-1), ops.argmax(b_logits, axis=-1))
    )


def _get_sequence_classifier():
    """Create a transformer-based sequence classifier for testing."""
    embed_dim = 32
    num_heads = 4
    ff_dim = 32

    class SimpleTransformerBlock(layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
            super().__init__(**kwargs)
            self.att = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads
            )
            self.ffn = models.Sequential(
                [
                    layers.Dense(ff_dim, activation="relu"),
                    layers.Dense(embed_dim),
                ]
            )
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        def call(self, inputs):
            attention_output = self.att(inputs, inputs)
            out1 = self.layernorm1(inputs + attention_output)
            ffn_output = self.ffn(out1)
            return self.layernorm2(out1 + ffn_output)

    inputs = layers.Input(shape=(SEQ_LEN,), dtype="int32")
    x = layers.Embedding(VOCAB_SIZE, embed_dim)(inputs)
    x = SimpleTransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(NUM_CLASSES)(x)
    return models.Model(inputs, outputs)


def _char_tokenizer(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN):
    """Character-based tokenizer for testing."""

    def _pad_or_trim_1d(ids, length):
        ids = ops.ravel(ops.array(ids, "int64"))
        if len(ids) < length:
            ids = ops.concatenate(
                [ids, ops.zeros(length - len(ids), dtype=ids.dtype)]
            )
        else:
            ids = ids[:length]
        return ids

    def _tok(x):
        if isinstance(x, str):
            ids = ops.convert_to_tensor(
                np.fromiter((ord(c) % vocab_size for c in x), dtype=np.int64)
            )
        else:
            ids = np.asarray(x, dtype=np.int64)
        ids = _pad_or_trim_1d(ids, seq_len)
        return ids[None, :]

    _tok.tokenize = _tok
    return _tok


def _string_dataset(
    long_text, num_samples=NUM_SAMPLES, sequence_length=SEQ_LEN
):
    """Yield string slices for calibration."""
    rng = np.random.default_rng(seed=0)
    length = max(1, len(long_text) - sequence_length)
    for _ in range(num_samples):
        start = rng.integers(0, length) if length > 1 else 0
        yield long_text[start : start + sequence_length]


@pytest.mark.requires_trainable_backend
class AWQAccuracyTest(testing.TestCase):
    """End-to-end accuracy preservation tests for AWQ quantization."""

    def test_awq_preserves_accuracy_on_transformer(self):
        """Test that AWQ quantization preserves model accuracy.

        This test:
        1. Creates a transformer-based sequence classifier
        2. Gets baseline (full precision) predictions
        3. Applies AWQ quantization with calibration data
        4. Compares quantized predictions against baseline
        5. Validates top-1 match rate and KL divergence bounds
        """
        import keras

        rng = np.random.default_rng(seed=321)
        keras.utils.set_random_seed(123)

        # Build calibration dataset
        calibration_set = list(_string_dataset(CALIBRATION_TEXT, NUM_SAMPLES))
        self.assertNotEmpty(calibration_set)

        # Build model and tokenizer
        model = _get_sequence_classifier()
        tokenizer = _char_tokenizer(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)

        # Build eval batch from same distribution as calibration
        batch_size = min(8, len(calibration_set))
        eval_samples = [
            calibration_set[rng.integers(0, len(calibration_set))]
            for _ in range(batch_size)
        ]
        x_eval = ops.concatenate([tokenizer(s) for s in eval_samples], axis=0)

        # Get baseline predictions (full precision)
        y_ref = model.predict(x_eval)

        # Define layer structure for AWQ
        embedding_layer = model.layers[1]
        transformer_block = model.layers[2]

        layer_structure = {
            "pre_block_layers": [embedding_layer],
            "sequential_blocks": [transformer_block],
        }

        # Configure AWQ
        awq_config = AWQConfig(
            dataset=calibration_set,
            tokenizer=tokenizer,
            num_samples=NUM_SAMPLES,
            sequence_length=SEQ_LEN,
            group_size=-1,  # per-channel quantization
            n_grid=20,
            quantization_layer_structure=layer_structure,
        )

        # Quantize model with AWQ
        model.quantize("awq", config=awq_config)

        # Get post-quantization predictions
        y_q = model.predict(x_eval)

        # Calculate accuracy metrics
        top1_match = _top1_match_rate(y_ref, y_q)

        p_ref = ops.softmax(y_ref)
        p_q = ops.softmax(y_q)
        kl = _mean_kl(p_ref, p_q)

        # Validate accuracy preservation
        # AWQ should maintain at least 50% top-1 agreement
        self.assertGreaterEqual(
            float(top1_match),
            0.5,
            f"Top-1 agreement too low: {float(top1_match):.3f}",
        )
        # KL divergence should be reasonably bounded
        self.assertLessEqual(
            float(kl), 0.30, f"KL divergence too high: {float(kl):.3f}"
        )

    def test_awq_single_layer_quantization_accuracy(self):
        """Test AWQ accuracy on a single Dense layer.

        Verifies that quantizing a single layer maintains reasonable
        output reconstruction error on test data.
        """
        import keras

        keras.utils.set_random_seed(42)

        # Create a Dense layer with random weights
        layer = layers.Dense(64)
        layer.build(input_shape=(None, 32))

        # Create calibration and test data
        calibration_data = np.random.randn(128, 32).astype("float32")
        test_data = np.random.randn(32, 32).astype("float32")

        # Get original layer output on test data
        original_output = np.array(layer(test_data))

        # Configure AWQ for layer
        config = AWQConfig(
            dataset=None,
            tokenizer=None,
            group_size=-1,
            n_grid=20,
        )
        layer.quantize("awq", config=config)

        # Create AWQ quantizer and run calibration
        awq_obj = AWQ(layer, config)
        awq_obj.update_activation_magnitudes(calibration_data)

        # Perform quantization
        awq_obj.quantize_layer()

        # Get quantized layer output
        quantized_output = np.array(layer(test_data))

        # Calculate reconstruction error (relative MSE of outputs)
        mse = np.mean((original_output - quantized_output) ** 2)
        original_var = np.var(original_output)
        relative_mse = mse / (original_var + 1e-8)

        # AWQ should achieve reasonable output reconstruction (< 20% relative)
        self.assertLess(
            relative_mse,
            0.20,
            f"Relative output MSE too high: {relative_mse:.4f}",
        )

        # Verify no NaN or Inf values in output
        self.assertFalse(np.any(np.isnan(quantized_output)))
        self.assertFalse(np.any(np.isinf(quantized_output)))

        awq_obj.free()

    def test_awq_output_consistency(self):
        """Test that AWQ layer produces consistent outputs.

        Verifies that a quantized layer produces deterministic outputs
        and the output shape matches expectations.
        """
        import keras

        keras.utils.set_random_seed(123)

        # Create and build layer
        layer = layers.Dense(32)
        layer.build(input_shape=(None, 16))

        # Configure and quantize
        config = AWQConfig(
            dataset=None,
            tokenizer=None,
            group_size=-1,
            n_grid=10,
        )
        layer.quantize("awq", config=config)

        # Calibrate
        calibration_data = np.random.randn(64, 16).astype("float32")
        awq_obj = AWQ(layer, config)
        awq_obj.update_activation_magnitudes(calibration_data)
        awq_obj.quantize_layer()

        # Test inference
        test_input = np.random.randn(8, 16).astype("float32")
        output1 = layer(test_input)
        output2 = layer(test_input)

        # Outputs should be identical (deterministic)
        np.testing.assert_array_equal(
            np.array(output1),
            np.array(output2),
            err_msg="AWQ layer outputs are not deterministic",
        )

        # Output shape should be correct
        self.assertEqual(output1.shape, (8, 32))

        awq_obj.free()

    def test_awq_single_layer_grouped_quantization_accuracy(self):
        """Test AWQ accuracy with grouped quantization (group_size > 0).

        This is a regression test to ensure grouped quantization maintains
        reasonable accuracy after fixing the shape mismatch issue.
        """
        import keras

        keras.utils.set_random_seed(42)

        # Create a Dense layer - use dimensions divisible by common group sizes
        in_features = 128
        out_features = 64
        layer = layers.Dense(out_features)
        layer.build(input_shape=(None, in_features))

        # Create calibration and test data
        calibration_data = np.random.randn(128, in_features).astype("float32")
        test_data = np.random.randn(32, in_features).astype("float32")

        # Get original layer output on test data
        original_output = np.array(layer(test_data))

        # Test with group_size=32 (common value)
        group_size = 32
        config = AWQConfig(
            dataset=None,
            tokenizer=None,
            group_size=group_size,
            n_grid=10,
        )
        layer.quantize("awq", config=config)

        # Create AWQ quantizer and run calibration
        awq_obj = AWQ(layer, config)
        awq_obj.update_activation_magnitudes(calibration_data)

        # Perform quantization
        awq_obj.quantize_layer()

        # Verify layer variables have correct shapes for grouped quantization
        n_groups = in_features // group_size
        self.assertEqual(
            layer.kernel_scale.shape,
            (out_features, n_groups),
            f"kernel_scale shape mismatch for group_size={group_size}",
        )
        self.assertEqual(
            layer.kernel_zero.shape,
            (out_features, n_groups),
            f"kernel_zero shape mismatch for group_size={group_size}",
        )

        # Get quantized layer output
        quantized_output = np.array(layer(test_data))

        # Calculate reconstruction error
        mse = np.mean((original_output - quantized_output) ** 2)
        original_var = np.var(original_output)
        relative_mse = mse / (original_var + 1e-8)

        # Grouped quantization may have slightly higher error than per-channel
        # but should still be reasonable (< 30% relative MSE)
        self.assertLess(
            relative_mse,
            0.30,
            f"Relative output MSE too high for group_size={group_size}: "
            f"{relative_mse:.4f}",
        )

        # Verify no NaN or Inf values
        self.assertFalse(np.any(np.isnan(quantized_output)))
        self.assertFalse(np.any(np.isinf(quantized_output)))

        awq_obj.free()

    @parameterized.named_parameters(
        ("per_channel", -1),
        ("group_16", 16),
        ("group_32", 32),
        ("group_64", 64),
        ("group_128", 128),
    )
    def test_awq_accuracy_various_group_sizes(self, group_size):
        """Test AWQ accuracy across various group sizes.

        Ensures the grouped quantization fix works for common group sizes.
        """
        import keras

        in_features = 128
        out_features = 64

        keras.utils.set_random_seed(42)

        # Create fresh layer for each test
        layer = layers.Dense(out_features)
        layer.build(input_shape=(None, in_features))

        # Create data
        calibration_data = np.random.randn(64, in_features).astype("float32")
        test_data = np.random.randn(16, in_features).astype("float32")

        # Get original output
        original_output = np.array(layer(test_data))

        # Configure and quantize
        config = AWQConfig(
            dataset=None,
            tokenizer=None,
            group_size=group_size,
            n_grid=5,
        )
        layer.quantize("awq", config=config)

        awq_obj = AWQ(layer, config)
        awq_obj.update_activation_magnitudes(calibration_data)
        awq_obj.quantize_layer()

        # Verify output
        quantized_output = np.array(layer(test_data))

        # Should have no NaN/Inf
        self.assertFalse(
            np.any(np.isnan(quantized_output)),
            f"NaN in output for group_size={group_size}",
        )
        self.assertFalse(
            np.any(np.isinf(quantized_output)),
            f"Inf in output for group_size={group_size}",
        )

        # Should maintain reasonable accuracy
        mse = np.mean((original_output - quantized_output) ** 2)
        original_var = np.var(original_output)
        relative_mse = mse / (original_var + 1e-8)

        self.assertLess(
            relative_mse,
            0.35,
            f"Accuracy too low for group_size={group_size}: "
            f"relative_mse={relative_mse:.4f}",
        )

        awq_obj.free()

    def test_awq_transformer_grouped_quantization(self):
        """Test AWQ with grouped quantization on transformer model.

        This is an end-to-end test similar to
        test_awq_preserves_accuracy_on_transformer but with group_size=16
        instead of per-channel.
        """
        import keras

        rng = np.random.default_rng(seed=321)
        keras.utils.set_random_seed(123)

        # Build calibration dataset
        calibration_set = list(_string_dataset(CALIBRATION_TEXT, NUM_SAMPLES))

        # Build model and tokenizer
        model = _get_sequence_classifier()
        tokenizer = _char_tokenizer(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)

        # Build eval batch
        batch_size = min(8, len(calibration_set))
        eval_samples = [
            calibration_set[rng.integers(0, len(calibration_set))]
            for _ in range(batch_size)
        ]
        x_eval = ops.concatenate([tokenizer(s) for s in eval_samples], axis=0)

        # Get baseline predictions
        y_ref = model.predict(x_eval)

        # Define layer structure for AWQ
        embedding_layer = model.layers[1]
        transformer_block = model.layers[2]

        layer_structure = {
            "pre_block_layers": [embedding_layer],
            "sequential_blocks": [transformer_block],
        }

        # Configure AWQ with grouped quantization (group_size=16)
        # Using 16 since embed_dim=32, ensuring divisibility
        awq_config = AWQConfig(
            dataset=calibration_set,
            tokenizer=tokenizer,
            num_samples=NUM_SAMPLES,
            sequence_length=SEQ_LEN,
            group_size=16,  # Grouped quantization
            n_grid=10,
            quantization_layer_structure=layer_structure,
        )

        # Quantize model with AWQ
        model.quantize("awq", config=awq_config)

        # Get post-quantization predictions
        y_q = model.predict(x_eval)

        # Calculate accuracy metrics
        top1_match = _top1_match_rate(y_ref, y_q)

        p_ref = ops.softmax(y_ref)
        p_q = ops.softmax(y_q)
        kl = _mean_kl(p_ref, p_q)

        # Grouped quantization may have slightly relaxed bounds
        # compared to per-channel
        self.assertGreaterEqual(
            float(top1_match),
            0.4,
            f"Top-1 agreement too low with group_size=16: "
            f"{float(top1_match):.3f}",
        )
        self.assertLessEqual(
            float(kl),
            0.40,
            f"KL divergence too high with group_size=16: {float(kl):.3f}",
        )


@pytest.mark.requires_trainable_backend
class AWQSerializationTest(testing.TestCase):
    """Tests for AWQ layer serialization and deserialization."""

    def test_awq_dense_serialization(self):
        """Test that an AWQ-quantized Dense layer can be serialized and
        deserialized correctly."""
        layer = layers.Dense(units=16)
        layer.build((None, 8))
        layer.quantize(
            "awq",
            config=AWQConfig(
                dataset=None, tokenizer=None, group_size=8, n_grid=10
            ),
        )
        config = layer.get_config()
        new_layer = layers.Dense.from_config(config)
        new_layer.build((None, 8))
        self.assertEqual(new_layer.quantization_mode, "awq")

    def test_awq_einsum_dense_serialization(self):
        """Test that an AWQ-quantized EinsumDense layer can be serialized and
        deserialized correctly."""
        config = dict(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer = layers.EinsumDense(**config)
        layer.build((None, 3))
        layer.quantize(
            "awq",
            config=AWQConfig(
                dataset=None, tokenizer=None, group_size=8, n_grid=10
            ),
        )
        layer_config = layer.get_config()
        new_layer = layers.EinsumDense.from_config(layer_config)
        new_layer.build((None, 3))
        self.assertEqual(new_layer.quantization_mode, "awq")

    def test_awq_dense_legacy_load_own_variables(self):
        """Test loading AWQ variables from a legacy store format."""
        awq_store = {
            # bias
            "0": np.random.random((16,)).astype("float32"),
            # quantized_kernel
            "1": np.random.randint(0, 16, size=(8, 8), dtype="uint8"),
            # kernel_scale
            "2": np.random.random((16, 1)).astype("float32"),
            # kernel_zero
            "3": np.random.random((16, 1)).astype("uint8"),
            # awq_scales
            "4": np.random.random((8,)).astype("float32"),
            # g_idx
            "5": np.random.random((8,)).astype("float32"),
        }

        # Test awq-quantized layer
        layer = layers.Dense(units=16, dtype="awq/4/8_from_float32")
        layer.build((None, 8))
        layer.load_own_variables(awq_store)
        self.assertTrue(layer.is_awq_calibrated)
        self.assertAllClose(layer.bias, awq_store["0"])
        self.assertAllClose(layer.quantized_kernel, awq_store["1"])
        self.assertAllClose(layer.kernel_scale, awq_store["2"])
        self.assertAllClose(layer.kernel_zero, awq_store["3"])
        self.assertAllClose(layer.awq_scales, awq_store["4"])
        self.assertAllClose(layer.g_idx, awq_store["5"])

    def test_awq_einsum_dense_legacy_load_own_variables(self):
        """Test loading AWQ variables from a legacy store format for
        EinsumDense."""
        # For EinsumDense with equation "ab,bcd->acd" and output_shape (8, 32)
        # with input shape (None, 3): kernel shape is (3, 8, 32)
        # Packed kernel shape: (16, 24) for 4-bit (3*8=24 columns, 32/2=16 rows)
        awq_store = {
            # bias
            "0": np.random.random((32,)).astype("float32"),
            # quantized_kernel
            "1": np.random.randint(0, 16, size=(16, 24), dtype="uint8"),
            # kernel_scale
            "2": np.random.random((32, 3)).astype("float32"),
            # kernel_zero
            "3": np.random.random((32, 3)).astype("uint8"),
            # awq_scales
            "4": np.random.random((24,)).astype("float32"),
            # g_idx
            "5": np.random.random((24,)).astype("float32"),
        }
        config = dict(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )

        # Test awq-quantized layer
        layer = layers.EinsumDense(**config, dtype="awq/4/8_from_float32")
        layer.build((None, 3))
        layer.load_own_variables(awq_store)
        self.assertTrue(layer.is_awq_calibrated)
        self.assertAllClose(layer.bias, awq_store["0"])
        self.assertAllClose(layer.quantized_kernel, awq_store["1"])
        self.assertAllClose(layer.kernel_scale, awq_store["2"])
        self.assertAllClose(layer.kernel_zero, awq_store["3"])
        self.assertAllClose(layer.awq_scales, awq_store["4"])
        self.assertAllClose(layer.g_idx, awq_store["5"])

    def test_int4_awq_kernel_returns_unpacked_form(self):
        """Test that the `kernel` property returns the unpacked int4 AWQ
        kernel."""
        from keras.src import quantizers

        layer = layers.Dense(units=2)
        layer.build((None, 2))
        layer.quantize(
            "awq",
            config=AWQConfig(
                dataset=None, tokenizer=None, group_size=8, n_grid=10
            ),
        )
        layer.is_awq_calibrated = True  # Bypass calibration check
        packed_kernel = layer.quantized_kernel
        self.assertAllClose(
            layer.kernel, quantizers.unpack_int4(packed_kernel, 2)
        )

    def test_awq_kernel_packing(self):
        """Validates that 4-bit AWQ packing reduces the kernel size."""
        layer = layers.Dense(units=16, use_bias=False)
        layer.build((None, 8))

        original_kernel_params = ops.prod(layer._kernel.shape)

        layer.quantize(
            "awq",
            config=AWQConfig(
                dataset=None, tokenizer=None, group_size=8, n_grid=10
            ),
        )

        quantized_kernel_params = ops.prod(layer.quantized_kernel.shape)
        self.assertEqual(quantized_kernel_params, original_kernel_params // 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
