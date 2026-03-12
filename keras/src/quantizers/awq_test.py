"""Tests for AWQ quantization."""

import numpy as np
import pytest
from absl.testing import parameterized

import keras
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.quantizers.awq import AWQ
from keras.src.quantizers.awq import awq_quantize_matrix
from keras.src.quantizers.awq import awq_search_optimal_scales
from keras.src.quantizers.awq_config import AWQConfig

# Shared RNG instance for reproducible tests
RNG = np.random.default_rng(seed=42)


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
        return ops.array([tokens], dtype="int32")

    def __call__(self, text):
        return self.tokenize(text)


@pytest.mark.requires_trainable_backend
class AWQAlgorithmTest(testing.TestCase):
    """Test AWQ algorithm core functionality."""

    def test_scale_search_returns_valid_scales(self):
        """Test that scale search returns valid positive scales."""
        weights = RNG.standard_normal((32, 16)).astype("float32")
        activations = ops.abs(
            ops.add(RNG.standard_normal((16,)).astype("float32"), 0.1)
        )

        scales = awq_search_optimal_scales(
            weights, activations, num_grid_points=10, group_size=-1
        )

        self.assertEqual(scales.shape, (16,))
        # All scales should be positive
        self.assertTrue(ops.all(ops.greater(scales, 0)))

    def test_scale_search_with_zero_activations(self):
        """Test scale search handles near-zero activations."""
        weights = ops.array(RNG.standard_normal((32, 16)).astype("float32"))
        # Some activations are very small
        activations = np.abs(RNG.standard_normal((16,)).astype("float32"))
        activations[:5] = 1e-10
        activations = ops.array(activations)

        scales = awq_search_optimal_scales(
            weights, activations, num_grid_points=10, group_size=-1
        )

        # Should handle gracefully without NaN or Inf
        self.assertFalse(ops.any(ops.isnan(scales)))
        self.assertFalse(ops.any(ops.isinf(scales)))

    def test_quantize_matrix_shapes(self):
        """Test that quantize_matrix returns correct shapes."""
        # weights_transpose has shape [out_features, in_features]
        weights = ops.array(RNG.standard_normal((32, 16)).astype("float32"))
        activations = ops.add(
            ops.abs(RNG.standard_normal((16,)).astype("float32")), 0.1
        )

        quantized, scale, zero, awq_scales, g_idx = awq_quantize_matrix(
            weights, activations, num_grid_points=10, group_size=-1
        )

        # Quantized shape: [out_features, in_features]
        self.assertEqual(quantized.shape, (32, 16))
        # Scale shape: [out_features, num_groups]
        self.assertEqual(scale.shape, (32, 1))
        # AWQ scales: per-channel for input features
        self.assertEqual(awq_scales.shape, (16,))
        # AWQ zero shape: [out_features, num_groups]
        self.assertEqual(zero.shape, (32, 1))
        # Group indices
        self.assertEqual(g_idx.shape, (16,))

    def test_quantize_matrix_with_grouping(self):
        """Test quantize_matrix with group size."""
        # Use dimensions divisible by group_size for cleaner test
        weights = ops.array(RNG.standard_normal((64, 32)).astype("float32"))
        activations = ops.add(
            ops.abs(RNG.standard_normal((32,)).astype("float32")), 0.1
        )

        # Test per-channel mode (group_size=-1) which is well-supported
        quantized, scale, zero, awq_scales, g_idx = awq_quantize_matrix(
            weights, activations, num_grid_points=5, group_size=8
        )

        # Quantized shape: [out_features, in_features]
        self.assertEqual(quantized.shape, (64, 32))
        # Scale shape: [out_features, num_groups]
        self.assertEqual(scale.shape, (64, 4))  # 32 in_features / 8 group_size
        # AWQ scales: per-channel for input features
        self.assertEqual(awq_scales.shape, (32,))
        # AWQ zero shape: [out_features, num_groups]
        self.assertEqual(zero.shape, (64, 4))
        # Group indices
        self.assertEqual(g_idx.shape, (32,))

        # Check g_idx values
        self.assertEqual(ops.max(g_idx), 3)  # 4 groups: 0,1,2,3
        self.assertEqual(awq_scales.shape, (32,))

    def test_quantize_matrix_grouped_shapes(self):
        """Test awq_quantize_matrix with positive group_size.

        This is a regression test for the InvalidArgumentError that occurred
        when group_size != -1 due to shape mismatch in broadcasting.
        """
        out_features = 768
        in_features = 768
        group_size = 128
        n_groups = in_features // group_size  # 6 groups

        weights = ops.array(
            RNG.standard_normal((out_features, in_features)).astype("float32")
        )
        activations = ops.array(
            np.abs(RNG.standard_normal((in_features,)).astype("float32")) + 0.1
        )

        quantized, scale, zero, awq_scales, g_idx = awq_quantize_matrix(
            weights, activations, num_grid_points=5, group_size=group_size
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
        expected_g_idx = ops.floor_divide(ops.arange(in_features), group_size)
        self.assertAllEqual(g_idx, expected_g_idx)

    def test_quantize_matrix_grouped_no_nan_inf(self):
        """Test grouped quantization produces no NaN or Inf values."""
        out_features = 256
        in_features = 512
        group_size = 64

        weights = ops.array(
            RNG.standard_normal((out_features, in_features)).astype("float32")
        )
        activations = ops.add(
            ops.abs(RNG.standard_normal((in_features,)).astype("float32")), 0.1
        )

        quantized, scale, _, awq_scales, _ = awq_quantize_matrix(
            weights, activations, num_grid_points=5, group_size=group_size
        )

        # Check for NaN/Inf in all outputs
        self.assertFalse(ops.any(ops.isnan(quantized)))
        self.assertFalse(ops.any(ops.isinf(quantized)))
        self.assertFalse(ops.any(ops.isnan(scale)))
        self.assertFalse(ops.any(ops.isinf(scale)))
        self.assertFalse(ops.any(ops.isnan(awq_scales)))
        self.assertFalse(ops.any(ops.isinf(awq_scales)))

    def test_scale_search_grouped_quantization(self):
        """Test awq_search_optimal_scales with grouped quantization."""
        out_features = 128
        in_features = 256
        group_size = 32

        weights = ops.array(
            RNG.standard_normal((out_features, in_features)).astype("float32")
        )
        activations = ops.add(
            ops.abs(RNG.standard_normal((in_features,)).astype("float32")), 0.1
        )

        scales = awq_search_optimal_scales(
            weights, activations, num_grid_points=5, group_size=group_size
        )

        # Scales should be [in_features]
        self.assertEqual(scales.shape, (in_features,))
        # All scales should be positive
        self.assertTrue(ops.all(ops.greater(scales, 0)))
        # No NaN or Inf
        self.assertFalse(ops.any(ops.isnan(scales)))
        self.assertFalse(ops.any(ops.isinf(scales)))

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

        weights = ops.array(
            RNG.standard_normal((out_features, in_features)).astype("float32")
        )
        activations = ops.add(
            ops.abs(RNG.standard_normal((in_features,)).astype("float32")), 0.1
        )

        _, scale, zero, _, _ = awq_quantize_matrix(
            weights, activations, num_grid_points=3, group_size=group_size
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
            num_grid_points=10,
        )

        layer.quantize(config=config)
        awq_obj = AWQ(layer, config)

        # Simulate activation capture
        calibration_data = RNG.standard_normal((64, 16)).astype("float32")
        awq_obj.update_activation_magnitudes(calibration_data)

        self.assertEqual(awq_obj.num_samples, 64)
        # Activation magnitudes should be non-negative
        self.assertTrue(
            ops.all(ops.greater_equal(awq_obj.activation_magnitudes, 0))
        )

    def test_awq_activation_accumulation(self):
        """Test that activation magnitudes accumulate correctly."""
        layer = layers.Dense(32)
        layer.build(input_shape=(None, 16))

        config = AWQConfig(
            dataset=None, tokenizer=None, group_size=-1, num_grid_points=10
        )
        layer.quantize(config=config)
        awq_obj = AWQ(layer, config)

        # First batch
        batch1 = ops.abs(RNG.standard_normal((10, 16)).astype("float32"))
        batch1_max = ops.max(batch1, axis=0)
        awq_obj.update_activation_magnitudes(batch1)

        # Second batch with higher values in some channels
        batch2 = ops.add(
            ops.abs(RNG.standard_normal((10, 16)).astype("float32")), 1.0
        )
        batch2_max = ops.max(batch2, axis=0)
        awq_obj.update_activation_magnitudes(batch2)

        # Accumulated magnitudes should be element-wise max
        expected_max = ops.maximum(batch1_max, batch2_max)
        self.assertAllClose(
            awq_obj.activation_magnitudes, expected_max, atol=1e-6
        )

    def test_awq_layer_variables_created(self):
        """Test that AWQ layer variables are properly created."""
        layer = layers.Dense(32)
        layer.build(input_shape=(None, 16))

        config = AWQConfig(
            dataset=None, tokenizer=None, group_size=-1, num_grid_points=10
        )
        layer.quantize(config=config)

        # Check that AWQ-specific variables exist
        self.assertTrue(hasattr(layer, "quantized_kernel"))
        self.assertTrue(hasattr(layer, "kernel_scale"))
        self.assertTrue(hasattr(layer, "kernel_zero"))
        self.assertTrue(hasattr(layer, "awq_scales"))
        self.assertTrue(hasattr(layer, "g_idx"))
        self.assertFalse(layer.is_awq_calibrated)


@pytest.mark.requires_trainable_backend
class AWQIntegrationTest(testing.TestCase):
    """Integration tests for AWQ quantization."""

    def test_dense_layer_quantize_awq(self):
        """Test Dense layer can be quantized with AWQ."""
        layer = layers.Dense(64)
        layer.build(input_shape=(None, 32))

        config = AWQConfig(
            dataset=None, tokenizer=None, group_size=16, num_grid_points=5
        )
        layer.quantize(config=config)

        # Check layer is properly configured
        self.assertEqual(layer.quantization_mode, "awq")
        self.assertTrue(hasattr(layer, "awq_scales"))

    def test_einsum_dense_layer_quantize_awq(self):
        """Test EinsumDense layer can be quantized with AWQ."""
        layer = layers.EinsumDense("ab,bc->ac", output_shape=(64,))
        layer.build(input_shape=(None, 32))

        config = AWQConfig(
            dataset=None, tokenizer=None, group_size=-1, num_grid_points=5
        )
        layer.quantize(config=config)

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
            model.quantize(config=config)


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
    length = max(1, len(long_text) - sequence_length)
    for _ in range(num_samples):
        start = RNG.integers(0, length) if length > 1 else 0
        yield long_text[start : start + sequence_length]


@pytest.mark.requires_trainable_backend
class AWQAccuracyTest(testing.TestCase):
    """End-to-end accuracy preservation tests for AWQ quantization."""

    @parameterized.named_parameters(
        ("per_channel", -1, 20, 0.5, 0.30),
        ("group_16", 16, 10, 0.4, 0.40),
    )
    def test_awq_transformer_accuracy(
        self, group_size, num_grid_points, min_top1, max_kl
    ):
        """Test that AWQ quantization preserves model accuracy.

        This test:
        1. Creates a transformer-based sequence classifier
        2. Gets baseline (full precision) predictions
        3. Applies AWQ quantization with calibration data
        4. Compares quantized predictions against baseline
        5. Validates top-1 match rate and KL divergence bounds
        """
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
            calibration_set[RNG.integers(0, len(calibration_set))]
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
            group_size=group_size,
            num_grid_points=num_grid_points,
            quantization_layer_structure=layer_structure,
        )

        # Quantize model with AWQ
        model.quantize(config=awq_config)

        # Get post-quantization predictions
        y_q = model.predict(x_eval)

        # Calculate accuracy metrics
        top1_match = _top1_match_rate(y_ref, y_q)

        p_ref = ops.softmax(y_ref)
        p_q = ops.softmax(y_q)
        kl = _mean_kl(p_ref, p_q)

        # Validate accuracy preservation
        self.assertGreaterEqual(
            float(top1_match),
            min_top1,
            f"Top-1 agreement too low for group_size={group_size}: "
            f"{float(top1_match):.3f}",
        )
        self.assertLessEqual(
            float(kl),
            max_kl,
            f"KL divergence too high for group_size={group_size}: "
            f"{float(kl):.3f}",
        )

    @parameterized.named_parameters(
        ("per_channel", -1, 0.35),
        ("group_16", 16, 0.35),
        ("group_32", 32, 0.35),
        ("group_64", 64, 0.35),
        ("group_128", 128, 0.35),
    )
    def test_awq_accuracy_various_group_sizes(
        self, group_size, max_relative_mse
    ):
        """Test AWQ accuracy across various group sizes.

        Verifies that quantizing a single layer maintains reasonable
        output reconstruction error and correct variable shapes.
        """
        in_features = 128
        out_features = 64

        keras.utils.set_random_seed(42)

        # Create fresh layer for each test
        layer = layers.Dense(out_features)
        layer.build(input_shape=(None, in_features))

        # Create data
        calibration_data = RNG.standard_normal((64, in_features)).astype(
            "float32"
        )
        test_data = RNG.standard_normal((16, in_features)).astype("float32")

        # Get original output
        original_output = layer(test_data)

        # Configure and quantize
        config = AWQConfig(
            dataset=None,
            tokenizer=None,
            group_size=group_size,
            num_grid_points=5,
        )
        layer.quantize(config=config)

        awq_obj = AWQ(layer, config)
        awq_obj.update_activation_magnitudes(calibration_data)
        awq_obj.quantize_layer()

        # Verify layer variables have correct shapes for grouped quantization
        if group_size > 0:
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

        # Verify output
        quantized_output = layer(test_data)

        # Should have no NaN/Inf
        self.assertFalse(
            ops.any(ops.isnan(quantized_output)),
            f"NaN in output for group_size={group_size}",
        )
        self.assertFalse(
            ops.any(ops.isinf(quantized_output)),
            f"Inf in output for group_size={group_size}",
        )

        # Should maintain reasonable accuracy
        mse = ops.mean(
            ops.power(ops.subtract(original_output, quantized_output), 2)
        )
        original_var = ops.var(original_output)
        relative_mse = ops.divide(mse, ops.add(original_var, 1e-8))

        self.assertLess(
            relative_mse,
            max_relative_mse,
            f"Accuracy too low for group_size={group_size}: "
            f"relative_mse={relative_mse:.4f}",
        )

        awq_obj.free()
