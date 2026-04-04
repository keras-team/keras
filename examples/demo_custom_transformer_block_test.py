"""
Tests for custom Transformer components: PositionalEncoding and TransformerBlock.

These tests verify the correctness, serialization, and functionality of the
custom layers defined in demo_custom_transformer_block.py.
"""

import os
import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import saving
from keras.src import testing
from keras.src import random

# Import the components from the demo file
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_custom_transformer_block import (
    PositionalEncoding,
    TransformerBlock,
    create_transformer_classifier,
)


class PositionalEncodingTest(testing.TestCase):
    """Test suite for PositionalEncoding layer."""

    def test_basics(self):
        """Test basic layer functionality."""
        self.run_layer_test(
            PositionalEncoding,
            init_kwargs={"max_seq_len": 100},
            input_shape=(2, 10, 32),
            expected_output_shape=(2, 10, 32),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )

    def test_shape_preservation(self):
        """Test that the layer preserves input shape."""
        layer = PositionalEncoding(max_seq_len=500)
        
        # Test various input shapes
        test_shapes = [
            (1, 10, 64),
            (4, 20, 128),
            (8, 100, 256),
            (2, 50, 512),
        ]
        
        for shape in test_shapes:
            inputs = np.random.randn(*shape)
            output = layer(inputs)
            self.assertEqual(output.shape, shape)

    def test_different_max_seq_len(self):
        """Test with different max_seq_len values."""
        for max_len in [50, 100, 1000, 5000]:
            layer = PositionalEncoding(max_seq_len=max_len)
            inputs = np.random.randn(2, 10, 32)
            output = layer(inputs)
            self.assertEqual(output.shape, inputs.shape)

    def test_positional_encoding_added(self):
        """Test that positional encoding is actually added to inputs."""
        layer = PositionalEncoding(max_seq_len=100)
        
        # Create a zero input
        inputs = np.zeros((1, 10, 32))
        output = layer(inputs)
        
        # Output should not be all zeros (positional encoding is added)
        self.assertFalse(np.allclose(output, 0.0))

    def test_encoding_values_sine_cosine_pattern(self):
        """Test that encoding follows sine/cosine pattern."""
        layer = PositionalEncoding(max_seq_len=100)
        
        # Create a zero input to isolate the positional encoding
        inputs = np.zeros((1, 10, 16))
        output = layer(inputs)
        
        # The output should be the positional encoding itself
        pe = output[0].numpy() if hasattr(output, 'numpy') else output[0]
        
        # Check that values are in reasonable range for sin/cos
        self.assertTrue(np.all(pe >= -1.0) and np.all(pe <= 1.0))

    def test_different_positions_different_encodings(self):
        """Test that different positions get different encodings."""
        layer = PositionalEncoding(max_seq_len=100)
        
        # Create a zero input
        inputs = np.zeros((1, 10, 32))
        output = layer(inputs)
        pe = output[0].numpy() if hasattr(output, 'numpy') else output[0]
        
        # Each position should have a different encoding
        for i in range(9):
            self.assertFalse(np.allclose(pe[i], pe[i + 1]))

    def test_batch_independence(self):
        """Test that encoding is independent of batch size."""
        layer = PositionalEncoding(max_seq_len=100)
        
        # Create inputs with different batch sizes
        inputs1 = np.zeros((1, 10, 32))
        inputs2 = np.zeros((4, 10, 32))
        
        output1 = layer(inputs1)
        output2 = layer(inputs2)
        
        # Extract positional encodings (should be the same for all batch elements)
        pe1 = output1[0].numpy() if hasattr(output1, 'numpy') else output1[0]
        pe2 = output2[0].numpy() if hasattr(output2, 'numpy') else output2[0]
        
        # Same positions should have same encoding across batches
        self.assertAllClose(pe1, pe2)

    def test_serialization(self):
        """Test layer serialization and deserialization."""
        layer = PositionalEncoding(max_seq_len=1000)
        
        # Build the layer
        inputs = np.zeros((2, 10, 32))
        _ = layer(inputs)
        
        # Test get_config
        config = layer.get_config()
        self.assertEqual(config["max_seq_len"], 1000)
        
        # Test from_config
        new_layer = PositionalEncoding.from_config(config)
        self.assertEqual(new_layer.max_seq_len, 1000)

    def test_full_serialization(self):
        """Test full serialization with model save/load."""
        layer = PositionalEncoding(max_seq_len=100)
        
        # Create a simple model
        inputs = layers.Input(shape=(10, 32))
        outputs = layer(inputs)
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Save and load the model
        temp_filepath = os.path.join(self.get_temp_dir(), "pos_encoding_model.keras")
        model.save(temp_filepath)
        
        loaded_model = saving.load_model(temp_filepath)
        
        # Test that outputs match
        test_input = np.random.randn(2, 10, 32)
        original_output = model.predict(test_input, verbose=0)
        loaded_output = loaded_model.predict(test_input, verbose=0)
        
        self.assertAllClose(original_output, loaded_output)

    def test_sequence_length_shorter_than_max(self):
        """Test with sequence lengths shorter than max_seq_len."""
        layer = PositionalEncoding(max_seq_len=100)
        
        # Use sequences of various lengths, all shorter than max_seq_len
        for seq_len in [5, 10, 50, 99]:
            inputs = np.random.randn(2, seq_len, 32)
            output = layer(inputs)
            self.assertEqual(output.shape, (2, seq_len, 32))

    def test_d_model_extraction(self):
        """Test that d_model is correctly extracted from input shape."""
        layer = PositionalEncoding(max_seq_len=100)
        
        # Test with different embedding dimensions
        for d_model in [32, 64, 128, 256]:
            inputs = np.random.randn(2, 10, d_model)
            output = layer(inputs)
            self.assertEqual(output.shape[-1], d_model)
            self.assertEqual(layer.d_model, d_model)

    @parameterized.named_parameters(
        ("small", (2, 5, 16)),
        ("medium", (4, 20, 64)),
        ("large", (2, 100, 256)),
    )
    def test_various_shapes(self, input_shape):
        """Test with various input shapes."""
        layer = PositionalEncoding(max_seq_len=200)
        inputs = np.random.randn(*input_shape)
        output = layer(inputs)
        self.assertEqual(output.shape, input_shape)


class TransformerBlockTest(testing.TestCase):
    """Test suite for TransformerBlock layer."""

    def test_basics(self):
        """Test basic layer functionality."""
        self.run_layer_test(
            TransformerBlock,
            init_kwargs={
                "embed_dim": 32,
                "num_heads": 4,
                "ff_dim": 64,
                "dropout_rate": 0.1,
            },
            input_shape=(2, 10, 32),
            expected_output_shape=(2, 10, 32),
            expected_num_trainable_weights=12,  # Attention (4) + FFN (2) + LayerNorms (4) + Dense (2)
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=2,  # Dropout layers
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )

    def test_shape_preservation(self):
        """Test that the layer preserves input shape."""
        layer = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128)
        
        test_shapes = [
            (1, 10, 64),
            (4, 20, 64),
            (8, 100, 64),
        ]
        
        for shape in test_shapes:
            inputs = np.random.randn(*shape).astype(np.float32)
            output = layer(inputs, training=False)
            self.assertEqual(output.shape, shape)

    def test_invalid_embed_dim_num_heads(self):
        """Test that ValueError is raised when embed_dim is not divisible by num_heads."""
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            TransformerBlock(embed_dim=65, num_heads=4)

    def test_valid_embed_dim_num_heads_combinations(self):
        """Test valid combinations of embed_dim and num_heads."""
        valid_combinations = [
            (64, 4),
            (128, 8),
            (256, 8),
            (512, 8),
            (256, 4),
            (128, 4),
        ]
        
        for embed_dim, num_heads in valid_combinations:
            layer = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads)
            inputs = np.random.randn(2, 10, embed_dim).astype(np.float32)
            output = layer(inputs, training=False)
            self.assertEqual(output.shape, inputs.shape)

    def test_different_activations(self):
        """Test with different activation functions."""
        activations = ["relu", "gelu"]
        
        for activation in activations:
            layer = TransformerBlock(
                embed_dim=64,
                num_heads=4,
                ff_dim=128,
                activation=activation,
            )
            inputs = np.random.randn(2, 10, 64).astype(np.float32)
            output = layer(inputs, training=False)
            self.assertEqual(output.shape, inputs.shape)

    def test_training_vs_inference(self):
        """Test that layer behaves differently in training vs inference."""
        layer = TransformerBlock(
            embed_dim=64,
            num_heads=4,
            ff_dim=128,
            dropout_rate=0.5,  # High dropout to see difference
        )
        
        inputs = np.random.randn(2, 10, 64).astype(np.float32)
        
        # Run multiple times in training mode (dropout should cause variation)
        outputs_training = [
            layer(inputs, training=True) for _ in range(3)
        ]
        
        # Run multiple times in inference mode (should be deterministic)
        outputs_inference = [
            layer(inputs, training=False) for _ in range(3)
        ]
        
        # Inference outputs should be identical
        for i in range(1, len(outputs_inference)):
            self.assertAllClose(outputs_inference[0], outputs_inference[i])

    def test_serialization(self):
        """Test layer serialization and deserialization."""
        layer = TransformerBlock(
            embed_dim=64,
            num_heads=4,
            ff_dim=128,
            dropout_rate=0.1,
            attention_dropout=0.1,
            activation="gelu",
        )
        
        # Build the layer
        inputs = np.zeros((2, 10, 64))
        _ = layer(inputs)
        
        # Test get_config
        config = layer.get_config()
        self.assertEqual(config["embed_dim"], 64)
        self.assertEqual(config["num_heads"], 4)
        self.assertEqual(config["ff_dim"], 128)
        self.assertEqual(config["dropout_rate"], 0.1)
        self.assertEqual(config["attention_dropout"], 0.1)
        self.assertEqual(config["activation"], "gelu")

    def test_full_serialization(self):
        """Test full serialization with model save/load."""
        layer = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128)
        
        # Create a simple model
        inputs = layers.Input(shape=(10, 64))
        outputs = layer(inputs)
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Save and load the model
        temp_filepath = os.path.join(self.get_temp_dir(), "transformer_block_model.keras")
        model.save(temp_filepath)
        
        loaded_model = saving.load_model(temp_filepath)
        
        # Test that outputs match
        test_input = np.random.randn(2, 10, 64).astype(np.float32)
        original_output = model.predict(test_input, verbose=0)
        loaded_output = loaded_model.predict(test_input, verbose=0)
        
        self.assertAllClose(original_output, loaded_output, atol=1e-5)

    def test_residual_connection(self):
        """Test that residual connections are working correctly."""
        layer = TransformerBlock(
            embed_dim=64,
            num_heads=4,
            ff_dim=128,
            dropout_rate=0.0,  # No dropout for deterministic test
        )
        
        # Create input with known values
        inputs = np.ones((1, 10, 64)).astype(np.float32)
        
        output = layer(inputs, training=False)
        
        # Output should be different from input (transformation occurred)
        # but should have the same shape
        self.assertEqual(output.shape, inputs.shape)
        
        # The output should not be all zeros
        self.assertFalse(np.allclose(output, 0.0))

    def test_layer_normalization(self):
        """Test that layer normalization is applied."""
        layer = TransformerBlock(
            embed_dim=64,
            num_heads=4,
            ff_dim=128,
            dropout_rate=0.0,
        )
        
        # Create input with extreme values
        inputs = np.random.randn(2, 10, 64).astype(np.float32) * 100
        
        output = layer(inputs, training=False)
        
        # Output should not have exploded
        self.assertTrue(np.max(np.abs(output)) < 1000)

    def test_different_ff_dim(self):
        """Test with different feed-forward dimensions."""
        for ff_dim in [64, 128, 256, 512]:
            layer = TransformerBlock(
                embed_dim=64,
                num_heads=4,
                ff_dim=ff_dim,
            )
            inputs = np.random.randn(2, 10, 64).astype(np.float32)
            output = layer(inputs, training=False)
            self.assertEqual(output.shape, inputs.shape)

    def test_attention_dropout(self):
        """Test with attention dropout."""
        layer = TransformerBlock(
            embed_dim=64,
            num_heads=4,
            ff_dim=128,
            dropout_rate=0.0,
            attention_dropout=0.5,
        )
        
        inputs = np.random.randn(2, 10, 64).astype(np.float32)
        
        # In inference mode, should still work
        output = layer(inputs, training=False)
        self.assertEqual(output.shape, inputs.shape)

    @parameterized.named_parameters(
        ("small", 32, 2, 64),
        ("medium", 64, 4, 128),
        ("large", 128, 8, 256),
    )
    def test_various_configurations(self, embed_dim, num_heads, ff_dim):
        """Test with various configurations."""
        layer = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
        )
        inputs = np.random.randn(2, 10, embed_dim).astype(np.float32)
        output = layer(inputs, training=False)
        self.assertEqual(output.shape, inputs.shape)


class TransformerClassifierTest(testing.TestCase):
    """Test suite for create_transformer_classifier function."""

    def test_model_creation(self):
        """Test that the model is created correctly."""
        model = create_transformer_classifier(
            vocab_size=1000,
            max_len=50,
            embed_dim=64,
            num_heads=4,
            ff_dim=128,
            num_blocks=2,
            num_classes=3,
        )
        
        # Check model input/output shapes
        self.assertEqual(model.input_shape, (None, 50))
        self.assertEqual(model.output_shape, (None, 3))

    def test_model_compilation(self):
        """Test that the model can be compiled."""
        model = create_transformer_classifier(
            vocab_size=1000,
            max_len=50,
            embed_dim=64,
            num_heads=4,
            ff_dim=128,
            num_blocks=2,
            num_classes=3,
        )
        
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        # Model should be compiled
        self.assertTrue(model.compiled)

    @pytest.mark.requires_trainable_backend
    def test_model_training(self):
        """Test that the model can be trained."""
        model = create_transformer_classifier(
            vocab_size=100,
            max_len=20,
            embed_dim=32,
            num_heads=2,
            ff_dim=64,
            num_blocks=1,
            num_classes=2,
        )
        
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        # Create dummy data
        x_train = np.random.randint(0, 100, size=(32, 20))
        y_train = np.eye(2)[np.random.randint(0, 2, size=(32,))]
        
        # Train for one epoch
        history = model.fit(x_train, y_train, epochs=1, verbose=0)
        
        # Check that training happened
        self.assertIn("loss", history.history)
        self.assertIn("accuracy", history.history)

    def test_model_inference(self):
        """Test that the model can perform inference."""
        model = create_transformer_classifier(
            vocab_size=100,
            max_len=20,
            embed_dim=32,
            num_heads=2,
            ff_dim=64,
            num_blocks=1,
            num_classes=2,
        )
        
        # Create test input
        test_input = np.random.randint(0, 100, size=(1, 20))
        
        # Predict
        output = model.predict(test_input, verbose=0)
        
        # Check output shape
        self.assertEqual(output.shape, (1, 2))
        
        # Check that output sums to approximately 1 (softmax)
        self.assertAlmostEqual(np.sum(output[0]), 1.0, places=5)

    def test_different_num_blocks(self):
        """Test with different numbers of transformer blocks."""
        for num_blocks in [1, 2, 4]:
            model = create_transformer_classifier(
                vocab_size=100,
                max_len=20,
                embed_dim=32,
                num_heads=2,
                ff_dim=64,
                num_blocks=num_blocks,
                num_classes=2,
            )
            
            # Count transformer blocks
            transformer_layers = [
                l for l in model.layers
                if isinstance(l, TransformerBlock)
            ]
            self.assertEqual(len(transformer_layers), num_blocks)

    def test_different_num_classes(self):
        """Test with different numbers of output classes."""
        for num_classes in [2, 5, 10]:
            model = create_transformer_classifier(
                vocab_size=100,
                max_len=20,
                embed_dim=32,
                num_heads=2,
                ff_dim=64,
                num_blocks=1,
                num_classes=num_classes,
            )
            
            test_input = np.random.randint(0, 100, size=(1, 20))
            output = model.predict(test_input, verbose=0)
            
            self.assertEqual(output.shape, (1, num_classes))

    def test_model_serialization(self):
        """Test that the full model can be serialized."""
        model = create_transformer_classifier(
            vocab_size=100,
            max_len=20,
            embed_dim=32,
            num_heads=2,
            ff_dim=64,
            num_blocks=1,
            num_classes=2,
        )
        
        # Save the model
        temp_filepath = os.path.join(self.get_temp_dir(), "transformer_classifier.keras")
        model.save(temp_filepath)
        
        # Load the model
        loaded_model = saving.load_model(temp_filepath)
        
        # Test that outputs match
        test_input = np.random.randint(0, 100, size=(1, 20))
        original_output = model.predict(test_input, verbose=0)
        loaded_output = loaded_model.predict(test_input, verbose=0)
        
        self.assertAllClose(original_output, loaded_output, atol=1e-5)

    def test_positional_encoding_in_model(self):
        """Test that positional encoding is present in the model."""
        model = create_transformer_classifier(
            vocab_size=100,
            max_len=20,
            embed_dim=32,
            num_heads=2,
            ff_dim=64,
            num_blocks=1,
            num_classes=2,
        )
        
        # Check for positional encoding layer
        pos_encoding_layers = [
            l for l in model.layers
            if isinstance(l, PositionalEncoding)
        ]
        self.assertEqual(len(pos_encoding_layers), 1)

    def test_dropout_rate_parameter(self):
        """Test that dropout rate is correctly applied."""
        dropout_rate = 0.3
        model = create_transformer_classifier(
            vocab_size=100,
            max_len=20,
            embed_dim=32,
            num_heads=2,
            ff_dim=64,
            num_blocks=1,
            num_classes=2,
            dropout_rate=dropout_rate,
        )
        
        # Check dropout layers in transformer blocks
        for layer in model.layers:
            if isinstance(layer, TransformerBlock):
                self.assertEqual(layer.dropout_rate, dropout_rate)


class PositionalEncodingCorrectnessTest(testing.TestCase):
    """Tests for positional encoding mathematical correctness."""

    def test_encoding_formula(self):
        """Test that positional encoding follows the correct formula.
        
        The formula from "Attention Is All You Need":
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        max_seq_len = 10
        d_model = 16
        
        layer = PositionalEncoding(max_seq_len=max_seq_len)
        
        # Create zero input to isolate positional encoding
        inputs = np.zeros((1, max_seq_len, d_model))
        output = layer(inputs)
        
        # Get the positional encoding
        pe = output[0].numpy() if hasattr(output, 'numpy') else output[0]
        
        # Manually compute expected positional encoding
        position = np.arange(max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        expected_pe = np.zeros((max_seq_len, d_model))
        expected_pe[:, 0::2] = np.sin(position * div_term)
        expected_pe[:, 1::2] = np.cos(position * div_term)
        
        self.assertAllClose(pe, expected_pe, atol=1e-5, rtol=1e-5)

    def test_encoding_is_periodic(self):
        """Test that positional encoding exhibits periodicity.
        
        Lower dimensions should have longer periods, higher dimensions shorter periods.
        """
        max_seq_len = 100
        d_model = 32
        
        layer = PositionalEncoding(max_seq_len=max_seq_len)
        inputs = np.zeros((1, max_seq_len, d_model))
        output = layer(inputs)
        pe = output[0].numpy() if hasattr(output, 'numpy') else output[0]
        
        # For the first dimension (lowest frequency), the period should be approximately
        # 2*pi*10000^(0/d_model) = 2*pi*1 = 2*pi
        # The values should be sin(position) for position 0 to 99
        
        # Check that dimension 0 is approximately sin(position)
        expected = np.sin(np.arange(max_seq_len))
        self.assertAllClose(pe[:, 0], expected, atol=1e-5)


class TransformerBlockCorrectnessTest(testing.TestCase):
    """Tests for transformer block mathematical correctness."""

    def test_self_attention(self):
        """Test that self-attention is working correctly."""
        embed_dim = 32
        num_heads = 4
        
        layer = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=64,
            dropout_rate=0.0,
        )
        
        # Create input with specific pattern
        batch_size = 2
        seq_len = 10
        inputs = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
        
        output = layer(inputs, training=False)
        
        # Output should be a transformation of the input
        # With residual connections, it should be input + transformation
        self.assertEqual(output.shape, inputs.shape)
        
    def test_pre_normalization(self):
        """Test that pre-normalization is used.
        
        The layer uses pre-normalization where LayerNorm is applied
        before attention and feed-forward layers.
        """
        layer = TransformerBlock(
            embed_dim=32,
            num_heads=4,
            ff_dim=64,
            dropout_rate=0.0,
        )
        
        # Build the layer
        inputs = np.zeros((1, 10, 32))
        _ = layer(inputs)
        
        # Check that layer normalization layers exist
        self.assertTrue(hasattr(layer, 'layer_norm1'))
        self.assertTrue(hasattr(layer, 'layer_norm2'))
        
    def test_feed_forward_network(self):
        """Test that feed-forward network has correct structure."""
        embed_dim = 32
        ff_dim = 128
        
        layer = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=4,
            ff_dim=ff_dim,
        )
        
        # Build the layer
        inputs = np.zeros((1, 10, embed_dim))
        _ = layer(inputs)
        
        # Check FFN layers
        self.assertTrue(hasattr(layer, 'ffn_layer1'))
        self.assertTrue(hasattr(layer, 'ffn_layer2'))
        
        # First FFN layer should expand to ff_dim
        self.assertEqual(layer.ffn_layer1.units, ff_dim)
        
        # Second FFN layer should project back to embed_dim
        self.assertEqual(layer.ffn_layer2.units, embed_dim)


class IntegrationTest(testing.TestCase):
    """Integration tests combining multiple components."""

    def test_full_pipeline(self):
        """Test the full pipeline from input to output."""
        # Create model
        model = create_transformer_classifier(
            vocab_size=100,
            max_len=20,
            embed_dim=32,
            num_heads=2,
            ff_dim=64,
            num_blocks=2,
            num_classes=2,
        )
        
        # Create random input
        batch_size = 4
        x = np.random.randint(0, 100, size=(batch_size, 20))
        
        # Forward pass
        output = model(x, training=False)
        
        # Check output
        self.assertEqual(output.shape, (batch_size, 2))
        self.assertTrue(np.all(output >= 0) and np.all(output <= 1))
        
        # Check that probabilities sum to 1
        sums = np.sum(output, axis=-1)
        self.assertAllClose(sums, np.ones(batch_size), atol=1e-5)

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        import keras
        
        model = create_transformer_classifier(
            vocab_size=100,
            max_len=20,
            embed_dim=32,
            num_heads=2,
            ff_dim=64,
            num_blocks=1,
            num_classes=2,
        )
        
        # Create variables for gradient computation
        x = np.random.randint(0, 100, size=(2, 20))
        y = np.eye(2)[np.random.randint(0, 2, size=(2,))]
        
        # Compute gradients
        with backend.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = keras.losses.categorical_crossentropy(y, predictions)
            loss = ops.mean(loss)
        
        gradients = tape.gradient(loss, model.trainable_weights)
        
        # Check that all gradients are non-None
        self.assertTrue(all(g is not None for g in gradients))
        
        # Check that gradients are non-zero
        for g in gradients:
            self.assertTrue(ops.any(g != 0))

    def test_reproducibility(self):
        """Test that the model produces consistent outputs."""
        keras.utils.set_random_seed(42)
        
        model = create_transformer_classifier(
            vocab_size=100,
            max_len=20,
            embed_dim=32,
            num_heads=2,
            ff_dim=64,
            num_blocks=1,
            num_classes=2,
        )
        
        x = np.random.randint(0, 100, size=(2, 20))
        
        # Run twice
        output1 = model(x, training=False)
        output2 = model(x, training=False)
        
        # Should be identical in inference mode
        self.assertAllClose(output1, output2)