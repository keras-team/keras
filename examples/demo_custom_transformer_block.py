"""
Title: Custom Transformer Block with Positional Encoding
Author: [Your Name]
Date created: 2024/03/26
Last modified: 2024/03/26
Description: Creating custom positional encoding and transformer block layers
    from scratch using Keras 3 backend-agnostic operations.
"""

"""
## Introduction

This example demonstrates how to create custom Keras layers that:

1. **Positional Encoding**: Add positional information to sequences - a crucial
   component for transformer models that has no built-in Keras equivalent.

2. **Transformer Block**: A composite layer combining self-attention,
   feed-forward networks, and residual connections.

Both layers are implemented using backend-agnostic Keras operations (`keras.ops`),
making them compatible with TensorFlow, JAX, and PyTorch backends.

## Background

Unlike RNNs that process sequences step-by-step, transformers process all positions
simultaneously. To give the model a sense of token order, we inject positional
information using sinusoidal functions at different frequencies.

The transformer block then uses self-attention to model relationships between
all positions, followed by position-wise feed-forward transformations.
"""

import numpy as np

import keras
from keras import layers
from keras import ops
from keras import initializers


# =============================================================================
# Positional Encoding Layer
# =============================================================================


class PositionalEncoding(layers.Layer):
    """Adds positional information to input sequences using sinusoidal functions.

    This layer implements the positional encoding from "Attention Is All You Need"
    (Vaswani et al., 2017). It creates a fixed encoding using sine and cosine
    functions at different frequencies, allowing the model to learn relative
    positions.

    Args:
        max_seq_len: Maximum sequence length. The positional encoding will be
            precomputed up to this length. Defaults to 5000.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Call arguments:
        inputs: A tensor of shape `(batch_size, seq_len, d_model)`.

    Returns:
        Tensor of same shape as input with positional encoding added:
        `(batch_size, seq_len, d_model)`.

    Example:
        ```python
        # Create a positional encoding layer
        pos_encoding = PositionalEncoding(max_seq_len=1000)

        # Input: batch of 32 sequences, each with 100 tokens, 512-dim embeddings
        x = np.random.randn(32, 100, 512)
        output = pos_encoding(x)  # Shape: (32, 100, 512)
        ```
    """

    def __init__(self, max_seq_len=5000, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len

    def build(self, input_shape):
        # Get the model dimension from input shape
        d_model = input_shape[-1]

        # Create position indices: (max_seq_len, 1)
        position = ops.arange(0, self.max_seq_len, dtype="float32")
        position = ops.expand_dims(position, axis=1)

        # Create dimension indices for computing frequencies: (d_model // 2,)
        # We use exp(-log(10000) * 2i / d_model) = 10000^(-2i/d_model)
        div_term = ops.exp(
            ops.arange(0, d_model, 2, dtype="float32")
            * (-ops.log(10000.0) / d_model)
        )

        # Compute positional encodings
        # pe[i, 2j] = sin(i / 10000^(2j/d_model))
        # pe[i, 2j+1] = cos(i / 10000^(2j/d_model))
        pe = ops.zeros((self.max_seq_len, d_model))

        # Use scatter_update to fill in sine values for even indices
        # and cosine values for odd indices
        # Note: We need to handle this carefully for backend compatibility
        pe_even = ops.sin(position * div_term)  # Shape: (max_seq_len, d_model//2)
        pe_odd = ops.cos(position * div_term)   # Shape: (max_seq_len, d_model//2)

        # Interleave even and odd: [sin, cos, sin, cos, ...]
        pe = ops.reshape(
            ops.stack([pe_even, pe_odd], axis=2),
            (self.max_seq_len, d_model)
        )

        # Add batch dimension for broadcasting: (1, max_seq_len, d_model)
        self.pe = ops.expand_dims(pe, axis=0)

        # Store d_model for serialization
        self.d_model = d_model
        self.built = True

    def call(self, inputs):
        # Get sequence length from inputs
        seq_len = ops.shape(inputs)[1]

        # Add positional encoding (broadcast over batch dimension)
        return inputs + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_seq_len": self.max_seq_len})
        return config


# =============================================================================
# Transformer Block Layer
# =============================================================================


class TransformerBlock(layers.Layer):
    """A single transformer block with self-attention and feed-forward network.

    This layer implements a standard transformer block consisting of:
    1. Multi-head self-attention mechanism
    2. Layer normalization
    3. Feed-forward network (two dense layers with ReLU/GELU)
    4. Residual connections around each sub-layer

    Implements the architecture from "Attention Is All You Need"
    (Vaswani et al., 2017) with pre-normalization (more stable training).

    Args:
        embed_dim: The dimension of the input and output embeddings.
        num_heads: Number of attention heads. Must divide embed_dim evenly.
        ff_dim: Hidden dimension of the feed-forward network. Defaults to 2048.
        dropout_rate: Dropout rate applied after attention and feed-forward.
            Defaults to 0.1.
        attention_dropout: Dropout rate within the attention mechanism.
            Defaults to 0.0.
        activation: Activation function for the feed-forward network.
            Can be "relu", "gelu", or a callable. Defaults to "relu".
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Call arguments:
        inputs: Tensor of shape `(batch_size, seq_len, embed_dim)`.
        training: Boolean indicating whether in training mode (affects dropout).
            Defaults to `None`.

    Returns:
        Tensor of shape `(batch_size, seq_len, embed_dim)`.

    Example:
        ```python
        # Create a transformer block
        transformer = TransformerBlock(
            embed_dim=512,
            num_heads=8,
            ff_dim=2048,
            dropout_rate=0.1
        )

        # Process a sequence
        x = np.random.randn(32, 100, 512)
        output = transformer(x, training=True)  # Shape: (32, 100, 512)
        ```
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim=2048,
        dropout_rate=0.1,
        attention_dropout=0.0,
        activation="relu",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.activation = activation

        # Validate that embed_dim is divisible by num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

    def build(self, input_shape):
        # Layer normalization for pre-normalization
        self.layer_norm1 = layers.LayerNormalization(
            epsilon=1e-6, name="layer_norm_1"
        )
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=1e-6, name="layer_norm_2"
        )

        # Multi-head self-attention
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.attention_dropout,
            name="self_attention"
        )

        # Feed-forward network
        self.ffn_layer1 = layers.Dense(
            self.ff_dim,
            activation=self.activation,
            name="ffn_dense_1"
        )
        self.ffn_layer2 = layers.Dense(
            self.embed_dim,
            name="ffn_dense_2"
        )

        # Dropout layers
        self.dropout1 = layers.Dropout(self.dropout_rate, name="dropout_1")
        self.dropout2 = layers.Dropout(self.dropout_rate, name="dropout_2")

        self.built = True

    def call(self, inputs, training=None):
        # Pre-normalization architecture (more stable than post-norm)

        # Self-attention sublayer with residual connection
        # Shape: (batch_size, seq_len, embed_dim)
        normed = self.layer_norm1(inputs)
        
        # Self-attention: query, key, value all come from the same input
        attention_output = self.attention(
            query=normed,
            value=normed,
            key=normed,
            training=training
        )
        attention_output = self.dropout1(attention_output, training=training)
        
        # First residual connection
        x = inputs + attention_output

        # Feed-forward sublayer with residual connection
        normed = self.layer_norm2(x)
        ffn_output = self.ffn_layer1(normed)
        ffn_output = self.ffn_layer2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)

        # Second residual connection
        return x + ffn_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
            "attention_dropout": self.attention_dropout,
            "activation": self.activation,
        })
        return config


# =============================================================================
# Example Usage: Building a Simple Transformer Classifier
# =============================================================================


def create_transformer_classifier(
    vocab_size=10000,
    max_len=100,
    embed_dim=128,
    num_heads=4,
    ff_dim=512,
    num_blocks=2,
    num_classes=2,
    dropout_rate=0.1
):
    """Creates a simple transformer-based text classifier.

    This demonstrates how to combine PositionalEncoding and TransformerBlock
    layers into a complete model for sequence classification.

    Args:
        vocab_size: Size of the vocabulary for the embedding layer.
        max_len: Maximum sequence length.
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        ff_dim: Feed-forward network hidden dimension.
        num_blocks: Number of transformer blocks to stack.
        num_classes: Number of output classes.
        dropout_rate: Dropout rate.

    Returns:
        A compiled Keras Model.
    """
    inputs = layers.Input(shape=(max_len,), name="input_ids")
    
    # Token embeddings
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        name="token_embedding"
    )(inputs)
    
    # Add positional encoding
    x = PositionalEncoding(max_seq_len=max_len, name="positional_encoding")(x)
    
    # Apply dropout to embeddings
    x = layers.Dropout(dropout_rate, name="embedding_dropout")(x)
    
    # Stack transformer blocks
    for i in range(num_blocks):
        x = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name=f"transformer_block_{i+1}"
        )(x)
    
    # Global average pooling over sequence dimension
    x = layers.GlobalAveragePooling1D(name="global_pooling")(x)
    
    # Classification head
    x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="transformer_classifier")
    return model


# =============================================================================
# Demo: Training the Model
# =============================================================================

if __name__ == "__main__":
    # Configuration
    VOCAB_SIZE = 1000
    MAX_LEN = 50
    EMBED_DIM = 64
    NUM_HEADS = 4
    FF_DIM = 128
    NUM_BLOCKS = 2
    NUM_CLASSES = 3
    BATCH_SIZE = 32
    NUM_SAMPLES = 1000
    EPOCHS = 3

    print("=" * 60)
    print("Custom Transformer Block Demo")
    print("=" * 60)

    # Generate random classification data
    # In practice, this would be real text data tokenized into integers
    print("\n1. Generating synthetic data...")
    x_train = np.random.randint(0, VOCAB_SIZE, size=(NUM_SAMPLES, MAX_LEN))
    y_train = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES,))
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)

    print(f"   Training data shape: {x_train.shape}")
    print(f"   Labels shape: {y_train.shape}")

    # Create the model
    print("\n2. Building transformer classifier...")
    model = create_transformer_classifier(
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_blocks=NUM_BLOCKS,
        num_classes=NUM_CLASSES,
        dropout_rate=0.1
    )

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # Print model summary
    print("\n3. Model architecture:")
    model.summary()

    # Train the model
    print(f"\n4. Training for {EPOCHS} epochs...")
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate
    print("\n5. Final metrics:")
    print(f"   Training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"   Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

    # Test inference
    print("\n6. Testing inference on sample input...")
    sample_input = np.random.randint(0, VOCAB_SIZE, size=(1, MAX_LEN))
    prediction = model.predict(sample_input, verbose=0)
    print(f"   Input shape: {sample_input.shape}")
    print(f"   Prediction shape: {prediction.shape}")
    print(f"   Predicted class: {np.argmax(prediction, axis=1)[0]}")
    print(f"   Class probabilities: {prediction[0]}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

    # Demonstrate layer reuse
    print("\n7. Demonstrating standalone layer usage:")
    
    # Create a standalone positional encoding layer
    pos_enc = PositionalEncoding(max_seq_len=100)
    
    # Test with random embeddings
    test_embeddings = np.random.randn(4, 20, 64)  # (batch, seq, dim)
    encoded = pos_enc(test_embeddings)
    print(f"   Input shape: {test_embeddings.shape}")
    print(f"   After positional encoding: {encoded.shape}")
    
    # Create a standalone transformer block
    trans_block = TransformerBlock(
        embed_dim=64,
        num_heads=4,
        ff_dim=128
    )
    
    # Process through transformer block
    transformed = trans_block(encoded, training=False)
    print(f"   After transformer block: {transformed.shape}")

    print("\n✓ All layers work correctly with backend-agnostic operations!")