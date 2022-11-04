"""Machine translation model.

Adapted from
https://keras.io/examples/nlp/neural_machine_translation_with_transformer/
"""
import tensorflow as tf
from tensorflow import keras

from keras.integration_test.models.input_spec import InputSpec

VOCAB_SIZE = 1500
SEQUENCE_LENGTH = 20


def get_data_spec(batch_size):
    return (
        (
            InputSpec((batch_size,), dtype="string"),
            InputSpec((batch_size,), dtype="string"),
        ),
        InputSpec(
            (batch_size, SEQUENCE_LENGTH), dtype="int64", range=[0, VOCAB_SIZE]
        ),
    )


def get_input_preprocessor():
    encoder_input_vectorizer = keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
    )
    decoder_input_vectorizer = keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
    )
    text_ds = tf.data.Dataset.from_tensor_slices(
        [
            "Lorem ipsum dolor sit amet",
            "consectetur adipiscing elit",
            "sed do eiusmod tempor incididunt ut",
            "labore et dolore magna aliqua.",
            "Ut enim ad minim veniam",
            "quis nostrud exercitation ullamco",
            "laboris nisi ut aliquip ex ea commodo consequat.",
        ]
    )
    encoder_input_vectorizer.adapt(text_ds)
    decoder_input_vectorizer.adapt(text_ds)
    return lambda x: (
        encoder_input_vectorizer(x[0]),
        decoder_input_vectorizer(x[1]),
    )


class TransformerEncoder(keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                keras.layers.Dense(dense_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(
                mask[:, tf.newaxis, tf.newaxis, :], dtype="int32"
            )
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = keras.layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoder(keras.layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                keras.layers.Dense(latent_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()
        self.layernorm_3 = keras.layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [
                tf.expand_dims(batch_size, -1),
                tf.constant([1, 1], dtype=tf.int32),
            ],
            axis=0,
        )
        return tf.tile(mask, mult)


def get_model(
    build=False, compile=False, jit_compile=False, include_preprocessing=True
):
    embed_dim = 256
    latent_dim = 256
    num_heads = 2

    if include_preprocessing:
        encoder_inputs = keras.Input(shape=(), dtype="string")
        decoder_inputs = keras.Input(shape=(), dtype="string")
        encoder_x, decoder_x = get_input_preprocessor()(
            (encoder_inputs, decoder_inputs)
        )
    else:
        encoder_inputs = keras.Input(shape=(None,), dtype="int64")
        decoder_inputs = keras.Input(shape=(None,), dtype="int64")
        encoder_x = encoder_inputs
        decoder_x = decoder_inputs

    x = PositionalEmbedding(SEQUENCE_LENGTH, VOCAB_SIZE, embed_dim)(encoder_x)
    encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)

    encoded_seq_inputs = keras.Input(shape=(None, embed_dim))
    x = PositionalEmbedding(SEQUENCE_LENGTH, VOCAB_SIZE, embed_dim)(decoder_x)
    x = TransformerDecoder(embed_dim, latent_dim, num_heads)(
        x, encoded_seq_inputs
    )
    x = keras.layers.Dropout(0.5)(x)
    decoder_outputs = keras.layers.Dense(VOCAB_SIZE, activation="softmax")(x)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    model = keras.Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
    )
    if compile:
        model.compile(
            "rmsprop",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            jit_compile=jit_compile,
        )
    return model


def get_custom_objects():
    return {
        "TransformerEncoder": TransformerEncoder,
        "TransformerDecoder": TransformerDecoder,
        "PositionalEmbedding": PositionalEmbedding,
    }
