"""Bert model.

Adapted from https://keras.io/examples/nlp/masked_language_modeling/
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.integration_test.models.input_spec import InputSpec

SEQUENCE_LENGTH = 16
VOCAB_SIZE = 1000
EMBED_DIM = 64
NUM_HEAD = 2
FF_DIM = 32
NUM_LAYERS = 2


def get_data_spec(batch_size):
    return (
        InputSpec((batch_size,), dtype="string"),
        InputSpec((batch_size, SEQUENCE_LENGTH, VOCAB_SIZE)),
    )


def get_input_preprocessor():
    input_vectorizer = keras.layers.TextVectorization(
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
    input_vectorizer.adapt(text_ds)
    return input_vectorizer


def bert_module(query, key, value, i):
    attention_output = keras.layers.MultiHeadAttention(
        num_heads=NUM_HEAD,
        key_dim=EMBED_DIM // NUM_HEAD,
    )(query, key, value)
    attention_output = keras.layers.Dropout(0.1)(attention_output)
    attention_output = keras.layers.LayerNormalization(epsilon=1e-6)(
        query + attention_output
    )

    ffn = keras.Sequential(
        [
            keras.layers.Dense(FF_DIM, activation="relu"),
            keras.layers.Dense(EMBED_DIM),
        ],
    )
    ffn_output = ffn(attention_output)
    ffn_output = keras.layers.Dropout(0.1)(ffn_output)
    sequence_output = keras.layers.LayerNormalization(epsilon=1e-6)(
        attention_output + ffn_output
    )
    return sequence_output


def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])
    return pos_enc


loss_fn = keras.losses.CategoricalCrossentropy()
loss_tracker = keras.metrics.Mean(name="loss")


class MaskedLanguageModel(keras.Model):
    def train_step(self, inputs):
        if len(inputs) == 3:
            features, labels, sample_weight = inputs
        else:
            features, labels = inputs
            sample_weight = None

        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            loss = loss_fn(labels, predictions, sample_weight=sample_weight)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        loss_tracker.update_state(loss, sample_weight=sample_weight)
        return {"loss": loss_tracker.result()}

    @property
    def metrics(self):
        return [loss_tracker]


def get_model(
    build=False, compile=False, jit_compile=False, include_preprocessing=True
):
    if include_preprocessing:
        inputs = keras.layers.Input((), dtype="string")
        x = get_input_preprocessor()(inputs)
    else:
        inputs = keras.layers.Input((SEQUENCE_LENGTH,), dtype=tf.int64)
        x = inputs
    word_embeddings = keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM)(x)
    position_embeddings = keras.layers.Embedding(
        input_dim=SEQUENCE_LENGTH,
        output_dim=EMBED_DIM,
        weights=[get_pos_encoding_matrix(SEQUENCE_LENGTH, EMBED_DIM)],
        trainable=False,
    )(tf.range(start=0, limit=SEQUENCE_LENGTH, delta=1))
    embeddings = word_embeddings + position_embeddings

    encoder_output = embeddings
    for i in range(NUM_LAYERS):
        encoder_output = bert_module(
            encoder_output, encoder_output, encoder_output, i
        )

    mlm_output = keras.layers.Dense(
        VOCAB_SIZE, name="mlm_cls", activation="softmax"
    )(encoder_output)
    model = MaskedLanguageModel(inputs, mlm_output)

    if compile:
        optimizer = keras.optimizers.Adam()
        model.compile(optimizer=optimizer, jit_compile=jit_compile)
    return model


def get_custom_objects():
    return {
        "MaskedLanguageModel": MaskedLanguageModel,
    }
