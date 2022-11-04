"""Text classification model.

Adapted from https://keras.io/examples/nlp/text_classification_from_scratch/
"""
import re
import string

import tensorflow as tf
from tensorflow import keras

from keras.integration_test.models.input_spec import InputSpec

MAX_FEATURES = 1000
EMBEDDING_DIM = 64
SEQUENCE_LENGTH = 32


def get_data_spec(batch_size):
    return (
        InputSpec((batch_size,), dtype="string"),
        InputSpec((batch_size, 1), dtype="int32", range=[0, 2]),
    )


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )


def get_input_preprocessor():
    input_vectorizer = keras.layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=MAX_FEATURES,
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


def get_model(
    build=False, compile=False, jit_compile=False, include_preprocessing=True
):
    if include_preprocessing:
        inputs = keras.Input(shape=(), dtype="string")
        x = get_input_preprocessor()(inputs)
    else:
        inputs = keras.Input(shape=(None,), dtype="int64")
        x = inputs
    x = keras.layers.Embedding(MAX_FEATURES, EMBEDDING_DIM)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv1D(
        128, 7, padding="valid", activation="relu", strides=3
    )(x)
    x = keras.layers.Conv1D(
        128, 7, padding="valid", activation="relu", strides=3
    )(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    predictions = keras.layers.Dense(
        1, activation="sigmoid", name="predictions"
    )(x)
    model = keras.Model(inputs, predictions)

    if compile:
        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
            jit_compile=jit_compile,
        )
    return model


def get_custom_objects():
    return {"custom_standardization": custom_standardization}
