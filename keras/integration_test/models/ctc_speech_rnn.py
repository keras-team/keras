import tensorflow as tf
from tensorflow import keras

from keras.integration_test.models.input_spec import InputSpec

TIMESTEPS = 64
INPUT_DIM = 50
OUTPUT_DIM = 40
NUM_RNN_LAYERS = 2
RNN_UNITS = 32


def get_input_preprocessor():
    return None


def get_data_spec(batch_size):
    return (
        InputSpec((batch_size, TIMESTEPS, INPUT_DIM)),
        InputSpec((batch_size, 1), dtype="int64", range=[0, OUTPUT_DIM]),
    )


def ctc_loss(y_true, y_pred):
    batch_length = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(
        shape=(batch_length, 1), dtype="int64"
    )
    label_length = label_length * tf.ones(
        shape=(batch_length, 1), dtype="int64"
    )

    return keras.backend.ctc_batch_cost(
        y_true, y_pred, input_length, label_length
    )


def get_model(
    build=False, compile=False, jit_compile=False, include_preprocessing=True
):
    input_spectrogram = keras.layers.Input((None, INPUT_DIM), name="input")
    x = keras.layers.Reshape((-1, INPUT_DIM, 1), name="expand_dim")(
        input_spectrogram
    )
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=[2, 2],
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x)
    x = keras.layers.BatchNormalization(name="conv_1_bn")(x)
    x = keras.layers.ReLU(name="conv_1_relu")(x)
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=[1, 2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = keras.layers.BatchNormalization(name="conv_2_bn")(x)
    x = keras.layers.ReLU(name="conv_2_relu")(x)
    x = keras.layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    for i in range(1, NUM_RNN_LAYERS + 1):
        recurrent = keras.layers.GRU(
            units=RNN_UNITS,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = keras.layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < NUM_RNN_LAYERS:
            x = keras.layers.Dropout(rate=0.5)(x)
    x = keras.layers.Dense(units=RNN_UNITS * 2, name="dense_1")(x)
    x = keras.layers.ReLU(name="dense_1_relu")(x)
    x = keras.layers.Dropout(rate=0.5)(x)
    output = keras.layers.Dense(units=OUTPUT_DIM + 1, activation="softmax")(x)
    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")

    if compile:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=ctc_loss,
            jit_compile=jit_compile,
        )
    return model


def get_custom_objects():
    return {"ctc_loss": ctc_loss}
