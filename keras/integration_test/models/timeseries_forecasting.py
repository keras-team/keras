from tensorflow import keras

from keras.integration_test.models.input_spec import InputSpec

TIMESTEPS = 32


def get_data_spec(batch_size):
    return (
        InputSpec((batch_size, TIMESTEPS, 1)),
        InputSpec((batch_size, 1)),
    )


def get_input_preprocessor():
    return None


def get_model(
    build=False, compile=False, jit_compile=False, include_preprocessing=True
):
    model = keras.Sequential(
        [
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.LSTM(32),
            keras.layers.Dense(1),
        ]
    )
    if build:
        model.build((None, TIMESTEPS, 1))
    if compile:
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss="mse",
            jit_compile=jit_compile,
        )
    return model


def get_custom_objects():
    return {}
