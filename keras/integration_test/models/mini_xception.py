"""Mini-Xception classification model.

Adapted from https://keras.io/examples/vision/image_classification_from_scratch/
"""
from tensorflow import keras

from keras.integration_test.models.input_spec import InputSpec

IMG_SIZE = (120, 120)


def get_data_spec(batch_size):
    return (
        InputSpec((batch_size,) + IMG_SIZE + (3,)),
        InputSpec((batch_size, 1), dtype="int32", range=[0, 2]),
    )


def get_input_preprocessor():
    return keras.Sequential(
        [
            keras.layers.RandomFlip(),
            keras.layers.RandomRotation(0.2),
            keras.layers.RandomZoom(0.2),
            keras.layers.Rescaling(1.0 / 255),
        ]
    )


def get_model(
    build=False, compile=False, jit_compile=False, include_preprocessing=True
):
    inputs = keras.Input(shape=IMG_SIZE + (3,))

    if include_preprocessing:
        x = get_input_preprocessor()(inputs)
    else:
        x = inputs

    x = keras.layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [128, 256, 512, 728]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = keras.layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = keras.layers.add([x, residual])
        previous_block_activation = x

    x = keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    if compile:
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
            jit_compile=jit_compile,
        )
    return model


def get_custom_objects():
    return {}
