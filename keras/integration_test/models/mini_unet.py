"""Segmentation model.

Adapted from https://keras.io/examples/vision/oxford_pets_image_segmentation/
"""
from tensorflow import keras

from keras.integration_test.models.input_spec import InputSpec

IMG_SIZE = (224, 224)
NUM_CLASSES = 5


def get_data_spec(batch_size):
    return (
        InputSpec((batch_size,) + IMG_SIZE + (3,)),
        InputSpec((batch_size,) + IMG_SIZE + (NUM_CLASSES,)),
    )


def get_input_preprocessor():
    return None


def get_model(
    build=False, compile=False, jit_compile=False, include_preprocessing=True
):
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    previous_block_activation = x
    for filters in [64, 128, 256]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = keras.layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = keras.layers.add([x, residual])
        previous_block_activation = x

    for filters in [256, 128, 64, 32]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.UpSampling2D(2)(x)

        residual = keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = keras.layers.Conv2D(filters, 1, padding="same")(residual)
        x = keras.layers.add([x, residual])
        previous_block_activation = x

    outputs = keras.layers.Conv2D(
        NUM_CLASSES, 3, activation="softmax", padding="same"
    )(x)
    model = keras.Model(inputs, outputs)
    if compile:
        model.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            jit_compile=jit_compile,
        )
    return model


def get_custom_objects():
    return {}
