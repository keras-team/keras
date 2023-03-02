"""Image classification with EfficientNetV2 architecture.

Adapted from the EfficientNetV2 Keras Application.
"""
import math

from tensorflow import keras

from keras.integration_test.models.input_spec import InputSpec

IMG_SIZE = (96, 96)
NUM_CLASSES = 5


def get_data_spec(batch_size):
    return (
        InputSpec((batch_size,) + IMG_SIZE + (3,)),
        InputSpec((batch_size, NUM_CLASSES)),
    )


def get_input_preprocessor():
    return keras.layers.Rescaling(scale=1.0 / 128.0, offset=-1)


def round_filters(filters, width_coefficient, min_depth, depth_divisor):
    filters *= width_coefficient
    minimum_depth = min_depth or depth_divisor
    new_filters = max(
        minimum_depth,
        int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
    )
    return int(new_filters)


def MBConvBlock(
    input_filters: int,
    output_filters: int,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    activation="swish",
    survival_probability: float = 0.8,
):
    def apply(inputs):
        filters = input_filters * expand_ratio
        if expand_ratio != 1:
            x = keras.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                padding="same",
                data_format="channels_last",
                use_bias=False,
            )(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(activation)(x)
        else:
            x = inputs

        x = keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            data_format="channels_last",
            use_bias=False,
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation)(x)

        if 0 < se_ratio <= 1:
            filters_se = max(1, int(input_filters * se_ratio))
            se = keras.layers.GlobalAveragePooling2D()(x)
            se = keras.layers.Reshape((1, 1, filters))(se)
            se = keras.layers.Conv2D(
                filters_se,
                1,
                padding="same",
                activation=activation,
            )(se)
            se = keras.layers.Conv2D(
                filters,
                1,
                padding="same",
                activation="sigmoid",
            )(se)
            x = keras.layers.multiply([x, se])
            x = keras.layers.Conv2D(
                filters=output_filters,
                kernel_size=1,
                strides=1,
                padding="same",
                data_format="channels_last",
                use_bias=False,
            )(x)
            x = keras.layers.BatchNormalization()(x)

            if strides == 1 and input_filters == output_filters:
                if survival_probability:
                    x = keras.layers.Dropout(
                        survival_probability,
                        noise_shape=(None, 1, 1, 1),
                    )(x)
                x = keras.layers.add([x, inputs])
        return x

    return apply


def FusedMBConvBlock(
    input_filters: int,
    output_filters: int,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    activation="swish",
    survival_probability: float = 0.8,
):
    def apply(inputs):
        filters = input_filters * expand_ratio
        if expand_ratio != 1:
            x = keras.layers.Conv2D(
                filters,
                kernel_size=kernel_size,
                strides=strides,
                data_format="channels_last",
                padding="same",
                use_bias=False,
            )(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(activation)(x)
        else:
            x = inputs

        if 0 < se_ratio <= 1:
            filters_se = max(1, int(input_filters * se_ratio))
            se = keras.layers.GlobalAveragePooling2D()(x)
            se = keras.layers.Reshape((1, 1, filters))(se)
            se = keras.layers.Conv2D(
                filters_se,
                1,
                padding="same",
                activation=activation,
            )(se)
            se = keras.layers.Conv2D(
                filters,
                1,
                padding="same",
                activation="sigmoid",
            )(se)
            x = keras.layers.multiply([x, se])

        x = keras.layers.Conv2D(
            output_filters,
            kernel_size=1 if expand_ratio != 1 else kernel_size,
            strides=1 if expand_ratio != 1 else strides,
            padding="same",
            use_bias=False,
        )(x)
        x = keras.layers.BatchNormalization()(x)

        if expand_ratio == 1:
            x = keras.layers.Activation(activation)(x)

        if strides == 1 and input_filters == output_filters:
            if survival_probability:
                x = keras.layers.Dropout(
                    survival_probability,
                    noise_shape=(None, 1, 1, 1),
                )(x)
            x = keras.layers.add([x, inputs])

        return x

    return apply


def get_model(
    build=False, compile=False, jit_compile=False, include_preprocessing=True
):
    width_coefficient = 1.0
    depth_coefficient = 1.0
    dropout_rate = 0.2
    drop_connect_rate = 0.2
    depth_divisor = 8
    min_depth = 8
    activation = "swish"
    blocks_args = [
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 24,
            "output_filters": 24,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 24,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "conv_type": 1,
            "expand_ratio": 4,
            "input_filters": 48,
            "kernel_size": 3,
            "num_repeat": 4,
            "output_filters": 64,
            "se_ratio": 0,
            "strides": 2,
        },
        {
            "conv_type": 0,
            "expand_ratio": 4,
            "input_filters": 64,
            "kernel_size": 3,
            "num_repeat": 6,
            "output_filters": 128,
            "se_ratio": 0.25,
            "strides": 2,
        },
    ]

    inputs = keras.layers.Input(shape=IMG_SIZE + (3,))
    if include_preprocessing:
        x = get_input_preprocessor()(inputs)
    else:
        x = inputs

    stem_filters = round_filters(
        filters=blocks_args[0]["input_filters"],
        width_coefficient=width_coefficient,
        min_depth=min_depth,
        depth_divisor=depth_divisor,
    )
    x = keras.layers.Conv2D(
        filters=stem_filters,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation, name="stem_activation")(x)

    b = 0
    blocks = float(sum(args["num_repeat"] for args in blocks_args))
    for _, args in enumerate(blocks_args):
        args["input_filters"] = round_filters(
            filters=args["input_filters"],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
        )
        args["output_filters"] = round_filters(
            filters=args["output_filters"],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
        )
        block = {0: MBConvBlock, 1: FusedMBConvBlock}[args.pop("conv_type")]
        repeats = int(math.ceil(depth_coefficient * args.pop("num_repeat")))
        for j in range(repeats):
            if j > 0:
                args["strides"] = 1
                args["input_filters"] = args["output_filters"]

            x = block(
                activation=activation,
                survival_probability=drop_connect_rate * b / blocks,
                **args,
            )(x)
            b += 1

    top_filters = round_filters(
        filters=1280,
        width_coefficient=width_coefficient,
        min_depth=min_depth,
        depth_divisor=depth_divisor,
    )
    x = keras.layers.Conv2D(
        filters=top_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        data_format="channels_last",
        use_bias=False,
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation=activation, name="top_activation")(x)
    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = keras.layers.Dropout(dropout_rate, name="top_dropout")(x)
    x = keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
    )(x)
    model = keras.Model(inputs, x)
    if compile:
        model.compile(
            "adam", loss="categorical_crossentropy", jit_compile=jit_compile
        )
    return model


def get_custom_objects():
    return {}
