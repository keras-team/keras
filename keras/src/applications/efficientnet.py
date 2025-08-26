import copy
import math

from keras.src import backend
from keras.src import layers
from keras.src.api_export import keras_export
from keras.src.applications import imagenet_utils
from keras.src.models import Functional
from keras.src.ops import operation_utils
from keras.src.utils import file_utils

BASE_WEIGHTS_PATH = "https://storage.googleapis.com/keras-applications/"

WEIGHTS_HASHES = {
    "b0": (
        "902e53a9f72be733fc0bcb005b3ebbac",
        "50bc09e76180e00e4465e1a485ddc09d",
    ),
    "b1": (
        "1d254153d4ab51201f1646940f018540",
        "74c4e6b3e1f6a1eea24c589628592432",
    ),
    "b2": (
        "b15cce36ff4dcbd00b6dd88e7857a6ad",
        "111f8e2ac8aa800a7a99e3239f7bfb39",
    ),
    "b3": (
        "ffd1fdc53d0ce67064dc6a9c7960ede0",
        "af6d107764bb5b1abb91932881670226",
    ),
    "b4": (
        "18c95ad55216b8f92d7e70b3a046e2fc",
        "ebc24e6d6c33eaebbd558eafbeedf1ba",
    ),
    "b5": (
        "ace28f2a6363774853a83a0b21b9421a",
        "38879255a25d3c92d5e44e04ae6cec6f",
    ),
    "b6": (
        "165f6e37dce68623721b423839de8be5",
        "9ecce42647a20130c1f39a5d4cb75743",
    ),
    "b7": (
        "8c03f828fec3ef71311cd463b6759d99",
        "cbcfe4450ddf6f3ad90b1b398090fe4a",
    ),
}

DEFAULT_BLOCKS_ARGS = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
]

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal",
    },
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 1.0 / 3.0,
        "mode": "fan_out",
        "distribution": "uniform",
    },
}

BASE_DOCSTRING = """Instantiates the {name} architecture.

Reference:
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
    https://arxiv.org/abs/1905.11946) (ICML 2019)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples](
https://keras.io/api/applications/#usage-examples-for-image-classification-models).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning](
https://keras.io/guides/transfer_learning/).

Note: each Keras Application expects a specific kind of input preprocessing.
For EfficientNet, input preprocessing is included as part of the model
(as a `Rescaling` layer), and thus
`keras.applications.efficientnet.preprocess_input` is actually a
pass-through function. EfficientNet models expect their inputs to be float
tensors of pixels with values in the `[0-255]` range.

Args:
    include_top: Whether to include the fully-connected
        layer at the top of the network. Defaults to `True`.
    weights: One of `None` (random initialization),
        `"imagenet"` (pre-training on ImageNet),
        or the path to the weights file to be loaded.
        Defaults to `"imagenet"`.
    input_tensor: Optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
    input_shape: Optional shape tuple, only to be specified
        if `include_top` is False.
        It should have exactly 3 inputs channels.
    pooling: Optional pooling mode for feature extraction
        when `include_top` is `False`. Defaults to `None`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional layer.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional layer, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
    classes: Optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified. 1000 is how many
        ImageNet classes there are. Defaults to `1000`.
    classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        Defaults to `'softmax'`.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.
    name: The name of the model (string).

Returns:
    A model instance.
"""


IMAGENET_STDDEV_RGB = [0.229, 0.224, 0.225]


def EfficientNet(
    width_coefficient,
    depth_coefficient,
    default_size,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    activation="swish",
    blocks_args="default",
    name="efficientnet",
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    weights_name=None,
):
    """Instantiates the EfficientNet architecture.

    Args:
      width_coefficient: float, scaling coefficient for network width.
      depth_coefficient: float, scaling coefficient for network depth.
      default_size: integer, default input image size.
      dropout_rate: float, dropout rate before final classifier layer.
      drop_connect_rate: float, dropout rate at skip connections.
      depth_divisor: integer, a unit of network width.
      activation: activation function.
      blocks_args: list of dicts, parameters to construct block modules.
      name: string, model name.
      include_top: whether to include the fully-connected
          layer at the top of the network.
      weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor
          (i.e. output of `layers.Input()`)
          to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
          if `include_top` is False.
          It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction
          when `include_top` is `False`.
          - `None` means that the output of the model will be
              the 4D tensor output of the
              last convolutional layer.
          - `avg` means that global average pooling
              will be applied to the output of the
              last convolutional layer, and thus
              the output of the model will be a 2D tensor.
          - `max` means that global max pooling will
              be applied.
      classes: optional number of classes to classify images
          into, only to be specified if `include_top` is True, and
          if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
          on the "top" layer. Ignored unless `include_top=True`. Set
          `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
        A model instance.
    """
    if blocks_args == "default":
        blocks_args = DEFAULT_BLOCKS_ARGS

    if not (weights in {"imagenet", None} or file_utils.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded."
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            'If using `weights="imagenet"` with `include_top`'
            " as true, `classes` should be 1000"
        )

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(
            divisor, int(filters + divisor / 2) // divisor * divisor
        )
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    # Build stem
    x = img_input
    x = layers.Rescaling(1.0 / 255.0)(x)
    x = layers.Normalization(axis=bn_axis)(x)

    if weights == "imagenet":
        # Note that the normalization layer uses square value of STDDEV as the
        # variance for the layer: result = (input - mean) / sqrt(var)
        # However, the original implementation uses (input - mean) / var to
        # normalize the input, we need to divide another sqrt(var) to match the
        # original implementation.
        # See https://github.com/tensorflow/tensorflow/issues/49930 for more
        # details
        x = layers.Rescaling(
            [1.0 / math.sqrt(stddev) for stddev in IMAGENET_STDDEV_RGB]
        )(x)

    x = layers.ZeroPadding2D(
        padding=imagenet_utils.correct_pad(x, 3), name="stem_conv_pad"
    )(x)
    x = layers.Conv2D(
        round_filters(32),
        3,
        strides=2,
        padding="valid",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name="stem_conv",
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name="stem_bn")(x)
    x = layers.Activation(activation, name="stem_activation")(x)

    # Build blocks
    blocks_args = copy.deepcopy(blocks_args)

    b = 0
    blocks = float(sum(round_repeats(args["repeats"]) for args in blocks_args))
    for i, args in enumerate(blocks_args):
        assert args["repeats"] > 0
        # Update block input and output filters based on depth multiplier.
        args["filters_in"] = round_filters(args["filters_in"])
        args["filters_out"] = round_filters(args["filters_out"])

        for j in range(round_repeats(args.pop("repeats"))):
            # The first block needs to take care of stride and filter size
            # increase.
            if j > 0:
                args["strides"] = 1
                args["filters_in"] = args["filters_out"]
            x = block(
                x,
                activation,
                drop_connect_rate * b / blocks,
                name=f"block{i + 1}{chr(j + 97)}_",
                **args,
            )
            b += 1

    # Build top
    x = layers.Conv2D(
        round_filters(1280),
        1,
        padding="same",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name="top_conv",
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name="top_bn")(x)
    x = layers.Activation(activation, name="top_activation")(x)
    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name="top_dropout")(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(
            classes,
            activation=classifier_activation,
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            name="predictions",
        )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = operation_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Functional(inputs, x, name=name)

    # Load weights.
    if weights == "imagenet":
        if include_top:
            file_suffix = ".h5"
            file_hash = WEIGHTS_HASHES[weights_name][0]
        else:
            file_suffix = "_notop.h5"
            file_hash = WEIGHTS_HASHES[weights_name][1]
        file_name = name + file_suffix
        weights_path = file_utils.get_file(
            file_name,
            BASE_WEIGHTS_PATH + file_name,
            cache_subdir="models",
            file_hash=file_hash,
        )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    return model


def block(
    inputs,
    activation="swish",
    drop_rate=0.0,
    name="",
    filters_in=32,
    filters_out=16,
    kernel_size=3,
    strides=1,
    expand_ratio=1,
    se_ratio=0.0,
    id_skip=True,
):
    """An inverted residual block.

    Args:
        inputs: input tensor.
        activation: activation function.
        drop_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.
        se_ratio: float between 0 and 1, fraction to squeeze the input filters.
        id_skip: boolean.

    Returns:
        output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=f"{name}expand_conv",
        )(inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=f"{name}expand_bn")(x)
        x = layers.Activation(activation, name=f"{name}expand_activation")(x)
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, kernel_size),
            name=f"{name}dwconv_pad",
        )(x)
        conv_pad = "valid"
    else:
        conv_pad = "same"
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding=conv_pad,
        use_bias=False,
        depthwise_initializer=CONV_KERNEL_INITIALIZER,
        name=f"{name}dwconv",
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=f"{name}bn")(x)
    x = layers.Activation(activation, name=f"{name}activation")(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(name=f"{name}se_squeeze")(x)
        if bn_axis == 1:
            se_shape = (filters, 1, 1)
        else:
            se_shape = (1, 1, filters)
        se = layers.Reshape(se_shape, name=f"{name}se_reshape")(se)
        se = layers.Conv2D(
            filters_se,
            1,
            padding="same",
            activation=activation,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=f"{name}se_reduce",
        )(se)
        se = layers.Conv2D(
            filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=f"{name}se_expand",
        )(se)
        x = layers.multiply([x, se], name=f"{name}se_excite")

    # Output phase
    x = layers.Conv2D(
        filters_out,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=f"{name}project_conv",
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=f"{name}project_bn")(x)
    if id_skip and strides == 1 and filters_in == filters_out:
        if drop_rate > 0:
            x = layers.Dropout(
                drop_rate, noise_shape=(None, 1, 1, 1), name=f"{name}drop"
            )(x)
        x = layers.add([x, inputs], name=f"{name}add")
    return x


@keras_export(
    [
        "keras.applications.efficientnet.EfficientNetB0",
        "keras.applications.EfficientNetB0",
    ]
)
def EfficientNetB0(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="efficientnetb0",
):
    return EfficientNet(
        1.0,
        1.0,
        224,
        0.2,
        name=name,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        weights_name="b0",
    )


@keras_export(
    [
        "keras.applications.efficientnet.EfficientNetB1",
        "keras.applications.EfficientNetB1",
    ]
)
def EfficientNetB1(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="efficientnetb1",
):
    return EfficientNet(
        1.0,
        1.1,
        240,
        0.2,
        name=name,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        weights_name="b1",
    )


@keras_export(
    [
        "keras.applications.efficientnet.EfficientNetB2",
        "keras.applications.EfficientNetB2",
    ]
)
def EfficientNetB2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="efficientnetb2",
):
    return EfficientNet(
        1.1,
        1.2,
        260,
        0.3,
        name=name,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        weights_name="b2",
    )


@keras_export(
    [
        "keras.applications.efficientnet.EfficientNetB3",
        "keras.applications.EfficientNetB3",
    ]
)
def EfficientNetB3(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="efficientnetb3",
):
    return EfficientNet(
        1.2,
        1.4,
        300,
        0.3,
        name=name,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        weights_name="b3",
    )


@keras_export(
    [
        "keras.applications.efficientnet.EfficientNetB4",
        "keras.applications.EfficientNetB4",
    ]
)
def EfficientNetB4(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="efficientnetb4",
):
    return EfficientNet(
        1.4,
        1.8,
        380,
        0.4,
        name=name,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        weights_name="b4",
    )


@keras_export(
    [
        "keras.applications.efficientnet.EfficientNetB5",
        "keras.applications.EfficientNetB5",
    ]
)
def EfficientNetB5(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="efficientnetb5",
):
    return EfficientNet(
        1.6,
        2.2,
        456,
        0.4,
        name=name,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        weights_name="b5",
    )


@keras_export(
    [
        "keras.applications.efficientnet.EfficientNetB6",
        "keras.applications.EfficientNetB6",
    ]
)
def EfficientNetB6(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="efficientnetb6",
):
    return EfficientNet(
        1.8,
        2.6,
        528,
        0.5,
        name=name,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        weights_name="b6",
    )


@keras_export(
    [
        "keras.applications.efficientnet.EfficientNetB7",
        "keras.applications.EfficientNetB7",
    ]
)
def EfficientNetB7(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="efficientnetb7",
):
    return EfficientNet(
        2.0,
        3.1,
        600,
        0.5,
        name=name,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        weights_name="b7",
    )


EfficientNetB0.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB0")
EfficientNetB1.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB1")
EfficientNetB2.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB2")
EfficientNetB3.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB3")
EfficientNetB4.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB4")
EfficientNetB5.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB5")
EfficientNetB6.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB6")
EfficientNetB7.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB7")


@keras_export("keras.applications.efficientnet.preprocess_input")
def preprocess_input(x, data_format=None):
    """A placeholder method for backward compatibility.

    The preprocessing logic has been included in the efficientnet model
    implementation. Users are no longer required to call this method to
    normalize the input data. This method does nothing and only kept as a
    placeholder to align the API surface between old and new version of model.

    Args:
        x: A floating point `numpy.array` or a tensor.
        data_format: Optional data format of the image tensor/array. `None`
            means the global setting `keras.backend.image_data_format()`
            is used (unless you changed it, it uses `"channels_last"`).
            Defaults to `None`.

    Returns:
        Unchanged `numpy.array` or tensor.
    """
    return x


@keras_export("keras.applications.efficientnet.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__
