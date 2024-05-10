from keras.src import backend
from keras.src import layers
from keras.src.api_export import keras_export
from keras.src.applications import imagenet_utils
from keras.src.models import Functional
from keras.src.ops import operation_utils
from keras.src.utils import file_utils

WEIGHTS_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/"
    "xception/xception_weights_tf_dim_ordering_tf_kernels.h5"
)
WEIGHTS_PATH_NO_TOP = (
    "https://storage.googleapis.com/tensorflow/keras-applications/"
    "xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
)


@keras_export(
    [
        "keras.applications.xception.Xception",
        "keras.applications.Xception",
    ]
)
def Xception(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="xception",
):
    """Instantiates the Xception architecture.

    Reference:
    - [Xception: Deep Learning with Depthwise Separable Convolutions](
        https://arxiv.org/abs/1610.02357) (CVPR 2017)

    For image classification use cases, see
    [this page for detailed examples](
      https://keras.io/api/applications/#usage-examples-for-image-classification-models).

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](
      https://keras.io/guides/transfer_learning/).

    The default input image size for this model is 299x299.

    Note: each Keras Application expects a specific kind of input preprocessing.
    For Xception, call `keras.applications.xception.preprocess_input`
    on your inputs before passing them to the model.
    `xception.preprocess_input` will scale input pixels between -1 and 1.

    Args:
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
            `"imagenet"` (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(299, 299, 3)`.
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 71.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified.
        classifier_activation: A `str` or callable. The activation function to
            use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top"
            layer.  When loading pretrained weights, `classifier_activation` can
            only be `None` or `"softmax"`.
        name: The name of the model (string).

    Returns:
        A model instance.
    """
    if not (weights in {"imagenet", None} or file_utils.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), 'imagenet' "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded."
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            "If using `weights='imagenet'` with `include_top=True`, "
            "`classes` should be 1000.  "
            f"Received classes={classes}"
        )

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=71,
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

    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    x = layers.Conv2D(
        32, (3, 3), strides=(2, 2), use_bias=False, name="block1_conv1"
    )(img_input)
    x = layers.BatchNormalization(axis=channel_axis, name="block1_conv1_bn")(x)
    x = layers.Activation("relu", name="block1_conv1_act")(x)
    x = layers.Conv2D(64, (3, 3), use_bias=False, name="block1_conv2")(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block1_conv2_bn")(x)
    x = layers.Activation("relu", name="block1_conv2_act")(x)

    residual = layers.Conv2D(
        128, (1, 1), strides=(2, 2), padding="same", use_bias=False
    )(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.SeparableConv2D(
        128, (3, 3), padding="same", use_bias=False, name="block2_sepconv1"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block2_sepconv1_bn")(
        x
    )
    x = layers.Activation("relu", name="block2_sepconv2_act")(x)
    x = layers.SeparableConv2D(
        128, (3, 3), padding="same", use_bias=False, name="block2_sepconv2"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block2_sepconv2_bn")(
        x
    )

    x = layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding="same", name="block2_pool"
    )(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(
        256, (1, 1), strides=(2, 2), padding="same", use_bias=False
    )(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation("relu", name="block3_sepconv1_act")(x)
    x = layers.SeparableConv2D(
        256, (3, 3), padding="same", use_bias=False, name="block3_sepconv1"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block3_sepconv1_bn")(
        x
    )
    x = layers.Activation("relu", name="block3_sepconv2_act")(x)
    x = layers.SeparableConv2D(
        256, (3, 3), padding="same", use_bias=False, name="block3_sepconv2"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block3_sepconv2_bn")(
        x
    )

    x = layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding="same", name="block3_pool"
    )(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(
        728, (1, 1), strides=(2, 2), padding="same", use_bias=False
    )(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation("relu", name="block4_sepconv1_act")(x)
    x = layers.SeparableConv2D(
        728, (3, 3), padding="same", use_bias=False, name="block4_sepconv1"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block4_sepconv1_bn")(
        x
    )
    x = layers.Activation("relu", name="block4_sepconv2_act")(x)
    x = layers.SeparableConv2D(
        728, (3, 3), padding="same", use_bias=False, name="block4_sepconv2"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block4_sepconv2_bn")(
        x
    )

    x = layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding="same", name="block4_pool"
    )(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = "block" + str(i + 5)

        x = layers.Activation("relu", name=prefix + "_sepconv1_act")(x)
        x = layers.SeparableConv2D(
            728,
            (3, 3),
            padding="same",
            use_bias=False,
            name=prefix + "_sepconv1",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=prefix + "_sepconv1_bn"
        )(x)
        x = layers.Activation("relu", name=prefix + "_sepconv2_act")(x)
        x = layers.SeparableConv2D(
            728,
            (3, 3),
            padding="same",
            use_bias=False,
            name=prefix + "_sepconv2",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=prefix + "_sepconv2_bn"
        )(x)
        x = layers.Activation("relu", name=prefix + "_sepconv3_act")(x)
        x = layers.SeparableConv2D(
            728,
            (3, 3),
            padding="same",
            use_bias=False,
            name=prefix + "_sepconv3",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=prefix + "_sepconv3_bn"
        )(x)

        x = layers.add([x, residual])

    residual = layers.Conv2D(
        1024, (1, 1), strides=(2, 2), padding="same", use_bias=False
    )(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation("relu", name="block13_sepconv1_act")(x)
    x = layers.SeparableConv2D(
        728, (3, 3), padding="same", use_bias=False, name="block13_sepconv1"
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name="block13_sepconv1_bn"
    )(x)
    x = layers.Activation("relu", name="block13_sepconv2_act")(x)
    x = layers.SeparableConv2D(
        1024, (3, 3), padding="same", use_bias=False, name="block13_sepconv2"
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name="block13_sepconv2_bn"
    )(x)

    x = layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding="same", name="block13_pool"
    )(x)
    x = layers.add([x, residual])

    x = layers.SeparableConv2D(
        1536, (3, 3), padding="same", use_bias=False, name="block14_sepconv1"
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name="block14_sepconv1_bn"
    )(x)
    x = layers.Activation("relu", name="block14_sepconv1_act")(x)

    x = layers.SeparableConv2D(
        2048, (3, 3), padding="same", use_bias=False, name="block14_sepconv2"
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name="block14_sepconv2_bn"
    )(x)
    x = layers.Activation("relu", name="block14_sepconv2_act")(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(
            classes, activation=classifier_activation, name="predictions"
        )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)

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
            weights_path = file_utils.get_file(
                "xception_weights_tf_dim_ordering_tf_kernels.h5",
                WEIGHTS_PATH,
                cache_subdir="models",
                file_hash="0a58e3b7378bc2990ea3b43d5981f1f6",
            )
        else:
            weights_path = file_utils.get_file(
                "xception_weights_tf_dim_ordering_tf_kernels_notop.h5",
                WEIGHTS_PATH_NO_TOP,
                cache_subdir="models",
                file_hash="b0042744bf5b25fce3cb969f33bebb97",
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


@keras_export("keras.applications.xception.preprocess_input")
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(
        x, data_format=data_format, mode="tf"
    )


@keras_export("keras.applications.xception.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__
