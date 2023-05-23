from tensorflow.io import gfile

from keras_core import backend
from keras_core import layers
from keras_core.api_export import keras_core_export
from keras_core.applications import imagenet_utils
from keras_core.models import Functional
from keras_core.operations import operation_utils
from keras_core.utils import file_utils

BASE_WEIGHTS_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/resnet/"
)
WEIGHTS_HASHES = {
    "resnet50": (
        "2cb95161c43110f7111970584f804107",
        "4d473c1dd8becc155b73f8504c6f6626",
    ),
}


def ResNet(
    stack_fn,
    preact,
    use_bias,
    model_name="resnet",
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    Args:
        stack_fn: A function that returns output tensor for the
            stacked residual blocks.
        preact: Whether to use pre-activation or not. `True` for ResNetV2,
            `False` for ResNet and ResNeXt.
        use_bias: Whether to use biases for convolutional layers or not.
            `True` for ResNet and ResNetV2, `False` for ResNeXt.
        model_name: Name of the model.
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: One of `None` (random initialization),
            `"imagenet"` (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: Optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `"channels_first"` data format). It
            should have exactly 3 inputs channels.
        pooling: Optional pooling mode for feature extraction
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
          into, only to be specified if `include_top` is `True`,
          and if no `weights` argument is specified.
        classifier_activation: A `str` or callable. The activation
            function to use on the "top" layer. Ignored unless
            `include_top=True`. Set `classifier_activation=None` to
            return the logits of the "top" layer. When loading
            pretrained weights, `classifier_activation` can only be
            `None` or `"softmax"`.

    Returns:
        A Model instance.
    """

    if not (weights in {"imagenet", None} or gfile.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), 'imagenet' "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded.  Received: "
            f"weights={weights}"
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
        default_size=224,
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

    if backend.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(
        img_input
    )
    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name="conv1_conv")(x)

    if not preact:
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name="conv1_bn"
        )(x)
        x = layers.Activation("relu", name="conv1_relu")(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1_pool")(x)

    x = stack_fn(x)

    if preact:
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name="post_bn"
        )(x)
        x = layers.Activation("relu", name="post_relu")(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)

        # Validate activation for the classifier layer
        imagenet_utils.validate_activation(classifier_activation, weights)

        x = layers.Dense(
            classes, activation=classifier_activation, name="predictions"
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
    model = Functional(inputs, x, name=model_name)

    # Load weights.
    if (weights == "imagenet") and (model_name in WEIGHTS_HASHES):
        if include_top:
            file_name = model_name + "_weights_tf_dim_ordering_tf_kernels.h5"
            file_hash = WEIGHTS_HASHES[model_name][0]
        else:
            file_name = (
                model_name + "_weights_tf_dim_ordering_tf_kernels_notop.h5"
            )
            file_hash = WEIGHTS_HASHES[model_name][1]
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


def residual_block_v1(
    x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None
):
    """A residual block for ResNet*_v1.

    Args:
        x: Input tensor.
        filters: No of filters in the bottleneck layer.
        kernel_size: Kernel size of the bottleneck layer. Defaults to 3
        stride: Stride of the first layer. Defaults to 1
        conv_shortcut: Use convolution shortcut if `True`, otherwise
            use identity shortcut. Defaults to `True`
        name(optional): Name of the block

    Returns:
        Output tensor for the residual block.
    """

    if backend.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    if conv_shortcut:
        shortcut = layers.Conv2D(
            4 * filters, 1, strides=stride, name=name + "_0_conv"
        )(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + "_1_conv")(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn"
    )(x)
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    x = layers.Conv2D(
        filters, kernel_size, padding="SAME", name=name + "_2_conv"
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + "_2_bn"
    )(x)
    x = layers.Activation("relu", name=name + "_2_relu")(x)

    x = layers.Conv2D(4 * filters, 1, name=name + "_3_conv")(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + "_3_bn"
    )(x)

    x = layers.Add(name=name + "_add")([shortcut, x])
    x = layers.Activation("relu", name=name + "_out")(x)
    return x


def stack_residual_blocks_v1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    Args:
        x: Input tensor.
        filters: Number of filters in the bottleneck layer in a block.
        blocks: Number of blocks in the stacked blocks.
        stride1: Stride of the first layer in the first block. Defaults to 2.
        name: Stack label.

    Returns:
        Output tensor for the stacked blocks.
    """

    x = residual_block_v1(x, filters, stride=stride1, name=name + "_block1")
    for i in range(2, blocks + 1):
        x = residual_block_v1(
            x, filters, conv_shortcut=False, name=name + "_block" + str(i)
        )
    return x


@keras_core_export(
    [
        "keras_core.applications.resnet50.ResNet50",
        "keras_core.applications.resnet.ResNet50",
        "keras_core.applications.ResNet50",
    ]
)
def ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """Instantiates the ResNet50 architecture."""

    def stack_fn(x):
        x = stack_residual_blocks_v1(x, 64, 3, stride1=1, name="conv2")
        x = stack_residual_blocks_v1(x, 128, 4, name="conv3")
        x = stack_residual_blocks_v1(x, 256, 6, name="conv4")
        return stack_residual_blocks_v1(x, 512, 3, name="conv5")

    return ResNet(
        stack_fn,
        False,
        True,
        "resnet50",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        classifier_activation=classifier_activation,
    )


@keras_core_export(
    [
        "keras_core.applications.resnet50.preprocess_input",
        "keras_core.applications.resnet.preprocess_input",
    ]
)
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(
        x, data_format=data_format, mode="caffe"
    )


@keras_core_export(
    [
        "keras_core.applications.resnet50.decode_predictions",
        "keras_core.applications.resnet.decode_predictions",
    ]
)
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_CAFFE,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__

DOC = """

Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2015)

For image classification use cases, see [this page for detailed examples](
    https://keras.io/api/applications/#usage-examples-for-image-classification-models).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning](
    https://keras.io/guides/transfer_learning/).

Note: each Keras Application expects a specific kind of input preprocessing.
For ResNet, call `keras_core.applications.resnet.preprocess_input` on your
inputs before passing them to the model. `resnet.preprocess_input` will convert
the input images from RGB to BGR, then will zero-center each color channel with
respect to the ImageNet dataset, without scaling.

Args:
    include_top: whether to include the fully-connected
        layer at the top of the network.
    weights: one of `None` (random initialization),
        `"imagenet"` (pre-training on ImageNet), or the path to the weights
        file to be loaded.
    input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
        to use as image input for the model.
    input_shape: optional shape tuple, only to be specified if `include_top`
        is `False` (otherwise the input shape has to be `(224, 224, 3)`
        (with `"channels_last"` data format) or `(3, 224, 224)`
        (with `"channels_first"` data format). It should have exactly 3
        inputs channels, and width and height should be no smaller than 32.
        E.g. `(200, 200, 3)` would be one valid value.
    pooling: Optional pooling mode for feature extraction when `include_top`
        is `False`.
        - `None` means that the output of the model will be the 4D tensor
                output of the last convolutional block.
        - `avg` means that global average pooling will be applied to the output
                of the last convolutional block, and thus the output of the
                model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
    classes: optional number of classes to classify images into, only to be
        specified if `include_top` is `True`, and if no `weights` argument is
        specified.
    classifier_activation: A `str` or callable. The activation function to
        use on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.

Returns:
    A Model instance.
"""

setattr(ResNet50, "__doc__", ResNet50.__doc__ + DOC)
