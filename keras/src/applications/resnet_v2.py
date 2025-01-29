from keras.src.api_export import keras_export
from keras.src.applications import imagenet_utils
from keras.src.applications import resnet


@keras_export(
    [
        "keras.applications.ResNet50V2",
        "keras.applications.resnet_v2.ResNet50V2",
    ]
)
def ResNet50V2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="resnet50v2",
):
    """Instantiates the ResNet50V2 architecture."""

    def stack_fn(x):
        x = resnet.stack_residual_blocks_v2(x, 64, 3, name="conv2")
        x = resnet.stack_residual_blocks_v2(x, 128, 4, name="conv3")
        x = resnet.stack_residual_blocks_v2(x, 256, 6, name="conv4")
        return resnet.stack_residual_blocks_v2(
            x, 512, 3, stride1=1, name="conv5"
        )

    return resnet.ResNet(
        stack_fn,
        True,
        True,
        name=name,
        weights_name="resnet50v2",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


@keras_export(
    [
        "keras.applications.ResNet101V2",
        "keras.applications.resnet_v2.ResNet101V2",
    ]
)
def ResNet101V2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="resnet101v2",
):
    """Instantiates the ResNet101V2 architecture."""

    def stack_fn(x):
        x = resnet.stack_residual_blocks_v2(x, 64, 3, name="conv2")
        x = resnet.stack_residual_blocks_v2(x, 128, 4, name="conv3")
        x = resnet.stack_residual_blocks_v2(x, 256, 23, name="conv4")
        return resnet.stack_residual_blocks_v2(
            x, 512, 3, stride1=1, name="conv5"
        )

    return resnet.ResNet(
        stack_fn,
        True,
        True,
        name=name,
        weights_name="resnet101v2",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


@keras_export(
    [
        "keras.applications.ResNet152V2",
        "keras.applications.resnet_v2.ResNet152V2",
    ]
)
def ResNet152V2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="resnet152v2",
):
    """Instantiates the ResNet152V2 architecture."""

    def stack_fn(x):
        x = resnet.stack_residual_blocks_v2(x, 64, 3, name="conv2")
        x = resnet.stack_residual_blocks_v2(x, 128, 8, name="conv3")
        x = resnet.stack_residual_blocks_v2(x, 256, 36, name="conv4")
        return resnet.stack_residual_blocks_v2(
            x, 512, 3, stride1=1, name="conv5"
        )

    return resnet.ResNet(
        stack_fn,
        True,
        True,
        name=name,
        weights_name="resnet152v2",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


@keras_export("keras.applications.resnet_v2.preprocess_input")
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(
        x, data_format=data_format, mode="tf"
    )


@keras_export("keras.applications.resnet_v2.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__


DOC = """

Reference:
- [Identity Mappings in Deep Residual Networks](
    https://arxiv.org/abs/1603.05027) (CVPR 2016)

For image classification use cases, see [this page for detailed examples](
    https://keras.io/api/applications/#usage-examples-for-image-classification-models).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning](
    https://keras.io/guides/transfer_learning/).

Note: each Keras Application expects a specific kind of input preprocessing.
For ResNet, call `keras.applications.resnet_v2.preprocess_input` on your
inputs before passing them to the model. `resnet_v2.preprocess_input` will
scale input pixels between -1 and 1.

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
    name: The name of the model (string).

Returns:
    A Model instance.
"""

setattr(ResNet50V2, "__doc__", ResNet50V2.__doc__ + DOC)
setattr(ResNet101V2, "__doc__", ResNet101V2.__doc__ + DOC)
setattr(ResNet152V2, "__doc__", ResNet152V2.__doc__ + DOC)
