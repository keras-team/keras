import warnings

from keras.src import backend
from keras.src import layers
from keras.src.api_export import keras_export
from keras.src.applications import imagenet_utils
from keras.src.models import Functional
from keras.src.ops import operation_utils
from keras.src.utils import file_utils

BASE_WEIGHT_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/"
)


@keras_export(
    [
        "keras.applications.mobilenet_v2.MobileNetV2",
        "keras.applications.MobileNetV2",
    ]
)
def MobileNetV2(
    input_shape=None,
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name=None,
):
    """Instantiates the MobileNetV2 architecture.

    MobileNetV2 is very similar to the original MobileNet,
    except that it uses inverted residual blocks with
    bottlenecking features. It has a drastically lower
    parameter count than the original MobileNet.
    MobileNets support any input size greater
    than 32 x 32, with larger image sizes
    offering better performance.

    Reference:
    - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](
        https://arxiv.org/abs/1801.04381) (CVPR 2018)

    This function returns a Keras image classification model,
    optionally loaded with weights pre-trained on ImageNet.

    For image classification use cases, see
    [this page for detailed examples](
      https://keras.io/api/applications/#usage-examples-for-image-classification-models).

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](
      https://keras.io/guides/transfer_learning/).

    Note: each Keras Application expects a specific kind of input preprocessing.
    For MobileNetV2, call
    `keras.applications.mobilenet_v2.preprocess_input`
    on your inputs before passing them to the model.
    `mobilenet_v2.preprocess_input` will scale input pixels between -1 and 1.

    Args:
        input_shape: Optional shape tuple, only to be specified if `include_top`
            is `False` (otherwise the input shape has to be `(224, 224, 3)`
            (with `"channels_last"` data format) or `(3, 224, 224)`
            (with `"channels_first"` data format).
            It should have exactly 3 inputs channels, and width and
            height should be no smaller than 32. E.g. `(200, 200, 3)` would
            be one valid value. Defaults to `None`.
            `input_shape` will be ignored if the `input_tensor` is provided.
        alpha: Controls the width of the network. This is known as the width
            multiplier in the MobileNet paper.
            - If `alpha < 1.0`, proportionally decreases the number
                of filters in each layer.
            - If `alpha > 1.0`, proportionally increases the number
                of filters in each layer.
            - If `alpha == 1`, default number of filters from the paper
                are used at each layer. Defaults to `1.0`.
        include_top: Boolean, whether to include the fully-connected layer
            at the top of the network. Defaults to `True`.
        weights: One of `None` (random initialization), `"imagenet"`
            (pre-training on ImageNet), or the path to the weights file
            to be loaded. Defaults to `"imagenet"`.
        input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model. `input_tensor` is useful
            for sharing inputs between multiple different networks.
            Defaults to `None`.
        pooling: Optional pooling mode for feature extraction when `include_top`
            is `False`.
            - `None` (default) means that the output of the model will be
                the 4D tensor output of the last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        classes: Optional number of classes to classify images into,
            only to be specified if `include_top` is `True`, and if
            no `weights` argument is specified. Defaults to `1000`.
        classifier_activation: A `str` or callable. The activation function
            to use on the "top" layer. Ignored unless `include_top=True`.
            Set `classifier_activation=None` to return the logits of the "top"
            layer. When loading pretrained weights, `classifier_activation`
            can only be `None` or `"softmax"`.
        name: String, the name of the model.

    Returns:
        A model instance.
    """
    if not (weights in {"imagenet", None} or file_utils.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded.  "
            f"Received `weights={weights}`"
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            'If using `weights="imagenet"` with `include_top` '
            f"as true, `classes` should be 1000. Received `classes={classes}`"
        )

    # Determine proper input shape and default size.
    # If both input_shape and input_tensor are used, they should match
    if input_shape is not None and input_tensor is not None:
        try:
            is_input_t_tensor = backend.is_keras_tensor(input_tensor)
        except ValueError:
            try:
                is_input_t_tensor = backend.is_keras_tensor(
                    operation_utils.get_source_inputs(input_tensor)
                )
            except ValueError:
                raise ValueError(
                    f"input_tensor: {input_tensor}"
                    "is not type input_tensor. "
                    f"Received `type(input_tensor)={type(input_tensor)}`"
                )
        if is_input_t_tensor:
            if backend.image_data_format() == "channels_first":
                if input_tensor.shape[1] != input_shape[1]:
                    raise ValueError(
                        "input_shape[1] must equal shape(input_tensor)[1] "
                        "when `image_data_format` is `channels_first`; "
                        "Received `input_tensor.shape="
                        f"{input_tensor.shape}`"
                        f", `input_shape={input_shape}`"
                    )
            else:
                if input_tensor.shape[2] != input_shape[1]:
                    raise ValueError(
                        "input_tensor.shape[2] must equal input_shape[1]; "
                        "Received `input_tensor.shape="
                        f"{input_tensor.shape}`, "
                        f"`input_shape={input_shape}`"
                    )
        else:
            raise ValueError(
                "input_tensor is not a Keras tensor; "
                f"Received `input_tensor={input_tensor}`"
            )

    # If input_shape is None, infer shape from input_tensor.
    if input_shape is None and input_tensor is not None:
        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError(
                "input_tensor must be a valid Keras tensor type; "
                f"Received {input_tensor} of type {type(input_tensor)}"
            )

        if input_shape is None and not backend.is_keras_tensor(input_tensor):
            default_size = 224
        elif input_shape is None and backend.is_keras_tensor(input_tensor):
            if backend.image_data_format() == "channels_first":
                rows = input_tensor.shape[2]
                cols = input_tensor.shape[3]
            else:
                rows = input_tensor.shape[1]
                cols = input_tensor.shape[2]

            if rows == cols and rows in [96, 128, 160, 192, 224]:
                default_size = rows
            else:
                default_size = 224

    # If input_shape is None and no input_tensor
    elif input_shape is None:
        default_size = 224

    # If input_shape is not None, assume default size.
    else:
        if backend.image_data_format() == "channels_first":
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [96, 128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    if backend.image_data_format() == "channels_last":
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == "imagenet":
        if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
            raise ValueError(
                "If imagenet weights are being loaded, "
                "alpha must be one of `0.35`, `0.50`, `0.75`, "
                "`1.0`, `1.3` or `1.4` only;"
                f" Received `alpha={alpha}`"
            )

        if rows != cols or rows not in [96, 128, 160, 192, 224]:
            rows = 224
            warnings.warn(
                "`input_shape` is undefined or non-square, "
                "or `rows` is not in [96, 128, 160, 192, 224]. "
                "Weights for input shape (224, 224) will be "
                "loaded as the default.",
                stacklevel=2,
            )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.Conv2D(
        first_block_filters,
        kernel_size=3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="Conv1",
    )(img_input)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name="bn_Conv1"
    )(x)
    x = layers.ReLU(6.0, name="Conv1_relu")(x)

    x = _inverted_res_block(
        x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0
    )

    x = _inverted_res_block(
        x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1
    )
    x = _inverted_res_block(
        x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2
    )

    x = _inverted_res_block(
        x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3
    )
    x = _inverted_res_block(
        x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4
    )
    x = _inverted_res_block(
        x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5
    )

    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6
    )
    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7
    )
    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8
    )
    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9
    )

    x = _inverted_res_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10
    )
    x = _inverted_res_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11
    )
    x = _inverted_res_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12
    )

    x = _inverted_res_block(
        x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13
    )
    x = _inverted_res_block(
        x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14
    )
    x = _inverted_res_block(
        x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15
    )

    x = _inverted_res_block(
        x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16
    )

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we increase the number of output
    # channels.
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = layers.Conv2D(
        last_block_filters, kernel_size=1, use_bias=False, name="Conv_1"
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name="Conv_1_bn"
    )(x)
    x = layers.ReLU(6.0, name="out_relu")(x)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(
            classes, activation=classifier_activation, name="predictions"
        )(x)

    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account any potential predecessors of
    # `input_tensor`.
    if input_tensor is not None:
        inputs = operation_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    if name is None:
        name = f"mobilenetv2_{alpha:0.2f}_{rows}"
    model = Functional(inputs, x, name=name)

    # Load weights.
    if weights == "imagenet":
        if include_top:
            model_name = (
                "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_"
                + str(float(alpha))
                + "_"
                + str(rows)
                + ".h5"
            )
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = file_utils.get_file(
                model_name, weight_path, cache_subdir="models"
            )
        else:
            model_name = (
                "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_"
                + str(float(alpha))
                + "_"
                + str(rows)
                + "_no_top"
                + ".h5"
            )
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = file_utils.get_file(
                model_name, weight_path, cache_subdir="models"
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    """Inverted ResNet block."""
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    in_channels = inputs.shape[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    # Ensure the number of filters on the last 1x1 convolution is divisible by
    # 8.
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = f"block_{block_id}_"

    if block_id:
        # Expand with a pointwise 1x1 convolution.
        x = layers.Conv2D(
            expansion * in_channels,
            kernel_size=1,
            padding="same",
            use_bias=False,
            activation=None,
            name=prefix + "expand",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + "expand_BN",
        )(x)
        x = layers.ReLU(6.0, name=prefix + "expand_relu")(x)
    else:
        prefix = "expanded_conv_"

    # Depthwise 3x3 convolution.
    if stride == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, 3), name=prefix + "pad"
        )(x)
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        activation=None,
        use_bias=False,
        padding="same" if stride == 1 else "valid",
        name=prefix + "depthwise",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "depthwise_BN",
    )(x)

    x = layers.ReLU(6.0, name=prefix + "depthwise_relu")(x)

    # Project with a pointwise 1x1 convolution.
    x = layers.Conv2D(
        pointwise_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        activation=None,
        name=prefix + "project",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "project_BN",
    )(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + "add")([inputs, x])
    return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@keras_export("keras.applications.mobilenet_v2.preprocess_input")
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(
        x, data_format=data_format, mode="tf"
    )


@keras_export("keras.applications.mobilenet_v2.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__
