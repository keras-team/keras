import numpy as np

from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src import random
from keras.src.api_export import keras_export
from keras.src.applications import imagenet_utils
from keras.src.layers.layer import Layer
from keras.src.models import Functional
from keras.src.models import Sequential
from keras.src.ops import operation_utils
from keras.src.utils import file_utils

BASE_WEIGHTS_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/convnext/"
)

WEIGHTS_HASHES = {
    "convnext_tiny": (
        "8ae6e78ce2933352b1ef4008e6dd2f17bc40771563877d156bc6426c7cf503ff",
        "d547c096cabd03329d7be5562c5e14798aa39ed24b474157cef5e85ab9e49ef1",
    ),
    "convnext_small": (
        "ce1277d8f1ee5a0ef0e171469089c18f5233860ceaf9b168049cb9263fd7483c",
        "6fc8009faa2f00c1c1dfce59feea9b0745eb260a7dd11bee65c8e20843da6eab",
    ),
    "convnext_base": (
        "52cbb006d3dadd03f6e095a8ca1aca47aecdd75acb4bc74bce1f5c695d0086e6",
        "40a20c5548a5e9202f69735ecc06c990e6b7c9d2de39f0361e27baeb24cb7c45",
    ),
    "convnext_large": (
        "070c5ed9ed289581e477741d3b34beffa920db8cf590899d6d2c67fba2a198a6",
        "96f02b6f0753d4f543261bc9d09bed650f24dd6bc02ddde3066135b63d23a1cd",
    ),
    "convnext_xlarge": (
        "c1f5ccab661354fc3a79a10fa99af82f0fbf10ec65cb894a3ae0815f17a889ee",
        "de3f8a54174130e0cecdc71583354753d557fcf1f4487331558e2a16ba0cfe05",
    ),
}


MODEL_CONFIGS = {
    "tiny": {
        "depths": [3, 3, 9, 3],
        "projection_dims": [96, 192, 384, 768],
        "default_size": 224,
    },
    "small": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [96, 192, 384, 768],
        "default_size": 224,
    },
    "base": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [128, 256, 512, 1024],
        "default_size": 224,
    },
    "large": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [192, 384, 768, 1536],
        "default_size": 224,
    },
    "xlarge": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [256, 512, 1024, 2048],
        "default_size": 224,
    },
}

BASE_DOCSTRING = """Instantiates the {name} architecture.

References:
- [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
(CVPR 2022)

For image classification use cases, see
[this page for detailed examples](
https://keras.io/api/applications/#usage-examples-for-image-classification-models).
For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning](
https://keras.io/guides/transfer_learning/).

The `base`, `large`, and `xlarge` models were first pre-trained on the
ImageNet-21k dataset and then fine-tuned on the ImageNet-1k dataset. The
pre-trained parameters of the models were assembled from the
[official repository](https://github.com/facebookresearch/ConvNeXt). To get a
sense of how these parameters were converted to Keras compatible parameters,
please refer to
[this repository](https://github.com/sayakpaul/keras-convnext-conversion).

Note: Each Keras Application expects a specific kind of input preprocessing.
For ConvNeXt, preprocessing is included in the model using a `Normalization`
layer.  ConvNeXt models expect their inputs to be float or uint8 tensors of
pixels with values in the [0-255] range.

When calling the `summary()` method after instantiating a ConvNeXt model,
prefer setting the `expand_nested` argument `summary()` to `True` to better
investigate the instantiated model.

Args:
    include_top: Whether to include the fully-connected
        layer at the top of the network. Defaults to `True`.
    weights: One of `None` (random initialization),
        `"imagenet"` (pre-training on ImageNet-1k), or the path to the weights
        file to be loaded. Defaults to `"imagenet"`.
    input_tensor: Optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
    input_shape: Optional shape tuple, only to be specified
        if `include_top` is `False`.
        It should have exactly 3 inputs channels.
    pooling: Optional pooling mode for feature extraction
        when `include_top` is `False`. Defaults to None.
        - `None` means that the output of the model will be
        the 4D tensor output of the last convolutional layer.
        - `avg` means that global average pooling
        will be applied to the output of the
        last convolutional layer, and thus
        the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
        be applied.
    classes: Optional number of classes to classify images
        into, only to be specified if `include_top` is `True`, and
        if no `weights` argument is specified. Defaults to 1000 (number of
        ImageNet classes).
    classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        Defaults to `"softmax"`.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.
    name: The name of the model (string).

Returns:
    A model instance.
"""


class StochasticDepth(Layer):
    """Stochastic Depth module.

    It performs batch-wise dropping rather than sample-wise. In libraries like
    `timm`, it's similar to `DropPath` layers that drops residual paths
    sample-wise.

    References:
    - https://github.com/rwightman/pytorch-image-models

    Args:
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].

    Returns:
      Tensor either with the residual path dropped or kept.
    """

    def __init__(self, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (ops.shape(x)[0],) + (1,) * (len(ops.shape(x)) - 1)
            random_tensor = keep_prob + random.uniform(shape, 0, 1)
            random_tensor = ops.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config


class LayerScale(Layer):
    """Layer scale module.

    References:

    - https://arxiv.org/abs/2103.17239

    Args:
        init_values (float): Initial value for layer scale. Should be within
            [0, 1].
        projection_dim (int): Projection dimensionality.

    Returns:
        Tensor multiplied to the scale.
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, _):
        self.gamma = self.add_weight(
            shape=(self.projection_dim,),
            initializer=initializers.Constant(self.init_values),
            trainable=True,
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config


def ConvNeXtBlock(
    projection_dim, drop_path_rate=0.0, layer_scale_init_value=1e-6, name=None
):
    """ConvNeXt block.

    References:
    - https://arxiv.org/abs/2201.03545
    - https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Notes:
        In the original ConvNeXt implementation (linked above), the authors use
        `Dense` layers for pointwise convolutions for increased efficiency.
        Following that, this implementation also uses the same.

    Args:
        projection_dim (int): Number of filters for convolution layers. In the
            ConvNeXt paper, this is referred to as projection dimension.
        drop_path_rate (float): Probability of dropping paths. Should be within
            [0, 1].
        layer_scale_init_value (float): Layer scale value.
            Should be a small float number.
        name: name to path to the keras layer.

    Returns:
        A function representing a ConvNeXtBlock block.
    """
    if name is None:
        name = f"prestem{str(backend.get_uid('prestem'))}"

    def apply(inputs):
        x = inputs

        x = layers.Conv2D(
            filters=projection_dim,
            kernel_size=7,
            padding="same",
            groups=projection_dim,
            name=f"{name}_depthwise_conv",
        )(x)
        x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_layernorm")(x)
        x = layers.Dense(4 * projection_dim, name=f"{name}_pointwise_conv_1")(x)
        x = layers.Activation("gelu", name=f"{name}_gelu")(x)
        x = layers.Dense(projection_dim, name=f"{name}_pointwise_conv_2")(x)

        if layer_scale_init_value is not None:
            x = LayerScale(
                layer_scale_init_value,
                projection_dim,
                name=f"{name}_layer_scale",
            )(x)
        if drop_path_rate:
            layer = StochasticDepth(
                drop_path_rate, name=f"{name}_stochastic_depth"
            )
        else:
            layer = layers.Activation("linear", name=f"{name}_identity")

        return inputs + layer(x)

    return apply


def PreStem(name=None):
    """Normalizes inputs with ImageNet-1k mean and std."""
    if name is None:
        name = "prestem{0}".format(str(backend.get_uid("prestem")))

    def apply(x):
        x = layers.Normalization(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            variance=[
                (0.229 * 255) ** 2,
                (0.224 * 255) ** 2,
                (0.225 * 255) ** 2,
            ],
            name=f"{name}_prestem_normalization",
        )(x)
        return x

    return apply


def Head(num_classes=1000, classifier_activation=None, name=None):
    """Implementation of classification head of ConvNeXt.

    Args:
        num_classes: number of classes for Dense layer
        classifier_activation: activation function for the Dense layer
        name: name prefix

    Returns:
        Classification head function.
    """
    if name is None:
        name = str(backend.get_uid("head"))

    def apply(x):
        x = layers.GlobalAveragePooling2D(name=f"{name}_head_gap")(x)
        x = layers.LayerNormalization(
            epsilon=1e-6, name=f"{name}_head_layernorm"
        )(x)
        x = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name=f"{name}_head_dense",
        )(x)
        return x

    return apply


def ConvNeXt(
    depths,
    projection_dims,
    drop_path_rate=0.0,
    layer_scale_init_value=1e-6,
    default_size=224,
    name="convnext",
    include_preprocessing=True,
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    weights_name=None,
):
    """Instantiates ConvNeXt architecture given specific configuration.

    Args:
        depths: An iterable containing depths for each individual stages.
        projection_dims: An iterable containing output number of channels of
        each individual stages.
        drop_path_rate: Stochastic depth probability. If 0.0, then stochastic
            depth won't be used.
        layer_scale_init_value: Layer scale coefficient. If 0.0, layer scaling
            won't be used.
        default_size: Default input image size.
        name: An optional name for the model.
        include_preprocessing: boolean denoting whether to
            include preprocessing in the model.
            When `weights="imagenet"` this should always be `True`.
            But for other models (e.g., randomly initialized) you should set it
            to `False` and apply preprocessing to data accordingly.
        include_top: Boolean denoting whether to include classification
            head to the model.
        weights: one of `None` (random initialization), `"imagenet"`
            (pre-training on ImageNet-1k),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to
            use as image input for the model.
        input_shape: optional shape tuple, only to be specified if `include_top`
            is `False`. It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction when `include_top`
            is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional layer.
            - `avg` means that global average pooling will be applied
                to the output of the last convolutional layer,
                and thus the output of the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        classes: optional number of classes to classify images into,
            only to be specified if `include_top` is `True`,
            and if no `weights` argument is specified.
        classifier_activation: A `str` or callable.
            The activation function to use
            on the "top" layer. Ignored unless `include_top=True`.
            Set `classifier_activation=None` to return the logits
            of the "top" layer.

    Returns:
        A model instance.
    """
    if backend.image_data_format() == "channels_first":
        raise ValueError(
            "ConvNeXt does not support the `channels_first` image data "
            "format. Switch to `channels_last` by editing your local "
            "config file at ~/.keras/keras.json"
        )
    if not (weights in {"imagenet", None} or file_utils.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded."
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            'If using `weights="imagenet"` with `include_top=True`, '
            "`classes` should be 1000. "
            f"Received classes={classes}"
        )

    # Determine proper input shape.
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

    if input_tensor is not None:
        inputs = operation_utils.get_source_inputs(input_tensor)[0]
        x = input_tensor
    else:
        inputs = img_input
        x = inputs

    if include_preprocessing:
        channel_axis = (
            3 if backend.image_data_format() == "channels_last" else 1
        )
        num_channels = input_shape[channel_axis - 1]
        if num_channels == 3:
            x = PreStem(name=name)(x)

    # Stem block.
    stem = Sequential(
        [
            layers.Conv2D(
                projection_dims[0],
                kernel_size=4,
                strides=4,
                name=f"{name}_stem_conv",
            ),
            layers.LayerNormalization(
                epsilon=1e-6, name=f"{name}_stem_layernorm"
            ),
        ],
        name=f"{name}_stem",
    )

    # Downsampling blocks.
    downsample_layers = []
    downsample_layers.append(stem)

    num_downsample_layers = 3
    for i in range(num_downsample_layers):
        downsample_layer = Sequential(
            [
                layers.LayerNormalization(
                    epsilon=1e-6,
                    name=f"{name}_downsampling_layernorm_{i}",
                ),
                layers.Conv2D(
                    projection_dims[i + 1],
                    kernel_size=2,
                    strides=2,
                    name=f"{name}_downsampling_conv_{i}",
                ),
            ],
            name=f"{name}_downsampling_block_{i}",
        )
        downsample_layers.append(downsample_layer)

    # Stochastic depth schedule.
    # This is referred from the original ConvNeXt codebase:
    # https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L86
    depth_drop_rates = [
        float(x) for x in np.linspace(0.0, drop_path_rate, sum(depths))
    ]

    # First apply downsampling blocks and then apply ConvNeXt stages.
    cur = 0

    num_convnext_blocks = 4
    for i in range(num_convnext_blocks):
        x = downsample_layers[i](x)
        for j in range(depths[i]):
            x = ConvNeXtBlock(
                projection_dim=projection_dims[i],
                drop_path_rate=depth_drop_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value,
                name=name + f"_stage_{i}_block_{j}",
            )(x)
        cur += depths[i]

    if include_top:
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = Head(
            num_classes=classes,
            classifier_activation=classifier_activation,
            name=name,
        )(x)

    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    model = Functional(inputs=inputs, outputs=x, name=name)

    # Validate weights before requesting them from the API
    if weights == "imagenet":
        expected_config = MODEL_CONFIGS[weights_name.split("convnext_")[-1]]
        if (
            depths != expected_config["depths"]
            or projection_dims != expected_config["projection_dims"]
        ):
            raise ValueError(
                f"Architecture configuration does not match {weights_name} "
                f"variant. When using pre-trained weights, the model "
                f"architecture must match the pre-trained configuration "
                f"exactly. Expected depths: {expected_config['depths']}, "
                f"got: {depths}. Expected projection_dims: "
                f"{expected_config['projection_dims']}, got: {projection_dims}."
            )

        if weights_name not in name:
            raise ValueError(
                f'Model name "{name}" does not match weights variant '
                f'"{weights_name}". When using imagenet weights, model name '
                f'must contain the weights variant (e.g., "convnext_'
                f'{weights_name.split("convnext_")[-1]}").'
            )

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


## Instantiating variants ##


@keras_export(
    [
        "keras.applications.convnext.ConvNeXtTiny",
        "keras.applications.ConvNeXtTiny",
    ]
)
def ConvNeXtTiny(
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="convnext_tiny",
):
    return ConvNeXt(
        weights_name="convnext_tiny",
        depths=MODEL_CONFIGS["tiny"]["depths"],
        projection_dims=MODEL_CONFIGS["tiny"]["projection_dims"],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        default_size=MODEL_CONFIGS["tiny"]["default_size"],
        name=name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


@keras_export(
    [
        "keras.applications.convnext.ConvNeXtSmall",
        "keras.applications.ConvNeXtSmall",
    ]
)
def ConvNeXtSmall(
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="convnext_small",
):
    return ConvNeXt(
        weights_name="convnext_small",
        depths=MODEL_CONFIGS["small"]["depths"],
        projection_dims=MODEL_CONFIGS["small"]["projection_dims"],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        default_size=MODEL_CONFIGS["small"]["default_size"],
        name=name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


@keras_export(
    [
        "keras.applications.convnext.ConvNeXtBase",
        "keras.applications.ConvNeXtBase",
    ]
)
def ConvNeXtBase(
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="convnext_base",
):
    return ConvNeXt(
        weights_name="convnext_base",
        depths=MODEL_CONFIGS["base"]["depths"],
        projection_dims=MODEL_CONFIGS["base"]["projection_dims"],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        default_size=MODEL_CONFIGS["base"]["default_size"],
        name=name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


@keras_export(
    [
        "keras.applications.convnext.ConvNeXtLarge",
        "keras.applications.ConvNeXtLarge",
    ]
)
def ConvNeXtLarge(
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="convnext_large",
):
    return ConvNeXt(
        weights_name="convnext_large",
        depths=MODEL_CONFIGS["large"]["depths"],
        projection_dims=MODEL_CONFIGS["large"]["projection_dims"],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        default_size=MODEL_CONFIGS["large"]["default_size"],
        name=name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


@keras_export(
    [
        "keras.applications.convnext.ConvNeXtXLarge",
        "keras.applications.ConvNeXtXLarge",
    ]
)
def ConvNeXtXLarge(
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="convnext_xlarge",
):
    return ConvNeXt(
        weights_name="convnext_xlarge",
        depths=MODEL_CONFIGS["xlarge"]["depths"],
        projection_dims=MODEL_CONFIGS["xlarge"]["projection_dims"],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        default_size=MODEL_CONFIGS["xlarge"]["default_size"],
        name=name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


ConvNeXtTiny.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtTiny")
ConvNeXtSmall.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtSmall")
ConvNeXtBase.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtBase")
ConvNeXtLarge.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtLarge")
ConvNeXtXLarge.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtXLarge")


@keras_export("keras.applications.convnext.preprocess_input")
def preprocess_input(x, data_format=None):
    """A placeholder method for backward compatibility.

    The preprocessing logic has been included in the convnext model
    implementation. Users are no longer required to call this method to
    normalize the input data. This method does nothing and only kept as a
    placeholder to align the API surface between old and new version of model.

    Args:
        x: A floating point `numpy.array` or a tensor.
        data_format: Optional data format of the image tensor/array. Defaults to
            None, in which case the global setting
            `keras.backend.image_data_format()` is used
            (unless you changed it, it defaults to `"channels_last"`).{mode}

    Returns:
        Unchanged `numpy.array` or tensor.
    """
    return x


@keras_export("keras.applications.convnext.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__
