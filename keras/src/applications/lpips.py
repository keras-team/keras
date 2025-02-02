from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.applications import imagenet_utils
from keras.src.applications import vgg16
from keras.src.models import Functional
from keras.src.utils import file_utils

WEIGHTS_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/"
    "lpips/lpips_vgg16_weights.h5"
)  # TODO: store weights at this location


def vgg_backbone(layer_names):
    """VGG backbone for LPIPS.

    Args:
        layer_names: list of layer names to extract features from

    Returns:
        Functional model with outputs at specified layers
    """
    vgg = vgg16.VGG16(include_top=False, weights=None)
    outputs = [
        layer.output for layer in vgg.layers if layer.name in layer_names
    ]
    return Functional(vgg.input, outputs)


def linear_model(channels):
    """Get the linear head model for LPIPS.
    Combines feature differences from VGG backbone.

    Args:
        channels: list of channel sizes for feature differences

    Returns:
        Functional model
    """
    inputs, outputs = [], []
    for ii, channel in enumerate(channels):
        x = layers.Input(shape=(None, None, channel))
        y = layers.Dropout(rate=0.5)(x)
        y = layers.Conv2D(
            filters=1,
            kernel_size=1,
            use_bias=False,
            name=f"linear_{ii}",
        )(y)
        inputs.append(x)
        outputs.append(y)

    model = Functional(inputs=inputs, outputs=outputs, name="linear_model")
    return model


@keras_export(["keras.applications.lpips.LPIPS", "keras.applications.LPIPS"])
def LPIPS(
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    network_type="vgg",
    name="lpips",
):
    """Instantiates the LPIPS model.

    Reference:
    - [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](
    https://arxiv.org/abs/1801.03924)

    Args:
        weights: one of `None` (random initialization),
            `"imagenet"` (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor for model input
        input_shape: optional shape tuple, defaults to (None, None, 3)
        network_type: backbone network type (currently only 'vgg' supported)
        name: model name string

    Returns:
        A `Model` instance.
    """
    if network_type != "vgg":
        raise ValueError(
            "Currently only VGG backbone is supported. "
            f"Got network_type={network_type}"
        )

    if not (weights in {"imagenet", None} or file_utils.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), 'imagenet' "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded."
        )

    # Define inputs
    if input_tensor is None:
        img_input1 = layers.Input(
            shape=input_shape or (None, None, 3), name="input1"
        )
        img_input2 = layers.Input(
            shape=input_shape or (None, None, 3), name="input2"
        )
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input1 = layers.Input(tensor=input_tensor, shape=input_shape)
            img_input2 = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input1 = input_tensor
            img_input2 = input_tensor

    # VGG feature extraction
    vgg_layers = [
        "block1_conv2",
        "block2_conv2",
        "block3_conv3",
        "block4_conv3",
        "block5_conv3",
    ]
    vgg_net = vgg_backbone(vgg_layers)

    # Process inputs
    feat1 = vgg_net(img_input1)
    feat2 = vgg_net(img_input2)

    # Normalize features
    def normalize(x):
        return x * ops.rsqrt(ops.sum(ops.square(x), axis=-1, keepdims=True))

    norm1 = [layers.Lambda(normalize)(f) for f in feat1]
    norm2 = [layers.Lambda(normalize)(f) for f in feat2]

    # Feature differences
    diffs = [
        layers.Lambda(lambda x: ops.square(x[0] - x[1]))([n1, n2])
        for n1, n2 in zip(norm1, norm2)
    ]

    # Get shapes for linear model
    channels = [f.shape[-1] for f in feat1]

    linear_net = linear_model(channels)

    lin_out = linear_net(diffs)

    spatial_average = [
        layers.Lambda(lambda x: ops.mean(x, axis=[1, 2]))(t) for t in lin_out
    ]

    output = layers.Lambda(
        lambda x: ops.squeeze(
            ops.sum(backend.convert_to_tensor(x), axis=0), axis=-1
        )
    )(spatial_average)

    # Create model
    model = Functional([img_input1, img_input2], output, name=name)

    # Load weights
    if weights == "imagenet":
        weights_path = file_utils.get_file(
            "lpips_vgg16_weights.h5",
            WEIGHTS_PATH,
            cache_subdir="models",
            file_hash=None,  # TODO: add hash
        )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


@keras_export("keras.applications.lpips.preprocess_input")
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(
        x, data_format=data_format, mode="torch"
    )


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_CAFFE,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
