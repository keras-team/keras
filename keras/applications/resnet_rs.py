# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
"""ResNet-RS models for Keras.

Reference:
- [Revisiting ResNets: Improved Training and Scaling Strategies](
    https://arxiv.org/pdf/2103.07579.pdf)
"""
import sys
from typing import Callable, Dict, List, Union

from keras import backend
from keras import layers
from keras.applications import imagenet_utils
from keras.engine import training
from keras.utils import data_utils
from keras.utils import layer_utils
import tensorflow.compat.v2 as tf


from tensorflow.python.util.tf_export import keras_export

BASE_WEIGHTS_URL = (
    "https://storage.googleapis.com/tensorflow/" "keras-applications/resnet_rs/"
)

WEIGHT_HASHES = {
    "resnet-rs-101-i160.h5": "544b3434d00efc199d66e9058c7f3379",
    "resnet-rs-101-i160_notop.h5": "82d5b90c5ce9d710da639d6216d0f979",
    "resnet-rs-101-i192.h5": "eb285be29ab42cf4835ff20a5e3b5d23",
    "resnet-rs-101-i192_notop.h5": "f9a0f6b85faa9c3db2b6e233c4eebb5b",
    "resnet-rs-152-i192.h5": "8d72a301ed8a6f11a47c4ced4396e338",
    "resnet-rs-152-i192_notop.h5": "5fbf7ac2155cb4d5a6180ee9e3aa8704",
    "resnet-rs-152-i224.h5": "31a46a92ab21b84193d0d71dd8c3d03b",
    "resnet-rs-152-i224_notop.h5": "dc8b2cba2005552eafa3167f00dc2133",
    "resnet-rs-152-i256.h5": "ba6271b99bdeb4e7a9b15c05964ef4ad",
    "resnet-rs-152-i256_notop.h5": "fa79794252dbe47c89130f65349d654a",
    "resnet-rs-200-i256.h5": "a76930b741884e09ce90fa7450747d5f",
    "resnet-rs-200-i256_notop.h5": "bbdb3994718dfc0d1cd45d7eff3f3d9c",
    "resnet-rs-270-i256.h5": "20d575825ba26176b03cb51012a367a8",
    "resnet-rs-270-i256_notop.h5": "2c42ecb22e35f3e23d2f70babce0a2aa",
    "resnet-rs-350-i256.h5": "f4a039dc3c421321b7fc240494574a68",
    "resnet-rs-350-i256_notop.h5": "6e44b55025bbdff8f51692a023143d66",
    "resnet-rs-350-i320.h5": "7ccb858cc738305e8ceb3c0140bee393",
    "resnet-rs-350-i320_notop.h5": "ab0c1f9079d2f85a9facbd2c88aa6079",
    "resnet-rs-420-i320.h5": "ae0eb9bed39e64fc8d7e0db4018dc7e8",
    "resnet-rs-420-i320_notop.h5": "fe6217c32be8305b1889657172b98884",
    "resnet-rs-50-i160.h5": "69d9d925319f00a8bdd4af23c04e4102",
    "resnet-rs-50-i160_notop.h5": "90daa68cd26c95aa6c5d25451e095529",
}

DEPTH_TO_WEIGHT_VARIANTS = {
    50: [160],
    101: [160, 192],
    152: [192, 224, 256],
    200: [256],
    270: [256],
    350: [256, 320],
    420: [320],
}
BLOCK_ARGS = {
    50: [
        {"input_filters": 64, "num_repeats": 3},
        {"input_filters": 128, "num_repeats": 4},
        {"input_filters": 256, "num_repeats": 6},
        {"input_filters": 512, "num_repeats": 3},
    ],
    101: [
        {"input_filters": 64, "num_repeats": 3},
        {"input_filters": 128, "num_repeats": 4},
        {"input_filters": 256, "num_repeats": 23},
        {"input_filters": 512, "num_repeats": 3},
    ],
    152: [
        {"input_filters": 64, "num_repeats": 3},
        {"input_filters": 128, "num_repeats": 8},
        {"input_filters": 256, "num_repeats": 36},
        {"input_filters": 512, "num_repeats": 3},
    ],
    200: [
        {"input_filters": 64, "num_repeats": 3},
        {"input_filters": 128, "num_repeats": 24},
        {"input_filters": 256, "num_repeats": 36},
        {"input_filters": 512, "num_repeats": 3},
    ],
    270: [
        {"input_filters": 64, "num_repeats": 4},
        {"input_filters": 128, "num_repeats": 29},
        {"input_filters": 256, "num_repeats": 53},
        {"input_filters": 512, "num_repeats": 4},
    ],
    350: [
        {"input_filters": 64, "num_repeats": 4},
        {"input_filters": 128, "num_repeats": 36},
        {"input_filters": 256, "num_repeats": 72},
        {"input_filters": 512, "num_repeats": 4},
    ],
    420: [
        {"input_filters": 64, "num_repeats": 4},
        {"input_filters": 128, "num_repeats": 44},
        {"input_filters": 256, "num_repeats": 87},
        {"input_filters": 512, "num_repeats": 4},
    ],
}
CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal",
    },
}

BASE_DOCSTRING = """Instantiates the {name} architecture.

    Reference:
    [Revisiting ResNets: Improved Training and Scaling Strategies](
    https://arxiv.org/pdf/2103.07579.pdf)

    For image classification use cases, see
    [this page for detailed examples](
    https://keras.io/api/applications/#usage-examples-for-image-classification-models).

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](
    https://keras.io/guides/transfer_learning/).

    Note: each Keras Application expects a specific kind of input preprocessing.
    For ResNetRs, by default input preprocessing is included as a part of the
    model (as a `Rescaling` layer), and thus
    `tf.keras.applications.resnet_rs.preprocess_input` is actually a
    pass-through function. In this use case, ResNetRS models expect their inputs
    to be float tensors of pixels with values in the [0-255] range.
    At the same time, preprocessing as a part of the model (i.e. `Rescaling`
    layer) can be disabled by setting `include_preprocessing` argument to False.
    With preprocessing disabled ResNetRS models expect their inputs to be float
    tensors of pixels with values in the [-1, 1] range.

    Args:
        depth: Depth of ResNet network.
        input_shape: optional shape tuple. It should have exactly 3 inputs
            channels, and width and height should be no smaller than 32.
            E.g. (200, 200, 3) would be one valid value.
        bn_momentum: Momentum parameter for Batch Normalization layers.
        bn_epsilon: Epsilon parameter for Batch Normalization layers.
        activation: activation function.
        se_ratio: Squeeze and Excitation layer ratio.
        dropout_rate: dropout rate before final classifier layer.
        drop_connect_rate: dropout rate at skip connections.
        include_top: whether to include the fully-connected layer at the top of
        the network.
        block_args: list of dicts, parameters to construct block modules.
        model_name: name of the model.
        pooling: optional pooling mode for feature extraction when `include_top`
            is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        weights: one of `None` (random initialization), `'imagenet'`
            (pre-training on ImageNet), or the path to the weights file to be
            loaded.  Note: one model can have multiple imagenet variants
            depending on input shape it was trained with. For input_shape
            224x224 pass `imagenet-i224` as argument. By default, highest input
            shape weights are downloaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to
            use as image input for the model.
        classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True, and if no `weights` argument is
            specified.
        classifier_activation: A `str` or callable. The activation function to
            use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.
        include_preprocessing: Boolean, whether to include the preprocessing layer
            (`Rescaling`) at the bottom of the network. Defaults to `True`.
            Note: Input image is normalized by ImageNet mean and standard deviation.

    Returns:
        A `keras.Model` instance.
"""


def Conv2DFixedPadding(filters, kernel_size, strides, name=None):
    """Conv2D block with fixed padding."""
    if name is None:
        counter = backend.get_uid("conv_")
        name = f"conv_{counter}"

    def apply(inputs):
        if strides > 1:
            inputs = fixed_padding(inputs, kernel_size)
        return layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same" if strides == 1 else "valid",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name,
        )(inputs)

    return apply


def STEM(
    bn_momentum: float = 0.0,
    bn_epsilon: float = 1e-5,
    activation: str = "relu",
    name=None,
):
    """ResNet-D type STEM block."""
    if name is None:
        counter = backend.get_uid("stem_")
        name = f"stem_{counter}"

    def apply(inputs):
        bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

        # First stem block
        x = Conv2DFixedPadding(
            filters=32, kernel_size=3, strides=2, name=name + "_stem_conv_1"
        )(inputs)
        x = layers.BatchNormalization(
            axis=bn_axis,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name=name + "_stem_batch_norm_1",
        )(x)
        x = layers.Activation(activation, name=name + "_stem_act_1")(x)

        # Second stem block
        x = Conv2DFixedPadding(
            filters=32, kernel_size=3, strides=1, name=name + "_stem_conv_2"
        )(x)
        x = layers.BatchNormalization(
            axis=bn_axis,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name=name + "_stem_batch_norm_2",
        )(x)
        x = layers.Activation(activation, name=name + "_stem_act_2")(x)

        # Final Stem block:
        x = Conv2DFixedPadding(
            filters=64, kernel_size=3, strides=1, name=name + "_stem_conv_3"
        )(x)
        x = layers.BatchNormalization(
            axis=bn_axis,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name=name + "_stem_batch_norm_3",
        )(x)
        x = layers.Activation(activation, name=name + "_stem_act_3")(x)

        # Replace stem max pool:
        x = Conv2DFixedPadding(
            filters=64, kernel_size=3, strides=2, name=name + "_stem_conv_4"
        )(x)
        x = layers.BatchNormalization(
            axis=bn_axis,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name=name + "_stem_batch_norm_4",
        )(x)
        x = layers.Activation(activation, name=name + "_stem_act_4")(x)
        return x

    return apply


def SE(
    in_filters: int, se_ratio: float = 0.25, expand_ratio: int = 1, name=None
):
    """Squeeze and Excitation block."""
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    if name is None:
        counter = backend.get_uid("se_")
        name = f"se_{counter}"

    def apply(inputs):
        x = layers.GlobalAveragePooling2D(name=name + "_se_squeeze")(inputs)
        if bn_axis == 1:
            se_shape = (x.shape[-1], 1, 1)
        else:
            se_shape = (1, 1, x.shape[-1])
        x = layers.Reshape(se_shape, name=name + "_se_reshape")(x)

        num_reduced_filters = max(1, int(in_filters * 4 * se_ratio))

        x = layers.Conv2D(
            filters=num_reduced_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=True,
            activation="relu",
            name=name + "_se_reduce",
        )(x)

        x = layers.Conv2D(
            filters=4
            * in_filters
            * expand_ratio,  # Expand ratio is 1 by default
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=True,
            activation="sigmoid",
            name=name + "_se_expand",
        )(x)

        return layers.multiply([inputs, x], name=name + "_se_excite")

    return apply


def BottleneckBlock(
    filters: int,
    strides: int,
    use_projection: bool,
    bn_momentum: float = 0.0,
    bn_epsilon: float = 1e-5,
    activation: str = "relu",
    se_ratio: float = 0.25,
    survival_probability: float = 0.8,
    name=None,
):
    """Bottleneck block variant for residual networks with BN."""
    if name is None:
        counter = backend.get_uid("block_0_")
        name = f"block_0_{counter}"

    def apply(inputs):
        bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

        shortcut = inputs

        if use_projection:
            filters_out = filters * 4
            if strides == 2:
                shortcut = layers.AveragePooling2D(
                    pool_size=(2, 2),
                    strides=(2, 2),
                    padding="same",
                    name=name + "_projection_pooling",
                )(inputs)
                shortcut = Conv2DFixedPadding(
                    filters=filters_out,
                    kernel_size=1,
                    strides=1,
                    name=name + "_projection_conv",
                )(shortcut)
            else:
                shortcut = Conv2DFixedPadding(
                    filters=filters_out,
                    kernel_size=1,
                    strides=strides,
                    name=name + "_projection_conv",
                )(inputs)

            shortcut = layers.BatchNormalization(
                axis=bn_axis,
                momentum=bn_momentum,
                epsilon=bn_epsilon,
                name=name + "_projection_batch_norm",
            )(shortcut)

        # First conv layer:
        x = Conv2DFixedPadding(
            filters=filters, kernel_size=1, strides=1, name=name + "_conv_1"
        )(inputs)
        x = layers.BatchNormalization(
            axis=bn_axis,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name=name + "batch_norm_1",
        )(x)
        x = layers.Activation(activation, name=name + "_act_1")(x)

        # Second conv layer:
        x = Conv2DFixedPadding(
            filters=filters,
            kernel_size=3,
            strides=strides,
            name=name + "_conv_2",
        )(x)
        x = layers.BatchNormalization(
            axis=bn_axis,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name=name + "_batch_norm_2",
        )(x)
        x = layers.Activation(activation, name=name + "_act_2")(x)

        # Third conv layer:
        x = Conv2DFixedPadding(
            filters=filters * 4, kernel_size=1, strides=1, name=name + "_conv_3"
        )(x)
        x = layers.BatchNormalization(
            axis=bn_axis,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name=name + "_batch_norm_3",
        )(x)

        if 0 < se_ratio < 1:
            x = SE(filters, se_ratio=se_ratio, name=name + "_se")(x)

        # Drop connect
        if survival_probability:
            x = layers.Dropout(
                survival_probability,
                noise_shape=(None, 1, 1, 1),
                name=name + "_drop",
            )(x)

        x = layers.Add()([x, shortcut])

        return layers.Activation(activation, name=name + "_output_act")(x)

    return apply


def BlockGroup(
    filters,
    strides,
    num_repeats,
    se_ratio: float = 0.25,
    bn_epsilon: float = 1e-5,
    bn_momentum: float = 0.0,
    activation: str = "relu",
    survival_probability: float = 0.8,
    name=None,
):
    """Create one group of blocks for the ResNet model."""
    if name is None:
        counter = backend.get_uid("block_group_")
        name = f"block_group_{counter}"

    def apply(inputs):
        # Only the first block per block_group uses projection shortcut and strides.
        x = BottleneckBlock(
            filters=filters,
            strides=strides,
            use_projection=True,
            se_ratio=se_ratio,
            bn_epsilon=bn_epsilon,
            bn_momentum=bn_momentum,
            activation=activation,
            survival_probability=survival_probability,
            name=name + "_block_0_",
        )(inputs)

        for i in range(1, num_repeats):
            x = BottleneckBlock(
                filters=filters,
                strides=1,
                use_projection=False,
                se_ratio=se_ratio,
                activation=activation,
                bn_epsilon=bn_epsilon,
                bn_momentum=bn_momentum,
                survival_probability=survival_probability,
                name=name + f"_block_{i}_",
            )(x)
        return x

    return apply


def get_survival_probability(init_rate, block_num, total_blocks):
    """Get survival probability based on block number and initial rate."""
    return init_rate * float(block_num) / total_blocks


def allow_bigger_recursion(target_limit: int):
    """Increase default recursion limit to create larger models."""
    current_limit = sys.getrecursionlimit()
    if current_limit < target_limit:
        sys.setrecursionlimit(target_limit)


def fixed_padding(inputs, kernel_size):
    """Pad the input along the spatial dimensions independently of input size."""
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    # Use ZeroPadding as to avoid TFOpLambda layer
    padded_inputs = layers.ZeroPadding2D(
        padding=((pad_beg, pad_end), (pad_beg, pad_end))
    )(inputs)

    return padded_inputs


def ResNetRS(
    depth: int,
    input_shape=None,
    bn_momentum=0.0,
    bn_epsilon=1e-5,
    activation: str = "relu",
    se_ratio=0.25,
    dropout_rate=0.25,
    drop_connect_rate=0.2,
    include_top=True,
    block_args: List[Dict[str, int]] = None,
    model_name="resnet-rs",
    pooling=None,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    # pylint: disable=g-bare-generic
    classifier_activation: Union[str, Callable] = "softmax",
    include_preprocessing=True,
):
    """Build Resnet-RS model, given provided parameters.

    Args:
        depth: Depth of ResNet network.
        input_shape: optional shape tuple. It should have exactly 3 inputs
          channels, and width and height should be no smaller than 32. E.g. (200,
          200, 3) would be one valid value.
        bn_momentum: Momentum parameter for Batch Normalization layers.
        bn_epsilon: Epsilon parameter for Batch Normalization layers.
        activation: activation function.
        se_ratio: Squeeze and Excitation layer ratio.
        dropout_rate: dropout rate before final classifier layer.
        drop_connect_rate: dropout rate at skip connections.
        include_top: whether to include the fully-connected layer at the top of
          the network.
        block_args: list of dicts, parameters to construct block modules.
        model_name: name of the model.
        pooling: optional pooling mode for feature extraction when `include_top`
          is `False`. - `None` means that the output of the model will be the 4D
          tensor output of the last convolutional layer. - `avg` means that global
          average pooling will be applied to the output of the last convolutional
          layer, and thus the output of the model will be a 2D tensor. - `max`
          means that global max pooling will be applied.
        weights: one of `None` (random initialization), `'imagenet'` (pre-training
          on ImageNet), or the path to the weights file to be loaded. Note- one
          model can have multiple imagenet variants depending on input shape it
          was trained with. For input_shape 224x224 pass `imagenet-i224` as
          argument. By default, highest input shape weights are downloaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to
          use as image input for the model.
        classes: optional number of classes to classify images into, only to be
          specified if `include_top` is True, and if no `weights` argument is
          specified.
        classifier_activation: A `str` or callable. The activation function to use
          on the "top" layer. Ignored unless `include_top=True`. Set
          `classifier_activation=None` to return the logits of the "top" layer.
        include_preprocessing: Boolean, whether to include the preprocessing layer
          (`Rescaling`) at the bottom of the network. Defaults to `True`. Note-
          Input image is normalized by ImageNet mean and standard deviation.

    Returns:
        A `tf.keras.Model` instance.

    Raises:
        ValueError: in case of invalid argument for `weights`, or invalid input
            shape.
        ValueError: if `classifier_activation` is not `softmax` or `None` when
            using a pretrained top layer.
    """
    # Validate parameters
    available_weight_variants = DEPTH_TO_WEIGHT_VARIANTS[depth]
    if weights == "imagenet":
        max_input_shape = max(available_weight_variants)
        # `imagenet` argument without explicit weights input size.
        # Picking weights trained with biggest available shape
        weights = f"{weights}-i{max_input_shape}"

    weights_allow_list = [f"imagenet-i{x}" for x in available_weight_variants]
    if not (
        weights in {*weights_allow_list, None} or tf.io.gfile.exists(weights)
    ):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `'imagenet'` "
            "(pre-training on ImageNet, with highest available input shape),"
            " or the path to the weights file to be loaded. "
            f"For ResNetRS{depth} the following weight variants are "
            f"available {weights_allow_list} (default=highest)."
            f" Received weights={weights}"
        )

    if weights in weights_allow_list and include_top and classes != 1000:
        raise ValueError(
            f"If using `weights` as `'imagenet'` or any of {weights_allow_list} "
            f"with `include_top` as true, `classes` should be 1000. "
            f"Received classes={classes}"
        )

    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )
    # Define input tensor
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    x = img_input

    if include_preprocessing:
        num_channels = input_shape[bn_axis - 1]
        x = layers.Rescaling(scale=1.0 / 255)(x)
        if num_channels == 3:
            x = layers.Normalization(
                mean=[0.485, 0.456, 0.406],
                variance=[0.229**2, 0.224**2, 0.225**2],
                axis=bn_axis,
            )(x)

    # Build stem
    x = STEM(
        bn_momentum=bn_momentum, bn_epsilon=bn_epsilon, activation=activation
    )(x)

    # Build blocks
    if block_args is None:
        block_args = BLOCK_ARGS[depth]

    for i, args in enumerate(block_args):
        survival_probability = get_survival_probability(
            init_rate=drop_connect_rate,
            block_num=i + 2,
            total_blocks=len(block_args) + 1,
        )

        x = BlockGroup(
            filters=args["input_filters"],
            activation=activation,
            strides=(1 if i == 0 else 2),
            num_repeats=args["num_repeats"],
            se_ratio=se_ratio,
            bn_momentum=bn_momentum,
            bn_epsilon=bn_epsilon,
            survival_probability=survival_probability,
            name=f"BlockGroup{i + 2}_",
        )(x)

    # Build head:
    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name="top_dropout")(x)

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
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = training.Model(inputs, x, name=model_name)

    # Download weights
    if weights in weights_allow_list:
        weights_input_shape = weights.split("-")[-1]  # e. g. "i160"
        weights_name = f"{model_name}-{weights_input_shape}"
        if not include_top:
            weights_name += "_notop"

        filename = f"{weights_name}.h5"
        download_url = BASE_WEIGHTS_URL + filename
        weights_path = data_utils.get_file(
            fname=filename,
            origin=download_url,
            cache_subdir="models",
            file_hash=WEIGHT_HASHES[filename],
        )
        model.load_weights(weights_path)

    elif weights is not None:
        model.load_weights(weights)

    return model


@keras_export(
    "keras.applications.resnet_rs.ResNetRS50", "keras.applications.ResNetRS50"
)
def ResNetRS50(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    """Build ResNet-RS50 model."""
    return ResNetRS(
        depth=50,
        include_top=include_top,
        drop_connect_rate=0.0,
        dropout_rate=0.25,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-50",
        include_preprocessing=include_preprocessing,
    )


@keras_export(
    "keras.applications.resnet_rs.ResNetRS101", "keras.applications.ResNetRS101"
)
def ResNetRS101(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    """Build ResNet-RS101 model."""
    return ResNetRS(
        depth=101,
        include_top=include_top,
        drop_connect_rate=0.0,
        dropout_rate=0.25,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-101",
        include_preprocessing=include_preprocessing,
    )


@keras_export(
    "keras.applications.resnet_rs.ResNetRS152", "keras.applications.ResNetRS152"
)
def ResNetRS152(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    """Build ResNet-RS152 model."""
    return ResNetRS(
        depth=152,
        include_top=include_top,
        drop_connect_rate=0.0,
        dropout_rate=0.25,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-152",
        include_preprocessing=include_preprocessing,
    )


@keras_export(
    "keras.applications.resnet_rs.ResNetRS200", "keras.applications.ResNetRS200"
)
def ResNetRS200(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    """Build ResNet-RS200 model."""
    return ResNetRS(
        depth=200,
        include_top=include_top,
        drop_connect_rate=0.1,
        dropout_rate=0.25,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-200",
        include_preprocessing=include_preprocessing,
    )


@keras_export(
    "keras.applications.resnet_rs.ResNetRS270", "keras.applications.ResNetRS270"
)
def ResNetRS270(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    """Build ResNet-RS-270 model."""
    allow_bigger_recursion(1300)
    return ResNetRS(
        depth=270,
        include_top=include_top,
        drop_connect_rate=0.1,
        dropout_rate=0.25,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-270",
        include_preprocessing=include_preprocessing,
    )


@keras_export(
    "keras.applications.resnet_rs.ResNetRS350", "keras.applications.ResNetRS350"
)
def ResNetRS350(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    """Build ResNet-RS350 model."""
    allow_bigger_recursion(1500)
    return ResNetRS(
        depth=350,
        include_top=include_top,
        drop_connect_rate=0.1,
        dropout_rate=0.4,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-350",
        include_preprocessing=include_preprocessing,
    )


@keras_export(
    "keras.applications.resnet_rs.ResNetRS420", "keras.applications.ResNetRS420"
)
def ResNetRS420(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    """Build ResNet-RS420 model."""
    allow_bigger_recursion(1800)
    return ResNetRS(
        depth=420,
        include_top=include_top,
        dropout_rate=0.4,
        drop_connect_rate=0.1,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-420",
        include_preprocessing=include_preprocessing,
    )


# pylint: disable=unused-argument
@keras_export("keras.applications.resnet_rs.preprocess_input")
def preprocess_input(x, data_format=None):
    """A placeholder method for backward compatibility.

    The preprocessing logic has been included in the ResnetRS model
    implementation. Users are no longer required to call this method to
    normalize
    the input data. This method does nothing and only kept as a placeholder to
    align the API surface between old and new version of model.

    Args:
      x: A floating point `numpy.array` or a `tf.Tensor`.
      data_format: Optional data format of the image tensor/array. Defaults to
        None, in which case the global setting
        `tf.keras.backend.image_data_format()` is used (unless you changed it,
        it defaults to "channels_last").{mode}

    Returns:
      Unchanged `numpy.array` or `tf.Tensor`.
    """
    return x


@keras_export("keras.applications.resnet_rs.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__

ResNetRS50.__doc__ = BASE_DOCSTRING.format(name="ResNetRS50")
ResNetRS152.__doc__ = BASE_DOCSTRING.format(name="ResNetRS152")
ResNetRS200.__doc__ = BASE_DOCSTRING.format(name="ResNetRS200")
ResNetRS270.__doc__ = BASE_DOCSTRING.format(name="ResNetRS270")
ResNetRS350.__doc__ = BASE_DOCSTRING.format(name="ResNetRS350")
ResNetRS420.__doc__ = BASE_DOCSTRING.format(name="ResNetRS420")
