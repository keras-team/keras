# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=missing-docstring
# pylint: disable=g-classes-have-attributes

"""RegNet models for Keras.

References:

- [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)
  (CVPR 2020)
- [Fast and Accurate Model Scaling](https://arxiv.org/abs/2103.06877)
  (CVPR 2021)
"""

from keras import backend
from keras import layers
from keras.applications import imagenet_utils
from keras.engine import training
from keras.utils import data_utils
from keras.utils import layer_utils
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export

BASE_WEIGHTS_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/regnet/"
)

WEIGHTS_HASHES = {
    "x002": (
        "49fb46e56cde07fdaf57bffd851461a86548f6a3a4baef234dd37290b826c0b8",
        "5445b66cd50445eb7ecab094c1e78d4d3d29375439d1a7798861c4af15ffff21",
    ),
    "x004": (
        "3523c7f5ac0dbbcc2fd6d83b3570e7540f7449d3301cc22c29547302114e4088",
        "de139bf07a66c9256f2277bf5c1b6dd2d5a3a891a5f8a925a10c8a0a113fd6f3",
    ),
    "x006": (
        "340216ef334a7bae30daac9f414e693c136fac9ab868704bbfcc9ce6a5ec74bb",
        "a43ec97ad62f86b2a96a783bfdc63a5a54de02eef54f26379ea05e1bf90a9505",
    ),
    "x008": (
        "8f145d6a5fae6da62677bb8d26eb92d0b9dfe143ec1ebf68b24a57ae50a2763d",
        "3c7e4b0917359304dc18e644475c5c1f5e88d795542b676439c4a3acd63b7207",
    ),
    "x016": (
        "31c386f4c7bfef4c021a583099aa79c1b3928057ba1b7d182f174674c5ef3510",
        "1b8e3d545d190271204a7b2165936a227d26b79bb7922bac5ee4d303091bf17a",
    ),
    "x032": (
        "6c025df1409e5ea846375bc9dfa240956cca87ef57384d93fef7d6fa90ca8c7f",
        "9cd4522806c0fcca01b37874188b2bd394d7c419956d77472a4e072b01d99041",
    ),
    "x040": (
        "ba128046c588a26dbd3b3a011b26cb7fa3cf8f269c184c132372cb20b6eb54c1",
        "b4ed0ca0b9a98e789e05000e830403a7ade4d8afa01c73491c44610195198afe",
    ),
    "x064": (
        "0f4489c3cd3ad979bd6b0324213998bcb36dc861d178f977997ebfe53c3ba564",
        "3e706fa416a18dfda14c713423eba8041ae2509db3e0a611d5f599b5268a46c4",
    ),
    "x080": (
        "76320e43272719df648db37271a247c22eb6e810fe469c37a5db7e2cb696d162",
        "7b1ce8e29ceefec10a6569640ee329dba7fbc98b5d0f6346aabade058b66cf29",
    ),
    "x120": (
        "5cafc461b78897d5e4f24e68cb406d18e75f31105ef620e7682b611bb355eb3a",
        "36174ddd0299db04a42631d028abcb1cc7afec2b705e42bd28fcd325e5d596bf",
    ),
    "x160": (
        "8093f57a5824b181fb734ea21ae34b1f7ee42c5298e63cf6d587c290973195d2",
        "9d1485050bdf19531ffa1ed7827c75850e0f2972118a996b91aa9264b088fd43",
    ),
    "x320": (
        "91fb3e6f4e9e44b3687e80977f7f4412ee9937c0c704232664fc83e4322ea01e",
        "9db7eacc37b85c98184070e1a172e6104c00846f44bcd4e727da9e50d9692398",
    ),
    "y002": (
        "1e8091c674532b1a61c04f6393a9c570113e0197f22bd1b98cc4c4fe800c6465",
        "f63221f63d625b8e201221499682587bfe29d33f50a4c4f4d53be00f66c0f12c",
    ),
    "y004": (
        "752fdbad21c78911bf1dcb8c513e5a0e14697b068e5d9e73525dbaa416d18d8e",
        "45e6ba8309a17a77e67afc05228454b2e0ee6be0dae65edc0f31f1da10cc066b",
    ),
    "y006": (
        "98942e07b273da500ff9699a1f88aca78dfad4375faabb0bab784bb0dace80a9",
        "b70261cba4e60013c99d130cc098d2fce629ff978a445663b6fa4f8fc099a2be",
    ),
    "y008": (
        "1b099377cc9a4fb183159a6f9b24bc998e5659d25a449f40c90cbffcbcfdcae4",
        "b11f5432a216ee640fe9be6e32939defa8d08b8d136349bf3690715a98752ca1",
    ),
    "y016": (
        "b7ce1f5e223f0941c960602de922bcf846288ce7a4c33b2a4f2e4ac4b480045b",
        "d7404f50205e82d793e219afb9eb2bfeb781b6b2d316a6128c6d7d7dacab7f57",
    ),
    "y032": (
        "6a6a545cf3549973554c9b94f0cd40e25f229fffb1e7f7ac779a59dcbee612bd",
        "eb3ac1c45ec60f4f031c3f5180573422b1cf7bebc26c004637517372f68f8937",
    ),
    "y040": (
        "98d00118b335162bbffe8f1329e54e5c8e75ee09b2a5414f97b0ddfc56e796f6",
        "b5be2a5e5f072ecdd9c0b8a437cd896df0efa1f6a1f77e41caa8719b7dfcb05d",
    ),
    "y064": (
        "65c948c7a18aaecaad2d1bd4fd978987425604ba6669ef55a1faa0069a2804b7",
        "885c4b7ed7ea339daca7dafa1a62cb7d41b1068897ef90a5a3d71b4a2e2db31a",
    ),
    "y080": (
        "7a2c62da2982e369a4984d3c7c3b32d6f8d3748a71cb37a31156c436c37f3e95",
        "3d119577e1e3bf8d153b895e8ea9e4ec150ff2d92abdca711b6e949c3fd7115d",
    ),
    "y120": (
        "a96ab0d27d3ae35a422ee7df0d789069b3e3217a99334e0ce861a96595bc5986",
        "4a6fa387108380b730b71feea2ad80b5224b5ea9dc21dc156c93fe3c6186485c",
    ),
    "y160": (
        "45067240ffbc7ca2591313fee2f80dbdda6d66ec1a7451446f9a6d00d8f7ac6e",
        "ead1e6b568be8f34447ec8941299a9df4368736ba9a8205de5427fa20a1fb316",
    ),
    "y320": (
        "b05e173e4ae635cfa22d06392ee3741284d17dadfee68f2aa6fd8cb2b7561112",
        "cad78f74a586e24c61d38be17f3ae53bb9674380174d2585da1a526b8c20e1fd",
    ),
}

# The widths and depths are deduced from a quantized linear function. For
# more information, please refer to "Designing Network Design Spaces" by
# Radosavovic et al.

# BatchNorm momentum and epsilon values taken from original implementation.

MODEL_CONFIGS = {
    "x002": {
        "depths": [1, 1, 4, 7],
        "widths": [24, 56, 152, 368],
        "group_width": 8,
        "default_size": 224,
        "block_type": "X",
    },
    "x004": {
        "depths": [1, 2, 7, 12],
        "widths": [32, 64, 160, 384],
        "group_width": 16,
        "default_size": 224,
        "block_type": "X",
    },
    "x006": {
        "depths": [1, 3, 5, 7],
        "widths": [48, 96, 240, 528],
        "group_width": 24,
        "default_size": 224,
        "block_type": "X",
    },
    "x008": {
        "depths": [1, 3, 7, 5],
        "widths": [64, 128, 288, 672],
        "group_width": 16,
        "default_size": 224,
        "block_type": "X",
    },
    "x016": {
        "depths": [2, 4, 10, 2],
        "widths": [72, 168, 408, 912],
        "group_width": 24,
        "default_size": 224,
        "block_type": "X",
    },
    "x032": {
        "depths": [2, 6, 15, 2],
        "widths": [96, 192, 432, 1008],
        "group_width": 48,
        "default_size": 224,
        "block_type": "X",
    },
    "x040": {
        "depths": [2, 5, 14, 2],
        "widths": [80, 240, 560, 1360],
        "group_width": 40,
        "default_size": 224,
        "block_type": "X",
    },
    "x064": {
        "depths": [2, 4, 10, 1],
        "widths": [168, 392, 784, 1624],
        "group_width": 56,
        "default_size": 224,
        "block_type": "X",
    },
    "x080": {
        "depths": [2, 5, 15, 1],
        "widths": [80, 240, 720, 1920],
        "group_width": 120,
        "default_size": 224,
        "block_type": "X",
    },
    "x120": {
        "depths": [2, 5, 11, 1],
        "widths": [224, 448, 896, 2240],
        "group_width": 112,
        "default_size": 224,
        "block_type": "X",
    },
    "x160": {
        "depths": [2, 6, 13, 1],
        "widths": [256, 512, 896, 2048],
        "group_width": 128,
        "default_size": 224,
        "block_type": "X",
    },
    "x320": {
        "depths": [2, 7, 13, 1],
        "widths": [336, 672, 1344, 2520],
        "group_width": 168,
        "default_size": 224,
        "block_type": "X",
    },
    "y002": {
        "depths": [1, 1, 4, 7],
        "widths": [24, 56, 152, 368],
        "group_width": 8,
        "default_size": 224,
        "block_type": "Y",
    },
    "y004": {
        "depths": [1, 3, 6, 6],
        "widths": [48, 104, 208, 440],
        "group_width": 8,
        "default_size": 224,
        "block_type": "Y",
    },
    "y006": {
        "depths": [1, 3, 7, 4],
        "widths": [48, 112, 256, 608],
        "group_width": 16,
        "default_size": 224,
        "block_type": "Y",
    },
    "y008": {
        "depths": [1, 3, 8, 2],
        "widths": [64, 128, 320, 768],
        "group_width": 16,
        "default_size": 224,
        "block_type": "Y",
    },
    "y016": {
        "depths": [2, 6, 17, 2],
        "widths": [48, 120, 336, 888],
        "group_width": 24,
        "default_size": 224,
        "block_type": "Y",
    },
    "y032": {
        "depths": [2, 5, 13, 1],
        "widths": [72, 216, 576, 1512],
        "group_width": 24,
        "default_size": 224,
        "block_type": "Y",
    },
    "y040": {
        "depths": [2, 6, 12, 2],
        "widths": [128, 192, 512, 1088],
        "group_width": 64,
        "default_size": 224,
        "block_type": "Y",
    },
    "y064": {
        "depths": [2, 7, 14, 2],
        "widths": [144, 288, 576, 1296],
        "group_width": 72,
        "default_size": 224,
        "block_type": "Y",
    },
    "y080": {
        "depths": [2, 4, 10, 1],
        "widths": [168, 448, 896, 2016],
        "group_width": 56,
        "default_size": 224,
        "block_type": "Y",
    },
    "y120": {
        "depths": [2, 5, 11, 1],
        "widths": [224, 448, 896, 2240],
        "group_width": 112,
        "default_size": 224,
        "block_type": "Y",
    },
    "y160": {
        "depths": [2, 4, 11, 1],
        "widths": [224, 448, 1232, 3024],
        "group_width": 112,
        "default_size": 224,
        "block_type": "Y",
    },
    "y320": {
        "depths": [2, 5, 12, 1],
        "widths": [232, 696, 1392, 3712],
        "group_width": 232,
        "default_size": 224,
        "block_type": "Y",
    },
}

BASE_DOCSTRING = """Instantiates the {name} architecture.

  Reference:
    - [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)
    (CVPR 2020)

  For image classification use cases, see
  [this page for detailed examples](
  https://keras.io/api/applications/#usage-examples-for-image-classification-models).

  For transfer learning use cases, make sure to read the
  [guide to transfer learning & fine-tuning](
    https://keras.io/guides/transfer_learning/).

  Note: Each Keras Application expects a specific kind of input preprocessing.
  For Regnets, preprocessing is included in the model using a `Rescaling` layer.
  RegNet models expect their inputs to be float or uint8 tensors of pixels with
  values in the [0-255] range.

  The naming of models is as follows: `RegNet<block_type><flops>` where
  `block_type` is one of `(X, Y)` and `flops` signifies hundred million
  floating point operations. For example RegNetY064 corresponds to RegNet with
  Y block and 6.4 giga flops (64 hundred million flops).

  Args:
    include_top: Whether to include the fully-connected
        layer at the top of the network. Defaults to True.
    weights: One of `None` (random initialization),
          `"imagenet"` (pre-training on ImageNet), or the path to the weights
          file to be loaded. Defaults to `"imagenet"`.
    input_tensor: Optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
    input_shape: Optional shape tuple, only to be specified
        if `include_top` is False.
        It should have exactly 3 inputs channels.
    pooling: Optional pooling mode for feature extraction
        when `include_top` is `False`. Defaults to None.
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
        if no `weights` argument is specified. Defaults to 1000 (number of
        ImageNet classes).
    classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        Defaults to `"softmax"`.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.

  Returns:
    A `keras.Model` instance.
"""


def PreStem(name=None):
    """Rescales and normalizes inputs to [0,1] and ImageNet mean and std.

    Args:
      name: name prefix

    Returns:
      Rescaled and normalized tensor
    """
    if name is None:
        name = "prestem" + str(backend.get_uid("prestem"))

    def apply(x):
        x = layers.Rescaling(
            scale=1.0 / 255.0, name=name + "_prestem_rescaling"
        )(x)
        return x

    return apply


def Stem(name=None):
    """Implementation of RegNet stem.

    (Common to all model variants)
    Args:
      name: name prefix

    Returns:
      Output tensor of the Stem
    """
    if name is None:
        name = "stem" + str(backend.get_uid("stem"))

    def apply(x):
        x = layers.Conv2D(
            32,
            (3, 3),
            strides=2,
            use_bias=False,
            padding="same",
            kernel_initializer="he_normal",
            name=name + "_stem_conv",
        )(x)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_stem_bn"
        )(x)
        x = layers.ReLU(name=name + "_stem_relu")(x)
        return x

    return apply


def SqueezeAndExciteBlock(filters_in, se_filters, name=None):
    """Implements the Squeeze and excite block (https://arxiv.org/abs/1709.01507).

    Args:
      filters_in: input filters to the block
      se_filters: filters to squeeze to
      name: name prefix

    Returns:
      A function object
    """
    if name is None:
        name = str(backend.get_uid("squeeze_and_excite"))

    def apply(inputs):
        x = layers.GlobalAveragePooling2D(
            name=name + "_squeeze_and_excite_gap", keepdims=True
        )(inputs)
        x = layers.Conv2D(
            se_filters,
            (1, 1),
            activation="relu",
            kernel_initializer="he_normal",
            name=name + "_squeeze_and_excite_squeeze",
        )(x)
        x = layers.Conv2D(
            filters_in,
            (1, 1),
            activation="sigmoid",
            kernel_initializer="he_normal",
            name=name + "_squeeze_and_excite_excite",
        )(x)
        x = tf.math.multiply(x, inputs)
        return x

    return apply


def XBlock(filters_in, filters_out, group_width, stride=1, name=None):
    """Implementation of X Block.

    Reference: [Designing Network Design
    Spaces](https://arxiv.org/abs/2003.13678)
    Args:
      filters_in: filters in the input tensor
      filters_out: filters in the output tensor
      group_width: group width
      stride: stride
      name: name prefix
    Returns:
      Output tensor of the block
    """
    if name is None:
        name = str(backend.get_uid("xblock"))

    def apply(inputs):
        if filters_in != filters_out and stride == 1:
            raise ValueError(
                f"Input filters({filters_in}) and output filters({filters_out}) "
                f"are not equal for stride {stride}. Input and output filters must "
                f"be equal for stride={stride}."
            )

        # Declare layers
        groups = filters_out // group_width

        if stride != 1:
            skip = layers.Conv2D(
                filters_out,
                (1, 1),
                strides=stride,
                use_bias=False,
                kernel_initializer="he_normal",
                name=name + "_skip_1x1",
            )(inputs)
            skip = layers.BatchNormalization(
                momentum=0.9, epsilon=1e-5, name=name + "_skip_bn"
            )(skip)
        else:
            skip = inputs

        # Build block
        # conv_1x1_1
        x = layers.Conv2D(
            filters_out,
            (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            name=name + "_conv_1x1_1",
        )(inputs)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_1x1_1_bn"
        )(x)
        x = layers.ReLU(name=name + "_conv_1x1_1_relu")(x)

        # conv_3x3
        x = layers.Conv2D(
            filters_out,
            (3, 3),
            use_bias=False,
            strides=stride,
            groups=groups,
            padding="same",
            kernel_initializer="he_normal",
            name=name + "_conv_3x3",
        )(x)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_3x3_bn"
        )(x)
        x = layers.ReLU(name=name + "_conv_3x3_relu")(x)

        # conv_1x1_2
        x = layers.Conv2D(
            filters_out,
            (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            name=name + "_conv_1x1_2",
        )(x)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_1x1_2_bn"
        )(x)

        x = layers.ReLU(name=name + "_exit_relu")(x + skip)

        return x

    return apply


def YBlock(
    filters_in,
    filters_out,
    group_width,
    stride=1,
    squeeze_excite_ratio=0.25,
    name=None,
):
    """Implementation of Y Block.

    Reference: [Designing Network Design
    Spaces](https://arxiv.org/abs/2003.13678)
    Args:
      filters_in: filters in the input tensor
      filters_out: filters in the output tensor
      group_width: group width
      stride: stride
      squeeze_excite_ratio: expansion ration for Squeeze and Excite block
      name: name prefix
    Returns:
      Output tensor of the block
    """
    if name is None:
        name = str(backend.get_uid("yblock"))

    def apply(inputs):
        if filters_in != filters_out and stride == 1:
            raise ValueError(
                f"Input filters({filters_in}) and output filters({filters_out}) "
                f"are not equal for stride {stride}. Input and output filters must  "
                f"be equal for stride={stride}."
            )

        groups = filters_out // group_width
        se_filters = int(filters_in * squeeze_excite_ratio)

        if stride != 1:
            skip = layers.Conv2D(
                filters_out,
                (1, 1),
                strides=stride,
                use_bias=False,
                kernel_initializer="he_normal",
                name=name + "_skip_1x1",
            )(inputs)
            skip = layers.BatchNormalization(
                momentum=0.9, epsilon=1e-5, name=name + "_skip_bn"
            )(skip)
        else:
            skip = inputs

        # Build block
        # conv_1x1_1
        x = layers.Conv2D(
            filters_out,
            (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            name=name + "_conv_1x1_1",
        )(inputs)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_1x1_1_bn"
        )(x)
        x = layers.ReLU(name=name + "_conv_1x1_1_relu")(x)

        # conv_3x3
        x = layers.Conv2D(
            filters_out,
            (3, 3),
            use_bias=False,
            strides=stride,
            groups=groups,
            padding="same",
            kernel_initializer="he_normal",
            name=name + "_conv_3x3",
        )(x)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_3x3_bn"
        )(x)
        x = layers.ReLU(name=name + "_conv_3x3_relu")(x)

        # Squeeze-Excitation block
        x = SqueezeAndExciteBlock(filters_out, se_filters, name=name)(x)

        # conv_1x1_2
        x = layers.Conv2D(
            filters_out,
            (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            name=name + "_conv_1x1_2",
        )(x)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_1x1_2_bn"
        )(x)

        x = layers.ReLU(name=name + "_exit_relu")(x + skip)

        return x

    return apply


def ZBlock(
    filters_in,
    filters_out,
    group_width,
    stride=1,
    squeeze_excite_ratio=0.25,
    bottleneck_ratio=0.25,
    name=None,
):
    """Implementation of Z block Reference: [Fast and Accurate Model Scaling](https://arxiv.org/abs/2103.06877).

    Args:
      filters_in: filters in the input tensor
      filters_out: filters in the output tensor
      group_width: group width
      stride: stride
      squeeze_excite_ratio: expansion ration for Squeeze and Excite block
      bottleneck_ratio: inverted bottleneck ratio
      name: name prefix
    Returns:
      Output tensor of the block
    """
    if name is None:
        name = str(backend.get_uid("zblock"))

    def apply(inputs):
        if filters_in != filters_out and stride == 1:
            raise ValueError(
                f"Input filters({filters_in}) and output filters({filters_out})"
                f"are not equal for stride {stride}. Input and output filters must be"
                f" equal for stride={stride}."
            )

        groups = filters_out // group_width
        se_filters = int(filters_in * squeeze_excite_ratio)

        inv_btlneck_filters = int(filters_out / bottleneck_ratio)

        # Build block
        # conv_1x1_1
        x = layers.Conv2D(
            inv_btlneck_filters,
            (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            name=name + "_conv_1x1_1",
        )(inputs)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_1x1_1_bn"
        )(x)
        x = tf.nn.silu(x)

        # conv_3x3
        x = layers.Conv2D(
            inv_btlneck_filters,
            (3, 3),
            use_bias=False,
            strides=stride,
            groups=groups,
            padding="same",
            kernel_initializer="he_normal",
            name=name + "_conv_3x3",
        )(x)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_3x3_bn"
        )(x)
        x = tf.nn.silu(x)

        # Squeeze-Excitation block
        x = SqueezeAndExciteBlock(inv_btlneck_filters, se_filters, name=name)

        # conv_1x1_2
        x = layers.Conv2D(
            filters_out,
            (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            name=name + "_conv_1x1_2",
        )(x)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_1x1_2_bn"
        )(x)

        if stride != 1:
            return x
        else:
            return x + inputs

    return apply


def Stage(block_type, depth, group_width, filters_in, filters_out, name=None):
    """Implementation of Stage in RegNet.

    Args:
      block_type: must be one of "X", "Y", "Z"
      depth: depth of stage, number of blocks to use
      group_width: group width of all blocks in  this stage
      filters_in: input filters to this stage
      filters_out: output filters from this stage
      name: name prefix

    Returns:
      Output tensor of Stage
    """
    if name is None:
        name = str(backend.get_uid("stage"))

    def apply(inputs):
        x = inputs
        if block_type == "X":
            x = XBlock(
                filters_in,
                filters_out,
                group_width,
                stride=2,
                name=f"{name}_XBlock_0",
            )(x)
            for i in range(1, depth):
                x = XBlock(
                    filters_out,
                    filters_out,
                    group_width,
                    name=f"{name}_XBlock_{i}",
                )(x)
        elif block_type == "Y":
            x = YBlock(
                filters_in,
                filters_out,
                group_width,
                stride=2,
                name=name + "_YBlock_0",
            )(x)
            for i in range(1, depth):
                x = YBlock(
                    filters_out,
                    filters_out,
                    group_width,
                    name=f"{name}_YBlock_{i}",
                )(x)
        elif block_type == "Z":
            x = ZBlock(
                filters_in,
                filters_out,
                group_width,
                stride=2,
                name=f"{name}_ZBlock_0",
            )(x)
            for i in range(1, depth):
                x = ZBlock(
                    filters_out,
                    filters_out,
                    group_width,
                    name=f"{name}_ZBlock_{i}",
                )(x)
        else:
            raise NotImplementedError(
                f"Block type `{block_type}` not recognized."
                f"block_type must be one of (`X`, `Y`, `Z`). "
            )
        return x

    return apply


def Head(num_classes=1000, name=None):
    """Implementation of classification head of RegNet.

    Args:
      num_classes: number of classes for Dense layer
      name: name prefix

    Returns:
      Output logits tensor.
    """
    if name is None:
        name = str(backend.get_uid("head"))

    def apply(x):
        x = layers.GlobalAveragePooling2D(name=name + "_head_gap")(x)
        x = layers.Dense(num_classes, name=name + "head_dense")(x)
        return x

    return apply


def RegNet(
    depths,
    widths,
    group_width,
    block_type,
    default_size,
    model_name="regnet",
    include_preprocessing=True,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """Instantiates RegNet architecture given specific configuration.

    Args:
      depths: An iterable containing depths for each individual stages.
      widths: An iterable containing output channel width of each individual
        stages
      group_width: Number of channels to be used in each group. See grouped
        convolutions for more information.
      block_type: Must be one of `{"X", "Y", "Z"}`. For more details see the
        papers "Designing network design spaces" and "Fast and Accurate Model
        Scaling"
      default_size: Default input image size.
      model_name: An optional name for the model.
      include_preprocessing: boolean denoting whther to include preprocessing in
        the model
      include_top: Boolean denoting whether to include classification head to the
        model.
      weights: one of `None` (random initialization), "imagenet" (pre-training on
        ImageNet), or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use
        as image input for the model.
      input_shape: optional shape tuple, only to be specified if `include_top` is
        False. It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction when `include_top` is
        `False`. - `None` means that the output of the model will be the 4D tensor
        output of the last convolutional layer. - `avg` means that global average
        pooling will be applied to the output of the last convolutional layer, and
        thus the output of the model will be a 2D tensor. - `max` means that
        global max pooling will be applied.
      classes: optional number of classes to classify images into, only to be
        specified if `include_top` is True, and if no `weights` argument is
        specified.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
      A `keras.Model` instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
          or invalid input shape.
        ValueError: if `classifier_activation` is not `softmax` or `None` when
          using a pretrained top layer.
        ValueError: if `include_top` is True but `num_classes` is not 1000.
        ValueError: if `block_type` is not one of `{"X", "Y", "Z"}`

    """
    if not (weights in {"imagenet", None} or tf.io.gfile.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded."
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            "If using `weights` as `'imagenet'` with `include_top`"
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

    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    x = inputs
    if include_preprocessing:
        x = PreStem(name=model_name)(x)
    x = Stem(name=model_name)(x)

    in_channels = 32  # Output from Stem

    for num_stage in range(4):
        depth = depths[num_stage]
        out_channels = widths[num_stage]

        x = Stage(
            block_type,
            depth,
            group_width,
            in_channels,
            out_channels,
            name=model_name + "_Stage_" + str(num_stage),
        )(x)
        in_channels = out_channels

    if include_top:
        x = Head(num_classes=classes)(x)
        imagenet_utils.validate_activation(classifier_activation, weights)

    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)

    model = training.Model(inputs=inputs, outputs=x, name=model_name)

    # Load weights.
    if weights == "imagenet":
        if include_top:
            file_suffix = ".h5"
            file_hash = WEIGHTS_HASHES[model_name[-4:]][0]
        else:
            file_suffix = "_notop.h5"
            file_hash = WEIGHTS_HASHES[model_name[-4:]][1]
        file_name = model_name + file_suffix
        weights_path = data_utils.get_file(
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
    "keras.applications.regnet.RegNetX002", "keras.applications.RegNetX002"
)
def RegNetX002(
    model_name="regnetx002",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["x002"]["depths"],
        MODEL_CONFIGS["x002"]["widths"],
        MODEL_CONFIGS["x002"]["group_width"],
        MODEL_CONFIGS["x002"]["block_type"],
        MODEL_CONFIGS["x002"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetX004", "keras.applications.RegNetX004"
)
def RegNetX004(
    model_name="regnetx004",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["x004"]["depths"],
        MODEL_CONFIGS["x004"]["widths"],
        MODEL_CONFIGS["x004"]["group_width"],
        MODEL_CONFIGS["x004"]["block_type"],
        MODEL_CONFIGS["x004"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetX006", "keras.applications.RegNetX006"
)
def RegNetX006(
    model_name="regnetx006",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["x006"]["depths"],
        MODEL_CONFIGS["x006"]["widths"],
        MODEL_CONFIGS["x006"]["group_width"],
        MODEL_CONFIGS["x006"]["block_type"],
        MODEL_CONFIGS["x006"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetX008", "keras.applications.RegNetX008"
)
def RegNetX008(
    model_name="regnetx008",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["x008"]["depths"],
        MODEL_CONFIGS["x008"]["widths"],
        MODEL_CONFIGS["x008"]["group_width"],
        MODEL_CONFIGS["x008"]["block_type"],
        MODEL_CONFIGS["x008"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetX016", "keras.applications.RegNetX016"
)
def RegNetX016(
    model_name="regnetx016",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["x016"]["depths"],
        MODEL_CONFIGS["x016"]["widths"],
        MODEL_CONFIGS["x016"]["group_width"],
        MODEL_CONFIGS["x016"]["block_type"],
        MODEL_CONFIGS["x016"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetX032", "keras.applications.RegNetX032"
)
def RegNetX032(
    model_name="regnetx032",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["x032"]["depths"],
        MODEL_CONFIGS["x032"]["widths"],
        MODEL_CONFIGS["x032"]["group_width"],
        MODEL_CONFIGS["x032"]["block_type"],
        MODEL_CONFIGS["x032"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetX040", "keras.applications.RegNetX040"
)
def RegNetX040(
    model_name="regnetx040",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["x040"]["depths"],
        MODEL_CONFIGS["x040"]["widths"],
        MODEL_CONFIGS["x040"]["group_width"],
        MODEL_CONFIGS["x040"]["block_type"],
        MODEL_CONFIGS["x040"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetX064", "keras.applications.RegNetX064"
)
def RegNetX064(
    model_name="regnetx064",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["x064"]["depths"],
        MODEL_CONFIGS["x064"]["widths"],
        MODEL_CONFIGS["x064"]["group_width"],
        MODEL_CONFIGS["x064"]["block_type"],
        MODEL_CONFIGS["x064"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetX080", "keras.applications.RegNetX080"
)
def RegNetX080(
    model_name="regnetx080",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["x080"]["depths"],
        MODEL_CONFIGS["x080"]["widths"],
        MODEL_CONFIGS["x080"]["group_width"],
        MODEL_CONFIGS["x080"]["block_type"],
        MODEL_CONFIGS["x080"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetX120", "keras.applications.RegNetX120"
)
def RegNetX120(
    model_name="regnetx120",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["x120"]["depths"],
        MODEL_CONFIGS["x120"]["widths"],
        MODEL_CONFIGS["x120"]["group_width"],
        MODEL_CONFIGS["x120"]["block_type"],
        MODEL_CONFIGS["x120"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetX160", "keras.applications.RegNetX160"
)
def RegNetX160(
    model_name="regnetx160",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["x160"]["depths"],
        MODEL_CONFIGS["x160"]["widths"],
        MODEL_CONFIGS["x160"]["group_width"],
        MODEL_CONFIGS["x160"]["block_type"],
        MODEL_CONFIGS["x160"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetX320", "keras.applications.RegNetX320"
)
def RegNetX320(
    model_name="regnetx320",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["x320"]["depths"],
        MODEL_CONFIGS["x320"]["widths"],
        MODEL_CONFIGS["x320"]["group_width"],
        MODEL_CONFIGS["x320"]["block_type"],
        MODEL_CONFIGS["x320"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetY002", "keras.applications.RegNetY002"
)
def RegNetY002(
    model_name="regnety002",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["y002"]["depths"],
        MODEL_CONFIGS["y002"]["widths"],
        MODEL_CONFIGS["y002"]["group_width"],
        MODEL_CONFIGS["y002"]["block_type"],
        MODEL_CONFIGS["y002"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetY004", "keras.applications.RegNetY004"
)
def RegNetY004(
    model_name="regnety004",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["y004"]["depths"],
        MODEL_CONFIGS["y004"]["widths"],
        MODEL_CONFIGS["y004"]["group_width"],
        MODEL_CONFIGS["y004"]["block_type"],
        MODEL_CONFIGS["y004"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetY006", "keras.applications.RegNetY006"
)
def RegNetY006(
    model_name="regnety006",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["y006"]["depths"],
        MODEL_CONFIGS["y006"]["widths"],
        MODEL_CONFIGS["y006"]["group_width"],
        MODEL_CONFIGS["y006"]["block_type"],
        MODEL_CONFIGS["y006"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetY008", "keras.applications.RegNetY008"
)
def RegNetY008(
    model_name="regnety008",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["y008"]["depths"],
        MODEL_CONFIGS["y008"]["widths"],
        MODEL_CONFIGS["y008"]["group_width"],
        MODEL_CONFIGS["y008"]["block_type"],
        MODEL_CONFIGS["y008"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetY016", "keras.applications.RegNetY016"
)
def RegNetY016(
    model_name="regnety016",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["y016"]["depths"],
        MODEL_CONFIGS["y016"]["widths"],
        MODEL_CONFIGS["y016"]["group_width"],
        MODEL_CONFIGS["y016"]["block_type"],
        MODEL_CONFIGS["y016"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetY032", "keras.applications.RegNetY032"
)
def RegNetY032(
    model_name="regnety032",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["y032"]["depths"],
        MODEL_CONFIGS["y032"]["widths"],
        MODEL_CONFIGS["y032"]["group_width"],
        MODEL_CONFIGS["y032"]["block_type"],
        MODEL_CONFIGS["y032"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetY040", "keras.applications.RegNetY040"
)
def RegNetY040(
    model_name="regnety040",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["y040"]["depths"],
        MODEL_CONFIGS["y040"]["widths"],
        MODEL_CONFIGS["y040"]["group_width"],
        MODEL_CONFIGS["y040"]["block_type"],
        MODEL_CONFIGS["y040"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetY064", "keras.applications.RegNetY064"
)
def RegNetY064(
    model_name="regnety064",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["y064"]["depths"],
        MODEL_CONFIGS["y064"]["widths"],
        MODEL_CONFIGS["y064"]["group_width"],
        MODEL_CONFIGS["y064"]["block_type"],
        MODEL_CONFIGS["y064"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetY080", "keras.applications.RegNetY080"
)
def RegNetY080(
    model_name="regnety080",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["y080"]["depths"],
        MODEL_CONFIGS["y080"]["widths"],
        MODEL_CONFIGS["y080"]["group_width"],
        MODEL_CONFIGS["y080"]["block_type"],
        MODEL_CONFIGS["y080"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetY120", "keras.applications.RegNetY120"
)
def RegNetY120(
    model_name="regnety120",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["y120"]["depths"],
        MODEL_CONFIGS["y120"]["widths"],
        MODEL_CONFIGS["y120"]["group_width"],
        MODEL_CONFIGS["y120"]["block_type"],
        MODEL_CONFIGS["y120"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetY160", "keras.applications.RegNetY160"
)
def RegNetY160(
    model_name="regnety160",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["y160"]["depths"],
        MODEL_CONFIGS["y160"]["widths"],
        MODEL_CONFIGS["y160"]["group_width"],
        MODEL_CONFIGS["y160"]["block_type"],
        MODEL_CONFIGS["y160"]["default_size"],
        model_name=model_name,
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
    "keras.applications.regnet.RegNetY320", "keras.applications.RegNetY320"
)
def RegNetY320(
    model_name="regnety320",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["y320"]["depths"],
        MODEL_CONFIGS["y320"]["widths"],
        MODEL_CONFIGS["y320"]["group_width"],
        MODEL_CONFIGS["y320"]["block_type"],
        MODEL_CONFIGS["y320"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


RegNetX002.__doc__ = BASE_DOCSTRING.format(name="RegNetX002")
RegNetX004.__doc__ = BASE_DOCSTRING.format(name="RegNetX004")
RegNetX006.__doc__ = BASE_DOCSTRING.format(name="RegNetX006")
RegNetX008.__doc__ = BASE_DOCSTRING.format(name="RegNetX008")
RegNetX016.__doc__ = BASE_DOCSTRING.format(name="RegNetX016")
RegNetX032.__doc__ = BASE_DOCSTRING.format(name="RegNetX032")
RegNetX040.__doc__ = BASE_DOCSTRING.format(name="RegNetX040")
RegNetX064.__doc__ = BASE_DOCSTRING.format(name="RegNetX064")
RegNetX080.__doc__ = BASE_DOCSTRING.format(name="RegNetX080")
RegNetX120.__doc__ = BASE_DOCSTRING.format(name="RegNetX120")
RegNetX160.__doc__ = BASE_DOCSTRING.format(name="RegNetX160")
RegNetX320.__doc__ = BASE_DOCSTRING.format(name="RegNetX320")

RegNetY002.__doc__ = BASE_DOCSTRING.format(name="RegNetY002")
RegNetY004.__doc__ = BASE_DOCSTRING.format(name="RegNetY004")
RegNetY006.__doc__ = BASE_DOCSTRING.format(name="RegNetY006")
RegNetY008.__doc__ = BASE_DOCSTRING.format(name="RegNetY008")
RegNetY016.__doc__ = BASE_DOCSTRING.format(name="RegNetY016")
RegNetY032.__doc__ = BASE_DOCSTRING.format(name="RegNetY032")
RegNetY040.__doc__ = BASE_DOCSTRING.format(name="RegNetY040")
RegNetY064.__doc__ = BASE_DOCSTRING.format(name="RegNetY064")
RegNetY080.__doc__ = BASE_DOCSTRING.format(name="RegNetY080")
RegNetY120.__doc__ = BASE_DOCSTRING.format(name="RegNetY120")
RegNetY160.__doc__ = BASE_DOCSTRING.format(name="RegNetY160")
RegNetY320.__doc__ = BASE_DOCSTRING.format(name="RegNetY320")


@keras_export("keras.applications.regnet.preprocess_input")
def preprocess_input(x, data_format=None):  # pylint: disable=unused-argument
    """A placeholder method for backward compatibility.

    The preprocessing logic has been included in the regnet model
    implementation. Users are no longer required to call this method to normalize
    the input data. This method does nothing and only kept as a placeholder to
    align the API surface between old and new version of model.

    Args:
      x: A floating point `numpy.array` or a `tf.Tensor`.
      data_format: Optional data format of the image tensor/array. Defaults to
        None, in which case the global setting
        `tf.keras.backend.image_data_format()` is used (unless you changed it, it
        defaults to "channels_last").{mode}

    Returns:
      Unchanged `numpy.array` or `tf.Tensor`.
    """
    return x


@keras_export("keras.applications.regnet.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__
