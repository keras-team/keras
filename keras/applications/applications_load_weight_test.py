# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Integration tests for Keras applications."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl import flags
from absl.testing import parameterized

from keras.applications import convnext
from keras.applications import densenet
from keras.applications import efficientnet
from keras.applications import efficientnet_v2
from keras.applications import inception_resnet_v2
from keras.applications import inception_v3
from keras.applications import mobilenet
from keras.applications import mobilenet_v2
from keras.applications import mobilenet_v3
from keras.applications import nasnet
from keras.applications import regnet
from keras.applications import resnet
from keras.applications import resnet_rs
from keras.applications import resnet_v2
from keras.applications import vgg16
from keras.applications import vgg19
from keras.applications import xception
from keras.utils import data_utils
from keras.utils import image_utils

ARG_TO_MODEL = {
    "resnet": (resnet, [resnet.ResNet50, resnet.ResNet101, resnet.ResNet152]),
    "resnet_v2": (
        resnet_v2,
        [resnet_v2.ResNet50V2, resnet_v2.ResNet101V2, resnet_v2.ResNet152V2],
    ),
    "vgg16": (vgg16, [vgg16.VGG16]),
    "vgg19": (vgg19, [vgg19.VGG19]),
    "xception": (xception, [xception.Xception]),
    "inception_v3": (inception_v3, [inception_v3.InceptionV3]),
    "inception_resnet_v2": (
        inception_resnet_v2,
        [inception_resnet_v2.InceptionResNetV2],
    ),
    "mobilenet": (mobilenet, [mobilenet.MobileNet]),
    "mobilenet_v2": (mobilenet_v2, [mobilenet_v2.MobileNetV2]),
    "mobilenet_v3_small": (mobilenet_v3, [mobilenet_v3.MobileNetV3Small]),
    "mobilenet_v3_large": (mobilenet_v3, [mobilenet_v3.MobileNetV3Large]),
    "convnext": (
        convnext,
        [
            convnext.ConvNeXtTiny,
            convnext.ConvNeXtSmall,
            convnext.ConvNeXtBase,
            convnext.ConvNeXtLarge,
            convnext.ConvNeXtXLarge,
        ],
    ),
    "densenet": (
        densenet,
        [densenet.DenseNet121, densenet.DenseNet169, densenet.DenseNet201],
    ),
    "nasnet_mobile": (nasnet, [nasnet.NASNetMobile]),
    "nasnet_large": (nasnet, [nasnet.NASNetLarge]),
    "efficientnet": (
        efficientnet,
        [
            efficientnet.EfficientNetB0,
            efficientnet.EfficientNetB1,
            efficientnet.EfficientNetB2,
            efficientnet.EfficientNetB3,
            efficientnet.EfficientNetB4,
            efficientnet.EfficientNetB5,
            efficientnet.EfficientNetB6,
            efficientnet.EfficientNetB7,
        ],
    ),
    "efficientnet_v2": (
        efficientnet_v2,
        [
            efficientnet_v2.EfficientNetV2B0,
            efficientnet_v2.EfficientNetV2B1,
            efficientnet_v2.EfficientNetV2B2,
            efficientnet_v2.EfficientNetV2B3,
            efficientnet_v2.EfficientNetV2S,
            efficientnet_v2.EfficientNetV2M,
            efficientnet_v2.EfficientNetV2L,
        ],
    ),
    "resnet_rs": (
        resnet_rs,
        [
            resnet_rs.ResNetRS50,
            resnet_rs.ResNetRS101,
            resnet_rs.ResNetRS152,
            resnet_rs.ResNetRS200,
            resnet_rs.ResNetRS270,
            resnet_rs.ResNetRS350,
            resnet_rs.ResNetRS420,
        ],
    ),
    "regnet": (
        regnet,
        [
            regnet.RegNetX002,
            regnet.RegNetX004,
            regnet.RegNetX006,
            regnet.RegNetX008,
            regnet.RegNetX016,
            regnet.RegNetX032,
            regnet.RegNetX040,
            regnet.RegNetX064,
            regnet.RegNetX080,
            regnet.RegNetX120,
            regnet.RegNetX160,
            regnet.RegNetX320,
            regnet.RegNetY002,
            regnet.RegNetY004,
            regnet.RegNetY006,
            regnet.RegNetY008,
            regnet.RegNetY016,
            regnet.RegNetY032,
            regnet.RegNetY040,
            regnet.RegNetY064,
            regnet.RegNetY080,
            regnet.RegNetY120,
            regnet.RegNetY160,
            regnet.RegNetY320,
        ],
    ),
}

TEST_IMAGE_PATH = (
    "https://storage.googleapis.com/tensorflow/"
    "keras-applications/tests/elephant.jpg"
)
_IMAGENET_CLASSES = 1000

# Add a flag to define which application module file is tested.
# This is set as an 'arg' in the build target to guarantee that
# it only triggers the tests of the application models in the module
# if that module file has been modified.
FLAGS = flags.FLAGS
flags.DEFINE_string("module", None, "Application module used in this test.")


def _get_elephant(target_size):
    # For models that don't include a Flatten step,
    # the default is to accept variable-size inputs
    # even when loading ImageNet weights (since it is possible).
    # In this case, default to 299x299.
    if target_size[0] is None:
        target_size = (299, 299)
    test_image = data_utils.get_file("elephant.jpg", TEST_IMAGE_PATH)
    img = image_utils.load_img(test_image, target_size=tuple(target_size))
    x = image_utils.img_to_array(img)
    return np.expand_dims(x, axis=0)


class ApplicationsLoadWeightTest(tf.test.TestCase, parameterized.TestCase):
    def assertShapeEqual(self, shape1, shape2):
        if len(shape1) != len(shape2):
            raise AssertionError(
                f"Shapes are different rank: {shape1} vs {shape2}"
            )
        if shape1 != shape2:
            raise AssertionError(f"Shapes differ: {shape1} vs {shape2}")

    def test_application_pretrained_weights_loading(self):
        app_module = ARG_TO_MODEL[FLAGS.module][0]
        apps = ARG_TO_MODEL[FLAGS.module][1]
        for app in apps:
            try:
                model = app(weights="imagenet")
            except Exception:
                self.skipTest("TODO(b/227700184): Re-enable.")
            self.assertShapeEqual(model.output_shape, (None, _IMAGENET_CLASSES))
            x = _get_elephant(model.input_shape[1:3])
            x = app_module.preprocess_input(x)
            preds = model.predict(x)
            names = [p[1] for p in app_module.decode_predictions(preds)[0]]
            # Test correct label is in top 3 (weak correctness test).
            self.assertIn("African_elephant", names[:3])


if __name__ == "__main__":
    tf.test.main()
