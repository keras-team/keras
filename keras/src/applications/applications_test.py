import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.applications import convnext
from keras.src.applications import densenet
from keras.src.applications import efficientnet
from keras.src.applications import efficientnet_v2
from keras.src.applications import inception_resnet_v2
from keras.src.applications import inception_v3
from keras.src.applications import mobilenet
from keras.src.applications import mobilenet_v2
from keras.src.applications import mobilenet_v3
from keras.src.applications import nasnet
from keras.src.applications import resnet
from keras.src.applications import resnet_v2
from keras.src.applications import vgg16
from keras.src.applications import vgg19
from keras.src.applications import xception
from keras.src.saving import serialization_lib
from keras.src.utils import file_utils
from keras.src.utils import image_utils

try:
    import PIL
except ImportError:
    PIL = None

MODEL_LIST = [
    # vgg
    (vgg16.VGG16, 512, vgg16),
    (vgg19.VGG19, 512, vgg19),
    # xception
    (xception.Xception, 2048, xception),
    # inception
    (inception_v3.InceptionV3, 2048, inception_v3),
    (inception_resnet_v2.InceptionResNetV2, 1536, inception_resnet_v2),
    # mobilenet
    (mobilenet.MobileNet, 1024, mobilenet),
    (mobilenet_v2.MobileNetV2, 1280, mobilenet_v2),
    (mobilenet_v3.MobileNetV3Small, 576, mobilenet_v3),
    (mobilenet_v3.MobileNetV3Large, 960, mobilenet_v3),
    # efficientnet
    (efficientnet.EfficientNetB0, 1280, efficientnet),
    (efficientnet.EfficientNetB1, 1280, efficientnet),
    (efficientnet.EfficientNetB2, 1408, efficientnet),
    (efficientnet.EfficientNetB3, 1536, efficientnet),
    (efficientnet.EfficientNetB4, 1792, efficientnet),
    (efficientnet.EfficientNetB5, 2048, efficientnet),
    (efficientnet.EfficientNetB6, 2304, efficientnet),
    (efficientnet.EfficientNetB7, 2560, efficientnet),
    (efficientnet_v2.EfficientNetV2B0, 1280, efficientnet_v2),
    (efficientnet_v2.EfficientNetV2B1, 1280, efficientnet_v2),
    (efficientnet_v2.EfficientNetV2B2, 1408, efficientnet_v2),
    (efficientnet_v2.EfficientNetV2B3, 1536, efficientnet_v2),
    (efficientnet_v2.EfficientNetV2S, 1280, efficientnet_v2),
    (efficientnet_v2.EfficientNetV2M, 1280, efficientnet_v2),
    (efficientnet_v2.EfficientNetV2L, 1280, efficientnet_v2),
    # densenet
    (densenet.DenseNet121, 1024, densenet),
    (densenet.DenseNet169, 1664, densenet),
    (densenet.DenseNet201, 1920, densenet),
    # convnext
    (convnext.ConvNeXtTiny, 768, convnext),
    (convnext.ConvNeXtSmall, 768, convnext),
    (convnext.ConvNeXtBase, 1024, convnext),
    (convnext.ConvNeXtLarge, 1536, convnext),
    (convnext.ConvNeXtXLarge, 2048, convnext),
    # nasnet
    (nasnet.NASNetMobile, 1056, nasnet),
    (nasnet.NASNetLarge, 4032, nasnet),
    # resnet
    (resnet.ResNet50, 2048, resnet),
    (resnet.ResNet101, 2048, resnet),
    (resnet.ResNet152, 2048, resnet),
    (resnet_v2.ResNet50V2, 2048, resnet_v2),
    (resnet_v2.ResNet101V2, 2048, resnet_v2),
    (resnet_v2.ResNet152V2, 2048, resnet_v2),
]
MODELS_UNSUPPORTED_CHANNELS_FIRST = ["ConvNeXt", "DenseNet", "NASNet"]

# Add names for `named_parameters`, and add each data format for each model
test_parameters = [
    (
        "{}_{}".format(model[0].__name__, image_data_format),
        *model,
        image_data_format,
    )
    for image_data_format in ["channels_first", "channels_last"]
    for model in MODEL_LIST
]


def _get_elephant(target_size):
    # For models that don't include a Flatten step,
    # the default is to accept variable-size inputs
    # even when loading ImageNet weights (since it is possible).
    # In this case, default to 299x299.
    TEST_IMAGE_PATH = (
        "https://storage.googleapis.com/tensorflow/"
        "keras-applications/tests/elephant.jpg"
    )

    if target_size[0] is None:
        target_size = (299, 299)
    test_image = file_utils.get_file("elephant.jpg", TEST_IMAGE_PATH)
    img = image_utils.load_img(test_image, target_size=tuple(target_size))
    x = image_utils.img_to_array(img)
    return np.expand_dims(x, axis=0)


@pytest.mark.skipif(
    os.environ.get("SKIP_APPLICATIONS_TESTS"),
    reason="Env variable set to skip.",
)
@pytest.mark.requires_trainable_backend
class ApplicationsTest(testing.TestCase, parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_image_data_format = backend.image_data_format()

    @classmethod
    def tearDownClass(cls):
        backend.set_image_data_format(cls.original_image_data_format)

    def skip_if_invalid_image_data_format_for_model(
        self, app, image_data_format
    ):
        does_not_support_channels_first = any(
            [
                unsupported_name.lower() in app.__name__.lower()
                for unsupported_name in MODELS_UNSUPPORTED_CHANNELS_FIRST
            ]
        )
        if (
            image_data_format == "channels_first"
            and does_not_support_channels_first
        ):
            self.skipTest(
                "{} does not support channels first".format(app.__name__)
            )

    @parameterized.named_parameters(test_parameters)
    def test_application_notop_variable_input_channels(
        self, app, last_dim, _, image_data_format
    ):
        if app == nasnet.NASNetMobile and backend.backend() == "torch":
            self.skipTest(
                "NASNetMobile pretrained incorrect with torch backend."
            )
        self.skip_if_invalid_image_data_format_for_model(app, image_data_format)
        backend.set_image_data_format(image_data_format)

        # Test compatibility with 1 channel
        if image_data_format == "channels_first":
            input_shape = (1, None, None)
            correct_output_shape = [None, last_dim, None, None]
        else:
            input_shape = (None, None, 1)
            correct_output_shape = [None, None, None, last_dim]

        model = app(weights=None, include_top=False, input_shape=input_shape)
        output_shape = list(model.outputs[0].shape)
        self.assertEqual(output_shape, correct_output_shape)

        # Test compatibility with 4 channels
        if image_data_format == "channels_first":
            input_shape = (4, None, None)
        else:
            input_shape = (None, None, 4)
        model = app(weights=None, include_top=False, input_shape=input_shape)
        output_shape = list(model.outputs[0].shape)
        self.assertEqual(output_shape, correct_output_shape)

    @parameterized.named_parameters(test_parameters)
    @pytest.mark.skipif(PIL is None, reason="Requires PIL.")
    def test_application_base(self, app, _, app_module, image_data_format):
        import tensorflow as tf

        if app == nasnet.NASNetMobile and backend.backend() == "torch":
            self.skipTest(
                "NASNetMobile pretrained incorrect with torch backend."
            )
        if (
            image_data_format == "channels_first"
            and len(tf.config.list_physical_devices("GPU")) == 0
            and backend.backend() == "tensorflow"
        ):
            self.skipTest(
                "Conv2D doesn't support channels_first using CPU with "
                "tensorflow backend"
            )
        self.skip_if_invalid_image_data_format_for_model(app, image_data_format)
        backend.set_image_data_format(image_data_format)

        # Can be instantiated with default arguments
        model = app(weights="imagenet")

        # Can run a correct inference on a test image
        if image_data_format == "channels_first":
            shape = model.input_shape[2:4]
        else:
            shape = model.input_shape[1:3]
        x = _get_elephant(shape)

        x = app_module.preprocess_input(x)
        preds = model.predict(x)
        names = [p[1] for p in app_module.decode_predictions(preds)[0]]
        # Test correct label is in top 3 (weak correctness test).
        self.assertIn("African_elephant", names[:3])

        # Can be serialized and deserialized
        config = serialization_lib.serialize_keras_object(model)
        reconstructed_model = serialization_lib.deserialize_keras_object(config)
        self.assertEqual(len(model.weights), len(reconstructed_model.weights))

    @parameterized.named_parameters(test_parameters)
    def test_application_notop_custom_input_shape(
        self, app, last_dim, _, image_data_format
    ):
        if app == nasnet.NASNetMobile and backend.backend() == "torch":
            self.skipTest(
                "NASNetMobile pretrained incorrect with torch backend."
            )
        self.skip_if_invalid_image_data_format_for_model(app, image_data_format)
        backend.set_image_data_format(image_data_format)

        if image_data_format == "channels_first":
            input_shape = (3, 123, 123)
            last_dim_axis = 1
        else:
            input_shape = (123, 123, 3)
            last_dim_axis = -1
        model = app(weights=None, include_top=False, input_shape=input_shape)
        output_shape = list(model.outputs[0].shape)
        self.assertEqual(output_shape[last_dim_axis], last_dim)

    @parameterized.named_parameters(test_parameters)
    def test_application_pooling(self, app, last_dim, _, image_data_format):
        if app == nasnet.NASNetMobile and backend.backend() == "torch":
            self.skipTest(
                "NASNetMobile pretrained incorrect with torch backend."
            )
        self.skip_if_invalid_image_data_format_for_model(app, image_data_format)
        backend.set_image_data_format(image_data_format)

        model = app(weights=None, include_top=False, pooling="max")
        output_shape = list(model.outputs[0].shape)
        self.assertEqual(output_shape, [None, last_dim])

    @parameterized.named_parameters(test_parameters)
    def test_application_classifier_activation(self, app, *_):
        if app == nasnet.NASNetMobile and backend.backend() == "torch":
            self.skipTest(
                "NASNetMobile pretrained incorrect with torch backend."
            )

        model = app(
            weights=None, include_top=True, classifier_activation="softmax"
        )
        last_layer_act = model.layers[-1].activation.__name__
        self.assertEqual(last_layer_act, "softmax")
