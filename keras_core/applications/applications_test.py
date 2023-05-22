import numpy as np
import pytest
from absl.testing import parameterized

from keras_core import backend
from keras_core import testing
from keras_core.applications import efficientnet
from keras_core.applications import efficientnet_v2
from keras_core.applications import mobilenet
from keras_core.applications import mobilenet_v2
from keras_core.applications import mobilenet_v3
from keras_core.applications import vgg16
from keras_core.applications import vgg19
from keras_core.applications import xception
from keras_core.utils import file_utils
from keras_core.utils import image_utils

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
    # mobilnet
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
]
# Add names for `named_parameters`.
MODEL_LIST = [(e[0].__name__, *e) for e in MODEL_LIST]


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


class ApplicationsTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(MODEL_LIST)
    def test_application_notop_variable_input_channels(self, app, last_dim, _):
        # Test compatibility with 1 channel
        if backend.image_data_format() == "channels_first":
            input_shape = (1, None, None)
        else:
            input_shape = (None, None, 1)
        model = app(weights=None, include_top=False, input_shape=input_shape)
        output_shape = list(model.outputs[0].shape)
        self.assertEqual(output_shape, [None, None, None, last_dim])

        # Test compatibility with 4 channels
        if backend.image_data_format() == "channels_first":
            input_shape = (4, None, None)
        else:
            input_shape = (None, None, 4)
        model = app(weights=None, include_top=False, input_shape=input_shape)
        output_shape = list(model.outputs[0].shape)
        self.assertEqual(output_shape, [None, None, None, last_dim])

    @parameterized.named_parameters(MODEL_LIST)
    @pytest.mark.skipif(PIL is None, reason="Requires PIL.")
    def test_application_base(self, app, _, app_module):
        # Can be instantiated with default arguments
        model = app(weights="imagenet")

        # Can run a correct inference on a test image
        x = _get_elephant(model.input_shape[1:3])
        x = app_module.preprocess_input(x)
        preds = model.predict(x)
        names = [p[1] for p in app_module.decode_predictions(preds)[0]]
        # Test correct label is in top 3 (weak correctness test).
        self.assertIn("African_elephant", names[:3])

        # Can be serialized and deserialized
        config = model.get_config()
        reconstructed_model = model.__class__.from_config(config)
        self.assertEqual(len(model.weights), len(reconstructed_model.weights))

    @parameterized.named_parameters(MODEL_LIST)
    def test_application_notop_custom_input_shape(self, app, last_dim, _):
        model = app(weights=None, include_top=False, input_shape=(123, 123, 3))
        output_shape = list(model.outputs[0].shape)
        self.assertEqual(output_shape[-1], last_dim)

    @parameterized.named_parameters(MODEL_LIST)
    def test_application_pooling(self, app, last_dim, _):
        model = app(weights=None, include_top=False, pooling="max")
        output_shape = list(model.outputs[0].shape)
        self.assertEqual(output_shape, [None, last_dim])

    @parameterized.named_parameters(MODEL_LIST)
    def test_application_classifier_activation(self, app, *_):
        model = app(
            weights=None, include_top=True, classifier_activation="softmax"
        )
        last_layer_act = model.layers[-1].activation.__name__
        self.assertEqual(last_layer_act, "softmax")
