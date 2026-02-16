"""Tests for LiteRT export of keras.applications models via PyTorch backend.

Tests one representative model from each family to verify export pipeline
works across diverse architectures.
"""

import os
import tempfile

import numpy as np
import pytest

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
from keras.src.applications import resnet
from keras.src.applications import resnet_v2
from keras.src.applications import vgg16
from keras.src.applications import vgg19
from keras.src.applications import xception


def _requires_litert_torch():
    """Skip helper — call at the top of every test."""
    try:
        import litert_torch  # noqa: F401
        import torch  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        pytest.skip("litert-torch or torch not installed")


def _get_interpreter(filepath):
    """Return an allocated LiteRT interpreter for the given file."""
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=filepath)
    interp.allocate_tensors()
    return interp


def _run_litert_inference(interpreter, input_array):
    """Feed a numpy array through the interpreter."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_array)
    interpreter.invoke()

    outputs = [interpreter.get_tensor(d["index"]) for d in output_details]
    return outputs[0] if len(outputs) == 1 else outputs


def _to_numpy(x):
    """Convert a Keras / torch tensor to numpy."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Requires torch backend"
)
class ApplicationsLiteRTTorchExportTest(testing.TestCase):
    """Test LiteRT export for keras.applications models via torch backend.

    Each test uses a representative model from a different family,
    with no pretrained weights for speed.
    """

    def test_vgg16_export(self):
        _requires_litert_torch()
        model = vgg16.VGG16(
            weights=None, include_top=True, input_shape=(224, 224, 3)
        )
        x = np.random.normal(size=(1, 224, 224, 3)).astype("float32")
        self._verify_litert_export(model, x)

    def test_vgg19_export(self):
        _requires_litert_torch()
        model = vgg19.VGG19(
            weights=None, include_top=True, input_shape=(224, 224, 3)
        )
        x = np.random.normal(size=(1, 224, 224, 3)).astype("float32")
        self._verify_litert_export(model, x)

    def test_resnet50_export(self):
        _requires_litert_torch()
        model = resnet.ResNet50(
            weights=None, include_top=True, input_shape=(224, 224, 3)
        )
        x = np.random.normal(size=(1, 224, 224, 3)).astype("float32")
        self._verify_litert_export(model, x)

    def test_resnet50v2_export(self):
        _requires_litert_torch()
        model = resnet_v2.ResNet50V2(
            weights=None, include_top=True, input_shape=(224, 224, 3)
        )
        x = np.random.normal(size=(1, 224, 224, 3)).astype("float32")
        self._verify_litert_export(model, x)

    def test_inceptionv3_export(self):
        _requires_litert_torch()
        model = inception_v3.InceptionV3(
            weights=None, include_top=True, input_shape=(299, 299, 3)
        )
        x = np.random.normal(size=(1, 299, 299, 3)).astype("float32")
        self._verify_litert_export(model, x)

    def test_inceptionresnetv2_export(self):
        _requires_litert_torch()
        model = inception_resnet_v2.InceptionResNetV2(
            weights=None, include_top=True, input_shape=(299, 299, 3)
        )
        x = np.random.normal(size=(1, 299, 299, 3)).astype("float32")
        self._verify_litert_export(model, x, atol=1e-2)

    def test_xception_export(self):
        _requires_litert_torch()
        model = xception.Xception(
            weights=None, include_top=True, input_shape=(299, 299, 3)
        )
        x = np.random.normal(size=(1, 299, 299, 3)).astype("float32")
        self._verify_litert_export(model, x)

    def test_mobilenet_export(self):
        _requires_litert_torch()
        model = mobilenet.MobileNet(
            weights=None, include_top=True, input_shape=(224, 224, 3)
        )
        x = np.random.normal(size=(1, 224, 224, 3)).astype("float32")
        self._verify_litert_export(model, x)

    def test_mobilenetv2_export(self):
        _requires_litert_torch()
        model = mobilenet_v2.MobileNetV2(
            weights=None, include_top=True, input_shape=(224, 224, 3)
        )
        x = np.random.normal(size=(1, 224, 224, 3)).astype("float32")
        self._verify_litert_export(model, x)

    def test_mobilenetv3small_export(self):
        _requires_litert_torch()
        model = mobilenet_v3.MobileNetV3Small(
            weights=None, include_top=True, input_shape=(224, 224, 3)
        )
        x = np.random.normal(size=(1, 224, 224, 3)).astype("float32")
        self._verify_litert_export(model, x)

    def test_mobilenetv3large_export(self):
        _requires_litert_torch()
        model = mobilenet_v3.MobileNetV3Large(
            weights=None, include_top=True, input_shape=(224, 224, 3)
        )
        x = np.random.normal(size=(1, 224, 224, 3)).astype("float32")
        self._verify_litert_export(model, x)

    def test_densenet121_export(self):
        _requires_litert_torch()
        model = densenet.DenseNet121(
            weights=None, include_top=True, input_shape=(224, 224, 3)
        )
        x = np.random.normal(size=(1, 224, 224, 3)).astype("float32")
        self._verify_litert_export(model, x)

    def test_convnexttiny_export(self):
        _requires_litert_torch()
        model = convnext.ConvNeXtTiny(
            weights=None, include_top=True, input_shape=(224, 224, 3)
        )
        x = np.random.normal(size=(1, 224, 224, 3)).astype("float32")
        self._verify_litert_export(model, x)

    def test_efficientnetb0_export(self):
        _requires_litert_torch()
        model = efficientnet.EfficientNetB0(
            weights=None, include_top=True, input_shape=(224, 224, 3)
        )
        x = np.random.normal(size=(1, 224, 224, 3)).astype("float32")
        self._verify_litert_export(model, x)

    def test_efficientnetv2b0_export(self):
        _requires_litert_torch()
        model = efficientnet_v2.EfficientNetV2B0(
            weights=None, include_top=True, input_shape=(224, 224, 3)
        )
        x = np.random.normal(size=(1, 224, 224, 3)).astype("float32")
        self._verify_litert_export(model, x)

    def _verify_litert_export(self, model, input_array, atol=1e-3):
        """Helper: export model to LiteRT and verify numeric parity."""

        # Get Keras reference output
        keras_out = _to_numpy(model(input_array))

        # Export and run inference
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = os.path.join(tmpdir, "model.tflite")
            model.export(tflite_path, format="litert")
            self.assertTrue(os.path.exists(tflite_path))

            interp = _get_interpreter(tflite_path)
            litert_out = _run_litert_inference(interp, input_array)

            # Compare outputs
            self.assertAllClose(keras_out, litert_out, atol=atol)
