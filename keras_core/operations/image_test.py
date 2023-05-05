import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_core import backend
from keras_core import testing
from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.operations import image as kimage


@pytest.mark.skipif(
    not backend.DYNAMIC_SHAPES_OK,
    reason="Backend does not support dynamic shapes",
)
class ImageOpsDynamicShapeTest(testing.TestCase):
    def test_resize(self):
        x = KerasTensor([None, 20, 20, 3])
        out = kimage.resize(x, size=(15, 15))
        self.assertEqual(out.shape, (None, 15, 15, 3))

        x = KerasTensor([None, None, 3])
        out = kimage.resize(x, size=(15, 15))
        self.assertEqual(out.shape, (15, 15, 3))


class ImageOpsStaticShapeTest(testing.TestCase):
    def test_resize(self):
        x = KerasTensor([20, 20, 3])
        out = kimage.resize(x, size=(15, 15))
        self.assertEqual(out.shape, (15, 15, 3))


class ImageOpsCorrectnessTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        [
            ("bilinear", True),
            ("nearest", True),
            ("lanczos3", True),
            ("lanczos5", True),
            ("bicubic", True),
            ("bilinear", False),
            ("nearest", False),
            ("lanczos3", False),
            ("lanczos5", False),
            ("bicubic", False),
        ]
    )
    def test_resize(self, method, antialias):
        x = np.random.random((50, 50, 3)) * 255
        out = kimage.resize(
            x, size=(25, 25), method=method, antialias=antialias
        )
        ref_out = tf.image.resize(
            x, size=(25, 25), method=method, antialias=antialias
        )
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out, atol=0.3)

        x = np.random.random((2, 50, 50, 3)) * 255
        out = kimage.resize(
            x, size=(25, 25), method=method, antialias=antialias
        )
        ref_out = tf.image.resize(
            x, size=(25, 25), method=method, antialias=antialias
        )
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out, atol=0.3)
