import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from keras_core import backend
from keras_core import testing
from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.ops import image as kimage


class ImageOpsDynamicShapeTest(testing.TestCase):
    def test_resize(self):
        x = KerasTensor([None, 20, 20, 3])
        out = kimage.resize(x, size=(15, 15))
        self.assertEqual(out.shape, (None, 15, 15, 3))

        x = KerasTensor([None, None, 3])
        out = kimage.resize(x, size=(15, 15))
        self.assertEqual(out.shape, (15, 15, 3))

    def test_affine_transform(self):
        x = KerasTensor([None, 20, 20, 3])
        transform = KerasTensor([None, 8])
        out = kimage.affine_transform(x, transform)
        self.assertEqual(out.shape, (None, 20, 20, 3))


class ImageOpsStaticShapeTest(testing.TestCase):
    def test_resize(self):
        x = KerasTensor([20, 20, 3])
        out = kimage.resize(x, size=(15, 15))
        self.assertEqual(out.shape, (15, 15, 3))

    def test_affine_transform(self):
        x = KerasTensor([20, 20, 3])
        transform = KerasTensor([8])
        out = kimage.affine_transform(x, transform)
        self.assertEqual(out.shape, (20, 20, 3))


class ImageOpsCorrectnessTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        [
            ("bilinear", True, "channels_last"),
            ("nearest", True, "channels_last"),
            ("lanczos3", True, "channels_last"),
            ("lanczos5", True, "channels_last"),
            ("bicubic", True, "channels_last"),
            ("bilinear", False, "channels_last"),
            ("nearest", False, "channels_last"),
            ("lanczos3", False, "channels_last"),
            ("lanczos5", False, "channels_last"),
            ("bicubic", False, "channels_last"),
            ("bilinear", True, "channels_first"),
        ]
    )
    def test_resize(self, interpolation, antialias, data_format):
        if backend.backend() == "torch":
            if "lanczos" in interpolation:
                self.skipTest(
                    "Resizing with Lanczos interpolation is "
                    "not supported by the PyTorch backend. "
                    f"Received: interpolation={interpolation}."
                )
            if interpolation == "bicubic" and antialias is False:
                self.skipTest(
                    "Resizing with Bicubic interpolation in "
                    "PyTorch backend produces noise. Please "
                    "turn on anti-aliasing. "
                    f"Received: interpolation={interpolation}, "
                    f"antialias={antialias}."
                )
        # Unbatched case
        if data_format == "channels_first":
            x = np.random.random((3, 50, 50)) * 255
        else:
            x = np.random.random((50, 50, 3)) * 255
        out = kimage.resize(
            x,
            size=(25, 25),
            interpolation=interpolation,
            antialias=antialias,
            data_format=data_format,
        )
        if data_format == "channels_first":
            x = np.transpose(x, (1, 2, 0))
        ref_out = tf.image.resize(
            x, size=(25, 25), method=interpolation, antialias=antialias
        )
        if data_format == "channels_first":
            ref_out = np.transpose(ref_out, (2, 0, 1))
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out, atol=0.3)

        # Batched case
        if data_format == "channels_first":
            x = np.random.random((2, 3, 50, 50)) * 255
        else:
            x = np.random.random((2, 50, 50, 3)) * 255
        out = kimage.resize(
            x,
            size=(25, 25),
            interpolation=interpolation,
            antialias=antialias,
            data_format=data_format,
        )
        if data_format == "channels_first":
            x = np.transpose(x, (0, 2, 3, 1))
        ref_out = tf.image.resize(
            x, size=(25, 25), method=interpolation, antialias=antialias
        )
        if data_format == "channels_first":
            ref_out = np.transpose(ref_out, (0, 3, 1, 2))
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out, atol=0.3)

    @parameterized.parameters(
        [
            ("bilinear", "constant", "channels_last"),
            ("nearest", "constant", "channels_last"),
            ("bilinear", "nearest", "channels_last"),
            ("nearest", "nearest", "channels_last"),
            ("bilinear", "wrap", "channels_last"),
            ("nearest", "wrap", "channels_last"),
            ("bilinear", "reflect", "channels_last"),
            ("nearest", "reflect", "channels_last"),
            ("bilinear", "constant", "channels_first"),
        ]
    )
    def test_affine_transform(self, interpolation, fill_mode, data_format):
        if fill_mode == "wrap" and backend.backend() == "torch":
            self.skipTest(
                "Applying affine transform with fill_mode=wrap is not support"
                " in torch backend"
            )
        if fill_mode == "wrap" and backend.backend() in ("jax", "numpy"):
            self.skipTest(
                "The numerical results of applying affine transform with "
                "fill_mode=wrap in tensorflow is inconsistent with jax and "
                "numpy backends"
            )

        # Unbatched case
        if data_format == "channels_first":
            x = np.random.random((3, 50, 50)) * 255
        else:
            x = np.random.random((50, 50, 3)) * 255
        transform = np.random.random(size=(6))
        transform = np.pad(transform, (0, 2))  # makes c0, c1 always 0
        out = kimage.affine_transform(
            x,
            transform,
            interpolation=interpolation,
            fill_mode=fill_mode,
            data_format=data_format,
        )
        if data_format == "channels_first":
            x = np.transpose(x, (1, 2, 0))
        ref_out = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(x, axis=0),
            transforms=tf.cast(tf.expand_dims(transform, axis=0), tf.float32),
            output_shape=tf.shape(x)[:-1],
            fill_value=0,
            interpolation=interpolation.upper(),
            fill_mode=fill_mode.upper(),
        )
        ref_out = ref_out[0]
        if data_format == "channels_first":
            ref_out = np.transpose(ref_out, (2, 0, 1))
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        if backend.backend() == "torch":
            # TODO: cannot pass with torch backend
            with self.assertRaises(AssertionError):
                self.assertAllClose(ref_out, out, atol=0.3)
        else:
            self.assertAllClose(ref_out, out, atol=0.3)

        # Batched case
        if data_format == "channels_first":
            x = np.random.random((2, 3, 50, 50)) * 255
        else:
            x = np.random.random((2, 50, 50, 3)) * 255
        transform = np.random.random(size=(2, 6))
        transform = np.pad(transform, [(0, 0), (0, 2)])  # makes c0, c1 always 0
        out = kimage.affine_transform(
            x,
            transform,
            interpolation=interpolation,
            fill_mode=fill_mode,
            data_format=data_format,
        )
        if data_format == "channels_first":
            x = np.transpose(x, (0, 2, 3, 1))
        ref_out = tf.raw_ops.ImageProjectiveTransformV3(
            images=x,
            transforms=tf.cast(transform, tf.float32),
            output_shape=tf.shape(x)[1:-1],
            fill_value=0,
            interpolation=interpolation.upper(),
            fill_mode=fill_mode.upper(),
        )
        if data_format == "channels_first":
            ref_out = np.transpose(ref_out, (0, 3, 1, 2))
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        if backend.backend() == "torch":
            # TODO: cannot pass with torch backend
            with self.assertRaises(AssertionError):
                self.assertAllClose(ref_out, out, atol=0.3)
        else:
            self.assertAllClose(ref_out, out, atol=0.3)
