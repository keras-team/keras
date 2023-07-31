import numpy as np
import pytest
import scipy.ndimage
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

    def test_extract_patches(self):
        x = KerasTensor([None, 20, 20, 3])
        p_h, p_w = 5, 5
        out = kimage.extract_patches(x, (p_h, p_w))
        self.assertEqual(out.shape, (None, 4, 4, 75))
        out = kimage.extract_patches(x, 5)
        self.assertEqual(out.shape, (None, 4, 4, 75))


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

    def test_extract_patches(self):
        x = KerasTensor([20, 20, 3])
        p_h, p_w = 5, 5
        out = kimage.extract_patches(x, (p_h, p_w))
        self.assertEqual(out.shape, (4, 4, 75))
        out = kimage.extract_patches(x, 5)
        self.assertEqual(out.shape, (4, 4, 75))


AFFINE_TRANSFORM_INTERPOLATIONS = {  # map to order
    "nearest": 0,
    "bilinear": 1,
}
AFFINE_TRANSFORM_FILL_MODES = {
    "constant": "grid-constant",
    "nearest": "nearest",
    "wrap": "grid-wrap",
    "mirror": "mirror",
    "reflect": "reflect",
}


def _compute_affine_transform_coordinates(image, transform):
    need_squeeze = False
    if len(image.shape) == 3:  # unbatched
        need_squeeze = True
        image = np.expand_dims(image, axis=0)
        transform = np.expand_dims(transform, axis=0)
    batch_size = image.shape[0]
    # get indices
    meshgrid = np.meshgrid(
        *[np.arange(size) for size in image.shape[1:]], indexing="ij"
    )
    indices = np.concatenate(
        [np.expand_dims(x, axis=-1) for x in meshgrid], axis=-1
    )
    indices = np.tile(indices, (batch_size, 1, 1, 1, 1))
    # swap the values
    transform[:, 4], transform[:, 0] = (
        transform[:, 0].copy(),
        transform[:, 4].copy(),
    )
    transform[:, 5], transform[:, 2] = (
        transform[:, 2].copy(),
        transform[:, 5].copy(),
    )
    # deal with transform
    transform = np.pad(transform, pad_width=[[0, 0], [0, 1]], constant_values=1)
    transform = np.reshape(transform, (batch_size, 3, 3))
    offset = np.pad(transform[:, 0:2, 2], pad_width=[[0, 0], [0, 1]])
    transform[:, 0:2, 2] = 0
    # transform the indices
    coordinates = np.einsum("Bhwij, Bjk -> Bhwik", indices, transform)
    coordinates = np.moveaxis(coordinates, source=-1, destination=1)
    coordinates += np.reshape(a=offset, newshape=(*offset.shape, 1, 1, 1))
    if need_squeeze:
        coordinates = np.squeeze(coordinates, axis=0)
    return coordinates


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
            ("bilinear", "mirror", "channels_last"),
            ("nearest", "mirror", "channels_last"),
            ("bilinear", "reflect", "channels_last"),
            ("nearest", "reflect", "channels_last"),
            ("bilinear", "constant", "channels_first"),
        ]
    )
    def test_affine_transform(self, interpolation, fill_mode, data_format):
        if backend.backend() == "torch" and fill_mode == "wrap":
            self.skipTest(
                "In torch backend, applying affine_transform with "
                "fill_mode=wrap is not supported"
            )
        if backend.backend() == "torch" and fill_mode == "reflect":
            self.skipTest(
                "In torch backend, applying affine_transform with "
                "fill_mode=reflect is redirected to fill_mode=mirror"
            )
        if backend.backend() == "tensorflow" and fill_mode == "mirror":
            self.skipTest(
                "In tensorflow backend, applying affine_transform with "
                "fill_mode=mirror is not supported"
            )
        if backend.backend() == "tensorflow" and fill_mode == "wrap":
            self.skipTest(
                "In tensorflow backend, the numerical results of applying "
                "affine_transform with fill_mode=wrap is inconsistent with"
                "scipy"
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
        coordinates = _compute_affine_transform_coordinates(x, transform)
        ref_out = scipy.ndimage.map_coordinates(
            x,
            coordinates,
            order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
            mode=AFFINE_TRANSFORM_FILL_MODES[fill_mode],
            prefilter=False,
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
        coordinates = _compute_affine_transform_coordinates(x, transform)
        ref_out = np.stack(
            [
                scipy.ndimage.map_coordinates(
                    x[i],
                    coordinates[i],
                    order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                    mode=AFFINE_TRANSFORM_FILL_MODES[fill_mode],
                    prefilter=False,
                )
                for i in range(x.shape[0])
            ],
            axis=0,
        )
        if data_format == "channels_first":
            ref_out = np.transpose(ref_out, (0, 3, 1, 2))
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out, atol=0.3)

    @parameterized.parameters(
        [
            ((5, 5), None, 1, "valid", "channels_last"),
            ((3, 3), (2, 2), 1, "valid", "channels_last"),
            ((5, 5), None, 1, "valid", "channels_first"),
            ((3, 3), (2, 2), 1, "valid", "channels_first"),
            ((5, 5), None, 1, "same", "channels_last"),
            ((3, 3), (2, 2), 1, "same", "channels_last"),
            ((5, 5), None, 1, "same", "channels_first"),
            ((3, 3), (2, 2), 1, "same", "channels_first"),
            ((5, 5), (1, 1), 3, "same", "channels_first"),
            ((5, 5), (2, 2), 3, "same", "channels_first"),
            ((5, 5), (2, 2), 3, "same", "channels_last"),
        ]
    )
    def test_extract_patches(
        self, size, strides, dilation_rate, padding, data_format
    ):
        if (
            data_format == "channels_first"
            and backend.backend() == "tensorflow"
        ):
            pytest.skip("channels_first unsupported on CPU with TF")

        if (
            isinstance(strides, tuple)
            and backend.backend() == "tensorflow"
            and dilation_rate > 1
        ):
            pytest.skip("dilation_rate>1 with strides>1 not supported with TF")
        if data_format == "channels_first":
            image = np.random.uniform(size=(1, 3, 20, 20))
        else:
            image = np.random.uniform(size=(1, 20, 20, 3))
        patch_h, patch_w = size[0], size[1]
        if strides is None:
            strides_h, strides_w = patch_h, patch_w
        else:
            strides_h, strides_w = strides[0], strides[1]

        patches_out = kimage.extract_patches(
            backend.convert_to_tensor(image, dtype="float32"),
            size=size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            data_format=data_format,
        )
        if data_format == "channels_first":
            patches_out = backend.numpy.transpose(
                patches_out, axes=[0, 2, 3, 1]
            )
        if data_format == "channels_first":
            image = np.transpose(image, [0, 2, 3, 1])
        patches_ref = tf.image.extract_patches(
            image,
            sizes=(1, patch_h, patch_w, 1),
            strides=(1, strides_h, strides_w, 1),
            rates=(1, dilation_rate, dilation_rate, 1),
            padding=padding.upper(),
        )
        self.assertEqual(tuple(patches_out.shape), tuple(patches_ref.shape))
        self.assertAllClose(
            patches_ref.numpy(), backend.convert_to_numpy(patches_out), atol=0.3
        )
