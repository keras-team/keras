import math

import numpy as np
import pytest
import scipy.ndimage
import tensorflow as tf
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.ops import image as kimage
from keras.src.testing.test_utils import named_product


class ImageOpsDynamicShapeTest(testing.TestCase):
    def setUp(self):
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")
        return super().setUp()

    def tearDown(self) -> None:
        backend.set_image_data_format(self.data_format)
        return super().tearDown()

    def test_rgb_to_grayscale(self):
        # Test channels_last
        x = KerasTensor([None, 20, 20, 3])
        out = kimage.rgb_to_grayscale(x)
        self.assertEqual(out.shape, (None, 20, 20, 1))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 20, 20])
        out = kimage.rgb_to_grayscale(x)
        self.assertEqual(out.shape, (None, 1, 20, 20))

    def test_rgb_to_hsv(self):
        # Test channels_last
        x = KerasTensor([None, 20, 20, 3])
        out = kimage.rgb_to_hsv(x)
        self.assertEqual(out.shape, (None, 20, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 20, 20])
        out = kimage.rgb_to_hsv(x)
        self.assertEqual(out.shape, (None, 3, 20, 20))

    def test_hsv_to_rgb(self):
        # Test channels_last
        x = KerasTensor([None, 20, 20, 3])
        out = kimage.hsv_to_rgb(x)
        self.assertEqual(out.shape, (None, 20, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 20, 20])
        out = kimage.hsv_to_rgb(x)
        self.assertEqual(out.shape, (None, 3, 20, 20))

    def test_resize(self):
        # Test channels_last
        x = KerasTensor([None, 20, 20, 3])
        out = kimage.resize(x, size=(15, 15))
        self.assertEqual(out.shape, (None, 15, 15, 3))

        x = KerasTensor([None, None, 3])
        out = kimage.resize(x, size=(15, 15))
        self.assertEqual(out.shape, (15, 15, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 20, 20])
        out = kimage.resize(x, size=(15, 15))
        self.assertEqual(out.shape, (None, 3, 15, 15))

        x = KerasTensor([3, None, None])
        out = kimage.resize(x, size=(15, 15))
        self.assertEqual(out.shape, (3, 15, 15))

    def test_affine_transform(self):
        # Test channels_last
        x = KerasTensor([None, 20, 20, 3])
        transform = KerasTensor([None, 8])
        out = kimage.affine_transform(x, transform)
        self.assertEqual(out.shape, (None, 20, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 20, 20])
        transform = KerasTensor([None, 8])
        out = kimage.affine_transform(x, transform)
        self.assertEqual(out.shape, (None, 3, 20, 20))

    def test_extract_patches(self):
        # Test channels_last
        x = KerasTensor([None, 20, 20, 3])
        p_h, p_w = 5, 5
        out = kimage.extract_patches(x, (p_h, p_w))
        self.assertEqual(out.shape, (None, 4, 4, 75))
        out = kimage.extract_patches(x, 5)
        self.assertEqual(out.shape, (None, 4, 4, 75))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 20, 20])
        p_h, p_w = 5, 5
        out = kimage.extract_patches(x, (p_h, p_w))
        self.assertEqual(out.shape, (None, 75, 4, 4))
        out = kimage.extract_patches(x, 5)
        self.assertEqual(out.shape, (None, 75, 4, 4))

    def test_map_coordinates(self):
        input = KerasTensor([20, 20, None])
        coordinates = KerasTensor([3, 15, 15, None])
        out = kimage.map_coordinates(input, coordinates, 0)
        self.assertEqual(out.shape, coordinates.shape[1:])

    def test_pad_images(self):
        # Test channels_last
        x = KerasTensor([None, 15, 25, 3])
        out = kimage.pad_images(x, 2, 3, target_height=20, target_width=30)
        self.assertEqual(out.shape, (None, 20, 30, 3))

        x = KerasTensor([None, None, 3])
        out = kimage.pad_images(x, 2, 3, target_height=20, target_width=30)
        self.assertEqual(out.shape, (20, 30, 3))

        # Test unknown shape
        x = KerasTensor([None, None, 3])
        out = kimage.pad_images(x, 2, 3, 2, 3)
        self.assertEqual(out.shape, (None, None, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 15, 25])
        out = kimage.pad_images(x, 2, 3, target_height=20, target_width=30)
        self.assertEqual(out.shape, (None, 3, 20, 30))

        x = KerasTensor([3, None, None])
        out = kimage.pad_images(x, 2, 3, target_height=20, target_width=30)
        self.assertEqual(out.shape, (3, 20, 30))

    def test_crop_images(self):
        # Test channels_last
        x = KerasTensor([None, 15, 25, 3])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (None, 10, 20, 3))

        x = KerasTensor([None, None, 3])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (10, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 15, 25])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (None, 3, 10, 20))

        x = KerasTensor([3, None, None])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (3, 10, 20))


class ImageOpsStaticShapeTest(testing.TestCase):
    def setUp(self):
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")
        return super().setUp()

    def tearDown(self) -> None:
        backend.set_image_data_format(self.data_format)
        return super().tearDown()

    def test_rgb_to_grayscale(self):
        # Test channels_last
        x = KerasTensor([20, 20, 3])
        out = kimage.rgb_to_grayscale(x)
        self.assertEqual(out.shape, (20, 20, 1))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 20, 20])
        out = kimage.rgb_to_grayscale(x)
        self.assertEqual(out.shape, (1, 20, 20))

    def test_rgb_to_hsv(self):
        # Test channels_last
        x = KerasTensor([20, 20, 3])
        out = kimage.rgb_to_hsv(x)
        self.assertEqual(out.shape, (20, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 20, 20])
        out = kimage.rgb_to_hsv(x)
        self.assertEqual(out.shape, (3, 20, 20))

    def test_hsv_to_rgb(self):
        # Test channels_last
        x = KerasTensor([20, 20, 3])
        out = kimage.hsv_to_rgb(x)
        self.assertEqual(out.shape, (20, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 20, 20])
        out = kimage.hsv_to_rgb(x)
        self.assertEqual(out.shape, (3, 20, 20))

    def test_resize(self):
        # Test channels_last
        x = KerasTensor([20, 20, 3])
        out = kimage.resize(x, size=(15, 15))
        self.assertEqual(out.shape, (15, 15, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 20, 20])
        out = kimage.resize(x, size=(15, 15))
        self.assertEqual(out.shape, (3, 15, 15))

    def test_affine_transform(self):
        # Test channels_last
        x = KerasTensor([20, 20, 3])
        transform = KerasTensor([8])
        out = kimage.affine_transform(x, transform)
        self.assertEqual(out.shape, (20, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 20, 20])
        transform = KerasTensor([8])
        out = kimage.affine_transform(x, transform)
        self.assertEqual(out.shape, (3, 20, 20))

    def test_extract_patches(self):
        # Test channels_last
        x = KerasTensor([20, 20, 3])
        p_h, p_w = 5, 5
        out = kimage.extract_patches(x, (p_h, p_w))
        self.assertEqual(out.shape, (4, 4, 75))
        out = kimage.extract_patches(x, 5)
        self.assertEqual(out.shape, (4, 4, 75))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 20, 20])
        p_h, p_w = 5, 5
        out = kimage.extract_patches(x, (p_h, p_w))
        self.assertEqual(out.shape, (75, 4, 4))
        out = kimage.extract_patches(x, 5)
        self.assertEqual(out.shape, (75, 4, 4))

    def test_map_coordinates(self):
        input = KerasTensor([20, 20, 3])
        coordinates = KerasTensor([3, 15, 15, 3])
        out = kimage.map_coordinates(input, coordinates, 0)
        self.assertEqual(out.shape, coordinates.shape[1:])

    def test_pad_images(self):
        # Test channels_last
        x = KerasTensor([15, 25, 3])
        out = kimage.pad_images(x, 2, 3, target_height=20, target_width=30)
        self.assertEqual(out.shape, (20, 30, 3))

        x_batch = KerasTensor([2, 15, 25, 3])
        out_batch = kimage.pad_images(
            x_batch, 2, 3, target_height=20, target_width=30
        )
        self.assertEqual(out_batch.shape, (2, 20, 30, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 15, 25])
        out = kimage.pad_images(x, 2, 3, target_height=20, target_width=30)
        self.assertEqual(out.shape, (3, 20, 30))

        x_batch = KerasTensor([2, 3, 15, 25])
        out_batch = kimage.pad_images(
            x_batch, 2, 3, target_height=20, target_width=30
        )
        self.assertEqual(out_batch.shape, (2, 3, 20, 30))

    def test_crop_images(self):
        # Test channels_last
        x = KerasTensor([15, 25, 3])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (10, 20, 3))

        x_batch = KerasTensor([2, 15, 25, 3])
        out_batch = kimage.crop_images(
            x_batch, 2, 3, target_height=10, target_width=20
        )
        self.assertEqual(out_batch.shape, (2, 10, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 15, 25])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (3, 10, 20))

        # Test channels_first and batched
        x_batch = KerasTensor([2, 3, 15, 25])
        out_batch = kimage.crop_images(
            x_batch, 2, 3, target_height=10, target_width=20
        )
        self.assertEqual(out_batch.shape, (2, 3, 10, 20))


AFFINE_TRANSFORM_INTERPOLATIONS = {  # map to order
    "nearest": 0,
    "bilinear": 1,
}


def _compute_affine_transform_coordinates(image, transform):
    image = image.copy()
    transform = transform.copy()
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


def _fixed_map_coordinates(
    input, coordinates, order, fill_mode="constant", fill_value=0.0
):
    # SciPy's implementation of map_coordinates handles boundaries incorrectly,
    # unless mode='reflect'. For order=1, this only affects interpolation
    # outside the bounds of the original array.
    # https://github.com/scipy/scipy/issues/2640
    padding = [
        (
            max(-np.floor(c.min()).astype(int) + 1, 0),
            max(np.ceil(c.max()).astype(int) + 1 - size, 0),
        )
        for c, size in zip(coordinates, input.shape)
    ]
    shifted_coords = [c + p[0] for p, c in zip(padding, coordinates)]
    pad_mode = {
        "nearest": "edge",
        "mirror": "reflect",
        "reflect": "symmetric",
    }.get(fill_mode, fill_mode)
    if fill_mode == "constant":
        padded = np.pad(
            input, padding, mode=pad_mode, constant_values=fill_value
        )
    else:
        padded = np.pad(input, padding, mode=pad_mode)
    result = scipy.ndimage.map_coordinates(
        padded, shifted_coords, order=order, mode=fill_mode, cval=fill_value
    )
    return result


class ImageOpsCorrectnessTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")
        return super().setUp()

    def tearDown(self) -> None:
        backend.set_image_data_format(self.data_format)
        return super().tearDown()

    def test_rgb_to_grayscale(self):
        # Test channels_last
        x = np.random.random((50, 50, 3)).astype("float32") * 255
        out = kimage.rgb_to_grayscale(x)
        ref_out = tf.image.rgb_to_grayscale(x)
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out)

        x = np.random.random((2, 50, 50, 3)).astype("float32") * 255
        out = kimage.rgb_to_grayscale(x)
        ref_out = tf.image.rgb_to_grayscale(x)
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.random((3, 50, 50)).astype("float32") * 255
        out = kimage.rgb_to_grayscale(x)
        ref_out = tf.image.rgb_to_grayscale(np.transpose(x, [1, 2, 0]))
        ref_out = tf.transpose(ref_out, [2, 0, 1])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out)

        x = np.random.random((2, 3, 50, 50)).astype("float32") * 255
        out = kimage.rgb_to_grayscale(x)
        ref_out = tf.image.rgb_to_grayscale(np.transpose(x, [0, 2, 3, 1]))
        ref_out = tf.transpose(ref_out, [0, 3, 1, 2])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out)

        # Test class
        out = kimage.RGBToGrayscale()(x)
        self.assertAllClose(ref_out, out)

    def test_rgb_to_hsv(self):
        # Test channels_last
        x = np.random.random((50, 50, 3)).astype("float32")
        out = kimage.rgb_to_hsv(x)
        ref_out = tf.image.rgb_to_hsv(x)
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out)

        x = np.random.random((2, 50, 50, 3)).astype("float32")
        out = kimage.rgb_to_hsv(x)
        ref_out = tf.image.rgb_to_hsv(x)
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.random((3, 50, 50)).astype("float32")
        out = kimage.rgb_to_hsv(x)
        ref_out = tf.image.rgb_to_hsv(np.transpose(x, [1, 2, 0]))
        ref_out = tf.transpose(ref_out, [2, 0, 1])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out)

        x = np.random.random((2, 3, 50, 50)).astype("float32")
        out = kimage.rgb_to_hsv(x)
        ref_out = tf.image.rgb_to_hsv(np.transpose(x, [0, 2, 3, 1]))
        ref_out = tf.transpose(ref_out, [0, 3, 1, 2])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out)

        # Test class
        out = kimage.RGBToHSV()(x)
        self.assertAllClose(ref_out, out)

    def test_hsv_to_rgb(self):
        # Test channels_last
        x = np.random.random((50, 50, 3)).astype("float32")
        out = kimage.hsv_to_rgb(x)
        ref_out = tf.image.hsv_to_rgb(x)
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out)

        x = np.random.random((2, 50, 50, 3)).astype("float32")
        out = kimage.hsv_to_rgb(x)
        ref_out = tf.image.hsv_to_rgb(x)
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.random((3, 50, 50)).astype("float32")
        out = kimage.hsv_to_rgb(x)
        ref_out = tf.image.hsv_to_rgb(np.transpose(x, [1, 2, 0]))
        ref_out = tf.transpose(ref_out, [2, 0, 1])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out)

        x = np.random.random((2, 3, 50, 50)).astype("float32")
        out = kimage.hsv_to_rgb(x)
        ref_out = tf.image.hsv_to_rgb(np.transpose(x, [0, 2, 3, 1]))
        ref_out = tf.transpose(ref_out, [0, 3, 1, 2])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out)

        # Test class
        out = kimage.HSVToRGB()(x)
        self.assertAllClose(ref_out, out)

    @parameterized.named_parameters(
        named_product(
            interpolation=[
                "bilinear",
                "nearest",
                "lanczos3",
                "lanczos5",
                "bicubic",
            ],
            antialias=[True, False],
        )
    )
    def test_resize(self, interpolation, antialias):
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
        # Test channels_last
        x = np.random.random((30, 30, 3)).astype("float32") * 255
        out = kimage.resize(
            x,
            size=(15, 15),
            interpolation=interpolation,
            antialias=antialias,
        )
        ref_out = tf.image.resize(
            x,
            size=(15, 15),
            method=interpolation,
            antialias=antialias,
        )
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out, atol=1e-4)

        x = np.random.random((2, 30, 30, 3)).astype("float32") * 255
        out = kimage.resize(
            x,
            size=(15, 15),
            interpolation=interpolation,
            antialias=antialias,
        )
        ref_out = tf.image.resize(
            x,
            size=(15, 15),
            method=interpolation,
            antialias=antialias,
        )
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out, atol=1e-4)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.random((3, 30, 30)).astype("float32") * 255
        out = kimage.resize(
            x,
            size=(15, 15),
            interpolation=interpolation,
            antialias=antialias,
        )
        ref_out = tf.image.resize(
            np.transpose(x, [1, 2, 0]),
            size=(15, 15),
            method=interpolation,
            antialias=antialias,
        )
        ref_out = tf.transpose(ref_out, [2, 0, 1])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out, atol=1e-4)

        x = np.random.random((2, 3, 30, 30)).astype("float32") * 255
        out = kimage.resize(
            x,
            size=(15, 15),
            interpolation=interpolation,
            antialias=antialias,
        )
        ref_out = tf.image.resize(
            np.transpose(x, [0, 2, 3, 1]),
            size=(15, 15),
            method=interpolation,
            antialias=antialias,
        )
        ref_out = tf.transpose(ref_out, [0, 3, 1, 2])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out, atol=1e-4)

        # Test class
        out = kimage.Resize(
            size=(15, 15),
            interpolation=interpolation,
            antialias=antialias,
        )(x)
        self.assertAllClose(ref_out, out, atol=1e-4)

    def test_resize_with_crop(self):
        # Test channels_last
        x = np.random.random((60, 50, 3)).astype("float32") * 255
        out = kimage.resize(x, size=(25, 25), crop_to_aspect_ratio=True)
        self.assertEqual(out.shape, (25, 25, 3))

        x = np.random.random((2, 50, 60, 3)).astype("float32") * 255
        out = kimage.resize(x, size=(25, 25), crop_to_aspect_ratio=True)
        self.assertEqual(out.shape, (2, 25, 25, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.random((3, 60, 50)).astype("float32") * 255
        out = kimage.resize(x, size=(25, 25), crop_to_aspect_ratio=True)
        self.assertEqual(out.shape, (3, 25, 25))

        x = np.random.random((2, 3, 50, 60)).astype("float32") * 255
        out = kimage.resize(x, size=(25, 25), crop_to_aspect_ratio=True)
        self.assertEqual(out.shape, (2, 3, 25, 25))

    @parameterized.named_parameters(named_product(fill_value=[1.0, 2.0]))
    def test_resize_with_pad(self, fill_value):
        # Test channels_last
        x = np.random.random((60, 50, 3)).astype("float32") * 255
        out = kimage.resize(
            x,
            size=(25, 25),
            pad_to_aspect_ratio=True,
            fill_value=fill_value,
        )
        self.assertEqual(out.shape, (25, 25, 3))

        x = np.random.random((2, 50, 60, 3)).astype("float32") * 255
        out = kimage.resize(
            x, size=(25, 25), pad_to_aspect_ratio=True, fill_value=fill_value
        )
        self.assertEqual(out.shape, (2, 25, 25, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.random((3, 60, 50)).astype("float32") * 255
        out = kimage.resize(
            x, size=(25, 25), pad_to_aspect_ratio=True, fill_value=fill_value
        )
        self.assertEqual(out.shape, (3, 25, 25))

        x = np.random.random((2, 3, 50, 60)).astype("float32") * 255
        out = kimage.resize(
            x, size=(25, 25), pad_to_aspect_ratio=True, fill_value=fill_value
        )
        self.assertEqual(out.shape, (2, 3, 25, 25))

    @parameterized.named_parameters(
        named_product(
            interpolation=["bilinear", "nearest"],
            fill_mode=["constant", "nearest", "wrap", "mirror", "reflect"],
        )
    )
    def test_affine_transform(self, interpolation, fill_mode):
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
        # TODO: `nearest` interpolation in jax and torch causes random index
        # shifting, resulting in significant differences in output which leads
        # to failure
        if backend.backend() in ("jax", "torch") and interpolation == "nearest":
            self.skipTest(
                f"In {backend.backend()} backend, "
                f"interpolation={interpolation} causes index shifting and "
                "leads test failure"
            )

        # Test channels_last
        np.random.seed(42)
        x = np.random.uniform(size=(50, 50, 3)).astype("float32") * 255
        transform = np.random.uniform(size=(6)).astype("float32")
        transform = np.pad(transform, (0, 2))  # makes c0, c1 always 0
        out = kimage.affine_transform(
            x, transform, interpolation=interpolation, fill_mode=fill_mode
        )
        coordinates = _compute_affine_transform_coordinates(x, transform)
        ref_out = _fixed_map_coordinates(
            x,
            coordinates,
            order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
            fill_mode=fill_mode,
        )
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out, atol=1e-2)

        x = np.random.uniform(size=(2, 50, 50, 3)).astype("float32") * 255
        transform = np.random.uniform(size=(2, 6)).astype("float32")
        transform = np.pad(transform, [(0, 0), (0, 2)])  # makes c0, c1 always 0
        out = kimage.affine_transform(
            x,
            transform,
            interpolation=interpolation,
            fill_mode=fill_mode,
        )
        coordinates = _compute_affine_transform_coordinates(x, transform)
        ref_out = np.stack(
            [
                _fixed_map_coordinates(
                    x[i],
                    coordinates[i],
                    order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                    fill_mode=fill_mode,
                )
                for i in range(x.shape[0])
            ],
            axis=0,
        )
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out, atol=1e-2)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.uniform(size=(3, 50, 50)).astype("float32") * 255
        transform = np.random.uniform(size=(6)).astype("float32")
        transform = np.pad(transform, (0, 2))  # makes c0, c1 always 0
        out = kimage.affine_transform(
            x, transform, interpolation=interpolation, fill_mode=fill_mode
        )
        coordinates = _compute_affine_transform_coordinates(
            np.transpose(x, [1, 2, 0]), transform
        )
        ref_out = _fixed_map_coordinates(
            np.transpose(x, [1, 2, 0]),
            coordinates,
            order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
            fill_mode=fill_mode,
        )
        ref_out = np.transpose(ref_out, [2, 0, 1])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out, atol=1e-2)

        x = np.random.uniform(size=(2, 3, 50, 50)).astype("float32") * 255
        transform = np.random.uniform(size=(2, 6)).astype("float32")
        transform = np.pad(transform, [(0, 0), (0, 2)])  # makes c0, c1 always 0
        out = kimage.affine_transform(
            x,
            transform,
            interpolation=interpolation,
            fill_mode=fill_mode,
        )
        coordinates = _compute_affine_transform_coordinates(
            np.transpose(x, [0, 2, 3, 1]), transform
        )
        ref_out = np.stack(
            [
                _fixed_map_coordinates(
                    np.transpose(x[i], [1, 2, 0]),
                    coordinates[i],
                    order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                    fill_mode=fill_mode,
                )
                for i in range(x.shape[0])
            ],
            axis=0,
        )
        ref_out = np.transpose(ref_out, [0, 3, 1, 2])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(ref_out, out, atol=1e-2)

        # Test class
        out = kimage.AffineTransform(
            interpolation=interpolation, fill_mode=fill_mode
        )(x, transform)
        self.assertAllClose(ref_out, out, atol=1e-2)

    @parameterized.named_parameters(
        named_product(
            size=[(3, 3), (5, 5)],
            strides=[None, (1, 1), (2, 2)],
            dilation_rate=[1, 3],
            padding=["valid", "same"],
        )
    )
    def test_extract_patches(self, size, strides, dilation_rate, padding):
        patch_h, patch_w = size[0], size[1]
        if strides is None:
            strides_h, strides_w = patch_h, patch_w
        else:
            strides_h, strides_w = strides[0], strides[1]
        if (
            backend.backend() == "tensorflow"
            and strides_h > 1
            or strides_w > 1
            and dilation_rate > 1
        ):
            pytest.skip("dilation_rate>1 with strides>1 not supported with TF")

        # Test channels_last
        image = np.random.uniform(size=(1, 20, 20, 3)).astype("float32")
        patches_out = kimage.extract_patches(
            image,
            size=size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
        )
        patches_ref = tf.image.extract_patches(
            image,
            sizes=(1, patch_h, patch_w, 1),
            strides=(1, strides_h, strides_w, 1),
            rates=(1, dilation_rate, dilation_rate, 1),
            padding=padding.upper(),
        )
        self.assertEqual(tuple(patches_out.shape), tuple(patches_ref.shape))
        self.assertAllClose(patches_ref, patches_out, atol=1e-2)

        # Test channels_first
        if backend.backend() == "tensorflow":
            # tensorflow doesn't support channels_first in
            # `kimage.extract_patches`
            return
        backend.set_image_data_format("channels_first")
        image = np.random.uniform(size=(1, 3, 20, 20)).astype("float32")
        patches_out = kimage.extract_patches(
            image,
            size=size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
        )
        patches_ref = tf.image.extract_patches(
            np.transpose(image, [0, 2, 3, 1]),
            sizes=(1, patch_h, patch_w, 1),
            strides=(1, strides_h, strides_w, 1),
            rates=(1, dilation_rate, dilation_rate, 1),
            padding=padding.upper(),
        )
        patches_ref = tf.transpose(patches_ref, [0, 3, 1, 2])
        self.assertEqual(tuple(patches_out.shape), tuple(patches_ref.shape))
        self.assertAllClose(patches_ref, patches_out, atol=1e-2)

        # Test class
        patches_out = kimage.ExtractPatches(
            size=size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
        )(image)
        self.assertAllClose(patches_ref, patches_out, atol=1e-2)

    @parameterized.named_parameters(
        named_product(
            # (input_shape, coordinates_shape)
            shape=[((5,), (7,)), ((3, 4, 5), (2, 3, 4))],
            # TODO: scipy.ndimage.map_coordinates does not support float16
            # TODO: torch cpu does not support round & floor for float16
            dtype=["uint8", "int32", "float32"],
            order=[0, 1],
            fill_mode=["constant", "nearest", "wrap", "mirror", "reflect"],
        )
    )
    def test_map_coordinates(self, shape, dtype, order, fill_mode):
        input_shape, coordinates_shape = shape
        input = np.arange(math.prod(input_shape), dtype=dtype).reshape(
            input_shape
        )
        coordinates_dtype = "float32" if "int" in dtype else dtype
        coordinates = [
            (size - 1)
            * np.random.uniform(size=coordinates_shape).astype(
                coordinates_dtype
            )
            for size in input_shape
        ]
        output = kimage.map_coordinates(input, coordinates, order, fill_mode)
        expected = _fixed_map_coordinates(input, coordinates, order, fill_mode)
        self.assertAllClose(output, expected)

        # Test class
        output = kimage.MapCoordinates(order, fill_mode)(input, coordinates)
        self.assertAllClose(output, expected)

    @parameterized.parameters(
        [
            (0, 0, 3, 3, None, None),
            (1, 0, 4, 3, None, None),
            (0, 1, 3, 4, None, None),
            (0, 0, 4, 3, None, None),
            (0, 0, 3, 4, None, None),
            (0, 0, None, None, 0, 1),
            (0, 0, None, None, 1, 0),
            (1, 2, None, None, 3, 4),
        ]
    )
    def test_pad_images(
        self,
        top_padding,
        left_padding,
        target_height,
        target_width,
        bottom_padding,
        right_padding,
    ):
        # Test channels_last
        image = np.random.uniform(size=(3, 3, 1)).astype("float32")
        _target_height = target_height  # For `tf.image.pad_to_bounding_box`
        _target_width = target_width  # For `tf.image.pad_to_bounding_box`
        if _target_height is None:
            _target_height = image.shape[0] + top_padding + bottom_padding
        if _target_width is None:
            _target_width = image.shape[1] + left_padding + right_padding
        padded_image = kimage.pad_images(
            image,
            top_padding,
            left_padding,
            bottom_padding,
            right_padding,
            target_height,
            target_width,
        )
        ref_padded_image = tf.image.pad_to_bounding_box(
            image, top_padding, left_padding, _target_height, _target_width
        )
        self.assertEqual(
            tuple(padded_image.shape), tuple(ref_padded_image.shape)
        )
        self.assertAllClose(ref_padded_image, padded_image)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        image = np.random.uniform(size=(1, 3, 3)).astype("float32")
        padded_image = kimage.pad_images(
            image,
            top_padding,
            left_padding,
            bottom_padding,
            right_padding,
            target_height,
            target_width,
        )
        ref_padded_image = tf.image.pad_to_bounding_box(
            np.transpose(image, [1, 2, 0]),
            top_padding,
            left_padding,
            _target_height,
            _target_width,
        )
        ref_padded_image = tf.transpose(ref_padded_image, [2, 0, 1])
        self.assertEqual(
            tuple(padded_image.shape), tuple(ref_padded_image.shape)
        )
        self.assertAllClose(ref_padded_image, padded_image)

        # Test class
        padded_image = kimage.PadImages(
            top_padding,
            left_padding,
            bottom_padding,
            right_padding,
            target_height,
            target_width,
        )(image)
        self.assertAllClose(ref_padded_image, padded_image)

    @parameterized.parameters(
        [
            (0, 0, 3, 3, None, None),
            (1, 0, 4, 3, None, None),
            (0, 1, 3, 4, None, None),
            (0, 0, 4, 3, None, None),
            (0, 0, 3, 4, None, None),
            (0, 0, None, None, 0, 1),
            (0, 0, None, None, 1, 0),
            (1, 2, None, None, 3, 4),
        ]
    )
    def test_crop_images(
        self,
        top_cropping,
        left_cropping,
        target_height,
        target_width,
        bottom_cropping,
        right_cropping,
    ):
        # Test channels_last
        image = np.random.uniform(size=(10, 10, 1)).astype("float32")
        _target_height = target_height  # For `tf.image.pad_to_bounding_box`
        _target_width = target_width  # For `tf.image.pad_to_bounding_box`
        if _target_height is None:
            _target_height = image.shape[0] - top_cropping - bottom_cropping
        if _target_width is None:
            _target_width = image.shape[1] - left_cropping - right_cropping
        cropped_image = kimage.crop_images(
            image,
            top_cropping,
            left_cropping,
            bottom_cropping,
            right_cropping,
            target_height,
            target_width,
        )
        ref_cropped_image = tf.image.crop_to_bounding_box(
            image, top_cropping, left_cropping, _target_height, _target_width
        )
        self.assertEqual(
            tuple(cropped_image.shape), tuple(ref_cropped_image.shape)
        )
        self.assertAllClose(ref_cropped_image, cropped_image)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        image = np.random.uniform(size=(1, 10, 10)).astype("float32")
        cropped_image = kimage.crop_images(
            image,
            top_cropping,
            left_cropping,
            bottom_cropping,
            right_cropping,
            target_height,
            target_width,
        )
        ref_cropped_image = tf.image.crop_to_bounding_box(
            np.transpose(image, [1, 2, 0]),
            top_cropping,
            left_cropping,
            _target_height,
            _target_width,
        )
        ref_cropped_image = tf.transpose(ref_cropped_image, [2, 0, 1])
        self.assertEqual(
            tuple(cropped_image.shape), tuple(ref_cropped_image.shape)
        )
        self.assertAllClose(ref_cropped_image, cropped_image)

        # Test class
        cropped_image = kimage.CropImages(
            top_cropping,
            left_cropping,
            bottom_cropping,
            right_cropping,
            target_height,
            target_width,
        )(image)
        self.assertAllClose(ref_cropped_image, cropped_image)


class ImageOpsBehaviorTests(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")
        return super().setUp()

    def tearDown(self) -> None:
        backend.set_image_data_format(self.data_format)
        return super().tearDown()

    @parameterized.named_parameters(named_product(rank=[2, 5]))
    def test_rgb_to_grayscale_invalid_rank(self, rank):
        shape = [3] * rank
        invalid_image = np.random.uniform(size=shape)
        with self.assertRaisesRegex(
            ValueError,
            "Invalid images rank: expected rank 3",
        ):
            kimage.rgb_to_grayscale(invalid_image)
        with self.assertRaisesRegex(
            ValueError,
            "Invalid images rank: expected rank 3",
        ):
            kimage.RGBToGrayscale()(invalid_image)
        invalid_image = KerasTensor(shape=shape)
        with self.assertRaisesRegex(
            ValueError,
            "Invalid images rank: expected rank 3",
        ):
            kimage.rgb_to_grayscale(invalid_image)

    @parameterized.named_parameters(named_product(rank=[2, 5]))
    def test_rgb_to_hsv_invalid_rank(self, rank):
        shape = [3] * rank
        invalid_image = np.random.uniform(size=shape)
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.rgb_to_hsv(invalid_image)
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.RGBToHSV()(invalid_image)
        invalid_image = KerasTensor(shape=shape)
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.rgb_to_hsv(invalid_image)

    def test_rgb_to_hsv_invalid_dtype(self):
        invalid_image = np.random.uniform(size=(10, 10, 3)).astype("int32")
        with self.assertRaisesRegex(
            ValueError, "Invalid images dtype: expected float dtype."
        ):
            kimage.rgb_to_hsv(invalid_image)
        with self.assertRaisesRegex(
            ValueError, "Invalid images dtype: expected float dtype."
        ):
            kimage.RGBToHSV()(invalid_image)
        invalid_image = KerasTensor(shape=(10, 10, 3), dtype="int32")
        with self.assertRaisesRegex(
            ValueError, "Invalid images dtype: expected float dtype."
        ):
            kimage.rgb_to_hsv(invalid_image)

    @parameterized.named_parameters(named_product(rank=[2, 5]))
    def test_hsv_to_rgb_invalid_rank(self, rank):
        shape = [3] * rank
        invalid_image = np.random.uniform(size=shape)
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.hsv_to_rgb(invalid_image)
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.HSVToRGB()(invalid_image)
        invalid_image = KerasTensor(shape=shape)
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.hsv_to_rgb(invalid_image)

    def test_hsv_to_rgb_invalid_dtype(self):
        invalid_image = np.random.uniform(size=(10, 10, 3)).astype("int32")
        with self.assertRaisesRegex(
            ValueError, "Invalid images dtype: expected float dtype."
        ):
            kimage.hsv_to_rgb(invalid_image)
        with self.assertRaisesRegex(
            ValueError, "Invalid images dtype: expected float dtype."
        ):
            kimage.HSVToRGB()(invalid_image)
        invalid_image = KerasTensor(shape=(10, 10, 3), dtype="int32")
        with self.assertRaisesRegex(
            ValueError, "Invalid images dtype: expected float dtype."
        ):
            kimage.hsv_to_rgb(invalid_image)

    def test_resize_invalid_rank(self):
        # Test rank=2
        invalid_image = np.random.uniform(size=(10, 10))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.resize(invalid_image, (5, 5))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.Resize((5, 5))(invalid_image)

        # Test rank=2, symbolic tensor
        invalid_image = KerasTensor(shape=(10, 10))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.resize(invalid_image, (5, 5))

    def test_affine_transform_invalid_images_rank(self):
        # Test rank=2
        invalid_image = np.random.uniform(size=(10, 10))
        transform = np.random.uniform(size=(6,))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.affine_transform(invalid_image, transform)
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.AffineTransform()(invalid_image, transform)

        # Test rank=5
        invalid_image = np.random.uniform(size=(2, 10, 10, 3, 1))
        transform = np.random.uniform(size=(6,))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.affine_transform(invalid_image, transform)
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.AffineTransform()(invalid_image, transform)

        # Test rank=2, symbolic tensor
        invalid_image = KerasTensor(shape=(10, 10))
        transform = KerasTensor(shape=(6,))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.affine_transform(invalid_image, transform)

    def test_affine_transform_invalid_transform_rank(self):
        # Test rank=3
        images = np.random.uniform(size=(10, 10, 3))
        invalid_transform = np.random.uniform(size=(2, 3, 2))
        with self.assertRaisesRegex(
            ValueError, "Invalid transform rank: expected rank 1"
        ):
            kimage.affine_transform(images, invalid_transform)
        with self.assertRaisesRegex(
            ValueError, "Invalid transform rank: expected rank 1"
        ):
            kimage.AffineTransform()(images, invalid_transform)

        # Test rank=0
        invalid_transform = np.random.uniform(size=())
        with self.assertRaisesRegex(
            ValueError, "Invalid transform rank: expected rank 1"
        ):
            kimage.affine_transform(images, invalid_transform)
        with self.assertRaisesRegex(
            ValueError, "Invalid transform rank: expected rank 1"
        ):
            kimage.AffineTransform()(images, invalid_transform)

        # Test rank=3, symbolic tensor
        images = KerasTensor(shape=(10, 10, 3))
        invalid_transform = KerasTensor(shape=(2, 3, 2))
        with self.assertRaisesRegex(
            ValueError, "Invalid transform rank: expected rank 1"
        ):
            kimage.affine_transform(images, invalid_transform)

    def test_extract_patches_invalid_size(self):
        size = (3, 3, 3)  # Invalid size, too many dimensions
        image = np.random.uniform(size=(2, 20, 20, 3))
        with self.assertRaisesRegex(
            TypeError, "Expected an int or a tuple of length 2"
        ):
            kimage.extract_patches(image, size)

        size = "5"  # Invalid size type
        with self.assertRaisesRegex(
            TypeError, "Expected an int or a tuple of length 2"
        ):
            kimage.extract_patches(image, size)

    def test_map_coordinates_invalid_coordinates_rank(self):
        # Test mismatched dim of coordinates
        image = np.random.uniform(size=(10, 10, 3))
        coordinates = np.random.uniform(size=(2, 10, 10))
        with self.assertRaisesRegex(
            ValueError, "must be the same as the rank of `inputs`"
        ):
            kimage.map_coordinates(image, coordinates, 0)
        with self.assertRaisesRegex(
            ValueError, "must be the same as the rank of `inputs`"
        ):
            kimage.MapCoordinates(0)(image, coordinates)

        # Test rank=1
        coordinates = np.random.uniform(size=(3,))
        with self.assertRaisesRegex(ValueError, "expected at least rank 2"):
            kimage.map_coordinates(image, coordinates, 0)
        with self.assertRaisesRegex(ValueError, "expected at least rank 2"):
            kimage.MapCoordinates(0)(image, coordinates)

    def test_crop_images_unknown_shape(self):
        # Test unknown height and target_height
        x = KerasTensor([None, 10, 3])
        with self.assertRaisesRegex(
            ValueError, "When the height of the images is unknown"
        ):
            kimage.crop_images(x, 2, 3, 4, 5)

        # Test unknown width and target_width
        x = KerasTensor([10, None, 3])
        with self.assertRaisesRegex(
            ValueError, "When the width of the images is unknown"
        ):
            kimage.crop_images(x, 2, 3, 4, 5)
