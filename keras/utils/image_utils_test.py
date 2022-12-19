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
"""Tests for image_utils."""

import io
import os
import pathlib

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras.utils import image_utils


@test_utils.run_v2_only
class TestImageUtils(test_combinations.TestCase):
    def test_smart_resize(self):
        test_input = np.random.random((20, 40, 3))
        output = image_utils.smart_resize(test_input, size=(50, 50))
        self.assertIsInstance(output, np.ndarray)
        self.assertListEqual(list(output.shape), [50, 50, 3])
        output = image_utils.smart_resize(test_input, size=(10, 10))
        self.assertListEqual(list(output.shape), [10, 10, 3])
        output = image_utils.smart_resize(test_input, size=(100, 50))
        self.assertListEqual(list(output.shape), [100, 50, 3])
        output = image_utils.smart_resize(test_input, size=(5, 15))
        self.assertListEqual(list(output.shape), [5, 15, 3])

    @parameterized.named_parameters(
        ("size1", (50, 50)),
        ("size2", (10, 10)),
        ("size3", (100, 50)),
        ("size4", (5, 15)),
    )
    def test_smart_resize_tf_dataset(self, size):
        test_input_np = np.random.random((2, 20, 40, 3))
        test_ds = tf.data.Dataset.from_tensor_slices(test_input_np)

        resize = lambda img: image_utils.smart_resize(img, size=size)
        test_ds = test_ds.map(resize)
        for sample in test_ds.as_numpy_iterator():
            self.assertIsInstance(sample, np.ndarray)
            self.assertListEqual(list(sample.shape), [size[0], size[1], 3])

    def test_smart_resize_batch(self):
        img = np.random.random((2, 20, 40, 3))
        out = image_utils.smart_resize(img, size=(20, 20))
        self.assertListEqual(list(out.shape), [2, 20, 20, 3])
        self.assertAllClose(out, img[:, :, 10:-10, :])

    def test_smart_resize_errors(self):
        with self.assertRaisesRegex(ValueError, "a tuple of 2 integers"):
            image_utils.smart_resize(
                np.random.random((20, 20, 2)), size=(10, 5, 3)
            )
        with self.assertRaisesRegex(ValueError, "incorrect rank"):
            image_utils.smart_resize(np.random.random((2, 4)), size=(10, 5))
        with self.assertRaisesRegex(ValueError, "incorrect rank"):
            image_utils.smart_resize(
                np.random.random((2, 4, 4, 5, 3)), size=(10, 5)
            )


@test_utils.run_v2_only
class TestImageLoading(test_combinations.TestCase):
    def test_load_img(self):
        tmpdir = self.create_tempdir()
        filename_rgb = os.path.join(tmpdir.full_path, "rgb_utils.png")
        filename_rgba = os.path.join(tmpdir.full_path, "rgba_utils.png")
        filename_grayscale_8bit = os.path.join(
            tmpdir.full_path, "grayscale_8bit_utils.png"
        )
        filename_grayscale_16bit = os.path.join(
            tmpdir.full_path, "grayscale_16bit_utils.tiff"
        )
        filename_grayscale_32bit = os.path.join(
            tmpdir.full_path, "grayscale_32bit_utils.tiff"
        )

        original_rgb_array = np.array(
            255 * np.random.rand(100, 100, 3), dtype=np.uint8
        )
        original_rgb = image_utils.array_to_img(original_rgb_array, scale=False)
        original_rgb.save(filename_rgb)

        original_rgba_array = np.array(
            255 * np.random.rand(100, 100, 4), dtype=np.uint8
        )
        original_rgba = image_utils.array_to_img(
            original_rgba_array, scale=False
        )
        original_rgba.save(filename_rgba)

        original_grayscale_8bit_array = np.array(
            255 * np.random.rand(100, 100, 1), dtype=np.uint8
        )
        original_grayscale_8bit = image_utils.array_to_img(
            original_grayscale_8bit_array, scale=False
        )
        original_grayscale_8bit.save(filename_grayscale_8bit)

        original_grayscale_16bit_array = np.array(
            np.random.randint(-2147483648, 2147483647, (100, 100, 1)),
            dtype=np.int16,
        )
        original_grayscale_16bit = image_utils.array_to_img(
            original_grayscale_16bit_array, scale=False, dtype="int16"
        )
        original_grayscale_16bit.save(filename_grayscale_16bit)

        original_grayscale_32bit_array = np.array(
            np.random.randint(-2147483648, 2147483647, (100, 100, 1)),
            dtype=np.int32,
        )
        original_grayscale_32bit = image_utils.array_to_img(
            original_grayscale_32bit_array, scale=False, dtype="int32"
        )
        original_grayscale_32bit.save(filename_grayscale_32bit)

        # Test that loaded image is exactly equal to original.

        loaded_im = image_utils.load_img(filename_rgb)
        loaded_im_array = image_utils.img_to_array(loaded_im)
        self.assertEqual(loaded_im_array.shape, original_rgb_array.shape)
        self.assertAllClose(loaded_im_array, original_rgb_array)

        loaded_im = image_utils.load_img(filename_rgba, color_mode="rgba")
        loaded_im_array = image_utils.img_to_array(loaded_im)
        self.assertEqual(loaded_im_array.shape, original_rgba_array.shape)
        self.assertAllClose(loaded_im_array, original_rgba_array)

        loaded_im = image_utils.load_img(filename_rgb, color_mode="grayscale")
        loaded_im_array = image_utils.img_to_array(loaded_im)
        self.assertEqual(
            loaded_im_array.shape,
            (original_rgb_array.shape[0], original_rgb_array.shape[1], 1),
        )

        loaded_im = image_utils.load_img(
            filename_grayscale_8bit, color_mode="grayscale"
        )
        loaded_im_array = image_utils.img_to_array(loaded_im)
        self.assertEqual(
            loaded_im_array.shape, original_grayscale_8bit_array.shape
        )
        self.assertAllClose(loaded_im_array, original_grayscale_8bit_array)

        loaded_im = image_utils.load_img(
            filename_grayscale_16bit, color_mode="grayscale"
        )
        loaded_im_array = image_utils.img_to_array(loaded_im, dtype="int16")
        self.assertEqual(
            loaded_im_array.shape, original_grayscale_16bit_array.shape
        )
        self.assertAllClose(loaded_im_array, original_grayscale_16bit_array)
        # test casting int16 image to float32
        loaded_im_array = image_utils.img_to_array(loaded_im)
        self.assertAllClose(loaded_im_array, original_grayscale_16bit_array)

        loaded_im = image_utils.load_img(
            filename_grayscale_32bit, color_mode="grayscale"
        )
        loaded_im_array = image_utils.img_to_array(loaded_im, dtype="int32")
        self.assertEqual(
            loaded_im_array.shape, original_grayscale_32bit_array.shape
        )
        self.assertAllClose(loaded_im_array, original_grayscale_32bit_array)
        # test casting int32 image to float32
        loaded_im_array = image_utils.img_to_array(loaded_im)
        self.assertAllClose(loaded_im_array, original_grayscale_32bit_array)

        # Test that nothing is changed when target size is equal to original.

        loaded_im = image_utils.load_img(filename_rgb, target_size=(100, 100))
        loaded_im_array = image_utils.img_to_array(loaded_im)
        self.assertEqual(loaded_im_array.shape, original_rgb_array.shape)
        self.assertAllClose(loaded_im_array, original_rgb_array)

        loaded_im = image_utils.load_img(
            filename_rgba, color_mode="rgba", target_size=(100, 100)
        )
        loaded_im_array = image_utils.img_to_array(loaded_im)
        self.assertEqual(loaded_im_array.shape, original_rgba_array.shape)
        self.assertAllClose(loaded_im_array, original_rgba_array)

        loaded_im = image_utils.load_img(
            filename_rgb, color_mode="grayscale", target_size=(100, 100)
        )
        loaded_im_array = image_utils.img_to_array(loaded_im)
        self.assertEqual(
            loaded_im_array.shape,
            (original_rgba_array.shape[0], original_rgba_array.shape[1], 1),
        )

        loaded_im = image_utils.load_img(
            filename_grayscale_8bit,
            color_mode="grayscale",
            target_size=(100, 100),
        )
        loaded_im_array = image_utils.img_to_array(loaded_im)
        self.assertEqual(
            loaded_im_array.shape, original_grayscale_8bit_array.shape
        )
        self.assertAllClose(loaded_im_array, original_grayscale_8bit_array)

        loaded_im = image_utils.load_img(
            filename_grayscale_16bit,
            color_mode="grayscale",
            target_size=(100, 100),
        )
        loaded_im_array = image_utils.img_to_array(loaded_im, dtype="int16")
        self.assertEqual(
            loaded_im_array.shape, original_grayscale_16bit_array.shape
        )
        self.assertAllClose(loaded_im_array, original_grayscale_16bit_array)

        loaded_im = image_utils.load_img(
            filename_grayscale_32bit,
            color_mode="grayscale",
            target_size=(100, 100),
        )
        loaded_im_array = image_utils.img_to_array(loaded_im, dtype="int32")
        self.assertEqual(
            loaded_im_array.shape, original_grayscale_32bit_array.shape
        )
        self.assertAllClose(loaded_im_array, original_grayscale_32bit_array)

        # Test down-sampling with bilinear interpolation.

        loaded_im = image_utils.load_img(filename_rgb, target_size=(25, 25))
        loaded_im_array = image_utils.img_to_array(loaded_im)
        self.assertEqual(loaded_im_array.shape, (25, 25, 3))

        loaded_im = image_utils.load_img(
            filename_rgba, color_mode="rgba", target_size=(25, 25)
        )
        loaded_im_array = image_utils.img_to_array(loaded_im)
        self.assertEqual(loaded_im_array.shape, (25, 25, 4))

        loaded_im = image_utils.load_img(
            filename_rgb, color_mode="grayscale", target_size=(25, 25)
        )
        loaded_im_array = image_utils.img_to_array(loaded_im)
        self.assertEqual(loaded_im_array.shape, (25, 25, 1))

        loaded_im = image_utils.load_img(
            filename_grayscale_8bit,
            color_mode="grayscale",
            target_size=(25, 25),
        )
        loaded_im_array = image_utils.img_to_array(loaded_im)
        self.assertEqual(loaded_im_array.shape, (25, 25, 1))

        loaded_im = image_utils.load_img(
            filename_grayscale_16bit,
            color_mode="grayscale",
            target_size=(25, 25),
        )
        loaded_im_array = image_utils.img_to_array(loaded_im, dtype="int16")
        self.assertEqual(loaded_im_array.shape, (25, 25, 1))

        loaded_im = image_utils.load_img(
            filename_grayscale_32bit,
            color_mode="grayscale",
            target_size=(25, 25),
        )
        loaded_im_array = image_utils.img_to_array(loaded_im, dtype="int32")
        self.assertEqual(loaded_im_array.shape, (25, 25, 1))

        # Test down-sampling with nearest neighbor interpolation.

        loaded_im_nearest = image_utils.load_img(
            filename_rgb, target_size=(25, 25), interpolation="nearest"
        )
        loaded_im_array_nearest = image_utils.img_to_array(loaded_im_nearest)
        self.assertEqual(loaded_im_array_nearest.shape, (25, 25, 3))
        self.assertTrue(np.any(loaded_im_array_nearest != loaded_im_array))

        loaded_im_nearest = image_utils.load_img(
            filename_rgba,
            color_mode="rgba",
            target_size=(25, 25),
            interpolation="nearest",
        )
        loaded_im_array_nearest = image_utils.img_to_array(loaded_im_nearest)
        self.assertEqual(loaded_im_array_nearest.shape, (25, 25, 4))
        self.assertTrue(np.any(loaded_im_array_nearest != loaded_im_array))

        loaded_im = image_utils.load_img(
            filename_grayscale_8bit,
            color_mode="grayscale",
            target_size=(25, 25),
            interpolation="nearest",
        )
        loaded_im_array = image_utils.img_to_array(loaded_im)
        self.assertEqual(loaded_im_array.shape, (25, 25, 1))

        loaded_im = image_utils.load_img(
            filename_grayscale_16bit,
            color_mode="grayscale",
            target_size=(25, 25),
            interpolation="nearest",
        )
        loaded_im_array = image_utils.img_to_array(loaded_im, dtype="int16")
        self.assertEqual(loaded_im_array.shape, (25, 25, 1))

        loaded_im = image_utils.load_img(
            filename_grayscale_32bit,
            color_mode="grayscale",
            target_size=(25, 25),
            interpolation="nearest",
        )
        loaded_im_array = image_utils.img_to_array(loaded_im, dtype="int32")
        self.assertEqual(loaded_im_array.shape, (25, 25, 1))

        # Test different path type
        with open(filename_grayscale_32bit, "rb") as f:
            path_ = io.BytesIO(f.read())  # io.Bytesio
        loaded_im = image_utils.load_img(path_, color_mode="grayscale")
        loaded_im_array = image_utils.img_to_array(loaded_im, dtype=np.int32)
        self.assertAllClose(loaded_im_array, original_grayscale_32bit_array)

        path_ = filename_grayscale_32bit  # str
        loaded_im = image_utils.load_img(path_, color_mode="grayscale")
        loaded_im_array = image_utils.img_to_array(loaded_im, dtype=np.int32)
        self.assertAllClose(loaded_im_array, original_grayscale_32bit_array)

        path_ = filename_grayscale_32bit.encode()  # bytes
        loaded_im = image_utils.load_img(path_, color_mode="grayscale")
        loaded_im_array = image_utils.img_to_array(loaded_im, dtype=np.int32)
        self.assertAllClose(loaded_im_array, original_grayscale_32bit_array)

        path_ = pathlib.Path(
            os.path.join(tmpdir.full_path, "grayscale_32bit_utils.tiff")
        )
        loaded_im = image_utils.load_img(path_, color_mode="grayscale")
        loaded_im_array = image_utils.img_to_array(loaded_im, dtype=np.int32)
        self.assertAllClose(loaded_im_array, original_grayscale_32bit_array)

        # Check that exception is raised if interpolation not supported.

        loaded_im = image_utils.load_img(
            filename_rgb, interpolation="unsupported"
        )
        with self.assertRaises(ValueError):
            loaded_im = image_utils.load_img(
                filename_rgb, target_size=(25, 25), interpolation="unsupported"
            )

        # Check that the aspect ratio of a square is the same

        filename_red_square = os.path.join(
            tmpdir.full_path, "red_square_utils.png"
        )
        arr = np.zeros((50, 100, 3), dtype=np.uint8)  # rectangle image 100x50
        arr[20:30, 45:55, 0] = 255  # red square 10x10
        red_square_array = np.array(arr)
        red_square = image_utils.array_to_img(red_square_array, scale=False)
        red_square.save(filename_red_square)

        loaded_im = image_utils.load_img(
            filename_red_square, target_size=(25, 25), keep_aspect_ratio=True
        )
        loaded_im_array = image_utils.img_to_array(loaded_im)
        self.assertEqual(loaded_im_array.shape, (25, 25, 3))

        red_channel_arr = loaded_im_array[:, :, 0].astype(bool)
        square_width = np.sum(np.sum(red_channel_arr, axis=0))
        square_height = np.sum(np.sum(red_channel_arr, axis=1))
        aspect_ratio_result = square_width / square_height

        # original square had 1:1 ratio
        self.assertNear(aspect_ratio_result, 1.0, 0.01)

    def test_array_to_img_and_img_to_array(self):
        height, width = 10, 8

        # Test the data format
        # Test RGB 3D
        x = np.random.random((3, height, width))
        img = image_utils.array_to_img(x, data_format="channels_first")
        self.assertEqual(img.size, (width, height))

        x = image_utils.img_to_array(img, data_format="channels_first")
        self.assertEqual(x.shape, (3, height, width))

        # Test RGBA 3D
        x = np.random.random((4, height, width))
        img = image_utils.array_to_img(x, data_format="channels_first")
        self.assertEqual(img.size, (width, height))

        x = image_utils.img_to_array(img, data_format="channels_first")
        self.assertEqual(x.shape, (4, height, width))

        # Test 2D
        x = np.random.random((1, height, width))
        img = image_utils.array_to_img(x, data_format="channels_first")
        self.assertEqual(img.size, (width, height))

        x = image_utils.img_to_array(img, data_format="channels_first")
        self.assertEqual(x.shape, (1, height, width))

        # grayscale 32-bit signed integer
        x = np.array(
            np.random.randint(-2147483648, 2147483647, (1, height, width)),
            dtype=np.int32,
        )
        img = image_utils.array_to_img(x, data_format="channels_first")
        self.assertEqual(img.size, (width, height))

        x = image_utils.img_to_array(img, data_format="channels_first")
        self.assertEqual(x.shape, (1, height, width))

        # Test tf data format
        # Test RGB 3D
        x = np.random.random((height, width, 3))
        img = image_utils.array_to_img(x, data_format="channels_last")
        self.assertEqual(img.size, (width, height))

        x = image_utils.img_to_array(img, data_format="channels_last")
        self.assertEqual(x.shape, (height, width, 3))

        # Test RGBA 3D
        x = np.random.random((height, width, 4))
        img = image_utils.array_to_img(x, data_format="channels_last")
        self.assertEqual(img.size, (width, height))

        x = image_utils.img_to_array(img, data_format="channels_last")
        self.assertEqual(x.shape, (height, width, 4))

        # Test 2D
        x = np.random.random((height, width, 1))
        img = image_utils.array_to_img(x, data_format="channels_last")
        self.assertEqual(img.size, (width, height))

        x = image_utils.img_to_array(img, data_format="channels_last")
        self.assertEqual(x.shape, (height, width, 1))

        # grayscale 16-bit signed integer
        x = np.array(
            np.random.randint(-2147483648, 2147483647, (height, width, 1)),
            dtype=np.int16,
        )
        img = image_utils.array_to_img(x, data_format="channels_last")
        self.assertEqual(img.size, (width, height))

        x = image_utils.img_to_array(img, data_format="channels_last")
        self.assertEqual(x.shape, (height, width, 1))

        # grayscale 32-bit signed integer
        x = np.array(
            np.random.randint(-2147483648, 2147483647, (height, width, 1)),
            dtype=np.int32,
        )
        img = image_utils.array_to_img(x, data_format="channels_last")
        self.assertEqual(img.size, (width, height))

        x = image_utils.img_to_array(img, data_format="channels_last")
        self.assertEqual(x.shape, (height, width, 1))

        # Test invalid use case
        with self.assertRaises(ValueError):
            x = np.random.random((height, width))  # not 3D
            img = image_utils.array_to_img(x, data_format="channels_first")

        with self.assertRaises(ValueError):
            x = np.random.random((height, width, 3))
            # unknown data_format
            img = image_utils.array_to_img(x, data_format="channels")

        with self.assertRaises(ValueError):
            # neither RGB, RGBA, or gray-scale
            x = np.random.random((height, width, 5))
            img = image_utils.array_to_img(x, data_format="channels_last")

        with self.assertRaises(ValueError):
            x = np.random.random((height, width, 3))
            # unknown data_format
            img = image_utils.img_to_array(x, data_format="channels")

        with self.assertRaises(ValueError):
            # neither RGB, RGBA, or gray-scale
            x = np.random.random((height, width, 5, 3))
            img = image_utils.img_to_array(x, data_format="channels_last")


if __name__ == "__main__":
    tf.test.main()
