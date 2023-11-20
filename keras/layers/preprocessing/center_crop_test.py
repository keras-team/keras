import numpy as np
import pytest
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras import backend
from keras import layers
from keras import testing


class CenterCropTest(testing.TestCase, parameterized.TestCase):
    def np_center_crop(self, img, h_new, w_new, data_format="channels_last"):
        img = np.array(img)
        if img.ndim == 4:
            if data_format == "channels_last":
                _, h, w = img.shape[:3]
            else:
                _, h, w = img.shape[1:]
        else:
            if data_format == "channels_last":
                h, w = img.shape[:2]
            else:
                h, w = img.shape[1:]
        h_start = (h - h_new) // 2
        w_start = (w - w_new) // 2
        if data_format == "channels_last":
            return img[
                ..., h_start : h_start + h_new, w_start : w_start + w_new, :
            ]
        else:
            return img[
                ..., h_start : h_start + h_new, w_start : w_start + w_new
            ]

    @pytest.mark.requires_trainable_backend
    def test_center_crop_basics(self):
        self.run_layer_test(
            layers.CenterCrop,
            init_kwargs={
                "height": 6,
                "width": 6,
                "data_format": "channels_last",
            },
            input_shape=(2, 12, 12, 3),
            expected_output_shape=(2, 6, 6, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )
        self.run_layer_test(
            layers.CenterCrop,
            init_kwargs={
                "height": 7,
                "width": 7,
                "data_format": "channels_first",
            },
            input_shape=(2, 3, 13, 13),
            expected_output_shape=(2, 3, 7, 7),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @parameterized.parameters(
        [
            ((5, 7), "channels_first"),
            ((5, 7), "channels_last"),
            ((4, 9), "channels_first"),
            ((9, 4), "channels_last"),
        ]
    )
    def test_center_crop_correctness(self, size, data_format):
        # batched case
        if data_format == "channels_first":
            img = np.random.random((2, 3, 9, 11))
        else:
            img = np.random.random((2, 9, 11, 3))
        out = layers.CenterCrop(
            size[0],
            size[1],
            data_format=data_format,
        )(img)
        if data_format == "channels_first":
            img_transpose = np.transpose(img, (0, 2, 3, 1))

            ref_out = np.transpose(
                self.np_center_crop(img_transpose, size[0], size[1]),
                (0, 3, 1, 2),
            )
        else:
            ref_out = self.np_center_crop(img, size[0], size[1])
        self.assertAllClose(ref_out, out)

        # unbatched case
        if data_format == "channels_first":
            img = np.random.random((3, 9, 11))
        else:
            img = np.random.random((9, 11, 3))
        out = layers.CenterCrop(
            size[0],
            size[1],
            data_format=data_format,
        )(img)
        if data_format == "channels_first":
            img_transpose = np.transpose(img, (1, 2, 0))
            ref_out = np.transpose(
                self.np_center_crop(
                    img_transpose,
                    size[0],
                    size[1],
                ),
                (2, 0, 1),
            )
        else:
            ref_out = self.np_center_crop(
                img,
                size[0],
                size[1],
            )
        self.assertAllClose(ref_out, out)

    @parameterized.parameters(
        [
            ((15, 10), "channels_first"),
            ((10, 17), "channels_last"),
        ]
    )
    def test_input_smaller_than_crop_box(self, size, data_format):
        """Output should equal resizing with crop_to_aspect ratio."""
        # batched case
        if data_format == "channels_first":
            img = np.random.random((2, 3, 9, 11))
        else:
            img = np.random.random((2, 9, 11, 3))
        out = layers.CenterCrop(
            size[0],
            size[1],
            data_format=data_format,
        )(img)
        ref_out = layers.Resizing(
            size[0], size[1], data_format=data_format, crop_to_aspect_ratio=True
        )(img)
        self.assertAllClose(ref_out, out)

        # unbatched case
        if data_format == "channels_first":
            img = np.random.random((3, 9, 11))
        else:
            img = np.random.random((9, 11, 3))
        out = layers.CenterCrop(
            size[0],
            size[1],
            data_format=data_format,
        )(img)
        ref_out = layers.Resizing(
            size[0], size[1], data_format=data_format, crop_to_aspect_ratio=True
        )(img)
        self.assertAllClose(ref_out, out)

    def test_tf_data_compatibility(self):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 10, 12, 3)
            output_shape = (2, 8, 9, 3)
        else:
            input_shape = (2, 3, 10, 12)
            output_shape = (2, 3, 8, 9)
        layer = layers.CenterCrop(8, 9)
        input_data = np.random.random(input_shape)
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output = output.numpy()
        self.assertEqual(tuple(output.shape), output_shape)

    def test_list_compatibility(self):
        if backend.config.image_data_format() == "channels_last":
            images = [
                np.random.rand(10, 10, 3),
                np.random.rand(10, 10, 3),
            ]
            output_shape = (2, 6, 5, 3)
        else:
            images = [
                np.random.rand(3, 10, 10),
                np.random.rand(3, 10, 10),
            ]
            output_shape = (2, 3, 6, 5)
        output = layers.CenterCrop(height=6, width=5)(images)
        ref_output = self.np_center_crop(
            images, 6, 5, data_format=backend.config.image_data_format()
        )
        self.assertEqual(tuple(output.shape), output_shape)
        self.assertAllClose(ref_output, output)
