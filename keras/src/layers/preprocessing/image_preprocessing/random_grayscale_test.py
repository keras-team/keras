import numpy as np
import pytest
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import testing


class RandomGrayscaleTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.RandomGrayscale,
            init_kwargs={
                "factor": 0.5,
                "data_format": "channels_last",
            },
            input_shape=(1, 2, 2, 3),
            supports_masking=False,
            expected_output_shape=(1, 2, 2, 3),
        )

        self.run_layer_test(
            layers.RandomGrayscale,
            init_kwargs={
                "factor": 0.5,
                "data_format": "channels_first",
            },
            input_shape=(1, 3, 2, 2),
            supports_masking=False,
            expected_output_shape=(1, 3, 2, 2),
        )

    @parameterized.named_parameters(
        ("channels_last", "channels_last"), ("channels_first", "channels_first")
    )
    def test_grayscale_conversion(self, data_format):
        if data_format == "channels_last":
            xs = np.random.uniform(0, 255, size=(2, 4, 4, 3)).astype(np.float32)
            layer = layers.RandomGrayscale(factor=1.0, data_format=data_format)
            transformed = ops.convert_to_numpy(layer(xs))
            self.assertEqual(transformed.shape[-1], 3)
            for img in transformed:
                r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                self.assertTrue(np.allclose(r, g) and np.allclose(g, b))
        else:
            xs = np.random.uniform(0, 255, size=(2, 3, 4, 4)).astype(np.float32)
            layer = layers.RandomGrayscale(factor=1.0, data_format=data_format)
            transformed = ops.convert_to_numpy(layer(xs))
            self.assertEqual(transformed.shape[1], 3)
            for img in transformed:
                r, g, b = img[0], img[1], img[2]
                self.assertTrue(np.allclose(r, g) and np.allclose(g, b))

    def test_invalid_factor(self):
        with self.assertRaises(ValueError):
            layers.RandomGrayscale(factor=-0.1)

        with self.assertRaises(ValueError):
            layers.RandomGrayscale(factor=1.1)

    def test_tf_data_compatibility(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3)) * 255
        else:
            input_data = np.random.random((2, 3, 8, 8)) * 255

        layer = layers.RandomGrayscale(factor=0.5, data_format=data_format)
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)

        for output in ds.take(1):
            output_array = output.numpy()
            self.assertEqual(output_array.shape, input_data.shape)

    def test_grayscale_with_single_color_image(self):
        test_cases = [
            # batched inputs
            (np.full((1, 4, 4, 3), 128, dtype=np.float32), "channels_last"),
            (np.full((1, 3, 4, 4), 128, dtype=np.float32), "channels_first"),
            # unbatched inputs
            (np.full((4, 4, 3), 128, dtype=np.float32), "channels_last"),
            (np.full((3, 4, 4), 128, dtype=np.float32), "channels_first"),
        ]

        for xs, data_format in test_cases:
            layer = layers.RandomGrayscale(factor=1.0, data_format=data_format)
            transformed = ops.convert_to_numpy(layer(xs))

            # Determine if the input was batched
            is_batched = len(xs.shape) == 4

            # If batched, select the first image from the batch for inspection.
            # Otherwise, use the transformed image directly.
            # `image_to_inspect` will always be a 3D tensor.
            if is_batched:
                image_to_inspect = transformed[0]
            else:
                image_to_inspect = transformed

            if data_format == "channels_last":
                # image_to_inspect has shape (H, W, C),
                # get the first channel [:, :, 0]
                channel_data = image_to_inspect[:, :, 0]
            else:  # data_format == "channels_first"
                # image_to_inspect has shape (C, H, W),
                # get the first channel [0, :, :]
                channel_data = image_to_inspect[0, :, :]

            unique_vals = np.unique(channel_data)
            self.assertEqual(len(unique_vals), 1)
