import numpy as np
import pytest
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras.src import layers
from keras.src import ops
from keras.src import testing


class EqualizationTest(testing.TestCase):
    def assertAllInRange(self, array, min_val, max_val):
        self.assertTrue(np.all(array >= min_val))
        self.assertTrue(np.all(array <= max_val))

    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.Equalization,
            init_kwargs={
                "value_range": (0, 255),
                "data_format": "channels_last",
            },
            input_shape=(1, 2, 2, 3),
            supports_masking=False,
            expected_output_shape=(1, 2, 2, 3),
        )

        self.run_layer_test(
            layers.Equalization,
            init_kwargs={
                "value_range": (0, 255),
                "data_format": "channels_first",
            },
            input_shape=(1, 3, 2, 2),
            supports_masking=False,
            expected_output_shape=(1, 3, 2, 2),
        )

    def test_equalizes_to_all_bins(self):
        xs = np.random.uniform(size=(2, 512, 512, 3), low=0, high=255).astype(
            np.float32
        )
        layer = layers.Equalization(value_range=(0, 255))
        xs = layer(xs)

        for i in range(0, 256):
            self.assertTrue(np.any(ops.convert_to_numpy(xs) == i))

    @parameterized.named_parameters(
        ("float32", np.float32), ("int32", np.int32), ("int64", np.int64)
    )
    def test_input_dtypes(self, dtype):
        xs = np.random.uniform(size=(2, 512, 512, 3), low=0, high=255).astype(
            dtype
        )
        layer = layers.Equalization(value_range=(0, 255))
        xs = ops.convert_to_numpy(layer(xs))

        for i in range(0, 256):
            self.assertTrue(np.any(xs == i))
        self.assertAllInRange(xs, 0, 255)

    @parameterized.named_parameters(("0_255", 0, 255), ("0_1", 0, 1))
    def test_output_range(self, lower, upper):
        xs = np.random.uniform(
            size=(2, 512, 512, 3), low=lower, high=upper
        ).astype(np.float32)
        layer = layers.Equalization(value_range=(lower, upper))
        xs = ops.convert_to_numpy(layer(xs))
        self.assertAllInRange(xs, lower, upper)

    def test_constant_regions(self):
        xs = np.zeros((1, 64, 64, 3), dtype=np.float32)
        xs[:, :21, :, :] = 50
        xs[:, 21:42, :, :] = 100
        xs[:, 42:, :, :] = 200

        layer = layers.Equalization(value_range=(0, 255))
        equalized = ops.convert_to_numpy(layer(xs))

        self.assertTrue(len(np.unique(equalized)) >= 3)
        self.assertAllInRange(equalized, 0, 255)

    def test_grayscale_images(self):
        xs_last = np.random.uniform(0, 255, size=(2, 64, 64, 1)).astype(
            np.float32
        )
        layer_last = layers.Equalization(
            value_range=(0, 255), data_format="channels_last"
        )
        equalized_last = ops.convert_to_numpy(layer_last(xs_last))
        self.assertEqual(equalized_last.shape[-1], 1)
        self.assertAllInRange(equalized_last, 0, 255)

        xs_first = np.random.uniform(0, 255, size=(2, 1, 64, 64)).astype(
            np.float32
        )
        layer_first = layers.Equalization(
            value_range=(0, 255), data_format="channels_first"
        )
        equalized_first = ops.convert_to_numpy(layer_first(xs_first))
        self.assertEqual(equalized_first.shape[1], 1)
        self.assertAllInRange(equalized_first, 0, 255)

    def test_single_color_image(self):
        xs_last = np.full((1, 64, 64, 3), 128, dtype=np.float32)
        layer_last = layers.Equalization(
            value_range=(0, 255), data_format="channels_last"
        )
        equalized_last = ops.convert_to_numpy(layer_last(xs_last))
        self.assertAllClose(equalized_last, 128.0)

        xs_first = np.full((1, 3, 64, 64), 128, dtype=np.float32)
        layer_first = layers.Equalization(
            value_range=(0, 255), data_format="channels_first"
        )
        equalized_first = ops.convert_to_numpy(layer_first(xs_first))
        self.assertAllClose(equalized_first, 128.0)

    def test_different_bin_sizes(self):
        xs = np.random.uniform(0, 255, size=(1, 64, 64, 3)).astype(np.float32)
        bin_sizes = [16, 64, 128, 256]
        for bins in bin_sizes:
            layer = layers.Equalization(value_range=(0, 255), bins=bins)
            equalized = ops.convert_to_numpy(layer(xs))
            self.assertAllInRange(equalized, 0, 255)

    def test_tf_data_compatibility(self):
        layer = layers.Equalization(value_range=(0, 255))
        input_data = np.random.random((2, 8, 8, 3)) * 255
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output_array = output.numpy()
            self.assertAllInRange(output_array, 0, 255)
