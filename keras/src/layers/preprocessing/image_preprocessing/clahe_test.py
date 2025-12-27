import numpy as np
import pytest
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras.src import layers
from keras.src import ops
from keras.src import testing


class CLAHETest(testing.TestCase):
    def assertAllInRange(self, array, min_val, max_val):
        self.assertTrue(np.all(array >= min_val))
        self.assertTrue(np.all(array <= max_val))

    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.CLAHE,
            init_kwargs={
                "value_range": (0, 255),
                "data_format": "channels_last",
                "tile_grid_size": (2, 2),
            },
            input_shape=(1, 4, 4, 3),
            supports_masking=False,
            expected_output_shape=(1, 4, 4, 3),
        )

        self.run_layer_test(
            layers.CLAHE,
            init_kwargs={
                "value_range": (0, 255),
                "data_format": "channels_first",
                "tile_grid_size": (2, 2),
            },
            input_shape=(1, 3, 4, 4),
            supports_masking=False,
            expected_output_shape=(1, 3, 4, 4),
        )

    def test_clahe_identity(self):
        xs = np.random.uniform(size=(2, 64, 64, 3), low=0, high=255).astype(
            np.float32
        )
        layer = layers.CLAHE(value_range=(0, 255))
        xs_out = layer(xs)
        self.assertAllInRange(ops.convert_to_numpy(xs_out), 0, 255)
        self.assertEqual(xs_out.shape, (2, 64, 64, 3))

    @parameterized.named_parameters(
        ("float32", np.float32), ("int32", np.int32), ("int64", np.int64)
    )
    def test_input_dtypes(self, dtype):
        xs = np.random.uniform(size=(2, 32, 32, 3), low=0, high=255).astype(
            dtype
        )
        layer = layers.CLAHE(value_range=(0, 255))
        xs_out = ops.convert_to_numpy(layer(xs))
        self.assertAllInRange(xs_out, 0, 255)

    @parameterized.named_parameters(("0_255", 0, 255), ("0_1", 0, 1))
    def test_output_range(self, lower, upper):
        xs = np.random.uniform(
            size=(2, 32, 32, 3), low=lower, high=upper
        ).astype(np.float32)
        layer = layers.CLAHE(value_range=(lower, upper))
        xs_out = ops.convert_to_numpy(layer(xs))
        self.assertAllInRange(xs_out, lower, upper)

    def test_grayscale_images(self):
        xs = np.random.uniform(0, 255, size=(2, 32, 32, 1)).astype(np.float32)
        layer = layers.CLAHE(value_range=(0, 255), data_format="channels_last")
        out = ops.convert_to_numpy(layer(xs))
        self.assertEqual(out.shape[-1], 1)
        self.assertAllInRange(out, 0, 255)

    def test_tf_data_compatibility(self):
        layer = layers.CLAHE(value_range=(0, 255))
        input_data = np.random.random((2, 16, 16, 3)) * 255
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output_array = output.numpy()
            self.assertAllInRange(output_array, 0, 255)
            self.assertEqual(output_array.shape, (2, 16, 16, 3))
