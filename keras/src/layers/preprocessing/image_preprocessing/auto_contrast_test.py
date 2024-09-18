import numpy as np
import pytest

from keras.src import layers
from keras.src import ops
from keras.src import testing


class AutoContrastTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.AutoContrast,
            init_kwargs={
                "value_range": (20, 200),
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    def test_constant_channels_dont_get_nanned(self):
        img = np.array([1, 1], dtype="float32")
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        layer = layers.AutoContrast(value_range=(0, 255))
        ys = layer(img)

        self.assertTrue(np.any(ops.convert_to_numpy(ys[0]) == 1.0))
        self.assertTrue(np.any(ops.convert_to_numpy(ys[0]) == 1.0))

    def test_auto_contrast_expands_value_range(self):
        img = np.array([0, 128], dtype="float32")
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        layer = layers.AutoContrast(value_range=(0, 255))
        ys = layer(img)

        self.assertTrue(np.any(ops.convert_to_numpy(ys[0]) == 0.0))
        self.assertTrue(np.any(ops.convert_to_numpy(ys[0]) == 255.0))

    def test_auto_contrast_different_values_per_channel(self):
        img = np.array(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            dtype="float32",
        )
        img = np.expand_dims(img, axis=0)

        layer = layers.AutoContrast(value_range=(0, 255))
        ys = layer(img)

        self.assertTrue(np.any(ops.convert_to_numpy(ys[0, ..., 0]) == 0.0))
        self.assertTrue(np.any(ops.convert_to_numpy(ys[0, ..., 1]) == 0.0))

        self.assertTrue(np.any(ops.convert_to_numpy(ys[0, ..., 0]) == 255.0))
        self.assertTrue(np.any(ops.convert_to_numpy(ys[0, ..., 1]) == 255.0))

        self.assertAllClose(
            ys,
            [
                [
                    [[0.0, 0.0, 0.0], [85.0, 85.0, 85.0]],
                    [[170.0, 170.0, 170.0], [255.0, 255.0, 255.0]],
                ]
            ],
        )

    def test_auto_contrast_expands_value_range_uint8(self):
        img = np.array([0, 128], dtype="uint8")
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        layer = layers.AutoContrast(value_range=(0, 255))
        ys = layer(img)

        self.assertTrue(np.any(ops.convert_to_numpy(ys[0]) == 0.0))
        self.assertTrue(np.any(ops.convert_to_numpy(ys[0]) == 255.0))

    def test_auto_contrast_properly_converts_value_range(self):
        img = np.array([0, 0.5], dtype="float32")
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        layer = layers.AutoContrast(value_range=(0, 1))
        ys = layer(img)
        self.assertAllClose(
            ops.convert_to_numpy(ys[0]), np.array([[[0.0]], [[1]]])
        )
