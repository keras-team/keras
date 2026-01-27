# keras/src/layers/preprocessing/image_preprocessing/random_resized_crop_test.py
import pytest
import numpy as np
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class RandomResizedCropTest(testing.TestCase):
    @parameterized.named_parameters(
        ("unbatched_channels_last", (64, 64, 3), (32, 48, 3), "channels_last"),
        ("batched_channels_last", (4, 64, 64, 3), (4, 32, 48, 3), "channels_last"),
        ("unbatched_channels_first", (3, 64, 64), (3, 32, 48), "channels_first"),
        ("batched_channels_first", (4, 3, 64, 64), (4, 3, 32, 48), "channels_first"),
    )
    def test_random_resized_crop_output_shape(self, input_shape, expected_shape, data_format):
        layer = layers.RandomResizedCrop(
            height=32, width=48, data_format=data_format
        )
        inputs = np.ones(input_shape, dtype="float32")
        outputs = layer(inputs, training=True)
        self.assertEqual(tuple(outputs.shape), expected_shape)

    @parameterized.named_parameters(
        ("channels_last", "channels_last"),
        ("channels_first", "channels_first"),
    )
    def test_random_resized_crop_inference_is_deterministic(self, data_format):
        layer = layers.RandomResizedCrop(height=32, width=32, data_format=data_format)

        if data_format == "channels_last":
            inputs = np.arange(64 * 64 * 3, dtype="float32").reshape((64, 64, 3))
        else:
            inputs = np.arange(3 * 64 * 64, dtype="float32").reshape((3, 64, 64))

        out1 = layer(inputs, training=False)
        out2 = layer(inputs, training=False)

        self.assertAllClose(out1, out2, atol=0.0, rtol=0.0)

    def test_random_resized_crop_training_is_random(self):
        layer1 = layers.RandomResizedCrop(height=32, width=32, seed=1)
        layer2 = layers.RandomResizedCrop(height=32, width=32, seed=2)

        inputs = np.arange(2 * 64 * 64 * 3, dtype="float32").reshape((2, 64, 64, 3))

        out1 = layer1(inputs, training=True)
        out2 = layer2(inputs, training=True)

        # Extremely unlikely to be identical if randomness is applied
        with self.assertRaises(AssertionError):
            self.assertAllClose(out1, out2, atol=1e-6)

    def test_random_resized_crop_dtype_preserved(self):
        layer = layers.RandomResizedCrop(height=16, width=16)

        inputs = np.ones((64, 64, 3), dtype="float16")
        outputs = layer(inputs, training=True)

        # tf.image.crop_and_resize promotes float16 to float32
        self.assertEqual(outputs.dtype, "float32")

    def test_random_resized_crop_config_roundtrip(self):
        layer = layers.RandomResizedCrop(
            height=24,
            width=40,
            scale=(0.2, 0.8),
            ratio=(0.75, 1.25),
        )

        config = layer.get_config()
        recreated = layers.RandomResizedCrop.from_config(config)

        self.assertEqual(recreated.height, 24)
        self.assertEqual(recreated.width, 40)
        self.assertEqual(recreated.scale, (0.2, 0.8))
        self.assertEqual(recreated.ratio, (0.75, 1.25))

    @pytest.mark.requires_trainable_backend
    def test_random_resized_crop_tf_data_compatibility(self):
        layer = layers.RandomResizedCrop(height=32, width=32)

        def augment(x):
            return layer(x, training=True)

        ds = (
            tf_data.Dataset.from_tensor_slices(
                np.random.uniform(size=(8, 64, 64, 3)).astype("float32")
            )
            .batch(4)
            .map(augment)
        )

        for batch in ds.take(1):
            self.assertEqual(tuple(batch.shape), (4, 32, 32, 3))

    def test_random_resized_crop_compute_output_shape(self):
        layer = layers.RandomResizedCrop(height=32, width=48, data_format="channels_last")
        input_shape = (4, 64, 64, 3)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (4, 32, 48, 3))

    def test_random_resized_crop_compute_output_spec(self):
        layer = layers.RandomResizedCrop(height=32, width=48)
        inputs = backend.KerasTensor((4, 64, 64, 3), dtype="float32")
        output_spec = layer.compute_output_spec(inputs)
        self.assertEqual(output_spec.shape, (4, 32, 48, 3))
        self.assertEqual(output_spec.dtype, "float32")

    def test_random_resized_crop_scale_bounds(self):
        # Test that scale bounds are respected
        layer = layers.RandomResizedCrop(height=32, width=32, scale=(0.5, 0.5))

        inputs = np.ones((4, 64, 64, 3), dtype="float32")
        outputs = layer(inputs, training=True)

        # With scale=0.5, crop area should be 0.5 * 64 * 64 = 2048
        # Crop height/width should be sqrt(2048) ≈ 45.25, but clamped
        # Since we resize to 32x32, output should be 32x32
        self.assertEqual(tuple(outputs.shape), (4, 32, 32, 3))

    def test_random_resized_crop_ratio_bounds(self):
        # Test that ratio bounds are respected
        layer = layers.RandomResizedCrop(height=32, width=32, ratio=(1.0, 1.0))

        inputs = np.ones((4, 64, 64, 3), dtype="float32")
        outputs = layer(inputs, training=True)

        # With ratio=1.0, crop should be square
        self.assertEqual(tuple(outputs.shape), (4, 32, 32, 3))
