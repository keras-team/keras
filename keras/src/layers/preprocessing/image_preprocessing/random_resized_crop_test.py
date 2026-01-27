"""Tests for RandomResizedCrop layer."""

import numpy as np
import pytest

from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import testing

SKIP_BACKENDS = ["openvino"]

pytestmark = pytest.mark.skipif(
    backend.backend() in SKIP_BACKENDS,
    reason=(
        "RandomResizedCrop tests not supported for backend: {}".format(
            backend.backend()
        )
    ),
)


class RandomResizedCropTest(testing.TestCase):
    """Test suite for RandomResizedCrop layer."""

    def test_random_resized_crop_inference_deterministic(self):
        """Test that inference mode is deterministic."""
        seed = 3481
        layer = layers.RandomResizedCrop(224, 224, seed=seed)
        np.random.seed(seed)
        inputs = np.random.random((2, 256, 256, 3)).astype("float32")

        output1 = layer(inputs, training=False)
        output2 = layer(inputs, training=False)

        self.assertAllClose(output1, output2)

    def test_random_resized_crop_training_random(self):
        """Test that training mode produces random results."""
        seed = 3481
        layer = layers.RandomResizedCrop(224, 224, seed=seed)
        np.random.seed(seed)
        inputs = np.random.random((1, 300, 300, 3)).astype("float32")

        output1 = layer(inputs, training=True)
        output2 = layer(inputs, training=True)

        output1_np = ops.convert_to_numpy(output1)
        output2_np = ops.convert_to_numpy(output2)
        diff = np.mean(np.abs(output1_np - output2_np))
        self.assertGreater(float(diff), 1e-4)

    def test_random_resized_crop_output_shape(self):
        """Test output shape is correct."""
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 256, 256, 3)).astype("float32")
        else:
            input_data = np.random.random((2, 3, 256, 256)).astype("float32")

        layer = layers.RandomResizedCrop(
            224,
            224,
            data_format=data_format,
            seed=1337,
        )
        output = layer(input_data, training=True)

        if data_format == "channels_last":
            self.assertEqual(output.shape, (2, 224, 224, 3))
        else:
            self.assertEqual(output.shape, (2, 3, 224, 224))

    def test_random_resized_crop_unbatched_input(self):
        """Test with unbatched (3D) input."""
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((256, 256, 3)).astype("float32")
        else:
            input_data = np.random.random((3, 256, 256)).astype("float32")

        layer = layers.RandomResizedCrop(
            224,
            224,
            data_format=data_format,
            seed=1337,
        )
        output = layer(input_data, training=True)

        if data_format == "channels_last":
            self.assertEqual(output.shape, (224, 224, 3))
        else:
            self.assertEqual(output.shape, (3, 224, 224))

    def test_random_resized_crop_seed_reproducibility(self):
        """Test that same seed produces same results."""
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 256, 256, 3)).astype("float32")
        else:
            input_data = np.random.random((2, 3, 256, 256)).astype("float32")

        layer1 = layers.RandomResizedCrop(
            224,
            224,
            scale=(0.08, 1.0),
            ratio=(0.75, 1.33),
            data_format=data_format,
            seed=1337,
        )
        layer2 = layers.RandomResizedCrop(
            224,
            224,
            scale=(0.08, 1.0),
            ratio=(0.75, 1.33),
            data_format=data_format,
            seed=1337,
        )

        output1 = layer1(input_data, training=True)
        output2 = layer2(input_data, training=True)

        self.assertAllClose(output1, output2)

    def test_random_resized_crop_different_seeds(self):
        """Test that different seeds produce different results."""
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 256, 256, 3)).astype("float32")
        else:
            input_data = np.random.random((2, 3, 256, 256)).astype("float32")

        layer1 = layers.RandomResizedCrop(
            224, 224, seed=123, data_format=data_format
        )
        layer2 = layers.RandomResizedCrop(
            224, 224, seed=456, data_format=data_format
        )

        output1 = layer1(input_data, training=True)
        output2 = layer2(input_data, training=True)

        output1_np = ops.convert_to_numpy(output1)
        output2_np = ops.convert_to_numpy(output2)
        diff = np.mean(np.abs(output1_np - output2_np))
        self.assertGreater(float(diff), 1e-4)

    def test_random_resized_crop_custom_parameters(self):
        """Test with custom scale and ratio parameters."""
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 256, 256, 3)).astype("float32")
        else:
            input_data = np.random.random((2, 3, 256, 256)).astype("float32")

        layer = layers.RandomResizedCrop(
            224,
            224,
            scale=(0.5, 1.0),
            ratio=(0.9, 1.1),
            data_format=data_format,
            seed=1337,
        )
        output = layer(input_data, training=True)

        if data_format == "channels_last":
            self.assertEqual(output.shape, (2, 224, 224, 3))
        else:
            self.assertEqual(output.shape, (2, 3, 224, 224))

    def test_random_resized_crop_config_serialization(self):
        """Test layer serialization and deserialization."""
        layer = layers.RandomResizedCrop(
            224,
            224,
            scale=(0.08, 1.0),
            ratio=(0.75, 1.33),
            interpolation="bilinear",
            seed=1337,
        )

        config = layer.get_config()
        new_layer = layers.RandomResizedCrop.from_config(config)

        self.assertEqual(new_layer.height, 224)
        self.assertEqual(new_layer.width, 224)
        self.assertEqual(new_layer.scale, (0.08, 1.0))
        self.assertEqual(new_layer.ratio, (0.75, 1.33))
        self.assertEqual(new_layer.interpolation, "bilinear")
        self.assertEqual(new_layer.seed, 1337)

    def test_random_resized_crop_compute_output_shape(self):
        """Test compute_output_shape method."""
        layer = layers.RandomResizedCrop(224, 224)

        output_shape = layer.compute_output_shape((8, 256, 256, 3))
        self.assertEqual(output_shape, (8, 224, 224, 3))

        output_shape = layer.compute_output_shape((256, 256, 3))
        self.assertEqual(output_shape, (224, 224, 3))

        output_shape = layer.compute_output_shape((None, 256, 256, 3))
        self.assertEqual(output_shape, (None, 224, 224, 3))

    def test_random_resized_crop_non_square_target(self):
        """Test with non-square target size."""
        layer = layers.RandomResizedCrop(320, 224, seed=1337)
        input_data = np.random.random((2, 400, 400, 3)).astype("float32")
        output = layer(input_data, training=True)

        output_np = ops.convert_to_numpy(output)
        self.assertEqual(output_np.shape, (2, 320, 224, 3))

    def test_random_resized_crop_rectangular_input(self):
        """Test with rectangular (non-square) input."""
        layer = layers.RandomResizedCrop(224, 224, seed=1337)
        input_data = np.random.random((2, 300, 500, 3)).astype("float32")
        output = layer(input_data, training=False)

        output_np = ops.convert_to_numpy(output)
        self.assertEqual(output_np.shape, (2, 224, 224, 3))

    def test_random_resized_crop_dtype_float32(self):
        """Test with float32 input."""
        layer = layers.RandomResizedCrop(224, 224, seed=1337)
        input_data = np.random.random((2, 256, 256, 3)).astype("float32")
        output = layer(input_data, training=True)

        output_np = ops.convert_to_numpy(output)
        self.assertEqual(output_np.dtype, np.float32)
