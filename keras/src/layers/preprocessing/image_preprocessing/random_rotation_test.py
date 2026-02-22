import numpy as np
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import testing


class RandomRotationTest(testing.TestCase):
    @parameterized.named_parameters(
        ("random_rotate_neg4", -0.4),
        ("random_rotate_neg2", -0.2),
        ("random_rotate_4", 0.4),
        ("random_rotate_2", 0.2),
        ("random_rotate_tuple", (-0.2, 0.4)),
    )
    def test_random_rotation_shapes(self, factor):
        self.run_layer_test(
            layers.RandomRotation,
            init_kwargs={
                "factor": factor,
            },
            input_shape=(2, 3, 4),
            expected_output_shape=(2, 3, 4),
            supports_masking=False,
            run_training_check=False,
        )

    def test_random_rotation_correctness(self):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (1, 5, 5, 1)
        else:
            input_shape = (1, 1, 5, 5)
        input_image = np.reshape(np.arange(0, 25), input_shape)
        layer = layers.RandomRotation(factor=(0.5, 0.5))
        actual_output = layer(input_image)
        expected_output = np.asarray(
            [
                [24, 23, 22, 21, 20],
                [19, 18, 17, 16, 15],
                [14, 13, 12, 11, 10],
                [9, 8, 7, 6, 5],
                [4, 3, 2, 1, 0],
            ]
        ).reshape(input_shape)

        self.assertAllClose(
            backend.convert_to_tensor(expected_output), actual_output, atol=1e-5
        )

    def test_training_false(self):
        input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1))
        layer = layers.RandomRotation(factor=(0.5, 0.5))
        actual_output = layer(input_image, training=False)
        self.assertAllClose(actual_output, input_image)

    def test_tf_data_compatibility(self):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (1, 5, 5, 1)
        else:
            input_shape = (1, 1, 5, 5)
        input_image = np.reshape(np.arange(0, 25), input_shape)
        layer = layers.RandomRotation(factor=(0.5, 0.5))

        ds = tf_data.Dataset.from_tensor_slices(input_image).map(layer)
        expected_output = np.asarray(
            [
                [24, 23, 22, 21, 20],
                [19, 18, 17, 16, 15],
                [14, 13, 12, 11, 10],
                [9, 8, 7, 6, 5],
                [4, 3, 2, 1, 0],
            ]
        ).reshape(input_shape[1:])
        output = next(iter(ds)).numpy()
        self.assertAllClose(expected_output, output)

    def test_random_rotation_fill_mode_crop(self):
        """Test that `fill_mode="crop"` is accepted by the layer."""
        layer = layers.RandomRotation(factor=0.2, fill_mode="crop")
        self.assertEqual(layer.fill_mode, "crop")

    def test_random_rotation_crop_output_shape(self):
        """Test that crop mode preserves the input shape for batched inputs."""
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 32, 48, 3)
        else:
            input_shape = (2, 3, 32, 48)

        layer = layers.RandomRotation(factor=0.2, fill_mode="crop", seed=42)
        images = np.random.randint(0, 256, input_shape, dtype="uint8")

        output = layer(images, training=True)
        self.assertEqual(output.shape, images.shape)

    def test_random_rotation_crop_dict_input(self):
        """Test crop mode with dict inputs (images + segmentation masks)."""
        if backend.config.image_data_format() == "channels_last":
            image_shape = (2, 32, 32, 3)
            mask_shape = (2, 32, 32, 1)
        else:
            image_shape = (2, 3, 32, 32)
            mask_shape = (2, 1, 32, 32)

        layer = layers.RandomRotation(factor=0.2, fill_mode="crop", seed=42)
        masks = np.random.randint(0, 5, mask_shape, dtype="uint8")
        data = {
            "images": np.random.randint(0, 256, image_shape, dtype="uint8"),
            "segmentation_masks": masks,
        }

        result = layer(data, training=True)
        self.assertIsInstance(result, dict)
        self.assertIn("images", result)
        self.assertIn("segmentation_masks", result)
        self.assertEqual(result["images"].shape, image_shape)
        self.assertEqual(result["segmentation_masks"].shape, mask_shape)

        out_masks = ops.convert_to_numpy(result["segmentation_masks"])
        in_unique = set(np.unique(masks.astype("float32")))
        out_unique = set(np.unique(out_masks))

        max_allowed = in_unique | {0}
        unexpected = out_unique - max_allowed

        self.assertEqual(
            len(unexpected),
            0,
            msg=f"Output has unexpected values: {unexpected}",
        )

    def test_random_rotation_invalid_fill_mode(self):
        """Test that an invalid `fill_mode` raises `NotImplementedError`."""
        with self.assertRaisesRegex(NotImplementedError, "Unknown `fill_mode`"):
            layers.RandomRotation(factor=0.2, fill_mode="invalid_mode")

    def test_random_rotation_crop_avoids_fill_artifacts(self):
        """Test that crop mode avoids introducing fill artifacts via zoom."""
        if backend.config.image_data_format() == "channels_last":
            image_shape = (1, 32, 48, 1)
        else:
            image_shape = (1, 1, 32, 48)

        images = np.ones(image_shape, dtype="float32")

        angle_factor = (0.25, 0.25)
        fill_value = 123.0

        layer_constant = layers.RandomRotation(
            factor=angle_factor,
            fill_mode="constant",
            fill_value=fill_value,
            interpolation="nearest",
            seed=1337,
        )
        layer_crop = layers.RandomRotation(
            factor=angle_factor,
            fill_mode="crop",
            interpolation="nearest",
            seed=1337,
        )

        out_constant = ops.convert_to_numpy(
            layer_constant(images, training=True)
        )
        out_crop = ops.convert_to_numpy(layer_crop(images, training=True))

        constant_fill_count = np.sum(np.isclose(out_constant, fill_value))
        crop_fill_count = np.sum(np.isclose(out_crop, fill_value))

        self.assertGreater(constant_fill_count, 0)
        self.assertEqual(crop_fill_count, 0)

    def test_random_rotation_crop_unbatched_preserves_shape(self):
        """Test that crop mode preserves shape for unbatched inputs."""
        if backend.config.image_data_format() == "channels_last":
            x = np.ones((32, 48, 1), dtype="float32")
        else:
            x = np.ones((1, 32, 48), dtype="float32")

        layer = layers.RandomRotation(
            factor=(0.25, 0.25), fill_mode="crop", seed=7
        )
        y = layer(x, training=True)
        self.assertEqual(y.shape, x.shape)

    def test_random_rotation_crop_with_bounding_boxes(self):
        """Test crop mode works with bounding boxes."""
        if backend.config.image_data_format() == "channels_last":
            image_shape = (2, 224, 224, 3)
        else:
            image_shape = (2, 3, 224, 224)

        layer = layers.RandomRotation(
            factor=0.2, fill_mode="crop", bounding_box_format="xyxy", seed=42
        )

        boxes = {
            "boxes": np.array(
                [
                    [[10, 10, 50, 50], [60, 60, 100, 100]],
                    [[20, 20, 80, 80], [120, 120, 180, 180]],
                ],
                dtype="float32",
            ),
            "labels": np.array([[1, 2], [3, 4]], dtype="int32"),
        }

        data = {
            "images": np.random.randint(0, 256, image_shape, dtype="uint8"),
            "bounding_boxes": boxes,
        }

        result = layer(data, training=True)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["images"].shape, image_shape)
        self.assertEqual(result["bounding_boxes"]["boxes"].shape, (2, 2, 4))
        self.assertEqual(result["bounding_boxes"]["labels"].shape, (2, 2))

    def test_random_rotation_crop_large_angle_no_artifacts(self):
        """Crop mode removes fill artifacts even for large angles."""
        if backend.config.image_data_format() == "channels_last":
            shape = (1, 100, 100, 1)
        else:
            shape = (1, 1, 100, 100)

        images = np.ones(shape, dtype="float32")
        layer = layers.RandomRotation(
            factor=(0.125, 0.125), fill_mode="crop", seed=42
        )
        output = ops.convert_to_numpy(layer(images, training=True))

        self.assertEqual(output.shape, shape)

        fill_ratio = np.mean(output < 1e-3)
        self.assertLess(
            fill_ratio,
            0.001,
            f"Crop output still contains fill artifacts: {fill_ratio:.2%}",
        )

    def test_random_rotation_crop_batch_no_artifacts(self):
        """Batch crop uses worst-case angle but still removes fill artifacts."""
        if backend.config.image_data_format() == "channels_last":
            shape = (2, 100, 100, 1)
        else:
            shape = (2, 1, 100, 100)

        images = np.ones(shape, dtype="float32")
        layer = layers.RandomRotation(
            factor=(0.125, 0.125), fill_mode="crop", seed=42
        )
        output = ops.convert_to_numpy(layer(images, training=True))

        self.assertEqual(output.shape, shape)

        fill_ratio = np.mean(output < 1e-3)
        self.assertLess(
            fill_ratio,
            0.06,
            f"Batch crop still contains fill artifacts: {fill_ratio:.2%}",
        )

    def test_crop_zoom_monotonic(self):
        """Scale decreases (more zoom-out) as rotation angle increases."""
        layer = layers.RandomRotation(factor=0.25, fill_mode="crop")
        height = ops.cast(64, "float32")
        width = ops.cast(64, "float32")
        small_angle = ops.convert_to_tensor([10.0])
        large_angle = ops.convert_to_tensor([40.0])

        small_scale = layer._get_rotation_scale(height, width, small_angle)
        large_scale = layer._get_rotation_scale(height, width, large_angle)

        self.assertLessEqual(
            large_scale,
            small_scale,
            "Larger angle should require smaller scale (more zoom-out)",
        )
        self.assertLess(small_scale, 1.0)
        self.assertGreater(large_scale, 0.0)
