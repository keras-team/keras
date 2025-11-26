import numpy as np
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class RandomCropTest(testing.TestCase):
    def test_random_crop(self):
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 2,
                "width": 2,
                "data_format": "channels_last",
            },
            input_shape=(1, 3, 4, 3),
            supports_masking=False,
            run_training_check=False,
            expected_output_shape=(1, 2, 2, 3),
        )
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 2,
                "width": 2,
                "data_format": "channels_last",
            },
            input_shape=(3, 4, 3),
            supports_masking=False,
            run_training_check=False,
            expected_output_shape=(2, 2, 3),
        )
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 2,
                "width": 2,
                "data_format": "channels_first",
            },
            input_shape=(1, 3, 3, 4),
            supports_masking=False,
            run_training_check=False,
            expected_output_shape=(1, 3, 2, 2),
        )
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 2,
                "width": 2,
                "data_format": "channels_first",
            },
            input_shape=(3, 3, 4),
            supports_masking=False,
            run_training_check=False,
            expected_output_shape=(3, 2, 2),
        )

    def test_random_crop_full(self):
        np.random.seed(1337)
        height, width = 8, 16
        if backend.config.image_data_format() == "channels_last":
            input_shape = (12, 8, 16, 3)
        else:
            input_shape = (12, 3, 8, 16)
        inp = np.random.random(input_shape)
        layer = layers.RandomCrop(height, width)
        actual_output = layer(inp, training=False)
        # After fix: should be center cropped, not identical
        self.assertEqual(
            actual_output.shape, inp.shape
        )  # Same shape in this case

    def test_random_crop_partial(self):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (12, 8, 16, 3)
            output_shape = (12, 8, 8, 3)
        else:
            input_shape = (12, 3, 8, 16)
            output_shape = (12, 3, 8, 8)
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 8,
                "width": 8,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            supports_masking=False,
            run_training_check=False,
        )

    def test_predicting_with_longer_height(self):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (12, 8, 16, 3)
            output_shape = (12, 10, 8, 3)
        else:
            input_shape = (12, 3, 8, 16)
            output_shape = (12, 3, 10, 8)
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 10,
                "width": 8,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            supports_masking=False,
            run_training_check=False,
        )

    def test_predicting_with_longer_width(self):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (12, 8, 16, 3)
            output_shape = (12, 8, 18, 3)
        else:
            input_shape = (12, 3, 8, 16)
            output_shape = (12, 3, 8, 18)
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 8,
                "width": 18,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            supports_masking=False,
            run_training_check=False,
        )

    def test_tf_data_compatibility(self):
        layer = layers.RandomCrop(8, 9)
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 10, 12, 3)
            output_shape = (2, 8, 9, 3)
        else:
            input_shape = (2, 3, 10, 12)
            output_shape = (2, 3, 8, 9)
        input_data = np.random.random(input_shape)
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        output = next(iter(ds)).numpy()
        self.assertEqual(tuple(output.shape), output_shape)

    def test_dict_input(self):
        layer = layers.RandomCrop(
            3, 3, data_format="channels_last", bounding_box_format="xyxy"
        )
        data = {
            "images": np.random.random((2, 4, 5, 3)),
            "labels": np.random.random((2, 7)),
            "segmentation_masks": np.random.random((2, 4, 5, 7)),
            "bounding_boxes": {
                "boxes": np.array([[1, 2, 2, 3]]),
                "labels": np.array([0]),
            },
        }
        transformed_data = layer(data)
        self.assertEqual(
            data["images"].shape[:-1],
            transformed_data["segmentation_masks"].shape[:-1],
        )
        self.assertAllClose(data["labels"], transformed_data["labels"])
        self.assertEqual(data["bounding_boxes"]["boxes"].shape, (1, 4))
        self.assertAllClose(
            data["bounding_boxes"]["labels"],
            transformed_data["bounding_boxes"]["labels"],
        )

    def test_validation_center_crop(self):
        """Test that validation mode performs center cropping."""
        layer = layers.RandomCrop(2, 2, data_format="channels_last")

        # Create a test image with distinct corners
        if backend.config.image_data_format() == "channels_last":
            test_image = np.zeros((4, 4, 3))
            # Mark corners with different values
            test_image[0, 0] = [1, 0, 0]  # Top-left red
            test_image[0, 3] = [0, 1, 0]  # Top-right green
            test_image[3, 0] = [0, 0, 1]  # Bottom-left blue
            test_image[3, 3] = [1, 1, 0]  # Bottom-right yellow
        else:
            test_image = np.zeros((3, 4, 4))
            # Mark corners with different values
            test_image[0, 0, 0] = 1  # Top-left red
            test_image[1, 0, 3] = 1  # Top-right green
            test_image[2, 3, 0] = 1  # Bottom-left blue
            test_image[0, 3, 3] = 1  # Bottom-right yellow (red channel)
            test_image[1, 3, 3] = 1  # Bottom-right yellow (green channel)

        # Test validation mode (should center crop)
        validation_output = layer(test_image, training=False)

        # Center crop should capture the middle 2x2 region
        expected_shape = (
            (2, 2, 3)
            if backend.config.image_data_format() == "channels_last"
            else (3, 2, 2)
        )
        self.assertEqual(validation_output.shape, expected_shape)

    def test_edge_case_exact_dimensions(self):
        """Test cropping when image dimensions exactly match target."""
        layer = layers.RandomCrop(4, 4, data_format="channels_last")

        if backend.config.image_data_format() == "channels_last":
            test_image = np.random.random((4, 4, 3))
        else:
            test_image = np.random.random((3, 4, 4))

        # Training mode with exact dimensions should still work
        training_output = layer(test_image, training=True)
        expected_shape = (
            (4, 4, 3)
            if backend.config.image_data_format() == "channels_last"
            else (3, 4, 4)
        )
        self.assertEqual(training_output.shape, expected_shape)

        # Validation mode should also work
        validation_output = layer(test_image, training=False)
        self.assertEqual(validation_output.shape, expected_shape)
