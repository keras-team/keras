import numpy as np
import pytest
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class RandomBrightnessTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.RandomBrightness,
            init_kwargs={
                "factor": 0.75,
                "value_range": (20, 200),
                "seed": 1,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    def test_random_brightness_inference(self):
        seed = 3481
        layer = layers.RandomBrightness([0, 1.0])
        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_correctness(self):
        seed = 2390

        # Always scale up, but randomly between 0 ~ 255
        layer = layers.RandomBrightness([0.1, 1.0])
        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = backend.convert_to_numpy(layer(inputs))
        diff = output - inputs
        diff = backend.convert_to_numpy(diff)
        self.assertTrue(np.amin(diff) >= 0)
        self.assertTrue(np.mean(diff) > 0)

        # Always scale down, but randomly between 0 ~ 255
        layer = layers.RandomBrightness([-1.0, -0.1])
        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = backend.convert_to_numpy(layer(inputs))
        diff = output - inputs
        self.assertTrue(np.amax(diff) <= 0)
        self.assertTrue(np.mean(diff) < 0)

    def test_tf_data_compatibility(self):
        layer = layers.RandomBrightness(factor=0.5, seed=1337)
        input_data = np.random.random((2, 8, 8, 3))
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output.numpy()

    def test_value_range_incorrect_type(self):
        with self.assertRaisesRegex(
            ValueError,
            "The `value_range` argument should be a list of two numbers.*",
        ):
            layers.RandomBrightness(factor=0.1, value_range="incorrect_type")

    def test_value_range_incorrect_length(self):
        with self.assertRaisesRegex(
            ValueError,
            "The `value_range` argument should be a list of two numbers.*",
        ):
            layers.RandomBrightness(factor=0.1, value_range=[10])

    def test_set_factor_incorrect_length(self):
        layer = layers.RandomBrightness(factor=0.5)
        with self.assertRaisesRegex(
            ValueError, "The `factor` argument should be a number.*"
        ):
            layer._set_factor([0.1])  # Only one element in list

    def test_set_factor_incorrect_type(self):
        layer = layers.RandomBrightness(factor=0.5)
        with self.assertRaisesRegex(
            ValueError, "The `factor` argument should be a number.*"
        ):
            layer._set_factor(
                "invalid_type"
            )  # Passing a string instead of a number or a list/tuple of numbers

    def test_factor_range_below_lower_bound(self):
        with self.assertRaisesRegex(
            ValueError, "The `factor` argument should be a number.*"
        ):
            # Passing a value less than -1.0
            layers.RandomBrightness(factor=-1.1)

    def test_factor_range_above_upper_bound(self):
        with self.assertRaisesRegex(
            ValueError, "The `factor` argument should be a number.*"
        ):
            # Passing a value more than 1.0
            layers.RandomBrightness(factor=1.1)

    def test_randomly_adjust_brightness_input_incorrect_rank(self):
        layer = layers.RandomBrightness(factor=0.1)
        wrong_rank_input = np.random.rand(10, 10)

        with self.assertRaisesRegex(
            ValueError,
            "Expected the input image to be rank 3 or 4.",
        ):
            layer(
                wrong_rank_input, training=True
            )  # Call the method that triggers the error

    def test_dict_input(self):
        layer = layers.RandomBrightness(factor=0.1, bounding_box_format="xyxy")
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
        self.assertAllClose(
            data["bounding_boxes"]["boxes"],
            transformed_data["bounding_boxes"]["boxes"],
        )
        self.assertAllClose(
            data["bounding_boxes"]["labels"],
            transformed_data["bounding_boxes"]["labels"],
        )
