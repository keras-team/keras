import numpy as np
import pytest
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class RandomContrastTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.RandomContrast,
            init_kwargs={
                "factor": 0.75,
                "value_range": (0, 255),
                "seed": 1,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )
        self.run_layer_test(
            layers.RandomContrast,
            init_kwargs={
                "factor": 0.75,
                "value_range": (0, 255),
                "seed": 1,
                "data_format": "channels_first",
            },
            input_shape=(8, 3, 4, 4),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 4),
        )

    def test_random_contrast_with_value_range_0_to_255(self):
        seed = 9809
        np.random.seed(seed)

        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            inputs = np.random.random((12, 8, 16, 3))
            height_axis = -3
            width_axis = -2
        else:
            inputs = np.random.random((12, 3, 8, 16))
            height_axis = -2
            width_axis = -1

        inputs = backend.convert_to_tensor(inputs, dtype="float32")
        layer = layers.RandomContrast(
            factor=0.5, value_range=(0, 255), seed=seed
        )
        transformation = layer.get_random_transformation(inputs, training=True)
        outputs = layer.transform_images(inputs, transformation, training=True)

        # Actual contrast arithmetic
        np.random.seed(seed)
        factor = backend.convert_to_numpy(transformation["contrast_factor"])
        inputs = backend.convert_to_numpy(inputs)
        inp_mean = np.mean(inputs, axis=height_axis, keepdims=True)
        inp_mean = np.mean(inp_mean, axis=width_axis, keepdims=True)
        actual_outputs = (inputs - inp_mean) * factor + inp_mean
        outputs = backend.convert_to_numpy(outputs)
        actual_outputs = np.clip(actual_outputs, 0, 255)

        self.assertAllClose(outputs, actual_outputs)

    def test_random_contrast_with_value_range_0_to_1(self):
        seed = 9809
        np.random.seed(seed)

        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            inputs = np.random.random((12, 8, 16, 3))
            height_axis = -3
            width_axis = -2
        else:
            inputs = np.random.random((12, 3, 8, 16))
            height_axis = -2
            width_axis = -1

        inputs = backend.convert_to_tensor(inputs, dtype="float32")
        layer = layers.RandomContrast(factor=0.5, value_range=(0, 1), seed=seed)
        transformation = layer.get_random_transformation(inputs, training=True)
        outputs = layer.transform_images(inputs, transformation, training=True)

        # Actual contrast arithmetic
        np.random.seed(seed)
        factor = backend.convert_to_numpy(transformation["contrast_factor"])
        inputs = backend.convert_to_numpy(inputs)
        inp_mean = np.mean(inputs, axis=height_axis, keepdims=True)
        inp_mean = np.mean(inp_mean, axis=width_axis, keepdims=True)
        actual_outputs = (inputs - inp_mean) * factor + inp_mean
        outputs = backend.convert_to_numpy(outputs)
        actual_outputs = np.clip(actual_outputs, 0, 1)

        self.assertAllClose(outputs, actual_outputs)

    def test_tf_data_compatibility(self):
        layer = layers.RandomContrast(factor=0.5, seed=1337)
        input_data = np.random.random((2, 8, 8, 3))
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        next(iter(ds)).numpy()

    def test_dict_input(self):
        layer = layers.RandomContrast(factor=0.1, bounding_box_format="xyxy")
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
