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
                "seed": 1,
                "data_format": "channels_first",
            },
            input_shape=(8, 3, 4, 4),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 4),
        )

    def test_random_contrast(self):
        seed = 9809
        np.random.seed(seed)
        inputs = np.random.random((12, 8, 16, 3))
        layer = layers.RandomContrast(factor=0.5, seed=seed)
        outputs = layer(inputs)

        # Actual contrast arithmetic
        np.random.seed(seed)
        factor = np.random.uniform(0.5, 1.5)
        inp_mean = np.mean(inputs, axis=-3, keepdims=True)
        inp_mean = np.mean(inp_mean, axis=-2, keepdims=True)
        actual_outputs = (inputs - inp_mean) * factor + inp_mean
        outputs = backend.convert_to_numpy(outputs)
        actual_outputs = np.clip(outputs, 0, 255)

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
