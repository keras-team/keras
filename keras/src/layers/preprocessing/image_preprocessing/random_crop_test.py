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
        self.assertAllClose(inp, actual_output)

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
