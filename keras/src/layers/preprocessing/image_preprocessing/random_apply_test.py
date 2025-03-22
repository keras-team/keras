import numpy as np
import pytest
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class RandomApplyTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.RandomApply,
            init_kwargs={
                "transforms": [],
                "factor": 1,
                "seed": 1,
                "data_format": "channels_last",
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    def test_random_apply_inference(self):
        seed = 3481
        layer = layers.RandomApply(
            transforms=[
                layers.RandomInvert(),
                layers.RandomBrightness(factor=1),
            ]
        )

        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_random_apply_no_operations(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3))
        else:
            input_data = np.random.random((2, 3, 8, 8))
        layer = layers.RandomApply(
            transforms=[
                layers.RandomInvert(),
                layers.RandomBrightness(factor=1),
            ],
            factor=(0, 0),
            data_format=data_format,
        )

        augmented_image = layer(input_data)
        self.assertAllClose(
            backend.convert_to_numpy(augmented_image), input_data
        )

    def test_random_apply_basic(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            inputs = np.ones((4, 4, 1))
            expected_output = np.asarray(
                [
                    [[1.0], [1.0], [0.0], [0.0]],
                    [[1.0], [1.0], [0.0], [0.0]],
                    [[0.0], [0.0], [0.0], [0.0]],
                    [[0.0], [0.0], [0.0], [0.0]],
                ],
            )

        else:
            inputs = np.ones((1, 4, 4))
            expected_output = np.array(
                [
                    [
                        [1.0, 1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            )

        layer = layers.RandomApply(
            transforms=[layers.RandomPerspective(data_format=data_format)],
            factor=(1, 1),
            data_format=data_format,
        )

        perspective_transformation = {
            "apply_perspective": np.array([True]),
            "start_points": np.array(
                [[[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.0, 3.0]]]
            ),
            "end_points": np.array([[[0.0, 0.0], [1, 0.0], [0.0, 1], [1, 1]]]),
            "input_shape": np.array((4, 4, 1)),
        }

        transformation = {
            "apply_transform": np.array([True]),
            "transform_values": {
                "RandomPerspective": perspective_transformation
            },
        }

        output = layer.transform_images(inputs, transformation)

        self.assertAllClose(expected_output, output, atol=1e-4, rtol=1e-4)

    def test_tf_data_compatibility(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3))
        else:
            input_data = np.random.random((2, 3, 8, 8))
        layer = layers.RandomApply(
            transforms=[
                layers.RandomInvert(),
                layers.RandomBrightness(factor=1),
            ],
            data_format=data_format,
        )

        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output.numpy()

    def test_random_apply_tf_data_bounding_boxes(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            image_shape = (1, 10, 8, 3)
        else:
            image_shape = (1, 3, 10, 8)
        input_image = np.random.random(image_shape)
        bounding_boxes = {
            "boxes": np.array(
                [
                    [
                        [2, 1, 4, 3],
                        [6, 4, 8, 6],
                    ]
                ]
            ),
            "labels": np.array([[1, 2]]),
        }

        input_data = {"images": input_image, "bounding_boxes": bounding_boxes}

        ds = tf_data.Dataset.from_tensor_slices(input_data)
        layer = layers.RandomApply(
            transforms=[
                layers.RandomInvert(),
                layers.RandomBrightness(factor=1),
            ],
            data_format=data_format,
            seed=42,
            bounding_box_format="xyxy",
        )
        ds.map(layer)
