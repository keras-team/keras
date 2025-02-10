import numpy as np
import pytest
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class RandomPerspectiveTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.RandomPerspective,
            init_kwargs={
                "factor": 1.0,
                "scale": 0.5,
                "interpolation": "bilinear",
                "fill_value": 0,
                "seed": 1,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    def test_random_perspective_inference(self):
        seed = 3481
        layer = layers.RandomPerspective()

        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_random_perspective_no_op(self):
        seed = 3481
        layer = layers.RandomPerspective(factor=0)

        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs)
        self.assertAllClose(inputs, output)

        layer = layers.RandomPerspective(scale=0)

        np.random.seed(seed)
        inputs = np.random.rand(224, 224, 3)
        output = layer(inputs)
        self.assertAllClose(inputs, output, atol=1e-2, rtol=1e-2)

    def test_random_perspective_basic(self):
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

        layer = layers.RandomPerspective(data_format=data_format)

        transformation = {
            "apply_perspective": np.asarray([True]),
            "input_shape": inputs.shape,
            "perspective_factor": np.asarray(
                [
                    [
                        [-0.5, -0.5],
                        [-0.5, -0.5],
                        [-0.5, -0.5],
                        [-0.5, -0.5],
                    ]
                ]
            ),
            "fill_value": 0,
        }

        output = layer.transform_images(inputs, transformation)

        self.assertAllClose(expected_output, output, atol=1e-4, rtol=1e-4)

    def test_tf_data_compatibility(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3))
        else:
            input_data = np.random.random((2, 3, 8, 8))
        layer = layers.RandomPerspective(data_format=data_format)

        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output.numpy()

    @parameterized.named_parameters(
        (
                "with_negative_shift",
                [
                    [-0.1319, -0.1157],
                    [-0.0469, -0.0745],
                    [-0.0491, -0.0047],
                    [-0.0586, -0.0155],
                ],
                [
                    [
                        [1.9133, 1.0001, 3.8251, 3.0013],
                        [5.6804, 3.9589, 7.5711, 5.9405],
                    ]
                ],
        ),
        (
                "with_positive_shift",
                [
                    [0.1319, 0.1157],
                    [0.0469, 0.0745],
                    [0.0491, 0.0047],
                    [0.0586, 0.0155],
                ],
                [
                    [
                        [2.0806, 0.9979, 4.1840, 3.0102],
                        [6.3028, 4.0308, 8.0000, 6.0797],
                    ]
                ],
        ),
    )
    def test_random_perspective_bounding_boxes(self, factor, expected_boxes):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            image_shape = (10, 8, 3)
        else:
            image_shape = (3, 10, 8)
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
        layer = layers.RandomPerspective(
            data_format=data_format,
            seed=42,
            bounding_box_format="xyxy",
        )

        transformation = {
            "apply_perspective": np.asarray([True]),
            "perspective_factor": np.asarray(factor),
            "input_shape": image_shape,
        }
        output = layer.transform_bounding_boxes(
            input_data["bounding_boxes"], transformation
        )

        self.assertAllClose(
            output["boxes"], expected_boxes, atol=1e-3, rtol=1e-3
        )

    @parameterized.named_parameters(
        (
                "with_negative_shift",
                [
                    [-0.1319, -0.1157],
                    [-0.0469, -0.0745],
                    [-0.0491, -0.0047],
                    [-0.0586, -0.0155],
                ],
                [
                    [
                        [1.9133, 1.0001, 3.8251, 3.0013],
                        [5.6804, 3.9589, 7.5711, 5.9405],
                    ]
                ],
        ),
        (
                "with_positive_shift",
                [
                    [0.1319, 0.1157],
                    [0.0469, 0.0745],
                    [0.0491, 0.0047],
                    [0.0586, 0.0155],
                ],
                [
                    [
                        [2.0806, 0.9979, 4.1840, 3.0102],
                        [6.3028, 4.0308, 8.0000, 6.0797],
                    ]
                ],
        ),
    )
    def test_random_flip_tf_data_bounding_boxes(self, factor, expected_boxes):
        data_format = backend.config.image_data_format()
        if backend.config.image_data_format() == "channels_last":
            image_shape = (1, 10, 8, 3)
        else:
            image_shape = (1, 3, 10, 8)
        input_image = np.random.random(image_shape)
        bounding_boxes = {
            "boxes": np.array(
                [
                    [
                        [
                            [2, 1, 4, 3],
                            [6, 4, 8, 6],
                        ]
                    ]
                ]
            ),
            "labels": np.array([[1, 2]]),
        }

        input_data = {"images": input_image, "bounding_boxes": bounding_boxes}

        ds = tf_data.Dataset.from_tensor_slices(input_data)
        layer = layers.RandomPerspective(
            data_format=data_format,
            seed=42,
            bounding_box_format="xyxy",
        )

        transformation = {
            "apply_perspective": np.asarray([True]),
            "perspective_factor": np.asarray(factor),
            "input_shape": image_shape,
        }

        ds = ds.map(
            lambda x: layer.transform_bounding_boxes(
                x["bounding_boxes"],
                transformation=transformation,
                training=True,
            )
        )

        output = next(iter(ds))
        expected_boxes = np.array(expected_boxes)
        self.assertAllClose(
            output["boxes"], expected_boxes, atol=1e-3, rtol=1e-3
        )
