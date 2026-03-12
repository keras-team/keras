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
            "apply_perspective": np.array([True]),
            "start_points": np.array(
                [[[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.0, 3.0]]]
            ),
            "end_points": np.array([[[0.0, 0.0], [1, 0.0], [0.0, 1], [1, 1]]]),
            "input_shape": np.array((4, 4, 1)),
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
            "with_large_scale",
            [
                [
                    [0.0, 0.0],
                    [8.151311, 0.0],
                    [0.0, 12.695701],
                    [9.2712054, 10.524198],
                ]
            ],
            [
                [
                    [2.6490488, 1.1149256, 5.2026834, 3.6187303],
                    [7.5547166, 4.2492595, 8.0, 6.869391],
                ]
            ],
        ),
        (
            "with_small_scale",
            [
                [
                    [0.0, 0.0],
                    [4.151311, 0.0],
                    [0.0, 6.695701],
                    [4.2712054, 7.524198],
                ]
            ],
            [
                [
                    [1.095408, 0.7504317, 2.2761598, 2.3389952],
                    [3.5416048, 3.2349987, 4.920989, 5.0568376],
                ]
            ],
        ),
    )
    def test_random_perspective_bounding_boxes(
        self, end_points, expected_boxes
    ):
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
            # data_format=data_format,
            seed=42,
            bounding_box_format="xyxy",
        )

        transformation = {
            "apply_perspective": np.array([True]),
            "end_points": np.array(end_points),
            "input_shape": np.array(image_shape),
            "start_points": np.array(
                [[[0.0, 0.0], [7.0, 0.0], [0.0, 9.0], [7.0, 9.0]]]
            ),
        }

        output = layer.transform_bounding_boxes(
            input_data["bounding_boxes"],
            transformation,
        )

        self.assertAllClose(
            output["boxes"],
            expected_boxes,
            atol=1e-3,
            rtol=1e-3,
            tpu_atol=1e-2,
            tpu_rtol=1e-2,
        )

    @parameterized.named_parameters(
        (
            "with_large_scale",
            [
                [
                    [0.0, 0.0],
                    [8.151311, 0.0],
                    [0.0, 12.695701],
                    [9.2712054, 10.524198],
                ]
            ],
            [
                [
                    [2.6490488, 1.1149256, 5.2026834, 3.6187303],
                    [7.5547166, 4.2492595, 8.0, 6.869391],
                ]
            ],
        ),
        (
            "with_small_scale",
            [
                [
                    [0.0, 0.0],
                    [4.151311, 0.0],
                    [0.0, 6.695701],
                    [4.2712054, 7.524198],
                ]
            ],
            [
                [
                    [1.095408, 0.7504317, 2.2761598, 2.3389952],
                    [3.5416048, 3.2349987, 4.920989, 5.0568376],
                ]
            ],
        ),
    )
    def test_random_flip_tf_data_bounding_boxes(
        self, end_points, expected_boxes
    ):
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
            "apply_perspective": np.array([True]),
            "end_points": np.array(end_points),
            "input_shape": np.array(image_shape),
            "start_points": np.array(
                [[[0.0, 0.0], [7.0, 0.0], [0.0, 9.0], [7.0, 9.0]]]
            ),
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
