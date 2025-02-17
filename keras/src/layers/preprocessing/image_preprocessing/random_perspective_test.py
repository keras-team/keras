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
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs)
        self.assertAllClose(inputs, output)

    def test_random_perspective_basic(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            inputs = np.ones((2, 2, 1))
            expected_output = np.array(
                [[[[1.0], [0.89476013]], [[0.84097195], [0.6412543]]]]
            )

        else:
            inputs = np.ones((1, 2, 2))
            expected_output = np.array([[[[1.0000, 0.8948], [0.8410, 0.6413]]]])

        layer = layers.RandomPerspective(data_format=data_format)

        transformation = {
            "apply_perspective": np.asarray([True]),
            "perspective_factor": np.asarray(
                [[0.05261999, 0.07951406, 0.05261999, 0.07951406]]
            ),
        }

        output = layer.transform_images(inputs, transformation)

        print(output)

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
            [[-0.5, -0.1, -0.3, -0.2]],
            [
                [
                    [6.6750007, 2.0750003, 8.0, 5.0750003],
                    [8.0, 6.8250003, 8.0, 9.825],
                ]
            ],
        ),
        (
            "with_positive_shift",
            [[0.5, 0.1, 0.3, 0.2]],
            [
                [
                    [0.29375008, 0.51874995, 2.2937498, 2.5187497],
                    [2.1062498, 3.0812497, 4.10625, 5.08125],
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
                    [2, 1, 4, 3],
                    [6, 4, 8, 6],
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
            input_data["bounding_boxes"],
            transformation=transformation,
            training=True,
        )

        print(output)

        self.assertAllClose(output["boxes"], expected_boxes)

    @parameterized.named_parameters(
        (
            "with_negative_shift",
            [[-0.5, -0.1, -0.3, -0.2]],
            [
                [
                    [6.6750007, 2.0750003, 8.0, 5.0750003],
                    [8.0, 6.8250003, 8.0, 9.825],
                ]
            ],
        ),
        (
            "with_positive_shift",
            [[0.5, 0.1, 0.3, 0.2]],
            [
                [
                    [0.29375008, 0.51874995, 2.2937498, 2.5187497],
                    [2.1062498, 3.0812497, 4.10625, 5.08125],
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
                        [2, 1, 4, 3],
                        [6, 4, 8, 6],
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
        self.assertAllClose(output["boxes"], expected_boxes)
