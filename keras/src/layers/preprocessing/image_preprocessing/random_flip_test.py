import unittest.mock

import numpy as np
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src import utils


class MockedRandomFlip(layers.RandomFlip):
    def call(self, inputs, training=True):
        unbatched = len(inputs.shape) == 3
        batch_size = 1 if unbatched else self.backend.shape(inputs)[0]
        mocked_value = self.backend.numpy.full(
            (batch_size, 1, 1, 1), 0.1, dtype="float32"
        )
        with unittest.mock.patch.object(
            self.backend.random,
            "uniform",
            return_value=mocked_value,
        ):
            out = super().call(inputs, training=training)
        return out


class RandomFlipTest(testing.TestCase):
    @parameterized.named_parameters(
        ("random_flip_horizontal", "horizontal"),
        ("random_flip_vertical", "vertical"),
        ("random_flip_both", "horizontal_and_vertical"),
    )
    def test_random_flip(self, mode):
        run_training_check = False if backend.backend() == "numpy" else True
        self.run_layer_test(
            layers.RandomFlip,
            init_kwargs={
                "mode": mode,
            },
            input_shape=(2, 3, 4),
            expected_output_shape=(2, 3, 4),
            supports_masking=False,
            run_training_check=run_training_check,
        )

    def test_random_flip_horizontal(self):
        run_training_check = False if backend.backend() == "numpy" else True
        utils.set_random_seed(0)
        # Test 3D input: shape (1*2*3)
        self.run_layer_test(
            MockedRandomFlip,
            init_kwargs={
                "mode": "horizontal",
                "data_format": "channels_last",
                "seed": 42,
            },
            input_data=np.asarray([[[2, 3, 4], [5, 6, 7]]]),
            expected_output=backend.convert_to_tensor([[[5, 6, 7], [2, 3, 4]]]),
            supports_masking=False,
            run_training_check=run_training_check,
        )
        # Test 4D input: shape (2*1*2*3)
        self.run_layer_test(
            MockedRandomFlip,
            init_kwargs={
                "mode": "horizontal",
                "data_format": "channels_last",
                "seed": 42,
            },
            input_data=np.asarray(
                [
                    [[[2, 3, 4], [5, 6, 7]]],
                    [[[2, 3, 4], [5, 6, 7]]],
                ]
            ),
            expected_output=backend.convert_to_tensor(
                [
                    [[[5, 6, 7], [2, 3, 4]]],
                    [[[5, 6, 7], [2, 3, 4]]],
                ]
            ),
            supports_masking=False,
            run_training_check=run_training_check,
        )

    def test_random_flip_vertical(self):
        run_training_check = False if backend.backend() == "numpy" else True
        utils.set_random_seed(0)
        # Test 3D input: shape (2*1*3)
        self.run_layer_test(
            MockedRandomFlip,
            init_kwargs={
                "mode": "vertical",
                "data_format": "channels_last",
                "seed": 42,
            },
            input_data=np.asarray([[[2, 3, 4]], [[5, 6, 7]]]),
            expected_output=backend.convert_to_tensor(
                [[[5, 6, 7]], [[2, 3, 4]]]
            ),
            supports_masking=False,
            run_training_check=run_training_check,
        )
        # Test 4D input: shape (2*2*1*3)
        self.run_layer_test(
            MockedRandomFlip,
            init_kwargs={
                "mode": "vertical",
                "data_format": "channels_last",
                "seed": 42,
            },
            input_data=np.asarray(
                [
                    [
                        [[2, 3, 4]],
                        [[5, 6, 7]],
                    ],
                    [
                        [[2, 3, 4]],
                        [[5, 6, 7]],
                    ],
                ]
            ),
            expected_output=backend.convert_to_tensor(
                [
                    [[[5, 6, 7]], [[2, 3, 4]]],
                    [[[5, 6, 7]], [[2, 3, 4]]],
                ]
            ),
            supports_masking=False,
            run_training_check=run_training_check,
        )

    def test_tf_data_compatibility(self):
        # Test 3D input: shape (2, 1, 3)
        layer = layers.RandomFlip(
            "vertical", data_format="channels_last", seed=42
        )
        input_data = np.array([[[2, 3, 4]], [[5, 6, 7]]])
        expected_output = np.array([[[5, 6, 7]], [[2, 3, 4]]])
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        output = next(iter(ds)).numpy()
        self.assertAllClose(output, expected_output)
        # Test 4D input: shape (2, 2, 1, 3)
        layer = layers.RandomFlip(
            "vertical", data_format="channels_last", seed=42
        )
        input_data = np.array(
            [
                [
                    [[2, 3, 4]],
                    [[5, 6, 7]],
                ],
                [
                    [[2, 3, 4]],
                    [[5, 6, 7]],
                ],
            ]
        )
        expected_output = np.array(
            [
                [[[5, 6, 7]], [[2, 3, 4]]],
                [[[5, 6, 7]], [[2, 3, 4]]],
            ]
        )
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        output = next(iter(ds)).numpy()
        self.assertAllClose(output, expected_output)

    @parameterized.named_parameters(
        (
            "with_horizontal",
            "horizontal",
            [[4, 1, 6, 3], [0, 4, 2, 6]],
        ),
        (
            "with_vertical",
            "vertical",
            [[2, 7, 4, 9], [6, 4, 8, 6]],
        ),
        (
            "with_horizontal_and_vertical",
            "horizontal_and_vertical",
            [[4, 7, 6, 9], [0, 4, 2, 6]],
        ),
    )
    def test_random_flip_bounding_boxes(self, mode, expected_boxes):
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
        random_flip_layer = layers.RandomFlip(
            mode,
            data_format=data_format,
            seed=42,
            bounding_box_format="xyxy",
        )

        transformation = {
            "flips": np.asarray([[True]]),
            "input_shape": input_image.shape,
        }
        output = random_flip_layer.transform_bounding_boxes(
            input_data["bounding_boxes"],
            transformation=transformation,
            training=True,
        )

        self.assertAllClose(output["boxes"], expected_boxes)

    @parameterized.named_parameters(
        (
            "with_horizontal",
            "horizontal",
            [[4, 1, 6, 3], [0, 4, 2, 6]],
        ),
        (
            "with_vertical",
            "vertical",
            [[2, 7, 4, 9], [6, 4, 8, 6]],
        ),
        (
            "with_horizontal_and_vertical",
            "horizontal_and_vertical",
            [[4, 7, 6, 9], [0, 4, 2, 6]],
        ),
    )
    def test_random_flip_tf_data_bounding_boxes(self, mode, expected_boxes):
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
        random_flip_layer = layers.RandomFlip(
            mode,
            data_format=data_format,
            seed=42,
            bounding_box_format="xyxy",
        )

        transformation = {
            "flips": np.asarray([[True]]),
            "input_shape": input_image.shape,
        }
        ds = ds.map(
            lambda x: random_flip_layer.transform_bounding_boxes(
                x["bounding_boxes"],
                transformation=transformation,
                training=True,
            )
        )

        output = next(iter(ds))
        expected_boxes = np.array(expected_boxes)
        self.assertAllClose(output["boxes"], expected_boxes)
