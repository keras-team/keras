import numpy as np
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.utils import backend_utils


class RandomTranslationTest(testing.TestCase):
    @parameterized.named_parameters(
        ("random_translate_4_by_6", 0.4, 0.6),
        ("random_translate_3_by_2", 0.3, 0.2),
        ("random_translate_tuple_factor", (-0.5, 0.4), (0.2, 0.3)),
    )
    def test_random_translation(self, height_factor, width_factor):
        self.run_layer_test(
            layers.RandomTranslation,
            init_kwargs={
                "height_factor": height_factor,
                "width_factor": width_factor,
            },
            input_shape=(2, 3, 4),
            expected_output_shape=(2, 3, 4),
            supports_masking=False,
            run_training_check=False,
        )

    @parameterized.named_parameters(
        ("bad_len", [0.1, 0.2, 0.3], 0.0),
        ("bad_type", {"dummy": 0.3}, 0.0),
        ("exceed_range_single", -1.1, 0.0),
        ("exceed_range_tuple", (-1.1, 0.0), 0.0),
    )
    def test_random_translation_with_bad_factor(
        self, height_factor, width_factor
    ):
        with self.assertRaises(ValueError):
            self.run_layer_test(
                layers.RandomTranslation,
                init_kwargs={
                    "height_factor": height_factor,
                    "width_factor": width_factor,
                },
                input_shape=(2, 3, 4),
                expected_output_shape=(2, 3, 4),
                supports_masking=False,
                run_training_check=False,
            )

    def test_random_translation_with_inference_mode(self):
        input_data = np.random.random((1, 4, 4, 3))
        expected_output = input_data
        layer = layers.RandomTranslation(0.2, 0.1)
        output = layer(input_data, training=False)
        self.assertAllClose(output, expected_output)

    @parameterized.parameters(["channels_first", "channels_last"])
    def test_random_translation_up_numeric_reflect(self, data_format):
        input_image = np.arange(0, 25)
        expected_output = np.asarray(
            [
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24],
                [20, 21, 22, 23, 24],
            ]
        )
        if data_format == "channels_last":
            input_image = np.reshape(input_image, (1, 5, 5, 1))
            expected_output = backend.convert_to_tensor(
                np.reshape(expected_output, (1, 5, 5, 1))
            )
        else:
            input_image = np.reshape(input_image, (1, 1, 5, 5))
            expected_output = backend.convert_to_tensor(
                np.reshape(expected_output, (1, 1, 5, 5))
            )
        self.run_layer_test(
            layers.RandomTranslation,
            init_kwargs={
                "height_factor": (-0.2, -0.2),
                "width_factor": 0.0,
                "data_format": data_format,
            },
            input_shape=None,
            input_data=input_image,
            expected_output=expected_output,
            supports_masking=False,
            run_training_check=False,
        )

    @parameterized.parameters(["channels_first", "channels_last"])
    def test_random_translation_up_numeric_constant(self, data_format):
        input_image = np.arange(0, 25).astype("float32")
        # Shifting by -.2 * 5 = 1 pixel.
        expected_output = np.asarray(
            [
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24],
                [0, 0, 0, 0, 0],
            ]
        )
        if data_format == "channels_last":
            input_image = np.reshape(input_image, (1, 5, 5, 1))
            expected_output = backend.convert_to_tensor(
                np.reshape(expected_output, (1, 5, 5, 1)), dtype="float32"
            )
        else:
            input_image = np.reshape(input_image, (1, 1, 5, 5))
            expected_output = backend.convert_to_tensor(
                np.reshape(expected_output, (1, 1, 5, 5)), dtype="float32"
            )
        self.run_layer_test(
            layers.RandomTranslation,
            init_kwargs={
                "height_factor": (-0.2, -0.2),
                "width_factor": 0.0,
                "fill_mode": "constant",
                "data_format": data_format,
            },
            input_shape=None,
            input_data=input_image,
            expected_output=expected_output,
            supports_masking=False,
            run_training_check=False,
        )

    @parameterized.parameters(["channels_first", "channels_last"])
    def test_random_translation_down_numeric_reflect(self, data_format):
        input_image = np.arange(0, 25)
        # Shifting by .2 * 5 = 1 pixel.
        expected_output = np.asarray(
            [
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
            ]
        )
        if data_format == "channels_last":
            input_image = np.reshape(input_image, (1, 5, 5, 1))
            expected_output = backend.convert_to_tensor(
                np.reshape(expected_output, (1, 5, 5, 1))
            )
        else:
            input_image = np.reshape(input_image, (1, 1, 5, 5))
            expected_output = backend.convert_to_tensor(
                np.reshape(expected_output, (1, 1, 5, 5))
            )
        self.run_layer_test(
            layers.RandomTranslation,
            init_kwargs={
                "height_factor": (0.2, 0.2),
                "width_factor": 0.0,
                "data_format": data_format,
            },
            input_shape=None,
            input_data=input_image,
            expected_output=expected_output,
            supports_masking=False,
            run_training_check=False,
        )

    @parameterized.parameters(["channels_first", "channels_last"])
    def test_random_translation_asymmetric_size_numeric_reflect(
        self, data_format
    ):
        input_image = np.arange(0, 16)
        # Shifting by .2 * 5 = 1 pixel.
        expected_output = np.asarray(
            [
                [6, 7],
                [4, 5],
                [2, 3],
                [0, 1],
                [0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
            ]
        )
        if data_format == "channels_last":
            input_image = np.reshape(input_image, (1, 8, 2, 1))
            expected_output = backend.convert_to_tensor(
                np.reshape(expected_output, (1, 8, 2, 1))
            )
        else:
            input_image = np.reshape(input_image, (1, 1, 8, 2))
            expected_output = backend.convert_to_tensor(
                np.reshape(expected_output, (1, 1, 8, 2))
            )
        self.run_layer_test(
            layers.RandomTranslation,
            init_kwargs={
                "height_factor": (0.5, 0.5),
                "width_factor": 0.0,
                "data_format": data_format,
            },
            input_shape=None,
            input_data=input_image,
            expected_output=expected_output,
            supports_masking=False,
            run_training_check=False,
        )

    @parameterized.parameters(["channels_first", "channels_last"])
    def test_random_translation_down_numeric_constant(self, data_format):
        input_image = np.arange(0, 25)
        # Shifting by .2 * 5 = 1 pixel.
        expected_output = np.asarray(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
            ]
        )
        if data_format == "channels_last":
            input_image = np.reshape(input_image, (1, 5, 5, 1))
            expected_output = backend.convert_to_tensor(
                np.reshape(expected_output, (1, 5, 5, 1))
            )
        else:
            input_image = np.reshape(input_image, (1, 1, 5, 5))
            expected_output = backend.convert_to_tensor(
                np.reshape(expected_output, (1, 1, 5, 5))
            )
        self.run_layer_test(
            layers.RandomTranslation,
            init_kwargs={
                "height_factor": (0.2, 0.2),
                "width_factor": 0.0,
                "fill_mode": "constant",
                "fill_value": 0.0,
                "data_format": data_format,
            },
            input_shape=None,
            input_data=input_image,
            expected_output=expected_output,
            supports_masking=False,
            run_training_check=False,
        )

    @parameterized.parameters(["channels_first", "channels_last"])
    def test_random_translation_left_numeric_reflect(self, data_format):
        input_image = np.arange(0, 25)
        # Shifting by .2 * 5 = 1 pixel.
        expected_output = np.asarray(
            [
                [1, 2, 3, 4, 4],
                [6, 7, 8, 9, 9],
                [11, 12, 13, 14, 14],
                [16, 17, 18, 19, 19],
                [21, 22, 23, 24, 24],
            ]
        )
        if data_format == "channels_last":
            input_image = np.reshape(input_image, (1, 5, 5, 1))
            expected_output = backend.convert_to_tensor(
                np.reshape(expected_output, (1, 5, 5, 1))
            )
        else:
            input_image = np.reshape(input_image, (1, 1, 5, 5))
            expected_output = backend.convert_to_tensor(
                np.reshape(expected_output, (1, 1, 5, 5))
            )
        self.run_layer_test(
            layers.RandomTranslation,
            init_kwargs={
                "height_factor": 0.0,
                "width_factor": (-0.2, -0.2),
                "data_format": data_format,
            },
            input_shape=None,
            input_data=input_image,
            expected_output=expected_output,
            supports_masking=False,
            run_training_check=False,
        )

    @parameterized.parameters(["channels_first", "channels_last"])
    def test_random_translation_left_numeric_constant(self, data_format):
        input_image = np.arange(0, 25)
        # Shifting by .2 * 5 = 1 pixel.
        expected_output = np.asarray(
            [
                [1, 2, 3, 4, 0],
                [6, 7, 8, 9, 0],
                [11, 12, 13, 14, 0],
                [16, 17, 18, 19, 0],
                [21, 22, 23, 24, 0],
            ]
        )
        if data_format == "channels_last":
            input_image = np.reshape(input_image, (1, 5, 5, 1))
            expected_output = backend.convert_to_tensor(
                np.reshape(expected_output, (1, 5, 5, 1))
            )
        else:
            input_image = np.reshape(input_image, (1, 1, 5, 5))
            expected_output = backend.convert_to_tensor(
                np.reshape(expected_output, (1, 1, 5, 5))
            )
        self.run_layer_test(
            layers.RandomTranslation,
            init_kwargs={
                "height_factor": 0.0,
                "width_factor": (-0.2, -0.2),
                "fill_mode": "constant",
                "fill_value": 0.0,
                "data_format": data_format,
            },
            input_shape=None,
            input_data=input_image,
            expected_output=expected_output,
            supports_masking=False,
            run_training_check=False,
        )

    def test_tf_data_compatibility(self):
        layer = layers.RandomTranslation(0.2, 0.1)
        input_data = np.random.random((1, 4, 4, 3))
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(1).map(layer)
        next(iter(ds)).numpy()

    @parameterized.named_parameters(
        (
            "with_positive_shift",
            [[1.0, 2.0]],
            [[3.0, 3.0, 5.0, 5.0], [7.0, 6.0, 8.0, 8.0]],
        ),
        (
            "with_negative_shift",
            [[-1.0, -2.0]],
            [[1.0, 0.0, 3.0, 1.0], [5.0, 2.0, 7.0, 4.0]],
        ),
    )
    def test_random_flip_bounding_boxes(self, translation, expected_boxes):
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
        random_translation_layer = layers.RandomTranslation(
            height_factor=0.5,
            width_factor=0.5,
            data_format=data_format,
            seed=42,
            bounding_box_format="xyxy",
        )

        transformation = {
            "translations": backend_utils.convert_tf_tensor(
                np.array(translation)
            ),
            "input_shape": image_shape,
        }
        output = random_translation_layer.transform_bounding_boxes(
            input_data["bounding_boxes"],
            transformation=transformation,
            training=True,
        )

        self.assertAllClose(output["boxes"], expected_boxes)

    @parameterized.named_parameters(
        (
            "with_positive_shift",
            [[1.0, 2.0]],
            [[3.0, 3.0, 5.0, 5.0], [7.0, 6.0, 8.0, 8.0]],
        ),
        (
            "with_negative_shift",
            [[-1.0, -2.0]],
            [[1.0, 0.0, 3.0, 1.0], [5.0, 2.0, 7.0, 4.0]],
        ),
    )
    def test_random_flip_tf_data_bounding_boxes(
        self, translation, expected_boxes
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
                        [2, 1, 4, 3],
                        [6, 4, 8, 6],
                    ]
                ]
            ),
            "labels": np.array([[1, 2]]),
        }

        input_data = {"images": input_image, "bounding_boxes": bounding_boxes}

        ds = tf_data.Dataset.from_tensor_slices(input_data)
        random_translation_layer = layers.RandomTranslation(
            height_factor=0.5,
            width_factor=0.5,
            data_format=data_format,
            seed=42,
            bounding_box_format="xyxy",
        )

        transformation = {
            "translations": np.array(translation),
            "input_shape": image_shape,
        }

        ds = ds.map(
            lambda x: random_translation_layer.transform_bounding_boxes(
                x["bounding_boxes"],
                transformation=transformation,
                training=True,
            )
        )

        output = next(iter(ds))
        expected_boxes = np.array(expected_boxes)
        self.assertAllClose(output["boxes"], expected_boxes)
