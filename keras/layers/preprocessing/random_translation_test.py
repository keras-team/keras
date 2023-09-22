import numpy as np
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras import backend
from keras import layers
from keras import testing


class RandomTranslationTest(testing.TestCase, parameterized.TestCase):
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
        for output in ds.take(1):
            output.numpy()
