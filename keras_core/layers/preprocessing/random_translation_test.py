import numpy as np
from absl.testing import parameterized

from keras_core import backend
from keras_core import layers
from keras_core import testing


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
        )

    def test_random_translation_up_numeric_reflect(self):
        input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1))
        expected_output = np.asarray(
            [
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24],
                [20, 21, 22, 23, 24],
            ]
        )
        expected_output = backend.convert_to_tensor(
            np.reshape(expected_output, (1, 5, 5, 1))
        )
        self.run_layer_test(
            layers.RandomTranslation,
            init_kwargs={
                "height_factor": (-0.2, -0.2),
                "width_factor": 0.0,
            },
            input_shape=None,
            input_data=input_image,
            expected_output=expected_output,
            supports_masking=False,
        )

    def test_random_translation_up_numeric_constant(self):
        input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1)).astype(
            "float32"
        )
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
        expected_output = backend.convert_to_tensor(
            np.reshape(expected_output, (1, 5, 5, 1)), dtype="float32"
        )
        self.run_layer_test(
            layers.RandomTranslation,
            init_kwargs={
                "height_factor": (-0.2, -0.2),
                "width_factor": 0.0,
                "fill_mode": "constant",
            },
            input_shape=None,
            input_data=input_image,
            expected_output=expected_output,
            supports_masking=False,
        )

    def test_random_translation_down_numeric_reflect(self):
        input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1))
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
        expected_output = backend.convert_to_tensor(
            np.reshape(expected_output, (1, 5, 5, 1))
        )
        self.run_layer_test(
            layers.RandomTranslation,
            init_kwargs={
                "height_factor": (0.2, 0.2),
                "width_factor": 0.0,
            },
            input_shape=None,
            input_data=input_image,
            expected_output=expected_output,
            supports_masking=False,
        )

    def test_random_translation_asymmetric_size_numeric_reflect(self):
        input_image = np.reshape(np.arange(0, 16), (1, 8, 2, 1))
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
        expected_output = backend.convert_to_tensor(
            np.reshape(expected_output, (1, 8, 2, 1))
        )
        self.run_layer_test(
            layers.RandomTranslation,
            init_kwargs={
                "height_factor": (0.5, 0.5),
                "width_factor": 0.0,
            },
            input_shape=None,
            input_data=input_image,
            expected_output=expected_output,
            supports_masking=False,
        )

    def test_random_translation_down_numeric_constant(self):
        input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1))
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
        expected_output = backend.convert_to_tensor(
            np.reshape(expected_output, (1, 5, 5, 1))
        )
        self.run_layer_test(
            layers.RandomTranslation,
            init_kwargs={
                "height_factor": (0.2, 0.2),
                "width_factor": 0.0,
                "fill_mode": "constant",
                "fill_value": 0.0,
            },
            input_shape=None,
            input_data=input_image,
            expected_output=expected_output,
            supports_masking=False,
        )

    def test_random_translation_left_numeric_reflect(self):
        input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1))
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
        expected_output = backend.convert_to_tensor(
            np.reshape(expected_output, (1, 5, 5, 1))
        )
        self.run_layer_test(
            layers.RandomTranslation,
            init_kwargs={
                "height_factor": 0.0,
                "width_factor": (-0.2, -0.2),
            },
            input_shape=None,
            input_data=input_image,
            expected_output=expected_output,
            supports_masking=False,
        )

    def test_random_translation_left_numeric_constant(self):
        input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1))
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
        expected_output = backend.convert_to_tensor(
            np.reshape(expected_output, (1, 5, 5, 1))
        )
        self.run_layer_test(
            layers.RandomTranslation,
            init_kwargs={
                "height_factor": 0.0,
                "width_factor": (-0.2, -0.2),
                "fill_mode": "constant",
                "fill_value": 0.0,
            },
            input_shape=None,
            input_data=input_image,
            expected_output=expected_output,
            supports_masking=False,
        )
