import numpy as np
from absl.testing import parameterized

from keras_core import backend
from keras_core import layers
from keras_core import testing


class RandomZoomTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("random_zoom_in_4_by_6", -0.4, -0.6),
        ("random_zoom_in_2_by_3", -0.2, -0.3),
        ("random_zoom_in_tuple_factor", (-0.4, -0.5), (-0.2, -0.3)),
        ("random_zoom_out_4_by_6", 0.4, 0.6),
        ("random_zoom_out_2_by_3", 0.2, 0.3),
        ("random_zoom_out_tuple_factor", (0.4, 0.5), (0.2, 0.3)),
    )
    def test_random_zoom(self, height_factor, width_factor):
        self.run_layer_test(
            layers.RandomZoom,
            init_kwargs={
                "height_factor": height_factor,
                "width_factor": width_factor,
            },
            input_shape=(2, 3, 4),
            expected_output_shape=(2, 3, 4),
            supports_masking=False,
        )

    def test_random_zoom_out_correctness(self):
        input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1))
        expected_output = np.asarray(
            [
                [0, 0, 0, 0, 0],
                [0, 5, 7, 9, 0],
                [0, 10, 12, 14, 0],
                [0, 20, 22, 24, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        expected_output = backend.convert_to_tensor(
            np.reshape(expected_output, (1, 5, 5, 1))
        )
        self.run_layer_test(
            layers.RandomZoom,
            init_kwargs={
                "height_factor": (0.5, 0.5),
                "width_factor": (0.8, 0.8),
                "interpolation": "nearest",
                "fill_mode": "constant",
            },
            input_shape=None,
            input_data=input_image,
            expected_output=expected_output,
            supports_masking=False,
        )

    def test_random_zoom_in_correctness(self):
        input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1))
        expected_output = np.asarray(
            [
                [6, 7, 7, 8, 8],
                [11, 12, 12, 13, 13],
                [11, 12, 12, 13, 13],
                [16, 17, 17, 18, 18],
                [16, 17, 17, 18, 18],
            ]
        )
        expected_output = backend.convert_to_tensor(
            np.reshape(expected_output, (1, 5, 5, 1))
        )
        self.run_layer_test(
            layers.RandomZoom,
            init_kwargs={
                "height_factor": (-0.5, -0.5),
                "width_factor": (-0.5, -0.5),
                "interpolation": "nearest",
            },
            input_shape=None,
            input_data=input_image,
            expected_output=expected_output,
            supports_masking=False,
        )
