import numpy as np
import tensorflow as tf
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
            run_training_check=False,
        )

    def test_random_zoom_out_correctness(self):
        input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1))
        expected_output = np.asarray(
            [
                [0, 0, 0, 0, 0],
                [0, 2.7, 4.5, 6.3, 0],
                [0, 10.2, 12.0, 13.8, 0],
                [0, 17.7, 19.5, 21.3, 0],
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
                "interpolation": "bilinear",
                "fill_mode": "constant",
            },
            input_shape=None,
            input_data=input_image,
            expected_output=expected_output,
            supports_masking=False,
            run_training_check=False,
        )

    def test_random_zoom_in_correctness(self):
        input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1))
        expected_output = np.asarray(
            [
                [6.0, 6.5, 7.0, 7.5, 8.0],
                [8.5, 9.0, 9.5, 10.0, 10.5],
                [11.0, 11.5, 12.0, 12.5, 13.0],
                [13.5, 14.0, 14.5, 15.0, 15.5],
                [16.0, 16.5, 17.0, 17.5, 18.0],
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
                "interpolation": "bilinear",
                "fill_mode": "constant",
            },
            input_shape=None,
            input_data=input_image,
            expected_output=expected_output,
            supports_masking=False,
            run_training_check=False,
        )

    def test_tf_data_compatibility(self):
        input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1))
        layer = layers.RandomZoom(
            height_factor=(0.5, 0.5),
            width_factor=(0.8, 0.8),
            interpolation="nearest",
            fill_mode="constant",
        )
        ds = tf.data.Dataset.from_tensor_slices(input_image).batch(1).map(layer)
        expected_output = np.asarray(
            [
                [0, 0, 0, 0, 0],
                [0, 5, 7, 9, 0],
                [0, 10, 12, 14, 0],
                [0, 20, 22, 24, 0],
                [0, 0, 0, 0, 0],
            ]
        ).reshape((1, 5, 5, 1))
        for output in ds.take(1):
            output = output.numpy()
        self.assertAllClose(expected_output, output)
