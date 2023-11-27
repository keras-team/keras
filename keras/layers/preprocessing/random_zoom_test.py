import numpy as np
import pytest
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras import backend
from keras import layers
from keras import models
from keras import testing


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
        if backend.config.image_data_format() == "channels_last":
            input_shape = (1, 5, 5, 1)
        else:
            input_shape = (1, 1, 5, 5)
        input_image = np.reshape(np.arange(0, 25), input_shape)
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
            np.reshape(expected_output, input_shape)
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
        if backend.config.image_data_format() == "channels_last":
            input_shape = (1, 5, 5, 1)
        else:
            input_shape = (1, 1, 5, 5)
        input_image = np.reshape(np.arange(0, 25), input_shape)
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
            np.reshape(expected_output, input_shape)
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
        if backend.config.image_data_format() == "channels_last":
            input_shape = (1, 5, 5, 1)
        else:
            input_shape = (1, 1, 5, 5)
        input_image = np.reshape(np.arange(0, 25), input_shape)
        layer = layers.RandomZoom(
            height_factor=(0.5, 0.5),
            width_factor=(0.8, 0.8),
            interpolation="nearest",
            fill_mode="constant",
        )
        ds = tf_data.Dataset.from_tensor_slices(input_image).batch(1).map(layer)
        expected_output = np.asarray(
            [
                [0, 0, 0, 0, 0],
                [0, 5, 7, 9, 0],
                [0, 10, 12, 14, 0],
                [0, 20, 22, 24, 0],
                [0, 0, 0, 0, 0],
            ]
        ).reshape(input_shape)
        for output in ds.take(1):
            output = output.numpy()
        self.assertAllClose(expected_output, output)

    def test_dynamic_shape(self):
        inputs = layers.Input((None, None, 3))
        outputs = layers.RandomZoom(
            height_factor=(0.5, 0.5),
            width_factor=(0.8, 0.8),
            interpolation="nearest",
            fill_mode="constant",
        )(inputs)
        model = models.Model(inputs, outputs)
        model.predict(np.random.random((1, 6, 6, 3)))

    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="The NumPy backend does not implement fit.",
    )
    def test_connect_with_flatten(self):
        model = models.Sequential(
            [
                layers.RandomZoom((-0.5, 0.0), (-0.5, 0.0)),
                layers.Flatten(),
                layers.Dense(1, activation="relu"),
            ],
        )

        model.compile(loss="mse")
        model.fit(np.random.random((2, 2, 2, 1)), y=np.random.random((2,)))
