import numpy as np
import pytest
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class RandomShearTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.RandomShear,
            init_kwargs={
                "x_factor": (0.5, 1),
                "y_factor": (0.5, 1),
                "interpolation": "bilinear",
                "fill_mode": "reflect",
                "data_format": "channels_last",
                "seed": 1,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    def test_random_posterization_inference(self):
        seed = 3481
        layer = layers.RandomShear(1, 1)
        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_shear_pixel_level(self):
        image = np.zeros((1, 5, 5, 3), dtype=np.float32)
        image[0, 1:4, 1:4, :] = 1.0
        image[0, 2, 2, :] = [0.0, 1.0, 0.0]

        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            image = np.transpose(image, (0, 2, 3, 1))

        shear_layer = layers.RandomShear(
            x_factor=(0.2, 0.3),
            y_factor=(0.2, 0.3),
            interpolation="bilinear",
            fill_mode="constant",
            fill_value=0.0,
            seed=42,
        )

        sheared_image = shear_layer(image)
        original_pixel = (
            image[0, 1, 2, 2]
            if data_format == "channels_first"
            else image[0, 2, 1, 2]
        )
        sheared_pixel = (
            sheared_image[0, 1, 2, 2]
            if data_format == "channels_first"
            else sheared_image[0, 2, 1, 2]
        )
        self.assertNotEqual(original_pixel, sheared_pixel)

    def test_tf_data_compatibility(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3))
        else:
            input_data = np.random.random((2, 3, 8, 8))
        layer = layers.RandomShear(1, 1)

        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output.numpy()
