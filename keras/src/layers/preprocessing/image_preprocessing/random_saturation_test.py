import numpy as np
import pytest
from tensorflow import data as tf_data

import keras
from keras.src import backend
from keras.src import layers
from keras.src import testing


class RandomSaturationTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.RandomSaturation,
            init_kwargs={
                "factor": 0.75,
                "seed": 1,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    def test_random_saturation_value_range(self):
        image = keras.random.uniform(shape=(3, 3, 3), minval=0, maxval=1)

        layer = layers.RandomSaturation(0.2)
        adjusted_image = layer(image)

        self.assertTrue(keras.ops.numpy.all(adjusted_image >= 0))
        self.assertTrue(keras.ops.numpy.all(adjusted_image <= 1))

    def test_random_saturation_no_op(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            inputs = np.random.random((2, 8, 8, 3))
        else:
            inputs = np.random.random((2, 3, 8, 8))

        layer = layers.RandomSaturation((0.5, 0.5))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output, atol=1e-3, rtol=1e-5)

    def test_random_saturation_full_grayscale(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            inputs = np.random.random((2, 8, 8, 3))
        else:
            inputs = np.random.random((2, 3, 8, 8))
        layer = layers.RandomSaturation(factor=(0.0, 0.0))
        result = layer(inputs)

        if data_format == "channels_last":
            self.assertAllClose(result[..., 0], result[..., 1])
            self.assertAllClose(result[..., 1], result[..., 2])
        else:
            self.assertAllClose(result[:, 0, :, :], result[:, 1, :, :])
            self.assertAllClose(result[:, 1, :, :], result[:, 2, :, :])

    def test_random_saturation_full_saturation(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            inputs = np.random.random((2, 8, 8, 3))
        else:
            inputs = np.random.random((2, 3, 8, 8))
        layer = layers.RandomSaturation(factor=(1.0, 1.0))
        result = layer(inputs)

        hsv = backend.image.rgb_to_hsv(result)
        s_channel = hsv[..., 1]

        self.assertAllClose(
            keras.ops.numpy.max(s_channel), layer.value_range[1]
        )

    def test_random_saturation_randomness(self):
        image = keras.random.uniform(shape=(3, 3, 3), minval=0, maxval=1)[:5]

        layer = layers.RandomSaturation(0.2)
        adjusted_images = layer(image)

        self.assertNotAllClose(adjusted_images, image)

    def test_tf_data_compatibility(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3))
        else:
            input_data = np.random.random((2, 3, 8, 8))
        layer = layers.RandomSaturation(
            factor=0.5, data_format=data_format, seed=1337
        )

        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output.numpy()
