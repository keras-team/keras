import numpy as np
import pytest
from tensorflow import data as tf_data

import keras
from keras.src import backend
from keras.src import layers
from keras.src import testing


class RandomHueTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.RandomHue,
            init_kwargs={
                "factor": 0.75,
                "value_range": (20, 200),
                "seed": 1,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    def test_random_hue_inference(self):
        seed = 3481
        layer = layers.RandomHue(0.2, [0, 1.0])
        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_random_hue_value_range_0_to_1(self):
        image = keras.random.uniform(shape=(3, 3, 3), minval=0, maxval=1)

        layer = layers.RandomHue(0.2, (0, 1))
        adjusted_image = layer(image)

        self.assertTrue(keras.ops.numpy.all(adjusted_image >= 0))
        self.assertTrue(keras.ops.numpy.all(adjusted_image <= 1))

    def test_random_hue_value_range_0_to_255(self):
        image = keras.random.uniform(shape=(3, 3, 3), minval=0, maxval=255)

        layer = layers.RandomHue(0.2, (0, 255))
        adjusted_image = layer(image)

        self.assertTrue(keras.ops.numpy.all(adjusted_image >= 0))
        self.assertTrue(keras.ops.numpy.all(adjusted_image <= 255))

    def test_random_hue_no_change_with_zero_factor(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            inputs = keras.random.randint((224, 224, 3), 0, 255)
        else:
            inputs = keras.random.randint((3, 224, 224), 0, 255)

        layer = layers.RandomHue(0, (0, 255), data_format=data_format)
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output, atol=1e-3, rtol=1e-5)

    def test_random_hue_randomness(self):
        image = keras.random.uniform(shape=(3, 3, 3), minval=0, maxval=1)[:5]

        layer = layers.RandomHue(0.2, (0, 255))
        adjusted_images = layer(image)

        self.assertNotAllClose(adjusted_images, image)

    def test_tf_data_compatibility(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3))
        else:
            input_data = np.random.random((2, 3, 8, 8))
        layer = layers.RandomHue(
            factor=0.5, value_range=[0, 1], data_format=data_format, seed=1337
        )

        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output.numpy()
