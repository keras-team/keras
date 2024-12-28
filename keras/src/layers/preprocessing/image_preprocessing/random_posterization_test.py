import numpy as np
import pytest
from tensorflow import data as tf_data

import keras
from keras.src import backend
from keras.src import layers
from keras.src import testing


class RandomPosterizationTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.RandomPosterization,
            init_kwargs={
                "factor": 1,
                "value_range": (20, 200),
                "seed": 1,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    def test_random_posterization_inference(self):
        seed = 3481
        layer = layers.RandomPosterization(1, [0, 255])
        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_random_posterization_basic(self):
        seed = 3481
        layer = layers.RandomPosterization(
            1, [0, 255], data_format="channels_last", seed=seed
        )
        np.random.seed(seed)
        inputs = np.asarray(
            [[[128.0, 235.0, 87.0], [12.0, 1.0, 23.0], [24.0, 18.0, 121.0]]]
        )
        output = layer(inputs)
        expected_output = np.asarray(
            [[[128.0, 128.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        )
        self.assertAllClose(expected_output, output)

    def test_random_posterization_value_range_0_to_1(self):
        image = keras.random.uniform(shape=(3, 3, 3), minval=0, maxval=1)

        layer = layers.RandomPosterization(1, [0, 1.0])
        adjusted_image = layer(image)

        self.assertTrue(keras.ops.numpy.all(adjusted_image >= 0))
        self.assertTrue(keras.ops.numpy.all(adjusted_image <= 1))

    def test_random_posterization_value_range_0_to_255(self):
        image = keras.random.uniform(shape=(3, 3, 3), minval=0, maxval=255)

        layer = layers.RandomPosterization(1, [0, 255])
        adjusted_image = layer(image)

        self.assertTrue(keras.ops.numpy.all(adjusted_image >= 0))
        self.assertTrue(keras.ops.numpy.all(adjusted_image <= 255))

    def test_random_posterization_randomness(self):
        image = keras.random.uniform(shape=(3, 3, 3), minval=0, maxval=1)

        layer = layers.RandomPosterization(1, [0, 255])
        adjusted_images = layer(image)

        self.assertNotAllClose(adjusted_images, image)

    def test_tf_data_compatibility(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3))
        else:
            input_data = np.random.random((2, 3, 8, 8))
        layer = layers.RandomPosterization(1, [0, 255])

        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output.numpy()
