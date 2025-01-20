import numpy as np
import pytest
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class RandomInvertTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.RandomInvert,
            init_kwargs={
                "factor": 0.75,
                "value_range": (20, 200),
                "seed": 1,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    def test_random_invert_inference(self):
        seed = 3481
        layer = layers.RandomInvert()
        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_random_invert_no_op(self):
        seed = 3481
        layer = layers.RandomInvert(factor=0)
        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs)
        self.assertAllClose(inputs, output)

    def test_random_invert_basic(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((1, 8, 8, 3))
        else:
            input_data = np.random.random((1, 3, 8, 8))
        layer = layers.RandomInvert(
            factor=(1, 1),
            value_range=[0, 1],
            data_format=data_format,
            seed=1337,
        )
        output = layer(input_data)
        self.assertAllClose(1 - input_data, output)

    def test_tf_data_compatibility(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3))
        else:
            input_data = np.random.random((2, 3, 8, 8))
        layer = layers.RandomInvert(
            factor=0.5, value_range=[0, 1], data_format=data_format, seed=1337
        )

        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output.numpy()
