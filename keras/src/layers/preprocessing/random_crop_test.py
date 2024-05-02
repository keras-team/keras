import numpy as np
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class RandomCropTest(testing.TestCase):
    def test_random_crop(self):
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 1,
                "width": 1,
            },
            input_shape=(2, 3, 4),
            supports_masking=False,
            run_training_check=False,
        )

    def test_random_crop_full(self):
        np.random.seed(1337)
        height, width = 8, 16
        if backend.config.image_data_format() == "channels_last":
            input_shape = (12, 8, 16, 3)
        else:
            input_shape = (12, 3, 8, 16)
        inp = np.random.random(input_shape)
        layer = layers.RandomCrop(height, width)
        actual_output = layer(inp, training=False)
        self.assertAllClose(inp, actual_output)

    def test_random_crop_partial(self):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (12, 8, 16, 3)
            output_shape = (12, 8, 8, 3)
        else:
            input_shape = (12, 3, 8, 16)
            output_shape = (12, 3, 8, 8)
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 8,
                "width": 8,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            supports_masking=False,
            run_training_check=False,
        )

    def test_predicting_with_longer_height(self):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (12, 8, 16, 3)
            output_shape = (12, 10, 8, 3)
        else:
            input_shape = (12, 3, 8, 16)
            output_shape = (12, 3, 10, 8)
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 10,
                "width": 8,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            supports_masking=False,
            run_training_check=False,
        )

    def test_predicting_with_longer_width(self):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (12, 8, 16, 3)
            output_shape = (12, 8, 18, 3)
        else:
            input_shape = (12, 3, 8, 16)
            output_shape = (12, 3, 8, 18)
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 8,
                "width": 18,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            supports_masking=False,
            run_training_check=False,
        )

    def test_tf_data_compatibility(self):
        layer = layers.RandomCrop(8, 9)
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 10, 12, 3)
            output_shape = (2, 8, 9, 3)
        else:
            input_shape = (2, 3, 10, 12)
            output_shape = (2, 3, 8, 9)
        input_data = np.random.random(input_shape)
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output = output.numpy()
        self.assertEqual(tuple(output.shape), output_shape)
