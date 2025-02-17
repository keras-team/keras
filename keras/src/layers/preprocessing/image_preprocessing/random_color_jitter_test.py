import numpy as np
import pytest
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class RandomColorJitterTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.RandomColorJitter,
            init_kwargs={
                "value_range": (20, 200),
                "brightness_factor": 0.2,
                "contrast_factor": 0.2,
                "saturation_factor": 0.2,
                "hue_factor": 0.2,
                "seed": 1,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    def test_random_color_jitter_inference(self):
        seed = 3481
        layer = layers.RandomColorJitter(
            value_range=(0, 1),
            brightness_factor=0.1,
            contrast_factor=0.2,
            saturation_factor=0.9,
            hue_factor=0.1,
        )

        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_brightness_only(self):
        seed = 2390
        np.random.seed(seed)

        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            inputs = np.random.random((12, 8, 16, 3))
        else:
            inputs = np.random.random((12, 3, 8, 16))

        layer = layers.RandomColorJitter(
            brightness_factor=[0.5, 0.5], seed=seed
        )
        output = backend.convert_to_numpy(layer(inputs))

        layer = layers.RandomBrightness(factor=[0.5, 0.5], seed=seed)
        sub_output = backend.convert_to_numpy(layer(inputs))

        self.assertAllClose(output, sub_output)

    def test_saturation_only(self):
        seed = 2390
        np.random.seed(seed)

        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            inputs = np.random.random((12, 8, 16, 3))
        else:
            inputs = np.random.random((12, 3, 8, 16))

        layer = layers.RandomColorJitter(
            saturation_factor=[0.5, 0.5], seed=seed
        )
        output = layer(inputs)

        layer = layers.RandomSaturation(factor=[0.5, 0.5], seed=seed)
        sub_output = layer(inputs)

        self.assertAllClose(output, sub_output)

    def test_hue_only(self):
        seed = 2390
        np.random.seed(seed)

        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            inputs = np.random.random((12, 8, 16, 3))
        else:
            inputs = np.random.random((12, 3, 8, 16))

        layer = layers.RandomColorJitter(hue_factor=[0.5, 0.5], seed=seed)
        output = layer(inputs)

        layer = layers.RandomHue(factor=[0.5, 0.5], seed=seed)
        sub_output = layer(inputs)

        self.assertAllClose(output, sub_output)

    def test_contrast_only(self):
        seed = 2390
        np.random.seed(seed)

        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            inputs = np.random.random((12, 8, 16, 3))
        else:
            inputs = np.random.random((12, 3, 8, 16))

        layer = layers.RandomColorJitter(contrast_factor=[0.5, 0.5], seed=seed)
        output = layer(inputs)

        layer = layers.RandomContrast(factor=[0.5, 0.5], seed=seed)
        sub_output = layer(inputs)

        self.assertAllClose(output, sub_output)

    def test_tf_data_compatibility(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3))
        else:
            input_data = np.random.random((2, 3, 8, 8))
        layer = layers.RandomColorJitter(
            value_range=(0, 1),
            brightness_factor=0.1,
            contrast_factor=0.2,
            saturation_factor=0.9,
            hue_factor=0.1,
        )

        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output.numpy()
