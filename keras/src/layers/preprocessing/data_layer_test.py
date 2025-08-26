import grain
import numpy as np
import pytest
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import testing
from keras.src.layers.preprocessing.data_layer import DataLayer
from keras.src.random import SeedGenerator


class RandomRGBToHSVLayer(DataLayer):
    def __init__(self, data_format=None, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.data_format = backend.standardize_data_format(data_format)
        self.seed = seed
        self.generator = SeedGenerator(seed)

    def call(self, inputs):
        images_shape = self.backend.shape(inputs)
        batch_size = 1 if len(images_shape) == 3 else images_shape[0]
        seed = self._get_seed_generator(self.backend._backend)

        probability = self.backend.random.uniform(
            shape=(batch_size,),
            minval=0.0,
            maxval=1.0,
            seed=seed,
        )
        hsv_images = self.backend.image.rgb_to_hsv(
            inputs, data_format=self.data_format
        )
        return self.backend.numpy.where(
            probability[:, None, None, None] > 0.5, hsv_images, inputs
        )

    def compute_output_shape(self, input_shape):
        return input_shape


class DataLayerTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            RandomRGBToHSVLayer,
            init_kwargs={
                "seed": 1337,
                "data_format": "channels_last",
            },
            input_shape=(1, 2, 2, 3),
            supports_masking=False,
            expected_output_shape=(1, 2, 2, 3),
        )

        self.run_layer_test(
            RandomRGBToHSVLayer,
            init_kwargs={
                "seed": 1337,
                "data_format": "channels_first",
            },
            input_shape=(1, 3, 2, 2),
            supports_masking=False,
            expected_output_shape=(1, 3, 2, 2),
        )

    def test_tf_data_compatibility(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3)).astype("float32")
        else:
            input_data = np.random.random((2, 3, 8, 8)).astype("float32")
        layer = RandomRGBToHSVLayer(data_format=data_format, seed=1337)

        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            self.assertDType(output, "float32")
            self.assertEqual(list(output.shape), list(input_data.shape))

    def test_grain_compatibility(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3)).astype("float32")
        else:
            input_data = np.random.random((2, 3, 8, 8)).astype("float32")
        layer = RandomRGBToHSVLayer(data_format=data_format, seed=1337)

        ds = grain.MapDataset.source(input_data).batch(2).map(layer)
        for output in ds[:1]:
            self.assertDType(output, "float32")
            self.assertEqual(list(output.shape), list(input_data.shape))
