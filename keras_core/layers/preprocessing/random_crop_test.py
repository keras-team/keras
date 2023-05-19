import numpy as np

from keras_core import layers
from keras_core import testing


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
        inp = np.random.random((12, 8, 16, 3))
        layer = layers.RandomCrop(height, width)
        actual_output = layer(inp, training=False)
        self.assertAllClose(inp, actual_output)

    def test_random_crop_partial(self):
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 8,
                "width": 8,
            },
            input_shape=(12, 8, 16, 3),
            expected_output_shape=(12, 8, 8, 3),
            supports_masking=False,
            run_training_check=False,
        )

    def test_predicting_with_longer_height(self):
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 10,
                "width": 8,
            },
            input_shape=(12, 8, 16, 3),
            expected_output_shape=(12, 10, 8, 3),
            supports_masking=False,
            run_training_check=False,
        )

    def test_predicting_with_longer_width(self):
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 8,
                "width": 18,
            },
            input_shape=(12, 8, 16, 3),
            expected_output_shape=(12, 8, 18, 3),
            supports_masking=False,
            run_training_check=False,
        )
