import numpy as np
import pytest

from keras.src import layers
from keras.src import testing


class MixUpTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.MixUp,
            init_kwargs={
                "alpha": 0.2,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    def test_mix_up_basic_functionality(self):
        image = np.random.random((64, 64, 3))
        mix_up_layer = layers.MixUp(alpha=1)
        transformation = {"mix_weight": 1, "permutation_order": [0]}
        output = mix_up_layer.transform_images(
            image, transformation=transformation
        )[0]
        self.assertAllClose(output, image)

        image = np.random.random((4, 64, 64, 3))
        mix_up_layer = layers.MixUp(alpha=0.2)
        transformation = {"mix_weight": 0.2, "permutation_order": [1, 0, 2, 3]}
        output = mix_up_layer.transform_images(
            image, transformation=transformation
        )
        self.assertNotAllClose(output, image)
        self.assertAllClose(output.shape, image.shape)

    def test_mix_up_basic_functionality_channel_first(self):
        image = np.random.random((3, 64, 64))
        mix_up_layer = layers.MixUp(alpha=1)
        transformation = {"mix_weight": 1, "permutation_order": [0]}
        output = mix_up_layer.transform_images(
            image, transformation=transformation
        )[0]
        self.assertAllClose(output, image)

        image = np.random.random((4, 3, 64, 64))
        mix_up_layer = layers.MixUp(alpha=0.2)
        transformation = {"mix_weight": 0.2, "permutation_order": [1, 0, 2, 3]}
        output = mix_up_layer.transform_images(
            image, transformation=transformation
        )
        self.assertNotAllClose(output, image)
        self.assertAllClose(output.shape, image.shape)
