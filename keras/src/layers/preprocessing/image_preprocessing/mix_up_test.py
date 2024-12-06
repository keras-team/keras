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
        mix_up_layer = layers.MixUp(alpha=0.2)
        output = mix_up_layer(image)
        self.assertAllClose(output, image)

        image = np.random.random((4, 64, 64, 3))
        mix_up_layer = layers.MixUp(alpha=0.2)
        output = mix_up_layer(image)
        self.assertNotAllClose(output, image)
        self.assertAllClose(output.shape, image.shape)

    def test_mix_up_basic_functionality_channel_first(self):
        image = np.random.random((3, 64, 64))
        mix_up_layer = layers.MixUp(alpha=0.2)
        output = mix_up_layer(image)
        self.assertAllClose(output, image)

        image = np.random.random((4, 3, 64, 64))
        mix_up_layer = layers.MixUp(alpha=0.2)
        output = mix_up_layer(image)
        self.assertNotAllClose(output, image)
        self.assertAllClose(output.shape, image.shape)

    def test_mix_up_random_alpha(self):
        image1 = np.ones((64, 64, 3))
        image2 = np.zeros((64, 64, 3))

        mix_up_layer = layers.MixUp(alpha=(0.1, 0.9))
        output1 = mix_up_layer([image1, image2])
        output2 = mix_up_layer([image1, image2])

        self.assertNotAllClose(output1, output2)
