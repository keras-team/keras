import numpy as np
import pytest

from keras import backend
from keras import layers
from keras import testing


class GaussianNoiseTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_gaussian_noise_basics(self):
        self.run_layer_test(
            layers.GaussianNoise,
            init_kwargs={
                "stddev": 0.2,
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=1,
            expected_num_losses=0,
            supports_masking=True,
        )

    def test_gaussian_noise_correctness(self):
        inputs = np.ones((20, 500))
        layer = layers.GaussianNoise(0.3, seed=1337)
        outputs = layer(inputs, training=True)
        self.assertAllClose(
            np.std(backend.convert_to_numpy(outputs)), 0.3, atol=0.02
        )
