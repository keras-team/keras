import numpy as np
import pytest

from keras_core import initializers
from keras_core import layers
from keras_core import testing


class SpectralNormalizationTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_basic_spectralnorm(self):
        self.run_layer_test(
            layers.SpectralNormalization,
            init_kwargs={"layer": layers.Dense(2)},
            input_shape=None,
            input_data=np.random.uniform(size=(10, 3, 4)),
            expected_output_shape=(10, 3, 2),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    def test_apply_layer(self):
        images = np.ones((1, 2, 2, 1))
        sn_wrapper = layers.SpectralNormalization(
            layers.Conv2D(
                1, (2, 2), kernel_initializer=initializers.Constant(value=1)
            ),
            power_iterations=8,
        )

        result = sn_wrapper(images, training=False)
        result_train = sn_wrapper(images, training=True)
        expected_output = np.array([[[[4.0]]]], dtype=np.float32)
        self.assertAllClose(result, expected_output)
        # max eigen value of 2x2 matrix of ones is 2
        self.assertAllClose(result_train, expected_output / 2)
