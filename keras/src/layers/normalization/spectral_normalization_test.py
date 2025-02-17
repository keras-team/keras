import numpy as np
import pytest

from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import models
from keras.src import testing


class SpectralNormalizationTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_basic_spectralnorm(self):
        self.run_layer_test(
            layers.SpectralNormalization,
            init_kwargs={"layer": layers.Dense(2)},
            input_data=np.random.uniform(size=(10, 3, 4)),
            expected_output_shape=(10, 3, 2),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=1,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )
        self.run_layer_test(
            layers.SpectralNormalization,
            init_kwargs={"layer": layers.Embedding(10, 4)},
            input_data=np.random.randint(10, size=(10,)).astype("float32"),
            expected_output_shape=(10, 4),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=1,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )

    def test_invalid_power_iterations(self):
        with self.assertRaisesRegex(
            ValueError, "`power_iterations` should be greater than zero."
        ):
            layers.SpectralNormalization(layers.Dense(2), power_iterations=0)

    def test_invalid_layer(self):
        layer = layers.SpectralNormalization(layers.ReLU())
        inputs = np.ones(shape=(4, 2))
        with self.assertRaisesRegex(
            ValueError, "object has no attribute 'kernel' nor 'embeddings'"
        ):
            layer(inputs)

    def test_apply_layer(self):
        if backend.config.image_data_format() == "channels_last":
            images = np.ones((1, 2, 2, 1))
        else:
            images = np.ones((1, 1, 2, 2))
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

    @pytest.mark.requires_trainable_backend
    def test_end_to_end(self):
        sn_wrapper = layers.SpectralNormalization(
            layers.Conv2D(
                3, (2, 2), padding="same", data_format="channels_last"
            ),
            power_iterations=2,
        )
        model = models.Sequential([sn_wrapper])
        model.compile("rmsprop", loss="mse")
        x = np.random.random((4, 8, 8, 3))
        y = np.random.random((4, 8, 8, 3))
        model.fit(x, y)
