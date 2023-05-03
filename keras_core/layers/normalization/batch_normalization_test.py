import numpy as np

from keras_core import layers
from keras_core import testing


class BatchNormalizationTest(testing.TestCase):
    def test_bn_basics(self):
        # vector case
        self.run_layer_test(
            layers.BatchNormalization,
            init_kwargs={
                "center": True,
                "scale": True,
            },
            call_kwargs={"training": True},
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.BatchNormalization,
            init_kwargs={
                "center": False,
                "scale": False,
            },
            call_kwargs={"training": True},
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        # image case, with regularizers
        self.run_layer_test(
            layers.BatchNormalization,
            init_kwargs={
                "center": True,
                "scale": True,
                "beta_regularizer": "l2",
                "gamma_regularizer": "l2",
            },
            call_kwargs={"training": True},
            input_shape=(2, 4, 4, 3),
            expected_output_shape=(2, 4, 4, 3),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=2,  # we have 2 regularizers.
            supports_masking=True,
        )

    def test_correctness(self):
        # Training
        layer = layers.BatchNormalization(axis=-1, momentum=0.8)
        # Random data centered on 5.0, variance 10.0
        x = np.random.normal(loc=5.0, scale=10.0, size=(200, 4, 4, 3))
        for _ in range(10):
            out = layer(x, training=True)

        out -= np.reshape(np.array(layer.beta), (1, 1, 1, 3))
        out /= np.reshape(np.array(layer.gamma), (1, 1, 1, 3))

        self.assertAllClose(np.mean(out, axis=(0, 1, 2)), 0.0, atol=1e-3)
        self.assertAllClose(np.std(out, axis=(0, 1, 2)), 1.0, atol=1e-3)

        # Inference
        out = layer(x, training=False)
        out -= np.reshape(np.array(layer.beta), (1, 1, 1, 3))
        out /= np.reshape(np.array(layer.gamma), (1, 1, 1, 3))

        self.assertAllClose(np.mean(out, axis=(0, 1, 2)), 0.0, atol=1e-1)
        self.assertAllClose(np.std(out, axis=(0, 1, 2)), 1.0, atol=1e-1)

    def test_masking_correctness(self):
        # TODO
        pass
