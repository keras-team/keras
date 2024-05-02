import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import testing
from keras.src.losses import MeanSquaredError
from keras.src.models import Model


class BatchNormalizationTest(testing.TestCase, parameterized.TestCase):
    @pytest.mark.requires_trainable_backend
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

    @parameterized.product(
        axis=(-1, 1),
        input_shape=((5, 2, 3), (5, 3, 3, 2)),
        moving_mean_initializer=("zeros", "ones"),
        moving_variance_initializer=("zeros", "ones"),
    )
    def test_correctness(
        self,
        axis,
        input_shape,
        moving_mean_initializer,
        moving_variance_initializer,
    ):
        # Training
        layer = layers.BatchNormalization(
            axis=axis,
            momentum=0,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
        )
        # Random data centered on 5.0, variance 10.0
        x = np.random.normal(loc=5.0, scale=10.0, size=input_shape)
        out = x
        for _ in range(3):
            out = layer(out, training=True)

        # Assert the normalization is correct.
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]
        out = backend.convert_to_numpy(out)
        out = out - np.reshape(
            backend.convert_to_numpy(layer.beta), broadcast_shape
        )
        out = out / np.reshape(
            backend.convert_to_numpy(layer.gamma), broadcast_shape
        )

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        reduction_axes = tuple(reduction_axes)
        self.assertAllClose(np.mean(out, axis=reduction_axes), 0.0, atol=1e-3)
        self.assertAllClose(np.std(out, axis=reduction_axes), 1.0, atol=1e-3)
        self.assertAllClose(layer.moving_mean, 0.0, atol=1e-3)
        self.assertAllClose(layer.moving_variance, 1.0, atol=1e-3)

        # Inference done before training shouldn't match.
        inference_out = layer(x, training=False)
        training_out = layer(x, training=True)
        self.assertNotAllClose(inference_out, training_out)

        # Since momentum is zero, inference after training should match.
        training_out = layer(x, training=True)
        inference_out = layer(x, training=False)
        self.assertAllClose(inference_out, training_out)

        # Masked result with no training should not differ
        x[:, 1, :] = 0.0
        unmasked_out = layer(x, training=False)
        masked = layers.Masking()(x)
        masked_out = layer(masked, training=False)
        self.assertAllClose(unmasked_out, masked_out)

        # Masked result should differ from unmasked result
        unmasked_out = layer(x, training=False)
        x[:, 1, :] = 0.0
        masked = layers.Masking()(x)
        masked_out = layer(masked, training=True)
        self.assertNotAllClose(unmasked_out, masked_out)

    @parameterized.product(
        synchronized=(
            (False, True) if backend.backend == "tensorflow" else (False,)
        ),
    )
    def test_input_fully_masked(self, synchronized):
        norm = layers.BatchNormalization(
            scale=False,
            center=False,
            synchronized=synchronized,
        )
        x = np.zeros((4, 5))
        mask = np.zeros((4,), dtype=np.float32)
        y = norm(x, mask=mask, training=True)
        self.assertAllClose(y, np.zeros_like(x, dtype=np.float32))

    @parameterized.product(run_eagerly=(True, False), mask_value=(0.0, 0.1, 1))
    @pytest.mark.requires_trainable_backend
    def test_bachnorm_ignore_masked_values(self, run_eagerly, mask_value):
        padded_data = np.array(
            [
                [
                    [1, 5],
                    [2, 5],
                    [mask_value, mask_value],
                    [mask_value, mask_value],
                ]
                for _ in range(10)
            ],
            dtype="float32",
        )

        inputs = layers.Input((None, 2))
        masked = layers.Masking(mask_value=mask_value)(inputs)
        normed = layers.BatchNormalization(momentum=0.0)(masked)
        model = Model(inputs, normed)
        loss = MeanSquaredError()
        model.compile(
            "rmsprop",
            loss=loss,
            run_eagerly=run_eagerly,
        )
        model.fit(x=padded_data, y=padded_data, batch_size=10, epochs=5)
        self.assertAllClose(model.layers[2].moving_mean.numpy(), [1.5, 5.0])
        self.assertAllClose(
            model.layers[2].moving_variance.numpy(), [0.25, 0.0]
        )

    def test_trainable_behavior(self):
        layer = layers.BatchNormalization(axis=-1, momentum=0.8, epsilon=1e-7)
        layer.build((1, 4, 4, 3))
        layer.trainable = False
        self.assertEqual(len(layer.weights), 4)
        self.assertEqual(len(layer.trainable_weights), 0)
        self.assertEqual(len(layer.non_trainable_weights), 4)

        # Random data centered on 5.0, variance 10.0
        x = np.random.normal(loc=5.0, scale=10.0, size=(200, 4, 4, 3))

        out = layer(x, training=True)
        self.assertAllClose(out, x)

        layer.trainable = True
        self.assertEqual(len(layer.weights), 4)
        self.assertEqual(len(layer.trainable_weights), 2)
        self.assertEqual(len(layer.non_trainable_weights), 2)

        for _ in range(10):
            out = layer(x, training=True)

        out = backend.convert_to_numpy(out)
        out = out - np.reshape(
            backend.convert_to_numpy(layer.beta), (1, 1, 1, 3)
        )
        out = out / np.reshape(
            backend.convert_to_numpy(layer.gamma), (1, 1, 1, 3)
        )

        self.assertAllClose(np.mean(out, axis=(0, 1, 2)), 0.0, atol=1e-3)
        self.assertAllClose(np.std(out, axis=(0, 1, 2)), 1.0, atol=1e-3)

    def test_large_value_within_autocast_scope(self):
        layer = layers.BatchNormalization()
        layer.build((1, 4, 4, 3))
        # Use 70000 to trigger overflow for float16
        large_value = ops.full(layer.moving_variance.shape, 70000)
        with backend.AutocastScope("float16"):
            layer.moving_variance.assign(large_value)
            self.assertAllClose(layer.moving_variance.value, large_value)
