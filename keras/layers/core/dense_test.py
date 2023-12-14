import os

import numpy as np
import pytest

from keras import backend
from keras import constraints
from keras import layers
from keras import models
from keras import saving
from keras import testing
from keras.backend.common import keras_tensor


class DenseTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_dense_basics(self):
        # 2D case, no bias.
        self.run_layer_test(
            layers.Dense,
            init_kwargs={
                "units": 4,
                "activation": "relu",
                "kernel_initializer": "random_uniform",
                "bias_initializer": "ones",
                "use_bias": False,
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 4),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        # 3D case, some regularizers.
        self.run_layer_test(
            layers.Dense,
            init_kwargs={
                "units": 5,
                "activation": "sigmoid",
                "kernel_regularizer": "l2",
                "bias_regularizer": "l2",
            },
            input_shape=(2, 3, 4),
            expected_output_shape=(2, 3, 5),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=2,  # we have 2 regularizers.
            supports_masking=True,
        )

    def test_dense_correctness(self):
        # With bias and activation.
        layer = layers.Dense(units=2, activation="relu")
        layer.build((1, 2))
        layer.set_weights(
            [
                np.array([[1.0, -2.0], [3.0, -4.0]]),
                np.array([5.0, -6.0]),
            ]
        )
        inputs = np.array(
            [[-1.0, 2.0]],
        )
        self.assertAllClose(layer(inputs), [[10.0, 0.0]])

        # Just a kernel matmul.
        layer = layers.Dense(units=2, use_bias=False)
        layer.build((1, 2))
        layer.set_weights(
            [
                np.array([[1.0, -2.0], [3.0, -4.0]]),
            ]
        )
        inputs = np.array(
            [[-1.0, 2.0]],
        )
        self.assertEqual(layer.bias, None)
        self.assertAllClose(layer(inputs), [[5.0, -6.0]])

    def test_dense_errors(self):
        with self.assertRaisesRegex(ValueError, "incompatible with the layer"):
            layer = layers.Dense(units=2, activation="relu")
            layer(keras_tensor.KerasTensor((1, 2)))
            layer(keras_tensor.KerasTensor((1, 3)))

    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason="Backend does not support sparse tensors.",
    )
    def test_dense_sparse(self):
        import tensorflow as tf

        self.run_layer_test(
            layers.Dense,
            init_kwargs={
                "units": 4,
            },
            input_shape=(2, 3),
            input_sparse=True,
            expected_output_shape=(2, 4),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
        )

        inputs = 4 * backend.random.uniform((10, 10))
        inputs = tf.sparse.from_dense(tf.nn.dropout(inputs, 0.8))

        layer = layers.Dense(units=5)
        outputs = layer(inputs)

        # Verify the computation is the same as if it had been a dense tensor
        expected_outputs = tf.add(
            tf.matmul(tf.sparse.to_dense(inputs), layer.kernel),
            layer.bias,
        )
        self.assertAllClose(outputs, expected_outputs)

        # Verify the gradient is sparse
        with tf.GradientTape() as g:
            outputs = layer(inputs)

        self.assertIsInstance(
            g.gradient(outputs, layer.kernel), tf.IndexedSlices
        )

    def test_dense_no_activation(self):
        layer = layers.Dense(units=2, use_bias=False, activation=None)
        layer.build((1, 2))
        layer.set_weights(
            [
                np.array([[1.0, -2.0], [3.0, -4.0]]),
            ]
        )
        inputs = np.array(
            [[-1.0, 2.0]],
        )
        self.assertEqual(layer.bias, None)
        self.assertAllClose(layer(inputs), [[5.0, -6.0]])

    def test_dense_without_activation_set(self):
        layer = layers.Dense(units=2, use_bias=False)
        layer.build((1, 2))
        layer.set_weights(
            [
                np.array([[1.0, -2.0], [3.0, -4.0]]),
            ]
        )
        layer.activation = None
        inputs = np.array(
            [[-1.0, 2.0]],
        )
        self.assertEqual(layer.bias, None)
        self.assertAllClose(layer(inputs), [[5.0, -6.0]])

    def test_dense_with_activation(self):
        layer = layers.Dense(units=2, use_bias=False, activation="relu")
        layer.build((1, 2))
        layer.set_weights(
            [
                np.array([[1.0, -2.0], [3.0, -4.0]]),
            ]
        )

        inputs = np.array(
            [[-1.0, 2.0]],
        )
        output = layer(inputs)
        expected_output = np.array([[5.0, 0.0]])
        self.assertAllClose(output, expected_output)

    def test_dense_constraints(self):
        layer = layers.Dense(units=2, kernel_constraint="non_neg")
        layer.build((None, 2))
        self.assertIsInstance(layer.kernel.constraint, constraints.NonNeg)
        layer = layers.Dense(units=2, bias_constraint="non_neg")
        layer.build((None, 2))
        self.assertIsInstance(layer.bias.constraint, constraints.NonNeg)

    @pytest.mark.requires_trainable_backend
    def test_enable_lora(self):
        layer = layers.Dense(units=16)
        layer.build((None, 8))
        layer.enable_lora(4)
        self.assertLen(layer.trainable_weights, 3)
        self.assertLen(layer.non_trainable_weights, 1)
        # Try eager call
        x = np.random.random((64, 8))
        y = np.random.random((64, 16))
        _ = layer(x[:2])

        # Try calling fit()
        model = models.Sequential(
            [
                layer,
            ]
        )
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y)

        # Try saving and reloading the model
        temp_filepath = os.path.join(self.get_temp_dir(), "lora_model.keras")
        model.save(temp_filepath)

        new_model = saving.load_model(temp_filepath)
        self.assertFalse(new_model.layers[0].lora_enabled)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "lora_model.weights.h5"
        )
        model.save_weights(temp_filepath)

        # Load the file into a fresh, non-lora model
        new_model = models.Sequential(
            [
                layers.Dense(units=16),
            ]
        )
        new_model.build((None, 8))
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try loading a normal checkpoint into a lora model
        new_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    @pytest.mark.requires_trainable_backend
    def test_lora_rank_argument(self):
        self.run_layer_test(
            layers.Dense,
            init_kwargs={
                "units": 5,
                "activation": "sigmoid",
                "kernel_regularizer": "l2",
                "lora_rank": 2,
            },
            input_shape=(2, 3, 4),
            expected_output_shape=(2, 3, 5),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=1,
            expected_num_seed_generators=0,
            expected_num_losses=2,  # we have 2 regularizers.
            supports_masking=True,
        )
