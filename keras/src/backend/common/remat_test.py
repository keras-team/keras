from contextlib import nullcontext

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.backend.common import global_state
from keras.src.backend.common.remat import RematScope
from keras.src.backend.common.remat import get_current_remat_mode
from keras.src.backend.common.remat import remat
from keras.src.backend.common.variables import AutocastScope
from keras.src.backend.common.variables import get_autocast_scope
from keras.src.layers import activations


class TestRematScope(testing.TestCase):
    def test_remat_scope_activation(self):
        self.assertIsNone(
            get_current_remat_mode()
        )  # Initially, no mode is active

        with RematScope(mode="full"):
            self.assertEqual(
                get_current_remat_mode().mode, "full"
            )  # Mode is set to "full"

        self.assertIsNone(
            get_current_remat_mode()
        )  # Mode is restored to None after scope ends

        with RematScope(mode=None):
            self.assertIsNone(
                get_current_remat_mode()
            )  # Mode is None when explicitly disabled

    def test_remat_scope_nested(self):
        """Test nested scopes with different rematerialization modes."""
        with RematScope(mode="full"):
            self.assertEqual(
                get_current_remat_mode().mode, "full"
            )  # Outer scope is "full"

            with RematScope(mode="activations"):
                self.assertEqual(
                    get_current_remat_mode().mode, "activations"
                )  # Inner scope is "activations"

            self.assertEqual(
                get_current_remat_mode().mode, "full"
            )  # Back to outer scope

        self.assertIsNone(
            get_current_remat_mode()
        )  # Mode is restored to None after all scopes

    def test_remat_scope_stack_management(self):
        """Test that the remat_scope_stack is managed correctly."""
        self.assertIsNone(
            global_state.get_global_attribute("remat_scope_stack")
        )  # No stack initially

        with RematScope(mode="full"):
            remat_stack = global_state.get_global_attribute("remat_scope_stack")
            self.assertIsNotNone(remat_stack)  # Stack is initialized
            self.assertEqual(len(remat_stack), 1)  # Stack contains one entry

            with RematScope(mode="activations"):
                remat_stack = global_state.get_global_attribute(
                    "remat_scope_stack"
                )
                self.assertEqual(
                    len(remat_stack), 2
                )  # Stack contains two entries

            remat_stack = global_state.get_global_attribute("remat_scope_stack")
            self.assertEqual(len(remat_stack), 1)  # Back to one entry

        self.assertEqual(
            global_state.get_global_attribute("remat_scope_stack"), []
        )  # Stack is cleared

    def test_invalid_mode(self):
        """Test that invalid rematerialization modes raise an error."""
        with self.assertRaises(ValueError):
            RematScope(mode="invalid")  # Invalid mode should raise ValueError


@pytest.mark.skipif(
    not backend.SUPPORTS_GRADIENT,
    reason="remat requires gradient support",
)
class RematTest(testing.TestCase):
    @parameterized.parameters("float32", "mixed_float16", "mixed_bfloat16")
    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="TensorFlow-specific test"
    )
    def test_remat_compiled_training(self, dtype):
        def make_model(mode):
            with RematScope(mode=mode):
                inputs = layers.Input((3,))
                outputs = layers.Dense(
                    2,
                    activation="relu",
                    dtype=dtype,
                    kernel_initializer=initializers.Constant(0.25),
                )(inputs)
                outputs = layers.Dense(
                    1,
                    dtype=dtype,
                    kernel_initializer=initializers.Constant(0.25),
                )(outputs)
                model = models.Model(inputs, outputs, dtype=dtype)
            model.compile(optimizer="sgd", loss="mse", jit_compile=False)
            return model

        plain = make_model(None)
        rematerialized = make_model("full")
        initial_weights = rematerialized.get_weights()
        inputs = np.ones((2, 3), dtype="float32") / 16
        targets = np.zeros((2, 1), dtype="float32")
        for _ in range(2):
            expected = plain.train_on_batch(inputs, targets)
            actual = rematerialized.train_on_batch(inputs, targets)
            self.assertAllClose(actual, expected)
            for actual_weight, expected_weight in zip(
                rematerialized.get_weights(), plain.get_weights()
            ):
                self.assertAllClose(actual_weight, expected_weight)
        self.assertFalse(
            np.array_equal(initial_weights[0], rematerialized.get_weights()[0])
        )

    @parameterized.product(
        layer_type=("dense", "conv1d"),
        dtype=("float32", "mixed_float16", "mixed_bfloat16"),
        traced=(False, True),
    )
    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="TensorFlow-specific test"
    )
    def test_remat_mixed_precision_gradients(self, layer_type, dtype, traced):
        import tensorflow as tf

        def make_layer():
            kwargs = {
                "dtype": dtype,
                "kernel_initializer": "ones",
                "bias_initializer": "zeros",
            }
            if layer_type == "dense":
                return layers.Dense(2, **kwargs)
            return layers.Conv1D(2, 2, padding="same", **kwargs)

        inputs = tf.reshape(tf.range(24, dtype="float32"), (2, 4, 3)) / 16
        plain = make_layer()
        with RematScope(mode="full"):
            rematerialized = make_layer()
        plain.build(inputs.shape)
        rematerialized.build(inputs.shape)
        rematerialized.set_weights(plain.get_weights())

        def value_and_grad(layer, mode):
            sources = [inputs] + [v.value for v in layer.trainable_variables]

            def compute():
                with tf.GradientTape() as tape:
                    tape.watch(inputs)
                    with RematScope(mode=mode):
                        outputs = layer(inputs)
                    loss = tf.reduce_sum(tf.square(tf.cast(outputs, "float32")))
                return outputs, tape.gradient(loss, sources)

            if traced:
                compute = tf.function(compute, jit_compile=False)
            return compute()

        expected_outputs, expected_gradients = value_and_grad(plain, None)
        actual_outputs, actual_gradients = value_and_grad(
            rematerialized, "full"
        )
        self.assertDType(actual_outputs, plain.compute_dtype)
        self.assertAllClose(actual_outputs, expected_outputs)
        for actual, expected in zip(actual_gradients, expected_gradients):
            self.assertIsNotNone(actual)
            self.assertDType(actual, "float32")
            self.assertTrue(np.isfinite(self.convert_to_numpy(actual)).all())
            self.assertAllClose(actual, expected, atol=1e-3, rtol=1e-3)

    @parameterized.parameters("float16", "bfloat16")
    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="TensorFlow-specific test"
    )
    def test_remat_reuses_forward_autocast_scope(self, backward_dtype):
        import tensorflow as tf

        variable = backend.Variable([1.0, 2.0], dtype="float64")
        inputs = tf.constant([0.25, 0.5], dtype="float64")
        sources = [inputs, variable.value]

        def function(x, scale=1.0):
            return (
                tf.cast(x, variable.dtype)
                * tf.cast(variable.value, variable.dtype)
                * scale
            )

        # The context when wrapping the function must not affect later calls.
        with AutocastScope("bfloat16"):
            rematerialized = remat(function)

        def value_and_grad(function, scope):
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                with scope:
                    outputs = function(inputs, scale=2.0)
                    loss = tf.reduce_sum(outputs)
            backward_scope = AutocastScope(backward_dtype)
            with backward_scope:
                gradients = tape.gradient(loss, sources)
                self.assertIs(get_autocast_scope(), backward_scope)
            self.assertIsNone(get_autocast_scope())
            return outputs, gradients

        def assert_matches(plain, rematerialized, scope):
            expected_outputs, expected_gradients = value_and_grad(plain, scope)
            actual_outputs, actual_gradients = value_and_grad(
                rematerialized, scope
            )
            self.assertDType(actual_outputs, expected_outputs.dtype)
            self.assertAllClose(actual_outputs, expected_outputs)
            for actual, expected in zip(actual_gradients, expected_gradients):
                self.assertIsNotNone(actual)
                self.assertAllClose(actual, expected)

        # An absent scope and an explicitly disabled scope have distinct
        # variable dtype behavior. Reuse the same wrapper across both and
        # across changing mixed-precision policies.
        for scope in (
            nullcontext(),
            AutocastScope(None),
            AutocastScope("float16"),
            AutocastScope("bfloat16"),
            AutocastScope("float16"),
        ):
            assert_matches(function, rematerialized, scope)

        def two_calls(function):
            def call(x, scale):
                with AutocastScope("float16"):
                    first = function(x, scale=scale)
                with AutocastScope("bfloat16"):
                    second = function(x, scale=scale)
                return tf.cast(first, "float32") + tf.cast(second, "float32")

            return call

        # Both forward calls must retain their scopes until backward runs.
        assert_matches(
            two_calls(function), two_calls(rematerialized), nullcontext()
        )

    def test_remat_basic_call(self):
        # Generate dummy data
        data_size = 10**5
        x_train = np.random.normal(size=(data_size, 4))
        y_train = np.random.normal(size=(data_size, 1))

        epochs = 5
        batch_size = 512
        # test applying remat
        output_with_remat = backend.ops.remat(activations.ReLU())(x_train)
        output_without_remat = activations.ReLU()(x_train)
        self.assertAllClose(output_with_remat, output_without_remat)
        # test remat in a model
        intermediate_function = backend.ops.remat(activations.ReLU())
        inputs = layers.Input(shape=(4,))
        x = layers.Dense(4)(inputs)
        x = layers.Lambda(intermediate_function)(x)
        outputs = layers.Dense(1)(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.predict(x_train)
        model.compile(optimizer="sgd", loss="mse")

        # Train model
        model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )

    def test_remat_with_kwargs(self):
        # Define a function that uses keyword arguments
        def fn_with_kwargs(x, scale=1.0, offset=0.0):
            return x * scale + offset

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Test with keyword arguments
        remat_fn = backend.ops.remat(fn_with_kwargs)
        result_with_kwargs = remat_fn(x, scale=2.0, offset=1.0)
        expected = fn_with_kwargs(x, scale=2.0, offset=1.0)
        self.assertAllClose(result_with_kwargs, expected)

        # Test with default keyword arguments
        result_with_defaults = remat_fn(x)
        expected_defaults = fn_with_kwargs(x)
        self.assertAllClose(result_with_defaults, expected_defaults)

        # Test with partial keyword arguments
        result_partial = remat_fn(x, scale=3.0)
        expected_partial = fn_with_kwargs(x, scale=3.0)
        self.assertAllClose(result_partial, expected_partial)
