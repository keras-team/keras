import numpy as np

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.backend.common import global_state
from keras.src.backend.common.remat import RematScope
from keras.src.backend.common.remat import get_current_remat_mode
from keras.src.layers import activations


class TestRematScope(testing.TestCase):
    def setUp(self):
        """Reset global state before each test."""
        global_state.clear_session()

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


class RematTest(testing.TestCase):
    def test_remat_basic_call(self):
        if backend.backend() in ("openvino", "numpy"):
            self.skipTest(
                "remat is not supported in openvino and numpy backends."
            )
        # Generate dummy data
        data_size = 10**5
        x_train = np.random.normal(size=(data_size, 4))
        y_train = np.random.normal(size=(data_size, 1))

        epochs = 5
        batch_size = 512
        # test applying remat
        output_with_remat = backend.core.remat(activations.ReLU())(x_train)
        output_without_remat = activations.ReLU()(x_train)
        self.assertAllClose(output_with_remat, output_without_remat)
        # test remat in a model
        intermediate_function = backend.core.remat(activations.ReLU())
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
        if backend.backend() in ("openvino", "numpy"):
            self.skipTest(
                "remat is not supported in openvino and numpy backends."
            )

        # Define a function that uses keyword arguments
        def fn_with_kwargs(x, scale=1.0, offset=0.0):
            return x * scale + offset

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Test with keyword arguments
        remat_fn = backend.core.remat(fn_with_kwargs)
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
