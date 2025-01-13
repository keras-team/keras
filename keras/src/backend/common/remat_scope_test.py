from keras.src import testing
from keras.src.backend.common import global_state
from keras.src.backend.common.remat_scope import RematScope
from keras.src.backend.common.remat_scope import get_current_remat_mode


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
                get_current_remat_mode(), "full"
            )  # Mode is set to "full"

        self.assertIsNone(
            get_current_remat_mode()
        )  # Mode is restored to None after scope ends

    def test_remat_scope_nested(self):
        """Test nested scopes with different rematerialization modes."""
        with RematScope(mode="full"):
            self.assertEqual(
                get_current_remat_mode(), "full"
            )  # Outer scope is "full"

            with RematScope(mode="activations"):
                self.assertEqual(
                    get_current_remat_mode(), "activations"
                )  # Inner scope is "activations"

            self.assertEqual(
                get_current_remat_mode(), "full"
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
