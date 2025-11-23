import threading

from keras.src import testing
from keras.src.backend.common import global_state
from keras.src.backend.common.name_scope import current_path
from keras.src.backend.common.name_scope import name_scope


class NameScopeTest(testing.TestCase):
    def test_stacking(self):
        self.assertEqual(current_path(), "")
        with name_scope("outer") as outer:
            self.assertEqual(outer.name, "outer")
            self.assertEqual(current_path(), "outer")
            with name_scope("middle") as middle:
                self.assertEqual(middle.name, "middle")
                self.assertEqual(current_path(), "outer/middle")
                with name_scope("inner") as inner:
                    self.assertEqual(inner.name, "inner")
                    self.assertEqual(current_path(), "outer/middle/inner")
                self.assertEqual(current_path(), "outer/middle")
            self.assertEqual(current_path(), "outer")
        self.assertEqual(current_path(), "")

    def test_deduplication(self):
        self.assertEqual(current_path(), "")
        with name_scope("name", caller=1):
            with name_scope("name", caller=1):
                self.assertEqual(current_path(), "name")
        self.assertEqual(current_path(), "")
        with name_scope("name"):
            with name_scope("name"):
                self.assertEqual(current_path(), "name/name")

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "must be a string"):
            name_scope("foo/bar")
        with self.assertRaisesRegex(ValueError, "must be a string"):
            name_scope(4)

    def test_override_parent(self):
        self.assertEqual(current_path(), "")
        with name_scope("outer"):
            self.assertEqual(current_path(), "outer")
            with name_scope("middle", override_parent="/absolute/path"):
                self.assertEqual(current_path(), "absolute/path/middle")
                with name_scope("inner"):
                    self.assertEqual(
                        current_path(), "absolute/path/middle/inner"
                    )
            self.assertEqual(current_path(), "outer")

    def test_exit_with_none_stack(self):
        """Test that __exit__ handles None name_scope_stack gracefully."""
        # Create a name_scope instance
        scope = name_scope("test")
        # Enter the scope normally
        scope.__enter__()

        # Simulate the scenario where global state is cleared
        # (e.g., in a different thread)
        global_state.set_global_attribute("name_scope_stack", None)

        # Exit should not raise an AttributeError
        scope.__exit__()

        # Clean up: reset the stack
        global_state.set_global_attribute("name_scope_stack", [])

    def test_exit_with_empty_stack(self):
        """Test that __exit__ handles empty name_scope_stack gracefully."""
        # Create a name_scope instance
        scope = name_scope("test")
        # Enter the scope normally
        scope.__enter__()

        # Simulate the scenario where the stack is cleared
        name_scope_stack = global_state.get_global_attribute("name_scope_stack")
        name_scope_stack.clear()

        # Exit should not raise an IndexError
        scope.__exit__()

        # Verify stack is still empty
        name_scope_stack = global_state.get_global_attribute(
            "name_scope_stack", default=[]
        )
        self.assertEqual(len(name_scope_stack), 0)

    def test_multithreaded_name_scope(self):
        """Test name_scope in multithreaded environment."""
        results = []

        def thread_function(thread_id):
            # Each thread should have its own name_scope_stack
            with name_scope(f"thread_{thread_id}"):
                path = current_path()
                results.append(path)
                # Verify we get the expected path
                self.assertEqual(path, f"thread_{thread_id}")

        # Create and start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=thread_function, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all threads executed successfully
        self.assertEqual(len(results), 5)

    def test_exit_without_pop_on_exit(self):
        """Test that __exit__ respects _pop_on_exit flag."""
        # Create a name_scope but don't enter it
        scope = name_scope("test")
        # _pop_on_exit should be False
        self.assertFalse(scope._pop_on_exit)

        # Set up a stack manually
        global_state.set_global_attribute("name_scope_stack", [scope])

        scope.__exit__()

        # Verify the stack still contains the scope
        name_scope_stack = global_state.get_global_attribute("name_scope_stack")
        self.assertEqual(len(name_scope_stack), 1)

        # Clean up
        global_state.set_global_attribute("name_scope_stack", [])
