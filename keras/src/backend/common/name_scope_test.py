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

    def test_exit_with_empty_stack(self):
        global_state.set_global_attribute("name_scope_stack", [])

        scope = name_scope("test")
        scope._pop_on_exit = True

        try:
            scope.__exit__()
            success = True
        except (AttributeError, IndexError):
            success = False

        self.assertTrue(success)

    def test_exit_with_none_stack(self):
        global_state.set_global_attribute("name_scope_stack", None)

        scope = name_scope("test")
        scope._pop_on_exit = True

        try:
            scope.__exit__()
            success = True
        except (AttributeError, IndexError):
            success = False

        self.assertTrue(success)

    def test_exit_without_pop_on_exit(self):
        global_state.set_global_attribute("name_scope_stack", ["dummy"])

        scope = name_scope("test")
        scope._pop_on_exit = False

        scope.__exit__()

        name_scope_stack = global_state.get_global_attribute("name_scope_stack")
        self.assertEqual(len(name_scope_stack), 1)

    def test_normal_exit_still_works(self):
        self.assertEqual(current_path(), "")

        with name_scope("test1"):
            self.assertEqual(current_path(), "test1")
            with name_scope("test2"):
                self.assertEqual(current_path(), "test1/test2")
            self.assertEqual(current_path(), "test1")

        self.assertEqual(current_path(), "")
