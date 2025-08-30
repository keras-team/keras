from keras.src.testing import test_case
from keras.src.utils import naming


class NamingUtilsTest(test_case.TestCase):
    def test_uniquify_unique_name(self):
        name = "the_unique_name"
        unique_name = naming.uniquify(name)
        self.assertEqual(unique_name, name)

    def test_auto_name(self):
        self.assertEqual(naming.auto_name("unique_name"), "unique_name")
        self.assertEqual(naming.auto_name("unique_name"), "unique_name_1")
        self.assertEqual(naming.auto_name("unique_name"), "unique_name_2")

    def test_get_uid(self):
        self.assertEqual(naming.get_uid("very_unique_name"), 1)
        self.assertEqual(naming.get_uid("very_unique_name"), 2)
        self.assertEqual(naming.get_uid("very_unique_name"), 3)

    def test_uniquify_non_unique_name(self):
        name = "non_unique_name"
        naming.uniquify(name)
        unique_name = naming.uniquify(name)
        self.assertEqual(unique_name, f"{name}_1")

    def test_to_snake_case_snake_case_name(self):
        name = "snake_case_name"
        snake_case_name = naming.to_snake_case(name)
        self.assertEqual(snake_case_name, name)

    def test_get_uid_existing_prefix(self):
        prefix = "existing_prefix"
        naming.get_uid(prefix)
        uid = naming.get_uid(prefix)
        self.assertEqual(uid, 2)

    def test_reset_uids(self):
        naming.get_uid("unique_name")
        naming.reset_uids()
        uid = naming.get_uid("unique_name")
        self.assertEqual(uid, 1)

    def test_get_object_name_no_name_attribute(self):
        class ObjectWithoutName:
            __name__ = "ObjectWithoutName"

        obj = ObjectWithoutName()
        object_name = naming.get_object_name(obj)
        self.assertEqual(object_name, "object_without_name")

    def test_get_object_name_no_name_or_class_attribute(self):
        class ObjectWithoutNameOrClass:
            pass

        obj = ObjectWithoutNameOrClass()
        object_name = naming.get_object_name(obj)
        self.assertEqual(object_name, "object_without_name_or_class")

    def test_uniquify_already_uniquified_name(self):
        name = "unique_name"
        unique_name = naming.uniquify(name)
        new_unique_name = naming.uniquify(unique_name)

        # first time `name` is uniquified so returns same name
        self.assertEqual(name, unique_name)

        # second time `name` is uniquified should be different
        # from the first output
        self.assertNotEqual(new_unique_name, unique_name)

    def test_to_snake_case_capital_after_any_character(self):
        name = "myVariableNameHere"
        snake_case_name = naming.to_snake_case(name)
        self.assertEqual(snake_case_name, "my_variable_name_here")

    def test_to_snake_case_lower_before_upper(self):
        name = "convertTHIS"
        snake_case_name = naming.to_snake_case(name)
        self.assertEqual(snake_case_name, "convert_this")

    def test_to_snake_case_already_snake_cased(self):
        name = "already_snake_cased"
        snake_case_name = naming.to_snake_case(name)
        self.assertEqual(snake_case_name, name)

    def test_to_snake_case_no_changes(self):
        name = "lowercase"
        snake_case_name = naming.to_snake_case(name)
        self.assertEqual(snake_case_name, name)

    def test_to_snake_case_single_uppercase_word(self):
        name = "UPPERCASE"
        snake_case_name = naming.to_snake_case(name)
        self.assertEqual(snake_case_name, "uppercase")

    def test_get_object_name_for_keras_objects(self):
        class MockKerasObject:
            name = "mock_object"

        obj = MockKerasObject()
        result = naming.get_object_name(obj)
        self.assertEqual(
            result, "mock_object", f"Expected 'mock_object' but got {result}"
        )

    # Test for function objects that have a `__name__` attribute.
    def test_get_object_name_for_functions(self):
        def mock_function():
            pass

        result = naming.get_object_name(mock_function)
        # Assumes to_snake_case works correctly.
        expected_name = naming.to_snake_case(mock_function.__name__)
        self.assertEqual(
            result,
            expected_name,
            f"Expected '{expected_name}' but got {result}",
        )
