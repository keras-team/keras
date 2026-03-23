"""Tests for keras.src.api_export."""

from keras.src import testing
from keras.src.api_export import REGISTERED_NAMES_TO_OBJS
from keras.src.api_export import REGISTERED_OBJS_TO_NAMES
from keras.src.api_export import get_name_from_symbol
from keras.src.api_export import get_symbol_from_name
from keras.src.api_export import keras_export
from keras.src.api_export import register_internal_serializable


class RegisterInternalSerializableTest(testing.TestCase):
    def test_register_string_path(self):
        sentinel = object()
        register_internal_serializable("keras._test.path.MyClass", sentinel)
        self.assertIs(
            REGISTERED_NAMES_TO_OBJS["keras._test.path.MyClass"], sentinel
        )
        self.assertEqual(
            REGISTERED_OBJS_TO_NAMES[sentinel], "keras._test.path.MyClass"
        )

    def test_register_list_path_uses_first(self):
        sentinel = object()
        register_internal_serializable(
            ["keras._test.path.First", "keras._test.path.Second"], sentinel
        )
        self.assertIs(
            REGISTERED_NAMES_TO_OBJS["keras._test.path.First"], sentinel
        )
        self.assertEqual(
            REGISTERED_OBJS_TO_NAMES[sentinel], "keras._test.path.First"
        )


class GetSymbolFromNameTest(testing.TestCase):
    def test_existing_name(self):
        sentinel = object()
        register_internal_serializable("keras._test.lookup.Foo", sentinel)
        self.assertIs(get_symbol_from_name("keras._test.lookup.Foo"), sentinel)

    def test_missing_name_returns_none(self):
        self.assertIsNone(get_symbol_from_name("completely.unknown.path"))


class GetNameFromSymbolTest(testing.TestCase):
    def test_existing_symbol(self):
        sentinel = object()
        register_internal_serializable("keras._test.reverse.Bar", sentinel)
        self.assertEqual(
            get_name_from_symbol(sentinel), "keras._test.reverse.Bar"
        )

    def test_missing_symbol_returns_none(self):
        self.assertIsNone(get_name_from_symbol(object()))


class KerasExportDecoratorTest(testing.TestCase):
    def test_decorator_registers_class(self):
        @keras_export("keras._test.decorator.MyTestClass")
        class MyTestClass:
            pass

        self.assertIs(
            get_symbol_from_name("keras._test.decorator.MyTestClass"),
            MyTestClass,
        )

    def test_decorator_registers_function(self):
        @keras_export("keras._test.decorator.my_test_func")
        def my_test_func():
            return 42

        self.assertIs(
            get_symbol_from_name("keras._test.decorator.my_test_func"),
            my_test_func,
        )
        self.assertEqual(my_test_func(), 42)

    def test_decorator_preserves_function(self):
        @keras_export("keras._test.decorator.identity_func")
        def identity_func(x):
            return x

        self.assertEqual(identity_func(10), 10)


if __name__ == "__main__":
    testing.run_tests()
