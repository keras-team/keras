from keras_core import testing
from keras_core.utils import python_utils


class PythonUtilsTest(testing.TestCase):
    def test_func_dump_and_load(self):
        def my_function(x, y=1, **kwargs):
            return x + y

        serialized = python_utils.func_dump(my_function)
        deserialized = python_utils.func_load(serialized)
        self.assertEqual(deserialized(2, y=3), 5)

    def test_removesuffix(self):
        x = "model.keras"
        self.assertEqual(python_utils.removesuffix(x, ".keras"), "model")
        self.assertEqual(python_utils.removesuffix(x, "model"), x)

    def test_removeprefix(self):
        x = "model.keras"
        self.assertEqual(python_utils.removeprefix(x, "model"), ".keras")
        self.assertEqual(python_utils.removeprefix(x, ".keras"), x)
