from keras_core import testing
from keras_core.utils import python_utils


class PythonUtilsTest(testing.TestCase):
    def test_func_dump_and_load(self):
        def my_function(x, y=1, **kwargs):
            return x + y

        serialized = python_utils.func_dump(my_function)
        deserialized = python_utils.func_load(serialized)
        self.assertEqual(deserialized(2, y=3), 5)
