from keras_core import backend
from keras_core import ops
from keras_core import testing
from keras_core.ops import operation_utils


class OperationUtilsTest(testing.TestCase):
    def test_get_source_inputs(self):
        x1 = backend.KerasTensor(shape=(2,))
        x2 = backend.KerasTensor(shape=(2,))
        x = x1 + x2
        x += 2
        x = ops.square(x)
        self.assertEqual(operation_utils.get_source_inputs(x), [x1, x2])
