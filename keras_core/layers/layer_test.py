from keras_core import testing
from keras_core.engine import keras_tensor
from keras_core.layers.layer import Layer
import numpy as np


class FunctionTest(testing.TestCase):
    def test_positional_arg_error(self):
        class SomeLayer(Layer):
            def call(self, x, bool_arg):
                if bool_arg:
                    return x
                return x + 1

        x = keras_tensor.KerasTensor(shape=(2, 3), name="x")
        with self.assertRaisesRegex(ValueError, "Only input tensors may be passed as"):
            SomeLayer()(x, True)

        # This works
        SomeLayer()(x, bool_arg=True)
