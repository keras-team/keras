from keras_core import backend
from keras_core import initializers
from keras_core.backend.common.variables import AutocastScope
from keras_core.testing import test_case


class VariablesTest(test_case.TestCase):
    def test_deferred_initialization(self):
        with backend.StatelessScope():
            v = backend.Variable(
                initializer=initializers.RandomNormal(), shape=(2, 2)
            )
            self.assertEqual(v._value, None)
            # Variables can nevertheless be accessed
            _ = v + 1
        self.assertEqual(v._value.shape, (2, 2))

        with self.assertRaisesRegex(ValueError, "while in a stateless scope"):
            with backend.StatelessScope():
                v = backend.Variable(initializer=0)

    def test_autocasting(self):
        v = backend.Variable(
            initializer=initializers.RandomNormal(),
            shape=(2, 2),
            dtype="float32",
        )
        self.assertEqual(v.dtype, "float32")
        self.assertEqual(v.value.dtype.name, "float32")

        print("open scope")
        with AutocastScope("float16"):
            self.assertEqual(v.value.dtype.name, "float16")
        self.assertEqual(v.value.dtype.name, "float32")

        # Test non-float variables are not affected
        v = backend.Variable(
            initializer=initializers.Ones(), shape=(2, 2), dtype="int32"
        )
        self.assertEqual(v.dtype, "int32")
        self.assertEqual(v.value.dtype.name, "int32")

        with AutocastScope("float16"):
            self.assertEqual(v.value.dtype.name, "int32")
