import numpy as np

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

    def test_deferred_assignment(self):
        with backend.StatelessScope() as scope:
            v = backend.Variable(
                initializer=initializers.RandomNormal(), shape=(2, 2)
            )
            self.assertEqual(v._value, None)
            v.assign(np.zeros((2, 2)))
            v.assign_add(2 * np.ones((2, 2)))
            v.assign_sub(np.ones((2, 2)))
        out = scope.get_current_value(v)
        self.assertAllClose(out, np.ones((2, 2)))

    def test_autocasting(self):
        v = backend.Variable(
            initializer=initializers.RandomNormal(),
            shape=(2, 2),
            dtype="float32",
        )
        self.assertEqual(v.dtype, "float32")
        self.assertEqual(backend.standardize_dtype(v.value.dtype), "float32")

        print("open scope")
        with AutocastScope("float16"):
            self.assertEqual(
                backend.standardize_dtype(v.value.dtype), "float16"
            )
        self.assertEqual(backend.standardize_dtype(v.value.dtype), "float32")

        # Test non-float variables are not affected
        v = backend.Variable(
            initializer=initializers.Ones(),
            shape=(2, 2),
            dtype="int32",
            trainable=False,
        )
        self.assertEqual(v.dtype, "int32")
        self.assertEqual(backend.standardize_dtype(v.value.dtype), "int32")

        with AutocastScope("float16"):
            self.assertEqual(backend.standardize_dtype(v.value.dtype), "int32")

    def test_standardize_dtype_with_torch_dtype(self):
        import torch

        x = torch.randn(4, 4)
        backend.standardize_dtype(x.dtype)
