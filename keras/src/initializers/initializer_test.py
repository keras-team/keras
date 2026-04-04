"""Tests for keras.src.initializers.initializer (base Initializer class)."""

from keras.src import testing
from keras.src.initializers.initializer import Initializer


class InitializerBaseTest(testing.TestCase):
    def test_call_raises_not_implemented(self):
        init = Initializer()
        with self.assertRaisesRegex(NotImplementedError, "__call__"):
            init((2, 3))

    def test_get_config_returns_empty_dict(self):
        init = Initializer()
        self.assertEqual(init.get_config(), {})

    def test_from_config_builds_instance(self):
        init = Initializer.from_config({})
        self.assertIsInstance(init, Initializer)


class CustomInitializerTest(testing.TestCase):
    """Test a concrete subclass to verify the base API contract."""

    def test_subclass_call(self):
        class ConstantInit(Initializer):
            def __init__(self, value=0.0):
                self.value = value

            def __call__(self, shape, dtype=None):
                import numpy as np

                return np.full(shape, self.value, dtype=dtype or "float32")

            def get_config(self):
                return {"value": self.value}

        init = ConstantInit(value=5.0)
        result = init((2, 3))
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(float(result[0, 0]), 5.0)

    def test_subclass_from_config(self):
        class ConstantInit(Initializer):
            def __init__(self, value=0.0):
                self.value = value

            def __call__(self, shape, dtype=None):
                import numpy as np

                return np.full(shape, self.value, dtype=dtype or "float32")

            def get_config(self):
                return {"value": self.value}

        init = ConstantInit(value=3.0)
        config = init.get_config()
        restored = ConstantInit.from_config(config)
        self.assertEqual(restored.value, 3.0)

    def test_subclass_clone(self):
        class ConstantInit(Initializer):
            def __init__(self, value=0.0):
                self.value = value

            def __call__(self, shape, dtype=None):
                import numpy as np

                return np.full(shape, self.value, dtype=dtype or "float32")

            def get_config(self):
                return {"value": self.value}

        init = ConstantInit(value=7.0)
        cloned = init.clone()
        self.assertIsInstance(cloned, ConstantInit)
        self.assertEqual(cloned.value, 7.0)
        self.assertIsNot(cloned, init)


if __name__ == "__main__":
    testing.run_tests()
