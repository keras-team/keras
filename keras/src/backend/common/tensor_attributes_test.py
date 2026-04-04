"""Tests for keras.src.backend.common.tensor_attributes."""

import numpy as np

from keras.src import backend
from keras.src import testing
from keras.src.backend.common.tensor_attributes import get_tensor_attr
from keras.src.backend.common.tensor_attributes import set_tensor_attr


class SetGetTensorAttrTest(testing.TestCase):
    def test_set_and_get_on_numpy_array(self):
        """NumPy arrays support setattr, so direct attr should work."""
        t = np.zeros((2, 3))
        set_tensor_attr(t, "my_custom_attr", "hello")
        self.assertEqual(get_tensor_attr(t, "my_custom_attr"), "hello")

    def test_set_none_value(self):
        t = np.zeros((2,))
        set_tensor_attr(t, "test_attr", "value")
        set_tensor_attr(t, "test_attr", None)
        # After setting None, should still be accessible but None
        result = get_tensor_attr(t, "test_attr")
        self.assertIsNone(result)

    def test_get_nonexistent_attr_returns_none(self):
        t = np.zeros((2,))
        result = get_tensor_attr(t, "nonexistent_attr_xyz")
        self.assertIsNone(result)

    def test_overwrite_attr(self):
        t = np.zeros((2,))
        set_tensor_attr(t, "counter", 1)
        set_tensor_attr(t, "counter", 2)
        self.assertEqual(get_tensor_attr(t, "counter"), 2)

    def test_different_attrs_on_same_tensor(self):
        t = np.zeros((2,))
        set_tensor_attr(t, "attr_a", "A")
        set_tensor_attr(t, "attr_b", "B")
        self.assertEqual(get_tensor_attr(t, "attr_a"), "A")
        self.assertEqual(get_tensor_attr(t, "attr_b"), "B")


class TensorAttrWithBackendTensorTest(testing.TestCase):
    def test_set_and_get_on_backend_tensor(self):
        """Backend tensors may not support setattr, using dict fallback."""
        t = backend.convert_to_tensor(
            np.array([1.0, 2.0, 3.0], dtype="float32")
        )
        set_tensor_attr(t, "custom_flag", True)
        self.assertTrue(get_tensor_attr(t, "custom_flag"))

    def test_set_none_on_backend_tensor(self):
        t = backend.convert_to_tensor(np.array([1.0], dtype="float32"))
        set_tensor_attr(t, "flag", "val")
        set_tensor_attr(t, "flag", None)
        result = get_tensor_attr(t, "flag")
        self.assertIsNone(result)

    def test_multiple_tensors_independent(self):
        t1 = backend.convert_to_tensor(np.array([1.0], dtype="float32"))
        t2 = backend.convert_to_tensor(np.array([2.0], dtype="float32"))
        set_tensor_attr(t1, "tag", "first")
        set_tensor_attr(t2, "tag", "second")
        self.assertEqual(get_tensor_attr(t1, "tag"), "first")
        self.assertEqual(get_tensor_attr(t2, "tag"), "second")


if __name__ == "__main__":
    testing.run_tests()
