import logging
import unittest

import numpy as np
import pytest

from keras.src import backend
from keras.src.backend.numpy.distributed_backend import NumpyDistributedBackend

logging.disable(logging.INFO)


class MockVariable:
    """A mock stateful variable with an `assign` method for testing."""

    def __init__(self, value):
        self.value = np.array(value, dtype=np.float32)

    def assign(self, new_value):
        self.value = np.array(new_value)

    def __sub__(self, other):
        return self.value - other


@pytest.mark.skipif(
    backend.backend() != "numpy",
    reason="NumPy-specific distributed backend tests",
)
class TestNumpyDistributedBackend(unittest.TestCase):
    """Unit tests for the NumpyDistributedBackend class."""

    def setUp(self):
        """Set up the test case by instantiating the backend."""
        self.backend = NumpyDistributedBackend()

    def tearDown(self):
        """Re-enable logging after tests are done."""
        logging.disable(logging.NOTSET)

    def test_get_tensor_lib(self):
        """Test if the correct tensor library (numpy) is returned."""
        self.assertIs(self.backend.get_tensor_lib(), np)

    def test_convert_to_backend_tensor(self):
        """Test tensor conversion to NumPy arrays."""
        py_list = [1.0, 2.0, 3.0]
        np_tensor = self.backend.convert_to_backend_tensor(py_list)
        self.assertIsInstance(np_tensor, np.ndarray)
        np.testing.assert_array_equal(np_tensor, np.array([1.0, 2.0, 3.0]))

    def test_compute_numpy_gradients_returns_zeros(self):
        loss = 15.0
        trainable_vars = [np.array([1.0, 2.0, 3.0]), np.array([[4.0], [5.0]])]

        gradients = self.backend.compute_gradients(loss, trainable_vars)

        self.assertEqual(len(gradients), 2)
        np.testing.assert_array_equal(
            gradients[0], np.zeros_like(trainable_vars[0])
        )
        np.testing.assert_array_equal(
            gradients[1], np.zeros_like(trainable_vars[1])
        )

    def test_apply_gradients_with_slice_assignment(self):
        """Test applying gradients to standard NumPy arrays."""
        var = np.array([10.0, 20.0])
        grad = np.array([0.5, 1.5])

        self.backend.apply_gradients([grad], [var], learning_rate=0.1)

        expected_var = np.array([10.0 - 0.1 * 0.5, 20.0 - 0.1 * 1.5])
        np.testing.assert_allclose(var, expected_var)

    def test_apply_gradients_with_assign_method(self):
        """Test applying gradients to mock objects with an .assign() method."""
        var = MockVariable([10.0, 20.0])
        grad = np.array([0.5, 1.5])

        self.backend.apply_gradients([grad], [var], learning_rate=0.1)

        expected_var = np.array([10.0 - 0.1 * 0.5, 20.0 - 0.1 * 1.5])
        np.testing.assert_allclose(var.value, expected_var)

    def test_create_optimizer(self):
        """Test the creation and functionality of the NumPy optimizer."""
        optimizer = self.backend.create_optimizer(
            optimizer_class="sgd", learning_rate=0.1
        )
        self.assertTrue(hasattr(optimizer, "apply_gradients"))

        var = np.array([10.0, 20.0])
        grad = np.array([2.0, 3.0])

        optimizer.apply_gradients([(grad, var)])

        expected_var = np.array([10.0 - 0.1 * 2.0, 20.0 - 0.1 * 3.0])
        np.testing.assert_allclose(var, expected_var)

    def test_get_device_info(self):
        """Test that device info is correctly reported for NumPy."""
        expected_info = {
            "backend": "numpy",
            "devices": ["cpu"],
            "device_count": 1,
        }
        self.assertDictEqual(self.backend.get_device_info(), expected_info)

    def test_is_multi_device_capable(self):
        """Test that the backend correctly reports single-device capability."""
        self.assertFalse(self.backend.is_multi_device_capable())

    def test_get_communication_ops(self):
        """Test the simulated communication operations."""
        ops = self.backend.get_communication_ops()

        x_reduce = np.array([[1.0, 2.0], [3.0, 4.0]])
        reduced = ops["all_reduce"](x_reduce)
        np.testing.assert_array_equal(reduced, np.array([4.0, 6.0]))

        x_gather = np.array([[1.0, 2.0]])
        gathered = ops["all_gather"](x_gather, axis=0)
        np.testing.assert_array_equal(
            gathered, np.array([[1.0, 2.0], [1.0, 2.0]])
        )

        x_broadcast = np.array([5.0, 6.0])
        broadcasted = ops["broadcast"](x_broadcast)
        np.testing.assert_array_equal(broadcasted, x_broadcast)

        x_scatter = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        scattered = ops["scatter"](x_scatter, num_devices=2)
        self.assertEqual(len(scattered), 2)
        np.testing.assert_array_equal(scattered[0], np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(scattered[1], np.array([[5, 6], [7, 8]]))


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
