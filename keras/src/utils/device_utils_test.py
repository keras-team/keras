"""Tests for keras.utils.device_utils."""

from keras.src import testing
from keras.src.utils import device_utils


class TestGetMemoryInfo(testing.TestCase):
    def test_returns_dict_with_required_keys(self):
        try:
            result = device_utils.get_memory_info("GPU:0")
            self.assertIsInstance(result, dict)
            self.assertIn("allocated", result)
            self.assertIn("peak", result)
        except NotImplementedError:
            self.skipTest("Backend does not support memory info")

    def test_allocated_and_peak_are_integers(self):
        try:
            result = device_utils.get_memory_info("GPU:0")
            self.assertIsInstance(result["allocated"], int)
            self.assertIsInstance(result["peak"], int)
        except NotImplementedError:
            self.skipTest("Backend does not support memory info")

    def test_peak_gte_allocated(self):
        try:
            result = device_utils.get_memory_info("GPU:0")
            self.assertGreaterEqual(result["peak"], result["allocated"])
        except NotImplementedError:
            self.skipTest("Backend does not support memory info")

    def test_numpy_backend_raises_not_implemented(self):
        from keras.src.backend.numpy.core import get_memory_info

        with self.assertRaises(NotImplementedError):
            get_memory_info("GPU:0")

    def test_openvino_backend_raises_not_implemented(self):
        from keras.src.backend.openvino.core import get_memory_info

        with self.assertRaises(NotImplementedError):
            get_memory_info("GPU:0")

    def test_torch_cpu_fallback_returns_dict(self):
        from keras.src.backend.torch.core import get_memory_info

        result = get_memory_info("cpu")
        self.assertIn("allocated", result)
        self.assertIn("peak", result)

    def test_tf_invalid_device_returns_zeros(self):
        from keras.src.backend.tensorflow.core import get_memory_info

        result = get_memory_info("GPU:99")
        self.assertEqual(result, {"allocated": 0, "peak": 0})

    def test_torch_device_object(self):
        import torch

        from keras.src.backend.torch.core import get_memory_info

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        result = get_memory_info(device)
        self.assertIn("allocated", result)
        self.assertIn("peak", result)
