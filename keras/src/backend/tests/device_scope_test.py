import pytest

from keras.src import backend
from keras.src import testing


class DeviceTest(testing.TestCase):
    @pytest.mark.skipif(backend.backend() != "tensorflow", reason="tf only")
    def test_tf_device_scope(self):
        import tensorflow as tf

        if not tf.config.list_physical_devices("GPU"):
            self.skipTest("Need at least one GPU for testing")

        with backend.device("cpu:0"):
            t = backend.numpy.ones((2, 1))
            self.assertIn("CPU:0", t.device)
        with backend.device("CPU:0"):
            t = backend.numpy.ones((2, 1))
            self.assertIn("CPU:0", t.device)

        # When leaving the scope, the device should be back with gpu:0
        t = backend.numpy.ones((2, 1))
        self.assertIn("GPU:0", t.device)

        # Also verify the explicit gpu device
        with backend.device("gpu:0"):
            t = backend.numpy.ones((2, 1))
            self.assertIn("GPU:0", t.device)

    @pytest.mark.skipif(backend.backend() != "jax", reason="jax only")
    def test_jax_device_scope(self):
        import jax

        def get_device(t):
            # After updating to Jax 0.4.33, Directly access via t.device attr.
            return list(t.devices())[0]

        platform = jax.default_backend()

        if platform != "gpu":
            self.skipTest("Need at least one GPU for testing")

        with backend.device("cpu:0"):
            t = backend.numpy.ones((2, 1))
            self.assertEqual(get_device(t), jax.devices("cpu")[0])
        with backend.device("CPU:0"):
            t = backend.numpy.ones((2, 1))
            self.assertEqual(get_device(t), jax.devices("cpu")[0])

        # When leaving the scope, the device should be back with gpu:0
        t = backend.numpy.ones((2, 1))
        self.assertEqual(get_device(t), jax.devices("gpu")[0])

        # Also verify the explicit gpu device
        with backend.device("gpu:0"):
            t = backend.numpy.ones((2, 1))
            self.assertEqual(get_device(t), jax.devices("gpu")[0])

    @pytest.mark.skipif(backend.backend() != "jax", reason="jax only")
    def test_invalid_jax_device(self):
        with self.assertRaisesRegex(ValueError, "Received: device_name='123'"):
            backend.device(123).__enter__()

    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_torch_device_scope(self):
        import torch

        with backend.device("cpu:0"):
            t = backend.numpy.ones((2, 1))
            self.assertEqual(t.device, torch.device("cpu"))
        with backend.device("CPU:0"):
            t = backend.numpy.ones((2, 1))
            self.assertEqual(t.device, torch.device("cpu"))

        # Need at least one GPU for the following testing.
        if not torch.cuda.is_available():
            return

        # When leaving the scope, the device should be back with gpu:0
        t = backend.numpy.ones((2, 1))
        self.assertEqual(t.device, torch.device("cuda", 0))

        # Also verify the explicit gpu -> cuda conversion
        with backend.device("gpu:0"):
            t = backend.numpy.ones((2, 1))
            self.assertEqual(t.device, torch.device("cuda", 0))

    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_invalid_torch_device(self):
        with self.assertRaisesRegex(ValueError, "Received: device_name='123'"):
            backend.device(123).__enter__()

    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_torch_meta_device(self):
        import torch

        with torch.device("meta"):
            x = torch.ones(5)

        t = backend.convert_to_tensor(x)

        if not torch.cuda.is_available():
            self.assertEqual(t.device, torch.device("cpu"))
        else:
            self.assertEqual(t.device, torch.device("cuda", 0))
