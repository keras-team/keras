import pytest

from keras import backend
from keras import testing


class DeviceTest(testing.TestCase):
    @pytest.mark.skipif(backend.backend() != "tensorflow", reason="tf only")
    def test_tf_device_scope(self):
        import tensorflow as tf

        if not tf.config.list_physical_devices("GPU"):
            self.skipTest("Need at least one GPU for testing")

        with backend.device_scope("cpu:0"):
            t = backend.numpy.ones((2, 1))
            self.assertIn("CPU:0", t.device)
        with backend.device_scope("CPU:0"):
            t = backend.numpy.ones((2, 1))
            self.assertIn("CPU:0", t.device)

        # When leaving the scope, the device should be back with gpu:0
        t = backend.numpy.ones((2, 1))
        self.assertIn("GPU:0", t.device)

        # Also verify the explicit gpu device
        with backend.device_scope("gpu:0"):
            t = backend.numpy.ones((2, 1))
            self.assertIn("GPU:0", t.device)

    @pytest.mark.skipif(backend.backend() != "jax", reason="jax only")
    def test_jax_device_scope(self):
        import jax
        from jax.lib import xla_bridge

        platform = xla_bridge.get_backend().platform

        if platform != "gpu":
            self.skipTest("Need at least one GPU for testing")

        with backend.device_scope("cpu:0"):
            t = backend.numpy.ones((2, 1))
            self.assertEqual(t.device(), jax.devices("cpu")[0])
        with backend.device_scope("CPU:0"):
            t = backend.numpy.ones((2, 1))
            self.assertEqual(t.device(), jax.devices("cpu")[0])

        # When leaving the scope, the device should be back with gpu:0
        t = backend.numpy.ones((2, 1))
        self.assertEqual(t.device(), jax.devices("gpu")[0])

        # Also verify the explicit gpu device
        with backend.device_scope("gpu:0"):
            t = backend.numpy.ones((2, 1))
            self.assertEqual(t.device(), jax.devices("gpu")[0])

    @pytest.mark.skipif(backend.backend() != "jax", reason="jax only")
    def test_invalid_jax_device(self):
        with self.assertRaisesRegex(ValueError, "Received: device_name='123'"):
            backend.device_scope(123).__enter__()

    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_torch_device_scope(self):
        import torch

        if not torch.cuda.device_count():
            self.skipTest("Need at least one GPU for testing")

        with backend.device_scope("cpu:0"):
            t = backend.numpy.ones((2, 1))
            self.assertEqual(t.device, torch.device("cpu"))
        with backend.device_scope("CPU:0"):
            t = backend.numpy.ones((2, 1))
            self.assertEqual(t.device, torch.device("cpu"))

        # When leaving the scope, the device should be back with gpu:0
        t = backend.numpy.ones((2, 1))
        self.assertEqual(t.device, torch.device("cuda", 0))

        # Also verify the explicit gpu -> cuda conversion
        with backend.device_scope("gpu:0"):
            t = backend.numpy.ones((2, 1))
            self.assertEqual(t.device, torch.device("cuda", 0))

    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_invalid_torch_device(self):
        with self.assertRaisesRegex(ValueError, "Received: device_name='123'"):
            backend.device_scope(123).__enter__()
