import unittest

from keras import backend
from keras import testing


class DeviceTest(testing.TestCase):
    def test_device_scope(self):
        import tensorflow as tf

        if not tf.config.list_physical_devices("GPU"):
            self.skipTest("Need at least one GPU for testing")

        with backend.device("cpu:0"):
            t = keras.backend.numpy.ones((2, 1))
            self._assertDevice(t, "gpu", 1)

        # When leaving the scope, the device should be back with gpu:0
        t = tf.ones((2, 1))
        self._assertDevice(t, "gpu", 0)

    def _assertDevice(self, t, expected_device_type, expected_device_id):
        if backend.backend() == "jax":
            import jax

            self.assertEqual(
                t.device(),
                jax.list_devices(expected_device_type)[expected_device_id],
            )
        elif backend.backend() == "tensorflow":
            self.assertIn(
                f"{expected_device_type}:{expected_device_id}", t.device
            )
        elif backend.backend() == "torch":
            import torch

            self.assertEqual(
                t.device, torch.device(expected_device_type, expected_device_id)
            )
