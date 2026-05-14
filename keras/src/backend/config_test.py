from keras.src import backend
from keras.src import testing
from keras.src.backend import config


class TF32ConfigTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        # Remember the starting state so each test restores it.
        self._tf32_was_enabled = config.is_tf32_enabled()

    def tearDown(self):
        if self._tf32_was_enabled is False:
            config.disable_tf32()
        else:
            config.enable_tf32()
        super().tearDown()

    def test_enabled_by_default(self):
        # `is_tf32_enabled()` returns `False` only when explicitly disabled.
        self.assertNotEqual(config.is_tf32_enabled(), False)

    def test_disable_then_enable(self):
        config.disable_tf32()
        self.assertEqual(config.is_tf32_enabled(), False)
        config.enable_tf32()
        self.assertNotEqual(config.is_tf32_enabled(), False)

    def test_toggle_applies_to_active_backend(self):
        """Toggling the Keras flag re-applies to the active backend."""
        if backend.backend() == "torch":
            import torch

            config.disable_tf32()
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
            config.enable_tf32()
            self.assertEqual(torch.get_float32_matmul_precision(), "high")
        elif backend.backend() == "tensorflow":
            import tensorflow as tf

            config.disable_tf32()
            self.assertFalse(
                tf.config.experimental.tensor_float_32_execution_enabled()
            )
            config.enable_tf32()
            self.assertTrue(
                tf.config.experimental.tensor_float_32_execution_enabled()
            )
        elif backend.backend() == "jax":
            import jax

            config.disable_tf32()
            self.assertEqual(jax.config.jax_default_matmul_precision, "highest")
            config.enable_tf32()
            self.assertEqual(jax.config.jax_default_matmul_precision, "default")
        else:
            # Other backends (e.g. numpy/openvino) have no TF32 knob. The
            # flag should still toggle without raising.
            config.disable_tf32()
            config.enable_tf32()
