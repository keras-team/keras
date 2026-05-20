from keras.src import backend
from keras.src import testing
from keras.src.backend import config
from keras.src.backend.common import global_state


class TF32ConfigTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        # Remember the starting state so each test restores it on teardown.
        self._saved = global_state.get_global_attribute("tf32")

    def tearDown(self):
        global_state.set_global_attribute("tf32", self._saved)
        super().tearDown()

    def test_default_is_disabled(self):
        # Keras does not opt the active backend into TF32 unless asked.
        global_state.set_global_attribute("tf32", None)
        self.assertIsInstance(config.is_tf32_enabled(), bool)
        self.assertFalse(config.is_tf32_enabled())

    def test_enable_then_disable(self):
        config.enable_tf32()
        self.assertIsInstance(config.is_tf32_enabled(), bool)
        self.assertTrue(config.is_tf32_enabled())
        config.disable_tf32()
        self.assertFalse(config.is_tf32_enabled())

    def test_toggle_applies_to_active_backend(self):
        """Toggling the Keras flag re-applies to the active backend."""
        if backend.backend() == "torch":
            import torch

            config.enable_tf32()
            self.assertEqual(torch.get_float32_matmul_precision(), "high")
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            config.disable_tf32()
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        elif backend.backend() == "tensorflow":
            import tensorflow as tf

            config.enable_tf32()
            self.assertTrue(
                tf.config.experimental.tensor_float_32_execution_enabled()
            )
            config.disable_tf32()
            self.assertFalse(
                tf.config.experimental.tensor_float_32_execution_enabled()
            )
        elif backend.backend() == "jax":
            import jax

            config.enable_tf32()
            self.assertEqual(jax.config.jax_default_matmul_precision, "default")
            config.disable_tf32()
            self.assertEqual(jax.config.jax_default_matmul_precision, "highest")
        else:
            # Other backends (e.g. numpy/openvino) have no TF32 knob. The
            # flag should still toggle without raising.
            config.enable_tf32()
            config.disable_tf32()
