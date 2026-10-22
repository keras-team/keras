import warnings

import pytest

from keras.src import backend
from keras.src import testing
from keras.src.backend import config
from keras.src.backend.common import dtypes
from keras.src.testing.test_utils import x64_mode_scope

X64_SUPPORTED = backend.backend() in config._X64_BACKENDS
JAX_ONLY = pytest.mark.skipif(backend.backend() != "jax", reason="JAX only.")


class X64ModeTest(testing.TestCase):
    def test_default_is_off(self):
        self.assertEqual(config.x64_mode(), "off")

    @pytest.mark.skipif(not X64_SUPPORTED, reason="Unsupported backend.")
    def test_set_and_restore(self):
        with x64_mode_scope("explicit_x64"):
            self.assertEqual(config.x64_mode(), "explicit_x64")
        self.assertEqual(config.x64_mode(), "off")

    def test_unknown_mode_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown `x64_mode`"):
            config.set_x64_mode("bogus")

    def test_enable_x64_is_not_supported_yet(self):
        with self.assertRaisesRegex(ValueError, "not supported yet"):
            config.set_x64_mode("enable_x64")

    @pytest.mark.skipif(X64_SUPPORTED, reason="Supported backend.")
    def test_unsupported_backend_raises(self):
        with self.assertRaisesRegex(ValueError, "not supported on the"):
            config.set_x64_mode("explicit_x64")
        self.assertEqual(config.x64_mode(), "off")

    @JAX_ONLY
    def test_explicit_x64_sets_jax_flag(self):
        import jax

        previous = jax.config.jax_explicit_x64_dtypes
        try:
            config.set_x64_mode("explicit_x64")
            self.assertEqual(jax.config.jax_explicit_x64_dtypes.name, "ALLOW")
        finally:
            config.set_x64_mode("off")
            jax.config.update("jax_explicit_x64_dtypes", previous)

    @JAX_ONLY
    def test_explicit_x64_rejects_jax_enable_x64(self):
        import jax

        jax.config.update("jax_enable_x64", True)
        try:
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                config.set_x64_mode("explicit_x64")
        finally:
            jax.config.update("jax_enable_x64", False)
        self.assertEqual(config.x64_mode(), "off")

    @JAX_ONLY
    def test_jax_flag_readers(self):
        import jax

        self.assertFalse(config._jax_explicit_x64_allowed())
        self.assertFalse(config._jax_x64_enabled())
        with x64_mode_scope("explicit_x64"):
            self.assertTrue(config._jax_explicit_x64_allowed())
            self.assertFalse(config._jax_x64_enabled())
        jax.config.update("jax_enable_x64", True)
        try:
            self.assertTrue(config._jax_x64_enabled())
        finally:
            jax.config.update("jax_enable_x64", False)

    @JAX_ONLY
    def test_jax_explicit_flag_alone_is_followed_with_one_warning(self):
        import jax

        dtypes._warn_jax_x64_without_keras_mode.cache_clear()
        previous = jax.config.jax_explicit_x64_dtypes
        jax.config.update("jax_explicit_x64_dtypes", "allow")
        try:
            with self.assertWarnsRegex(UserWarning, "Keras follows it"):
                self.assertEqual(backend.result_type("int64", "int64"), "int64")
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # Must not warn a second time.
                # The `explicit_x64` rules apply, not `enable_x64`.
                self.assertEqual(backend.result_type("int64", int), "int64")
                self.assertEqual(
                    backend.result_type("uint32", "int32"), "int32"
                )
        finally:
            jax.config.update("jax_explicit_x64_dtypes", previous)

    @JAX_ONLY
    def test_jax_enable_x64_alone_raises_on_64bit_input(self):
        import jax

        jax.config.update("jax_enable_x64", True)
        try:
            self.assertEqual(backend.result_type("int32", "int32"), "int32")
            with self.assertRaisesRegex(ValueError, "not support"):
                backend.result_type("int64", "int64")
        finally:
            jax.config.update("jax_enable_x64", False)
