import os

from keras.src import backend
from keras.src import testing
from keras.src.backend.common import global_state
from keras.src.backend.config import disable_jit_cache
from keras.src.backend.config import enable_jit_cache
from keras.src.backend.config import is_jit_cache_enabled


class JitCacheConfigTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self._saved = global_state.get_global_attribute(
            "jit_cache_dir", default=None
        )
        disable_jit_cache()

    def tearDown(self):
        super().tearDown()
        global_state.set_global_attribute("jit_cache_dir", self._saved)

    def test_default_is_disabled(self):
        self.assertFalse(is_jit_cache_enabled())

    def test_enable_default_path(self):
        enable_jit_cache()
        path = is_jit_cache_enabled()
        self.assertTrue(path)
        self.assertEqual(path, os.path.join(os.path.expanduser("~"), ".keras", "jit_cache"))

    def test_enable_custom_path(self):
        enable_jit_cache("/tmp/keras_jit_cache_test")
        self.assertEqual(is_jit_cache_enabled(), "/tmp/keras_jit_cache_test")

    def test_disable_after_enable(self):
        enable_jit_cache("/tmp/keras_jit_cache_test")
        disable_jit_cache()
        self.assertFalse(is_jit_cache_enabled())

    def test_jax_runtime_picks_up_path(self):
        # On the JAX backend, enable_jit_cache should push the path into
        # jax.config. On other backends this call is a no-op and we only
        # check the global_state stash.
        enable_jit_cache("/tmp/keras_jit_cache_test")
        if backend.backend() == "jax":
            import jax

            self.assertEqual(
                jax.config.jax_compilation_cache_dir,
                "/tmp/keras_jit_cache_test",
            )
