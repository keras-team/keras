"""Tests for keras.src.utils.jax_utils."""

from keras.src import backend
from keras.src import testing
from keras.src.utils.jax_utils import is_in_jax_tracing_scope


class IsInJaxTracingScopeTest(testing.TestCase):
    def test_not_in_tracing_scope(self):
        """Outside of jit/vmap, should return False."""
        result = is_in_jax_tracing_scope()
        self.assertFalse(result)

    def test_with_concrete_tensor(self):
        """Concrete tensor should not be a Tracer."""
        import numpy as np

        t = backend.convert_to_tensor(np.array(1.0, dtype=np.float32))
        result = is_in_jax_tracing_scope(t)
        self.assertFalse(result)


if __name__ == "__main__":
    testing.run_tests()
