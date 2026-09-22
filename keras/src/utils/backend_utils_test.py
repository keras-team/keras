import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.utils import backend_utils


class BackendUtilsTest(testing.TestCase):
    @parameterized.named_parameters(
        ("numpy", "numpy"),
        ("jax", "jax"),
        ("tensorflow", "tensorflow"),
        ("torch", "torch"),
    )
    def test_dynamic_backend(self, name):
        dynamic_backend = backend_utils.DynamicBackend()
        x = np.random.uniform(size=[1, 2, 3]).astype("float32")

        if name == "numpy":
            dynamic_backend.set_backend(name)
            y = dynamic_backend.ops.numpy.log10(x)
            self.assertIsInstance(y, np.ndarray)
        elif name == "jax":
            import jax

            dynamic_backend.set_backend(name)
            y = dynamic_backend.ops.numpy.log10(x)
            self.assertIsInstance(y, jax.Array)
        elif name == "tensorflow":
            import tensorflow as tf

            dynamic_backend.set_backend(name)
            y = dynamic_backend.ops.numpy.log10(x)
            self.assertIsInstance(y, tf.Tensor)
        elif name == "torch":
            import torch

            dynamic_backend.set_backend(name)
            y = dynamic_backend.ops.numpy.log10(x)
            self.assertIsInstance(y, torch.Tensor)

    @parameterized.named_parameters(
        ("numpy", "numpy"),
        ("jax", "jax"),
        ("tensorflow", "tensorflow"),
        ("torch", "torch"),
    )
    def test_dynamic_backend_op_attribute_fallback(self, name):
        # Ops live in `keras.src.backend.<backend>.ops` and are no longer
        # re-exported on the backend package itself. `DynamicBackend`
        # documents `backend.<op>` access, and subclasses of the public
        # preprocessing layers reach ops that way via `self.backend`, so it
        # has to keep resolving.
        dynamic_backend = backend_utils.DynamicBackend()
        dynamic_backend.set_backend(name)
        x = np.random.uniform(size=[1, 2, 3]).astype("float32")

        y = dynamic_backend.cast(x, "float16")
        self.assertEqual(backend.standardize_dtype(y.dtype), "float16")
        self.assertAllClose(dynamic_backend.numpy.log10(x), np.log10(x))
        self.assertAllClose(dynamic_backend.nn.relu(-x), np.zeros_like(x))

        # `numerical_utils.encode_categorical_inputs` branches on
        # `backend_module.__name__`, so dunders have to resolve too.
        self.assertEqual(dynamic_backend.__name__, f"keras.src.backend.{name}")

        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            dynamic_backend.not_an_op

    def test_dynamic_backend_invalid_name(self):
        dynamic_backend = backend_utils.DynamicBackend()
        with self.assertRaisesRegex(ValueError, "Available backends are"):
            dynamic_backend.set_backend("abc")
