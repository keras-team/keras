import numpy as np
from absl.testing import parameterized

from keras.src import testing
from keras.src.utils import backend_utils


class BackendUtilsTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("numpy", "numpy"),
        ("jax", "jax"),
        ("tensorflow", "tensorflow"),
        ("torch", "torch"),
    )
    def test_dynamic_backend(self, name):
        backend = backend_utils.DynamicBackend()
        x = np.random.uniform(size=[1, 2, 3])

        if name == "numpy":
            backend.set_backend(name)
            y = backend.numpy.log10(x)
            self.assertIsInstance(y, np.ndarray)
        elif name == "jax":
            import jax

            backend.set_backend(name)
            y = backend.numpy.log10(x)
            self.assertIsInstance(y, jax.Array)
        elif name == "tensorflow":
            import tensorflow as tf

            backend.set_backend(name)
            y = backend.numpy.log10(x)
            self.assertIsInstance(y, tf.Tensor)
        elif name == "torch":
            import torch

            backend.set_backend(name)
            y = backend.numpy.log10(x)
            self.assertIsInstance(y, torch.Tensor)

    def test_dynamic_backend_invalid_name(self):
        backend = backend_utils.DynamicBackend()
        with self.assertRaisesRegex(ValueError, "Avaiable backends are"):
            backend.set_backend("abc")
