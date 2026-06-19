import keras
from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.types import Shape
from keras.src.types import Tensor


class TypesTest(testing.TestCase):
    def test_exposed_in_public_api(self):
        self.assertIs(keras.types.Tensor, Tensor)
        self.assertIs(keras.types.Shape, Shape)

    def test_tensor_matches_active_backend_runtime_type(self):
        # `Tensor` should be the concrete tensor class of the active backend
        # so that `isinstance(t, keras.types.Tensor)` works at runtime.
        t = ops.zeros((2, 3))
        self.assertIsInstance(t, Tensor)

    def test_tensor_is_a_class_on_every_backend(self):
        # All five backends resolve `Tensor` to a real class. Tests written
        # against `Tensor` can rely on `isinstance`.
        self.assertTrue(isinstance(Tensor, type))

    def test_tensor_resolution_per_backend(self):
        name = backend.backend()
        if name == "tensorflow":
            import tensorflow as tf

            self.assertIs(Tensor, tf.Tensor)
        elif name == "jax":
            import jax

            self.assertIs(Tensor, jax.Array)
        elif name == "torch":
            import torch

            self.assertIs(Tensor, torch.Tensor)
        elif name == "numpy":
            import numpy as np

            self.assertIs(Tensor, np.ndarray)
        elif name == "openvino":
            from keras.src.backend.openvino.core import OpenVINOKerasTensor

            self.assertIs(Tensor, OpenVINOKerasTensor)
