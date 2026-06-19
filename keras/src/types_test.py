import keras
from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.types import _BACKEND_TENSOR_TYPE
from keras.src.types import Shape
from keras.src.types import Tensor


class TypesTest(testing.TestCase):
    def test_exposed_in_public_api(self):
        self.assertIs(keras.types.Tensor, Tensor)
        self.assertIs(keras.types.Shape, Shape)

    def test_tensor_isinstance_matches_active_backend(self):
        # Tensors produced by `keras.ops` on the active backend should
        # satisfy `isinstance(t, keras.types.Tensor)`.
        t = ops.zeros((2, 3))
        self.assertIsInstance(t, Tensor)

    def test_tensor_issubclass_matches_active_backend(self):
        self.assertTrue(issubclass(_BACKEND_TENSOR_TYPE, Tensor))

    def test_tensor_isinstance_rejects_non_tensor(self):
        self.assertNotIsInstance("not a tensor", Tensor)
        self.assertNotIsInstance(123, Tensor)

    def test_backend_tensor_type_resolution(self):
        name = backend.backend()
        if name == "tensorflow":
            import tensorflow as tf

            self.assertIs(_BACKEND_TENSOR_TYPE, tf.Tensor)
        elif name == "jax":
            import jax

            self.assertIs(_BACKEND_TENSOR_TYPE, jax.Array)
        elif name == "torch":
            import torch

            self.assertIs(_BACKEND_TENSOR_TYPE, torch.Tensor)
        elif name == "numpy":
            import numpy as np

            self.assertIs(_BACKEND_TENSOR_TYPE, np.ndarray)
        elif name == "openvino":
            from keras.src.backend.openvino.core import OpenVINOKerasTensor

            self.assertIs(_BACKEND_TENSOR_TYPE, OpenVINOKerasTensor)

    def test_tensor_cannot_be_instantiated(self):
        with self.assertRaisesRegex(TypeError, "type alias"):
            Tensor()

    def test_shape_cannot_be_instantiated(self):
        with self.assertRaisesRegex(TypeError, "type alias"):
            Shape()
