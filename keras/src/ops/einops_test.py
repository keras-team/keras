import numpy as np

from conftest import skip_if_backend
from keras.src import testing
from keras.src.backend.common import keras_tensor
from keras.src.ops.einops import rearrange


@skip_if_backend("openvino", "NumPy ops not supported with openvino backend")
class RearrangeTest(testing.TestCase):
    def test_basic_rearrangement_symbolic(self):
        x = keras_tensor.KerasTensor((2, 3, 4))
        y = rearrange(x, "b c h -> b h c")
        self.assertIsInstance(y, keras_tensor.KerasTensor)
        self.assertEqual(y.shape, (2, 4, 3))

    def test_basic_rearrangement(self):
        x = np.random.rand(2, 3, 4)
        y = rearrange(x, "b c h -> b h c")
        self.assertEqual(y.shape, (2, 4, 3))
        np.testing.assert_array_equal(y, x.transpose(0, 2, 1))

    def test_output_composition(self):
        x = np.random.rand(2, 4, 4, 3)
        y = rearrange(x, "b h w c -> (b h) w c")
        target_shape = (8, 4, 3)
        self.assertEqual(y.shape, target_shape)
        np.testing.assert_array_equal(y, x.reshape(8, 4, 3))

    def test_basic_decomposition_and_rearrangement_symbolic(self):
        x = keras_tensor.KerasTensor((6, 8))
        y = rearrange(x, "(h w) c -> h w c", h=2, w=3)
        self.assertIsInstance(y, keras_tensor.KerasTensor)
        self.assertEqual(y.shape, (2, 3, 8))

    def test_basic_decomposition_and_rearrangement(self):
        x = np.random.rand(6, 8)
        y = rearrange(x, "(h w) c -> h w c", h=2, w=3)
        self.assertEqual(y.shape, (2, 3, 8))

    def test_unchanged_shape(self):
        x = np.ones([2, 3, 4])
        y = rearrange(x, "b h c -> b h c")
        np.testing.assert_array_equal(y, x)
        self.assertTrue(x.shape, y.shape)

    def test_unchanged_shape_symbolic(self):
        x = keras_tensor.KerasTensor((2, 3, 4))
        y = rearrange(x, "b h c -> b h c")
        self.assertTrue(x.shape, y.shape)
