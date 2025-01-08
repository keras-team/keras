import numpy as np

from keras.src import testing
from keras.src.backend.common import keras_tensor
from keras.src.ops.einops import rearrange


class RearrangeTest(testing.TestCase):
    def test_basic_rearrangement(self):
        x = keras_tensor.KerasTensor((2, 3, 4))
        y = rearrange(x, "b c h -> b h c")
        self.assertIsInstance(y, keras_tensor.KerasTensor)
        self.assertEqual(y.shape, (2, 4, 3))

    def test_basic_decomposition_and_rearrangement(self):
        x = keras_tensor.KerasTensor((6, 8))
        y = rearrange(x, "(h w) c -> h w c", h=2, w=3)
        self.assertIsInstance(y, keras_tensor.KerasTensor)
        self.assertEqual(y.shape, (2, 3, 8))

    def test_static_shape(self):
        x = np.ones([2, 3, 4])
        y = rearrange(x, "b c h -> b h c")
        np.testing.assert_array_equal(y, np.transpose(x, (0, 2, 1)))
