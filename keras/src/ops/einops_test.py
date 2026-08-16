from conftest import skip_if_backend
from keras.src import ops
from keras.src import testing
from keras.src.backend.common import keras_tensor
from keras.src.ops.einops import rearrange


class RearrangeTest(testing.TestCase):
    def test_basic_rearrangement_symbolic(self):
        x = keras_tensor.KerasTensor((2, 3, 4))
        y = rearrange(x, "b c h -> b h c")
        self.assertIsInstance(y, keras_tensor.KerasTensor)
        self.assertEqual(y.shape, (2, 4, 3))

    @skip_if_backend("openvino", "Test operation not supported by openvino")
    def test_basic_rearrangement(self):
        x = ops.random.uniform((2, 3, 4))
        y = rearrange(x, "b c h -> b h c")
        self.assertEqual(y.shape, (2, 4, 3))
        self.assertTrue(ops.all(ops.equal(y, ops.transpose(x, (0, 2, 1)))))

    @skip_if_backend("openvino", "Test operation not supported by openvino")
    def test_output_composition(self):
        x = ops.random.uniform((2, 4, 4, 3))
        y = rearrange(x, "b h w c -> (b h) w c")
        target_shape = (8, 4, 3)
        self.assertEqual(y.shape, target_shape)
        self.assertTrue(ops.all(ops.equal(y, ops.reshape(x, (8, 4, 3)))))

    def test_basic_decomposition_and_rearrangement_symbolic(self):
        x = keras_tensor.KerasTensor((6, 8))
        y = rearrange(x, "(h w) c -> h w c", h=2, w=3)
        self.assertIsInstance(y, keras_tensor.KerasTensor)
        self.assertEqual(y.shape, (2, 3, 8))

    def test_basic_decomposition_and_rearrangement(self):
        x = ops.random.uniform((6, 8))
        y = rearrange(x, "(h w) c -> h w c", h=2, w=3)
        self.assertEqual(y.shape, (2, 3, 8))

    @skip_if_backend("openvino", "Test operation not supported by openvino")
    def test_unchanged_shape(self):
        x = ops.ones([2, 3, 4])
        y = rearrange(x, "b h c -> b h c")
        self.assertTrue(ops.all(ops.equal(y, x)))
        self.assertTrue(x.shape, y.shape)

    def test_unchanged_shape_symbolic(self):
        x = keras_tensor.KerasTensor((2, 3, 4))
        y = rearrange(x, "b h c -> b h c")
        self.assertTrue(x.shape, y.shape)
