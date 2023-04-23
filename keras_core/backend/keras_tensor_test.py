import numpy as np
import tensorflow as tf
from jax import numpy as jnp

from keras_core import operations as ops
from keras_core import testing
from keras_core.backend import keras_tensor


class KerasTensorTest(testing.TestCase):
    def test_attributes(self):
        x = keras_tensor.KerasTensor(shape=(3,), dtype="float32")
        self.assertEqual(x.dtype, "float32")
        self.assertEqual(x.shape, (3,))

    def test_numpy_methods(self):
        x = keras_tensor.KerasTensor(shape=(3, 2), dtype="float32")

        # reshape
        x = x.reshape((6,))
        self.assertEqual(x.shape, (6,))

        # expand_dims, squeeze
        x = ops.expand_dims(x, -1)
        self.assertEqual(x.shape, (6, 1))
        x = x.squeeze()
        self.assertEqual(x.shape, (6,))
        x = ops.expand_dims(x, axis=0)
        self.assertEqual(x.shape, (1, 6))
        x = x.squeeze(axis=0)
        self.assertEqual(x.shape, (6,))

    def test_invalid_usage(self):
        x = keras_tensor.KerasTensor(shape=(3,), dtype="float32")
        with self.assertRaisesRegex(
            ValueError, "doesn't have any actual numerical value"
        ):
            np.array(x)

        with self.assertRaisesRegex(
            ValueError, "cannot be used as input to a JAX function"
        ):
            jnp.array(x)

        with self.assertRaisesRegex(
            ValueError, "cannot be used as input to a TensorFlow function"
        ):
            tf.convert_to_tensor(x)
