import unittest

from keras import backend
from keras.backend.common.keras_tensor import KerasTensor


def single_arg_test_fn(x):
    return backend.numpy.concatenate([(x + 1) ** 2, x], axis=-1)


def three_args_2_kwarg_test_fn(x1, x2, x3=None):
    x1 = backend.numpy.max(x1, axis=1)
    x2 = backend.numpy.max(x2, axis=1)
    if x3 is not None:
        x1 += backend.numpy.max(x3, axis=1)
    return x1 + x2


class ComputeOutputSpecTest(unittest.TestCase):
    def test_dynamic_batch_size(self):
        x = KerasTensor(shape=(None, 3, 5))
        y = backend.compute_output_spec(single_arg_test_fn, x)
        self.assertEqual(y.shape, (None, 3, 10))

        x1 = KerasTensor(shape=(None, 3, 5))
        x2 = KerasTensor(shape=(None, 3, 5))
        x3 = KerasTensor(shape=(None, 3, 5))
        y = backend.compute_output_spec(
            three_args_2_kwarg_test_fn, x1, x2, x3=x3
        )
        self.assertEqual(y.shape, (None, 5))

    def test_dynamic_everything(self):
        x = KerasTensor(shape=(2, None, 3))
        y = backend.compute_output_spec(single_arg_test_fn, x)
        self.assertEqual(y.shape, (2, None, 6))

        x1 = KerasTensor(shape=(None, None, 5))
        x2 = KerasTensor(shape=(None, None, 5))
        x3 = KerasTensor(shape=(None, None, 5))
        y = backend.compute_output_spec(
            three_args_2_kwarg_test_fn, x1, x2, x3=x3
        )
        self.assertEqual(y.shape, (None, 5))
