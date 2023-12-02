import unittest

import pytest

from keras import backend
from keras import ops
from keras.backend.common.keras_tensor import KerasTensor


def single_arg_test_fn(x):
    return ops.concatenate([(x + 1) ** 2, x], axis=-1)


def three_args_2_kwarg_test_fn(x1, x2, x3=None):
    x1 = ops.max(x1, axis=1)
    x2 = ops.max(x2, axis=1)
    if x3 is not None:
        x1 += ops.max(x3, axis=1)
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

    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason="Backend does not support sparse tensors.",
    )
    def test_sparse_to_sparse(self):
        def single_arg_sparse_fn(x):
            y0 = ops.transpose(x, axes=(0, 2, 1))
            y1 = ops.squeeze(ops.expand_dims(x, axis=3), axis=3)
            y2 = ops.reshape(ops.reshape(x, (-1, 9)), (-1, 3, 3))
            return (y0, y1, y2)

        x = KerasTensor(shape=(None, 3, 3), sparse=True)
        ys = backend.compute_output_spec(single_arg_sparse_fn, x)
        for y in ys:
            self.assertEqual(y.shape, (None, 3, 3))
            self.assertTrue(y.sparse)

        def three_args_2_kwarg_sparse_fn(x1, x2, x3=None):
            y0 = ops.add(x1, x2)  # sparse, sparse
            y1 = ops.concatenate([x1, x2], axis=0)  # sparse, sparse
            y2 = ops.divide(x1, x3)  # sparse, dense
            y3 = ops.matmul(x1, x2)  # sparse, sparse
            y4 = ops.multiply(x1, x2)  # sparse, sparse
            y5 = ops.multiply(x1, x3)  # sparse, dense
            return (y0, y1, y2, y3, y4, y5)

        x1 = KerasTensor(shape=(None, 3, 3), sparse=True)
        x2 = KerasTensor(shape=(None, 3, 3), sparse=True)
        x3 = KerasTensor(shape=(None, 3, 3), sparse=False)
        ys = backend.compute_output_spec(
            three_args_2_kwarg_sparse_fn, x1, x2, x3=x3
        )
        for y in ys:
            self.assertEqual(y.shape, (None, 3, 3))
            self.assertTrue(y.sparse)

    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason="Backend does not support sparse tensors.",
    )
    def test_sparse_to_dense(self):
        def single_arg_dense_fn(x):
            y0 = ops.exp(x)
            return (y0,)

        x = KerasTensor(shape=(None, 3, 3), sparse=True)
        ys = backend.compute_output_spec(single_arg_dense_fn, x)
        for y in ys:
            self.assertEqual(y.shape, (None, 3, 3))
            self.assertFalse(y.sparse)

        def three_args_2_kwarg_dense_fn(x1, x2, x3=None):
            y0 = ops.add(x1, x3)  # sparse, dense
            y1 = ops.add(x3, x1)  # dense, sparse
            y2 = ops.concatenate([x1, x3], axis=0)  # sparse, dense
            y3 = ops.matmul(x1, x3)  # sparse, dense
            y4 = ops.matmul(x3, x1)  # dense, sparse
            indices = KerasTensor((3,), "int64", sparse=True)
            y5 = ops.take(x3, indices=indices, axis=1)  # dense, sparse
            y6 = ops.divide(x1, x2)
            return (y0, y1, y2, y3, y4, y5, y6)

        x1 = KerasTensor(shape=(None, 3, 3), sparse=True)
        x2 = KerasTensor(shape=(None, 3, 3), sparse=True)
        x3 = KerasTensor(shape=(None, 3, 3), sparse=False)
        ys = backend.compute_output_spec(
            three_args_2_kwarg_dense_fn, x1, x2, x3=x3
        )
        for y in ys:
            self.assertEqual(y.shape, (None, 3, 3))
            self.assertFalse(y.sparse)
