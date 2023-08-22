import numpy as np
from tensorflow.python.ops.numpy_ops import np_config

from keras_core import backend
from keras_core import testing
from keras_core.backend.common import standardize_dtype
from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.ops import numpy as knp

# TODO: remove reliance on this (or alternatively, turn it on by default).
np_config.enable_numpy_behavior()


class NumpyTwoInputOpsDynamicShapeTest(testing.TestCase):
    def test_add(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.add(x, y).shape, (2, 3))

    def test_subtract(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.subtract(x, y).shape, (2, 3))

    def test_multiply(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.multiply(x, y).shape, (2, 3))

    def test_matmul(self):
        x = KerasTensor([None, 3, 4])
        y = KerasTensor([3, None, 4, 5])
        self.assertEqual(knp.matmul(x, y).shape, (3, None, 3, 5))

    def test_power(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.power(x, y).shape, (2, 3))

    def test_divide(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.divide(x, y).shape, (2, 3))

    def test_true_divide(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.true_divide(x, y).shape, (2, 3))

    def test_append(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.append(x, y).shape, (None,))

    def test_arctan2(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.arctan2(x, y).shape, (2, 3))

    def test_cross(self):
        x1 = KerasTensor([2, 3, 3])
        x2 = KerasTensor([1, 3, 2])
        y = KerasTensor([None, 1, 2])
        self.assertEqual(knp.cross(x1, y).shape, (2, 3, 3))
        self.assertEqual(knp.cross(x2, y).shape, (None, 3))

    def test_einsum(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([3, 4])
        self.assertEqual(knp.einsum("ij,jk->ik", x, y).shape, (None, 4))
        self.assertEqual(knp.einsum("ij,jk->ikj", x, y).shape, (None, 4, 3))
        self.assertEqual(knp.einsum("ii", x).shape, ())
        self.assertEqual(knp.einsum(",ij", 5, x).shape, (None, 3))

        x = KerasTensor([None, 3, 4])
        y = KerasTensor([None, 4, 5])
        z = KerasTensor([1, 1, 1, 9])
        self.assertEqual(knp.einsum("ijk,jkl->li", x, y).shape, (5, None))
        self.assertEqual(knp.einsum("ijk,jkl->lij", x, y).shape, (5, None, 3))
        self.assertEqual(
            knp.einsum("...,...j->...j", x, y).shape, (None, 3, 4, 5)
        )
        self.assertEqual(
            knp.einsum("i...,...j->i...j", x, y).shape, (None, 3, 4, 5)
        )
        self.assertEqual(knp.einsum("i...,...j", x, y).shape, (3, 4, None, 5))
        self.assertEqual(
            knp.einsum("i...,...j,...k", x, y, z).shape, (1, 3, 4, None, 5, 9)
        )
        self.assertEqual(
            knp.einsum("mij,ijk,...", x, y, z).shape, (1, 1, 1, 9, 5, None)
        )

        with self.assertRaises(ValueError):
            x = KerasTensor([None, 3])
            y = KerasTensor([3, 4])
            knp.einsum("ijk,jk->ik", x, y)

    def test_full_like(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.full_like(x, KerasTensor([1, 3])).shape, (None, 3))

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.full_like(x, 2).shape, (None, 3, 3))

    def test_greater(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.greater(x, y).shape, (2, 3))

    def test_greater_equal(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.greater_equal(x, y).shape, (2, 3))

    def test_isclose(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.isclose(x, y).shape, (2, 3))

    def test_less(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.less(x, y).shape, (2, 3))

    def test_less_equal(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.less_equal(x, y).shape, (2, 3))

    def test_linspace(self):
        start = KerasTensor([None, 3, 4])
        stop = KerasTensor([2, 3, 4])
        self.assertEqual(
            knp.linspace(start, stop, 10, axis=1).shape, (2, 10, 3, 4)
        )

        start = KerasTensor([None, 3])
        stop = 2
        self.assertEqual(
            knp.linspace(start, stop, 10, axis=1).shape, (None, 10, 3)
        )

    def test_logical_and(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.logical_and(x, y).shape, (2, 3))

    def test_logical_or(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.logical_or(x, y).shape, (2, 3))

    def test_logspace(self):
        start = KerasTensor([None, 3, 4])
        stop = KerasTensor([2, 3, 4])
        self.assertEqual(
            knp.logspace(start, stop, 10, axis=1).shape, (2, 10, 3, 4)
        )

        start = KerasTensor([None, 3])
        stop = 2
        self.assertEqual(
            knp.logspace(start, stop, 10, axis=1).shape, (None, 10, 3)
        )

    def test_maximum(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.maximum(x, y).shape, (2, 3))

    def test_minimum(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.minimum(x, y).shape, (2, 3))

    def test_mod(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.mod(x, y).shape, (2, 3))

    def test_not_equal(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.not_equal(x, y).shape, (2, 3))

    def test_outer(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.outer(x, y).shape, (None, None))

    def test_take(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.take(x, 1).shape, ())
        self.assertEqual(knp.take(x, [1, 2]).shape, (2,))
        self.assertEqual(
            knp.take(x, [[1, 2], [1, 2]], axis=1).shape, (None, 2, 2)
        )

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.take(x, 1, axis=1).shape, (None, 3))
        self.assertEqual(knp.take(x, [1, 2]).shape, (2,))
        self.assertEqual(
            knp.take(x, [[1, 2], [1, 2]], axis=1).shape, (None, 2, 2, 3)
        )

        # test with negative axis
        self.assertEqual(knp.take(x, 1, axis=-2).shape, (None, 3))

        # test with multi-dimensional indices
        x = KerasTensor([None, 3, None, 5])
        indices = KerasTensor([6, 7])
        self.assertEqual(knp.take(x, indices, axis=2).shape, (None, 3, 6, 7, 5))

    def test_take_along_axis(self):
        x = KerasTensor([None, 3])
        indices = KerasTensor([1, 3])
        self.assertEqual(knp.take_along_axis(x, indices, axis=0).shape, (1, 3))
        self.assertEqual(
            knp.take_along_axis(x, indices, axis=1).shape, (None, 3)
        )

        x = KerasTensor([None, 3, 3])
        indices = KerasTensor([1, 3, None])
        self.assertEqual(
            knp.take_along_axis(x, indices, axis=1).shape, (None, 3, 3)
        )

    def test_tensordot(self):
        x = KerasTensor([None, 3, 4])
        y = KerasTensor([3, 4])
        self.assertEqual(knp.tensordot(x, y, axes=1).shape, (None, 3, 4))
        self.assertEqual(knp.tensordot(x, y, axes=[[0, 1], [1, 0]]).shape, (4,))

    def test_vdot(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([None, 3])
        self.assertEqual(knp.vdot(x, y).shape, ())

        x = KerasTensor([None, 3, 3])
        y = KerasTensor([None, 3, 3])
        self.assertEqual(knp.vdot(x, y).shape, ())

    def test_where(self):
        condition = KerasTensor([2, None, 1])
        x = KerasTensor([None, 1])
        y = KerasTensor([None, 3])
        self.assertEqual(knp.where(condition, x, y).shape, (2, None, 3))
        self.assertEqual(knp.where(condition).shape, (2, None, 1))

    def test_floordiv(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.floor_divide(x, y).shape, (2, 3))

    def test_xor(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.logical_xor(x, y).shape, (2, 3))

    def test_shape_equal_basic_equality(self):
        x = KerasTensor([3, 4]).shape
        y = KerasTensor([3, 4]).shape
        self.assertTrue(knp.shape_equal(x, y))
        y = KerasTensor([3, 5]).shape
        self.assertFalse(knp.shape_equal(x, y))

    def test_shape_equal_allow_none(self):
        x = KerasTensor([3, 4, None]).shape
        y = KerasTensor([3, 4, 5]).shape
        self.assertTrue(knp.shape_equal(x, y, allow_none=True))
        self.assertFalse(knp.shape_equal(x, y, allow_none=False))

    def test_shape_equal_different_shape_lengths(self):
        x = KerasTensor([3, 4]).shape
        y = KerasTensor([3, 4, 5]).shape
        self.assertFalse(knp.shape_equal(x, y))

    def test_shape_equal_ignore_axes(self):
        x = KerasTensor([3, 4, 5]).shape
        y = KerasTensor([3, 6, 5]).shape
        self.assertTrue(knp.shape_equal(x, y, axis=1))
        y = KerasTensor([3, 6, 7]).shape
        self.assertTrue(knp.shape_equal(x, y, axis=(1, 2)))
        self.assertFalse(knp.shape_equal(x, y, axis=1))


class NumpyTwoInputOpsStaticShapeTest(testing.TestCase):
    def test_add(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.add(x, y).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.add(x, y)

    def test_subtract(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.subtract(x, y).shape, (2, 3))

        x = KerasTensor([2, 3])
        self.assertEqual(knp.subtract(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.subtract(x, y)

    def test_multiply(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.multiply(x, y).shape, (2, 3))

        x = KerasTensor([2, 3])
        self.assertEqual(knp.multiply(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.multiply(x, y)

    def test_matmul(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([3, 2])
        self.assertEqual(knp.matmul(x, y).shape, (2, 2))

        with self.assertRaises(ValueError):
            x = KerasTensor([3, 4])
            y = KerasTensor([2, 3, 4])
            knp.matmul(x, y)

    def test_power(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.power(x, y).shape, (2, 3))

        x = KerasTensor([2, 3])
        self.assertEqual(knp.power(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.power(x, y)

    def test_divide(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.divide(x, y).shape, (2, 3))

        x = KerasTensor([2, 3])
        self.assertEqual(knp.divide(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.divide(x, y)

    def test_true_divide(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.true_divide(x, y).shape, (2, 3))

        x = KerasTensor([2, 3])
        self.assertEqual(knp.true_divide(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.true_divide(x, y)

    def test_append(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.append(x, y).shape, (12,))

        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.append(x, y, axis=0).shape, (4, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.append(x, y, axis=2)

    def test_arctan2(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.arctan2(x, y).shape, (2, 3))

        x = KerasTensor([2, 3])
        self.assertEqual(knp.arctan2(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.arctan2(x, y)

    def test_cross(self):
        x1 = KerasTensor([2, 3, 3])
        x2 = KerasTensor([1, 3, 2])
        y1 = KerasTensor([2, 3, 3])
        y2 = KerasTensor([2, 3, 2])
        self.assertEqual(knp.cross(x1, y1).shape, (2, 3, 3))
        self.assertEqual(knp.cross(x2, y2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.cross(x, y)

        with self.assertRaises(ValueError):
            x = KerasTensor([4, 3, 3])
            y = KerasTensor([2, 3, 3])
            knp.cross(x, y)

    def test_einsum(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([3, 4])
        self.assertEqual(knp.einsum("ij,jk->ik", x, y).shape, (2, 4))
        self.assertEqual(knp.einsum("ij,jk->ikj", x, y).shape, (2, 4, 3))
        self.assertEqual(knp.einsum("ii", x).shape, ())
        self.assertEqual(knp.einsum(",ij", 5, x).shape, (2, 3))

        x = KerasTensor([2, 3, 4])
        y = KerasTensor([3, 4, 5])
        z = KerasTensor([1, 1, 1, 9])
        self.assertEqual(knp.einsum("ijk,jkl->li", x, y).shape, (5, 2))
        self.assertEqual(knp.einsum("ijk,jkl->lij", x, y).shape, (5, 2, 3))
        self.assertEqual(knp.einsum("...,...j->...j", x, y).shape, (2, 3, 4, 5))
        self.assertEqual(
            knp.einsum("i...,...j->i...j", x, y).shape, (2, 3, 4, 5)
        )
        self.assertEqual(knp.einsum("i...,...j", x, y).shape, (3, 4, 2, 5))
        self.assertEqual(knp.einsum("i...,...j", x, y).shape, (3, 4, 2, 5))
        self.assertEqual(
            knp.einsum("i...,...j,...k", x, y, z).shape, (1, 3, 4, 2, 5, 9)
        )
        self.assertEqual(
            knp.einsum("mij,ijk,...", x, y, z).shape, (1, 1, 1, 9, 5, 2)
        )

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([3, 4])
            knp.einsum("ijk,jk->ik", x, y)

    def test_full_like(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.full_like(x, 2).shape, (2, 3))

    def test_greater(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.greater(x, y).shape, (2, 3))

        x = KerasTensor([2, 3])
        self.assertEqual(knp.greater(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.greater(x, y)

    def test_greater_equal(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.greater_equal(x, y).shape, (2, 3))

        x = KerasTensor([2, 3])
        self.assertEqual(knp.greater_equal(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.greater_equal(x, y)

    def test_isclose(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.isclose(x, y).shape, (2, 3))

        x = KerasTensor([2, 3])
        self.assertEqual(knp.isclose(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.isclose(x, y)

    def test_less(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.less(x, y).shape, (2, 3))

        x = KerasTensor([2, 3])
        self.assertEqual(knp.less(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.less(x, y)

    def test_less_equal(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.less_equal(x, y).shape, (2, 3))

        x = KerasTensor([2, 3])
        self.assertEqual(knp.less_equal(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.less_equal(x, y)

    def test_linspace(self):
        start = KerasTensor([2, 3, 4])
        stop = KerasTensor([2, 3, 4])
        self.assertEqual(knp.linspace(start, stop, 10).shape, (10, 2, 3, 4))

        with self.assertRaises(ValueError):
            start = KerasTensor([2, 3])
            stop = KerasTensor([2, 3, 4])
            knp.linspace(start, stop)

    def test_logical_and(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.logical_and(x, y).shape, (2, 3))

        x = KerasTensor([2, 3])
        self.assertEqual(knp.logical_and(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.logical_and(x, y)

    def test_logical_or(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.logical_or(x, y).shape, (2, 3))

        x = KerasTensor([2, 3])
        self.assertEqual(knp.logical_or(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.logical_or(x, y)

    def test_logspace(self):
        start = KerasTensor([2, 3, 4])
        stop = KerasTensor([2, 3, 4])
        self.assertEqual(knp.logspace(start, stop, 10).shape, (10, 2, 3, 4))

        with self.assertRaises(ValueError):
            start = KerasTensor([2, 3])
            stop = KerasTensor([2, 3, 4])
            knp.logspace(start, stop)

    def test_maximum(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.maximum(x, y).shape, (2, 3))

        x = KerasTensor([2, 3])
        self.assertEqual(knp.maximum(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.maximum(x, y)

    def test_minimum(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.minimum(x, y).shape, (2, 3))

        x = KerasTensor([2, 3])
        self.assertEqual(knp.minimum(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.minimum(x, y)

    def test_mod(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.mod(x, y).shape, (2, 3))

        x = KerasTensor([2, 3])
        self.assertEqual(knp.mod(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.mod(x, y)

    def test_not_equal(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.not_equal(x, y).shape, (2, 3))

        x = KerasTensor([2, 3])
        self.assertEqual(knp.not_equal(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.not_equal(x, y)

    def test_outer(self):
        x = KerasTensor([3])
        y = KerasTensor([4])
        self.assertEqual(knp.outer(x, y).shape, (3, 4))

        x = KerasTensor([2, 3])
        y = KerasTensor([4, 5])
        self.assertEqual(knp.outer(x, y).shape, (6, 20))

        x = KerasTensor([2, 3])
        self.assertEqual(knp.outer(x, 2).shape, (6, 1))

    def test_take(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.take(x, 1).shape, ())
        self.assertEqual(knp.take(x, [1, 2]).shape, (2,))
        self.assertEqual(knp.take(x, [[1, 2], [1, 2]], axis=1).shape, (2, 2, 2))

        # test with multi-dimensional indices
        x = KerasTensor([2, 3, 4, 5])
        indices = KerasTensor([6, 7])
        self.assertEqual(knp.take(x, indices, axis=2).shape, (2, 3, 6, 7, 5))

    def test_take_along_axis(self):
        x = KerasTensor([2, 3])
        indices = KerasTensor([1, 3])
        self.assertEqual(knp.take_along_axis(x, indices, axis=0).shape, (1, 3))
        self.assertEqual(knp.take_along_axis(x, indices, axis=1).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            indices = KerasTensor([1, 4])
            knp.take_along_axis(x, indices, axis=0)

    def test_tensordot(self):
        x = KerasTensor([2, 3, 3])
        y = KerasTensor([3, 3, 4])
        self.assertEqual(knp.tensordot(x, y, axes=1).shape, (2, 3, 3, 4))
        self.assertEqual(knp.tensordot(x, y, axes=2).shape, (2, 4))
        self.assertEqual(
            knp.tensordot(x, y, axes=[[1, 2], [0, 1]]).shape, (2, 4)
        )

    def test_vdot(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.vdot(x, y).shape, ())

    def test_where(self):
        condition = KerasTensor([2, 3])
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.where(condition, x, y).shape, (2, 3))
        self.assertAllEqual(knp.where(condition).shape, (2, 3))

    def test_floordiv(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.floor_divide(x, y).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.floor_divide(x, y)

    def test_xor(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.logical_xor(x, y).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.logical_xor(x, y)

    def test_digitize(self):
        x = KerasTensor((2, 3))
        bins = KerasTensor((3,))
        self.assertEqual(knp.digitize(x, bins).shape, (2, 3))
        self.assertTrue(knp.digitize(x, bins).dtype == "int32")

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            bins = KerasTensor([2, 3, 4])
            knp.digitize(x, bins)


class NumpyOneInputOpsDynamicShapeTest(testing.TestCase):
    def test_mean(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.mean(x).shape, ())

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.mean(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.mean(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_all(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.all(x).shape, ())

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.all(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.all(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_any(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.any(x).shape, ())

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.any(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.any(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_var(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.var(x).shape, ())

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.var(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.var(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_sum(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.sum(x).shape, ())

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.sum(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.sum(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_amax(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.amax(x).shape, ())

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.amax(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.amax(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_amin(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.amin(x).shape, ())

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.amin(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.amin(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_square(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.square(x).shape, (None, 3))

    def test_negative(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.negative(x).shape, (None, 3))

    def test_abs(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.abs(x).shape, (None, 3))

    def test_absolute(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.absolute(x).shape, (None, 3))

    def test_squeeze(self):
        x = KerasTensor([None, 1])
        self.assertEqual(knp.squeeze(x).shape, (None,))
        self.assertEqual(knp.squeeze(x, axis=1).shape, (None,))

        with self.assertRaises(ValueError):
            x = KerasTensor([None, 1])
            knp.squeeze(x, axis=0)

    def test_transpose(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.transpose(x).shape, (3, None))

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.transpose(x, (2, 0, 1)).shape, (3, None, 3))

    def test_arccos(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.arccos(x).shape, (None, 3))

    def test_arccosh(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.arccosh(x).shape, (None, 3))

    def test_arcsin(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.arcsin(x).shape, (None, 3))

    def test_arcsinh(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.arcsinh(x).shape, (None, 3))

    def test_arctan(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.arctan(x).shape, (None, 3))

    def test_arctanh(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.arctanh(x).shape, (None, 3))

    def test_argmax(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.argmax(x).shape, ())

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.argmax(x, axis=1).shape, (None, 3))

    def test_argmin(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.argmin(x).shape, ())

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.argmin(x, axis=1).shape, (None, 3))

    def test_argsort(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.argsort(x).shape, (None, 3))

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.argsort(x, axis=1).shape, (None, 3, 3))

    def test_array(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.array(x).shape, (None, 3))

    def test_average(self):
        x = KerasTensor([None, 3])
        weights = KerasTensor([None, 3])
        self.assertEqual(knp.average(x, weights=weights).shape, ())

        x = KerasTensor([None, 3])
        weights = KerasTensor([3])
        self.assertEqual(knp.average(x, axis=1, weights=weights).shape, (None,))

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.average(x, axis=1).shape, (None, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([None, 3, 3])
            weights = KerasTensor([None, 4])
            knp.average(x, weights=weights)

    def test_broadcast_to(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.broadcast_to(x, (2, 3, 3)).shape, (2, 3, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([3, 3])
            knp.broadcast_to(x, (2, 2, 3))

    def test_ceil(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.ceil(x).shape, (None, 3))

    def test_clip(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.clip(x, 1, 2).shape, (None, 3))

    def test_concatenate(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([None, 3])
        self.assertEqual(
            knp.concatenate(
                [x, y],
            ).shape,
            (None, 3),
        )
        self.assertEqual(knp.concatenate([x, y], axis=1).shape, (None, 6))

        with self.assertRaises(ValueError):
            self.assertEqual(knp.concatenate([x, y], axis=None).shape, (None,))

        with self.assertRaises(ValueError):
            x = KerasTensor([None, 3, 5])
            y = KerasTensor([None, 4, 6])
            knp.concatenate([x, y], axis=1)

    def test_conjugate(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.conjugate(x).shape, (None, 3))

    def test_conj(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.conj(x).shape, (None, 3))

    def test_copy(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.copy(x).shape, (None, 3))

    def test_cos(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.cos(x).shape, (None, 3))

    def test_cosh(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.cosh(x).shape, (None, 3))

    def test_count_nonzero(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.count_nonzero(x).shape, ())

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.count_nonzero(x, axis=1).shape, (None, 3))

    def test_cumprod(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.cumprod(x).shape, (None,))

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.cumprod(x, axis=1).shape, (None, 3, 3))

    def test_cumsum(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.cumsum(x).shape, (None,))

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.cumsum(x, axis=1).shape, (None, 3, 3))

    def test_diag(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.diag(x).shape, (None,))
        self.assertEqual(knp.diag(x, k=3).shape, (None,))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3, 4])
            knp.diag(x)

    def test_diagonal(self):
        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.diagonal(x).shape, (3, None))

    def test_dot(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([3, 2])
        z = KerasTensor([None, None, 2])
        self.assertEqual(knp.dot(x, y).shape, (None, 2))
        self.assertEqual(knp.dot(x, 2).shape, (None, 3))
        self.assertEqual(knp.dot(x, z).shape, (None, None, 2))

        x = KerasTensor([None])
        y = KerasTensor([5])
        self.assertEqual(knp.dot(x, y).shape, ())

    def test_exp(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.exp(x).shape, (None, 3))

    def test_expand_dims(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.expand_dims(x, -1).shape, (None, 3, 1))
        self.assertEqual(knp.expand_dims(x, 0).shape, (1, None, 3))
        self.assertEqual(knp.expand_dims(x, 1).shape, (None, 1, 3))
        self.assertEqual(knp.expand_dims(x, -2).shape, (None, 1, 3))

    def test_expm1(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.expm1(x).shape, (None, 3))

    def test_flip(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.flip(x).shape, (None, 3))

    def test_floor(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.floor(x).shape, (None, 3))

    def test_get_item(self):
        x = KerasTensor([None, 5, 16])
        # Simple slice.
        sliced = knp.get_item(x, 5)
        self.assertEqual(sliced.shape, (5, 16))
        # Ellipsis slice.
        sliced = knp.get_item(x, np.s_[..., -1])
        self.assertEqual(sliced.shape, (None, 5))
        # `newaxis` slice.
        sliced = knp.get_item(x, np.s_[:, np.newaxis, ...])
        self.assertEqual(sliced.shape, (None, 1, 5, 16))
        # Strided slice.
        sliced = knp.get_item(x, np.s_[:5, 3:, 3:12:2])
        self.assertEqual(sliced.shape, (None, 2, 5))
        # Error states.
        with self.assertRaises(ValueError):
            sliced = knp.get_item(x, np.s_[:, 17, :])
        with self.assertRaises(ValueError):
            sliced = knp.get_item(x, np.s_[..., 5, ...])
        with self.assertRaises(ValueError):
            sliced = knp.get_item(x, np.s_[:, :, :, :])

    def test_hstack(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([None, 3])
        self.assertEqual(knp.hstack([x, y]).shape, (None, 6))

        x = KerasTensor([None, 3])
        y = KerasTensor([None, None])
        self.assertEqual(knp.hstack([x, y]).shape, (None, None))

    def test_imag(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.imag(x).shape, (None, 3))

    def test_isfinite(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.isfinite(x).shape, (None, 3))

    def test_isinf(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.isinf(x).shape, (None, 3))

    def test_isnan(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.isnan(x).shape, (None, 3))

    def test_log(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.log(x).shape, (None, 3))

    def test_log10(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.log10(x).shape, (None, 3))

    def test_log1p(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.log1p(x).shape, (None, 3))

    def test_log2(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.log2(x).shape, (None, 3))

    def test_logaddexp(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.logaddexp(x, x).shape, (None, 3))

    def test_logical_not(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.logical_not(x).shape, (None, 3))

    def test_max(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.max(x).shape, ())

    def test_meshgrid(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([None, 3])
        self.assertEqual(knp.meshgrid(x, y)[0].shape, (None, None))
        self.assertEqual(knp.meshgrid(x, y)[1].shape, (None, None))

        with self.assertRaises(ValueError):
            knp.meshgrid(x, y, indexing="kk")

    def test_moveaxis(self):
        x = KerasTensor([None, 3, 4, 5])
        self.assertEqual(knp.moveaxis(x, 0, -1).shape, (3, 4, 5, None))
        self.assertEqual(knp.moveaxis(x, -1, 0).shape, (5, None, 3, 4))
        self.assertEqual(
            knp.moveaxis(x, [0, 1], [-1, -2]).shape, (4, 5, 3, None)
        )
        self.assertEqual(knp.moveaxis(x, [0, 1], [1, 0]).shape, (3, None, 4, 5))
        self.assertEqual(
            knp.moveaxis(x, [0, 1], [-2, -1]).shape, (4, 5, None, 3)
        )

    def test_ndim(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.ndim(x).shape, (2,))

    def test_ones_like(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.ones_like(x).shape, (None, 3))
        self.assertEqual(knp.ones_like(x).dtype, x.dtype)

    def test_zeros_like(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.zeros_like(x).shape, (None, 3))
        self.assertEqual(knp.zeros_like(x).dtype, x.dtype)

    def test_pad(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.pad(x, 1).shape, (None, 5))
        self.assertEqual(knp.pad(x, (1, 2)).shape, (None, 6))
        self.assertEqual(knp.pad(x, ((1, 2), (3, 4))).shape, (None, 10))

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.pad(x, 1).shape, (None, 5, 5))
        self.assertEqual(knp.pad(x, (1, 2)).shape, (None, 6, 6))
        self.assertEqual(
            knp.pad(x, ((1, 2), (3, 4), (5, 6))).shape, (None, 10, 14)
        )

    def test_prod(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.prod(x).shape, ())
        self.assertEqual(knp.prod(x, axis=0).shape, (3,))
        self.assertEqual(knp.prod(x, axis=1, keepdims=True).shape, (None, 1))

    def test_ravel(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.ravel(x).shape, (None,))

    def test_real(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.real(x).shape, (None, 3))

    def test_reciprocal(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.reciprocal(x).shape, (None, 3))

    def test_repeat(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.repeat(x, 2).shape, (None,))
        self.assertEqual(knp.repeat(x, 3, axis=1).shape, (None, 9))
        self.assertEqual(knp.repeat(x, [1, 2], axis=0).shape, (3, 3))
        self.assertEqual(knp.repeat(x, 2, axis=0).shape, (None, 3))

    def test_reshape(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.reshape(x, (3, 2)).shape, (3, 2))
        self.assertEqual(knp.reshape(x, (3, -1)).shape, (3, None))

    def test_roll(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.roll(x, 1).shape, (None, 3))
        self.assertEqual(knp.roll(x, 1, axis=1).shape, (None, 3))
        self.assertEqual(knp.roll(x, 1, axis=0).shape, (None, 3))

    def test_round(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.round(x).shape, (None, 3))

    def test_sign(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.sign(x).shape, (None, 3))

    def test_sin(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.sin(x).shape, (None, 3))

    def test_sinh(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.sinh(x).shape, (None, 3))

    def test_size(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.size(x).shape, ())

    def test_sort(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.sort(x).shape, (None, 3))
        self.assertEqual(knp.sort(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.sort(x, axis=0).shape, (None, 3))

    def test_split(self):
        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.split(x, 2)[0].shape, (None, 3, 3))
        self.assertEqual(knp.split(x, 3, axis=1)[0].shape, (None, 1, 3))
        self.assertEqual(knp.split(x, [1, 3], axis=1)[1].shape, (None, 2, 3))

    def test_sqrt(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.sqrt(x).shape, (None, 3))

    def test_stack(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([None, 3])
        self.assertEqual(knp.stack([x, y]).shape, (2, None, 3))
        self.assertEqual(knp.stack([x, y], axis=-1).shape, (None, 3, 2))

    def test_std(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.std(x).shape, ())

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.std(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.std(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_swapaxes(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.swapaxes(x, 0, 1).shape, (3, None))

    def test_tan(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.tan(x).shape, (None, 3))

    def test_tanh(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.tanh(x).shape, (None, 3))

    def test_tile(self):
        x = KerasTensor([None, 3])
        self.assertEqual(knp.tile(x, [2]).shape, (None, 6))
        self.assertEqual(knp.tile(x, [1, 2]).shape, (None, 6))
        self.assertEqual(knp.tile(x, [2, 1, 2]).shape, (2, None, 6))

    def test_trace(self):
        x = KerasTensor([None, 3, None, 5])
        self.assertEqual(knp.trace(x).shape, (None, 5))
        self.assertEqual(knp.trace(x, axis1=2, axis2=3).shape, (None, 3))

    def test_tril(self):
        x = KerasTensor([None, 3, None, 5])
        self.assertEqual(knp.tril(x).shape, (None, 3, None, 5))
        self.assertEqual(knp.tril(x, k=1).shape, (None, 3, None, 5))
        self.assertEqual(knp.tril(x, k=-1).shape, (None, 3, None, 5))

    def test_triu(self):
        x = KerasTensor([None, 3, None, 5])
        self.assertEqual(knp.triu(x).shape, (None, 3, None, 5))
        self.assertEqual(knp.triu(x, k=1).shape, (None, 3, None, 5))
        self.assertEqual(knp.triu(x, k=-1).shape, (None, 3, None, 5))

    def test_vstack(self):
        x = KerasTensor([None, 3])
        y = KerasTensor([None, 3])
        self.assertEqual(knp.vstack([x, y]).shape, (None, 3))

        x = KerasTensor([None, 3])
        y = KerasTensor([None, None])
        self.assertEqual(knp.vstack([x, y]).shape, (None, 3))


class NumpyOneInputOpsStaticShapeTest(testing.TestCase):
    def test_mean(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.mean(x).shape, ())

    def test_all(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.all(x).shape, ())

    def test_any(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.any(x).shape, ())

    def test_var(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.var(x).shape, ())

    def test_sum(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.sum(x).shape, ())

    def test_amax(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.amax(x).shape, ())

    def test_amin(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.amin(x).shape, ())

    def test_square(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.square(x).shape, (2, 3))

    def test_negative(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.negative(x).shape, (2, 3))

    def test_abs(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.abs(x).shape, (2, 3))

    def test_absolute(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.absolute(x).shape, (2, 3))

    def test_squeeze(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.squeeze(x).shape, (2, 3))

    def test_transpose(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.transpose(x).shape, (3, 2))

    def test_arccos(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.arccos(x).shape, (2, 3))

    def test_arccosh(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.arccosh(x).shape, (2, 3))

    def test_arcsin(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.arcsin(x).shape, (2, 3))

    def test_arcsinh(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.arcsinh(x).shape, (2, 3))

    def test_arctan(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.arctan(x).shape, (2, 3))

    def test_arctanh(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.arctanh(x).shape, (2, 3))

    def test_argmax(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.argmax(x).shape, ())

    def test_argmin(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.argmin(x).shape, ())

    def test_argsort(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.argsort(x).shape, (2, 3))
        self.assertEqual(knp.argsort(x, axis=None).shape, (6,))

    def test_array(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.array(x).shape, (2, 3))

    def test_average(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.average(x).shape, ())

    def test_broadcast_to(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.broadcast_to(x, (2, 2, 3)).shape, (2, 2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([3, 3])
            knp.broadcast_to(x, (2, 2, 3))

    def test_ceil(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.ceil(x).shape, (2, 3))

    def test_clip(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.clip(x, 1, 2).shape, (2, 3))

    def test_concatenate(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.concatenate([x, y]).shape, (4, 3))
        self.assertEqual(knp.concatenate([x, y], axis=1).shape, (2, 6))

        with self.assertRaises(ValueError):
            self.assertEqual(knp.concatenate([x, y], axis=None).shape, (None,))

    def test_conjugate(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.conjugate(x).shape, (2, 3))

    def test_conj(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.conj(x).shape, (2, 3))

    def test_copy(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.copy(x).shape, (2, 3))

    def test_cos(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.cos(x).shape, (2, 3))

    def test_cosh(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.cosh(x).shape, (2, 3))

    def test_count_nonzero(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.count_nonzero(x).shape, ())

    def test_cumprod(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.cumprod(x).shape, (6,))

    def test_cumsum(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.cumsum(x).shape, (6,))

    def test_diag(self):
        x = KerasTensor([3])
        self.assertEqual(knp.diag(x).shape, (3, 3))
        self.assertEqual(knp.diag(x, k=3).shape, (6, 6))
        self.assertEqual(knp.diag(x, k=-2).shape, (5, 5))

        x = KerasTensor([3, 5])
        self.assertEqual(knp.diag(x).shape, (3,))
        self.assertEqual(knp.diag(x, k=3).shape, (2,))
        self.assertEqual(knp.diag(x, k=-2).shape, (1,))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3, 4])
            knp.diag(x)

    def test_diagonal(self):
        x = KerasTensor([3, 3])
        self.assertEqual(knp.diagonal(x).shape, (3,))
        self.assertEqual(knp.diagonal(x, offset=1).shape, (2,))

        x = KerasTensor([3, 5, 5])
        self.assertEqual(knp.diagonal(x).shape, (5, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([3])
            knp.diagonal(x)

    def test_dot(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([3, 2])
        z = KerasTensor([4, 3, 2])
        self.assertEqual(knp.dot(x, y).shape, (2, 2))
        self.assertEqual(knp.dot(x, 2).shape, (2, 3))
        self.assertEqual(knp.dot(x, z).shape, (2, 4, 2))

        x = KerasTensor([5])
        y = KerasTensor([5])
        self.assertEqual(knp.dot(x, y).shape, ())

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3])
            knp.dot(x, y)

    def test_exp(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.exp(x).shape, (2, 3))

    def test_expand_dims(self):
        x = KerasTensor([2, 3, 4])
        self.assertEqual(knp.expand_dims(x, 0).shape, (1, 2, 3, 4))
        self.assertEqual(knp.expand_dims(x, 1).shape, (2, 1, 3, 4))
        self.assertEqual(knp.expand_dims(x, -2).shape, (2, 3, 1, 4))

    def test_expm1(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.expm1(x).shape, (2, 3))

    def test_flip(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.flip(x).shape, (2, 3))

    def test_floor(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.floor(x).shape, (2, 3))

    def test_get_item(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.get_item(x, 1).shape, (3,))

        x = KerasTensor([5, 3, 2])
        self.assertEqual(knp.get_item(x, 3).shape, (3, 2))

        x = KerasTensor(
            [
                2,
            ]
        )
        self.assertEqual(knp.get_item(x, 0).shape, ())

    def test_hstack(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.hstack([x, y]).shape, (2, 6))

    def test_imag(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.imag(x).shape, (2, 3))

    def test_isfinite(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.isfinite(x).shape, (2, 3))

    def test_isinf(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.isinf(x).shape, (2, 3))

    def test_isnan(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.isnan(x).shape, (2, 3))

    def test_log(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.log(x).shape, (2, 3))

    def test_log10(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.log10(x).shape, (2, 3))

    def test_log1p(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.log1p(x).shape, (2, 3))

    def test_log2(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.log2(x).shape, (2, 3))

    def test_logaddexp(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.logaddexp(x, x).shape, (2, 3))

    def test_logical_not(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.logical_not(x).shape, (2, 3))

    def test_max(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.max(x).shape, ())

    def test_meshgrid(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3, 4])
        z = KerasTensor([2, 3, 4, 5])
        self.assertEqual(knp.meshgrid(x, y)[0].shape, (24, 6))
        self.assertEqual(knp.meshgrid(x, y)[1].shape, (24, 6))
        self.assertEqual(knp.meshgrid(x, y, indexing="ij")[0].shape, (6, 24))
        self.assertEqual(
            knp.meshgrid(x, y, z, indexing="ij")[0].shape, (6, 24, 120)
        )
        with self.assertRaises(ValueError):
            knp.meshgrid(x, y, indexing="kk")

    def test_moveaxis(self):
        x = KerasTensor([2, 3, 4, 5])
        self.assertEqual(knp.moveaxis(x, 0, -1).shape, (3, 4, 5, 2))
        self.assertEqual(knp.moveaxis(x, -1, 0).shape, (5, 2, 3, 4))
        self.assertEqual(knp.moveaxis(x, [0, 1], [-1, -2]).shape, (4, 5, 3, 2))
        self.assertEqual(knp.moveaxis(x, [0, 1], [1, 0]).shape, (3, 2, 4, 5))
        self.assertEqual(knp.moveaxis(x, [0, 1], [-2, -1]).shape, (4, 5, 2, 3))

    def test_ndim(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.ndim(x).shape, (2,))

    def test_ones_like(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.ones_like(x).shape, (2, 3))
        self.assertEqual(knp.ones_like(x).dtype, x.dtype)

    def test_zeros_like(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.zeros_like(x).shape, (2, 3))
        self.assertEqual(knp.zeros_like(x).dtype, x.dtype)

    def test_pad(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.pad(x, 1).shape, (4, 5))
        self.assertEqual(knp.pad(x, (1, 2)).shape, (5, 6))
        self.assertEqual(knp.pad(x, ((1, 2), (3, 4))).shape, (5, 10))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            knp.pad(x, ((1, 2), (3, 4), (5, 6)))

    def test_prod(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.prod(x).shape, ())
        self.assertEqual(knp.prod(x, axis=0).shape, (3,))
        self.assertEqual(knp.prod(x, axis=1).shape, (2,))

    def test_ravel(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.ravel(x).shape, (6,))

    def test_real(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.real(x).shape, (2, 3))

    def test_reciprocal(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.reciprocal(x).shape, (2, 3))

    def test_repeat(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.repeat(x, 2).shape, (12,))
        self.assertEqual(knp.repeat(x, 3, axis=1).shape, (2, 9))
        self.assertEqual(knp.repeat(x, [1, 2], axis=0).shape, (3, 3))

    def test_reshape(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.reshape(x, (3, 2)).shape, (3, 2))
        self.assertEqual(knp.reshape(x, (3, -1)).shape, (3, 2))
        self.assertEqual(knp.reshape(x, (6,)).shape, (6,))
        self.assertEqual(knp.reshape(x, (-1,)).shape, (6,))

    def test_roll(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.roll(x, 1).shape, (2, 3))
        self.assertEqual(knp.roll(x, 1, axis=1).shape, (2, 3))
        self.assertEqual(knp.roll(x, 1, axis=0).shape, (2, 3))

    def test_round(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.round(x).shape, (2, 3))

    def test_sign(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.sign(x).shape, (2, 3))

    def test_sin(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.sin(x).shape, (2, 3))

    def test_sinh(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.sinh(x).shape, (2, 3))

    def test_size(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.size(x).shape, ())

    def test_sort(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.sort(x).shape, (2, 3))
        self.assertEqual(knp.sort(x, axis=1).shape, (2, 3))
        self.assertEqual(knp.sort(x, axis=0).shape, (2, 3))

    def test_split(self):
        x = KerasTensor([2, 3])
        self.assertEqual(len(knp.split(x, 2)), 2)
        self.assertEqual(knp.split(x, 2)[0].shape, (1, 3))
        self.assertEqual(knp.split(x, 3, axis=1)[0].shape, (2, 1))
        self.assertEqual(knp.split(x, [1, 3], axis=1)[1].shape, (2, 2))

        with self.assertRaises(ValueError):
            knp.split(x, 2, axis=1)

    def test_sqrt(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.sqrt(x).shape, (2, 3))

    def test_stack(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.stack([x, y]).shape, (2, 2, 3))
        self.assertEqual(knp.stack([x, y], axis=-1).shape, (2, 3, 2))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([3, 3])
            knp.stack([x, y])

    def test_std(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.std(x).shape, ())

    def test_swapaxes(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.swapaxes(x, 0, 1).shape, (3, 2))

    def test_tan(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.tan(x).shape, (2, 3))

    def test_tanh(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.tanh(x).shape, (2, 3))

    def test_tile(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.tile(x, [2]).shape, (2, 6))
        self.assertEqual(knp.tile(x, [1, 2]).shape, (2, 6))
        self.assertEqual(knp.tile(x, [2, 1, 2]).shape, (2, 2, 6))

    def test_trace(self):
        x = KerasTensor([2, 3, 4, 5])
        self.assertEqual(knp.trace(x).shape, (4, 5))
        self.assertEqual(knp.trace(x, axis1=2, axis2=3).shape, (2, 3))

    def test_tril(self):
        x = KerasTensor([2, 3, 4, 5])
        self.assertEqual(knp.tril(x).shape, (2, 3, 4, 5))
        self.assertEqual(knp.tril(x, k=1).shape, (2, 3, 4, 5))
        self.assertEqual(knp.tril(x, k=-1).shape, (2, 3, 4, 5))

    def test_triu(self):
        x = KerasTensor([2, 3, 4, 5])
        self.assertEqual(knp.triu(x).shape, (2, 3, 4, 5))
        self.assertEqual(knp.triu(x, k=1).shape, (2, 3, 4, 5))
        self.assertEqual(knp.triu(x, k=-1).shape, (2, 3, 4, 5))

    def test_vstack(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.vstack([x, y]).shape, (4, 3))


class NumpyTwoInputOpsCorretnessTest(testing.TestCase):
    def test_add(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        z = np.array([[[1, 2, 3], [3, 2, 1]]])
        self.assertAllClose(knp.add(x, y), np.add(x, y))
        self.assertAllClose(knp.add(x, z), np.add(x, z))

        self.assertAllClose(knp.Add()(x, y), np.add(x, y))
        self.assertAllClose(knp.Add()(x, z), np.add(x, z))

    def test_subtract(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        z = np.array([[[1, 2, 3], [3, 2, 1]]])
        self.assertAllClose(knp.subtract(x, y), np.subtract(x, y))
        self.assertAllClose(knp.subtract(x, z), np.subtract(x, z))

        self.assertAllClose(knp.Subtract()(x, y), np.subtract(x, y))
        self.assertAllClose(knp.Subtract()(x, z), np.subtract(x, z))

    def test_multiply(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        z = np.array([[[1, 2, 3], [3, 2, 1]]])
        self.assertAllClose(knp.multiply(x, y), np.multiply(x, y))
        self.assertAllClose(knp.multiply(x, z), np.multiply(x, z))

        self.assertAllClose(knp.Multiply()(x, y), np.multiply(x, y))
        self.assertAllClose(knp.Multiply()(x, z), np.multiply(x, z))

    def test_matmul(self):
        x = np.ones([2, 3, 4, 5])
        y = np.ones([2, 3, 5, 6])
        z = np.ones([5, 6])
        self.assertAllClose(knp.matmul(x, y), np.matmul(x, y))
        self.assertAllClose(knp.matmul(x, z), np.matmul(x, z))

        self.assertAllClose(knp.Matmul()(x, y), np.matmul(x, y))
        self.assertAllClose(knp.Matmul()(x, z), np.matmul(x, z))

    def test_power(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        z = np.array([[[1, 2, 3], [3, 2, 1]]])
        self.assertAllClose(knp.power(x, y), np.power(x, y))
        self.assertAllClose(knp.power(x, z), np.power(x, z))

        self.assertAllClose(knp.Power()(x, y), np.power(x, y))
        self.assertAllClose(knp.Power()(x, z), np.power(x, z))

    def test_divide(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        z = np.array([[[1, 2, 3], [3, 2, 1]]])
        self.assertAllClose(knp.divide(x, y), np.divide(x, y))
        self.assertAllClose(knp.divide(x, z), np.divide(x, z))

        self.assertAllClose(knp.Divide()(x, y), np.divide(x, y))
        self.assertAllClose(knp.Divide()(x, z), np.divide(x, z))

    def test_true_divide(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        z = np.array([[[1, 2, 3], [3, 2, 1]]])
        self.assertAllClose(knp.true_divide(x, y), np.true_divide(x, y))
        self.assertAllClose(knp.true_divide(x, z), np.true_divide(x, z))

        self.assertAllClose(knp.TrueDivide()(x, y), np.true_divide(x, y))
        self.assertAllClose(knp.TrueDivide()(x, z), np.true_divide(x, z))

    def test_append(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        z = np.array([[[1, 2, 3], [3, 2, 1]], [[4, 5, 6], [3, 2, 1]]])
        self.assertAllClose(knp.append(x, y), np.append(x, y))
        self.assertAllClose(knp.append(x, y, axis=1), np.append(x, y, axis=1))
        self.assertAllClose(knp.append(x, z), np.append(x, z))

        self.assertAllClose(knp.Append()(x, y), np.append(x, y))
        self.assertAllClose(knp.Append(axis=1)(x, y), np.append(x, y, axis=1))
        self.assertAllClose(knp.Append()(x, z), np.append(x, z))

    def test_arctan2(self):
        x = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        y = np.array([[4.0, 5.0, 6.0], [3.0, 2.0, 1.0]])
        self.assertAllClose(knp.arctan2(x, y), np.arctan2(x, y))

        self.assertAllClose(knp.Arctan2()(x, y), np.arctan2(x, y))

    def test_cross(self):
        x1 = np.ones([2, 1, 4, 3])
        x2 = np.ones([2, 1, 4, 2])
        y1 = np.ones([2, 1, 4, 3])
        y2 = np.ones([1, 5, 4, 3])
        y3 = np.ones([1, 5, 4, 2])
        self.assertAllClose(knp.cross(x1, y1), np.cross(x1, y1))
        self.assertAllClose(knp.cross(x1, y2), np.cross(x1, y2))
        if backend.backend() != "torch":
            # API divergence between `torch.cross` and `np.cross`
            # `torch.cross` only allows dim 3, `np.cross` allows dim 2 or 3
            self.assertAllClose(knp.cross(x1, y3), np.cross(x1, y3))
            self.assertAllClose(knp.cross(x2, y3), np.cross(x2, y3))

        self.assertAllClose(knp.Cross()(x1, y1), np.cross(x1, y1))
        self.assertAllClose(knp.Cross()(x1, y2), np.cross(x1, y2))
        if backend.backend() != "torch":
            # API divergence between `torch.cross` and `np.cross`
            # `torch.cross` only allows dim 3, `np.cross` allows dim 2 or 3
            self.assertAllClose(knp.Cross()(x1, y3), np.cross(x1, y3))
            self.assertAllClose(knp.Cross()(x2, y3), np.cross(x2, y3))

    def test_einsum(self):
        x = np.arange(24).reshape([2, 3, 4]).astype("float32")
        y = np.arange(24).reshape([2, 4, 3]).astype("float32")
        self.assertAllClose(
            knp.einsum("ijk,lkj->il", x, y),
            np.einsum("ijk,lkj->il", x, y),
        )
        self.assertAllClose(
            knp.einsum("ijk,ikj->i", x, y),
            np.einsum("ijk,ikj->i", x, y),
        )
        self.assertAllClose(
            knp.einsum("i...,j...k->...ijk", x, y),
            np.einsum("i..., j...k->...ijk", x, y),
        )
        self.assertAllClose(knp.einsum(",ijk", 5, y), np.einsum(",ijk", 5, y))

        self.assertAllClose(
            knp.Einsum("ijk,lkj->il")(x, y),
            np.einsum("ijk,lkj->il", x, y),
        )
        self.assertAllClose(
            knp.Einsum("ijk,ikj->i")(x, y),
            np.einsum("ijk,ikj->i", x, y),
        )
        self.assertAllClose(
            knp.Einsum("i...,j...k->...ijk")(x, y),
            np.einsum("i...,j...k->...ijk", x, y),
        )
        self.assertAllClose(knp.Einsum(",ijk")(5, y), np.einsum(",ijk", 5, y))

    def test_full_like(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.full_like(x, 2), np.full_like(x, 2))
        self.assertAllClose(
            knp.full_like(x, 2, dtype="float32"),
            np.full_like(x, 2, dtype="float32"),
        )
        self.assertAllClose(
            knp.full_like(x, np.ones([2, 3])),
            np.full_like(x, np.ones([2, 3])),
        )

        self.assertAllClose(knp.FullLike()(x, 2), np.full_like(x, 2))
        self.assertAllClose(
            knp.FullLike()(x, 2, dtype="float32"),
            np.full_like(x, 2, dtype="float32"),
        )
        self.assertAllClose(
            knp.FullLike()(x, np.ones([2, 3])),
            np.full_like(x, np.ones([2, 3])),
        )

    def test_greater(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        self.assertAllClose(knp.greater(x, y), np.greater(x, y))
        self.assertAllClose(knp.greater(x, 2), np.greater(x, 2))
        self.assertAllClose(knp.greater(2, x), np.greater(2, x))

        self.assertAllClose(knp.Greater()(x, y), np.greater(x, y))
        self.assertAllClose(knp.Greater()(x, 2), np.greater(x, 2))
        self.assertAllClose(knp.Greater()(2, x), np.greater(2, x))

    def test_greater_equal(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        self.assertAllClose(
            knp.greater_equal(x, y),
            np.greater_equal(x, y),
        )
        self.assertAllClose(
            knp.greater_equal(x, 2),
            np.greater_equal(x, 2),
        )
        self.assertAllClose(
            knp.greater_equal(2, x),
            np.greater_equal(2, x),
        )

        self.assertAllClose(
            knp.GreaterEqual()(x, y),
            np.greater_equal(x, y),
        )
        self.assertAllClose(
            knp.GreaterEqual()(x, 2),
            np.greater_equal(x, 2),
        )
        self.assertAllClose(
            knp.GreaterEqual()(2, x),
            np.greater_equal(2, x),
        )

    def test_isclose(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        self.assertAllClose(knp.isclose(x, y), np.isclose(x, y))
        self.assertAllClose(knp.isclose(x, 2), np.isclose(x, 2))
        self.assertAllClose(knp.isclose(2, x), np.isclose(2, x))

        self.assertAllClose(knp.Isclose()(x, y), np.isclose(x, y))
        self.assertAllClose(knp.Isclose()(x, 2), np.isclose(x, 2))
        self.assertAllClose(knp.Isclose()(2, x), np.isclose(2, x))

    def test_less(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        self.assertAllClose(knp.less(x, y), np.less(x, y))
        self.assertAllClose(knp.less(x, 2), np.less(x, 2))
        self.assertAllClose(knp.less(2, x), np.less(2, x))

        self.assertAllClose(knp.Less()(x, y), np.less(x, y))
        self.assertAllClose(knp.Less()(x, 2), np.less(x, 2))
        self.assertAllClose(knp.Less()(2, x), np.less(2, x))

    def test_less_equal(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        self.assertAllClose(knp.less_equal(x, y), np.less_equal(x, y))
        self.assertAllClose(knp.less_equal(x, 2), np.less_equal(x, 2))
        self.assertAllClose(knp.less_equal(2, x), np.less_equal(2, x))

        self.assertAllClose(knp.LessEqual()(x, y), np.less_equal(x, y))
        self.assertAllClose(knp.LessEqual()(x, 2), np.less_equal(x, 2))
        self.assertAllClose(knp.LessEqual()(2, x), np.less_equal(2, x))

    def test_linspace(self):
        self.assertAllClose(knp.linspace(0, 10, 5), np.linspace(0, 10, 5))
        self.assertAllClose(
            knp.linspace(0, 10, 5, endpoint=False),
            np.linspace(0, 10, 5, endpoint=False),
        )
        self.assertAllClose(knp.Linspace(num=5)(0, 10), np.linspace(0, 10, 5))
        self.assertAllClose(
            knp.Linspace(num=5, endpoint=False)(0, 10),
            np.linspace(0, 10, 5, endpoint=False),
        )

        start = np.zeros([2, 3, 4])
        stop = np.ones([2, 3, 4])
        self.assertAllClose(
            knp.linspace(start, stop, 5, retstep=True)[0],
            np.linspace(start, stop, 5, retstep=True)[0],
        )
        self.assertAllClose(
            backend.convert_to_numpy(
                knp.linspace(start, stop, 5, endpoint=False, retstep=True)[0]
            ),
            np.linspace(start, stop, 5, endpoint=False, retstep=True)[0],
        )
        self.assertAllClose(
            backend.convert_to_numpy(
                knp.linspace(
                    start, stop, 5, endpoint=False, retstep=True, dtype="int32"
                )[0]
            ),
            np.linspace(
                start, stop, 5, endpoint=False, retstep=True, dtype="int32"
            )[0],
        )

        self.assertAllClose(
            knp.Linspace(5, retstep=True)(start, stop)[0],
            np.linspace(start, stop, 5, retstep=True)[0],
        )
        self.assertAllClose(
            backend.convert_to_numpy(
                knp.Linspace(5, endpoint=False, retstep=True)(start, stop)[0]
            ),
            np.linspace(start, stop, 5, endpoint=False, retstep=True)[0],
        )
        self.assertAllClose(
            backend.convert_to_numpy(
                knp.Linspace(5, endpoint=False, retstep=True, dtype="int32")(
                    start, stop
                )[0]
            ),
            np.linspace(
                start, stop, 5, endpoint=False, retstep=True, dtype="int32"
            )[0],
        )

    def test_logical_and(self):
        x = np.array([[True, False], [True, True]])
        y = np.array([[False, False], [True, False]])
        self.assertAllClose(knp.logical_and(x, y), np.logical_and(x, y))
        self.assertAllClose(knp.logical_and(x, True), np.logical_and(x, True))
        self.assertAllClose(knp.logical_and(True, x), np.logical_and(True, x))

        self.assertAllClose(knp.LogicalAnd()(x, y), np.logical_and(x, y))
        self.assertAllClose(knp.LogicalAnd()(x, True), np.logical_and(x, True))
        self.assertAllClose(knp.LogicalAnd()(True, x), np.logical_and(True, x))

    def test_logical_or(self):
        x = np.array([[True, False], [True, True]])
        y = np.array([[False, False], [True, False]])
        self.assertAllClose(knp.logical_or(x, y), np.logical_or(x, y))
        self.assertAllClose(knp.logical_or(x, True), np.logical_or(x, True))
        self.assertAllClose(knp.logical_or(True, x), np.logical_or(True, x))

        self.assertAllClose(knp.LogicalOr()(x, y), np.logical_or(x, y))
        self.assertAllClose(knp.LogicalOr()(x, True), np.logical_or(x, True))
        self.assertAllClose(knp.LogicalOr()(True, x), np.logical_or(True, x))

    def test_logspace(self):
        self.assertAllClose(knp.logspace(0, 10, 5), np.logspace(0, 10, 5))
        self.assertAllClose(
            knp.logspace(0, 10, 5, endpoint=False),
            np.logspace(0, 10, 5, endpoint=False),
        )
        self.assertAllClose(knp.Logspace(num=5)(0, 10), np.logspace(0, 10, 5))
        self.assertAllClose(
            knp.Logspace(num=5, endpoint=False)(0, 10),
            np.logspace(0, 10, 5, endpoint=False),
        )

        start = np.zeros([2, 3, 4])
        stop = np.ones([2, 3, 4])

        self.assertAllClose(
            knp.logspace(start, stop, 5, base=10),
            np.logspace(start, stop, 5, base=10),
        )
        self.assertAllClose(
            knp.logspace(start, stop, 5, endpoint=False, base=10),
            np.logspace(start, stop, 5, endpoint=False, base=10),
        )

        self.assertAllClose(
            knp.Logspace(5, base=10)(start, stop),
            np.logspace(start, stop, 5, base=10),
        )
        self.assertAllClose(
            knp.Logspace(5, endpoint=False, base=10)(start, stop),
            np.logspace(start, stop, 5, endpoint=False, base=10),
        )

    def test_maximum(self):
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])
        self.assertAllClose(knp.maximum(x, y), np.maximum(x, y))
        self.assertAllClose(knp.maximum(x, 1), np.maximum(x, 1))
        self.assertAllClose(knp.maximum(1, x), np.maximum(1, x))

        self.assertAllClose(knp.Maximum()(x, y), np.maximum(x, y))
        self.assertAllClose(knp.Maximum()(x, 1), np.maximum(x, 1))
        self.assertAllClose(knp.Maximum()(1, x), np.maximum(1, x))

    def test_minimum(self):
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])
        self.assertAllClose(knp.minimum(x, y), np.minimum(x, y))
        self.assertAllClose(knp.minimum(x, 1), np.minimum(x, 1))
        self.assertAllClose(knp.minimum(1, x), np.minimum(1, x))

        self.assertAllClose(knp.Minimum()(x, y), np.minimum(x, y))
        self.assertAllClose(knp.Minimum()(x, 1), np.minimum(x, 1))
        self.assertAllClose(knp.Minimum()(1, x), np.minimum(1, x))

    def test_mod(self):
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])
        self.assertAllClose(knp.mod(x, y), np.mod(x, y))
        self.assertAllClose(knp.mod(x, 1), np.mod(x, 1))
        self.assertAllClose(knp.mod(1, x), np.mod(1, x))

        self.assertAllClose(knp.Mod()(x, y), np.mod(x, y))
        self.assertAllClose(knp.Mod()(x, 1), np.mod(x, 1))
        self.assertAllClose(knp.Mod()(1, x), np.mod(1, x))

    def test_not_equal(self):
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])
        self.assertAllClose(knp.not_equal(x, y), np.not_equal(x, y))
        self.assertAllClose(knp.not_equal(x, 1), np.not_equal(x, 1))
        self.assertAllClose(knp.not_equal(1, x), np.not_equal(1, x))

        self.assertAllClose(knp.NotEqual()(x, y), np.not_equal(x, y))
        self.assertAllClose(knp.NotEqual()(x, 1), np.not_equal(x, 1))
        self.assertAllClose(knp.NotEqual()(1, x), np.not_equal(1, x))

    def test_outer(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        self.assertAllClose(knp.outer(x, y), np.outer(x, y))
        self.assertAllClose(knp.Outer()(x, y), np.outer(x, y))

        x = np.ones([2, 3, 4])
        y = np.ones([2, 3, 4, 5, 6])
        self.assertAllClose(knp.outer(x, y), np.outer(x, y))
        self.assertAllClose(knp.Outer()(x, y), np.outer(x, y))

    def test_take(self):
        x = np.arange(24).reshape([1, 2, 3, 4])
        indices = np.array([0, 1])
        self.assertAllClose(knp.take(x, indices), np.take(x, indices))
        self.assertAllClose(knp.take(x, 0), np.take(x, 0))
        self.assertAllClose(knp.take(x, 0, axis=1), np.take(x, 0, axis=1))

        self.assertAllClose(knp.Take()(x, indices), np.take(x, indices))
        self.assertAllClose(knp.Take()(x, 0), np.take(x, 0))
        self.assertAllClose(knp.Take(axis=1)(x, 0), np.take(x, 0, axis=1))

        # test with multi-dimensional indices
        rng = np.random.default_rng(0)
        x = rng.standard_normal((2, 3, 4, 5))
        indices = rng.integers(0, 4, (6, 7))
        self.assertAllClose(
            knp.take(x, indices, axis=2),
            np.take(x, indices, axis=2),
        )

        # test with negative axis
        self.assertAllClose(
            knp.take(x, indices, axis=-2),
            np.take(x, indices, axis=-2),
        )

    def test_take_along_axis(self):
        x = np.arange(24).reshape([1, 2, 3, 4])
        indices = np.ones([1, 4, 1, 1], dtype=np.int32)
        self.assertAllClose(
            knp.take_along_axis(x, indices, axis=1),
            np.take_along_axis(x, indices, axis=1),
        )
        self.assertAllClose(
            knp.TakeAlongAxis(axis=1)(x, indices),
            np.take_along_axis(x, indices, axis=1),
        )

        x = np.arange(12).reshape([1, 1, 3, 4])
        indices = np.ones([1, 4, 1, 1], dtype=np.int32)
        self.assertAllClose(
            knp.take_along_axis(x, indices, axis=2),
            np.take_along_axis(x, indices, axis=2),
        )
        self.assertAllClose(
            knp.TakeAlongAxis(axis=2)(x, indices),
            np.take_along_axis(x, indices, axis=2),
        )

    def test_tensordot(self):
        x = np.arange(24).reshape([1, 2, 3, 4]).astype("float32")
        y = np.arange(24).reshape([3, 4, 1, 2]).astype("float32")
        self.assertAllClose(
            knp.tensordot(x, y, axes=2), np.tensordot(x, y, axes=2)
        )
        self.assertAllClose(
            knp.tensordot(x, y, axes=([0, 1], [2, 3])),
            np.tensordot(x, y, axes=([0, 1], [2, 3])),
        )
        self.assertAllClose(
            knp.Tensordot(axes=2)(x, y),
            np.tensordot(x, y, axes=2),
        )
        self.assertAllClose(
            knp.Tensordot(axes=([0, 1], [2, 3]))(x, y),
            np.tensordot(x, y, axes=([0, 1], [2, 3])),
        )

    def test_vdot(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        self.assertAllClose(knp.vdot(x, y), np.vdot(x, y))
        self.assertAllClose(knp.Vdot()(x, y), np.vdot(x, y))

    def test_where(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        self.assertAllClose(knp.where(x > 1, x, y), np.where(x > 1, x, y))
        self.assertAllClose(knp.Where()(x > 1, x, y), np.where(x > 1, x, y))
        self.assertAllClose(knp.where(x > 1), np.where(x > 1))
        self.assertAllClose(knp.Where()(x > 1), np.where(x > 1))

    def test_digitize(self):
        x = np.array([0.0, 1.0, 3.0, 1.6])
        bins = np.array([0.0, 3.0, 4.5, 7.0])
        self.assertAllClose(knp.digitize(x, bins), np.digitize(x, bins))
        self.assertAllClose(knp.Digitize()(x, bins), np.digitize(x, bins))
        self.assertTrue(
            standardize_dtype(knp.digitize(x, bins).dtype) == "int32"
        )
        self.assertTrue(
            standardize_dtype(knp.Digitize()(x, bins).dtype) == "int32"
        )

        x = np.array([0.2, 6.4, 3.0, 1.6])
        bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        self.assertAllClose(knp.digitize(x, bins), np.digitize(x, bins))
        self.assertAllClose(knp.Digitize()(x, bins), np.digitize(x, bins))
        self.assertTrue(
            standardize_dtype(knp.digitize(x, bins).dtype) == "int32"
        )
        self.assertTrue(
            standardize_dtype(knp.Digitize()(x, bins).dtype) == "int32"
        )

        x = np.array([1, 4, 10, 15])
        bins = np.array([4, 10, 14, 15])
        self.assertAllClose(knp.digitize(x, bins), np.digitize(x, bins))
        self.assertAllClose(knp.Digitize()(x, bins), np.digitize(x, bins))
        self.assertTrue(
            standardize_dtype(knp.digitize(x, bins).dtype) == "int32"
        )
        self.assertTrue(
            standardize_dtype(knp.Digitize()(x, bins).dtype) == "int32"
        )


class NumpyOneInputOpsCorrectnessTest(testing.TestCase):
    def test_mean(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.mean(x), np.mean(x))
        self.assertAllClose(knp.mean(x, axis=()), np.mean(x, axis=()))
        self.assertAllClose(knp.mean(x, axis=1), np.mean(x, axis=1))
        self.assertAllClose(knp.mean(x, axis=(1,)), np.mean(x, axis=(1,)))
        self.assertAllClose(
            knp.mean(x, axis=1, keepdims=True),
            np.mean(x, axis=1, keepdims=True),
        )

        self.assertAllClose(knp.Mean()(x), np.mean(x))
        self.assertAllClose(knp.Mean(axis=1)(x), np.mean(x, axis=1))
        self.assertAllClose(
            knp.Mean(axis=1, keepdims=True)(x),
            np.mean(x, axis=1, keepdims=True),
        )

    def test_all(self):
        x = np.array([[True, False, True], [True, True, True]])
        self.assertAllClose(knp.all(x), np.all(x))
        self.assertAllClose(knp.all(x, axis=()), np.all(x, axis=()))
        self.assertAllClose(knp.all(x, axis=1), np.all(x, axis=1))
        self.assertAllClose(knp.all(x, axis=(1,)), np.all(x, axis=(1,)))
        self.assertAllClose(
            knp.all(x, axis=1, keepdims=True),
            np.all(x, axis=1, keepdims=True),
        )

        self.assertAllClose(knp.All()(x), np.all(x))
        self.assertAllClose(knp.All(axis=1)(x), np.all(x, axis=1))
        self.assertAllClose(
            knp.All(axis=1, keepdims=True)(x),
            np.all(x, axis=1, keepdims=True),
        )

    def test_any(self):
        x = np.array([[True, False, True], [True, True, True]])
        self.assertAllClose(knp.any(x), np.any(x))
        self.assertAllClose(knp.any(x, axis=()), np.any(x, axis=()))
        self.assertAllClose(knp.any(x, axis=1), np.any(x, axis=1))
        self.assertAllClose(knp.any(x, axis=(1,)), np.any(x, axis=(1,)))
        self.assertAllClose(
            knp.any(x, axis=1, keepdims=True),
            np.any(x, axis=1, keepdims=True),
        )

        self.assertAllClose(knp.Any()(x), np.any(x))
        self.assertAllClose(knp.Any(axis=1)(x), np.any(x, axis=1))
        self.assertAllClose(
            knp.Any(axis=1, keepdims=True)(x),
            np.any(x, axis=1, keepdims=True),
        )

    def test_var(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.var(x), np.var(x))
        self.assertAllClose(knp.var(x, axis=()), np.var(x, axis=()))
        self.assertAllClose(knp.var(x, axis=1), np.var(x, axis=1))
        self.assertAllClose(knp.var(x, axis=(1,)), np.var(x, axis=(1,)))
        self.assertAllClose(
            knp.var(x, axis=1, keepdims=True),
            np.var(x, axis=1, keepdims=True),
        )

        self.assertAllClose(knp.Var()(x), np.var(x))
        self.assertAllClose(knp.Var(axis=1)(x), np.var(x, axis=1))
        self.assertAllClose(
            knp.Var(axis=1, keepdims=True)(x),
            np.var(x, axis=1, keepdims=True),
        )

    def test_sum(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.sum(x), np.sum(x))
        self.assertAllClose(knp.sum(x, axis=()), np.sum(x, axis=()))
        self.assertAllClose(knp.sum(x, axis=1), np.sum(x, axis=1))
        self.assertAllClose(knp.sum(x, axis=(1,)), np.sum(x, axis=(1,)))
        self.assertAllClose(
            knp.sum(x, axis=1, keepdims=True),
            np.sum(x, axis=1, keepdims=True),
        )

        self.assertAllClose(knp.Sum()(x), np.sum(x))
        self.assertAllClose(knp.Sum(axis=1)(x), np.sum(x, axis=1))
        self.assertAllClose(
            knp.Sum(axis=1, keepdims=True)(x),
            np.sum(x, axis=1, keepdims=True),
        )

    def test_amax(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.amax(x), np.amax(x))
        self.assertAllClose(knp.amax(x, axis=()), np.amax(x, axis=()))
        self.assertAllClose(knp.amax(x, axis=1), np.amax(x, axis=1))
        self.assertAllClose(knp.amax(x, axis=(1,)), np.amax(x, axis=(1,)))
        self.assertAllClose(
            knp.amax(x, axis=1, keepdims=True),
            np.amax(x, axis=1, keepdims=True),
        )

        self.assertAllClose(knp.Amax()(x), np.amax(x))
        self.assertAllClose(knp.Amax(axis=1)(x), np.amax(x, axis=1))
        self.assertAllClose(
            knp.Amax(axis=1, keepdims=True)(x),
            np.amax(x, axis=1, keepdims=True),
        )

    def test_amin(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.amin(x), np.amin(x))
        self.assertAllClose(knp.amin(x, axis=()), np.amin(x, axis=()))
        self.assertAllClose(knp.amin(x, axis=1), np.amin(x, axis=1))
        self.assertAllClose(knp.amin(x, axis=(1,)), np.amin(x, axis=(1,)))
        self.assertAllClose(
            knp.amin(x, axis=1, keepdims=True),
            np.amin(x, axis=1, keepdims=True),
        )

        self.assertAllClose(knp.Amin()(x), np.amin(x))
        self.assertAllClose(knp.Amin(axis=1)(x), np.amin(x, axis=1))
        self.assertAllClose(
            knp.Amin(axis=1, keepdims=True)(x),
            np.amin(x, axis=1, keepdims=True),
        )

    def test_square(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.square(x), np.square(x))

        self.assertAllClose(knp.Square()(x), np.square(x))

    def test_negative(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.negative(x), np.negative(x))

        self.assertAllClose(knp.Negative()(x), np.negative(x))

    def test_abs(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.abs(x), np.abs(x))

        self.assertAllClose(knp.Abs()(x), np.abs(x))

    def test_absolute(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.absolute(x), np.absolute(x))

        self.assertAllClose(knp.Absolute()(x), np.absolute(x))

    def test_squeeze(self):
        x = np.ones([1, 2, 3, 4, 5])
        self.assertAllClose(knp.squeeze(x), np.squeeze(x))
        self.assertAllClose(knp.squeeze(x, axis=0), np.squeeze(x, axis=0))

        self.assertAllClose(knp.Squeeze()(x), np.squeeze(x))
        self.assertAllClose(knp.Squeeze(axis=0)(x), np.squeeze(x, axis=0))

    def test_transpose(self):
        x = np.ones([1, 2, 3, 4, 5])
        self.assertAllClose(knp.transpose(x), np.transpose(x))
        self.assertAllClose(
            knp.transpose(x, axes=(1, 0, 3, 2, 4)),
            np.transpose(x, axes=(1, 0, 3, 2, 4)),
        )

        self.assertAllClose(knp.Transpose()(x), np.transpose(x))
        self.assertAllClose(
            knp.Transpose(axes=(1, 0, 3, 2, 4))(x),
            np.transpose(x, axes=(1, 0, 3, 2, 4)),
        )

    def test_arccos(self):
        x = np.array([[1, 0.5, -0.7], [0.9, 0.2, -1]])
        self.assertAllClose(knp.arccos(x), np.arccos(x))

        self.assertAllClose(knp.Arccos()(x), np.arccos(x))

    def test_arccosh(self):
        x = np.array([[1, 0.5, -0.7], [0.9, 0.2, -1]])
        self.assertAllClose(knp.arccosh(x), np.arccosh(x))

        self.assertAllClose(knp.Arccosh()(x), np.arccosh(x))

    def test_arcsin(self):
        x = np.array([[1, 0.5, -0.7], [0.9, 0.2, -1]])
        self.assertAllClose(knp.arcsin(x), np.arcsin(x))

        self.assertAllClose(knp.Arcsin()(x), np.arcsin(x))

    def test_arcsinh(self):
        x = np.array([[1, 0.5, -0.7], [0.9, 0.2, -1]])
        self.assertAllClose(knp.arcsinh(x), np.arcsinh(x))

        self.assertAllClose(knp.Arcsinh()(x), np.arcsinh(x))

    def test_arctan(self):
        x = np.array([[1, 0.5, -0.7], [0.9, 0.2, -1]])
        self.assertAllClose(knp.arctan(x), np.arctan(x))

        self.assertAllClose(knp.Arctan()(x), np.arctan(x))

    def test_arctanh(self):
        x = np.array([[1, 0.5, -0.7], [0.9, 0.2, -1]])
        self.assertAllClose(knp.arctanh(x), np.arctanh(x))

        self.assertAllClose(knp.Arctanh()(x), np.arctanh(x))

    def test_argmax(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.argmax(x), np.argmax(x))
        self.assertAllClose(knp.argmax(x, axis=1), np.argmax(x, axis=1))

        self.assertAllClose(knp.Argmax()(x), np.argmax(x))
        self.assertAllClose(knp.Argmax(axis=1)(x), np.argmax(x, axis=1))

    def test_argmin(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.argmin(x), np.argmin(x))
        self.assertAllClose(knp.argmin(x, axis=1), np.argmin(x, axis=1))

        self.assertAllClose(knp.Argmin()(x), np.argmin(x))
        self.assertAllClose(knp.Argmin(axis=1)(x), np.argmin(x, axis=1))

    def test_argsort(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.argsort(x), np.argsort(x))
        self.assertAllClose(knp.argsort(x, axis=1), np.argsort(x, axis=1))
        self.assertAllClose(
            knp.argsort(x, axis=None),
            np.argsort(x, axis=None),
        )

        self.assertAllClose(knp.Argsort()(x), np.argsort(x))
        self.assertAllClose(knp.Argsort(axis=1)(x), np.argsort(x, axis=1))
        self.assertAllClose(
            knp.Argsort(axis=None)(x),
            np.argsort(x, axis=None),
        )

    def test_array(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.array(x), np.array(x))
        self.assertAllClose(knp.Array()(x), np.array(x))
        self.assertTrue(backend.is_tensor(knp.array(x)))
        self.assertTrue(backend.is_tensor(knp.Array()(x)))

    def test_average(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        weights = np.ones([2, 3])
        weights_1d = np.ones([3])
        self.assertAllClose(knp.average(x), np.average(x))
        self.assertAllClose(knp.average(x, axis=()), np.average(x, axis=()))
        self.assertAllClose(knp.average(x, axis=1), np.average(x, axis=1))
        self.assertAllClose(knp.average(x, axis=(1,)), np.average(x, axis=(1,)))
        self.assertAllClose(
            knp.average(x, axis=1, weights=weights),
            np.average(x, axis=1, weights=weights),
        )
        self.assertAllClose(
            knp.average(x, axis=1, weights=weights_1d),
            np.average(x, axis=1, weights=weights_1d),
        )

        self.assertAllClose(knp.Average()(x), np.average(x))
        self.assertAllClose(knp.Average(axis=1)(x), np.average(x, axis=1))
        self.assertAllClose(
            knp.Average(axis=1)(x, weights=weights),
            np.average(x, axis=1, weights=weights),
        )
        self.assertAllClose(
            knp.Average(axis=1)(x, weights=weights_1d),
            np.average(x, axis=1, weights=weights_1d),
        )

    def test_bincount(self):
        x = np.array([1, 1, 2, 3, 2, 4, 4, 5])
        weights = np.array([0, 0, 3, 2, 1, 1, 4, 2])
        minlength = 3
        self.assertAllClose(
            knp.bincount(x, weights=weights, minlength=minlength),
            np.bincount(x, weights=weights, minlength=minlength),
        )
        self.assertAllClose(
            knp.Bincount(weights=weights, minlength=minlength)(x),
            np.bincount(x, weights=weights, minlength=minlength),
        )

    def test_broadcast_to(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(
            knp.broadcast_to(x, [2, 2, 3]),
            np.broadcast_to(x, [2, 2, 3]),
        )

        self.assertAllClose(
            knp.BroadcastTo([2, 2, 3])(x),
            np.broadcast_to(x, [2, 2, 3]),
        )

    def test_ceil(self):
        x = np.array([[1.2, 2.1, -2.5], [2.4, -11.9, -5.5]])
        self.assertAllClose(knp.ceil(x), np.ceil(x))
        self.assertAllClose(knp.Ceil()(x), np.ceil(x))

    def test_clip(self):
        x = np.array([[1.2, 2.1, -2.5], [2.4, -11.9, -5.5]])
        self.assertAllClose(knp.clip(x, -2, 2), np.clip(x, -2, 2))
        self.assertAllClose(knp.clip(x, -2, 2), np.clip(x, -2, 2))

        self.assertAllClose(knp.Clip(0, 1)(x), np.clip(x, 0, 1))
        self.assertAllClose(knp.Clip(0, 1)(x), np.clip(x, 0, 1))

    def test_concatenate(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [6, 5, 4]])
        z = np.array([[7, 8, 9], [9, 8, 7]])
        self.assertAllClose(
            knp.concatenate([x, y], axis=0),
            np.concatenate([x, y], axis=0),
        )
        self.assertAllClose(
            knp.concatenate([x, y, z], axis=0),
            np.concatenate([x, y, z], axis=0),
        )
        self.assertAllClose(
            knp.concatenate([x, y], axis=1),
            np.concatenate([x, y], axis=1),
        )

        self.assertAllClose(
            knp.Concatenate(axis=0)([x, y]),
            np.concatenate([x, y], axis=0),
        )
        self.assertAllClose(
            knp.Concatenate(axis=0)([x, y, z]),
            np.concatenate([x, y, z], axis=0),
        )
        self.assertAllClose(
            knp.Concatenate(axis=1)([x, y]),
            np.concatenate([x, y], axis=1),
        )

    def test_conjugate(self):
        x = np.array([[1 + 2j, 2 + 3j], [3 + 4j, 4 + 5j]])
        self.assertAllClose(knp.conjugate(x), np.conjugate(x))
        self.assertAllClose(knp.Conjugate()(x), np.conjugate(x))

    def test_conj(self):
        x = np.array([[1 + 2j, 2 + 3j], [3 + 4j, 4 + 5j]])
        self.assertAllClose(knp.conj(x), np.conj(x))
        self.assertAllClose(knp.Conj()(x), np.conj(x))

    def test_copy(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.copy(x), np.copy(x))
        self.assertAllClose(knp.Copy()(x), np.copy(x))

    def test_cos(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.cos(x), np.cos(x))
        self.assertAllClose(knp.Cos()(x), np.cos(x))

    def test_cosh(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.cosh(x), np.cosh(x))
        self.assertAllClose(knp.Cosh()(x), np.cosh(x))

    def test_count_nonzero(self):
        x = np.array([[0, 2, 3], [3, 2, 0]])
        self.assertAllClose(knp.count_nonzero(x), np.count_nonzero(x))
        self.assertAllClose(
            knp.count_nonzero(x, axis=()), np.count_nonzero(x, axis=())
        )
        self.assertAllClose(
            knp.count_nonzero(x, axis=1),
            np.count_nonzero(x, axis=1),
        )
        self.assertAllClose(
            knp.count_nonzero(x, axis=(1,)),
            np.count_nonzero(x, axis=(1,)),
        )

        self.assertAllClose(
            knp.CountNonzero()(x),
            np.count_nonzero(x),
        )
        self.assertAllClose(
            knp.CountNonzero(axis=1)(x),
            np.count_nonzero(x, axis=1),
        )

    def test_cumprod(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.cumprod(x), np.cumprod(x))
        self.assertAllClose(
            knp.cumprod(x, axis=0),
            np.cumprod(x, axis=0),
        )
        self.assertAllClose(
            knp.cumprod(x, axis=None),
            np.cumprod(x, axis=None),
        )

        self.assertAllClose(knp.Cumprod()(x), np.cumprod(x))
        self.assertAllClose(
            knp.Cumprod(axis=0)(x),
            np.cumprod(x, axis=0),
        )
        self.assertAllClose(
            knp.Cumprod(axis=None)(x),
            np.cumprod(x, axis=None),
        )

    def test_cumsum(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.cumsum(x), np.cumsum(x))
        self.assertAllClose(
            knp.cumsum(x, axis=0),
            np.cumsum(x, axis=0),
        )
        self.assertAllClose(
            knp.cumsum(x, axis=1),
            np.cumsum(x, axis=1),
        )

        self.assertAllClose(knp.Cumsum()(x), np.cumsum(x))
        self.assertAllClose(
            knp.Cumsum(axis=0)(x),
            np.cumsum(x, axis=0),
        )
        self.assertAllClose(
            knp.Cumsum(axis=1)(x),
            np.cumsum(x, axis=1),
        )

    def test_diag(self):
        x = np.array([1, 2, 3])
        self.assertAllClose(knp.diag(x), np.diag(x))
        self.assertAllClose(knp.diag(x, k=1), np.diag(x, k=1))
        self.assertAllClose(knp.diag(x, k=-1), np.diag(x, k=-1))

        self.assertAllClose(knp.Diag()(x), np.diag(x))
        self.assertAllClose(knp.Diag(k=1)(x), np.diag(x, k=1))
        self.assertAllClose(knp.Diag(k=-1)(x), np.diag(x, k=-1))

        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.diag(x), np.diag(x))
        self.assertAllClose(knp.diag(x, k=1), np.diag(x, k=1))
        self.assertAllClose(knp.diag(x, k=-1), np.diag(x, k=-1))

        self.assertAllClose(knp.Diag()(x), np.diag(x))
        self.assertAllClose(knp.Diag(k=1)(x), np.diag(x, k=1))
        self.assertAllClose(knp.Diag(k=-1)(x), np.diag(x, k=-1))

    def test_diagonal(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.diagonal(x), np.diagonal(x))
        self.assertAllClose(
            knp.diagonal(x, offset=1),
            np.diagonal(x, offset=1),
        )
        self.assertAllClose(
            knp.diagonal(x, offset=-1), np.diagonal(x, offset=-1)
        )

        self.assertAllClose(knp.Diagonal()(x), np.diagonal(x))
        self.assertAllClose(knp.Diagonal(offset=1)(x), np.diagonal(x, offset=1))
        self.assertAllClose(
            knp.Diagonal(offset=-1)(x), np.diagonal(x, offset=-1)
        )

        x = np.ones([2, 3, 4, 5])
        self.assertAllClose(knp.diagonal(x), np.diagonal(x))
        self.assertAllClose(
            knp.diagonal(x, offset=1, axis1=2, axis2=3),
            np.diagonal(x, offset=1, axis1=2, axis2=3),
        )
        self.assertAllClose(
            knp.diagonal(x, offset=-1, axis1=2, axis2=3),
            np.diagonal(x, offset=-1, axis1=2, axis2=3),
        )

    def test_dot(self):
        x = np.arange(24).reshape([2, 3, 4]).astype("float32")
        y = np.arange(12).reshape([4, 3]).astype("float32")
        z = np.arange(4).astype("float32")
        self.assertAllClose(knp.dot(x, y), np.dot(x, y))
        self.assertAllClose(knp.dot(x, z), np.dot(x, z))
        self.assertAllClose(knp.dot(x, 2), np.dot(x, 2))

        self.assertAllClose(knp.Dot()(x, y), np.dot(x, y))
        self.assertAllClose(knp.Dot()(x, z), np.dot(x, z))
        self.assertAllClose(knp.Dot()(x, 2), np.dot(x, 2))

    def test_exp(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.exp(x), np.exp(x))
        self.assertAllClose(knp.Exp()(x), np.exp(x))

    def test_expand_dims(self):
        x = np.ones([2, 3, 4])
        self.assertAllClose(knp.expand_dims(x, 0), np.expand_dims(x, 0))
        self.assertAllClose(knp.expand_dims(x, 1), np.expand_dims(x, 1))
        self.assertAllClose(knp.expand_dims(x, -2), np.expand_dims(x, -2))

        self.assertAllClose(knp.ExpandDims(0)(x), np.expand_dims(x, 0))
        self.assertAllClose(knp.ExpandDims(1)(x), np.expand_dims(x, 1))
        self.assertAllClose(knp.ExpandDims(-2)(x), np.expand_dims(x, -2))

    def test_expm1(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.expm1(x), np.expm1(x))
        self.assertAllClose(knp.Expm1()(x), np.expm1(x))

    def test_flip(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.flip(x), np.flip(x))
        self.assertAllClose(knp.flip(x, 0), np.flip(x, 0))
        self.assertAllClose(knp.flip(x, 1), np.flip(x, 1))

        self.assertAllClose(knp.Flip()(x), np.flip(x))
        self.assertAllClose(knp.Flip(0)(x), np.flip(x, 0))
        self.assertAllClose(knp.Flip(1)(x), np.flip(x, 1))

    def test_floor(self):
        x = np.array([[1.1, 2.2, -3.3], [3.3, 2.2, -1.1]])
        self.assertAllClose(knp.floor(x), np.floor(x))
        self.assertAllClose(knp.Floor()(x), np.floor(x))

    def test_hstack(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [6, 5, 4]])
        self.assertAllClose(knp.hstack([x, y]), np.hstack([x, y]))
        self.assertAllClose(knp.Hstack()([x, y]), np.hstack([x, y]))

        x = np.ones([2, 3, 4])
        y = np.ones([2, 5, 4])
        self.assertAllClose(knp.hstack([x, y]), np.hstack([x, y]))
        self.assertAllClose(knp.Hstack()([x, y]), np.hstack([x, y]))

    def test_imag(self):
        x = np.array([[1 + 1j, 2 + 2j, 3 + 3j], [3 + 3j, 2 + 2j, 1 + 1j]])
        self.assertAllClose(knp.imag(x), np.imag(x))
        self.assertAllClose(knp.Imag()(x), np.imag(x))

    def test_isfinite(self):
        x = np.array([[1, 2, np.inf], [np.nan, np.nan, np.nan]])
        self.assertAllClose(knp.isfinite(x), np.isfinite(x))
        self.assertAllClose(knp.Isfinite()(x), np.isfinite(x))

    def test_isinf(self):
        x = np.array([[1, 2, np.inf], [np.nan, np.nan, np.nan]])
        self.assertAllClose(knp.isinf(x), np.isinf(x))
        self.assertAllClose(knp.Isinf()(x), np.isinf(x))

    def test_isnan(self):
        x = np.array([[1, 2, np.inf], [np.nan, np.nan, np.nan]])
        self.assertAllClose(knp.isnan(x), np.isnan(x))
        self.assertAllClose(knp.Isnan()(x), np.isnan(x))

    def test_log(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.log(x), np.log(x))
        self.assertAllClose(knp.Log()(x), np.log(x))

    def test_log10(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.log10(x), np.log10(x))
        self.assertAllClose(knp.Log10()(x), np.log10(x))

    def test_log1p(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.log1p(x), np.log1p(x))
        self.assertAllClose(knp.Log1p()(x), np.log1p(x))

    def test_log2(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.log2(x), np.log2(x))
        self.assertAllClose(knp.Log2()(x), np.log2(x))

    def test_logaddexp(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.logaddexp(x, y), np.logaddexp(x, y))
        self.assertAllClose(knp.Logaddexp()(x, y), np.logaddexp(x, y))

    def test_logical_not(self):
        x = np.array([[True, False], [False, True]])
        self.assertAllClose(knp.logical_not(x), np.logical_not(x))
        self.assertAllClose(knp.LogicalNot()(x), np.logical_not(x))

    def test_max(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.max(x), np.max(x))
        self.assertAllClose(knp.Max()(x), np.max(x))

        self.assertAllClose(knp.max(x, 0), np.max(x, 0))
        self.assertAllClose(knp.Max(0)(x), np.max(x, 0))

        self.assertAllClose(knp.max(x, 1), np.max(x, 1))
        self.assertAllClose(knp.Max(1)(x), np.max(x, 1))

        # test max with initial
        self.assertAllClose(knp.max(x, initial=4), 4)

        # test empty tensor
        x = np.array([[]])
        self.assertAllClose(knp.max(x, initial=1), np.max(x, initial=1))
        self.assertAllClose(
            knp.max(x, initial=1, keepdims=True),
            np.max(x, initial=1, keepdims=True),
        )

    def test_min(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.min(x), np.min(x))
        self.assertAllClose(knp.Min()(x), np.min(x))

        self.assertAllClose(knp.min(x, 0), np.min(x, 0))
        self.assertAllClose(knp.Min(0)(x), np.min(x, 0))

        self.assertAllClose(knp.min(x, 1), np.min(x, 1))
        self.assertAllClose(knp.Min(1)(x), np.min(x, 1))

        # test min with initial
        self.assertAllClose(knp.min(x, initial=0), 0)

        # test empty tensor
        x = np.array([[]])
        self.assertAllClose(knp.min(x, initial=1), np.min(x, initial=1))
        self.assertAllClose(
            knp.min(x, initial=1, keepdims=True),
            np.min(x, initial=1, keepdims=True),
        )

    def test_meshgrid(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        z = np.array([7, 8, 9])
        self.assertAllClose(knp.meshgrid(x, y), np.meshgrid(x, y))
        self.assertAllClose(knp.meshgrid(x, z), np.meshgrid(x, z))
        self.assertAllClose(
            knp.meshgrid(x, y, z, indexing="ij"),
            np.meshgrid(x, y, z, indexing="ij"),
        )
        self.assertAllClose(knp.Meshgrid()(x, y), np.meshgrid(x, y))
        self.assertAllClose(knp.Meshgrid()(x, z), np.meshgrid(x, z))
        self.assertAllClose(
            knp.Meshgrid(indexing="ij")(x, y, z),
            np.meshgrid(x, y, z, indexing="ij"),
        )

        if backend.backend() == "tensorflow":
            # Arguments to `jax.numpy.meshgrid` must be 1D now.
            x = np.ones([1, 2, 3])
            y = np.ones([4, 5, 6, 6])
            z = np.ones([7, 8])
            self.assertAllClose(knp.meshgrid(x, y), np.meshgrid(x, y))
            self.assertAllClose(knp.meshgrid(x, z), np.meshgrid(x, z))
            self.assertAllClose(
                knp.meshgrid(x, y, z, indexing="ij"),
                np.meshgrid(x, y, z, indexing="ij"),
            )
            self.assertAllClose(knp.Meshgrid()(x, y), np.meshgrid(x, y))
            self.assertAllClose(knp.Meshgrid()(x, z), np.meshgrid(x, z))
            self.assertAllClose(
                knp.Meshgrid(indexing="ij")(x, y, z),
                np.meshgrid(x, y, z, indexing="ij"),
            )

    def test_moveaxis(self):
        x = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
        self.assertAllClose(knp.moveaxis(x, 0, -1), np.moveaxis(x, 0, -1))
        self.assertAllClose(knp.moveaxis(x, -1, 0), np.moveaxis(x, -1, 0))
        self.assertAllClose(
            knp.moveaxis(x, (0, 1), (1, 0)),
            np.moveaxis(x, (0, 1), (1, 0)),
        )
        self.assertAllClose(
            knp.moveaxis(x, [0, 1, 2], [2, 0, 1]),
            np.moveaxis(x, [0, 1, 2], [2, 0, 1]),
        )
        self.assertAllClose(knp.Moveaxis(-1, 0)(x), np.moveaxis(x, -1, 0))
        self.assertAllClose(
            knp.Moveaxis((0, 1), (1, 0))(x),
            np.moveaxis(x, (0, 1), (1, 0)),
        )

        self.assertAllClose(
            knp.Moveaxis([0, 1, 2], [2, 0, 1])(x),
            np.moveaxis(x, [0, 1, 2], [2, 0, 1]),
        )

    def test_ndim(self):
        x = np.array([1, 2, 3])
        self.assertEqual(knp.ndim(x), np.ndim(x))
        self.assertEqual(knp.Ndim()(x), np.ndim(x))

    def test_nonzero(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.nonzero(x), np.nonzero(x))
        self.assertAllClose(knp.Nonzero()(x), np.nonzero(x))

    def test_ones_like(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.ones_like(x), np.ones_like(x))
        self.assertAllClose(knp.OnesLike()(x), np.ones_like(x))

    def test_pad(self):
        x = np.array([[1, 2], [3, 4]])
        self.assertAllClose(
            knp.pad(x, ((1, 1), (1, 1))),
            np.pad(x, ((1, 1), (1, 1))),
        )
        self.assertAllClose(
            knp.pad(x, ((1, 1), (1, 1))),
            np.pad(x, ((1, 1), (1, 1))),
        )

        self.assertAllClose(
            knp.Pad(((1, 1), (1, 1)))(x),
            np.pad(x, ((1, 1), (1, 1))),
        )
        self.assertAllClose(
            knp.Pad(((1, 1), (1, 1)))(x),
            np.pad(x, ((1, 1), (1, 1))),
        )

        self.assertAllClose(
            knp.pad(x, ((1, 1), (1, 1)), mode="reflect"),
            np.pad(x, ((1, 1), (1, 1)), mode="reflect"),
        )
        self.assertAllClose(
            knp.pad(x, ((1, 1), (1, 1)), mode="symmetric"),
            np.pad(x, ((1, 1), (1, 1)), mode="symmetric"),
        )

        self.assertAllClose(
            knp.Pad(((1, 1), (1, 1)), mode="reflect")(x),
            np.pad(x, ((1, 1), (1, 1)), mode="reflect"),
        )
        self.assertAllClose(
            knp.Pad(((1, 1), (1, 1)), mode="symmetric")(x),
            np.pad(x, ((1, 1), (1, 1)), mode="symmetric"),
        )

        x = np.ones([2, 3, 4, 5])
        self.assertAllClose(
            knp.pad(x, ((2, 3), (1, 1), (1, 1), (1, 1))),
            np.pad(x, ((2, 3), (1, 1), (1, 1), (1, 1))),
        )
        self.assertAllClose(
            knp.Pad(((2, 3), (1, 1), (1, 1), (1, 1)))(x),
            np.pad(x, ((2, 3), (1, 1), (1, 1), (1, 1))),
        )

    def test_prod(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.prod(x), np.prod(x))
        self.assertAllClose(knp.prod(x, axis=()), np.prod(x, axis=()))
        self.assertAllClose(knp.prod(x, axis=1), np.prod(x, axis=1))
        self.assertAllClose(knp.prod(x, axis=(1,)), np.prod(x, axis=(1,)))
        self.assertAllClose(
            knp.prod(x, axis=1, keepdims=True),
            np.prod(x, axis=1, keepdims=True),
        )

        self.assertAllClose(knp.Prod()(x), np.prod(x))
        self.assertAllClose(knp.Prod(axis=1)(x), np.prod(x, axis=1))
        self.assertAllClose(
            knp.Prod(axis=1, keepdims=True)(x),
            np.prod(x, axis=1, keepdims=True),
        )

    def test_ravel(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.ravel(x), np.ravel(x))
        self.assertAllClose(knp.Ravel()(x), np.ravel(x))

    def test_real(self):
        x = np.array([[1, 2, 3 - 3j], [3, 2, 1 + 5j]])
        self.assertAllClose(knp.real(x), np.real(x))
        self.assertAllClose(knp.Real()(x), np.real(x))

    def test_reciprocal(self):
        x = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        self.assertAllClose(knp.reciprocal(x), np.reciprocal(x))
        self.assertAllClose(knp.Reciprocal()(x), np.reciprocal(x))

    def test_repeat(self):
        x = np.array([[1, 2], [3, 4]])
        self.assertAllClose(knp.repeat(x, 2), np.repeat(x, 2))
        self.assertAllClose(knp.repeat(x, 3, axis=1), np.repeat(x, 3, axis=1))
        self.assertAllClose(
            knp.repeat(x, np.array([1, 2]), axis=-1),
            np.repeat(x, np.array([1, 2]), axis=-1),
        )
        self.assertAllClose(knp.Repeat(2)(x), np.repeat(x, 2))
        self.assertAllClose(knp.Repeat(3, axis=1)(x), np.repeat(x, 3, axis=1))
        self.assertAllClose(
            knp.Repeat(np.array([1, 2]), axis=0)(x),
            np.repeat(x, np.array([1, 2]), axis=0),
        )

    def test_reshape(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.reshape(x, [3, 2]), np.reshape(x, [3, 2]))
        self.assertAllClose(knp.Reshape([3, 2])(x), np.reshape(x, [3, 2]))

    def test_roll(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.roll(x, 1), np.roll(x, 1))
        self.assertAllClose(knp.roll(x, 1, axis=1), np.roll(x, 1, axis=1))
        self.assertAllClose(knp.roll(x, -1, axis=0), np.roll(x, -1, axis=0))
        self.assertAllClose(knp.Roll(1)(x), np.roll(x, 1))
        self.assertAllClose(knp.Roll(1, axis=1)(x), np.roll(x, 1, axis=1))
        self.assertAllClose(knp.Roll(-1, axis=0)(x), np.roll(x, -1, axis=0))

    def test_round(self):
        x = np.array([[1.1, 2.5, 3.9], [3.2, 2.3, 1.8]])
        self.assertAllClose(knp.round(x), np.round(x))
        self.assertAllClose(knp.Round()(x), np.round(x))

    def test_sign(self):
        x = np.array([[1, -2, 3], [-3, 2, -1]])
        self.assertAllClose(knp.sign(x), np.sign(x))
        self.assertAllClose(knp.Sign()(x), np.sign(x))

    def test_sin(self):
        x = np.array([[1, -2, 3], [-3, 2, -1]])
        self.assertAllClose(knp.sin(x), np.sin(x))
        self.assertAllClose(knp.Sin()(x), np.sin(x))

    def test_sinh(self):
        x = np.array([[1, -2, 3], [-3, 2, -1]])
        self.assertAllClose(knp.sinh(x), np.sinh(x))
        self.assertAllClose(knp.Sinh()(x), np.sinh(x))

    def test_size(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.size(x), np.size(x))
        self.assertAllClose(knp.Size()(x), np.size(x))

    def test_sort(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.sort(x), np.sort(x))
        self.assertAllClose(knp.Sort()(x), np.sort(x))
        self.assertAllClose(knp.sort(x, axis=0), np.sort(x, axis=0))
        self.assertAllClose(knp.Sort(axis=0)(x), np.sort(x, axis=0))

    def test_split(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        if backend.backend() == "torch":
            self.assertAllClose(
                [backend.convert_to_numpy(t) for t in knp.split(x, 2)],
                np.split(x, 2),
            )
            self.assertAllClose(
                [backend.convert_to_numpy(t) for t in knp.Split(2)(x)],
                np.split(x, 2),
            )
            self.assertAllClose(
                [
                    backend.convert_to_numpy(t)
                    for t in knp.split(x, [1, 2], axis=1)
                ],
                np.split(x, [1, 2], axis=1),
            )
            self.assertAllClose(
                [
                    backend.convert_to_numpy(t)
                    for t in knp.Split([1, 2], axis=1)(x)
                ],
                np.split(x, [1, 2], axis=1),
            )
        else:
            self.assertAllClose(knp.split(x, 2), np.split(x, 2))
            self.assertAllClose(knp.Split(2)(x), np.split(x, 2))
            self.assertAllClose(
                knp.split(x, [1, 2], axis=1),
                np.split(x, [1, 2], axis=1),
            )
            self.assertAllClose(
                knp.Split([1, 2], axis=1)(x),
                np.split(x, [1, 2], axis=1),
            )

    def test_sqrt(self):
        x = np.array([[1, 4, 9], [16, 25, 36]])
        self.assertAllClose(knp.sqrt(x), np.sqrt(x))
        self.assertAllClose(knp.Sqrt()(x), np.sqrt(x))

    def test_stack(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [6, 5, 4]])
        self.assertAllClose(knp.stack([x, y]), np.stack([x, y]))
        self.assertAllClose(knp.stack([x, y], axis=1), np.stack([x, y], axis=1))
        self.assertAllClose(knp.Stack()([x, y]), np.stack([x, y]))
        self.assertAllClose(knp.Stack(axis=1)([x, y]), np.stack([x, y], axis=1))

    def test_std(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.std(x), np.std(x))
        self.assertAllClose(knp.std(x, axis=1), np.std(x, axis=1))
        self.assertAllClose(
            knp.std(x, axis=1, keepdims=True),
            np.std(x, axis=1, keepdims=True),
        )

        self.assertAllClose(knp.Std()(x), np.std(x))
        self.assertAllClose(knp.Std(axis=1)(x), np.std(x, axis=1))
        self.assertAllClose(
            knp.Std(axis=1, keepdims=True)(x),
            np.std(x, axis=1, keepdims=True),
        )

    def test_swapaxes(self):
        x = np.arange(24).reshape([1, 2, 3, 4])
        self.assertAllClose(
            knp.swapaxes(x, 0, 1),
            np.swapaxes(x, 0, 1),
        )
        self.assertAllClose(
            knp.Swapaxes(0, 1)(x),
            np.swapaxes(x, 0, 1),
        )

    def test_tan(self):
        x = np.array([[1, -2, 3], [-3, 2, -1]])
        self.assertAllClose(knp.tan(x), np.tan(x))
        self.assertAllClose(knp.Tan()(x), np.tan(x))

    def test_tanh(self):
        x = np.array([[1, -2, 3], [-3, 2, -1]])
        self.assertAllClose(knp.tanh(x), np.tanh(x))
        self.assertAllClose(knp.Tanh()(x), np.tanh(x))

    def test_tile(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.tile(x, [2, 3]), np.tile(x, [2, 3]))
        self.assertAllClose(knp.Tile([2, 3])(x), np.tile(x, [2, 3]))

    def test_trace(self):
        x = np.arange(24).reshape([1, 2, 3, 4])
        self.assertAllClose(knp.trace(x), np.trace(x))
        self.assertAllClose(
            knp.trace(x, axis1=2, axis2=3),
            np.trace(x, axis1=2, axis2=3),
        )
        self.assertAllClose(
            knp.Trace(axis1=2, axis2=3)(x),
            np.trace(x, axis1=2, axis2=3),
        )

    def test_tril(self):
        x = np.arange(24).reshape([1, 2, 3, 4])
        self.assertAllClose(knp.tril(x), np.tril(x))
        self.assertAllClose(knp.tril(x, -1), np.tril(x, -1))
        self.assertAllClose(knp.Tril(-1)(x), np.tril(x, -1))

    def test_triu(self):
        x = np.arange(24).reshape([1, 2, 3, 4])
        self.assertAllClose(knp.triu(x), np.triu(x))
        self.assertAllClose(knp.triu(x, -1), np.triu(x, -1))
        self.assertAllClose(knp.Triu(-1)(x), np.triu(x, -1))

    def test_vstack(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [6, 5, 4]])
        self.assertAllClose(knp.vstack([x, y]), np.vstack([x, y]))
        self.assertAllClose(knp.Vstack()([x, y]), np.vstack([x, y]))

    def test_floordiv(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        z = np.array([[[1, 2, 3], [3, 2, 1]]])
        self.assertAllClose(knp.floor_divide(x, y), np.floor_divide(x, y))
        self.assertAllClose(knp.floor_divide(x, z), np.floor_divide(x, z))

        self.assertAllClose(knp.FloorDivide()(x, y), np.floor_divide(x, y))
        self.assertAllClose(knp.FloorDivide()(x, z), np.floor_divide(x, z))

    def test_xor(self):
        x = np.array([[True, False], [True, True]])
        y = np.array([[False, False], [True, False]])
        self.assertAllClose(knp.logical_xor(x, y), np.logical_xor(x, y))
        self.assertAllClose(knp.logical_xor(x, True), np.logical_xor(x, True))
        self.assertAllClose(knp.logical_xor(True, x), np.logical_xor(True, x))

        self.assertAllClose(knp.LogicalXor()(x, y), np.logical_xor(x, y))
        self.assertAllClose(knp.LogicalXor()(x, True), np.logical_xor(x, True))
        self.assertAllClose(knp.LogicalXor()(True, x), np.logical_xor(True, x))


class NumpyArrayCreateOpsCorrectnessTest(testing.TestCase):
    def test_ones(self):
        self.assertAllClose(knp.ones([2, 3]), np.ones([2, 3]))
        self.assertAllClose(knp.Ones()([2, 3]), np.ones([2, 3]))

    def test_zeros(self):
        self.assertAllClose(knp.zeros([2, 3]), np.zeros([2, 3]))
        self.assertAllClose(knp.Zeros()([2, 3]), np.zeros([2, 3]))

    def test_eye(self):
        self.assertAllClose(knp.eye(3), np.eye(3))
        self.assertAllClose(knp.eye(3, 4), np.eye(3, 4))
        self.assertAllClose(knp.eye(3, 4, 1), np.eye(3, 4, 1))

        self.assertAllClose(knp.Eye()(3), np.eye(3))
        self.assertAllClose(knp.Eye()(3, 4), np.eye(3, 4))
        self.assertAllClose(knp.Eye()(3, 4, 1), np.eye(3, 4, 1))

    def test_arange(self):
        self.assertAllClose(knp.arange(3), np.arange(3))
        self.assertAllClose(knp.arange(3, 7), np.arange(3, 7))
        self.assertAllClose(knp.arange(3, 7, 2), np.arange(3, 7, 2))

        self.assertAllClose(knp.Arange()(3), np.arange(3))
        self.assertAllClose(knp.Arange()(3, 7), np.arange(3, 7))
        self.assertAllClose(knp.Arange()(3, 7, 2), np.arange(3, 7, 2))

    def test_full(self):
        self.assertAllClose(knp.full([2, 3], 0), np.full([2, 3], 0))
        self.assertAllClose(knp.full([2, 3], 0.1), np.full([2, 3], 0.1))
        self.assertAllClose(
            knp.full([2, 3], np.array([1, 4, 5])),
            np.full([2, 3], np.array([1, 4, 5])),
        )

        self.assertAllClose(knp.Full()([2, 3], 0), np.full([2, 3], 0))
        self.assertAllClose(knp.Full()([2, 3], 0.1), np.full([2, 3], 0.1))
        self.assertAllClose(
            knp.Full()([2, 3], np.array([1, 4, 5])),
            np.full([2, 3], np.array([1, 4, 5])),
        )

    def test_identity(self):
        self.assertAllClose(knp.identity(3), np.identity(3))
        self.assertAllClose(knp.Identity()(3), np.identity(3))

    def test_tri(self):
        self.assertAllClose(knp.tri(3), np.tri(3))
        self.assertAllClose(knp.tri(3, 4), np.tri(3, 4))
        self.assertAllClose(knp.tri(3, 4, 1), np.tri(3, 4, 1))

        self.assertAllClose(knp.Tri()(3), np.tri(3))
        self.assertAllClose(knp.Tri()(3, 4), np.tri(3, 4))
        self.assertAllClose(knp.Tri()(3, 4, 1), np.tri(3, 4, 1))
