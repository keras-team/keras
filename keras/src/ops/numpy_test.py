import functools
import itertools
import math
import warnings

import numpy as np
import pytest
from absl.testing import parameterized

import keras
from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.backend.common import dtypes
from keras.src.backend.common import is_int_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.ops import numpy as knp
from keras.src.testing.test_utils import named_product


class NumPyTestRot90(testing.TestCase):
    def test_basic_rotation(self):
        array = np.array([[1, 2, 3], [4, 5, 6]])
        rotated = knp.rot90(array)
        expected = np.rot90(array)
        self.assertAllClose(rotated, expected)

    @parameterized.named_parameters(
        ("k_0", 0, [[1, 2], [3, 4]]),
        ("k_1", 1, [[2, 4], [1, 3]]),
        ("k_2", 2, [[4, 3], [2, 1]]),
        ("k_neg1", -1, [[3, 1], [4, 2]]),
        ("k_5", 5, [[2, 4], [1, 3]]),  # k=5 ≡ k=1 (mod 4)
        ("k_6", 6, [[4, 3], [2, 1]]),  # k=6 ≡ k=2 (mod 4)
    )
    def test_k_parameter_variations(self, k, expected):
        array = np.array([[1, 2], [3, 4]])
        rotated = knp.rot90(array, k=k)
        expected = np.array(expected)
        self.assertAllClose(rotated, expected)

    @parameterized.named_parameters(
        ("axes_0_1", (0, 1)), ("axes_1_2", (1, 2)), ("axes_0_2", (0, 2))
    )
    def test_3d_operations(self, axes):
        array_3d = np.arange(12).reshape(3, 2, 2)
        rotated = knp.rot90(array_3d, axes=axes)
        expected = np.rot90(array_3d, axes=axes)
        self.assertAllClose(rotated, expected)

    @parameterized.named_parameters(
        ("single_image", np.random.random((4, 4, 3))),
        ("batch_images", np.random.random((2, 4, 4, 3))),
    )
    def test_image_processing(self, array):
        np.random.seed(0)
        rotated = knp.rot90(array, axes=(0, 1))
        expected = np.rot90(array, axes=(0, 1))
        self.assertAllClose(rotated, expected)

    @parameterized.named_parameters(
        ("single_row", [[1, 2, 3]]),
        ("single_column", [[1], [2], [3]]),
        ("negative_values", [[-1, 0], [1, -2]]),
    )
    def test_edge_conditions(self, array):
        numpy_array = np.array(array)
        rotated = knp.rot90(numpy_array)
        expected = np.rot90(numpy_array)
        self.assertAllClose(rotated, expected)

    @parameterized.named_parameters(
        ("1D_array", np.array([1, 2, 3]), None),
        ("duplicate_axes", np.array([[1, 2], [3, 4]]), (0, 0)),
    )
    def test_error_conditions(self, array, axes):
        if axes is None:
            with self.assertRaises(ValueError):
                knp.rot90(array)
        else:
            with self.assertRaises(ValueError):
                knp.rot90(array, axes=axes)


class NumpyTwoInputOpsDynamicShapeTest(testing.TestCase):
    def test_add(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.add(x, y).shape, (2, 3))

    def test_heaviside(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((None, 3))
        self.assertEqual(knp.heaviside(x, y).shape, (None, 3))

    def test_hypot(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((None, 3))
        self.assertEqual(knp.hypot(x, y).shape, (None, 3))

    def test_subtract(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.subtract(x, y).shape, (2, 3))

    def test_multiply(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.multiply(x, y).shape, (2, 3))

    def test_matmul(self):
        x = KerasTensor((None, 3, 4))
        y = KerasTensor((3, None, 4, 5))
        self.assertEqual(knp.matmul(x, y).shape, (3, None, 3, 5))

    def test_power(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.power(x, y).shape, (2, 3))

    def test_divide(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.divide(x, y).shape, (2, 3))

    def test_divide_no_nan(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.divide_no_nan(x, y).shape, (2, 3))

    def test_true_divide(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.true_divide(x, y).shape, (2, 3))

    def test_append(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.append(x, y).shape, (None,))

    def test_arctan2(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.arctan2(x, y).shape, (2, 3))

    def test_bitwise_and(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((None, 3))
        self.assertEqual(knp.bitwise_and(x, y).shape, (None, 3))

    def test_bitwise_or(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((None, 3))
        self.assertEqual(knp.bitwise_or(x, y).shape, (None, 3))

    def test_bitwise_xor(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((None, 3))
        self.assertEqual(knp.bitwise_xor(x, y).shape, (None, 3))

    def test_bitwise_left_shift(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((None, 3))
        self.assertEqual(knp.bitwise_left_shift(x, y).shape, (None, 3))

    # left_shift is same as bitwise_left_shift

    def test_bitwise_right_shift(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((None, 3))
        self.assertEqual(knp.bitwise_right_shift(x, y).shape, (None, 3))

    # right_shift is same as bitwise_right_shift

    def test_cross(self):
        x1 = KerasTensor((2, 3, 3))
        x2 = KerasTensor((1, 3, 2))
        y = KerasTensor((None, 1, 2))
        self.assertEqual(knp.cross(x1, y).shape, (2, 3, 3))
        self.assertEqual(knp.cross(x2, y).shape, (None, 3))

    def test_einsum(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((3, 4))
        self.assertEqual(knp.einsum("ij,jk->ik", x, y).shape, (None, 4))
        self.assertEqual(knp.einsum("ij,jk->ikj", x, y).shape, (None, 4, 3))
        self.assertEqual(knp.einsum("ii", x).shape, ())
        self.assertEqual(knp.einsum(",ij", 5, x).shape, (None, 3))

        x = KerasTensor((None, 3, 4))
        y = KerasTensor((None, 4, 5))
        z = KerasTensor((1, 1, 1, 9))
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
            x = KerasTensor((None, 3))
            y = KerasTensor((3, 4))
            knp.einsum("ijk,jk->ik", x, y)

    def test_full_like(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.full_like(x, KerasTensor((1, 3))).shape, (None, 3))

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.full_like(x, 2).shape, (None, 3, 3))

    def test_gcd(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.gcd(x, y).shape, (2, 3))

    def test_geomspace(self):
        start = KerasTensor((None, 3, 4))
        stop = KerasTensor((2, 3, 4))
        self.assertEqual(
            knp.geomspace(start, stop, 10, axis=1).shape, (2, 10, 3, 4)
        )

        start = KerasTensor((None, 3))
        stop = 2
        self.assertEqual(
            knp.geomspace(start, stop, 10, axis=1).shape, (None, 10, 3)
        )

    def test_greater(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.greater(x, y).shape, (2, 3))

    def test_greater_equal(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.greater_equal(x, y).shape, (2, 3))

    def test_allclose(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.allclose(x, y).shape, ())

    def test_isclose(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.isclose(x, y).shape, (2, 3))

    def test_isin(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.isin(x, y).shape, (None, 3))

    def test_kron(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.kron(x, y).shape, (None, None))

    def test_lcm(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.lcm(x, y).shape, (2, 3))

    def test_ldexp(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((1, 3))
        self.assertEqual(knp.ldexp(x, y).shape, (None, 3))

    def test_less(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.less(x, y).shape, (2, 3))

    def test_less_equal(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.less_equal(x, y).shape, (2, 3))

    def test_linspace(self):
        start = KerasTensor((None, 3, 4))
        stop = KerasTensor((2, 3, 4))
        self.assertEqual(
            knp.linspace(start, stop, 10, axis=1).shape, (2, 10, 3, 4)
        )

        start = KerasTensor((None, 3))
        stop = 2
        self.assertEqual(
            knp.linspace(start, stop, 10, axis=1).shape, (None, 10, 3)
        )

    def test_logical_and(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.logical_and(x, y).shape, (2, 3))

    def test_logical_or(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.logical_or(x, y).shape, (2, 3))

    def test_logspace(self):
        start = KerasTensor((None, 3, 4))
        stop = KerasTensor((2, 3, 4))
        self.assertEqual(
            knp.logspace(start, stop, 10, axis=1).shape, (2, 10, 3, 4)
        )

        start = KerasTensor((None, 3))
        stop = 2
        self.assertEqual(
            knp.logspace(start, stop, 10, axis=1).shape, (None, 10, 3)
        )

    def test_maximum(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.maximum(x, y).shape, (2, 3))

    def test_minimum(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.minimum(x, y).shape, (2, 3))

    def test_mod(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.mod(x, y).shape, (2, 3))

    def test_nextafter(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((1, 3))
        self.assertEqual(knp.nextafter(x, y).shape, (None, 3))

    def test_not_equal(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.not_equal(x, y).shape, (2, 3))

    def test_outer(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.outer(x, y).shape, (None, None))

    def test_quantile(self):
        x = KerasTensor((None, 3))

        # q as scalar
        q = KerasTensor(())
        self.assertEqual(knp.quantile(x, q).shape, ())

        # q as 1D tensor
        q = KerasTensor((2,))
        self.assertEqual(knp.quantile(x, q).shape, (2,))
        self.assertEqual(knp.quantile(x, q, axis=1).shape, (2, None))
        self.assertEqual(
            knp.quantile(x, q, axis=1, keepdims=True).shape,
            (2, None, 1),
        )

    def test_searchsorted(self):
        a = KerasTensor((None,))
        v = KerasTensor((2, 3))

        output = knp.searchsorted(a, v)
        self.assertEqual(output.shape, v.shape)
        self.assertEqual(output.dtype, "int64")

    def test_take(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.take(x, 1).shape, ())
        self.assertEqual(knp.take(x, [1, 2]).shape, (2,))
        self.assertEqual(
            knp.take(x, [[1, 2], [1, 2]], axis=1).shape, (None, 2, 2)
        )

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.take(x, 1, axis=1).shape, (None, 3))
        self.assertEqual(knp.take(x, [1, 2]).shape, (2,))
        self.assertEqual(
            knp.take(x, [[1, 2], [1, 2]], axis=1).shape, (None, 2, 2, 3)
        )

        # test with negative axis
        self.assertEqual(knp.take(x, 1, axis=-2).shape, (None, 3))

        # test with multi-dimensional indices
        x = KerasTensor((None, 3, None, 5))
        indices = KerasTensor((6, 7))
        self.assertEqual(knp.take(x, indices, axis=2).shape, (None, 3, 6, 7, 5))

    def test_take_along_axis(self):
        x = KerasTensor((None, 3))
        indices = KerasTensor((1, 3))
        self.assertEqual(knp.take_along_axis(x, indices, axis=0).shape, (1, 3))
        self.assertEqual(
            knp.take_along_axis(x, indices, axis=1).shape, (None, 3)
        )

        x = KerasTensor((None, 3, 3))
        indices = KerasTensor((1, 3, None))
        self.assertEqual(
            knp.take_along_axis(x, indices, axis=1).shape, (None, 3, 3)
        )

    def test_tensordot(self):
        x = KerasTensor((None, 3, 4))
        y = KerasTensor((3, 4))
        self.assertEqual(knp.tensordot(x, y, axes=1).shape, (None, 3, 4))
        self.assertEqual(knp.tensordot(x, y, axes=[[0, 1], [1, 0]]).shape, (4,))

    def test_vdot(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((None, 3))
        self.assertEqual(knp.vdot(x, y).shape, ())

        x = KerasTensor((None, 3, 3))
        y = KerasTensor((None, 3, 3))
        self.assertEqual(knp.vdot(x, y).shape, ())

    def test_inner(self):
        x = KerasTensor((None,))
        y = KerasTensor((3,))
        self.assertEqual(knp.inner(x, y).shape, ())

    def test_where(self):
        condition = KerasTensor((2, None, 1))
        x = KerasTensor((None, 1))
        y = KerasTensor((None, 3))
        self.assertEqual(knp.where(condition, x, y).shape, (2, None, 3))
        self.assertEqual(knp.where(condition).shape, (2, None, 1))

    def test_floor_divide(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.floor_divide(x, y).shape, (2, 3))

    def test_xor(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.logical_xor(x, y).shape, (2, 3))

    def test_shape_equal_basic_equality(self):
        x = KerasTensor((3, 4)).shape
        y = KerasTensor((3, 4)).shape
        self.assertTrue(knp.shape_equal(x, y))
        y = KerasTensor((3, 5)).shape
        self.assertFalse(knp.shape_equal(x, y))

    def test_shape_equal_allow_none(self):
        x = KerasTensor((3, 4, None)).shape
        y = KerasTensor((3, 4, 5)).shape
        self.assertTrue(knp.shape_equal(x, y, allow_none=True))
        self.assertFalse(knp.shape_equal(x, y, allow_none=False))

    def test_shape_equal_different_shape_lengths(self):
        x = KerasTensor((3, 4)).shape
        y = KerasTensor((3, 4, 5)).shape
        self.assertFalse(knp.shape_equal(x, y))

    def test_shape_equal_ignore_axes(self):
        x = KerasTensor((3, 4, 5)).shape
        y = KerasTensor((3, 6, 5)).shape
        self.assertTrue(knp.shape_equal(x, y, axis=1))
        y = KerasTensor((3, 6, 7)).shape
        self.assertTrue(knp.shape_equal(x, y, axis=(1, 2)))
        self.assertFalse(knp.shape_equal(x, y, axis=1))

    def test_shape_equal_only_none(self):
        x = KerasTensor((None, None)).shape
        y = KerasTensor((5, 6)).shape
        self.assertTrue(knp.shape_equal(x, y, allow_none=True))

    def test_shape_equal_axis_as_list(self):
        x = KerasTensor((3, 4, 5)).shape
        y = KerasTensor((3, 6, 5)).shape
        self.assertTrue(knp.shape_equal(x, y, axis=[1]))

    def test_shape_non_equal_with_negative_axis(self):
        x = KerasTensor((3, 4, 5)).shape
        y = KerasTensor((3, 4, 6)).shape
        self.assertFalse(knp.shape_equal(x, y, axis=-2))

    def test_shape_equal_with_negative_axis(self):
        x = KerasTensor((3, 4, 5)).shape
        y = KerasTensor((3, 4, 5)).shape
        self.assertTrue(knp.shape_equal(x, y, axis=-1))

    def test_shape_equal_zeros(self):
        x = KerasTensor((0, 4)).shape
        y = KerasTensor((0, 4)).shape
        self.assertTrue(knp.shape_equal(x, y))
        y = KerasTensor((0, 5)).shape
        self.assertFalse(knp.shape_equal(x, y))

    def test_broadcast_shapes_conversion_to_list(self):
        shape1 = KerasTensor((1, 2)).shape
        shape2 = KerasTensor((3, 1)).shape
        expected_output = [3, 2]
        self.assertEqual(knp.broadcast_shapes(shape1, shape2), expected_output)

    def test_broadcast_shapes_shape1_longer_than_shape2(self):
        shape1 = KerasTensor((5, 3, 2)).shape
        shape2 = KerasTensor((1, 3)).shape
        with self.assertRaisesRegex(ValueError, "Cannot broadcast shape"):
            knp.broadcast_shapes(shape1, shape2)

    def test_broadcast_shapes_shape2_longer_than_shape1(self):
        shape1 = KerasTensor((5, 3)).shape
        shape2 = KerasTensor((2, 5, 3)).shape
        expected_output = [2, 5, 3]
        self.assertEqual(knp.broadcast_shapes(shape1, shape2), expected_output)

    def test_broadcast_shapes_broadcasting_shape1_is_1(self):
        shape1 = KerasTensor((1, 3)).shape
        shape2 = KerasTensor((5, 1)).shape
        expected_output = [5, 3]
        self.assertEqual(knp.broadcast_shapes(shape1, shape2), expected_output)

    def test_broadcast_shapes_broadcasting_shape1_is_none(self):
        shape1 = KerasTensor((None, 3)).shape
        shape2 = KerasTensor((5, 1)).shape
        expected_output = [5, 3]
        self.assertEqual(knp.broadcast_shapes(shape1, shape2), expected_output)

        shape1 = KerasTensor((None, 3)).shape
        shape2 = KerasTensor((5, 3)).shape
        expected_output = [5, 3]
        self.assertEqual(knp.broadcast_shapes(shape1, shape2), expected_output)

    def test_broadcast_shapes_broadcasting_shape2_conditions(self):
        shape1 = KerasTensor((5, 3, 2)).shape
        shape2 = KerasTensor((1, 3, 2)).shape
        expected_output = [5, 3, 2]
        self.assertEqual(knp.broadcast_shapes(shape1, shape2), expected_output)

        shape1 = KerasTensor((5, 3, 2)).shape
        shape2 = KerasTensor((1, None, 2)).shape
        expected_output = [5, 3, 2]
        self.assertEqual(knp.broadcast_shapes(shape1, shape2), expected_output)


class NumpyTwoInputOpsStaticShapeTest(testing.TestCase):
    def test_add(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.add(x, y).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.add(x, y)

    def test_heaviside(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.heaviside(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        y = KerasTensor((3,))
        self.assertEqual(knp.heaviside(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        y = KerasTensor((1, 3))
        self.assertEqual(knp.heaviside(x, y).shape, (2, 3))

    def test_hypot(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.hypot(x, y).shape, (2, 3))

    def test_subtract(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.subtract(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.subtract(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.subtract(x, y)

    def test_multiply(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.multiply(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.multiply(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.multiply(x, y)

    def test_matmul(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((3, 2))
        self.assertEqual(knp.matmul(x, y).shape, (2, 2))

        with self.assertRaises(ValueError):
            x = KerasTensor((3, 4))
            y = KerasTensor((2, 3, 4))
            knp.matmul(x, y)

    def test_matmul_sparse(self):
        x = KerasTensor((2, 3), sparse=True)
        y = KerasTensor((3, 2))
        result = knp.matmul(x, y)
        self.assertEqual(result.shape, (2, 2))

        x = KerasTensor((2, 3))
        y = KerasTensor((3, 2), sparse=True)
        result = knp.matmul(x, y)
        self.assertEqual(result.shape, (2, 2))

        x = KerasTensor((2, 3), sparse=True)
        y = KerasTensor((3, 2), sparse=True)
        result = knp.matmul(x, y)
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(result.sparse)

    def test_power(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.power(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.power(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.power(x, y)

    def test_divide(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.divide(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.divide(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.divide(x, y)

    def test_divide_no_nan(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.divide_no_nan(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.divide_no_nan(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.divide_no_nan(x, y)

    def test_true_divide(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.true_divide(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.true_divide(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.true_divide(x, y)

    def test_append(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.append(x, y).shape, (12,))

        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.append(x, y, axis=0).shape, (4, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.append(x, y, axis=2)

    def test_arctan2(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.arctan2(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.arctan2(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.arctan2(x, y)

    def test_bitwise_and(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.bitwise_and(x, y).shape, (2, 3))

    def test_bitwise_or(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.bitwise_or(x, y).shape, (2, 3))

    def test_bitwise_xor(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.bitwise_xor(x, y).shape, (2, 3))

    def test_bitwise_left_shift(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.bitwise_left_shift(x, y).shape, (2, 3))

    # left_shift is same as bitwise_left_shift

    def test_bitwise_right_shift(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.bitwise_right_shift(x, y).shape, (2, 3))

    # right_shift is same as bitwise_right_shift

    def test_cross(self):
        x1 = KerasTensor((2, 3, 3))
        x2 = KerasTensor((1, 3, 2))
        y1 = KerasTensor((2, 3, 3))
        y2 = KerasTensor((2, 3, 2))
        self.assertEqual(knp.cross(x1, y1).shape, (2, 3, 3))
        self.assertEqual(knp.cross(x2, y2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.cross(x, y)

        with self.assertRaises(ValueError):
            x = KerasTensor((4, 3, 3))
            y = KerasTensor((2, 3, 3))
            knp.cross(x, y)

    def test_einsum(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((3, 4))
        self.assertEqual(knp.einsum("ij,jk->ik", x, y).shape, (2, 4))
        self.assertEqual(knp.einsum("ij,jk->ikj", x, y).shape, (2, 4, 3))
        self.assertEqual(knp.einsum("ii", x).shape, ())
        self.assertEqual(knp.einsum(",ij", 5, x).shape, (2, 3))

        x = KerasTensor((2, 3, 4))
        y = KerasTensor((3, 4, 5))
        z = KerasTensor((1, 1, 1, 9))
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
            x = KerasTensor((2, 3))
            y = KerasTensor((3, 4))
            knp.einsum("ijk,jk->ik", x, y)

    def test_full_like(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.full_like(x, 2).shape, (2, 3))

    def test_gcd(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.gcd(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.gcd(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.gcd(x, y)

    def test_geomspace(self):
        start = KerasTensor((2, 3, 4))
        stop = KerasTensor((2, 3, 4))
        self.assertEqual(knp.geomspace(start, stop, 10).shape, (10, 2, 3, 4))

        with self.assertRaises(ValueError):
            start = KerasTensor((2, 3))
            stop = KerasTensor((2, 3, 4))
            knp.geomspace(start, stop)

    def test_greater(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.greater(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.greater(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.greater(x, y)

    def test_greater_equal(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.greater_equal(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.greater_equal(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.greater_equal(x, y)

    def test_allclose(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.allclose(x, y).shape, ())

        x = KerasTensor((2, 3))
        self.assertEqual(knp.allclose(x, 2).shape, ())

    def test_isclose(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.isclose(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.isclose(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.isclose(x, y)

    def test_isin(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((3, 3))
        self.assertEqual(knp.isin(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.isin(x, 2).shape, (2, 3))

    def test_kron(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.kron(x, y).shape, (4, 9))

    def test_lcm(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.lcm(x, y).shape, (2, 3))

    def test_ldexp(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.ldexp(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        y = KerasTensor((1, 3))
        self.assertEqual(knp.ldexp(x, y).shape, (2, 3))

    def test_less(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.less(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.less(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.less(x, y)

    def test_less_equal(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.less_equal(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.less_equal(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.less_equal(x, y)

    def test_linspace(self):
        start = KerasTensor((2, 3, 4))
        stop = KerasTensor((2, 3, 4))
        self.assertEqual(knp.linspace(start, stop, 10).shape, (10, 2, 3, 4))

        with self.assertRaises(ValueError):
            start = KerasTensor((2, 3))
            stop = KerasTensor((2, 3, 4))
            knp.linspace(start, stop)

    def test_logical_and(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.logical_and(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.logical_and(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.logical_and(x, y)

    def test_logical_or(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.logical_or(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.logical_or(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.logical_or(x, y)

    def test_logspace(self):
        start = KerasTensor((2, 3, 4))
        stop = KerasTensor((2, 3, 4))
        self.assertEqual(knp.logspace(start, stop, 10).shape, (10, 2, 3, 4))

        with self.assertRaises(ValueError):
            start = KerasTensor((2, 3))
            stop = KerasTensor((2, 3, 4))
            knp.logspace(start, stop)

    def test_maximum(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.maximum(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.maximum(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.maximum(x, y)

    def test_minimum(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.minimum(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.minimum(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.minimum(x, y)

    def test_mod(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.mod(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.mod(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.mod(x, y)

    def test_nextafter(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.nextafter(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        y = KerasTensor((1, 3))
        self.assertEqual(knp.nextafter(x, y).shape, (2, 3))

    def test_not_equal(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.not_equal(x, y).shape, (2, 3))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.not_equal(x, 2).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.not_equal(x, y)

    def test_outer(self):
        x = KerasTensor((3,))
        y = KerasTensor((4,))
        self.assertEqual(knp.outer(x, y).shape, (3, 4))

        x = KerasTensor((2, 3))
        y = KerasTensor((4, 5))
        self.assertEqual(knp.outer(x, y).shape, (6, 20))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.outer(x, 2).shape, (6, 1))

    def test_quantile(self):
        x = KerasTensor((3, 3))

        # q as scalar
        q = KerasTensor(())
        self.assertEqual(knp.quantile(x, q).shape, ())

        # q as 1D tensor
        q = KerasTensor((2,))
        self.assertEqual(knp.quantile(x, q).shape, (2,))
        self.assertEqual(knp.quantile(x, q, axis=1).shape, (2, 3))
        self.assertEqual(
            knp.quantile(x, q, axis=1, keepdims=True).shape,
            (2, 3, 1),
        )

    def test_searchsorted(self):
        a = KerasTensor((3,))
        v = KerasTensor((2, 3))

        self.assertEqual(knp.searchsorted(a, v).shape, v.shape)

    def test_take(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.take(x, 1).shape, ())
        self.assertEqual(knp.take(x, [1, 2]).shape, (2,))
        self.assertEqual(knp.take(x, [[1, 2], [1, 2]], axis=1).shape, (2, 2, 2))

        # test with multi-dimensional indices
        x = KerasTensor((2, 3, 4, 5))
        indices = KerasTensor((6, 7))
        self.assertEqual(knp.take(x, indices, axis=2).shape, (2, 3, 6, 7, 5))

    def test_take_along_axis(self):
        x = KerasTensor((2, 3))
        indices = KerasTensor((1, 3))
        self.assertEqual(knp.take_along_axis(x, indices, axis=0).shape, (1, 3))
        self.assertEqual(knp.take_along_axis(x, indices, axis=1).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            indices = KerasTensor((1, 4))
            knp.take_along_axis(x, indices, axis=0)

    def test_tensordot(self):
        x = KerasTensor((2, 3, 3))
        y = KerasTensor((3, 3, 4))
        self.assertEqual(knp.tensordot(x, y, axes=1).shape, (2, 3, 3, 4))
        self.assertEqual(knp.tensordot(x, y, axes=2).shape, (2, 4))
        self.assertEqual(
            knp.tensordot(x, y, axes=[[1, 2], [0, 1]]).shape, (2, 4)
        )

    def test_vdot(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.vdot(x, y).shape, ())

    def test_inner(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.inner(x, y).shape, ())

    def test_where(self):
        condition = KerasTensor((2, 3))
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.where(condition, x, y).shape, (2, 3))
        self.assertAllEqual(knp.where(condition).shape, (2, 3))

    def test_floor_divide(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.floor_divide(x, y).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.floor_divide(x, y)

    def test_xor(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.logical_xor(x, y).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3, 4))
            knp.logical_xor(x, y)

    def test_digitize(self):
        x = KerasTensor((2, 3))
        bins = KerasTensor((3,))
        self.assertEqual(knp.digitize(x, bins).shape, (2, 3))
        self.assertTrue(knp.digitize(x, bins).dtype == "int32")

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            bins = KerasTensor((2, 3, 4))
            knp.digitize(x, bins)

    def test_correlate_mode_valid(self):
        x = KerasTensor((3,))
        y = KerasTensor((3,))
        self.assertEqual(knp.correlate(x, y).shape, (1,))
        self.assertTrue(knp.correlate(x, y).dtype == "float32")

        with self.assertRaises(ValueError):
            x = KerasTensor((3,))
            y = KerasTensor((3, 4))
            knp.correlate(x, y)

    def test_correlate_mode_same(self):
        x = KerasTensor((3,))
        y = KerasTensor((3,))
        self.assertEqual(knp.correlate(x, y, mode="same").shape, (3,))
        self.assertTrue(knp.correlate(x, y, mode="same").dtype == "float32")

        with self.assertRaises(ValueError):
            x = KerasTensor((3,))
            y = KerasTensor((3, 4))
            knp.correlate(x, y, mode="same")

    def test_correlate_mode_full(self):
        x = KerasTensor((3,))
        y = KerasTensor((3,))
        self.assertEqual(knp.correlate(x, y, mode="full").shape, (5,))
        self.assertTrue(knp.correlate(x, y, mode="full").dtype == "float32")

        with self.assertRaises(ValueError):
            x = KerasTensor((3))
            y = KerasTensor((3, 4))
            knp.correlate(x, y, mode="full")


class NumpyOneInputOpsDynamicShapeTest(testing.TestCase):
    def test_mean(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.mean(x).shape, ())

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.mean(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.mean(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_all(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.all(x).shape, ())

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.all(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.all(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_any(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.any(x).shape, ())

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.any(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.any(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_trapezoid(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.trapezoid(x).shape, (None,))

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.trapezoid(x, axis=1).shape, (None, 3))

    def test_vander(self):
        x = KerasTensor((None,))
        self.assertEqual(knp.vander(x).shape, (None, None))

    def test_var(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.var(x).shape, ())

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.var(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.var(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_sum(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.sum(x).shape, ())

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.sum(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.sum(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_amax(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.amax(x).shape, ())

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.amax(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.amax(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_amin(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.amin(x).shape, ())

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.amin(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.amin(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_square(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.square(x).shape, (None, 3))

    def test_negative(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.negative(x).shape, (None, 3))

    def test_abs(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.abs(x).shape, (None, 3))

    def test_absolute(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.absolute(x).shape, (None, 3))

    def test_squeeze(self):
        x = KerasTensor((None, 1))
        self.assertEqual(knp.squeeze(x).shape, (None,))
        self.assertEqual(knp.squeeze(x, axis=1).shape, (None,))

        with self.assertRaises(ValueError):
            x = KerasTensor((None, 1))
            knp.squeeze(x, axis=0)

        # Multiple axes
        x = KerasTensor((None, 1, 1, 1))
        self.assertEqual(knp.squeeze(x, (1, 2)).shape, (None, 1))
        self.assertEqual(knp.squeeze(x, (-1, -2)).shape, (None, 1))
        self.assertEqual(knp.squeeze(x, (1, 2, 3)).shape, (None,))
        self.assertEqual(knp.squeeze(x, (-1, 1)).shape, (None, 1))

    def test_transpose(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.transpose(x).shape, (3, None))

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.transpose(x, (2, 0, 1)).shape, (3, None, 3))

    def test_arccos(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.arccos(x).shape, (None, 3))

    def test_arccosh(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.arccosh(x).shape, (None, 3))

    def test_arcsin(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.arcsin(x).shape, (None, 3))

    def test_arcsinh(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.arcsinh(x).shape, (None, 3))

    def test_arctan(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.arctan(x).shape, (None, 3))

    def test_arctanh(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.arctanh(x).shape, (None, 3))

    def test_argmax(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.argmax(x).shape, ())
        self.assertEqual(knp.argmax(x, keepdims=True).shape, (None, 3))

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.argmax(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.argmax(x, keepdims=True).shape, (None, 3, 3))

    @pytest.mark.skipif(
        keras.config.backend() == "openvino" or testing.jax_uses_tpu(),
        reason="OpenVINO and JAX TPU don't support this",
    )
    def test_argmax_negative_zero(self):
        input_data = np.array(
            [-1.0, -0.0, 1.401298464324817e-45], dtype=np.float32
        )
        self.assertEqual(knp.argmax(input_data), 2)

    @pytest.mark.skipif(
        keras.config.backend() == "openvino" or testing.jax_uses_tpu(),
        reason="OpenVINO and JAX TPU don't support this",
    )
    def test_argmin_negative_zero(self):
        input_data = np.array(
            [
                0.0,
                1.1754943508222875e-38,
                -1.401298464324817e-45,
                0.0,
                459367.0,
            ],
            dtype=np.float32,
        )
        self.assertEqual(knp.argmin(input_data), 2)

    def test_argmin(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.argmin(x).shape, ())
        self.assertEqual(knp.argmin(x, keepdims=True).shape, (None, 3))

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.argmin(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.argmin(x, keepdims=True).shape, (None, 3, 3))

    def test_argsort(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.argsort(x).shape, (None, 3))

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.argsort(x, axis=1).shape, (None, 3, 3))

    def test_array(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.array(x).shape, (None, 3))

    def test_average(self):
        x = KerasTensor((None, 3))
        weights = KerasTensor((None, 3))
        self.assertEqual(knp.average(x, weights=weights).shape, ())

        x = KerasTensor((None, 3))
        weights = KerasTensor((3,))
        self.assertEqual(knp.average(x, axis=1, weights=weights).shape, (None,))

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.average(x, axis=1).shape, (None, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((None, 3, 3))
            weights = KerasTensor((None, 4))
            knp.average(x, weights=weights)

    def test_bartlett(self):
        x = np.random.randint(1, 100 + 1)
        self.assertEqual(knp.bartlett(x).shape[0], x)

    def test_blackman(self):
        x = np.random.randint(1, 100 + 1)
        self.assertEqual(knp.blackman(x).shape[0], x)

    def test_hamming(self):
        x = np.random.randint(1, 100 + 1)
        self.assertEqual(knp.hamming(x).shape[0], x)

    def test_hanning(self):
        x = np.random.randint(1, 100 + 1)
        self.assertEqual(knp.hanning(x).shape[0], x)

    def test_kaiser(self):
        x = np.random.randint(1, 100 + 1)
        beta = float(np.random.randint(10, 20 + 1))
        self.assertEqual(knp.kaiser(x, beta).shape[0], x)

    def test_bitwise_invert(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.bitwise_invert(x).shape, (None, 3))

    # bitwise_not is same as bitwise_invert

    def test_broadcast_to(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.broadcast_to(x, (2, 3, 3)).shape, (2, 3, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((3, 3))
            knp.broadcast_to(x, (2, 2, 3))

    def test_cbrt(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.cbrt(x).shape, (None, 3))

    def test_ceil(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.ceil(x).shape, (None, 3))

    def test_clip(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.clip(x, 1, 2).shape, (None, 3))

    def test_concatenate(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((None, 3))
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
            x = KerasTensor((None, 3, 5))
            y = KerasTensor((None, 4, 6))
            knp.concatenate([x, y], axis=1)

    def test_concatenate_sparse(self):
        x = KerasTensor((2, 3), sparse=True)
        y = KerasTensor((2, 3))
        result = knp.concatenate([x, y], axis=1)
        self.assertEqual(result.shape, (2, 6))
        self.assertFalse(result.sparse)

        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3), sparse=True)
        result = knp.concatenate([x, y], axis=1)
        self.assertEqual(result.shape, (2, 6))
        self.assertFalse(result.sparse)

        x = KerasTensor((2, 3), sparse=True)
        y = KerasTensor((2, 3), sparse=True)
        result = knp.concatenate([x, y], axis=1)
        self.assertEqual(result.shape, (2, 6))
        self.assertTrue(result.sparse)

    def test_conjugate(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.conjugate(x).shape, (None, 3))

    def test_conj(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.conj(x).shape, (None, 3))

    def test_copy(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.copy(x).shape, (None, 3))

    def test_corrcoef(self):
        x = KerasTensor((3, None))
        self.assertEqual(knp.corrcoef(x).shape, (3, None))

    def test_cos(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.cos(x).shape, (None, 3))

    def test_cosh(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.cosh(x).shape, (None, 3))

    def test_count_nonzero(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.count_nonzero(x).shape, ())

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.count_nonzero(x, axis=1).shape, (None, 3))

    def test_cumprod(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.cumprod(x).shape, (None,))

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.cumprod(x, axis=1).shape, (None, 3, 3))

    def test_cumsum(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.cumsum(x).shape, (None,))

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.cumsum(x, axis=1).shape, (None, 3, 3))

    def test_deg2rad(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.deg2rad(x).shape, (None, 3))

    def test_diag(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.diag(x).shape, (None,))
        self.assertEqual(knp.diag(x, k=3).shape, (None,))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3, 4))
            knp.diag(x)

    def test_diagflat(self):
        x = KerasTensor((3,))
        self.assertEqual(knp.diagflat(x).shape, (3, 3))
        self.assertEqual(knp.diagflat(x, k=1).shape, (4, 4))
        self.assertEqual(knp.diagflat(x, k=-1).shape, (4, 4))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.diagflat(x).shape, (6, 6))
        self.assertEqual(knp.diagflat(x, k=2).shape, (8, 8))

        x = KerasTensor((None, 3))
        self.assertEqual(knp.diagflat(x).shape, (None, None))

    def test_diagonal(self):
        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.diagonal(x).shape, (3, None))

    def test_diff(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.diff(x).shape, (None, 2))
        self.assertEqual(knp.diff(x, n=2).shape, (None, 1))
        self.assertEqual(knp.diff(x, n=3).shape, (None, 0))
        self.assertEqual(knp.diff(x, n=4).shape, (None, 0))

        self.assertEqual(knp.diff(x, axis=0).shape, (None, 3))
        self.assertEqual(knp.diff(x, n=2, axis=0).shape, (None, 3))

    def test_dot(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((3, 2))
        z = KerasTensor((None, None, 2))
        self.assertEqual(knp.dot(x, y).shape, (None, 2))
        self.assertEqual(knp.dot(x, 2).shape, (None, 3))
        self.assertEqual(knp.dot(x, z).shape, (None, None, 2))

        x = KerasTensor((None,))
        y = KerasTensor((5,))
        self.assertEqual(knp.dot(x, y).shape, ())

    def test_empty_like(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.empty_like(x).shape, (None, 3))
        self.assertEqual(knp.empty_like(x).dtype, x.dtype)

    def test_exp(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.exp(x).shape, (None, 3))

    def test_exp2(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.exp2(x).shape, (None, 3))

    def test_expand_dims(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.expand_dims(x, -1).shape, (None, 3, 1))
        self.assertEqual(knp.expand_dims(x, 0).shape, (1, None, 3))
        self.assertEqual(knp.expand_dims(x, 1).shape, (None, 1, 3))
        self.assertEqual(knp.expand_dims(x, -2).shape, (None, 1, 3))

        # Multiple axes
        self.assertEqual(knp.expand_dims(x, (1, 2)).shape, (None, 1, 1, 3))
        self.assertEqual(knp.expand_dims(x, (-1, -2)).shape, (None, 3, 1, 1))
        self.assertEqual(knp.expand_dims(x, (-1, 1)).shape, (None, 1, 3, 1))

    def test_expm1(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.expm1(x).shape, (None, 3))

    def test_flip(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.flip(x).shape, (None, 3))

    def test_floor(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.floor(x).shape, (None, 3))

    def test_get_item(self):
        x = KerasTensor((None, 5, 16))
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
        x = KerasTensor((None, 3))
        y = KerasTensor((None, 3))
        self.assertEqual(knp.hstack([x, y]).shape, (None, 6))

        x = KerasTensor((None, 3))
        y = KerasTensor((None, None))
        self.assertEqual(knp.hstack([x, y]).shape, (None, None))

    def test_imag(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.imag(x).shape, (None, 3))

    def test_isfinite(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.isfinite(x).shape, (None, 3))

    def test_isinf(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.isinf(x).shape, (None, 3))

    def test_isnan(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.isnan(x).shape, (None, 3))

    def test_isneginf(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.isneginf(x).shape, (None, 3))

    def test_isposinf(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.isposinf(x).shape, (None, 3))

    def test_isreal(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.isreal(x).shape, (None, 3))

    def test_log(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.log(x).shape, (None, 3))

    def test_log10(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.log10(x).shape, (None, 3))

    def test_log1p(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.log1p(x).shape, (None, 3))

    def test_log2(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.log2(x).shape, (None, 3))

    def test_logaddexp(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.logaddexp(x, x).shape, (None, 3))

    def test_logaddexp2(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.logaddexp2(x, x).shape, (None, 3))

    def test_logical_not(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.logical_not(x).shape, (None, 3))

    def test_max(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.max(x).shape, ())

    def test_median(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.median(x).shape, ())

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.median(x, axis=1).shape, (None, 3))
        self.assertEqual(
            knp.median(x, axis=1, keepdims=True).shape, (None, 1, 3)
        )

    def test_meshgrid(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((None, 3))
        self.assertEqual(knp.meshgrid(x, y)[0].shape, (None, None))
        self.assertEqual(knp.meshgrid(x, y)[1].shape, (None, None))

        with self.assertRaises(ValueError):
            knp.meshgrid(x, y, indexing="kk")

    def test_moveaxis(self):
        x = KerasTensor((None, 3, 4, 5))
        self.assertEqual(knp.moveaxis(x, 0, -1).shape, (3, 4, 5, None))
        self.assertEqual(knp.moveaxis(x, -1, 0).shape, (5, None, 3, 4))
        self.assertEqual(
            knp.moveaxis(x, [0, 1], [-1, -2]).shape, (4, 5, 3, None)
        )
        self.assertEqual(knp.moveaxis(x, [0, 1], [1, 0]).shape, (3, None, 4, 5))
        self.assertEqual(
            knp.moveaxis(x, [0, 1], [-2, -1]).shape, (4, 5, None, 3)
        )

    def test_nanmax(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.nanmax(x).shape, ())

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.nanmax(x, axis=1).shape, (None, 3))
        self.assertEqual(
            knp.nanmax(x, axis=1, keepdims=True).shape, (None, 1, 3)
        )

        self.assertEqual(knp.nanmax(x, axis=(1,)).shape, (None, 3))

        self.assertEqual(knp.nanmax(x, axis=(1, 2)).shape, (None,))
        self.assertEqual(
            knp.nanmax(x, axis=(1, 2), keepdims=True).shape, (None, 1, 1)
        )

        self.assertEqual(knp.nanmax(x, axis=()).shape, (None, 3, 3))

        x4 = KerasTensor((None, 2, 3, 4))
        self.assertEqual(knp.nanmax(x4, axis=2).shape, (None, 2, 4))
        self.assertEqual(knp.nanmax(x4, axis=(1, 3)).shape, (None, 3))

    def test_nanmean(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.nanmean(x).shape, ())

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.nanmean(x, axis=1).shape, (None, 3))
        self.assertEqual(
            knp.nanmean(x, axis=1, keepdims=True).shape, (None, 1, 3)
        )

        self.assertEqual(knp.nanmean(x, axis=(1,)).shape, (None, 3))

        self.assertEqual(knp.nanmean(x, axis=(1, 2)).shape, (None,))
        self.assertEqual(
            knp.nanmean(x, axis=(1, 2), keepdims=True).shape, (None, 1, 1)
        )

        self.assertEqual(knp.nanmean(x, axis=()).shape, (None, 3, 3))

        x4 = KerasTensor((None, 2, 3, 4))
        self.assertEqual(knp.nanmean(x4, axis=2).shape, (None, 2, 4))
        self.assertEqual(knp.nanmean(x4, axis=(1, 3)).shape, (None, 3))

    def test_nanmin(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.nanmin(x).shape, ())

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.nanmin(x, axis=1).shape, (None, 3))
        self.assertEqual(
            knp.nanmin(x, axis=1, keepdims=True).shape, (None, 1, 3)
        )

        self.assertEqual(knp.nanmin(x, axis=(1,)).shape, (None, 3))

        self.assertEqual(knp.nanmin(x, axis=(1, 2)).shape, (None,))
        self.assertEqual(
            knp.nanmin(x, axis=(1, 2), keepdims=True).shape, (None, 1, 1)
        )

        self.assertEqual(knp.nanmin(x, axis=()).shape, (None, 3, 3))

        x4 = KerasTensor((None, 2, 3, 4))
        self.assertEqual(knp.nanmin(x4, axis=2).shape, (None, 2, 4))
        self.assertEqual(knp.nanmin(x4, axis=(1, 3)).shape, (None, 3))

    def test_nanprod(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.nanprod(x).shape, ())

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.nanprod(x, axis=1).shape, (None, 3))
        self.assertEqual(
            knp.nanprod(x, axis=1, keepdims=True).shape, (None, 1, 3)
        )

        self.assertEqual(knp.nanprod(x, axis=(1,)).shape, (None, 3))

        self.assertEqual(knp.nanprod(x, axis=(1, 2)).shape, (None,))
        self.assertEqual(
            knp.nanprod(x, axis=(1, 2), keepdims=True).shape, (None, 1, 1)
        )

        self.assertEqual(knp.nanprod(x, axis=()).shape, (None, 3, 3))

        x4 = KerasTensor((None, 2, 3, 4))
        self.assertEqual(knp.nanprod(x4, axis=2).shape, (None, 2, 4))
        self.assertEqual(knp.nanprod(x4, axis=(1, 3)).shape, (None, 3))

    def test_nanstd(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.nanstd(x).shape, ())

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.nanstd(x, axis=1).shape, (None, 3))
        self.assertEqual(
            knp.nanstd(x, axis=1, keepdims=True).shape, (None, 1, 3)
        )

        self.assertEqual(knp.nanstd(x, axis=(1,)).shape, (None, 3))

        self.assertEqual(knp.nanstd(x, axis=(1, 2)).shape, (None,))
        self.assertEqual(
            knp.nanstd(x, axis=(1, 2), keepdims=True).shape, (None, 1, 1)
        )

        self.assertEqual(knp.nanstd(x, axis=()).shape, (None, 3, 3))

        x4 = KerasTensor((None, 2, 3, 4))
        self.assertEqual(knp.nanstd(x4, axis=2).shape, (None, 2, 4))
        self.assertEqual(knp.nanstd(x4, axis=(1, 3)).shape, (None, 3))

    def test_nansum(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.nansum(x).shape, ())

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.nansum(x, axis=1).shape, (None, 3))
        self.assertEqual(
            knp.nansum(x, axis=1, keepdims=True).shape, (None, 1, 3)
        )

        self.assertEqual(knp.nansum(x, axis=(1,)).shape, (None, 3))

        self.assertEqual(knp.nansum(x, axis=(1, 2)).shape, (None,))
        self.assertEqual(
            knp.nansum(x, axis=(1, 2), keepdims=True).shape, (None, 1, 1)
        )

        self.assertEqual(knp.nansum(x, axis=()).shape, (None, 3, 3))

        x4 = KerasTensor((None, 2, 3, 4))
        self.assertEqual(knp.nansum(x4, axis=2).shape, (None, 2, 4))
        self.assertEqual(knp.nansum(x4, axis=(1, 3)).shape, (None, 3))

    def test_nanvar(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.nanvar(x).shape, ())

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.nanvar(x, axis=1).shape, (None, 3))
        self.assertEqual(
            knp.nanvar(x, axis=1, keepdims=True).shape, (None, 1, 3)
        )

        self.assertEqual(knp.nanvar(x, axis=(1,)).shape, (None, 3))

        self.assertEqual(knp.nanvar(x, axis=(1, 2)).shape, (None,))
        self.assertEqual(
            knp.nanvar(x, axis=(1, 2), keepdims=True).shape, (None, 1, 1)
        )

        self.assertEqual(knp.nanvar(x, axis=()).shape, (None, 3, 3))

        x4 = KerasTensor((None, 2, 3, 4))
        self.assertEqual(knp.nanvar(x4, axis=2).shape, (None, 2, 4))
        self.assertEqual(knp.nanvar(x4, axis=(1, 3)).shape, (None, 3))

    def test_ndim(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.ndim(x).shape, (2,))

    def test_nonzero(self):
        x = KerasTensor((None, 5, 6))
        result = knp.nonzero(x)
        self.assertLen(result, 3)
        self.assertEqual(result[0].shape, (None,))
        self.assertEqual(result[1].shape, (None,))
        self.assertEqual(result[2].shape, (None,))

    def test_ones_like(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.ones_like(x).shape, (None, 3))
        self.assertEqual(knp.ones_like(x).dtype, x.dtype)

    def test_zeros_like(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.zeros_like(x).shape, (None, 3))
        self.assertEqual(knp.zeros_like(x).dtype, x.dtype)

    def test_pad(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.pad(x, 1).shape, (None, 5))
        self.assertEqual(knp.pad(x, (1, 2)).shape, (None, 6))
        self.assertEqual(knp.pad(x, ((1, 2), (3, 4))).shape, (None, 10))

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.pad(x, 1).shape, (None, 5, 5))
        self.assertEqual(knp.pad(x, (1, 2)).shape, (None, 6, 6))
        self.assertEqual(
            knp.pad(x, ((1, 2), (3, 4), (5, 6))).shape, (None, 10, 14)
        )

    def test_prod(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.prod(x).shape, ())
        self.assertEqual(knp.prod(x, axis=0).shape, (3,))
        self.assertEqual(knp.prod(x, axis=1, keepdims=True).shape, (None, 1))

    def test_ptp(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.ptp(x).shape, ())
        self.assertEqual(knp.ptp(x, axis=0).shape, (3,))
        self.assertEqual(knp.ptp(x, axis=1, keepdims=True).shape, (None, 1))

    def test_ravel(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.ravel(x).shape, (None,))

    def test_unravel_index(self):
        x = KerasTensor((None,))
        indices = knp.unravel_index(x, (2, 3))
        self.assertEqual(len(indices), 2)
        self.assertEqual(indices[0].shape, (None,))
        self.assertEqual(indices[1].shape, (None,))

        x = KerasTensor((None, 4))
        indices = knp.unravel_index(x, (3, 4))
        self.assertEqual(len(indices), 2)
        self.assertEqual(indices[0].shape, (None, 4))
        self.assertEqual(indices[1].shape, (None, 4))

        x = KerasTensor((None, 3, 2))
        indices = knp.unravel_index(x, (5, 6, 4))
        self.assertEqual(len(indices), 3)
        self.assertEqual(indices[0].shape, (None, 3, 2))
        self.assertEqual(indices[1].shape, (None, 3, 2))
        self.assertEqual(indices[2].shape, (None, 3, 2))

    def test_real(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.real(x).shape, (None, 3))

    def test_reciprocal(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.reciprocal(x).shape, (None, 3))

    def test_repeat(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.repeat(x, 2).shape, (None,))
        self.assertEqual(knp.repeat(x, 3, axis=1).shape, (None, 9))
        self.assertEqual(knp.repeat(x, [1, 2], axis=0).shape, (None, 3))
        self.assertEqual(knp.repeat(x, 2, axis=0).shape, (None, 3))

    def test_reshape(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.reshape(x, (3, 2)).shape, (3, 2))
        self.assertEqual(knp.reshape(x, (3, -1)).shape, (3, None))

    def test_roll(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.roll(x, 1).shape, (None, 3))
        self.assertEqual(knp.roll(x, 1, axis=1).shape, (None, 3))
        self.assertEqual(knp.roll(x, 1, axis=0).shape, (None, 3))

    def test_round(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.round(x).shape, (None, 3))

    def test_sign(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.sign(x).shape, (None, 3))

    def test_signbit(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.signbit(x).shape, (None, 3))

    def test_sin(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.sin(x).shape, (None, 3))

    def test_sinh(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.sinh(x).shape, (None, 3))

    def test_size(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.size(x).shape, ())

    def test_sort(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.sort(x).shape, (None, 3))
        self.assertEqual(knp.sort(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.sort(x, axis=0).shape, (None, 3))

    def test_split(self):
        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.split(x, 2)[0].shape, (None, 3, 3))
        self.assertEqual(knp.split(x, 3, axis=1)[0].shape, (None, 1, 3))
        self.assertEqual(len(knp.split(x, [1, 3], axis=1)), 3)
        self.assertEqual(knp.split(x, [1, 3], axis=1)[0].shape, (None, 1, 3))
        self.assertEqual(knp.split(x, [1, 3], axis=1)[1].shape, (None, 2, 3))
        self.assertEqual(knp.split(x, [1, 3], axis=1)[2].shape, (None, 0, 3))

    def test_sqrt(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.sqrt(x).shape, (None, 3))

    def test_stack(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((None, 3))
        self.assertEqual(knp.stack([x, y]).shape, (2, None, 3))
        self.assertEqual(knp.stack([x, y], axis=-1).shape, (None, 3, 2))

    def test_std(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.std(x).shape, ())

        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.std(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.std(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_swapaxes(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.swapaxes(x, 0, 1).shape, (3, None))

    def test_tan(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.tan(x).shape, (None, 3))

    def test_tanh(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.tanh(x).shape, (None, 3))

    def test_tile(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.tile(x, 2).shape, (None, 6))
        self.assertEqual(knp.tile(x, [2]).shape, (None, 6))
        self.assertEqual(knp.tile(x, [1, 2]).shape, (None, 6))
        self.assertEqual(knp.tile(x, [2, 1, 2]).shape, (2, None, 6))

        # Test with multi-dimensional input
        x = KerasTensor((None, 3, 2, 2))
        self.assertEqual(knp.tile(x, [1, 2, 1, 1]).shape, (None, 6, 2, 2))

    def test_trace(self):
        x = KerasTensor((None, 3, None, 5))
        self.assertEqual(knp.trace(x).shape, (None, 5))
        self.assertEqual(knp.trace(x, axis1=2, axis2=3).shape, (None, 3))

    def test_tril(self):
        x = KerasTensor((None, 3, None, 5))
        self.assertEqual(knp.tril(x).shape, (None, 3, None, 5))
        self.assertEqual(knp.tril(x, k=1).shape, (None, 3, None, 5))
        self.assertEqual(knp.tril(x, k=-1).shape, (None, 3, None, 5))

    def test_triu(self):
        x = KerasTensor((None, 3, None, 5))
        self.assertEqual(knp.triu(x).shape, (None, 3, None, 5))
        self.assertEqual(knp.triu(x, k=1).shape, (None, 3, None, 5))
        self.assertEqual(knp.triu(x, k=-1).shape, (None, 3, None, 5))

    def test_trunc(self):
        x = KerasTensor((None, 3, None, 5))
        self.assertEqual(knp.trunc(x).shape, (None, 3, None, 5))

    def test_vstack(self):
        x = KerasTensor((None, 3))
        y = KerasTensor((None, 3))
        self.assertEqual(knp.vstack([x, y]).shape, (None, 3))

        x = KerasTensor((None, 3))
        y = KerasTensor((None, None))
        self.assertEqual(knp.vstack([x, y]).shape, (None, 3))

    def test_dstack(self):
        x = KerasTensor((None,))
        y = KerasTensor((None,))
        self.assertEqual(knp.dstack([x, y]).shape, (1, None, 2))

        x = KerasTensor((None, 3))
        y = KerasTensor((None, 3))
        self.assertEqual(knp.dstack([x, y]).shape, (None, 3, 2))

        x = KerasTensor((None, 3))
        y = KerasTensor((None, None))
        self.assertEqual(knp.dstack([x, y]).shape, (None, 3, 2))

    def test_hsplit(self):
        x = KerasTensor((3, None, 3))
        self.assertEqual(knp.hsplit(x, 2)[0].shape, (3, None, 3))
        self.assertEqual(len(knp.hsplit(x, [1, 3])), 3)
        self.assertEqual(knp.hsplit(x, [1, 3])[0].shape, (3, 1, 3))
        self.assertEqual(knp.hsplit(x, [1, 3])[1].shape, (3, 2, 3))
        self.assertEqual(knp.hsplit(x, [1, 3])[2].shape, (3, None, 3))

        # test 1D case
        x_1d = KerasTensor((None,))
        self.assertEqual(knp.hsplit(x_1d, 2)[0].shape, (None,))

        splits_1d = knp.hsplit(x_1d, [2, 5])
        self.assertEqual(splits_1d[0].shape, (2,))
        self.assertEqual(splits_1d[1].shape, (3,))
        self.assertEqual(splits_1d[2].shape, (None,))

    def test_vsplit(self):
        x = KerasTensor((None, 3, 3))
        self.assertEqual(knp.vsplit(x, 2)[0].shape, (None, 3, 3))
        self.assertEqual(len(knp.vsplit(x, [1, 3])), 3)
        self.assertEqual(knp.vsplit(x, [1, 3])[0].shape, (1, 3, 3))
        self.assertEqual(knp.vsplit(x, [1, 3])[1].shape, (2, 3, 3))
        self.assertEqual(knp.vsplit(x, [1, 3])[2].shape, (None, 3, 3))

    def test_argpartition(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.argpartition(x, 3).shape, (None, 3))
        self.assertEqual(knp.argpartition(x, 1, axis=1).shape, (None, 3))

        with self.assertRaises(ValueError):
            knp.argpartition(x, (1, 3))

    def test_angle(self):
        x = KerasTensor((None, 3))
        self.assertEqual(knp.angle(x).shape, (None, 3))

    def test_view(self):
        x = knp.array(KerasTensor((None, 3)), dtype="int32")
        self.assertEqual(knp.view(x, dtype="uint32").shape, (None, 3))
        self.assertEqual(knp.view(x, dtype="uint32").dtype, "uint32")
        x = knp.array(KerasTensor((None, 3)), dtype="int32")
        self.assertEqual(knp.view(x, dtype="int16").shape, (None, 6))
        self.assertEqual(knp.view(x, dtype="int16").dtype, "int16")
        x = knp.array(KerasTensor((None, 4)), dtype="int16")
        self.assertEqual(knp.view(x, dtype="int32").shape, (None, 2))
        self.assertEqual(knp.view(x, dtype="int32").dtype, "int32")

    def test_array_split(self):
        x = KerasTensor((None, 4))
        splits = knp.array_split(x, 2, axis=0)
        self.assertEqual(len(splits), 2)
        self.assertEqual(splits[0].shape, (None, 4))
        self.assertEqual(splits[1].shape, (None, 4))


class NumpyOneInputOpsStaticShapeTest(testing.TestCase):
    def test_mean(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.mean(x).shape, ())

    def test_all(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.all(x).shape, ())

    def test_any(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.any(x).shape, ())

    def test_trapezoid(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.trapezoid(x).shape, (2,))

    def test_vander(self):
        x = KerasTensor((2,))
        self.assertEqual(knp.vander(x).shape, (2, 2))

    def test_var(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.var(x).shape, ())

    def test_sum(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.sum(x).shape, ())

    def test_amax(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.amax(x).shape, ())

    def test_amin(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.amin(x).shape, ())

    def test_square(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.square(x).shape, (2, 3))

    def test_negative(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.negative(x).shape, (2, 3))

    def test_abs(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.abs(x).shape, (2, 3))

    def test_absolute(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.absolute(x).shape, (2, 3))

    def test_squeeze(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.squeeze(x).shape, (2, 3))

        x = KerasTensor((2, 1, 3))
        self.assertEqual(knp.squeeze(x).shape, (2, 3))
        self.assertEqual(knp.squeeze(x, axis=1).shape, (2, 3))
        self.assertEqual(knp.squeeze(x, axis=-2).shape, (2, 3))

        with self.assertRaises(ValueError):
            knp.squeeze(x, axis=0)

        # Multiple axes
        x = KerasTensor((2, 1, 1, 1))
        self.assertEqual(knp.squeeze(x, (1, 2)).shape, (2, 1))
        self.assertEqual(knp.squeeze(x, (-1, -2)).shape, (2, 1))
        self.assertEqual(knp.squeeze(x, (1, 2, 3)).shape, (2,))
        self.assertEqual(knp.squeeze(x, (-1, 1)).shape, (2, 1))

    def test_transpose(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.transpose(x).shape, (3, 2))

    def test_arccos(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.arccos(x).shape, (2, 3))

    def test_arccosh(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.arccosh(x).shape, (2, 3))

    def test_arcsin(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.arcsin(x).shape, (2, 3))

    def test_arcsinh(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.arcsinh(x).shape, (2, 3))

    def test_arctan(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.arctan(x).shape, (2, 3))

    def test_arctanh(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.arctanh(x).shape, (2, 3))

    def test_argmax(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.argmax(x).shape, ())
        self.assertEqual(knp.argmax(x, keepdims=True).shape, (2, 3))

    def test_argmin(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.argmin(x).shape, ())
        self.assertEqual(knp.argmin(x, keepdims=True).shape, (2, 3))

    def test_argsort(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.argsort(x).shape, (2, 3))
        self.assertEqual(knp.argsort(x, axis=None).shape, (6,))

    def test_array(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.array(x).shape, (2, 3))

    def test_average(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.average(x).shape, ())

    def test_bitwise_invert(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.bitwise_invert(x).shape, (2, 3))

    # bitwise_not is same as bitwise_invert

    def test_broadcast_to(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.broadcast_to(x, (2, 2, 3)).shape, (2, 2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((3, 3))
            knp.broadcast_to(x, (2, 2, 3))

    def test_cbrt(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.cbrt(x).shape, (2, 3))

    def test_ceil(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.ceil(x).shape, (2, 3))

    def test_clip(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.clip(x, 1, 2).shape, (2, 3))

    def test_concatenate(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.concatenate([x, y]).shape, (4, 3))
        self.assertEqual(knp.concatenate([x, y], axis=1).shape, (2, 6))

        with self.assertRaises(ValueError):
            self.assertEqual(knp.concatenate([x, y], axis=None).shape, (None,))

    def test_conjugate(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.conjugate(x).shape, (2, 3))

    def test_conj(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.conj(x).shape, (2, 3))

    def test_copy(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.copy(x).shape, (2, 3))

    def test_cos(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.cos(x).shape, (2, 3))

    def test_cosh(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.cosh(x).shape, (2, 3))

    def test_count_nonzero(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.count_nonzero(x).shape, ())

    def test_cumprod(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.cumprod(x).shape, (6,))

    def test_cumsum(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.cumsum(x).shape, (6,))

    def test_deg2rad(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.deg2rad(x).shape, (2, 3))

    def test_diag(self):
        x = KerasTensor((3,))
        self.assertEqual(knp.diag(x).shape, (3, 3))
        self.assertEqual(knp.diag(x, k=3).shape, (6, 6))
        self.assertEqual(knp.diag(x, k=-2).shape, (5, 5))

        x = KerasTensor((3, 5))
        self.assertEqual(knp.diag(x).shape, (3,))
        self.assertEqual(knp.diag(x, k=3).shape, (2,))
        self.assertEqual(knp.diag(x, k=-2).shape, (1,))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3, 4))
            knp.diag(x)

    def test_diagflat(self):
        x = KerasTensor((3,))
        self.assertEqual(knp.diagflat(x).shape, (3, 3))
        self.assertEqual(knp.diagflat(x, k=1).shape, (4, 4))
        self.assertEqual(knp.diagflat(x, k=-1).shape, (4, 4))

        x = KerasTensor((2, 3))
        self.assertEqual(knp.diagflat(x).shape, (6, 6))
        self.assertEqual(knp.diagflat(x, k=1).shape, (7, 7))
        self.assertEqual(knp.diagflat(x, k=-1).shape, (7, 7))

        x = KerasTensor((None, 3))
        self.assertEqual(knp.diagflat(x).shape, (None, None))

        x = KerasTensor(())
        self.assertEqual(knp.diagflat(x).shape, (1, 1))

    def test_diagonal(self):
        x = KerasTensor((3, 3))
        self.assertEqual(knp.diagonal(x).shape, (3,))
        self.assertEqual(knp.diagonal(x, offset=1).shape, (2,))

        x = KerasTensor((3, 5, 5))
        self.assertEqual(knp.diagonal(x).shape, (5, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor((3,))
            knp.diagonal(x)

    def test_diff(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.diff(x).shape, (2, 2))
        self.assertEqual(knp.diff(x, n=2).shape, (2, 1))
        self.assertEqual(knp.diff(x, n=3).shape, (2, 0))
        self.assertEqual(knp.diff(x, n=4).shape, (2, 0))

        self.assertEqual(knp.diff(x, axis=0).shape, (1, 3))
        self.assertEqual(knp.diff(x, n=2, axis=0).shape, (0, 3))
        self.assertEqual(knp.diff(x, n=3, axis=0).shape, (0, 3))

    def test_dot(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((3, 2))
        z = KerasTensor((4, 3, 2))
        self.assertEqual(knp.dot(x, y).shape, (2, 2))
        self.assertEqual(knp.dot(x, 2).shape, (2, 3))
        self.assertEqual(knp.dot(x, z).shape, (2, 4, 2))

        x = KerasTensor((5,))
        y = KerasTensor((5,))
        self.assertEqual(knp.dot(x, y).shape, ())

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((2, 3))
            knp.dot(x, y)

    def test_empty_like(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.empty_like(x).shape, (2, 3))
        self.assertEqual(knp.empty_like(x).dtype, x.dtype)

    def test_exp(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.exp(x).shape, (2, 3))

    def test_exp2(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.exp2(x).shape, (2, 3))

    def test_expand_dims(self):
        x = KerasTensor((2, 3, 4))
        self.assertEqual(knp.expand_dims(x, 0).shape, (1, 2, 3, 4))
        self.assertEqual(knp.expand_dims(x, 1).shape, (2, 1, 3, 4))
        self.assertEqual(knp.expand_dims(x, -2).shape, (2, 3, 1, 4))

        # Multiple axes
        self.assertEqual(knp.expand_dims(x, (1, 2)).shape, (2, 1, 1, 3, 4))
        self.assertEqual(knp.expand_dims(x, (-1, -2)).shape, (2, 3, 4, 1, 1))
        self.assertEqual(knp.expand_dims(x, (-1, 1)).shape, (2, 1, 3, 4, 1))

    def test_expm1(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.expm1(x).shape, (2, 3))

    def test_flip(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.flip(x).shape, (2, 3))

    def test_floor(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.floor(x).shape, (2, 3))

    def test_get_item(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.get_item(x, 1).shape, (3,))

        x = KerasTensor((5, 3, 2))
        self.assertEqual(knp.get_item(x, 3).shape, (3, 2))

        x = KerasTensor(
            [
                2,
            ]
        )
        self.assertEqual(knp.get_item(x, 0).shape, ())

    def test_hstack(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.hstack([x, y]).shape, (2, 6))

    def test_imag(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.imag(x).shape, (2, 3))

    def test_isfinite(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.isfinite(x).shape, (2, 3))

    def test_isinf(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.isinf(x).shape, (2, 3))

    def test_isnan(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.isnan(x).shape, (2, 3))

    def test_isneginf(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.isneginf(x).shape, (2, 3))

    def test_isposinf(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.isposinf(x).shape, (2, 3))

    def test_isreal(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.isreal(x).shape, (2, 3))

    def test_log(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.log(x).shape, (2, 3))

    def test_log10(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.log10(x).shape, (2, 3))

    def test_log1p(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.log1p(x).shape, (2, 3))

    def test_log2(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.log2(x).shape, (2, 3))

    def test_logaddexp(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.logaddexp(x, x).shape, (2, 3))

    def test_logaddexp2(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.logaddexp2(x, x).shape, (2, 3))

    def test_logical_not(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.logical_not(x).shape, (2, 3))

    def test_max(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.max(x).shape, ())

    def test_median(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.median(x).shape, ())

        x = KerasTensor((2, 3, 3))
        self.assertEqual(knp.median(x, axis=1).shape, (2, 3))
        self.assertEqual(knp.median(x, axis=1, keepdims=True).shape, (2, 1, 3))

    def test_meshgrid(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3, 4))
        z = KerasTensor((2, 3, 4, 5))
        self.assertEqual(knp.meshgrid(x, y)[0].shape, (24, 6))
        self.assertEqual(knp.meshgrid(x, y)[1].shape, (24, 6))
        self.assertEqual(knp.meshgrid(x, y, indexing="ij")[0].shape, (6, 24))
        self.assertEqual(
            knp.meshgrid(x, y, z, indexing="ij")[0].shape, (6, 24, 120)
        )
        with self.assertRaises(ValueError):
            knp.meshgrid(x, y, indexing="kk")

    def test_moveaxis(self):
        x = KerasTensor((2, 3, 4, 5))
        self.assertEqual(knp.moveaxis(x, 0, -1).shape, (3, 4, 5, 2))
        self.assertEqual(knp.moveaxis(x, -1, 0).shape, (5, 2, 3, 4))
        self.assertEqual(knp.moveaxis(x, [0, 1], [-1, -2]).shape, (4, 5, 3, 2))
        self.assertEqual(knp.moveaxis(x, [0, 1], [1, 0]).shape, (3, 2, 4, 5))
        self.assertEqual(knp.moveaxis(x, [0, 1], [-2, -1]).shape, (4, 5, 2, 3))

    def test_nanmax(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.nanmax(x).shape, ())
        self.assertEqual(knp.nanmax(x, axis=0).shape, (3,))
        self.assertEqual(knp.nanmax(x, axis=1).shape, (2,))
        self.assertEqual(knp.nanmax(x, axis=1, keepdims=True).shape, (2, 1))

    def test_nanmean(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.nanmean(x).shape, ())
        self.assertEqual(knp.nanmean(x, axis=0).shape, (3,))
        self.assertEqual(knp.nanmean(x, axis=1).shape, (2,))
        self.assertEqual(knp.nanmean(x, axis=1, keepdims=True).shape, (2, 1))

    def test_nanmin(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.nanmin(x).shape, ())
        self.assertEqual(knp.nanmin(x, axis=0).shape, (3,))
        self.assertEqual(knp.nanmin(x, axis=1).shape, (2,))
        self.assertEqual(knp.nanmin(x, axis=1, keepdims=True).shape, (2, 1))

    def test_nanprod_(self):
        x = KerasTensor((2, 3))

        self.assertEqual(knp.nanprod(x).shape, ())
        self.assertEqual(knp.nanprod(x, axis=0).shape, (3,))
        self.assertEqual(knp.nanprod(x, axis=1).shape, (2,))
        self.assertEqual(knp.nanprod(x, axis=1, keepdims=True).shape, (2, 1))

    def test_nanstd_(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.nanstd(x).shape, ())
        self.assertEqual(knp.nanstd(x, axis=0).shape, (3,))
        self.assertEqual(knp.nanstd(x, axis=1).shape, (2,))
        self.assertEqual(knp.nanstd(x, axis=1, keepdims=True).shape, (2, 1))

    def test_nansum_(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.nansum(x).shape, ())
        self.assertEqual(knp.nansum(x, axis=0).shape, (3,))
        self.assertEqual(knp.nansum(x, axis=1).shape, (2,))
        self.assertEqual(knp.nansum(x, axis=1, keepdims=True).shape, (2, 1))

    def test_nanvar_(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.nanvar(x).shape, ())
        self.assertEqual(knp.nanvar(x, axis=0).shape, (3,))
        self.assertEqual(knp.nanvar(x, axis=1).shape, (2,))
        self.assertEqual(knp.nanvar(x, axis=1, keepdims=True).shape, (2, 1))

    def test_ndim(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.ndim(x).shape, (2,))

    def test_ones_like(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.ones_like(x).shape, (2, 3))
        self.assertEqual(knp.ones_like(x).dtype, x.dtype)

    def test_zeros_like(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.zeros_like(x).shape, (2, 3))
        self.assertEqual(knp.zeros_like(x).dtype, x.dtype)

    def test_pad(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.pad(x, 1).shape, (4, 5))
        self.assertEqual(knp.pad(x, (1, 2)).shape, (5, 6))
        self.assertEqual(knp.pad(x, ((1, 2), (3, 4))).shape, (5, 10))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            knp.pad(x, ((1, 2), (3, 4), (5, 6)))

    def test_prod(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.prod(x).shape, ())
        self.assertEqual(knp.prod(x, axis=0).shape, (3,))
        self.assertEqual(knp.prod(x, axis=1).shape, (2,))

    def test_ptp(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.ptp(x).shape, ())
        self.assertEqual(knp.ptp(x, axis=0).shape, (3,))
        self.assertEqual(knp.ptp(x, axis=1).shape, (2,))

    def test_ravel(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.ravel(x).shape, (6,))

    def test_unravel_index(self):
        x = KerasTensor((6,))
        indices = knp.unravel_index(x, (2, 3))
        self.assertEqual(len(indices), 2)
        self.assertEqual(indices[0].shape, (6,))
        self.assertEqual(indices[1].shape, (6,))

        x = KerasTensor((2, 3))
        indices = knp.unravel_index(x, (3, 4))
        self.assertEqual(len(indices), 2)
        self.assertEqual(indices[0].shape, (2, 3))
        self.assertEqual(indices[1].shape, (2, 3))

    def test_real(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.real(x).shape, (2, 3))

    def test_reciprocal(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.reciprocal(x).shape, (2, 3))

    def test_repeat(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.repeat(x, 2).shape, (12,))
        self.assertEqual(knp.repeat(x, [2]).shape, (12,))
        self.assertEqual(knp.repeat(x, 3, axis=1).shape, (2, 9))
        self.assertEqual(knp.repeat(x, [1, 2], axis=0).shape, (3, 3))

        with self.assertRaises(ValueError):
            knp.repeat(x, [1, 1])
        with self.assertRaises(ValueError):
            knp.repeat(x, [1, 1, 1], axis=0)

    def test_reshape(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.reshape(x, (3, 2)).shape, (3, 2))
        self.assertEqual(knp.reshape(x, (3, -1)).shape, (3, 2))
        self.assertEqual(knp.reshape(x, (6,)).shape, (6,))
        self.assertEqual(knp.reshape(x, (-1,)).shape, (6,))

    def test_roll(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.roll(x, 1).shape, (2, 3))
        self.assertEqual(knp.roll(x, 1, axis=1).shape, (2, 3))
        self.assertEqual(knp.roll(x, 1, axis=0).shape, (2, 3))

    def test_round(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.round(x).shape, (2, 3))

    def test_sign(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.sign(x).shape, (2, 3))

    def test_signbit(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.signbit(x).shape, (2, 3))

    def test_sin(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.sin(x).shape, (2, 3))

    def test_sinh(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.sinh(x).shape, (2, 3))

    def test_size(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.size(x).shape, ())

    def test_sort(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.sort(x).shape, (2, 3))
        self.assertEqual(knp.sort(x, axis=1).shape, (2, 3))
        self.assertEqual(knp.sort(x, axis=0).shape, (2, 3))

    def test_split(self):
        x = KerasTensor((2, 3))
        self.assertEqual(len(knp.split(x, 2)), 2)
        self.assertEqual(knp.split(x, 2)[0].shape, (1, 3))
        self.assertEqual(knp.split(x, 3, axis=1)[0].shape, (2, 1))
        self.assertEqual(len(knp.split(x, [1, 3], axis=1)), 3)
        self.assertEqual(knp.split(x, [1, 3], axis=1)[0].shape, (2, 1))
        self.assertEqual(knp.split(x, [1, 3], axis=1)[1].shape, (2, 2))
        self.assertEqual(knp.split(x, [1, 3], axis=1)[2].shape, (2, 0))

        with self.assertRaises(ValueError):
            knp.split(x, 2, axis=1)

    def test_hsplit(self):
        x = KerasTensor((3, 5))

        splits = knp.hsplit(x, 5)
        self.assertEqual(len(splits), 5)
        for split in splits:
            self.assertEqual(split.shape, (3, 1))

        splits = knp.hsplit(x, [1, 3])
        self.assertEqual(len(splits), 3)
        self.assertEqual(splits[0].shape, (3, 1))
        self.assertEqual(splits[1].shape, (3, 2))
        self.assertEqual(splits[2].shape, (3, 2))

        # test 1D case
        x_1d = KerasTensor((10,))
        splits = knp.hsplit(x_1d, 2)
        self.assertEqual(len(splits), 2)
        for split in splits:
            self.assertEqual(split.shape, (5,))

        splits = knp.hsplit(x_1d, [2, 5])
        self.assertEqual(len(splits), 3)
        self.assertEqual(splits[0].shape, (2,))
        self.assertEqual(splits[1].shape, (3,))
        self.assertEqual(splits[2].shape, (5,))

    def test_vsplit(self):
        x = KerasTensor((5, 3))

        splits = knp.vsplit(x, 5)
        self.assertEqual(len(splits), 5)
        for split in splits:
            self.assertEqual(split.shape, (1, 3))

        splits = knp.vsplit(x, [1, 3])
        self.assertEqual(len(splits), 3)
        self.assertEqual(splits[0].shape, (1, 3))
        self.assertEqual(splits[1].shape, (2, 3))
        self.assertEqual(splits[2].shape, (2, 3))

    def test_sqrt(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.sqrt(x).shape, (2, 3))

    def test_stack(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.stack([x, y]).shape, (2, 2, 3))
        self.assertEqual(knp.stack([x, y], axis=-1).shape, (2, 3, 2))

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            y = KerasTensor((3, 3))
            knp.stack([x, y])

    def test_std(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.std(x).shape, ())

    def test_swapaxes(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.swapaxes(x, 0, 1).shape, (3, 2))

    def test_tan(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.tan(x).shape, (2, 3))

    def test_tanh(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.tanh(x).shape, (2, 3))

    def test_tile(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.tile(x, 2).shape, (2, 6))
        self.assertEqual(knp.tile(x, [2]).shape, (2, 6))
        self.assertEqual(knp.tile(x, [1, 2]).shape, (2, 6))
        self.assertEqual(knp.tile(x, [2, 1, 2]).shape, (2, 2, 6))

    def test_trace(self):
        x = KerasTensor((2, 3, 4, 5))
        self.assertEqual(knp.trace(x).shape, (4, 5))
        self.assertEqual(knp.trace(x, axis1=2, axis2=3).shape, (2, 3))

    def test_tril(self):
        x = KerasTensor((2, 3, 4, 5))
        self.assertEqual(knp.tril(x).shape, (2, 3, 4, 5))
        self.assertEqual(knp.tril(x, k=1).shape, (2, 3, 4, 5))
        self.assertEqual(knp.tril(x, k=-1).shape, (2, 3, 4, 5))

    def test_triu(self):
        x = KerasTensor((2, 3, 4, 5))
        self.assertEqual(knp.triu(x).shape, (2, 3, 4, 5))
        self.assertEqual(knp.triu(x, k=1).shape, (2, 3, 4, 5))
        self.assertEqual(knp.triu(x, k=-1).shape, (2, 3, 4, 5))

    def test_trunc(self):
        x = KerasTensor((2, 3, 4, 5))
        self.assertEqual(knp.trunc(x).shape, (2, 3, 4, 5))

    def test_vstack(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.vstack([x, y]).shape, (4, 3))

    def test_dstack(self):
        x = KerasTensor((3,))
        y = KerasTensor((3,))
        self.assertEqual(knp.dstack([x, y]).shape, (1, 3, 2))

        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.dstack([x, y]).shape, (2, 3, 2))

        x = KerasTensor((2, 3, 4))
        y = KerasTensor((2, 3, 5))
        self.assertEqual(knp.dstack([x, y]).shape, (2, 3, 9))

    def test_argpartition(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.argpartition(x, 3).shape, (2, 3))
        self.assertEqual(knp.argpartition(x, 1, axis=1).shape, (2, 3))

        with self.assertRaises(ValueError):
            knp.argpartition(x, (1, 3))

    def test_angle(self):
        x = KerasTensor((2, 3))
        self.assertEqual(knp.angle(x).shape, (2, 3))

    def test_view(self):
        x = knp.array(KerasTensor((2, 3)), dtype="int32")
        self.assertEqual(knp.view(x, dtype="uint32").shape, (2, 3))
        self.assertEqual(knp.view(x, dtype="uint32").dtype, "uint32")
        x = knp.array(KerasTensor((2, 3)), dtype="int32")
        self.assertEqual(knp.view(x, dtype="int16").shape, (2, 6))
        self.assertEqual(knp.view(x, dtype="int16").dtype, "int16")
        x = knp.array(KerasTensor((2, 4)), dtype="int16")
        self.assertEqual(knp.view(x, dtype="int32").shape, (2, 2))
        self.assertEqual(knp.view(x, dtype="int32").dtype, "int32")

    def test_array_split(self):
        x = KerasTensor((8, 4))
        splits = knp.array_split(x, 3, axis=0)
        self.assertEqual(len(splits), 3)
        self.assertEqual(splits[0].shape, (3, 4))
        self.assertEqual(splits[1].shape, (3, 4))
        self.assertEqual(splits[2].shape, (2, 4))


class NumpyTwoInputOpsCorrectnessTest(testing.TestCase):
    def test_add(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        z = np.array([[[1, 2, 3], [3, 2, 1]]])
        self.assertAllClose(knp.add(x, y), np.add(x, y))
        self.assertAllClose(knp.add(x, z), np.add(x, z))

        self.assertAllClose(knp.Add()(x, y), np.add(x, y))
        self.assertAllClose(knp.Add()(x, z), np.add(x, z))

    def test_heaviside(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [6, 5, 4]])
        self.assertAllClose(knp.heaviside(x, y), np.heaviside(x, y))
        self.assertAllClose(knp.Heaviside()(x, y), np.heaviside(x, y))

        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array(4)
        self.assertAllClose(knp.heaviside(x, y), np.heaviside(x, y))
        self.assertAllClose(knp.Heaviside()(x, y), np.heaviside(x, y))

    def test_hypot(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [6, 5, 4]])
        self.assertAllClose(knp.hypot(x, y), np.hypot(x, y))
        self.assertAllClose(knp.Hypot()(x, y), np.hypot(x, y))

        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array(4)
        self.assertAllClose(knp.hypot(x, y), np.hypot(x, y))
        self.assertAllClose(knp.Hypot()(x, y), np.hypot(x, y))

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
        p = np.ones([4])
        self.assertAllClose(knp.matmul(x, y), np.matmul(x, y))
        self.assertAllClose(knp.matmul(x, z), np.matmul(x, z))
        self.assertAllClose(knp.matmul(p, x), np.matmul(p, x))

        self.assertAllClose(knp.Matmul()(x, y), np.matmul(x, y))
        self.assertAllClose(knp.Matmul()(x, z), np.matmul(x, z))
        self.assertAllClose(knp.Matmul()(p, x), np.matmul(p, x))

    @parameterized.named_parameters(
        named_product(
            (
                {
                    "testcase_name": "rank2",
                    "x_shape": (5, 3),
                    "y_shape": (3, 4),
                },
                {
                    "testcase_name": "rank3",
                    "x_shape": (2, 5, 3),
                    "y_shape": (2, 3, 4),
                },
                {
                    "testcase_name": "rank4",
                    "x_shape": (2, 2, 5, 3),
                    "y_shape": (2, 2, 3, 4),
                },
            ),
            dtype=["float16", "float32", "float64", "int32"],
            x_sparse=[False, True],
            y_sparse=[False, True],
        )
    )
    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason="Backend does not support sparse tensors.",
    )
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_matmul_sparse(self, dtype, x_shape, y_shape, x_sparse, y_sparse):
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            if x_sparse and y_sparse and dtype in ("float16", "int32"):
                pytest.skip(
                    f"Sparse sparse matmul unsupported for {dtype}"
                    " with TensorFlow backend"
                )

            dense_to_sparse = tf.sparse.from_dense
        elif backend.backend() == "jax":
            import jax.experimental.sparse as jax_sparse

            dense_to_sparse = functools.partial(
                jax_sparse.BCOO.fromdense, n_batch=len(x_shape) - 2
            )

        rng = np.random.default_rng(0)

        x = x_np = (4 * rng.standard_normal(x_shape)).astype(dtype)
        if x_sparse:
            x_np = np.multiply(x_np, rng.random(x_shape) < 0.7)
            x = dense_to_sparse(x_np)

        y = y_np = (4 * rng.standard_normal(y_shape)).astype(dtype)
        if y_sparse:
            y_np = np.multiply(y_np, rng.random(y_shape) < 0.7)
            y = dense_to_sparse(y_np)

        atol = 0.1 if dtype == "float16" else 1e-4
        tpu_atol = 1 if dtype == "float16" else 1e-1
        self.assertAllClose(
            knp.matmul(x, y),
            np.matmul(x_np, y_np),
            atol=atol,
            tpu_atol=tpu_atol,
            tpu_rtol=tpu_atol,
        )
        self.assertSparse(knp.matmul(x, y), x_sparse and y_sparse)

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

    def test_divide_no_nan(self):
        x = np.array(
            [[2, 1, 0], [np.inf, -np.inf, np.nan], [np.inf, -np.inf, np.nan]]
        )
        y = np.array([[2, 0, 0], [0, 0, 0], [3, 2, 1]])
        expected_result = np.array(
            [[1, 0, 0], [0, 0, 0], [np.inf, -np.inf, np.nan]]
        )
        self.assertAllClose(knp.divide_no_nan(x, y), expected_result)
        self.assertAllClose(knp.DivideNoNan()(x, y), expected_result)

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

        a = np.array([0.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
        b = np.array([0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0])

        self.assertAllClose(knp.arctan2(a, b), np.arctan2(a, b))
        self.assertAllClose(knp.Arctan2()(a, b), np.arctan2(a, b))

        m = np.array([[3, 4], [7, 8]], dtype=np.int8)
        n = np.array([[1, 2], [3, 4]], dtype=float)

        self.assertAllClose(knp.arctan2(m, n), np.arctan2(m, n))
        self.assertAllClose(knp.Arctan2()(m, n), np.arctan2(m, n))

        x = np.array([1.0, 2.0, np.nan])
        y = np.array([3.0, np.nan, 4.0])
        self.assertAllClose(knp.arctan2(x, y), np.arctan2(x, y))
        self.assertAllClose(knp.Arctan2()(x, y), np.arctan2(x, y))

    def test_bitwise_and(self):
        x = np.array([2, 5, 255])
        y = np.array([3, 14, 16])
        self.assertAllClose(knp.bitwise_and(x, y), np.bitwise_and(x, y))
        self.assertAllClose(knp.BitwiseAnd()(x, y), np.bitwise_and(x, y))

    def test_bitwise_or(self):
        x = np.array([2, 5, 255])
        y = np.array([3, 14, 16])
        self.assertAllClose(knp.bitwise_or(x, y), np.bitwise_or(x, y))
        self.assertAllClose(knp.BitwiseOr()(x, y), np.bitwise_or(x, y))

    def test_bitwise_xor(self):
        x = np.array([2, 5, 255])
        y = np.array([3, 14, 16])
        self.assertAllClose(knp.bitwise_xor(x, y), np.bitwise_xor(x, y))
        self.assertAllClose(knp.BitwiseXor()(x, y), np.bitwise_xor(x, y))

    def test_bitwise_left_shift(self):
        x = np.array([50, 60, 70])
        y = np.array([1, 2, 3])
        self.assertAllClose(knp.bitwise_left_shift(x, y), np.left_shift(x, y))
        self.assertAllClose(knp.BitwiseLeftShift()(x, y), np.left_shift(x, y))

    # left_shift is same as bitwise_left_shift

    def test_bitwise_right_shift(self):
        x = np.array([5, 6, 7])
        y = np.array([1, 2, 3])
        self.assertAllClose(knp.bitwise_right_shift(x, y), np.right_shift(x, y))
        self.assertAllClose(knp.BitwiseRightShift()(x, y), np.right_shift(x, y))

    # right_shift is same as bitwise_right_shift

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

        # Test axis is not None
        self.assertAllClose(
            knp.cross(x1, y1, axis=-1), np.cross(x1, y1, axis=-1)
        )
        self.assertAllClose(
            knp.Cross(axis=-1)(x1, y1), np.cross(x1, y1, axis=-1)
        )

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

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason=f"{backend.backend()} doesn't implement custom ops for einsum.",
    )
    def test_einsum_custom_ops_for_tensorflow(self):
        subscripts = "a,b->ab"
        x = np.arange(2).reshape([2]).astype("float32")
        y = np.arange(3).reshape([3]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

        subscripts = "ab,b->a"
        x = np.arange(6).reshape([2, 3]).astype("float32")
        y = np.arange(3).reshape([3]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

        subscripts = "ab,bc->ac"
        x = np.arange(6).reshape([2, 3]).astype("float32")
        y = np.arange(12).reshape([3, 4]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

        subscripts = "ab,cb->ac"
        x = np.arange(6).reshape([2, 3]).astype("float32")
        y = np.arange(12).reshape([4, 3]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

        subscripts = "abc,cd->abd"
        x = np.arange(24).reshape([2, 3, 4]).astype("float32")
        y = np.arange(20).reshape([4, 5]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

        subscripts = "abc,cde->abde"
        x = np.arange(24).reshape([2, 3, 4]).astype("float32")
        y = np.arange(120).reshape([4, 5, 6]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

        subscripts = "abc,dc->abd"
        x = np.arange(24).reshape([2, 3, 4]).astype("float32")
        y = np.arange(20).reshape([5, 4]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

        subscripts = "abc,dce->abde"
        x = np.arange(24).reshape([2, 3, 4]).astype("float32")
        y = np.arange(120).reshape([5, 4, 6]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

        subscripts = "abc,dec->abde"
        x = np.arange(24).reshape([2, 3, 4]).astype("float32")
        y = np.arange(120).reshape([5, 6, 4]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

        subscripts = "abcd,abde->abce"
        x = np.arange(120).reshape([2, 3, 4, 5]).astype("float32")
        y = np.arange(180).reshape([2, 3, 5, 6]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

        subscripts = "abcd,abed->abce"
        x = np.arange(120).reshape([2, 3, 4, 5]).astype("float32")
        y = np.arange(180).reshape([2, 3, 6, 5]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

        subscripts = "abcd,acbe->adbe"
        x = np.arange(120).reshape([2, 3, 4, 5]).astype("float32")
        y = np.arange(144).reshape([2, 4, 3, 6]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

        subscripts = "abcd,adbe->acbe"
        x = np.arange(120).reshape([2, 3, 4, 5]).astype("float32")
        y = np.arange(180).reshape([2, 5, 3, 6]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

        subscripts = "abcd,aecd->acbe"
        x = np.arange(120).reshape([2, 3, 4, 5]).astype("float32")
        y = np.arange(240).reshape([2, 6, 4, 5]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

        subscripts = "abcd,aecd->aceb"
        x = np.arange(120).reshape([2, 3, 4, 5]).astype("float32")
        y = np.arange(240).reshape([2, 6, 4, 5]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

        subscripts = "abcd,cde->abe"
        x = np.arange(120).reshape([2, 3, 4, 5]).astype("float32")
        y = np.arange(120).reshape([4, 5, 6]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

        subscripts = "abcd,ced->abe"
        x = np.arange(120).reshape([2, 3, 4, 5]).astype("float32")
        y = np.arange(120).reshape([4, 6, 5]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

        subscripts = "abcd,ecd->abe"
        x = np.arange(120).reshape([2, 3, 4, 5]).astype("float32")
        y = np.arange(120).reshape([6, 4, 5]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

        subscripts = "abcde,aebf->adbcf"
        x = np.arange(720).reshape([2, 3, 4, 5, 6]).astype("float32")
        y = np.arange(252).reshape([2, 6, 3, 7]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

        subscripts = "abcde,afce->acdbf"
        x = np.arange(720).reshape([2, 3, 4, 5, 6]).astype("float32")
        y = np.arange(336).reshape([2, 7, 4, 6]).astype("float32")
        self.assertAllClose(
            knp.einsum(subscripts, x, y), np.einsum(subscripts, x, y)
        )

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
            knp.FullLike(dtype="float32")(x, 2),
            np.full_like(x, 2, dtype="float32"),
        )
        self.assertAllClose(
            knp.FullLike()(x, np.ones([2, 3])),
            np.full_like(x, np.ones([2, 3])),
        )

    def test_gcd(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        self.assertAllClose(knp.gcd(x, y), np.gcd(x, y))
        self.assertAllClose(knp.gcd(x, 2), np.gcd(x, 2))
        self.assertAllClose(knp.gcd(2, x), np.gcd(2, x))

        self.assertAllClose(knp.Gcd()(x, y), np.gcd(x, y))
        self.assertAllClose(knp.Gcd()(x, 2), np.gcd(x, 2))
        self.assertAllClose(knp.Gcd()(2, x), np.gcd(2, x))

    def test_geomspace(self):
        self.assertAllClose(knp.geomspace(1, 1000, 4), np.geomspace(1, 1000, 4))
        self.assertAllClose(
            knp.geomspace(1, 1000, 4, endpoint=False),
            np.geomspace(1, 1000, 4, endpoint=False),
        )
        self.assertAllClose(
            knp.Geomspace(num=4)(1, 1000), np.geomspace(1, 1000, 4)
        )
        self.assertAllClose(
            knp.Geomspace(num=4, endpoint=False)(1, 1000),
            np.geomspace(1, 1000, 4, endpoint=False),
        )

        start = np.array([1.0, 2.0, 3.0])
        stop = np.array([1000.0, 2000.0, 3000.0])

        self.assertAllClose(
            knp.geomspace(start, stop, 4),
            np.geomspace(start, stop, 4),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertAllClose(
            knp.geomspace(start, stop, 4, endpoint=False),
            np.geomspace(start, stop, 4, endpoint=False),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertAllClose(
            knp.Geomspace(num=4)(start, stop),
            np.geomspace(start, stop, 4),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertAllClose(
            knp.Geomspace(num=4, endpoint=False)(start, stop),
            np.geomspace(start, stop, 4, endpoint=False),
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

    def test_allclose(self):
        x = np.array([1], dtype="int32")
        y = np.array([2], dtype="int32")
        self.assertAllClose(knp.allclose(x, y, rtol=0.1, atol=1e-8), False)

        x = np.array([1.0], dtype="float32")
        y = np.array([1.0000001], dtype="float32")
        self.assertAllClose(knp.allclose(x, y, rtol=0.1, atol=1e-8), True)

        # Test with NaNs
        x_nan = np.array([np.nan, 1.0])
        y_nan = np.array([np.nan, 1.0])
        self.assertAllClose(knp.allclose(x_nan, y_nan), False)
        self.assertAllClose(knp.allclose(x_nan, y_nan, equal_nan=True), True)

    def test_isclose(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        self.assertAllClose(knp.isclose(x, y), np.isclose(x, y))
        self.assertAllClose(knp.isclose(x, 2), np.isclose(x, 2))
        self.assertAllClose(knp.isclose(2, x), np.isclose(2, x))

        self.assertAllClose(knp.Isclose()(x, y), np.isclose(x, y))
        self.assertAllClose(knp.Isclose()(x, 2), np.isclose(x, 2))
        self.assertAllClose(knp.Isclose()(2, x), np.isclose(2, x))

    def test_isin(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        self.assertAllClose(knp.isin(x, y), np.isin(x, y))
        self.assertAllClose(knp.isin(x, 2), np.isin(x, 2))
        self.assertAllClose(knp.isin(2, x), np.isin(2, x))

        self.assertAllClose(
            knp.isin(x, y, assume_unique=True),
            np.isin(x, y, assume_unique=True),
        )
        self.assertAllClose(
            knp.isin(x, 2, assume_unique=True),
            np.isin(x, 2, assume_unique=True),
        )
        self.assertAllClose(
            knp.isin(2, x, assume_unique=True),
            np.isin(2, x, assume_unique=True),
        )

        self.assertAllClose(
            knp.isin(x, y, invert=True), np.isin(x, y, invert=True)
        )
        self.assertAllClose(
            knp.isin(x, 2, invert=True), np.isin(x, 2, invert=True)
        )
        self.assertAllClose(
            knp.isin(2, x, invert=True), np.isin(2, x, invert=True)
        )

        self.assertAllClose(
            knp.isin(x, y, assume_unique=True, invert=True),
            np.isin(x, y, assume_unique=True, invert=True),
        )
        self.assertAllClose(
            knp.isin(x, 2, assume_unique=True, invert=True),
            np.isin(x, 2, assume_unique=True, invert=True),
        )
        self.assertAllClose(
            knp.isin(2, x, assume_unique=True, invert=True),
            np.isin(2, x, assume_unique=True, invert=True),
        )

        self.assertAllClose(knp.IsIn()(x, y), np.isin(x, y))
        self.assertAllClose(knp.IsIn()(x, 2), np.isin(x, 2))
        self.assertAllClose(knp.IsIn()(2, x), np.isin(2, x))

        self.assertAllClose(
            knp.IsIn(assume_unique=True)(x, y),
            np.isin(x, y, assume_unique=True),
        )
        self.assertAllClose(
            knp.IsIn(invert=True)(x, y),
            np.isin(x, y, invert=True),
        )
        self.assertAllClose(
            knp.IsIn(assume_unique=True, invert=True)(x, y),
            np.isin(x, y, assume_unique=True, invert=True),
        )

    def test_kron(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        self.assertAllClose(knp.kron(x, y), np.kron(x, y))
        self.assertAllClose(knp.Kron()(x, y), np.kron(x, y))

    def test_lcm(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        self.assertAllClose(knp.lcm(x, y), np.lcm(x, y))
        self.assertAllClose(knp.Lcm()(x, y), np.lcm(x, y))

        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array(4)
        self.assertAllClose(knp.lcm(x, y), np.lcm(x, y))
        self.assertAllClose(knp.Lcm()(x, y), np.lcm(x, y))

        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([4])
        self.assertAllClose(knp.lcm(x, y), np.lcm(x, y))
        self.assertAllClose(knp.Lcm()(x, y), np.lcm(x, y))

    def test_ldexp(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        self.assertAllClose(knp.ldexp(x, y), np.ldexp(x, y))
        self.assertAllClose(knp.Ldexp()(x, y), np.ldexp(x, y))

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
        self.assertAllClose(
            knp.Linspace(num=0, endpoint=False)(0, 10),
            np.linspace(0, 10, 0, endpoint=False),
        )

        start = np.zeros([2, 3, 4])
        stop = np.ones([2, 3, 4])
        self.assertAllClose(
            knp.linspace(start, stop, 5, retstep=True)[0],
            np.linspace(start, stop, 5, retstep=True)[0],
        )
        self.assertAllClose(
            knp.linspace(start, stop, 5, endpoint=False, retstep=True)[0],
            np.linspace(start, stop, 5, endpoint=False, retstep=True)[0],
        )
        self.assertAllClose(
            knp.linspace(
                start, stop, 5, endpoint=False, retstep=True, dtype="int32"
            )[0],
            np.linspace(
                start, stop, 5, endpoint=False, retstep=True, dtype="int32"
            )[0],
        )

        self.assertAllClose(
            knp.Linspace(5, retstep=True)(start, stop)[0],
            np.linspace(start, stop, 5, retstep=True)[0],
        )
        self.assertAllClose(
            knp.Linspace(5, endpoint=False, retstep=True)(start, stop)[0],
            np.linspace(start, stop, 5, endpoint=False, retstep=True)[0],
        )
        self.assertAllClose(
            knp.Linspace(5, endpoint=False, retstep=True, dtype="int32")(
                start, stop
            )[0],
            np.linspace(
                start, stop, 5, endpoint=False, retstep=True, dtype="int32"
            )[0],
        )

        # Test `num` as a tensor
        # https://github.com/keras-team/keras/issues/19772
        self.assertAllClose(
            knp.linspace(0, 10, backend.convert_to_tensor(5)),
            np.linspace(0, 10, 5),
        )
        self.assertAllClose(
            knp.linspace(0, 10, backend.convert_to_tensor(5), endpoint=False),
            np.linspace(0, 10, 5, endpoint=False),
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
        self.assertAllClose(
            knp.logspace(0, 10, 5),
            np.logspace(0, 10, 5),
            tpu_atol=1e-4,
            tpu_rtol=1e-4,
        )
        self.assertAllClose(
            knp.logspace(0, 10, 5, endpoint=False),
            np.logspace(0, 10, 5, endpoint=False),
        )
        self.assertAllClose(
            knp.Logspace(num=5)(0, 10),
            np.logspace(0, 10, 5),
            tpu_atol=1e-4,
            tpu_rtol=1e-4,
        )
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

    def test_nextafter(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        self.assertAllClose(knp.nextafter(x, y), np.nextafter(x, y))
        self.assertAllClose(knp.Nextafter()(x, y), np.nextafter(x, y))

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

    def test_quantile(self):
        x = np.arange(24).reshape([2, 3, 4]).astype("float32")

        # q as scalar
        q = np.array(0.5, dtype="float32")
        self.assertAllClose(knp.quantile(x, q), np.quantile(x, q))
        self.assertAllClose(
            knp.quantile(x, q, keepdims=True), np.quantile(x, q, keepdims=True)
        )

        # q as 1D tensor
        q = np.array([0.5, 1.0], dtype="float32")
        self.assertAllClose(knp.quantile(x, q), np.quantile(x, q))
        self.assertAllClose(
            knp.quantile(x, q, keepdims=True), np.quantile(x, q, keepdims=True)
        )
        self.assertAllClose(
            knp.quantile(x, q, axis=1), np.quantile(x, q, axis=1)
        )
        self.assertAllClose(
            knp.quantile(x, q, axis=1, keepdims=True),
            np.quantile(x, q, axis=1, keepdims=True),
        )

        # multiple axes
        self.assertAllClose(
            knp.quantile(x, q, axis=(1, 2)), np.quantile(x, q, axis=(1, 2))
        )

        # test all supported methods
        q = np.array([0.501, 1.0], dtype="float32")
        for method in ["linear", "lower", "higher", "midpoint", "nearest"]:
            self.assertAllClose(
                knp.quantile(x, q, method=method),
                np.quantile(x, q, method=method),
            )
            self.assertAllClose(
                knp.quantile(x, q, axis=1, method=method),
                np.quantile(x, q, axis=1, method=method),
            )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Only test tensorflow backend",
    )
    def test_quantile_in_tf_function(self):
        import tensorflow as tf

        x = knp.array([[1, 2, 3], [4, 5, 6]])
        q = [0.5]
        expected_output = np.array([[2, 5]])

        @tf.function
        def run_quantile(x, q, axis):
            return knp.quantile(x, q, axis=axis)

        result = run_quantile(x, q, axis=1)
        self.assertAllClose(result, expected_output)

    def test_take(self):
        x = np.arange(24).reshape([1, 2, 3, 4])
        indices = np.array([0, 1])
        self.assertAllClose(knp.take(x, indices), np.take(x, indices))
        self.assertAllClose(knp.take(x, 0), np.take(x, 0))
        self.assertAllClose(knp.take(x, 0, axis=1), np.take(x, 0, axis=1))

        self.assertAllClose(knp.Take()(x, indices), np.take(x, indices))
        self.assertAllClose(knp.Take()(x, 0), np.take(x, 0))
        self.assertAllClose(knp.Take(axis=1)(x, 0), np.take(x, 0, axis=1))

        # Test with multi-dimensional indices
        rng = np.random.default_rng(0)
        x = rng.standard_normal((2, 3, 4, 5))
        indices = rng.integers(0, 4, (6, 7))
        self.assertAllClose(
            knp.take(x, indices, axis=2), np.take(x, indices, axis=2)
        )

        # Test with negative axis
        self.assertAllClose(
            knp.take(x, indices, axis=-2), np.take(x, indices, axis=-2)
        )

        # Test with axis=None & x.ndim=2
        x = np.array(([1, 2], [3, 4]))
        indices = np.array([2, 3])
        self.assertAllClose(
            knp.take(x, indices, axis=None), np.take(x, indices, axis=None)
        )

        # Test with negative indices
        x = rng.standard_normal((2, 3, 4, 5))
        indices = rng.integers(-3, 0, (6, 7))
        self.assertAllClose(
            knp.take(x, indices, axis=2), np.take(x, indices, axis=2)
        )

    @parameterized.named_parameters(
        named_product(
            [
                {"testcase_name": "axis_none", "axis": None},
                {"testcase_name": "axis_0", "axis": 0},
                {"testcase_name": "axis_1", "axis": 1},
                {"testcase_name": "axis_minus1", "axis": -1},
            ],
            dtype=[
                "float16",
                "float32",
                "float64",
                "uint8",
                "int8",
                "int16",
                "int32",
            ],
        )
    )
    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason="Backend does not support sparse tensors.",
    )
    def test_take_sparse(self, dtype, axis):
        rng = np.random.default_rng(0)
        x = (4 * rng.standard_normal((3, 4, 5))).astype(dtype)

        if backend.backend() == "tensorflow":
            import tensorflow as tf

            indices = tf.SparseTensor([[0, 0], [1, 2]], [-1, 2], (2, 3))
        elif backend.backend() == "jax":
            import jax.experimental.sparse as jax_sparse

            indices = jax_sparse.BCOO(([-1, 2], [[0, 0], [1, 2]]), shape=(2, 3))

        self.assertAllClose(
            knp.take(x, indices, axis=axis),
            np.take(x, backend.convert_to_numpy(indices), axis=axis),
        )

    @parameterized.named_parameters(
        named_product(
            [
                {"testcase_name": "axis_none", "axis": None},
                {"testcase_name": "axis_0", "axis": 0},
                {"testcase_name": "axis_1", "axis": 1},
                {"testcase_name": "axis_minus1", "axis": -1},
            ],
            dtype=[
                "float16",
                "float32",
                "float64",
                "uint8",
                "int8",
                "int16",
                "int32",
            ],
        )
    )
    @pytest.mark.skipif(
        not backend.SUPPORTS_RAGGED_TENSORS,
        reason="Backend does not support ragged tensors.",
    )
    def test_take_ragged(self, dtype, axis):
        rng = np.random.default_rng(0)
        x = (4 * rng.standard_normal((3, 4, 5))).astype(dtype)

        if backend.backend() == "tensorflow":
            import tensorflow as tf

            indices = tf.ragged.constant([[2], [0, -1, 1]])
            mask = backend.convert_to_numpy(tf.ones_like(indices))

        if axis == 0:
            mask = np.expand_dims(mask, (2, 3))
        elif axis == 1:
            mask = np.expand_dims(mask, (2,))

        self.assertAllClose(
            knp.take(x, indices, axis=axis),
            np.take(x, backend.convert_to_numpy(indices), axis=axis)
            * mask.astype(dtype),
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

        # Test with axis=None
        x = np.arange(12).reshape([1, 1, 3, 4])
        indices = np.array([1, 2, 3], dtype=np.int32)
        self.assertAllClose(
            knp.take_along_axis(x, indices, axis=None),
            np.take_along_axis(x, indices, axis=None),
        )
        self.assertAllClose(
            knp.TakeAlongAxis(axis=None)(x, indices),
            np.take_along_axis(x, indices, axis=None),
        )

        # Test with negative indices
        x = np.arange(12).reshape([1, 1, 3, 4])
        indices = np.full([1, 4, 1, 1], -1, dtype=np.int32)
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
        self.assertAllClose(
            knp.Tensordot(axes=[0, 2])(x, y),
            np.tensordot(x, y, axes=[0, 2]),
        )

    def test_vdot(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        self.assertAllClose(knp.vdot(x, y), np.vdot(x, y))
        self.assertAllClose(knp.Vdot()(x, y), np.vdot(x, y))

    def test_inner(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        self.assertAllClose(knp.inner(x, y), np.inner(x, y))
        self.assertAllClose(knp.Inner()(x, y), np.inner(x, y))

    def test_where(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        self.assertAllClose(knp.where(x > 1, x, y), np.where(x > 1, x, y))
        self.assertAllClose(knp.Where()(x > 1, x, y), np.where(x > 1, x, y))
        self.assertAllClose(knp.where(x > 1), np.where(x > 1))
        self.assertAllClose(knp.Where()(x > 1), np.where(x > 1))

        with self.assertRaisesRegex(
            ValueError, "`x1` and `x2` either both should be `None`"
        ):
            knp.where(x > 1, x, None)

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

        # test overflow
        x = np.array([65504, 65504, 65504], dtype="float16")
        self.assertAllClose(knp.mean(x), np.mean(x))

    def test_array_split(self):
        x = np.array([[1, 2, 3], [4, 5, 6]])

        # Even split (axis=0)
        knp_res1 = knp.array_split(x, 2)
        np_res1 = np.array_split(x, 2)
        self.assertEqual(len(knp_res1), len(np_res1))
        for k_arr, n_arr in zip(knp_res1, np_res1):
            self.assertAllClose(k_arr, n_arr)

        # Even split (axis=1)
        knp_res2 = knp.array_split(x, 3, axis=1)
        np_res2 = np.array_split(x, 3, axis=1)
        self.assertEqual(len(knp_res2), len(np_res2))
        for k_arr, n_arr in zip(knp_res2, np_res2):
            self.assertAllClose(k_arr, n_arr)

        # Uneven split (axis=1) - 3 columns into 2 sections
        knp_res3 = knp.array_split(x, 2, axis=1)
        np_res3 = np.array_split(x, 2, axis=1)
        self.assertEqual(len(knp_res3), len(np_res3))
        for k_arr, n_arr in zip(knp_res3, np_res3):
            self.assertAllClose(k_arr, n_arr)

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

    def test_trapezoid(self):
        y = np.random.random((3, 3, 3))
        x = np.random.random((3, 3, 3))
        dx = 2.0

        self.assertAllClose(knp.trapezoid(y), np.trapezoid(y))
        self.assertAllClose(knp.trapezoid(y, x=x), np.trapezoid(y, x=x))
        self.assertAllClose(knp.trapezoid(y, dx=dx), np.trapezoid(y, dx=dx))
        self.assertAllClose(
            knp.trapezoid(y, x=x, axis=1),
            np.trapezoid(y, x=x, axis=1),
        )

    def test_vander(self):
        x = np.random.random((3,))
        N = 6

        self.assertAllClose(knp.vander(x), np.vander(x))
        self.assertAllClose(knp.vander(x, N=N), np.vander(x, N=N))
        self.assertAllClose(
            knp.vander(x, N=N, increasing=True),
            np.vander(x, N=N, increasing=True),
        )

        self.assertAllClose(knp.Vander().call(x), np.vander(x))
        self.assertAllClose(knp.Vander(N=N).call(x), np.vander(x, N=N))
        self.assertAllClose(
            knp.Vander(N=N, increasing=True).call(x),
            np.vander(x, N=N, increasing=True),
        )
        self.assertAllClose(
            knp.Vander(N=N, increasing=False).call(x),
            np.vander(x, N=N, increasing=False),
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
        x = np.ones([1, 3, 1, 5])
        self.assertAllClose(knp.squeeze(x), np.squeeze(x))
        self.assertAllClose(knp.squeeze(x, axis=0), np.squeeze(x, axis=0))

        self.assertAllClose(knp.Squeeze()(x), np.squeeze(x))
        self.assertAllClose(knp.Squeeze(axis=0)(x), np.squeeze(x, axis=0))

        # Multiple axes
        x = np.ones([2, 1, 1, 1])
        self.assertAllClose(knp.squeeze(x, (1, 2)), np.squeeze(x, (1, 2)))
        self.assertAllClose(knp.squeeze(x, (-1, -2)), np.squeeze(x, (-1, -2)))
        self.assertAllClose(knp.squeeze(x, (1, 2, 3)), np.squeeze(x, (1, 2, 3)))
        self.assertAllClose(knp.squeeze(x, (-1, 1)), np.squeeze(x, (-1, 1)))

        self.assertAllClose(knp.Squeeze((1, 2))(x), np.squeeze(x, (1, 2)))
        self.assertAllClose(knp.Squeeze((-1, -2))(x), np.squeeze(x, (-1, -2)))
        self.assertAllClose(knp.Squeeze((1, 2, 3))(x), np.squeeze(x, (1, 2, 3)))
        self.assertAllClose(knp.Squeeze((-1, 1))(x), np.squeeze(x, (-1, 1)))

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
        x = np.array([[1, 2, 3], [3, 2, 1], [4, 5, 6]])
        self.assertAllClose(knp.argmax(x), np.argmax(x))
        self.assertAllClose(knp.argmax(x, axis=1), np.argmax(x, axis=1))
        self.assertAllClose(
            knp.argmax(x, axis=1, keepdims=True),
            np.argmax(x, axis=1, keepdims=True),
        )
        self.assertAllClose(
            knp.argmax(x, keepdims=True), np.argmax(x, keepdims=True)
        )

        self.assertAllClose(knp.Argmax()(x), np.argmax(x))
        self.assertAllClose(knp.Argmax(axis=1)(x), np.argmax(x, axis=1))

        self.assertAllClose(knp.Argmax()(x), np.argmax(x))
        self.assertAllClose(
            knp.Argmax(keepdims=True)(x), np.argmax(x, keepdims=True)
        )

    def test_argmin(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.argmin(x), np.argmin(x))
        self.assertAllClose(knp.argmin(x, axis=1), np.argmin(x, axis=1))
        self.assertAllClose(
            knp.argmin(x, keepdims=True), np.argmin(x, keepdims=True)
        )

        self.assertAllClose(knp.Argmin()(x), np.argmin(x))
        self.assertAllClose(knp.Argmin(axis=1)(x), np.argmin(x, axis=1))
        self.assertAllClose(
            knp.Argmin(keepdims=True)(x), np.argmin(x, keepdims=True)
        )

    def test_argsort(self):
        x = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertAllClose(knp.argsort(x), np.argsort(x))
        self.assertAllClose(knp.argsort(x, axis=1), np.argsort(x, axis=1))
        self.assertAllClose(knp.argsort(x, axis=None), np.argsort(x, axis=None))

        self.assertAllClose(knp.Argsort()(x), np.argsort(x))
        self.assertAllClose(knp.Argsort(axis=1)(x), np.argsort(x, axis=1))
        self.assertAllClose(knp.Argsort(axis=None)(x), np.argsort(x, axis=None))

        x = np.array(1)  # rank == 0
        self.assertAllClose(knp.argsort(x), np.argsort(x))
        self.assertAllClose(knp.Argsort()(x), np.argsort(x))

    def test_array(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.array(x), np.array(x))
        self.assertAllClose(knp.Array()(x), np.array(x))
        self.assertTrue(backend.is_tensor(knp.array(x)))
        self.assertTrue(backend.is_tensor(knp.Array()(x)))

        # Check dtype conversion.
        x = [[1, 0, 1], [1, 1, 0]]
        output = knp.array(x, dtype="int32")
        self.assertEqual(standardize_dtype(output.dtype), "int32")
        x = [[1, 0, 1], [1, 1, 0]]
        output = knp.array(x, dtype="float32")
        self.assertEqual(standardize_dtype(output.dtype), "float32")
        x = [[1, 0, 1], [1, 1, 0]]
        output = knp.array(x, dtype="bool")
        self.assertEqual(standardize_dtype(output.dtype), "bool")

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

    def test_bartlett(self):
        x = np.random.randint(1, 100 + 1)
        self.assertAllClose(knp.bartlett(x), np.bartlett(x))

        self.assertAllClose(knp.Bartlett()(x), np.bartlett(x))

    def test_blackman(self):
        x = np.random.randint(1, 100 + 1)
        self.assertAllClose(knp.blackman(x), np.blackman(x))

        self.assertAllClose(knp.Blackman()(x), np.blackman(x))

    def test_hamming(self):
        x = np.random.randint(1, 100 + 1)
        self.assertAllClose(knp.hamming(x), np.hamming(x))

        self.assertAllClose(knp.Hamming()(x), np.hamming(x))

    def test_hanning(self):
        x = np.random.randint(1, 100 + 1)
        self.assertAllClose(knp.hanning(x), np.hanning(x))

        self.assertAllClose(knp.Hanning()(x), np.hanning(x))

    def test_kaiser(self):
        x = np.random.randint(1, 100 + 1)
        beta = float(np.random.randint(10, 20 + 1))
        self.assertAllClose(knp.kaiser(x, beta), np.kaiser(x, beta))

        self.assertAllClose(knp.Kaiser(beta)(x), np.kaiser(x, beta))

    @parameterized.named_parameters(
        named_product(sparse_input=(False, True), sparse_arg=(False, True))
    )
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(),
        reason="bincount not supported on TensorFlow GPU",
    )
    def test_bincount(self, sparse_input, sparse_arg):
        if (sparse_input or sparse_arg) and not backend.SUPPORTS_SPARSE_TENSORS:
            pytest.skip("Backend does not support sparse tensors")

        x = x_np = np.array([1, 1, 2, 3, 2, 4, 4, 6])
        weights = weights_np = np.array([0, 0, 3, 2, 1, 1, 4, 2])
        if sparse_input:
            indices = np.array([[1], [3], [5], [7], [9], [11], [13], [15]])

            if backend.backend() == "tensorflow":
                import tensorflow as tf

                x = tf.SparseTensor(indices, x, (16,))
                weights = tf.SparseTensor(indices, weights, (16,))
            elif backend.backend() == "jax":
                from jax.experimental import sparse as jax_sparse

                x = jax_sparse.BCOO((x, indices), shape=(16,))
                weights = jax_sparse.BCOO((weights, indices), shape=(16,))

        minlength = 3
        output = knp.bincount(
            x, weights=weights, minlength=minlength, sparse=sparse_arg
        )
        self.assertAllClose(
            output, np.bincount(x_np, weights=weights_np, minlength=minlength)
        )
        self.assertSparse(output, sparse_input or sparse_arg)
        output = knp.Bincount(
            weights=weights, minlength=minlength, sparse=sparse_arg
        )(x)
        self.assertAllClose(
            output, np.bincount(x_np, weights=weights_np, minlength=minlength)
        )
        self.assertSparse(output, sparse_input or sparse_arg)

        x = knp.expand_dims(x, 0)
        weights = knp.expand_dims(weights, 0)

        expected_output = np.array([[0, 0, 4, 2, 5, 0, 2]])
        output = knp.bincount(
            x, weights=weights, minlength=minlength, sparse=sparse_arg
        )
        self.assertAllClose(output, expected_output)
        self.assertSparse(output, sparse_input or sparse_arg)
        output = knp.Bincount(
            weights=weights, minlength=minlength, sparse=sparse_arg
        )(x)
        self.assertAllClose(output, expected_output)
        self.assertSparse(output, sparse_input or sparse_arg)

        # test with weights=None
        expected_output = np.array([[0, 2, 2, 1, 2, 0, 1]])
        output = knp.Bincount(
            weights=None, minlength=minlength, sparse=sparse_arg
        )(x)
        self.assertAllClose(output, expected_output)
        self.assertSparse(output, sparse_input or sparse_arg)

    def test_bitwise_invert(self):
        x = np.array([2, 5, 255])
        self.assertAllClose(knp.bitwise_invert(x), np.bitwise_not(x))
        self.assertAllClose(knp.BitwiseInvert()(x), np.bitwise_not(x))

    # bitwise_not is same as bitwise_invert

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

    def test_cbrt(self):
        x = np.array([[-8, -1, 0], [1, 8, 27]], dtype="float32")
        ref_y = np.sign(x) * np.abs(x) ** (1.0 / 3.0)
        y = knp.cbrt(x)
        self.assertEqual(standardize_dtype(y.dtype), "float32")
        self.assertAllClose(y, ref_y)

        y = knp.Cbrt()(x)
        self.assertEqual(standardize_dtype(y.dtype), "float32")
        self.assertAllClose(y, ref_y)

    def test_ceil(self):
        x = np.array([[1.2, 2.1, -2.5], [2.4, -11.9, -5.5]])
        self.assertAllClose(knp.ceil(x), np.ceil(x))
        self.assertAllClose(knp.Ceil()(x), np.ceil(x))

    def test_clip(self):
        x = np.array([[1.2, 2.1, 0.5], [2.4, 11.9, 0.5]])
        self.assertAllClose(knp.clip(x, 1, 2), np.clip(x, 1, 2))
        self.assertAllClose(knp.clip(x, 1, 2), np.clip(x, 1, 2))

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

    def test_view(self):
        x = np.array(1, dtype="int16")
        result = knp.view(x, dtype="float16")
        assert backend.standardize_dtype(result.dtype) == "float16"

        with self.assertRaises(Exception):
            result = knp.view(x, dtype="int8")

        with self.assertRaises(Exception):
            result = knp.view(x, dtype="int32")

        x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype="int16")
        result = knp.view(x, dtype="int16")
        assert backend.standardize_dtype(result.dtype) == "int16"

        self.assertEqual(
            backend.standardize_dtype(knp.view(x, dtype="int16").dtype), "int16"
        )
        self.assertAllClose(knp.view(x, dtype="int16"), x.view("int16"))

        self.assertEqual(
            backend.standardize_dtype(knp.view(x, dtype="float16").dtype),
            "float16",
        )
        self.assertAllClose(knp.view(x, dtype="float16"), x.view("float16"))

        self.assertEqual(
            backend.standardize_dtype(knp.view(x, dtype="int8").dtype), "int8"
        )
        self.assertAllClose(knp.view(x, dtype="int8"), x.view("int8"))

        self.assertEqual(
            backend.standardize_dtype(knp.view(x, dtype="int32").dtype), "int32"
        )
        self.assertAllClose(knp.view(x, dtype="int32"), x.view("int32"))

    @parameterized.named_parameters(
        [
            {"testcase_name": "axis_0", "axis": 0},
            {"testcase_name": "axis_1", "axis": 1},
        ]
    )
    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason="Backend does not support sparse tensors.",
    )
    def test_concatenate_sparse(self, axis):
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            x = tf.SparseTensor([[0, 0], [1, 2]], [1.0, 2.0], (2, 3))
            y = tf.SparseTensor([[0, 0], [1, 1]], [4.0, 5.0], (2, 3))
        elif backend.backend() == "jax":
            import jax.experimental.sparse as jax_sparse

            x = jax_sparse.BCOO(([1.0, 2.0], [[0, 0], [1, 2]]), shape=(2, 3))
            y = jax_sparse.BCOO(([4.0, 5.0], [[0, 0], [1, 1]]), shape=(2, 3))

        x_np = backend.convert_to_numpy(x)
        y_np = backend.convert_to_numpy(y)
        z = np.random.rand(2, 3).astype("float32")

        self.assertAllClose(
            knp.concatenate([x, z], axis=axis),
            np.concatenate([x_np, z], axis=axis),
        )
        self.assertAllClose(
            knp.concatenate([z, x], axis=axis),
            np.concatenate([z, x_np], axis=axis),
        )
        self.assertAllClose(
            knp.concatenate([x, y], axis=axis),
            np.concatenate([x_np, y_np], axis=axis),
        )

        self.assertAllClose(
            knp.Concatenate(axis=axis)([x, z]),
            np.concatenate([x_np, z], axis=axis),
        )
        self.assertAllClose(
            knp.Concatenate(axis=axis)([z, x]),
            np.concatenate([z, x_np], axis=axis),
        )
        self.assertAllClose(
            knp.Concatenate(axis=axis)([x, y]),
            np.concatenate([x_np, y_np], axis=axis),
        )

        self.assertSparse(knp.concatenate([x, y], axis=axis))
        self.assertSparse(knp.Concatenate(axis=axis)([x, y]))

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

    def test_corrcoef(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.corrcoef(x), np.corrcoef(x))
        self.assertAllClose(knp.Corrcoef()(x), np.corrcoef(x))

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

    @parameterized.product(
        axis=[None, 0, 1, -1],
        dtype=[None, "int32", "float32"],
    )
    def test_cumprod(self, axis, dtype):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(
            knp.cumprod(x, axis=axis, dtype=dtype),
            np.cumprod(x, axis=axis, dtype=dtype or x.dtype),
        )
        self.assertAllClose(
            knp.Cumprod(axis=axis, dtype=dtype)(x),
            np.cumprod(x, axis=axis, dtype=dtype or x.dtype),
        )

    @parameterized.product(
        axis=[None, 0, 1, -1],
        dtype=[None, "int32", "float32"],
    )
    def test_cumsum(self, axis, dtype):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(
            knp.cumsum(x, axis=axis, dtype=dtype),
            np.cumsum(x, axis=axis, dtype=dtype or x.dtype),
        )
        self.assertAllClose(
            knp.Cumsum(axis=axis, dtype=dtype)(x),
            np.cumsum(x, axis=axis, dtype=dtype or x.dtype),
        )

    def test_deg2rad(self):
        x = np.random.uniform(-360, 360, size=(3, 3))
        self.assertAllClose(knp.deg2rad(x), np.deg2rad(x))
        self.assertAllClose(knp.Deg2rad()(x), np.deg2rad(x))

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

    def test_diagflat(self):
        x = np.array([1, 2, 3])
        self.assertAllClose(knp.diagflat(x), np.diagflat(x))
        self.assertAllClose(knp.diagflat(x, k=1), np.diagflat(x, k=1))
        self.assertAllClose(knp.diagflat(x, k=-1), np.diagflat(x, k=-1))

        x = np.array([[1, 2], [3, 4]])
        self.assertAllClose(knp.diagflat(x), np.diagflat(x))
        self.assertAllClose(knp.diagflat(x, k=1), np.diagflat(x, k=1))
        self.assertAllClose(knp.diagflat(x, k=-1), np.diagflat(x, k=-1))

        x = np.array([1, 2, 3, 4])
        self.assertAllClose(knp.diagflat(x), np.diagflat(x))
        self.assertAllClose(knp.diagflat(x, k=2), np.diagflat(x, k=2))
        self.assertAllClose(knp.diagflat(x, k=-2), np.diagflat(x, k=-2))

        x_float = np.array([1.1, 2.2, 3.3])
        self.assertAllClose(knp.diagflat(x_float), np.diagflat(x_float))

        x = np.array([1, 2, 3])
        self.assertAllClose(knp.Diagflat()(x), np.diagflat(x))
        self.assertAllClose(knp.Diagflat(k=1)(x), np.diagflat(x, k=1))
        self.assertAllClose(knp.Diagflat(k=-1)(x), np.diagflat(x, k=-1))

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

    def test_diff(self):
        x = np.array([1, 2, 4, 7, 0])
        self.assertAllClose(knp.diff(x), np.diff(x))
        self.assertAllClose(knp.diff(x, n=2), np.diff(x, n=2))
        self.assertAllClose(knp.diff(x, n=3), np.diff(x, n=3))

        x = np.array([[1, 3, 6, 10], [0, 5, 6, 8]])
        self.assertAllClose(knp.diff(x), np.diff(x))
        self.assertAllClose(knp.diff(x, axis=0), np.diff(x, axis=0))
        self.assertAllClose(knp.diff(x, n=2, axis=0), np.diff(x, n=2, axis=0))
        self.assertAllClose(knp.diff(x, n=2, axis=1), np.diff(x, n=2, axis=1))

        # Test n=0
        x = np.array([1, 2, 4, 7, 0])
        self.assertAllClose(knp.diff(x, n=0), np.diff(x, n=0))

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

    def test_exp2(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.exp2(x), np.exp2(x))
        self.assertAllClose(knp.Exp2()(x), np.exp2(x))

    def test_expand_dims(self):
        x = np.ones([2, 3, 4])
        self.assertAllClose(knp.expand_dims(x, 0), np.expand_dims(x, 0))
        self.assertAllClose(knp.expand_dims(x, 1), np.expand_dims(x, 1))
        self.assertAllClose(knp.expand_dims(x, -2), np.expand_dims(x, -2))

        self.assertAllClose(knp.ExpandDims(0)(x), np.expand_dims(x, 0))
        self.assertAllClose(knp.ExpandDims(1)(x), np.expand_dims(x, 1))
        self.assertAllClose(knp.ExpandDims(-2)(x), np.expand_dims(x, -2))

        # Multiple axes
        self.assertAllClose(
            knp.expand_dims(x, (1, 2)), np.expand_dims(x, (1, 2))
        )
        self.assertAllClose(
            knp.expand_dims(x, (-1, -2)), np.expand_dims(x, (-1, -2))
        )
        self.assertAllClose(
            knp.expand_dims(x, (-1, 1)), np.expand_dims(x, (-1, 1))
        )

        self.assertAllClose(
            knp.ExpandDims((1, 2))(x), np.expand_dims(x, (1, 2))
        )
        self.assertAllClose(
            knp.ExpandDims((-1, -2))(x), np.expand_dims(x, (-1, -2))
        )
        self.assertAllClose(
            knp.ExpandDims((-1, 1))(x), np.expand_dims(x, (-1, 1))
        )

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

    def test_isneginf(self):
        x = np.array(
            [[1, 2, np.inf, -np.inf], [np.nan, np.nan, np.nan, np.nan]]
        )
        self.assertAllClose(knp.isneginf(x), np.isneginf(x))
        self.assertAllClose(knp.Isneginf()(x), np.isneginf(x))

    def test_isposinf(self):
        x = np.array(
            [[1, 2, np.inf, -np.inf], [np.nan, np.nan, np.nan, np.nan]]
        )
        self.assertAllClose(knp.isposinf(x), np.isposinf(x))
        self.assertAllClose(knp.Isposinf()(x), np.isposinf(x))

    def test_isreal(self):
        x = np.array([1 + 1j, 1 + 0j, 4.5, 3, 2, 2j], dtype=complex)
        self.assertAllClose(knp.isreal(x), np.isreal(x))
        self.assertAllClose(knp.Isreal()(x), np.isreal(x))

        x = np.array([1.0, 2.0, 3.0])
        self.assertAllClose(knp.isreal(x), np.isreal(x))
        self.assertAllClose(knp.Isreal()(x), np.isreal(x))

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

    def test_logaddexp2(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.logaddexp2(x, y), np.logaddexp2(x, y))
        self.assertAllClose(knp.Logaddexp2()(x, y), np.logaddexp2(x, y))

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

        self.assertAllClose(knp.min(x, axis=(0, 1)), np.min(x, (0, 1)))
        self.assertAllClose(knp.Min((0, 1))(x), np.min(x, (0, 1)))

        self.assertAllClose(knp.min(x, axis=()), np.min(x, axis=()))
        self.assertAllClose(knp.Min(())(x), np.min(x, axis=()))

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

    def test_median(self):
        x = np.array([[1, 2, 3], [3, 2, 1]]).astype("float32")
        self.assertAllClose(knp.median(x), np.median(x))
        self.assertAllClose(
            knp.median(x, keepdims=True), np.median(x, keepdims=True)
        )
        self.assertAllClose(knp.median(x, axis=1), np.median(x, axis=1))
        self.assertAllClose(knp.median(x, axis=(1,)), np.median(x, axis=(1,)))
        self.assertAllClose(
            knp.median(x, axis=1, keepdims=True),
            np.median(x, axis=1, keepdims=True),
        )

        self.assertAllClose(knp.Median()(x), np.median(x))
        self.assertAllClose(knp.Median(axis=1)(x), np.median(x, axis=1))
        self.assertAllClose(
            knp.Median(axis=1, keepdims=True)(x),
            np.median(x, axis=1, keepdims=True),
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

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Only test tensorflow backend",
    )
    def test_ndim_tf_ragged(self):
        import tensorflow as tf

        # Rank 2
        ragged_2d = tf.ragged.constant([[1, 2, 3], [4]])
        self.assertEqual(knp.ndim(ragged_2d), 2)
        self.assertEqual(knp.Ndim()(ragged_2d), 2)
        # Rank 0
        ragged_scalar = tf.ragged.constant(1)
        self.assertEqual(knp.ndim(ragged_scalar), 0)
        self.assertEqual(knp.Ndim()(ragged_scalar), 0)
        # Rank 3
        ragged_3d = tf.ragged.constant([[[1], [2, 3]], [[4, 5, 6]]])
        self.assertEqual(knp.ndim(ragged_3d), 3)
        self.assertEqual(knp.Ndim()(ragged_3d), 3)

    def test_nonzero(self):
        x = np.array([[0, 0, 3], [3, 0, 0]])
        self.assertAllClose(knp.nonzero(x), np.nonzero(x))
        self.assertAllClose(knp.Nonzero()(x), np.nonzero(x))

    def test_ones_like(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.ones_like(x), np.ones_like(x))
        self.assertAllClose(knp.OnesLike()(x), np.ones_like(x))

    @parameterized.named_parameters(
        named_product(
            dtype=[
                "float16",
                "float32",
                "float64",
                "uint8",
                "int8",
                "int16",
                "int32",
            ],
            mode=["constant", "reflect", "symmetric"],
            constant_values=[None, 0, 2],
        )
    )
    def test_pad(self, dtype, mode, constant_values):
        # 2D
        x = np.ones([2, 3], dtype=dtype)
        pad_width = ((1, 1), (1, 1))

        if mode != "constant":
            if constant_values is not None:
                with self.assertRaisesRegex(
                    ValueError,
                    "Argument `constant_values` can only be "
                    "provided when `mode == 'constant'`",
                ):
                    knp.pad(
                        x, pad_width, mode=mode, constant_values=constant_values
                    )
                return
            # constant_values is None
            kwargs = {}
        else:
            # mode is constant
            kwargs = {"constant_values": constant_values or 0}

        self.assertAllClose(
            knp.pad(x, pad_width, mode=mode, constant_values=constant_values),
            np.pad(x, pad_width, mode=mode, **kwargs),
        )
        self.assertAllClose(
            knp.Pad(pad_width, mode=mode)(x, constant_values=constant_values),
            np.pad(x, pad_width, mode=mode, **kwargs),
        )

        # 5D (pad last 3D)
        x = np.ones([2, 3, 4, 5, 6], dtype=dtype)
        pad_width = ((0, 0), (0, 0), (2, 3), (1, 1), (1, 1))
        self.assertAllClose(
            knp.pad(x, pad_width, mode=mode, constant_values=constant_values),
            np.pad(x, pad_width, mode=mode, **kwargs),
        )
        self.assertAllClose(
            knp.Pad(pad_width, mode=mode)(x, constant_values=constant_values),
            np.pad(x, pad_width, mode=mode, **kwargs),
        )

        # 5D (pad arbitrary dimensions)
        if backend.backend() == "torch" and mode != "constant":
            self.skipTest(
                "reflect and symmetric padding for arbitrary dimensions "
                "are not supported by torch"
            )
        x = np.ones([2, 3, 4, 5, 6], dtype=dtype)
        pad_width = ((1, 1), (2, 1), (3, 2), (4, 3), (5, 4))
        self.assertAllClose(
            knp.pad(x, pad_width, mode=mode, constant_values=constant_values),
            np.pad(x, pad_width, mode=mode, **kwargs),
        )
        self.assertAllClose(
            knp.Pad(pad_width, mode=mode)(x, constant_values=constant_values),
            np.pad(x, pad_width, mode=mode, **kwargs),
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

    def test_ptp(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])

        self.assertAllClose(knp.ptp(x), np.ptp(x))
        self.assertAllClose(knp.ptp(x, axis=None), np.ptp(x, axis=None))

        self.assertAllClose(knp.ptp(x, axis=0), np.ptp(x, axis=0))
        self.assertAllClose(knp.ptp(x, axis=1), np.ptp(x, axis=1))
        self.assertAllClose(knp.ptp(x, axis=(1,)), np.ptp(x, axis=(1,)))

        self.assertAllClose(knp.ptp(x, axis=()), np.ptp(x, axis=()))

        self.assertAllClose(
            knp.ptp(x, axis=1, keepdims=True),
            np.ptp(x, axis=1, keepdims=True),
        )

        self.assertAllClose(knp.Ptp()(x), np.ptp(x))
        self.assertAllClose(knp.Ptp(axis=1)(x), np.ptp(x, axis=1))
        self.assertAllClose(knp.Ptp(axis=(0, 1))(x), np.ptp(x, axis=(0, 1)))
        self.assertAllClose(
            knp.Ptp(axis=1, keepdims=True)(x),
            np.ptp(x, axis=1, keepdims=True),
        )

    def test_ravel(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        self.assertAllClose(knp.ravel(x), np.ravel(x))
        self.assertAllClose(knp.Ravel()(x), np.ravel(x))

    def test_unravel_index(self):
        x = np.array([0, 1, 2, 3])
        shape = (2, 2)
        self.assertAllClose(
            knp.unravel_index(x, shape), np.unravel_index(x, shape)
        )

        x = np.array([[0, 1], [2, 3]])
        shape = (2, 2)
        self.assertAllClose(
            knp.unravel_index(x, shape), np.unravel_index(x, shape)
        )

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
        self.assertAllClose(
            knp.Repeat(np.array([2]))(x),
            np.repeat(x, np.array([2])),
        )
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
        self.assertAllClose(knp.Reshape(-1)(x), np.reshape(x, -1))

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

        # Test with decimal=1
        self.assertAllClose(knp.round(x, decimals=1), np.round(x, decimals=1))
        self.assertAllClose(knp.Round(decimals=1)(x), np.round(x, decimals=1))

        # Test with integers
        x = np.array([[1, 2, 3], [3, 2, 1]], dtype="int32")
        self.assertAllClose(knp.round(x, decimals=1), np.round(x, decimals=1))
        self.assertAllClose(knp.Round(decimals=1)(x), np.round(x, decimals=1))

        # Test with integers and decimal < 0
        x = np.array([[123, 234, 345], [345, 234, 123]], dtype="int32")
        self.assertAllClose(knp.round(x, decimals=-1), np.round(x, decimals=-1))
        self.assertAllClose(knp.Round(decimals=-1)(x), np.round(x, decimals=-1))

    def test_searchsorted(self):
        a = np.array([1, 2, 2, 3, 4, 5, 5])
        v = np.array([4, 3, 5, 1, 2])
        expected = np.searchsorted(a, v).astype("int32")
        self.assertAllEqual(knp.searchsorted(a, v), expected)
        self.assertAllEqual(knp.SearchSorted()(a, v), expected)

    def test_sign(self):
        x = np.array([[1, -2, 3], [-3, 2, -1]])
        self.assertAllClose(knp.sign(x), np.sign(x))
        self.assertAllClose(knp.Sign()(x), np.sign(x))

    def test_signbit(self):
        x = np.array([[0.0, -0.0, -1.1e-45], [1.1e-38, 2, -1]])
        self.assertAllClose(knp.signbit(x), np.signbit(x))
        self.assertAllClose(knp.Signbit()(x), np.signbit(x))

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
        self.assertIsInstance(knp.split(x, 2), list)
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

        # test invalid indices_or_sections
        with self.assertRaises(Exception):
            knp.split(x, 3)

        # test zero dimension
        x = np.ones(shape=(0,))
        self.assertEqual(len(knp.split(x, 2)), 2)
        self.assertEqual(len(knp.Split(2)(x)), 2)

        # test indices_or_sections as tensor
        x = knp.array([[1, 2, 3], [3, 2, 1]])
        indices_or_sections = knp.array([1, 2])
        x_np = np.array([[1, 2, 3], [3, 2, 1]])
        indices_or_sections_np = np.array([1, 2])
        self.assertAllClose(
            knp.split(x, indices_or_sections, axis=1),
            np.split(x_np, indices_or_sections_np, axis=1),
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Only test tensorflow backend",
    )
    def test_split_with_jit_in_tf(self):
        import tensorflow as tf

        x = knp.array([[1, 2, 3], [3, 2, 1]])
        indices = knp.array([1, 2])
        x_np = np.array([[1, 2, 3], [3, 2, 1]])
        indices_np = np.array([1, 2])

        @tf.function(jit_compile=True)
        def fn(x, indices, axis):
            return knp.split(x, indices, axis=axis)

        self.assertAllClose(
            fn(x, indices, axis=1),
            np.split(x_np, indices_np, axis=1),
        )

    def test_hsplit(self):
        x = np.arange(18).reshape((3, 6))

        self.assertIsInstance(knp.hsplit(x, 3), list)
        self.assertAllClose(knp.hsplit(x, 3), np.hsplit(x, 3))
        self.assertAllClose(knp.Hsplit(3)(x), np.hsplit(x, 3))

        indices = [1, 3, 5]

        # Compare each split
        for split_knp, split_np in zip(
            knp.hsplit(x, indices), np.hsplit(x, indices)
        ):
            self.assertAllClose(split_knp, split_np)

        for split_knp, split_np in zip(
            knp.Hsplit(indices)(x), np.hsplit(x, indices)
        ):
            self.assertAllClose(split_knp, split_np)

        with self.assertRaises(Exception):
            knp.hsplit(x, 4)

        x_kr = knp.array(x)
        indices_kr = knp.array(indices)
        indices_np = np.array(indices)

        for split_knp, split_np in zip(
            knp.hsplit(x_kr, indices_kr), np.hsplit(x, indices_np)
        ):
            self.assertAllClose(split_knp, split_np)

        # Test 1D case
        x_1d = np.arange(10)
        indices_1d = [2, 5, 9]

        self.assertIsInstance(knp.hsplit(x_1d, 2), list)
        self.assertAllClose(knp.hsplit(x_1d, 2), np.hsplit(x_1d, 2))
        self.assertAllClose(knp.Hsplit(2)(x_1d), np.hsplit(x_1d, 2))

        for split_knp, split_np in zip(
            knp.hsplit(x_1d, indices_1d), np.hsplit(x_1d, indices_1d)
        ):
            self.assertAllClose(split_knp, split_np)

        for split_knp, split_np in zip(
            knp.Hsplit(indices_1d)(x_1d), np.hsplit(x_1d, indices_1d)
        ):
            self.assertAllClose(split_knp, split_np)

        with self.assertRaises(Exception):
            knp.hsplit(x_1d, 3)

        x_kr = knp.array(x_1d)
        indices_kr = knp.array(indices_1d)
        indices_np = np.array(indices_1d)

        for split_knp, split_np in zip(
            knp.hsplit(x_kr, indices_kr), np.hsplit(x_1d, indices_np)
        ):
            self.assertAllClose(split_knp, split_np)

    def test_vsplit(self):
        x = np.arange(18).reshape((6, 3))

        self.assertIsInstance(knp.vsplit(x, 3), list)
        self.assertAllClose(knp.vsplit(x, 3), np.vsplit(x, 3))
        self.assertAllClose(knp.Vsplit(3)(x), np.vsplit(x, 3))

        indices = [1, 3, 5]

        # Compare each split
        for split_knp, split_np in zip(
            knp.vsplit(x, indices), np.vsplit(x, indices)
        ):
            self.assertAllClose(split_knp, split_np)

        for split_knp, split_np in zip(
            knp.Vsplit(indices)(x), np.vsplit(x, indices)
        ):
            self.assertAllClose(split_knp, split_np)

        with self.assertRaises(Exception):
            knp.vsplit(x, 4)

        x_kr = knp.array(x)
        indices_kr = knp.array(indices)
        indices_np = np.array(indices)

        for split_knp, split_np in zip(
            knp.vsplit(x_kr, indices_kr), np.vsplit(x, indices_np)
        ):
            self.assertAllClose(split_knp, split_np)

    def test_sqrt(self):
        x = np.array([[1, 4, 9], [16, 25, 36]], dtype="float32")
        ref_y = np.sqrt(x)
        y = knp.sqrt(x)
        self.assertEqual(standardize_dtype(y.dtype), "float32")
        self.assertAllClose(y, ref_y)
        y = knp.Sqrt()(x)
        self.assertEqual(standardize_dtype(y.dtype), "float32")
        self.assertAllClose(y, ref_y)

    def test_sqrt_int32(self):
        x = np.array([[1, 4, 9], [16, 25, 36]], dtype="int32")
        ref_y = np.sqrt(x)
        y = knp.sqrt(x)
        self.assertEqual(standardize_dtype(y.dtype), "float32")
        self.assertAllClose(y, ref_y)
        y = knp.Sqrt()(x)
        self.assertEqual(standardize_dtype(y.dtype), "float32")
        self.assertAllClose(y, ref_y)

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
        self.assertAllClose(knp.tile(x, 2), np.tile(x, 2))
        self.assertAllClose(knp.tile(x, [2, 3]), np.tile(x, [2, 3]))
        self.assertAllClose(knp.Tile([2, 3])(x), np.tile(x, [2, 3]))

        # If repeats.ndim > x.ndim
        self.assertAllClose(knp.tile(x, [2, 3, 4]), np.tile(x, [2, 3, 4]))
        self.assertAllClose(knp.Tile([2, 3, 4])(x), np.tile(x, [2, 3, 4]))

        # If repeats.ndim < x.ndim
        self.assertAllClose(knp.tile(x, [2]), np.tile(x, [2]))
        self.assertAllClose(knp.Tile([2])(x), np.tile(x, [2]))

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

        x = np.ones([5, 5])
        self.assertAllClose(knp.tril(x), np.tril(x))
        self.assertAllClose(knp.tril(x, -1), np.tril(x, -1))
        self.assertAllClose(knp.Tril(-1)(x), np.tril(x, -1))

    def test_tril_in_layer(self):
        # https://github.com/keras-team/keras/issues/18890
        x = keras.Input((None, 3))
        y1 = keras.layers.Lambda(
            lambda x: keras.ops.tril(
                keras.ops.ones((keras.ops.shape(x)[1], keras.ops.shape(x)[1]))
            ),
            output_shape=(None, None, 3),
        )(x)
        y2 = keras.layers.Lambda(
            lambda x: keras.ops.tril(
                keras.ops.ones((keras.ops.shape(x)[1], keras.ops.shape(x)[1])),
                k=-1,
            ),
            output_shape=(None, None, 3),
        )(x)
        model = keras.Model(x, [y1, y2])

        result = model(np.ones((1, 2, 3), "float32"))
        self.assertAllClose(
            result, [np.tril(np.ones((2, 2))), np.tril(np.ones((2, 2)), k=-1)]
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Only test tensorflow backend",
    )
    def test_tril_with_jit_in_tf(self):
        import tensorflow as tf

        x = knp.reshape(knp.arange(24), [1, 2, 3, 4])
        k = knp.array(0)
        x_np = np.reshape(np.arange(24), [1, 2, 3, 4])
        k_np = np.array(0)

        @tf.function(jit_compile=True)
        def fn(x, k):
            return knp.tril(x, k=k)

        self.assertAllClose(fn(x, k), np.tril(x_np, k_np))

    def test_triu(self):
        x = np.arange(24).reshape([1, 2, 3, 4])
        self.assertAllClose(knp.triu(x), np.triu(x))
        self.assertAllClose(knp.triu(x, -1), np.triu(x, -1))
        self.assertAllClose(knp.Triu(-1)(x), np.triu(x, -1))

        x = np.ones([5, 5])
        self.assertAllClose(knp.triu(x), np.triu(x))
        self.assertAllClose(knp.triu(x, -1), np.triu(x, -1))
        self.assertAllClose(knp.Triu(-1)(x), np.triu(x, -1))

    def test_triu_in_layer(self):
        # https://github.com/keras-team/keras/issues/18890
        x = keras.Input((None, 3))
        y1 = keras.layers.Lambda(
            lambda x: keras.ops.triu(
                keras.ops.ones((keras.ops.shape(x)[1], keras.ops.shape(x)[1]))
            ),
            output_shape=(None, None, 3),
        )(x)
        y2 = keras.layers.Lambda(
            lambda x: keras.ops.triu(
                keras.ops.ones((keras.ops.shape(x)[1], keras.ops.shape(x)[1])),
                k=-1,
            ),
            output_shape=(None, None, 3),
        )(x)
        model = keras.Model(x, [y1, y2])

        result = model(np.ones((1, 2, 3), "float32"))
        self.assertAllClose(
            result, [np.triu(np.ones((2, 2))), np.triu(np.ones((2, 2)), k=-1)]
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Only test tensorflow backend",
    )
    def test_triu_with_jit_in_tf(self):
        import tensorflow as tf

        x = knp.reshape(knp.arange(24), [1, 2, 3, 4])
        k = knp.array(0)
        x_np = np.reshape(np.arange(24), [1, 2, 3, 4])
        k_np = np.array(0)

        @tf.function(jit_compile=True)
        def fn(x, k):
            return knp.triu(x, k=k)

        self.assertAllClose(fn(x, k), np.triu(x_np, k_np))

    def test_trunc(self):
        x = np.array([-1.7, -2.5, -0.2, 0.2, 1.5, 1.7, 2.0])
        self.assertAllClose(knp.trunc(x), np.trunc(x))
        self.assertAllClose(knp.Trunc()(x), np.trunc(x))

        x = np.array([-1, -2, -0, 0, 1, 1, 2], dtype="int32")
        self.assertAllClose(knp.trunc(x), np.trunc(x))
        self.assertAllClose(knp.Trunc()(x), np.trunc(x))

    def test_vstack(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [6, 5, 4]])
        self.assertAllClose(knp.vstack([x, y]), np.vstack([x, y]))
        self.assertAllClose(knp.Vstack()([x, y]), np.vstack([x, y]))

    def test_dstack(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [6, 5, 4]])
        self.assertAllClose(knp.dstack([x, y]), np.dstack([x, y]))
        self.assertAllClose(knp.Dstack()([x, y]), np.dstack([x, y]))

        x = np.array([1, 2, 3])
        y = np.array([[4, 5, 6]])
        self.assertAllClose(knp.dstack([x, y]), np.dstack([x, y]))

        x = np.ones([2, 3, 4])
        y = np.ones([2, 3, 5])
        self.assertAllClose(knp.dstack([x, y]), np.dstack([x, y]))

    def test_floor_divide(self):
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

    def test_correlate(self):
        x = np.array([1, 2, 3])
        y = np.array([0, 1, 0.5])
        self.assertAllClose(knp.correlate(x, y), np.correlate(x, y))
        self.assertAllClose(
            knp.correlate(x, y, mode="same"), np.correlate(x, y, mode="same")
        )
        self.assertAllClose(
            knp.correlate(x, y, mode="full"), np.correlate(x, y, mode="full")
        )

        self.assertAllClose(knp.Correlate()(x, y), np.correlate(x, y))
        self.assertAllClose(
            knp.Correlate(mode="same")(x, y), np.correlate(x, y, mode="same")
        )
        self.assertAllClose(
            knp.Correlate(mode="full")(x, y), np.correlate(x, y, mode="full")
        )

    def test_correlate_different_size(self):
        x = np.array([1, 3, 5])
        y = np.array([7, 9])
        self.assertAllClose(knp.correlate(x, y), np.correlate(x, y))
        self.assertAllClose(
            knp.correlate(x, y, mode="same"), np.correlate(x, y, mode="same")
        )
        self.assertAllClose(
            knp.correlate(x, y, mode="full"), np.correlate(x, y, mode="full")
        )

        self.assertAllClose(knp.Correlate()(x, y), np.correlate(x, y))
        self.assertAllClose(
            knp.Correlate(mode="same")(x, y), np.correlate(x, y, mode="same")
        )
        self.assertAllClose(
            knp.Correlate(mode="full")(x, y), np.correlate(x, y, mode="full")
        )

    def test_select(self):
        x = np.arange(6)
        condlist = [x < 3, x > 3]
        choicelist = [x, x**2]
        y = knp.select(condlist, choicelist, 42)
        self.assertAllClose(y, [0, 1, 2, 42, 16, 25])

        # Test with tuples
        condlist = (x < 3, x > 3)
        choicelist = (x, x**2)
        y = knp.select(condlist, choicelist, 42)
        self.assertAllClose(y, [0, 1, 2, 42, 16, 25])

        # Test with symbolic tensors
        x = backend.KerasTensor((6,))
        condlist = [x < 3, x > 3]
        choicelist = [x, x**2]
        y = knp.select(condlist, choicelist, 42)
        self.assertEqual(y.shape, (6,))

    def test_slogdet(self):
        x = np.ones((4, 4)) * 2.0
        out = knp.slogdet(x)
        self.assertAllClose(out[0], 0)
        self.assertAllClose(out[0], 0)

        x = backend.KerasTensor((3, 3))
        out = knp.slogdet(x)
        self.assertEqual(out[0].shape, ())
        self.assertEqual(out[1].shape, ())

        x = backend.KerasTensor((2, 4, 3, 3))
        out = knp.slogdet(x)
        self.assertEqual(out[0].shape, ())
        self.assertEqual(out[1].shape, (2, 4))

    def test_nanmax(self):
        x = np.array([[1.0, np.nan, 3.0], [np.nan, 2.0, -np.inf]])

        self.assertAllClose(knp.nanmax(x), np.nanmax(x))
        self.assertAllClose(knp.nanmax(x, axis=()), np.nanmax(x, axis=()))
        self.assertAllClose(knp.nanmax(x, axis=1), np.nanmax(x, axis=1))
        self.assertAllClose(knp.nanmax(x, axis=(1,)), np.nanmax(x, axis=(1,)))
        self.assertAllClose(
            knp.nanmax(x, axis=1, keepdims=True),
            np.nanmax(x, axis=1, keepdims=True),
        )

        self.assertAllClose(knp.Nanmax()(x), np.nanmax(x))
        self.assertAllClose(knp.Nanmax(axis=1)(x), np.nanmax(x, axis=1))
        self.assertAllClose(
            knp.Nanmax(axis=1, keepdims=True)(x),
            np.nanmax(x, axis=1, keepdims=True),
        )

        x_all_nan = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        self.assertAllClose(knp.nanmax(x_all_nan), np.nanmax(x_all_nan))
        self.assertAllClose(
            knp.nanmax(x_all_nan, axis=1),
            np.nanmax(x_all_nan, axis=1),
        )

        x_3d = np.array(
            [
                [[1.0, np.nan], [2.0, 3.0]],
                [[np.nan, 4.0], [5.0, np.nan]],
            ]
        )
        self.assertAllClose(knp.nanmax(x_3d), np.nanmax(x_3d))
        self.assertAllClose(
            knp.nanmax(x_3d, axis=(1, 2)),
            np.nanmax(x_3d, axis=(1, 2)),
        )

    def test_nanmean(self):
        x = np.array([[1.0, np.nan, 3.0, 4.0], [np.nan, 2.0, np.inf, -np.inf]])

        self.assertAllClose(knp.nanmean(x), np.nanmean(x))
        self.assertAllClose(knp.nanmean(x, axis=()), np.nanmean(x, axis=()))
        self.assertAllClose(knp.nanmean(x, axis=1), np.nanmean(x, axis=1))
        self.assertAllClose(knp.nanmean(x, axis=(1,)), np.nanmean(x, axis=(1,)))
        self.assertAllClose(
            knp.nanmean(x, axis=1, keepdims=True),
            np.nanmean(x, axis=1, keepdims=True),
        )

        self.assertAllClose(knp.Nanmean()(x), np.nanmean(x))
        self.assertAllClose(knp.Nanmean(axis=1)(x), np.nanmean(x, axis=1))
        self.assertAllClose(
            knp.Nanmean(axis=1, keepdims=True)(x),
            np.nanmean(x, axis=1, keepdims=True),
        )

        x_all_nan = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        self.assertAllClose(knp.nanmean(x_all_nan), np.nanmean(x_all_nan))
        self.assertAllClose(
            knp.nanmean(x_all_nan, axis=1),
            np.nanmean(x_all_nan, axis=1),
        )

        x_3d = np.array(
            [
                [[1.0, np.nan], [2.0, 3.0]],
                [[np.nan, 4.0], [5.0, np.nan]],
            ]
        )
        self.assertAllClose(knp.nanmean(x_3d), np.nanmean(x_3d))
        self.assertAllClose(
            knp.nanmean(x_3d, axis=(1, 2)),
            np.nanmean(x_3d, axis=(1, 2)),
        )

    def test_nanmin(self):
        x = np.array([[1.0, np.nan, 3.0], [np.nan, 2.0, np.inf]])

        self.assertAllClose(knp.nanmin(x), np.nanmin(x))
        self.assertAllClose(knp.nanmin(x, axis=()), np.nanmin(x, axis=()))
        self.assertAllClose(knp.nanmin(x, axis=1), np.nanmin(x, axis=1))
        self.assertAllClose(knp.nanmin(x, axis=(1,)), np.nanmin(x, axis=(1,)))
        self.assertAllClose(
            knp.nanmin(x, axis=1, keepdims=True),
            np.nanmin(x, axis=1, keepdims=True),
        )

        self.assertAllClose(knp.Nanmin()(x), np.nanmin(x))
        self.assertAllClose(knp.Nanmin(axis=1)(x), np.nanmin(x, axis=1))
        self.assertAllClose(
            knp.Nanmin(axis=1, keepdims=True)(x),
            np.nanmin(x, axis=1, keepdims=True),
        )

        x_all_nan = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        self.assertAllClose(knp.nanmin(x_all_nan), np.nanmin(x_all_nan))
        self.assertAllClose(
            knp.nanmin(x_all_nan, axis=1),
            np.nanmin(x_all_nan, axis=1),
        )

        x_3d = np.array(
            [
                [[1.0, np.nan], [2.0, 3.0]],
                [[np.nan, 4.0], [5.0, np.nan]],
            ]
        )
        self.assertAllClose(knp.nanmin(x_3d), np.nanmin(x_3d))
        self.assertAllClose(
            knp.nanmin(x_3d, axis=(1, 2)),
            np.nanmin(x_3d, axis=(1, 2)),
        )

    def test_nanprod(self):
        x = np.array([[1.0, np.nan, 3.0], [np.nan, 2.0, 1.0]])

        self.assertAllClose(knp.nanprod(x), np.nanprod(x))
        self.assertAllClose(knp.nanprod(x, axis=()), np.nanprod(x, axis=()))
        self.assertAllClose(knp.nanprod(x, axis=1), np.nanprod(x, axis=1))
        self.assertAllClose(knp.nanprod(x, axis=(1,)), np.nanprod(x, axis=(1,)))
        self.assertAllClose(
            knp.nanprod(x, axis=1, keepdims=True),
            np.nanprod(x, axis=1, keepdims=True),
        )

        self.assertAllClose(knp.Nanprod()(x), np.nanprod(x))
        self.assertAllClose(knp.Nanprod(axis=1)(x), np.nanprod(x, axis=1))
        self.assertAllClose(
            knp.Nanprod(axis=1, keepdims=True)(x),
            np.nanprod(x, axis=1, keepdims=True),
        )

        x_all_nan = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        self.assertAllClose(knp.nanprod(x_all_nan), np.nanprod(x_all_nan))
        self.assertAllClose(
            knp.nanprod(x_all_nan, axis=1),
            np.nanprod(x_all_nan, axis=1),
        )

        x_3d = np.array(
            [
                [[1.0, np.nan], [2.0, 3.0]],
                [[np.nan, 4.0], [5.0, np.nan]],
            ]
        )
        self.assertAllClose(knp.nanprod(x_3d), np.nanprod(x_3d))
        self.assertAllClose(
            knp.nanprod(x_3d, axis=(1, 2)),
            np.nanprod(x_3d, axis=(1, 2)),
        )

    def test_nanstd(self):
        x = np.array([[[1.0, np.nan, 3.0], [np.nan, 2.0, 1.0]]])

        self.assertAllClose(knp.nanstd(x), np.nanstd(x))
        self.assertAllClose(knp.nanstd(x, axis=()), np.nanstd(x, axis=()))
        self.assertAllClose(knp.nanstd(x, axis=0), np.nanstd(x, axis=0))
        self.assertAllClose(knp.nanstd(x, axis=1), np.nanstd(x, axis=1))
        self.assertAllClose(knp.nanstd(x, axis=(1,)), np.nanstd(x, axis=(1,)))
        self.assertAllClose(
            knp.nanstd(x, axis=1, keepdims=True),
            np.nanstd(x, axis=1, keepdims=True),
        )

        self.assertAllClose(knp.Nanstd()(x), np.nanstd(x))
        self.assertAllClose(knp.Nanstd(axis=1)(x), np.nanstd(x, axis=1))
        self.assertAllClose(
            knp.Nanstd(axis=1, keepdims=True)(x),
            np.nanstd(x, axis=1, keepdims=True),
        )

        x_all_nan = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        self.assertAllClose(knp.nanstd(x_all_nan), np.nanstd(x_all_nan))
        self.assertAllClose(
            knp.nanstd(x_all_nan, axis=1),
            np.nanstd(x_all_nan, axis=1),
        )

        x_3d = np.array(
            [
                [[1.0, np.nan], [2.0, 3.0]],
                [[np.nan, 4.0], [5.0, np.nan]],
            ]
        )
        self.assertAllClose(knp.nanstd(x_3d), np.nanstd(x_3d))
        self.assertAllClose(
            knp.nanstd(x_3d, axis=(1, 2)),
            np.nanstd(x_3d, axis=(1, 2)),
        )

    def test_nansum(self):
        x = np.array([[1.0, np.nan, 3.0], [np.nan, 2.0, 1.0]])

        self.assertAllClose(knp.nansum(x), np.nansum(x))
        self.assertAllClose(knp.nansum(x, axis=()), np.nansum(x, axis=()))
        self.assertAllClose(knp.nansum(x, axis=1), np.nansum(x, axis=1))
        self.assertAllClose(knp.nansum(x, axis=(1,)), np.nansum(x, axis=(1,)))
        self.assertAllClose(
            knp.nansum(x, axis=1, keepdims=True),
            np.nansum(x, axis=1, keepdims=True),
        )

        self.assertAllClose(knp.Nansum()(x), np.nansum(x))
        self.assertAllClose(knp.Nansum(axis=1)(x), np.nansum(x, axis=1))
        self.assertAllClose(
            knp.Nansum(axis=1, keepdims=True)(x),
            np.nansum(x, axis=1, keepdims=True),
        )

        x_all_nan = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        self.assertAllClose(knp.nansum(x_all_nan), np.nansum(x_all_nan))
        self.assertAllClose(
            knp.nansum(x_all_nan, axis=1),
            np.nansum(x_all_nan, axis=1),
        )

        x_3d = np.array(
            [
                [[1.0, np.nan], [2.0, 3.0]],
                [[np.nan, 4.0], [5.0, np.nan]],
            ]
        )
        self.assertAllClose(knp.nansum(x_3d), np.nansum(x_3d))
        self.assertAllClose(
            knp.nansum(x_3d, axis=(1, 2)),
            np.nansum(x_3d, axis=(1, 2)),
        )

    def test_nanvar(self):
        x = np.array([[[1.0, np.nan, 3.0], [np.nan, 2.0, 1.0]]])

        self.assertAllClose(knp.nanvar(x), np.nanvar(x))
        self.assertAllClose(knp.nanvar(x, axis=()), np.nanvar(x, axis=()))
        self.assertAllClose(knp.nanvar(x, axis=0), np.nanvar(x, axis=0))
        self.assertAllClose(knp.nanvar(x, axis=1), np.nanvar(x, axis=1))
        self.assertAllClose(knp.nanvar(x, axis=(1,)), np.nanvar(x, axis=(1,)))
        self.assertAllClose(
            knp.nanvar(x, axis=1, keepdims=True),
            np.nanvar(x, axis=1, keepdims=True),
        )

        self.assertAllClose(knp.Nanvar()(x), np.nanvar(x))
        self.assertAllClose(knp.Nanvar(axis=1)(x), np.nanvar(x, axis=1))
        self.assertAllClose(
            knp.Nanvar(axis=1, keepdims=True)(x),
            np.nanvar(x, axis=1, keepdims=True),
        )

        x_all_nan = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        self.assertAllClose(knp.nanvar(x_all_nan), np.nanvar(x_all_nan))
        self.assertAllClose(
            knp.nanvar(x_all_nan, axis=1),
            np.nanvar(x_all_nan, axis=1),
        )

        x_3d = np.array(
            [
                [[1.0, np.nan], [2.0, 3.0]],
                [[np.nan, 4.0], [5.0, np.nan]],
            ]
        )
        self.assertAllClose(knp.nanvar(x_3d), np.nanvar(x_3d))
        self.assertAllClose(
            knp.nanvar(x_3d, axis=(1, 2)),
            np.nanvar(x_3d, axis=(1, 2)),
        )

    def test_nan_to_num(self):
        x = knp.array([1.0, np.nan, np.inf, -np.inf])
        self.assertAllClose(
            knp.nan_to_num(x), [1.0, 0.0, 3.402823e38, -3.402823e38]
        )
        self.assertAllClose(
            knp.NanToNum()(x), [1.0, 0.0, 3.402823e38, -3.402823e38]
        )
        self.assertAllClose(
            knp.nan_to_num(x, nan=2, posinf=3, neginf=4), [1.0, 2.0, 3.0, 4.0]
        )
        self.assertAllClose(
            knp.NanToNum(nan=2, posinf=3, neginf=4)(x), [1.0, 2.0, 3.0, 4.0]
        )

        x = backend.KerasTensor((3, 4))
        self.assertEqual(
            knp.NanToNum(nan=2, posinf=3, neginf=4)(x).shape, (3, 4)
        )

    def test_vectorize(self):
        # Basic functionality
        def myfunc(a, b):
            return a + b

        vfunc = np.vectorize(myfunc)
        y = vfunc([1, 2, 3, 4], 2)
        self.assertAllClose(y, [3, 4, 5, 6])

        # Test signature arg
        vfunc = knp.vectorize(knp.trace, signature="(d,d)->()")
        out = vfunc(np.eye(4))
        self.assertAllClose(
            out, np.vectorize(np.trace, signature="(d,d)->()")(np.eye(4))
        )

        vfunc = knp.vectorize(knp.diag, signature="(d,d)->(d)")
        out = vfunc(np.eye(4))
        self.assertAllClose(
            out, np.vectorize(np.diag, signature="(d,d)->(d)")(np.eye(4))
        )

    def test_argpartition(self):
        x = np.array([3, 4, 2, 1])
        self.assertAllClose(knp.argpartition(x, 2), np.argpartition(x, 2))
        self.assertAllClose(knp.Argpartition(2)(x), np.argpartition(x, 2))

        x = np.array([[3, 4, 2], [1, 3, 4]])
        self.assertAllClose(knp.argpartition(x, 1), np.argpartition(x, 1))
        self.assertAllClose(knp.Argpartition(1)(x), np.argpartition(x, 1))

        x = np.array([[[3, 4], [2, 3]], [[1, 2], [0, 1]]])
        self.assertAllClose(knp.argpartition(x, 1), np.argpartition(x, 1))
        self.assertAllClose(knp.Argpartition(1)(x), np.argpartition(x, 1))

    def test_angle(self):
        x = np.array([[1, 0.5, -0.7], [0.9, 0.2, -1]])
        self.assertAllClose(knp.angle(x), np.angle(x))

        self.assertAllClose(knp.Angle()(x), np.angle(x))


class NumpyArrayCreateOpsCorrectnessTest(testing.TestCase):
    def test_ones(self):
        self.assertAllClose(knp.ones([2, 3]), np.ones([2, 3]))

    def test_zeros(self):
        self.assertAllClose(knp.zeros([2, 3]), np.zeros([2, 3]))

    def test_eye(self):
        self.assertAllClose(knp.eye(3), np.eye(3))
        self.assertAllClose(knp.eye(3, 4), np.eye(3, 4))
        self.assertAllClose(knp.eye(3, 4, 1), np.eye(3, 4, 1))

        # Test k >= N
        self.assertAllClose(knp.eye(3, k=3), np.eye(3, k=3))

        # Test k > 0 and N >= M
        self.assertAllClose(knp.eye(3, k=1), np.eye(3, k=1))

        # Test k > 0 and N < M and N + k > M
        self.assertAllClose(knp.eye(3, 4, k=2), np.eye(3, 4, k=2))

        # Test k < 0 and M >= N
        self.assertAllClose(knp.eye(3, k=-1), np.eye(3, k=-1))

        # Test k < 0 and M < N and M - k > N
        self.assertAllClose(knp.eye(4, 3, k=-2), np.eye(4, 3, k=-2))

    def test_eye_raises_error_with_floats(self):
        with self.assertRaises(TypeError):
            knp.eye(3.0)
        with self.assertRaises(TypeError):
            knp.eye(3.0, 2.0)
        with self.assertRaises(TypeError):
            knp.eye(3, 2.0)
        with self.assertRaises(TypeError):
            v = knp.max(knp.arange(4.0))
            knp.eye(v)
        with self.assertRaises(TypeError):
            knp.eye(knp.array(3, dtype="bfloat16"))

    def test_arange(self):
        self.assertAllClose(knp.arange(3), np.arange(3))
        self.assertAllClose(knp.arange(3, 7), np.arange(3, 7))
        self.assertAllClose(knp.arange(3, 7, 2), np.arange(3, 7, 2))

        self.assertAllClose(knp.Arange()(3), np.arange(3))
        self.assertAllClose(knp.Arange()(3, 7), np.arange(3, 7))
        self.assertAllClose(knp.Arange()(3, 7, 2), np.arange(3, 7, 2))

        self.assertEqual(standardize_dtype(knp.arange(3).dtype), "int32")
        with warnings.catch_warnings(record=True) as record:
            knp.arange(3, dtype="int")
        self.assertEqual(len(record), 0)

    def test_full(self):
        self.assertAllClose(knp.full([2, 3], 0), np.full([2, 3], 0))
        self.assertAllClose(knp.full([2, 3], 0.1), np.full([2, 3], 0.1))
        self.assertAllClose(
            knp.full([2, 3], np.array([1, 4, 5])),
            np.full([2, 3], np.array([1, 4, 5])),
        )

        self.assertAllClose(knp.Full([2, 3])(0), np.full([2, 3], 0))
        self.assertAllClose(knp.Full([2, 3])(0.1), np.full([2, 3], 0.1))
        self.assertAllClose(
            knp.Full([2, 3])(np.array([1, 4, 5])),
            np.full([2, 3], np.array([1, 4, 5])),
        )

    def test_identity(self):
        self.assertAllClose(knp.identity(3), np.identity(3))

    def test_tri(self):
        self.assertAllClose(knp.tri(3), np.tri(3))
        self.assertAllClose(knp.tri(3, 4), np.tri(3, 4))
        self.assertAllClose(knp.tri(3, 4, 1), np.tri(3, 4, 1))

        # Test k < 0
        self.assertAllClose(knp.tri(3, k=-1), np.tri(3, k=-1))

        # Test -k-1 > N
        self.assertAllClose(knp.tri(3, k=-5), np.tri(3, k=-5))

        # Test k > M
        self.assertAllClose(knp.tri(3, k=4), np.tri(3, k=4))


def create_sparse_tensor(x, indices_from=None, start=0, delta=2):
    if indices_from is not None:
        indices = indices_from.indices
    else:
        size = math.prod(x.shape)
        flat_indices = np.arange(start, size, delta)
        indices = np.stack(np.where(np.ones_like(x)), axis=1)[flat_indices]

    if backend.backend() == "tensorflow":
        import tensorflow as tf

        return tf.SparseTensor(indices, tf.gather_nd(x, indices), x.shape)
    elif backend.backend() == "jax":
        import jax
        import jax.experimental.sparse as jax_sparse

        values = x[tuple(jax.numpy.moveaxis(indices, -1, 0))]
        return jax_sparse.BCOO((values, indices), shape=x.shape)


def create_indexed_slices(x, indices_from=None, start=0, delta=2):
    indices = np.arange(start, x.shape[0], delta)

    if backend.backend() == "tensorflow":
        import tensorflow as tf

        if indices_from is not None:
            indices = indices_from.indices
        return tf.IndexedSlices(tf.gather(x, indices), indices, x.shape)
    elif backend.backend() == "jax":
        import jax
        import jax.experimental.sparse as jax_sparse

        if indices_from is not None:
            indices = indices_from.indices
        else:
            indices = jax.numpy.expand_dims(indices, axis=1)
        values = jax.numpy.take(x, jax.numpy.squeeze(indices, axis=1), axis=0)
        return jax_sparse.BCOO((values, indices), shape=x.shape)


def get_sparseness_combinations(dense_to_sparse_fn):
    x = np.array([[1, 2, 3], [3, 2, 1]])
    y = np.array([[4, 5, 6], [3, 2, 1]])
    scalar = backend.convert_to_tensor(2)
    x_sp = dense_to_sparse_fn(x)
    y_sp = dense_to_sparse_fn(y, indices_from=x_sp)
    x_sp_sup = dense_to_sparse_fn(x, start=0, delta=1)
    y_sp_dis = dense_to_sparse_fn(y, start=1)
    y_sp_sup = dense_to_sparse_fn(y, start=0, delta=1)
    x = backend.convert_to_tensor(x)
    y = backend.convert_to_tensor(y)
    return [
        {"testcase_name": "sparse_dense", "x": x_sp, "y": y},
        {"testcase_name": "dense_sparse", "x": x, "y": y_sp},
        {"testcase_name": "sparse_scalar", "x": x_sp, "y": scalar},
        {"testcase_name": "scalar_sparse", "x": scalar, "y": y_sp},
        {"testcase_name": "sparse_sparse_same", "x": x_sp, "y": y_sp},
        {"testcase_name": "sparse_sparse_disjoint", "x": x_sp, "y": y_sp_dis},
        {"testcase_name": "sparse_sparse_superset", "x": x_sp, "y": y_sp_sup},
        {"testcase_name": "sparse_sparse_subset", "x": x_sp_sup, "y": y_sp},
    ]


def sparseness(x):
    if isinstance(x, KerasTensor):
        return "sparse" if x.sparse else "dense"
    elif x.__class__.__name__ == "BCOO":
        if x.n_dense > 0:
            return "slices"
        else:
            return "sparse"
    elif x.__class__.__name__ == "SparseTensor":
        return "sparse"
    elif x.__class__.__name__ == "IndexedSlices":
        return "slices"
    elif not hasattr(x, "shape") or not x.shape:
        return "scalar"
    else:
        return "dense"


def union_sparseness(x1, x2):
    x1_sparseness = sparseness(x1)
    x2_sparseness = sparseness(x2)
    if any(s in ("scalar", "dense") for s in (x1_sparseness, x2_sparseness)):
        return "dense"
    if x1_sparseness != x2_sparseness:
        raise ValueError(f"Illegal combination of operands: {x1} {x2}")
    return x1_sparseness


def intersection_sparseness(x1, x2):
    x1_sparseness = sparseness(x1)
    x2_sparseness = sparseness(x2)
    if x1_sparseness == "scalar":
        return x2_sparseness
    if x2_sparseness in ("scalar", "dense"):
        return x1_sparseness
    if x1_sparseness == "dense":
        return x2_sparseness
    if x1_sparseness != x2_sparseness:
        raise ValueError(f"Illegal combination of operands: {x1} {x2}")
    return x1_sparseness


def division_sparseness(x1, x2):
    x1_sparseness = sparseness(x1)
    x2_sparseness = sparseness(x2)
    if x2_sparseness in ("sparse", "slices"):
        return "dense"
    return "dense" if x1_sparseness == "scalar" else x1_sparseness


def snake_to_pascal_case(name):
    return "".join(w.capitalize() for w in name.split("_"))


@pytest.mark.skipif(
    not backend.SUPPORTS_SPARSE_TENSORS,
    reason="Backend does not support sparse tensors.",
)
class SparseTest(testing.TestCase):
    DTYPES = ["int32", "float32"]
    DENSIFYING_UNARY_OPS = [
        "arccos",
        "arccosh",
        "cos",
        "cosh",
        "exp",
        "isfinite",
        "log",
        "log10",
        "log2",
        "reciprocal",
    ]
    DENSIFYING_UNARY_OPS_TESTS = [
        {
            "testcase_name": op,
            "op_function": getattr(knp, op),
            "op_class": getattr(knp, op.capitalize()),
            "np_op": getattr(np, op),
        }
        for op in DENSIFYING_UNARY_OPS
    ]
    ELEMENTWISE_UNARY_OPS = [
        "abs",
        "absolute",
        "arcsin",
        "arcsinh",
        "arctan",
        "arctanh",
        "ceil",
        "conj",
        "conjugate",
        "copy",
        "expm1",
        "floor",
        "imag",
        "log1p",
        "negative",
        "real",
        "round",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "square",
        "tan",
        "tanh",
    ]
    ELEMENTWISE_UNARY_OPS_TESTS = [
        {
            "testcase_name": op,
            "op_function": getattr(knp, op),
            "op_class": getattr(knp, snake_to_pascal_case(op)),
            "np_op": getattr(np, op),
        }
        for op in ELEMENTWISE_UNARY_OPS
    ]
    OTHER_UNARY_OPS_ARGS = [
        ("digitize", "", {}, {"bins": np.array([0.1, 0.2, 1.0])}, (4, 2, 3)),
        ("mean", "none", {"axis": None}, {}, (4, 2, 3)),
        ("mean", "none_k", {"axis": None, "keepdims": True}, {}, (4, 2, 3)),
        ("mean", "empty", {"axis": ()}, {}, (4, 2, 3)),
        ("mean", "empty_k", {"axis": (), "keepdims": True}, {}, (4, 2, 3)),
        ("mean", "0", {"axis": 0}, {}, (4, 2, 3)),
        ("mean", "0_k", {"axis": 0, "keepdims": True}, {}, (4, 2, 3)),
        ("mean", "1", {"axis": 1}, {}, (4, 2, 3)),
        ("mean", "1_k", {"axis": 1, "keepdims": True}, {}, (4, 2, 3)),
        ("mean", "01", {"axis": (0, 1)}, {}, (4, 2, 3)),
        ("mean", "01_k", {"axis": (0, 1), "keepdims": True}, {}, (4, 2, 3)),
        ("mean", "02", {"axis": (1, 2)}, {}, (4, 2, 3)),
        ("mean", "02_k", {"axis": (1, 2), "keepdims": True}, {}, (4, 2, 3)),
        ("mean", "all", {"axis": (0, 1, 2)}, {}, (4, 2, 3)),
        ("mean", "all_k", {"axis": (0, 1, 2), "keepdims": True}, {}, (4, 2, 3)),
        ("sum", "none", {"axis": None}, {}, (4, 2, 3)),
        ("sum", "none_k", {"axis": None, "keepdims": True}, {}, (4, 2, 3)),
        ("sum", "empty", {"axis": ()}, {}, (4, 2, 3)),
        ("sum", "empty_k", {"axis": (), "keepdims": True}, {}, (4, 2, 3)),
        ("sum", "0", {"axis": 0}, {}, (4, 2, 3)),
        ("sum", "0_k", {"axis": 0, "keepdims": True}, {}, (4, 2, 3)),
        ("sum", "1", {"axis": 1}, {}, (4, 2, 3)),
        ("sum", "1_k", {"axis": 1, "keepdims": True}, {}, (4, 2, 3)),
        ("sum", "01", {"axis": (0, 1)}, {}, (4, 2, 3)),
        ("sum", "01_k", {"axis": (0, 1), "keepdims": True}, {}, (4, 2, 3)),
        ("sum", "02", {"axis": (1, 2)}, {}, (4, 2, 3)),
        ("sum", "02_k", {"axis": (1, 2), "keepdims": True}, {}, (4, 2, 3)),
        ("sum", "all", {"axis": (0, 1, 2)}, {}, (4, 2, 3)),
        ("sum", "all_k", {"axis": (0, 1, 2), "keepdims": True}, {}, (4, 2, 3)),
        ("expand_dims", "zero", {"axis": 0}, {}, (2, 3)),
        ("expand_dims", "one", {"axis": 1}, {}, (2, 3)),
        ("expand_dims", "minus_two", {"axis": -2}, {}, (2, 3)),
        ("reshape", "basic", {"newshape": (4, 3, 2)}, {}, (4, 2, 3)),
        ("reshape", "minus_one", {"newshape": (4, 3, -1)}, {}, (4, 2, 3)),
        ("reshape", "fewer_dims", {"newshape": (4, 6)}, {}, (4, 2, 3)),
        ("squeeze", "no_axis_no_op", {}, {}, (2, 3)),
        ("squeeze", "one", {"axis": 1}, {}, (2, 1, 3)),
        ("squeeze", "minus_two", {"axis": -2}, {}, (2, 1, 3)),
        ("squeeze", "no_axis", {}, {}, (2, 1, 3)),
        ("transpose", "no_axes", {}, {}, (1, 2, 3, 4)),
        ("transpose", "axes", {"axes": (0, 3, 2, 1)}, {}, (1, 2, 3, 4)),
    ]
    OTHER_UNARY_OPS_TESTS = [
        {
            "testcase_name": "_".join([op, testcase_name]),
            "op_function": getattr(knp, op),
            "op_class": getattr(knp, snake_to_pascal_case(op)),
            "np_op": getattr(np, op),
            "init_kwargs": init_kwargs,
            "op_kwargs": op_kwargs,
            "input_shape": input_shape,
        }
        for op, testcase_name, init_kwargs, op_kwargs, input_shape in (
            OTHER_UNARY_OPS_ARGS
        )
    ]

    BINARY_OPS = [
        ("add", union_sparseness),
        ("subtract", union_sparseness),
        ("maximum", union_sparseness),
        ("minimum", union_sparseness),
        ("multiply", intersection_sparseness),
        ("divide", division_sparseness),
        ("true_divide", division_sparseness),
    ]
    BINARY_OPS_TESTS = [
        {
            "testcase_name": op,
            "op_function": getattr(knp, op),
            "op_class": getattr(knp, snake_to_pascal_case(op)),
            "np_op": getattr(np, op),
            "op_sparseness": op_sparseness,
        }
        for op, op_sparseness in BINARY_OPS
    ]

    def assertSameSparseness(self, x, y):
        self.assertEqual(sparseness(x), sparseness(y))

    def assertSparseness(self, x, expected_sparseness):
        self.assertEqual(sparseness(x), expected_sparseness)

    @parameterized.named_parameters(ELEMENTWISE_UNARY_OPS_TESTS)
    def test_elementwise_unary_symbolic_static_shape(
        self, op_function, op_class, np_op
    ):
        x = KerasTensor([2, 3], sparse=True)
        self.assertEqual(op_function(x).shape, (2, 3))
        self.assertTrue(op_function(x).sparse)
        self.assertEqual(op_class()(x).shape, (2, 3))
        self.assertTrue(op_class()(x).sparse)

    @parameterized.named_parameters(ELEMENTWISE_UNARY_OPS_TESTS)
    def test_elementwise_unary_symbolic_dynamic_shape(
        self, op_function, op_class, np_op
    ):
        x = KerasTensor([None, 3], sparse=True)
        self.assertEqual(op_function(x).shape, (None, 3))
        self.assertTrue(op_function(x).sparse)
        self.assertEqual(op_class()(x).shape, (None, 3))
        self.assertTrue(op_class()(x).sparse)

    @parameterized.named_parameters(OTHER_UNARY_OPS_TESTS)
    def test_other_unary_symbolic_static_shape(
        self, op_function, op_class, np_op, init_kwargs, op_kwargs, input_shape
    ):
        expected_shape = op_function(
            KerasTensor(input_shape), **init_kwargs, **op_kwargs
        ).shape
        x = KerasTensor(input_shape, sparse=True)
        self.assertEqual(
            op_function(x, **init_kwargs, **op_kwargs).shape, expected_shape
        )
        self.assertTrue(op_function(x, **init_kwargs, **op_kwargs).sparse)
        self.assertEqual(
            op_class(**init_kwargs)(x, **op_kwargs).shape, expected_shape
        )
        self.assertTrue(op_class(**init_kwargs)(x, **op_kwargs).sparse)

    @parameterized.named_parameters(OTHER_UNARY_OPS_TESTS)
    def test_other_unary_symbolic_dynamic_shape(
        self, op_function, op_class, np_op, init_kwargs, op_kwargs, input_shape
    ):
        input_shape = (None,) + input_shape[1:]
        expected_shape = op_function(
            KerasTensor(input_shape), **init_kwargs, **op_kwargs
        ).shape
        x = KerasTensor(input_shape, sparse=True)
        self.assertEqual(
            op_function(x, **init_kwargs, **op_kwargs).shape, expected_shape
        )
        self.assertTrue(op_function(x, **init_kwargs, **op_kwargs).sparse)
        self.assertEqual(
            op_class(**init_kwargs)(x, **op_kwargs).shape, expected_shape
        )
        self.assertTrue(op_class(**init_kwargs)(x, **op_kwargs).sparse)

    @parameterized.named_parameters(DENSIFYING_UNARY_OPS_TESTS)
    def test_densifying_unary_sparse_correctness(
        self, op_function, op_class, np_op
    ):
        x = np.array([[1, 0.5, -0.7], [0.9, 0.2, -1]])
        x = create_sparse_tensor(x)
        x_np = backend.convert_to_numpy(x)

        self.assertAllClose(op_function(x), np_op(x_np))
        self.assertAllClose(op_class()(x), np_op(x_np))

    @parameterized.named_parameters(DENSIFYING_UNARY_OPS_TESTS)
    def test_densifying_unary_indexed_slices_correctness(
        self, op_function, op_class, np_op
    ):
        x = np.array([[1, 0.5, -0.7], [0.9, 0.2, -1]])
        x = create_indexed_slices(x)
        x_np = backend.convert_to_numpy(x)

        self.assertAllClose(op_function(x), np_op(x_np))
        self.assertAllClose(op_class()(x), np_op(x_np))

    @parameterized.named_parameters(ELEMENTWISE_UNARY_OPS_TESTS)
    def test_elementwise_unary_sparse_correctness(
        self, op_function, op_class, np_op
    ):
        if op_function.__name__ in ("conj", "conjugate", "imag", "real"):
            x = np.array([[1 + 1j, 2 + 2j, 3 + 3j], [3 + 3j, 2 + 2j, 1 + 1j]])
        else:
            x = np.array([[1, 0.5, -0.7], [0.9, 0.2, -1]])
        x = create_sparse_tensor(x)
        x_np = backend.convert_to_numpy(x)

        self.assertAllClose(op_function(x), np_op(x_np))
        self.assertSameSparseness(op_function(x), x)
        self.assertAllClose(op_class()(x), np_op(x_np))
        self.assertSameSparseness(op_class()(x), x)

    @parameterized.named_parameters(ELEMENTWISE_UNARY_OPS_TESTS)
    def test_elementwise_unary_indexed_slices_correctness(
        self, op_function, op_class, np_op
    ):
        if op_function.__name__ in ("conj", "conjugate", "imag", "real"):
            x = np.array([[1 + 1j, 2 + 2j, 3 + 3j], [3 + 3j, 2 + 2j, 1 + 1j]])
        else:
            x = np.array([[1, 0.5, -0.7], [0.9, 0.2, -1]])
        x = create_indexed_slices(x)
        x_np = backend.convert_to_numpy(x)

        self.assertAllClose(op_function(x), np_op(x_np))
        self.assertSameSparseness(op_function(x), x)
        self.assertAllClose(op_class()(x), np_op(x_np))
        self.assertSameSparseness(op_class()(x), x)

    @parameterized.named_parameters(OTHER_UNARY_OPS_TESTS)
    def test_other_unary_sparse_correctness(
        self, op_function, op_class, np_op, init_kwargs, op_kwargs, input_shape
    ):
        x = np.random.random(input_shape)
        if op_function is knp.mean:
            x = create_indexed_slices(x)
        else:
            x = create_sparse_tensor(x)
        x_np = backend.convert_to_numpy(x)

        # `newshape` was renamed `shape` in Numpy.
        np_init_kwargs = init_kwargs.copy()
        if "newshape" in init_kwargs:
            np_init_kwargs["shape"] = np_init_kwargs.pop("newshape")

        self.assertAllClose(
            op_function(x, **init_kwargs, **op_kwargs),
            np_op(x_np, **np_init_kwargs, **op_kwargs),
        )
        self.assertAllClose(
            op_class(**init_kwargs)(x, **op_kwargs),
            np_op(x_np, **np_init_kwargs, **op_kwargs),
        )
        # Reduction operations have complex and backend dependent rules about
        # when the result is sparse and it is dense.
        if op_function is not knp.mean:
            self.assertSameSparseness(
                op_function(x, **init_kwargs, **op_kwargs), x
            )
            self.assertSameSparseness(
                op_class(**init_kwargs)(x, **op_kwargs), x
            )

    @parameterized.named_parameters(
        named_product(
            BINARY_OPS_TESTS, x_sparse=[True, False], y_sparse=[True, False]
        )
    )
    def test_binary_symbolic_static_shape(
        self, x_sparse, y_sparse, op_function, op_class, np_op, op_sparseness
    ):
        x = KerasTensor([2, 3], sparse=x_sparse)
        y = KerasTensor([2, 3], sparse=y_sparse)
        self.assertEqual(op_function(x, y).shape, (2, 3))
        self.assertSparseness(op_function(x, y), op_sparseness(x, y))
        self.assertEqual(op_class()(x, y).shape, (2, 3))
        self.assertSparseness(op_class()(x, y), op_sparseness(x, y))

    @parameterized.named_parameters(
        named_product(
            BINARY_OPS_TESTS, x_sparse=[True, False], y_sparse=[True, False]
        )
    )
    def test_binary_symbolic_dynamic_shape(
        self, x_sparse, y_sparse, op_function, op_class, np_op, op_sparseness
    ):
        x = KerasTensor([None, 3], sparse=x_sparse)
        y = KerasTensor([2, None], sparse=y_sparse)
        self.assertEqual(op_function(x, y).shape, (2, 3))
        self.assertSparseness(op_function(x, y), op_sparseness(x, y))
        self.assertEqual(op_class()(x, y).shape, (2, 3))
        self.assertSparseness(op_class()(x, y), op_sparseness(x, y))

    @parameterized.named_parameters(
        named_product(
            BINARY_OPS_TESTS,
            get_sparseness_combinations(create_sparse_tensor),
            dtype=DTYPES,
        )
    )
    def test_binary_correctness_sparse_tensor(
        self, x, y, op_function, op_class, np_op, op_sparseness, dtype
    ):
        x = backend.cast(x, dtype)
        y = backend.cast(y, dtype)
        expected_result = np_op(
            backend.convert_to_numpy(x), backend.convert_to_numpy(y)
        )

        self.assertAllClose(op_function(x, y), expected_result)
        self.assertSparseness(op_function(x, y), op_sparseness(x, y))
        self.assertAllClose(op_class()(x, y), expected_result)
        self.assertSparseness(op_class()(x, y), op_sparseness(x, y))

    @parameterized.named_parameters(
        named_product(
            BINARY_OPS_TESTS,
            get_sparseness_combinations(create_indexed_slices),
            dtype=DTYPES,
        )
    )
    def test_binary_correctness_indexed_slices(
        self, x, y, op_function, op_class, np_op, op_sparseness, dtype
    ):
        x = backend.cast(x, dtype)
        y = backend.cast(y, dtype)
        expected_result = np_op(
            backend.convert_to_numpy(x), backend.convert_to_numpy(y)
        )

        self.assertAllClose(op_function(x, y), expected_result)
        self.assertSparseness(op_function(x, y), op_sparseness(x, y))
        self.assertAllClose(op_class()(x, y), expected_result)
        self.assertSparseness(op_class()(x, y), op_sparseness(x, y))

    @parameterized.named_parameters(
        named_product(
            sparse_type=["sparse_tensor", "indexed_slices"],
            dtype=["int32", "float32"],
        )
    )
    def test_divide_with_zeros_nans(self, sparse_type, dtype):
        x = backend.convert_to_tensor([[0, 2, 3], [3, 2, 1]], dtype=dtype)
        if sparse_type == "indexed_slices":
            x = create_indexed_slices(x, start=0, delta=2)
        else:
            x = create_sparse_tensor(x, start=0, delta=2)
        if dtype.startswith("int"):
            y = [[0, 0, 3], [0, 0, 1]]
        else:
            y = [[np.nan, np.nan, 3], [0, 0, 1]]
        y = backend.convert_to_tensor(y, dtype=dtype)
        expected_result = np.divide(
            backend.convert_to_numpy(x), backend.convert_to_numpy(y)
        )

        self.assertAllClose(knp.divide(x, y), expected_result)
        self.assertAllClose(knp.Divide()(x, y), expected_result)


class NumpyDtypeTest(testing.TestCase):
    """Test the dtype to verify that the behavior matches JAX."""

    ALL_DTYPES = [
        x
        for x in dtypes.ALLOWED_DTYPES
        if x
        not in (
            "string",
            "complex64",
            "complex128",
            # Remove 64-bit dtypes.
            "float64",
            "uint64",
            "int64",
        )
        + dtypes.FLOAT8_TYPES  # Remove float8 dtypes for the following tests
    ] + [None]
    INT_DTYPES = [x for x in dtypes.INT_TYPES if x not in ("uint64", "int64")]
    FLOAT_DTYPES = [x for x in dtypes.FLOAT_TYPES if x not in ("float64",)]

    if backend.backend() == "torch":
        ALL_DTYPES = [x for x in ALL_DTYPES if x not in ("uint16", "uint32")]
        INT_DTYPES = [x for x in INT_DTYPES if x not in ("uint16", "uint32")]
    elif backend.backend() == "tensorflow":
        # TODO(hongyu): Re-enable uint32 tests once we determine how to handle
        # dtypes.result_type(uint32, int*) -> int64 promotion.
        # Since TF variables require int64 to be placed on the GPU, we
        # exclusively enable the int64 dtype for TF. However, JAX does not
        # natively support int64, which prevents us from comparing the dtypes.
        ALL_DTYPES = [x for x in ALL_DTYPES if x not in ("uint32",)]
        INT_DTYPES = [x for x in INT_DTYPES if x not in ("uint32",)]

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_add(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1,), dtype=dtype1)
        x2 = knp.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.add(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.add(x1, x2).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Add().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_array_split(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1, 2), dtype=dtype)
        x_jax = jnp.ones((1, 2), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.split(x_jax, 2, -1)[0].dtype)

        self.assertEqual(
            standardize_dtype(knp.split(x, 2, -1)[0].dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_add_python_types(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)

        # python int
        expected_dtype = standardize_dtype(jnp.add(x_jax, 1).dtype)

        self.assertDType(knp.add(x, 1), expected_dtype)
        self.assertDType(knp.Add().symbolic_call(x, 1), expected_dtype)

        # python float
        expected_dtype = standardize_dtype(jnp.add(x_jax, 1.0).dtype)

        self.assertDType(knp.add(x, 1.0), expected_dtype)
        self.assertDType(knp.Add().symbolic_call(x, 1.0), expected_dtype)

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_bartlett(self, dtype):
        x = knp.ones((), dtype=dtype)
        expected_dtype = backend.floatx()

        self.assertEqual(
            standardize_dtype(knp.bartlett(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Bartlett().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_blackman(self, dtype):
        x = knp.ones((), dtype=dtype)
        expected_dtype = backend.floatx()

        self.assertEqual(
            standardize_dtype(knp.blackman(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Blackman().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_hamming(self, dtype):
        x = knp.ones((), dtype=dtype)
        expected_dtype = backend.floatx()

        self.assertEqual(
            standardize_dtype(knp.hamming(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Hamming().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_hanning(self, dtype):
        x = knp.ones((), dtype=dtype)
        expected_dtype = backend.floatx()

        self.assertEqual(
            standardize_dtype(knp.hanning(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Hanning().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_kaiser(self, dtype):
        x = knp.ones((), dtype=dtype)
        beta = knp.ones((), dtype=dtype)
        expected_dtype = backend.floatx()

        self.assertEqual(
            standardize_dtype(knp.kaiser(x, beta).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Kaiser(beta).symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=INT_DTYPES))
    def test_bincount(self, dtype):
        import jax.numpy as jnp

        if backend.backend() == "tensorflow":
            import tensorflow as tf

            if tf.test.is_gpu_available():
                self.skipTest("bincount does not work in tensorflow gpu")

        x = np.array([1, 1, 2, 3, 2, 4, 4, 5], dtype=dtype)
        weights = np.array([0, 0, 3, 2, 1, 1, 4, 2], dtype=dtype)
        minlength = 3
        self.assertEqual(
            standardize_dtype(
                knp.bincount(x, weights=weights, minlength=minlength).dtype
            ),
            standardize_dtype(
                jnp.bincount(x, weights=weights, minlength=minlength).dtype
            ),
        )
        self.assertEqual(
            knp.Bincount(weights=weights, minlength=minlength)
            .symbolic_call(x)
            .dtype,
            standardize_dtype(
                jnp.bincount(x, weights=weights, minlength=minlength).dtype
            ),
        )

        # test float32 weights
        weights = np.array([0, 0, 3, 2, 1, 1, 4, 2], dtype="float32")
        self.assertEqual(
            standardize_dtype(knp.bincount(x, weights=weights).dtype),
            standardize_dtype(jnp.bincount(x, weights=weights).dtype),
        )
        self.assertEqual(
            knp.Bincount(weights=weights).symbolic_call(x).dtype,
            standardize_dtype(jnp.bincount(x, weights=weights).dtype),
        )

        # test float16 weights
        weights = np.array([0, 0, 3, 2, 1, 1, 4, 2], dtype="float16")
        self.assertEqual(
            standardize_dtype(knp.bincount(x, weights=weights).dtype),
            standardize_dtype(jnp.bincount(x, weights=weights).dtype),
        )
        self.assertEqual(
            knp.Bincount(weights=weights).symbolic_call(x).dtype,
            standardize_dtype(jnp.bincount(x, weights=weights).dtype),
        )

        # test weights=None
        self.assertEqual(
            standardize_dtype(knp.bincount(x).dtype),
            standardize_dtype(jnp.bincount(x).dtype),
        )
        self.assertEqual(
            knp.Bincount().symbolic_call(x).dtype,
            standardize_dtype(jnp.bincount(x).dtype),
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_subtract(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        if dtype1 == "bool" and dtype2 == "bool":
            self.skipTest("subtract does not support bool")

        x1 = knp.ones((1,), dtype=dtype1)
        x2 = knp.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.subtract(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.subtract(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            knp.Subtract().symbolic_call(x1, x2).dtype, expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_subtract_python_types(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)

        # python int
        expected_dtype = standardize_dtype(jnp.subtract(x_jax, 1).dtype)

        self.assertDType(knp.subtract(x, 1), expected_dtype)
        self.assertDType(knp.Subtract().symbolic_call(x, 1), expected_dtype)

        # python float
        expected_dtype = standardize_dtype(jnp.subtract(x_jax, 1.0).dtype)

        self.assertDType(knp.subtract(x, 1.0), expected_dtype)
        self.assertDType(knp.Subtract().symbolic_call(x, 1.0), expected_dtype)

    @parameterized.named_parameters(
        named_product(
            dtypes=list(itertools.combinations(ALL_DTYPES, 2))
            + [("int8", "int8")]
        )
    )
    def test_matmul(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        # The shape of the matrix needs to meet the requirements of
        # torch._int_mm to test hardware-accelerated matmul
        x1 = knp.ones((17, 16), dtype=dtype1)
        x2 = knp.ones((16, 8), dtype=dtype2)
        x1_jax = jnp.ones((17, 16), dtype=dtype1)
        x2_jax = jnp.ones((16, 8), dtype=dtype2)
        if dtype1 == "int8" and dtype2 == "int8":
            preferred_element_type = "int32"
        else:
            preferred_element_type = None
        expected_dtype = standardize_dtype(
            jnp.matmul(
                x1_jax, x2_jax, preferred_element_type=preferred_element_type
            ).dtype
        )

        self.assertEqual(
            standardize_dtype(knp.matmul(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            knp.Matmul().symbolic_call(x1, x2).dtype, expected_dtype
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_multiply(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1,), dtype=dtype1)
        x2 = knp.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.multiply(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.multiply(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            knp.Multiply().symbolic_call(x1, x2).dtype, expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_multiply_python_types(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)

        # python int
        expected_dtype = standardize_dtype(jnp.multiply(x_jax, 1).dtype)

        self.assertDType(knp.multiply(x, 1), expected_dtype)
        self.assertDType(knp.Multiply().symbolic_call(x, 1), expected_dtype)

        # python float
        expected_dtype = standardize_dtype(jnp.multiply(x_jax, 1.0).dtype)

        self.assertDType(knp.multiply(x, 1.0), expected_dtype)
        self.assertDType(knp.Multiply().symbolic_call(x, 1.0), expected_dtype)

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_mean(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.mean(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = "float32"

        self.assertEqual(standardize_dtype(knp.mean(x).dtype), expected_dtype)
        self.assertEqual(knp.Mean().symbolic_call(x).dtype, expected_dtype)

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_max(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.max(x_jax).dtype)

        self.assertEqual(standardize_dtype(knp.max(x).dtype), expected_dtype)
        self.assertEqual(knp.Max().symbolic_call(x).dtype, expected_dtype)

        # Test with initial
        initial = 1
        expected_dtype = standardize_dtype(
            jnp.max(x_jax, initial=initial).dtype
        )
        self.assertEqual(
            standardize_dtype(knp.max(x, initial=initial).dtype), expected_dtype
        )
        self.assertEqual(
            knp.Max(initial=initial).symbolic_call(x).dtype, expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_ones(self, dtype):
        import jax.numpy as jnp

        expected_dtype = standardize_dtype(jnp.ones([2, 3], dtype=dtype).dtype)

        self.assertEqual(
            standardize_dtype(knp.ones([2, 3], dtype=dtype).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_zeros(self, dtype):
        import jax.numpy as jnp

        expected_dtype = standardize_dtype(jnp.zeros([2, 3], dtype=dtype).dtype)

        self.assertEqual(
            standardize_dtype(knp.zeros([2, 3], dtype=dtype).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_absolute(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.absolute(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.absolute(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Absolute().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_all(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.all(x_jax).dtype)

        self.assertEqual(standardize_dtype(knp.all(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.All().symbolic_call(x).dtype), expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_amax(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.amax(x_jax).dtype)

        self.assertEqual(standardize_dtype(knp.amax(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Amax().symbolic_call(x).dtype), expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_amin(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.amin(x_jax).dtype)

        self.assertEqual(standardize_dtype(knp.amin(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Amin().symbolic_call(x).dtype), expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_any(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.any(x_jax).dtype)

        self.assertEqual(standardize_dtype(knp.any(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Any().symbolic_call(x).dtype), expected_dtype
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_append(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1,), dtype=dtype1)
        x2 = knp.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.append(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.append(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            knp.Append().symbolic_call(x1, x2).dtype, expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_argmax(self, dtype):
        import jax.numpy as jnp

        if dtype == "bool":
            value = [[True, False, True], [False, True, False]]
        else:
            value = [[1, 2, 3], [3, 2, 1]]
        x = knp.array(value, dtype=dtype)
        x_jax = jnp.array(value, dtype=dtype)
        expected_dtype = standardize_dtype(jnp.argmax(x_jax).dtype)

        self.assertEqual(standardize_dtype(knp.argmax(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Argmax().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_argmin(self, dtype):
        import jax.numpy as jnp

        if dtype == "bool":
            value = [[True, False, True], [False, True, False]]
        else:
            value = [[1, 2, 3], [3, 2, 1]]
        x = knp.array(value, dtype=dtype)
        x_jax = jnp.array(value, dtype=dtype)
        expected_dtype = standardize_dtype(jnp.argmin(x_jax).dtype)

        self.assertEqual(standardize_dtype(knp.argmin(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Argmin().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_argpartition(self, dtype):
        import jax.numpy as jnp

        if dtype == "bool":
            self.skipTest("argpartition doesn't support bool dtype")

        x = knp.array([1, 2, 3], dtype=dtype)
        x_jax = jnp.array([1, 2, 3], dtype=dtype)
        expected_dtype = standardize_dtype(jnp.argpartition(x_jax, 1).dtype)

        self.assertEqual(
            standardize_dtype(knp.argpartition(x, 1).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Argpartition(1).symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_argsort(self, dtype):
        import jax.numpy as jnp

        if dtype == "bool":
            value = [[True, False, True], [False, True, False]]
        else:
            value = [[1, 2, 3], [4, 5, 6]]
        x = knp.array(value, dtype=dtype)
        x_jax = jnp.array(value, dtype=dtype)
        expected_dtype = standardize_dtype(jnp.argsort(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.argsort(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Argsort().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.parameters(
        (10, None, None, None),  # stop
        (2, 10, None, None),  # start, stop
        (10, None, 2, None),  # stop, step
        (0, 10, 2, None),  # start, stop, step
        (0, 10, 0.5, None),
        (10.0, None, 1, None),
        (0, 10.0, 1, None),
        (0.0, 10, 1, None),
        (10, None, 1, "float32"),
        (10, None, 1, "int32"),
        (10, None, 1, "int16"),
        (10, None, 1, "float16"),
    )
    def test_arange(self, start, stop, step, dtype):
        import jax.numpy as jnp

        expected_dtype = standardize_dtype(
            jnp.arange(start, stop, step, dtype).dtype
        )

        self.assertEqual(
            standardize_dtype(knp.arange(start, stop, step, dtype).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(
                knp.Arange(dtype).symbolic_call(start, stop, step).dtype
            ),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_arccos(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.arccos(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.arccos(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Arccos().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_arccosh(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.arccosh(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(
            standardize_dtype(knp.arccosh(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Arccosh().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_arcsin(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.arcsin(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.arcsin(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Arcsin().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_arcsinh(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.arcsinh(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(
            standardize_dtype(knp.arcsinh(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Arcsinh().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_arctan(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.arctan(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.arctan(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Arctan().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_arctan2(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1,), dtype=dtype1)
        x2 = knp.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.arctan2(x1_jax, x2_jax).dtype)
        if dtype1 is not None and "float" not in dtype1:
            if dtype2 is not None and "float" not in dtype2:
                if "int64" in (dtype1, dtype2) or "uint32" in (dtype1, dtype2):
                    expected_dtype = backend.floatx()

        self.assertEqual(
            standardize_dtype(knp.arctan2(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            knp.Arctan2().symbolic_call(x1, x2).dtype, expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_arctanh(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.arctanh(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(
            standardize_dtype(knp.arctanh(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Arctanh().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.parameters(
        (bool(0), "bool"),
        (int(0), "int32"),
        (float(0), backend.floatx()),
        ([False, True, False], "bool"),
        ([1, 2, 3], "int32"),
        ([1.0, 2.0, 3.0], backend.floatx()),
        ([1, 2.0, 3], backend.floatx()),
        ([[False], [True], [False]], "bool"),
        ([[1], [2], [3]], "int32"),
        ([[1], [2.0], [3]], backend.floatx()),
        *[
            (np.array(0, dtype=dtype), dtype)
            for dtype in ALL_DTYPES
            if dtype is not None
        ],
    )
    def test_array(self, x, expected_dtype):
        self.assertDType(knp.array(x), expected_dtype)
        # TODO: support the assertion of knp.Array

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_average(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1,), dtype=dtype1)
        x2 = knp.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(
            jnp.average(x1_jax, weights=x2_jax).dtype
        )
        if dtype1 is not None and "float" not in dtype1:
            if dtype2 is not None and "float" not in dtype2:
                if "int64" in (dtype1, dtype2) or "uint32" in (dtype1, dtype2):
                    expected_dtype = backend.floatx()

        self.assertEqual(
            standardize_dtype(knp.average(x1, weights=x2).dtype), expected_dtype
        )
        self.assertEqual(
            knp.Average().symbolic_call(x1, weights=x2).dtype, expected_dtype
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(INT_DTYPES, 2))
    )
    def test_bitwise_and(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1,), dtype=dtype1)
        x2 = knp.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(
            jnp.bitwise_and(x1_jax, x2_jax).dtype
        )

        self.assertDType(knp.bitwise_and(x1, x2), expected_dtype)
        self.assertDType(knp.BitwiseAnd().symbolic_call(x1, x2), expected_dtype)

    @parameterized.named_parameters(named_product(dtype=INT_DTYPES))
    def test_bitwise_invert(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.invert(x_jax).dtype)

        self.assertDType(knp.bitwise_invert(x), expected_dtype)
        self.assertDType(knp.BitwiseInvert().symbolic_call(x), expected_dtype)

    # bitwise_not is same as bitwise_invert

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(INT_DTYPES, 2))
    )
    def test_bitwise_or(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1,), dtype=dtype1)
        x2 = knp.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.bitwise_or(x1_jax, x2_jax).dtype)

        self.assertDType(knp.bitwise_or(x1, x2), expected_dtype)
        self.assertDType(knp.BitwiseOr().symbolic_call(x1, x2), expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(INT_DTYPES, 2))
    )
    def test_bitwise_xor(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1,), dtype=dtype1)
        x2 = knp.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(
            jnp.bitwise_xor(x1_jax, x2_jax).dtype
        )

        self.assertDType(knp.bitwise_xor(x1, x2), expected_dtype)
        self.assertDType(knp.BitwiseXor().symbolic_call(x1, x2), expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.product(INT_DTYPES, INT_DTYPES + [None]))
    )
    def test_bitwise_left_shift(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1,), dtype=dtype1)
        x2 = knp.ones((1,), dtype=dtype2) if dtype2 else 1
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2) if dtype2 else 1
        expected_dtype = standardize_dtype(jnp.left_shift(x1_jax, x2_jax).dtype)

        self.assertDType(knp.bitwise_left_shift(x1, x2), expected_dtype)
        self.assertDType(
            knp.BitwiseLeftShift().symbolic_call(x1, x2), expected_dtype
        )

    # left_shift is same as bitwise_left_shift

    @parameterized.named_parameters(
        named_product(dtypes=itertools.product(INT_DTYPES, INT_DTYPES + [None]))
    )
    def test_bitwise_right_shift(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1,), dtype=dtype1)
        x2 = knp.ones((1,), dtype=dtype2) if dtype2 else 1
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2) if dtype2 else 1
        expected_dtype = standardize_dtype(
            jnp.right_shift(x1_jax, x2_jax).dtype
        )

        self.assertDType(knp.bitwise_right_shift(x1, x2), expected_dtype)
        self.assertDType(
            knp.BitwiseRightShift().symbolic_call(x1, x2), expected_dtype
        )

    # right_shift is same as bitwise_right_shift

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_broadcast_to(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((3,), dtype=dtype)
        x_jax = jnp.ones((3,), dtype=dtype)
        expected_dtype = standardize_dtype(
            jnp.broadcast_to(x_jax, (3, 3)).dtype
        )

        self.assertEqual(
            standardize_dtype(knp.broadcast_to(x, (3, 3)).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.BroadcastTo((3, 3)).symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_cbrt(self, dtype):
        import jax.numpy as jnp

        x1 = knp.ones((1,), dtype=dtype)
        x1_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.cbrt(x1_jax).dtype)

        self.assertEqual(standardize_dtype(knp.cbrt(x1).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Cbrt().symbolic_call(x1).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_ceil(self, dtype):
        import jax.numpy as jnp

        if dtype is None:
            dtype = backend.floatx()
        if dtype == "bool":
            value = [[True, False, True], [True, False, True]]
        elif "int" in dtype:
            value = [[1, 2, 2], [2, 11, 5]]
        else:
            value = [[1.2, 2.1, 2.5], [2.4, 11.9, 5.5]]
        x = knp.array(value, dtype=dtype)
        x_jax = jnp.array(value, dtype=dtype)
        expected_dtype = standardize_dtype(jnp.ceil(x_jax).dtype)
        # Here, we follow Numpy's rule, not JAX's; ints are promoted to floats.
        if dtype == "bool" or is_int_dtype(dtype):
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.ceil(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Ceil().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_clip(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.clip(x_jax, 1, 2).dtype)
        if dtype == "bool":
            expected_dtype = "int32"

        self.assertEqual(
            standardize_dtype(knp.clip(x, 1, 2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Clip(1, 2).symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_concatenate(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1,), dtype=dtype1)
        x2 = knp.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(
            jnp.concatenate([x1_jax, x2_jax]).dtype
        )

        self.assertEqual(
            standardize_dtype(knp.concatenate([x1, x2]).dtype), expected_dtype
        )
        self.assertEqual(
            knp.Concatenate().symbolic_call([x1, x2]).dtype, expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_cos(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.cos(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.cos(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Cos().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_cosh(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.cosh(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.cosh(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Cosh().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_copy(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.copy(x_jax).dtype)

        self.assertEqual(standardize_dtype(knp.copy(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Copy().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_corrcoef(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((2, 4), dtype=dtype)
        x_jax = jnp.ones((2, 4), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.corrcoef(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.corrcoef(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Corrcoef().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_correlate(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((3,), dtype=dtype1)
        x2 = knp.ones((3,), dtype=dtype2)
        x1_jax = jnp.ones((3,), dtype=dtype1)
        x2_jax = jnp.ones((3,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.correlate(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.correlate(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Correlate().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_count_nonzero(self, dtype):
        x = knp.ones((1,), dtype=dtype)
        expected_dtype = "int32"

        self.assertEqual(
            standardize_dtype(knp.count_nonzero(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.CountNonzero().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_cross(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1, 1, 3), dtype=dtype1)
        x2 = knp.ones((1, 1, 3), dtype=dtype2)
        x1_jax = jnp.ones((1, 1, 3), dtype=dtype1)
        x2_jax = jnp.ones((1, 1, 3), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.cross(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.cross(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            knp.Cross().symbolic_call(x1, x2).dtype, expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_cumprod(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.cumprod(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.cumprod(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Cumprod().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_cumsum(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.cumsum(x_jax).dtype)

        self.assertEqual(standardize_dtype(knp.cumsum(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Cumsum().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_deg2rad(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.deg2rad(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.deg2rad(x).dtype), expected_dtype
        )

        self.assertEqual(
            standardize_dtype(knp.Deg2rad().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_diag(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.diag(x_jax).dtype)

        self.assertEqual(standardize_dtype(knp.diag(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Diag().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_diagflat(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.diagflat(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.diagflat(x).dtype), expected_dtype
        )

        self.assertEqual(
            standardize_dtype(knp.Diagflat().symbolic_call(x).dtype),
            expected_dtype,
        )

        x_2d = knp.ones((1, 1), dtype=dtype)
        x_jax_2d = jnp.ones((1, 1), dtype=dtype)
        expected_dtype_2d = standardize_dtype(jnp.diagflat(x_jax_2d).dtype)

        self.assertEqual(
            standardize_dtype(knp.diagflat(x_2d).dtype), expected_dtype_2d
        )
        self.assertEqual(
            standardize_dtype(knp.Diagflat().symbolic_call(x_2d).dtype),
            expected_dtype_2d,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_diagonal(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1, 1, 1), dtype=dtype)
        x_jax = jnp.ones((1, 1, 1), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.diagonal(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.diagonal(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Diagonal().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_diff(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.diff(x_jax).dtype)

        self.assertEqual(standardize_dtype(knp.diff(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Diff().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_digitize(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        bins = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        x_bins = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.digitize(x_jax, x_bins).dtype)

        self.assertEqual(
            standardize_dtype(knp.digitize(x, bins).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Digitize().symbolic_call(x, bins).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_divide(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1,), dtype=dtype1)
        x2 = knp.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.divide(x1_jax, x2_jax).dtype)

        self.assertDType(knp.divide(x1, x2), expected_dtype)
        self.assertDType(knp.Divide().symbolic_call(x1, x2), expected_dtype)

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_divide_python_types(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)

        # python int
        expected_dtype = standardize_dtype(jnp.divide(x_jax, 1).dtype)

        self.assertDType(knp.divide(x, 1), expected_dtype)
        self.assertDType(knp.Divide().symbolic_call(x, 1), expected_dtype)

        # python float
        expected_dtype = standardize_dtype(jnp.divide(x_jax, 1.0).dtype)

        self.assertDType(knp.divide(x, 1.0), expected_dtype)
        self.assertDType(knp.Divide().symbolic_call(x, 1.0), expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_dot(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((2, 3, 4), dtype=dtype1)
        x2 = knp.ones((4, 3), dtype=dtype2)
        x1_jax = jnp.ones((2, 3, 4), dtype=dtype1)
        x2_jax = jnp.ones((4, 3), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.dot(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.dot(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(knp.Dot().symbolic_call(x1, x2).dtype, expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_dstack(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1, 1), dtype=dtype1)
        x2 = knp.ones((1, 1), dtype=dtype2)
        x1_jax = jnp.ones((1, 1), dtype=dtype1)
        x2_jax = jnp.ones((1, 1), dtype=dtype2)

        expected_dtype = standardize_dtype(jnp.dstack([x1_jax, x2_jax]).dtype)

        self.assertEqual(
            standardize_dtype(knp.dstack([x1, x2]).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Dstack().symbolic_call([x1, x2]).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(
            dtypes=list(itertools.combinations(ALL_DTYPES, 2))
            + [("int8", "int8")]
        )
    )
    def test_einsum(self, dtypes):
        import jax.numpy as jnp

        def get_input_shapes(subscripts):
            x1_labels = subscripts.split(",")[0]
            x2_labels = subscripts.split("->")[0][len(x1_labels) + 1 :]
            x1_shape = [1] * len(x1_labels)
            x2_shape = [1] * len(x2_labels)
            return x1_shape, x2_shape

        dtype1, dtype2 = dtypes
        subscripts = "ijk,lkj->il"
        x1_shape, x2_shape = get_input_shapes(subscripts)
        x1 = knp.ones(x1_shape, dtype=dtype1)
        x2 = knp.ones(x2_shape, dtype=dtype2)
        x1_jax = jnp.ones(x1_shape, dtype=dtype1)
        x2_jax = jnp.ones(x2_shape, dtype=dtype2)
        if dtype1 == "int8" and dtype2 == "int8":
            preferred_element_type = "int32"
        else:
            preferred_element_type = None
        expected_dtype = standardize_dtype(
            jnp.einsum(
                subscripts,
                x1_jax,
                x2_jax,
                preferred_element_type=preferred_element_type,
            ).dtype
        )

        self.assertEqual(
            standardize_dtype(knp.einsum(subscripts, x1, x2).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(
                knp.Einsum(subscripts).symbolic_call(x1, x2).dtype
            ),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(
            dtypes=list(itertools.combinations(ALL_DTYPES, 2))
            + [("int8", "int8")]
        )
    )
    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason=f"{backend.backend()} doesn't implement custom ops for einsum.",
    )
    def test_einsum_custom_ops_for_tensorflow(self, dtypes):
        import jax.numpy as jnp

        def get_input_shapes(subscripts):
            x1_labels = subscripts.split(",")[0]
            x2_labels = subscripts.split("->")[0][len(x1_labels) + 1 :]
            x1_shape = [1] * len(x1_labels)
            x2_shape = [1] * len(x2_labels)
            return x1_shape, x2_shape

        dtype1, dtype2 = dtypes
        for subscripts in [
            "a,b->ab",
            "ab,b->a",
            "ab,bc->ac",
            "ab,cb->ac",
            "abc,cd->abd",
            "abc,cde->abde",
            "abc,dc->abd",
            "abc,dce->abde",
            "abc,dec->abde",
            "abcd,abde->abce",
            "abcd,abed->abce",
            "abcd,acbe->adbe",
            "abcd,adbe->acbe",
            "abcd,aecd->acbe",
            "abcd,aecd->aceb",
            "abcd,cde->abe",
            "abcd,ced->abe",
            "abcd,ecd->abe",
            "abcde,aebf->adbcf",
            "abcde,afce->acdbf",
        ]:
            x1_shape, x2_shape = get_input_shapes(subscripts)
            x1 = knp.ones(x1_shape, dtype=dtype1)
            x2 = knp.ones(x2_shape, dtype=dtype2)
            x1_jax = jnp.ones(x1_shape, dtype=dtype1)
            x2_jax = jnp.ones(x2_shape, dtype=dtype2)
            if dtype1 == "int8" and dtype2 == "int8":
                preferred_element_type = "int32"
            else:
                preferred_element_type = None
            expected_dtype = standardize_dtype(
                jnp.einsum(
                    subscripts,
                    x1_jax,
                    x2_jax,
                    preferred_element_type=preferred_element_type,
                ).dtype
            )

            self.assertEqual(
                standardize_dtype(knp.einsum(subscripts, x1, x2).dtype),
                expected_dtype,
            )
            self.assertEqual(
                standardize_dtype(
                    knp.Einsum(subscripts).symbolic_call(x1, x2).dtype
                ),
                expected_dtype,
            )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_empty(self, dtype):
        import jax.numpy as jnp

        expected_dtype = standardize_dtype(jnp.empty([2, 3], dtype=dtype).dtype)

        self.assertEqual(
            standardize_dtype(knp.empty([2, 3], dtype=dtype).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_empty_like(self, dtype):
        import jax.numpy as jnp

        x_jax = jnp.empty([2, 3, 4], dtype=dtype)
        x = knp.ones([2, 3, 4], dtype=dtype)
        expected_dtype = standardize_dtype(
            jnp.empty_like(x_jax, dtype=dtype).dtype
        )

        self.assertEqual(
            standardize_dtype(knp.empty_like(x, dtype=dtype).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.EmptyLike().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_equal(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.equal(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.equal(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Equal().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_exp(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.exp(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.exp(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Exp().symbolic_call(x).dtype), expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_exp2(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.exp2(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.exp2(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Exp2().symbolic_call(x).dtype), expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_expand_dims(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.expand_dims(x_jax, -1).dtype)

        self.assertEqual(
            standardize_dtype(knp.expand_dims(x, -1).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.ExpandDims(-1).symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_expm1(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.expm1(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.expm1(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Expm1().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_eye(self, dtype):
        import jax.numpy as jnp

        expected_dtype = standardize_dtype(jnp.eye(3, dtype=dtype).dtype)
        if dtype is None:
            expected_dtype = backend.floatx()

        self.assertEqual(
            standardize_dtype(knp.eye(3, dtype=dtype).dtype),
            expected_dtype,
        )

        expected_dtype = standardize_dtype(jnp.eye(3, 4, 1, dtype=dtype).dtype)
        if dtype is None:
            expected_dtype = backend.floatx()

        self.assertEqual(
            standardize_dtype(knp.eye(3, 4, k=1, dtype=dtype).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_flip(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.flip(x_jax, -1).dtype)

        self.assertEqual(
            standardize_dtype(knp.flip(x, -1).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Flip(-1).symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_floor(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.floor(x_jax).dtype)
        # Here, we follow Numpy's rule, not JAX's; ints are promoted to floats.
        if dtype == "bool" or is_int_dtype(dtype):
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.floor(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Floor().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_floor_divide(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(
            jnp.floor_divide(x1_jax, x2_jax).dtype
        )

        self.assertEqual(
            standardize_dtype(knp.floor_divide(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.FloorDivide().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_floor_divide_python_types(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)

        # python int
        expected_dtype = standardize_dtype(jnp.floor_divide(x_jax, 1).dtype)

        self.assertDType(knp.floor_divide(x, 1), expected_dtype)
        self.assertDType(knp.FloorDivide().symbolic_call(x, 1), expected_dtype)

        # python float
        expected_dtype = standardize_dtype(jnp.floor_divide(x_jax, 1.0).dtype)

        self.assertDType(knp.floor_divide(x, 1.0), expected_dtype)
        self.assertDType(
            knp.FloorDivide().symbolic_call(x, 1.0), expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_full(self, dtype):
        import jax.numpy as jnp

        expected_dtype = standardize_dtype(jnp.full((), 0, dtype=dtype).dtype)
        if dtype is None:
            expected_dtype = backend.floatx()

        self.assertEqual(
            standardize_dtype(knp.full((), 0, dtype=dtype).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Full((), dtype=dtype).symbolic_call(0).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_full_like(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.full_like(x_jax, 0).dtype)

        self.assertEqual(
            standardize_dtype(knp.full_like(x, 0).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.FullLike().symbolic_call(x, 0).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(INT_DTYPES, 2))
    )
    def test_gcd(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.gcd(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.gcd(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Gcd().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_greater(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.greater(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.greater(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Greater().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_greater_equal(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(
            jnp.greater_equal(x1_jax, x2_jax).dtype
        )

        self.assertEqual(
            standardize_dtype(knp.greater_equal(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.GreaterEqual().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_heaviside(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1, 1), dtype=dtype1)
        x2 = knp.ones((1, 1), dtype=dtype2)
        x1_jax = jnp.ones((1, 1), dtype=dtype1)
        x2_jax = jnp.ones((1, 1), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.heaviside(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.heaviside(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Heaviside().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_hstack(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1, 1), dtype=dtype1)
        x2 = knp.ones((1, 1), dtype=dtype2)
        x1_jax = jnp.ones((1, 1), dtype=dtype1)
        x2_jax = jnp.ones((1, 1), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.hstack([x1_jax, x2_jax]).dtype)

        self.assertEqual(
            standardize_dtype(knp.hstack([x1, x2]).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Hstack().symbolic_call([x1, x2]).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_hypot(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1, 1), dtype=dtype1)
        x2 = knp.ones((1, 1), dtype=dtype2)
        x1_jax = jnp.ones((1, 1), dtype=dtype1)
        x2_jax = jnp.ones((1, 1), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.hypot(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.hypot(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Hypot().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_identity(self, dtype):
        import jax.numpy as jnp

        expected_dtype = standardize_dtype(jnp.identity(3, dtype=dtype).dtype)
        if dtype is None:
            expected_dtype = backend.floatx()

        self.assertEqual(
            standardize_dtype(knp.identity(3, dtype=dtype).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_isclose(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.isclose(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.isclose(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Isclose().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_isfinite(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.isfinite(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.isfinite(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Isfinite().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_isin(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.isin(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.isin(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.IsIn().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_isinf(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.isinf(x_jax).dtype)

        self.assertEqual(standardize_dtype(knp.isinf(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Isinf().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_isnan(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.isnan(x_jax).dtype)

        self.assertEqual(standardize_dtype(knp.isnan(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Isnan().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_isneginf(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.isneginf(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.isneginf(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Isneginf().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_isposinf(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.isposinf(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.isposinf(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Isposinf().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_isreal(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.isreal(x_jax).dtype)

        self.assertEqual(standardize_dtype(knp.isreal(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Isreal().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(INT_DTYPES, 2))
    )
    def test_kron(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.kron(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.kron(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Kron().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(INT_DTYPES, 2))
    )
    def test_lcm(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.lcm(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.lcm(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Lcm().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=list(itertools.product(ALL_DTYPES, INT_DTYPES)))
    )
    def test_ldexp(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.ldexp(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.ldexp(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Ldexp().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_less(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.less(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.less(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Less().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_less_equal(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.less_equal(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.less_equal(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.LessEqual().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(
            start_and_stop=[
                [0, 10],
                [0.5, 10.5],
                [np.array([0, 1], "int32"), np.array([10, 20], "int32")],
                [np.array([0, 1], "float32"), np.array([10, 20], "float32")],
            ],
            num=[0, 1, 5],
            dtype=FLOAT_DTYPES + [None],
        )
    )
    def test_linspace(self, start_and_stop, num, dtype):
        import jax.numpy as jnp

        start, stop = start_and_stop
        expected_dtype = standardize_dtype(
            jnp.linspace(start, stop, num, dtype=dtype).dtype
        )

        self.assertEqual(
            standardize_dtype(
                knp.linspace(start, stop, num, dtype=dtype).dtype
            ),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(
                knp.Linspace(num, dtype=dtype).symbolic_call(start, stop).dtype
            ),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_log(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((3, 3), dtype=dtype)
        x_jax = jnp.ones((3, 3), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.log(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.log(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Log().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_log10(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((3, 3), dtype=dtype)
        x_jax = jnp.ones((3, 3), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.log10(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.log10(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Log10().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_log1p(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((3, 3), dtype=dtype)
        x_jax = jnp.ones((3, 3), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.log1p(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.log1p(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Log1p().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_log2(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((3, 3), dtype=dtype)
        x_jax = jnp.ones((3, 3), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.log2(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.log2(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Log2().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_logaddexp(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((3, 3), dtype=dtype1)
        x2 = knp.ones((3, 3), dtype=dtype2)
        x1_jax = jnp.ones((3, 3), dtype=dtype1)
        x2_jax = jnp.ones((3, 3), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.logaddexp(x1_jax, x2_jax).dtype)
        # jnp.logaddexp will promote "int64" and "uint32" to "float64"
        # force the promotion to `backend.floatx()`
        if dtype1 is not None and "float" not in dtype1:
            if dtype2 is not None and "float" not in dtype2:
                if "int64" in (dtype1, dtype2) or "uint32" in (dtype1, dtype2):
                    expected_dtype = backend.floatx()

        self.assertEqual(
            standardize_dtype(knp.logaddexp(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Logaddexp().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_logaddexp2(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((3, 3), dtype=dtype1)
        x2 = knp.ones((3, 3), dtype=dtype2)
        x1_jax = jnp.ones((3, 3), dtype=dtype1)
        x2_jax = jnp.ones((3, 3), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.logaddexp2(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.logaddexp2(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Logaddexp2().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(
            start_and_stop=[
                [0, 10],
                [0.5, 10.5],
                [np.array([0, 1], "int32"), np.array([10, 20], "int32")],
                [np.array([0, 1], "float32"), np.array([10, 20], "float32")],
            ],
            num=[0, 1, 5],
            dtype=FLOAT_DTYPES + [None],
        )
    )
    def test_logspace(self, start_and_stop, num, dtype):
        import jax.numpy as jnp

        start, stop = start_and_stop
        expected_dtype = standardize_dtype(
            jnp.logspace(start, stop, num, dtype=dtype).dtype
        )

        self.assertEqual(
            standardize_dtype(
                knp.logspace(start, stop, num, dtype=dtype).dtype
            ),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(
                knp.Logspace(num, dtype=dtype).symbolic_call(start, stop).dtype
            ),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(
            start_and_stop=[
                [1, 1000],
                [0.5, 10.5],
                [
                    np.array([1, 2], "float32"),
                    np.array([100, 200], "float32"),
                ],
            ],
            num=[0, 1, 5],
            dtype=FLOAT_DTYPES + [None],
        )
    )
    def test_geomspace(self, start_and_stop, num, dtype):
        import jax.numpy as jnp

        start, stop = start_and_stop
        expected_dtype = standardize_dtype(
            jnp.geomspace(start, stop, num, dtype=dtype).dtype
        )

        self.assertEqual(
            standardize_dtype(
                knp.geomspace(start, stop, num, dtype=dtype).dtype
            ),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(
                knp.Geomspace(num, dtype=dtype).symbolic_call(start, stop).dtype
            ),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_logical_and(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(
            jnp.logical_and(x1_jax, x2_jax).dtype
        )

        self.assertEqual(
            standardize_dtype(knp.logical_and(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.LogicalAnd().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_logical_not(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.logical_not(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.logical_not(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.LogicalNot().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_logical_or(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.logical_or(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.logical_or(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.LogicalOr().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_logical_xor(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(
            jnp.logical_xor(x1_jax, x2_jax).dtype
        )

        self.assertEqual(
            standardize_dtype(knp.logical_xor(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.LogicalXor().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_maximum(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.maximum(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.maximum(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Maximum().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_maximum_python_types(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)

        # python int
        expected_dtype = standardize_dtype(jnp.maximum(x_jax, 1).dtype)

        self.assertDType(knp.maximum(x, 1), expected_dtype)
        self.assertDType(knp.Maximum().symbolic_call(x, 1), expected_dtype)

        # python float
        expected_dtype = standardize_dtype(jnp.maximum(x_jax, 1.0).dtype)

        self.assertDType(knp.maximum(x, 1.0), expected_dtype)
        self.assertDType(knp.Maximum().symbolic_call(x, 1.0), expected_dtype)

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_median(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((3, 3), dtype=dtype)
        x_jax = jnp.ones((3, 3), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.median(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.median(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Median().symbolic_call(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.median(x, axis=1).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Median(axis=1).symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_meshgrid(self, dtype):
        import jax.numpy as jnp

        if dtype == "bool":
            self.skipTest("meshgrid doesn't support bool dtype")
        elif dtype is None:
            dtype = backend.floatx()
        x = knp.array([1, 2, 3], dtype=dtype)
        y = knp.array([4, 5, 6], dtype=dtype)
        x_jax = jnp.array([1, 2, 3], dtype=dtype)
        y_jax = jnp.array([4, 5, 6], dtype=dtype)
        expected_dtype = standardize_dtype(jnp.meshgrid(x_jax, y_jax)[0].dtype)

        self.assertEqual(
            standardize_dtype(knp.meshgrid(x, y)[0].dtype), expected_dtype
        )
        self.assertEqual(
            knp.Meshgrid().symbolic_call(x, y)[0].dtype, expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_min(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.min(x_jax).dtype)

        self.assertEqual(standardize_dtype(knp.min(x).dtype), expected_dtype)
        self.assertEqual(knp.Min().symbolic_call(x).dtype, expected_dtype)

        # Test with initial
        initial = 0
        expected_dtype = standardize_dtype(
            jnp.min(x_jax, initial=initial).dtype
        )
        self.assertEqual(
            standardize_dtype(knp.min(x, initial=initial).dtype), expected_dtype
        )
        self.assertEqual(
            knp.Min(initial=initial).symbolic_call(x).dtype, expected_dtype
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_minimum(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.minimum(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.minimum(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Minimum().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_minimum_python_types(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)

        # python int
        expected_dtype = standardize_dtype(jnp.minimum(x_jax, 1).dtype)

        self.assertDType(knp.minimum(x, 1), expected_dtype)
        self.assertDType(knp.Minimum().symbolic_call(x, 1), expected_dtype)

        # python float
        expected_dtype = standardize_dtype(jnp.minimum(x_jax, 1.0).dtype)

        self.assertDType(knp.minimum(x, 1.0), expected_dtype)
        self.assertDType(knp.Minimum().symbolic_call(x, 1.0), expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_mod(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.mod(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.mod(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Mod().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_moveaxis(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1, 1, 1), dtype=dtype)
        x_jax = jnp.ones((1, 1, 1), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.moveaxis(x_jax, -2, -1).dtype)

        self.assertEqual(
            standardize_dtype(knp.moveaxis(x, -2, -1).dtype), expected_dtype
        )
        self.assertEqual(
            knp.Moveaxis(-2, -1).symbolic_call(x).dtype, expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_nanmax(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.nanmax(x_jax).dtype)

        if backend.backend() == "torch" and expected_dtype == "uint32":
            expected_dtype = "int32"

        self.assertEqual(standardize_dtype(knp.nanmax(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Nanmax().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_nanmean(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.nanmean(x_jax).dtype)

        if backend.backend() == "torch" and expected_dtype == "uint32":
            expected_dtype = "int32"

        self.assertEqual(
            standardize_dtype(knp.nanmean(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Nanmean().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_nanmin(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.nanmin(x_jax).dtype)

        if backend.backend() == "torch" and expected_dtype == "uint32":
            expected_dtype = "int32"

        self.assertEqual(standardize_dtype(knp.nanmin(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Nanmin().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_nanprod(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1, 1, 1), dtype=dtype)
        x_jax = jnp.ones((1, 1, 1), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.nanprod(x_jax).dtype)

        if backend.backend() == "torch" and expected_dtype == "uint32":
            expected_dtype = "int32"

        self.assertEqual(
            standardize_dtype(knp.nanprod(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Nanprod().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_nanstd(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)

        expected_dtype = standardize_dtype(jnp.nanstd(x_jax).dtype)

        if backend.backend() == "torch" and expected_dtype == "uint32":
            expected_dtype = "int32"

        self.assertEqual(
            standardize_dtype(knp.nanstd(x).dtype),
            expected_dtype,
        )

        self.assertEqual(
            standardize_dtype(knp.Nanstd().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_nansum(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.nansum(x_jax).dtype)

        if backend.backend() == "torch" and expected_dtype == "uint32":
            expected_dtype = "int32"

        self.assertEqual(standardize_dtype(knp.nansum(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Nansum().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_nanvar(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)

        expected_dtype = standardize_dtype(jnp.nanvar(x_jax).dtype)

        if backend.backend() == "torch" and expected_dtype == "uint32":
            expected_dtype = "int32"

        self.assertEqual(standardize_dtype(knp.nanvar(x).dtype), expected_dtype)

        self.assertEqual(
            standardize_dtype(knp.Nanvar().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_nan_to_num(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.nan_to_num(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.nan_to_num(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.NanToNum().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=list(itertools.product(ALL_DTYPES, ALL_DTYPES)))
    )
    def test_nextafter(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.nextafter(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.nextafter(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Nextafter().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_nonzero(self, dtype):
        import jax.numpy as jnp

        x = knp.zeros((1,), dtype=dtype)
        x_jax = jnp.zeros((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.nonzero(x_jax)[0].dtype)

        self.assertEqual(
            standardize_dtype(knp.nonzero(x)[0].dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Nonzero().symbolic_call(x)[0].dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_not_equal(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((), dtype=dtype1)
        x2 = knp.ones((), dtype=dtype2)
        x1_jax = jnp.ones((), dtype=dtype1)
        x2_jax = jnp.ones((), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.not_equal(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.not_equal(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.NotEqual().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_ones_like(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.ones_like(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.ones_like(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.OnesLike().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_outer(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1, 2), dtype=dtype1)
        x2 = knp.ones((3, 4), dtype=dtype2)
        x1_jax = jnp.ones((1, 2), dtype=dtype1)
        x2_jax = jnp.ones((3, 4), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.outer(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.outer(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Outer().symbolic_call(x1, x2).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_pad(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((2, 2, 2, 2), dtype=dtype)
        x_jax = jnp.ones((2, 2, 2, 2), dtype=dtype)
        pad_width = ((0, 0), (1, 1), (1, 1), (1, 1))

        for mode in ("constant", "symmetric", "reflect"):
            expected_dtype = standardize_dtype(
                jnp.pad(x_jax, pad_width, mode).dtype
            )

            self.assertEqual(
                standardize_dtype(knp.pad(x, pad_width, mode).dtype),
                expected_dtype,
            )
            self.assertEqual(
                standardize_dtype(
                    knp.Pad(pad_width, mode).symbolic_call(x).dtype
                ),
                expected_dtype,
            )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_power(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x = knp.ones((1,), dtype=dtype1)
        power = knp.ones((1,), dtype2)
        x_jax = jnp.ones((1,), dtype=dtype1)
        power_jax = jnp.ones((1,), dtype2)
        expected_dtype = standardize_dtype(jnp.power(x_jax, power_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.power(x, power).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Power().symbolic_call(x, power).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_power_python_types(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)

        # python int
        expected_dtype = standardize_dtype(jnp.power(x_jax, 1).dtype)

        self.assertDType(knp.power(x, 1), expected_dtype)
        self.assertDType(knp.Power().symbolic_call(x, 1), expected_dtype)

        # python float
        expected_dtype = standardize_dtype(jnp.power(x_jax, 1.0).dtype)

        self.assertDType(knp.power(x, 1.0), expected_dtype)
        self.assertDType(knp.Power().symbolic_call(x, 1.0), expected_dtype)

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_prod(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1, 1, 1), dtype=dtype)
        x_jax = jnp.ones((1, 1, 1), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.prod(x_jax).dtype)
        # TODO: torch doesn't support uint32
        if backend.backend() == "torch" and expected_dtype == "uint32":
            expected_dtype = "int32"

        self.assertEqual(
            standardize_dtype(knp.prod(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Prod().symbolic_call(x).dtype), expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_ptp(self, dtype):
        import jax.numpy as jnp

        if dtype == "bool":
            self.skipTest("ptp doesn't support bool dtype")

        x = knp.ones((1, 1, 1), dtype=dtype)
        x_jax = jnp.ones((1, 1, 1), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.ptp(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.ptp(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Ptp().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_quantile(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((3,), dtype=dtype)
        x_jax = jnp.ones((3,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.quantile(x_jax, 0.5).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(
            standardize_dtype(knp.quantile(x, 0.5).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Quantile().symbolic_call(x, 0.5).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_searchsorted(self, dtype):
        import jax.numpy as jnp

        if dtype == "bool":
            self.skipTest("searchsorted doesn't support bool dtype")

        a = knp.ones((3,), dtype=dtype)
        v = knp.ones((3,), dtype=dtype)

        a_jax = jnp.ones((3,), dtype=dtype)
        v_jax = jnp.ones((3,), dtype=dtype)

        expected_dtype = standardize_dtype(jnp.searchsorted(a_jax, v_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.searchsorted(a, v).dtype), expected_dtype
        )

        self.assertEqual(
            standardize_dtype(knp.SearchSorted().symbolic_call(a, v).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_ravel(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.ravel(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.ravel(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Ravel().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=INT_DTYPES))
    def test_unravel_index(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((3,), dtype=dtype)
        x_jax = jnp.ones((3,), dtype=dtype)

        indices = knp.array([2, 0], dtype=dtype)
        indices_jax = jnp.array([2, 0], dtype=dtype)

        unravel_result_knp = knp.unravel_index(indices, x.shape)
        unravel_result_jax = jnp.unravel_index(indices_jax, x_jax.shape)

        expected_dtype_knp = standardize_dtype(unravel_result_knp[0].dtype)
        expected_dtype_jax = standardize_dtype(unravel_result_jax[0].dtype)

        self.assertEqual(expected_dtype_knp, expected_dtype_jax)

        unravel_result_knp_symbolic = knp.UnravelIndex(x.shape).symbolic_call(
            indices
        )
        expected_dtype_symbolic = standardize_dtype(
            unravel_result_knp_symbolic[0].dtype
        )

        self.assertEqual(expected_dtype_symbolic, expected_dtype_jax)

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_repeat(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1, 1), dtype=dtype)
        x_jax = jnp.ones((1, 1), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.repeat(x_jax, 2).dtype)

        self.assertEqual(
            standardize_dtype(knp.repeat(x, 2).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Repeat(2).symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_reshape(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1, 1), dtype=dtype)
        x_jax = jnp.ones((1, 1), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.reshape(x_jax, [1]).dtype)

        self.assertEqual(
            standardize_dtype(knp.reshape(x, [1]).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Reshape([1]).symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_roll(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((5,), dtype=dtype)
        x_jax = jnp.ones((5,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.roll(x_jax, 2).dtype)

        self.assertEqual(
            standardize_dtype(knp.roll(x, 2).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Roll(2).symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_round(self, dtype):
        import jax.numpy as jnp

        if dtype == "bool":
            self.skipTest("round doesn't support bool dtype")
        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.round(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.round(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Round().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_sign(self, dtype):
        import jax.numpy as jnp

        if dtype == "bool":
            self.skipTest("sign doesn't support bool dtype")
        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.sign(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.sign(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Sign().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_signbit(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.signbit(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.signbit(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Signbit().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_sin(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.sin(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.sin(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Sin().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_sinh(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.sinh(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.sinh(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Sinh().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_sort(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((2,), dtype=dtype)
        x_jax = jnp.ones((2,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.sort(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.sort(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Sort().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_split(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1, 2), dtype=dtype)
        x_jax = jnp.ones((1, 2), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.split(x_jax, 2, -1)[0].dtype)

        self.assertEqual(
            standardize_dtype(knp.split(x, 2, -1)[0].dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Split(2, -1).symbolic_call(x)[0].dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_hsplit(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((2, 1), dtype=dtype)
        x_jax = jnp.ones((2, 1), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.hsplit(x_jax, [1])[0].dtype)

        self.assertEqual(
            standardize_dtype(knp.hsplit(x, [1])[0].dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Hsplit([1]).symbolic_call(x)[0].dtype),
            expected_dtype,
        )

        # test 1d case
        x_1d = knp.ones((4,), dtype=dtype)
        x_1d_jax = jnp.ones((4,), dtype=dtype)
        expected_dtype_1d = standardize_dtype(
            jnp.hsplit(x_1d_jax, [2])[0].dtype
        )

        self.assertEqual(
            standardize_dtype(knp.hsplit(x_1d, [2])[0].dtype),
            expected_dtype_1d,
        )
        self.assertEqual(
            standardize_dtype(knp.Hsplit([2]).symbolic_call(x_1d)[0].dtype),
            expected_dtype_1d,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_vsplit(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1, 2), dtype=dtype)
        x_jax = jnp.ones((1, 2), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.vsplit(x_jax, [1])[0].dtype)

        self.assertEqual(
            standardize_dtype(knp.vsplit(x, [1])[0].dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Vsplit([1]).symbolic_call(x)[0].dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_sqrt(self, dtype):
        import jax.numpy as jnp

        x1 = knp.ones((1,), dtype=dtype)
        x1_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.sqrt(x1_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.sqrt(x1).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Sqrt().symbolic_call(x1).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_square(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.square(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.square(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Square().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_squeeze(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1, 1), dtype=dtype)
        x_jax = jnp.ones((1, 1), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.squeeze(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.squeeze(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Squeeze().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_stack(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1,), dtype=dtype1)
        x2 = knp.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.stack([x1_jax, x2_jax]).dtype)

        self.assertEqual(
            standardize_dtype(knp.stack([x1, x2]).dtype), expected_dtype
        )
        self.assertEqual(
            knp.Stack().symbolic_call([x1, x2]).dtype, expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_std(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.std(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(
            standardize_dtype(knp.std(x).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Std().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_sum(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.sum(x_jax).dtype)

        # TODO: torch doesn't support uint32
        if backend.backend() == "torch" and expected_dtype == "uint32":
            expected_dtype = "int32"

        self.assertEqual(standardize_dtype(knp.sum(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Sum().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_swapaxes(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1, 1), dtype=dtype)
        x_jax = jnp.ones((1, 1), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.swapaxes(x_jax, -1, -2).dtype)

        self.assertEqual(
            standardize_dtype(knp.swapaxes(x, -1, -2).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Swapaxes(-1, -2).symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_take(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.take(x_jax, 0).dtype)

        self.assertEqual(
            standardize_dtype(knp.take(x, 0).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(knp.Take().symbolic_call(x, 0).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtype=ALL_DTYPES, indices_dtype=INT_DTYPES)
    )
    def test_take_along_axis(self, dtype, indices_dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        indices = knp.zeros((1,), dtype=indices_dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        indices_jax = jnp.zeros((1,), dtype=indices_dtype)
        expected_dtype = standardize_dtype(
            jnp.take_along_axis(x_jax, indices_jax, 0).dtype
        )

        self.assertEqual(
            standardize_dtype(knp.take_along_axis(x, indices, 0).dtype),
            expected_dtype,
        )
        self.assertEqual(
            standardize_dtype(
                knp.TakeAlongAxis(0).symbolic_call(x, indices).dtype
            ),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_tan(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.tan(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.tan(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Tan().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_tanh(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.tanh(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.tanh(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Tanh().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_tensordot(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1, 1), dtype=dtype1)
        x2 = knp.ones((1, 1), dtype=dtype2)
        x1_jax = jnp.ones((1, 1), dtype=dtype1)
        x2_jax = jnp.ones((1, 1), dtype=dtype2)
        expected_dtype = standardize_dtype(
            jnp.tensordot(x1_jax, x2_jax, 2).dtype
        )

        self.assertEqual(
            standardize_dtype(knp.tensordot(x1, x2, 2).dtype), expected_dtype
        )
        self.assertEqual(
            knp.Tensordot(2).symbolic_call(x1, x2).dtype, expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_tile(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.tile(x_jax, [1]).dtype)

        self.assertEqual(
            standardize_dtype(knp.tile(x, [1]).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Tile([1]).symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_trace(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1, 1, 1), dtype=dtype)
        x_jax = jnp.ones((1, 1, 1), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.trace(x_jax).dtype)
        # jnp.trace is buggy with bool. We set the expected_dtype to int32
        # for bool inputs
        if dtype == "bool":
            expected_dtype = "int32"
        if dtype == "uint8" and backend.backend() == "torch":
            # Torch backend doesn't support uint32 dtype.
            expected_dtype = "int32"

        self.assertDType(knp.trace(x), expected_dtype)
        self.assertDType(knp.Trace().symbolic_call(x), expected_dtype)

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_transpose(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1, 1), dtype=dtype)
        x_jax = jnp.ones((1, 1), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.transpose(x_jax, [1, 0]).dtype)

        self.assertEqual(
            standardize_dtype(knp.transpose(x, [1, 0]).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Transpose([1, 0]).symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_tri(self, dtype):
        import jax.numpy as jnp

        expected_dtype = standardize_dtype(jnp.tri(3, dtype=dtype).dtype)

        self.assertEqual(
            standardize_dtype(knp.tri(3, dtype=dtype).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_tril(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1, 1), dtype=dtype)
        x_jax = jnp.ones((1, 1), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.tril(x_jax, 0).dtype)

        self.assertEqual(
            standardize_dtype(knp.tril(x, 0).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Tril(0).symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_triu(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1, 1), dtype=dtype)
        x_jax = jnp.ones((1, 1), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.triu(x_jax, 0).dtype)

        self.assertEqual(
            standardize_dtype(knp.triu(x, 0).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Triu(0).symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_true_divide(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1,), dtype=dtype1)
        x2 = knp.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(
            jnp.true_divide(x1_jax, x2_jax).dtype
        )

        self.assertDType(knp.true_divide(x1, x2), expected_dtype)
        self.assertDType(knp.TrueDivide().symbolic_call(x1, x2), expected_dtype)

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_trunc(self, dtype):
        x = knp.ones((1, 1), dtype=dtype)
        # TODO: jax <= 0.30.0 doesn't preserve the original dtype.
        expected_dtype = dtype or backend.floatx()

        self.assertEqual(standardize_dtype(knp.trunc(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Trunc().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_trapezoid(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((2,), dtype=dtype)
        x_jax = jnp.ones((2,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.trapezoid(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.trapezoid(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Trapezoid().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_vander(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((2,), dtype=dtype)
        x_jax = jnp.ones((2,), dtype=dtype)

        if dtype == "bool":
            self.skipTest("vander does not support bool")

        expected_dtype = standardize_dtype(jnp.vander(x_jax).dtype)

        self.assertEqual(standardize_dtype(knp.vander(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Vander().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_var(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((2,), dtype=dtype)
        x_jax = jnp.ones((2,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.var(x_jax).dtype)
        if dtype == "int64":
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.var(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Var().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_vdot(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1,), dtype=dtype1)
        x2 = knp.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.vdot(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.vdot(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(knp.Vdot().symbolic_call(x1, x2).dtype, expected_dtype)

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_inner(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1,), dtype=dtype1)
        x2 = knp.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.inner(x1_jax, x2_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.inner(x1, x2).dtype), expected_dtype
        )
        self.assertEqual(
            knp.Inner().symbolic_call(x1, x2).dtype, expected_dtype
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_vstack(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = knp.ones((1,), dtype=dtype1)
        x2 = knp.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.vstack([x1_jax, x2_jax]).dtype)

        self.assertEqual(
            standardize_dtype(knp.vstack([x1, x2]).dtype), expected_dtype
        )
        self.assertEqual(
            knp.Vstack().symbolic_call([x1, x2]).dtype, expected_dtype
        )

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(ALL_DTYPES, 2))
    )
    def test_where(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        condition = knp.ones((10,), dtype="bool")
        x1 = knp.ones((10,), dtype=dtype1)
        x2 = knp.ones((10,), dtype=dtype2)
        condition_jax = jnp.ones((10,), dtype="bool")
        x1_jax = jnp.ones((10,), dtype=dtype1)
        x2_jax = jnp.ones((10,), dtype=dtype2)
        expected_dtype = standardize_dtype(
            jnp.where(condition_jax, x1_jax, x2_jax).dtype
        )

        self.assertEqual(
            standardize_dtype(knp.where(condition, x1, x2).dtype),
            expected_dtype,
        )
        self.assertEqual(
            knp.Where().symbolic_call(condition, x1, x2).dtype, expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_where_python_types(self, dtype):
        import jax.numpy as jnp

        condition = knp.ones((10,), dtype="bool")
        x = knp.ones((10,), dtype=dtype)
        condition_jax = jnp.ones((10,), dtype="bool")
        x_jax = jnp.ones((10,), dtype=dtype)

        # python int
        expected_dtype = standardize_dtype(
            jnp.where(condition_jax, x_jax, 1).dtype
        )

        self.assertDType(knp.where(condition, x, 1), expected_dtype)
        self.assertDType(
            knp.Where().symbolic_call(condition, x, 1), expected_dtype
        )

        # python float
        expected_dtype = standardize_dtype(
            jnp.where(condition_jax, x_jax, 1.0).dtype
        )

        self.assertDType(knp.where(condition, x, 1.0), expected_dtype)
        self.assertDType(
            knp.Where().symbolic_call(condition, x, 1.0), expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_zeros_like(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((), dtype=dtype)
        x_jax = jnp.ones((), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.ones_like(x_jax).dtype)

        self.assertEqual(
            standardize_dtype(knp.zeros_like(x).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.ZerosLike().symbolic_call(x).dtype),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_angle(self, dtype):
        if dtype == "bfloat16" and testing.torch_uses_gpu():
            self.skipTest("Torch cuda does not support bfloat16")

        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.angle(x_jax).dtype)
        if dtype == "bool" or is_int_dtype(dtype):
            expected_dtype = backend.floatx()

        self.assertEqual(standardize_dtype(knp.angle(x).dtype), expected_dtype)
        self.assertEqual(
            standardize_dtype(knp.Angle().symbolic_call(x).dtype),
            expected_dtype,
        )

    VIEW_DTYPES = [x for x in ALL_DTYPES if x != "bool" and x is not None]

    @parameterized.named_parameters(
        named_product(dtypes=itertools.combinations(VIEW_DTYPES, 2))
    )
    def test_view(self, dtypes):
        import jax.numpy as jnp

        input_dtype, output_dtype = dtypes
        x = knp.ones((2, 8), dtype=input_dtype)
        x_jax = jnp.ones((2, 8), dtype=input_dtype)

        keras_output = knp.view(x, output_dtype)
        symbolic_output = knp.View(output_dtype).symbolic_call(x)
        expected_output = x_jax.view(output_dtype)
        self.assertEqual(
            standardize_dtype(keras_output.dtype),
            standardize_dtype(expected_output.dtype),
        )
        self.assertEqual(
            keras_output.shape,
            expected_output.shape,
        )
        self.assertEqual(
            standardize_dtype(symbolic_output.dtype),
            standardize_dtype(expected_output.dtype),
        )


@pytest.mark.skipif(
    testing.torch_uses_gpu(),
    reason="histogram op not implemented for torch on gpu",
)
class HistogramTest(testing.TestCase):
    def test_histogram_default_args(self):
        hist_op = knp.histogram
        input_tensor = np.random.rand(8)

        # Expected output
        expected_counts, expected_edges = np.histogram(input_tensor)

        counts, edges = hist_op(input_tensor)

        self.assertEqual(counts.shape, expected_counts.shape)
        self.assertAllClose(counts, expected_counts)
        self.assertEqual(edges.shape, expected_edges.shape)
        self.assertAllClose(edges, expected_edges)

    def test_histogram_custom_bins(self):
        hist_op = knp.histogram
        input_tensor = np.random.rand(8)
        bins = 5

        # Expected output
        expected_counts, expected_edges = np.histogram(input_tensor, bins=bins)

        counts, edges = hist_op(input_tensor, bins=bins)

        self.assertEqual(counts.shape, expected_counts.shape)
        self.assertAllClose(counts, expected_counts)
        self.assertEqual(edges.shape, expected_edges.shape)
        self.assertAllClose(edges, expected_edges)

    def test_histogram_custom_range(self):
        hist_op = knp.histogram
        input_tensor = np.random.rand(10)
        range_specified = (2, 8)

        # Expected output
        expected_counts, expected_edges = np.histogram(
            input_tensor, range=range_specified
        )

        counts, edges = hist_op(input_tensor, range=range_specified)

        self.assertEqual(counts.shape, expected_counts.shape)
        self.assertAllClose(counts, expected_counts)
        self.assertEqual(edges.shape, expected_edges.shape)
        self.assertAllClose(edges, expected_edges)

    def test_histogram_symbolic_input(self):
        hist_op = knp.histogram
        input_tensor = KerasTensor(shape=(None,), dtype="float32")

        counts, edges = hist_op(input_tensor)

        self.assertEqual(counts.shape, (10,))
        self.assertEqual(edges.shape, (11,))

    def test_histogram_non_integer_bins_raises_error(self):
        hist_op = knp.histogram
        input_tensor = np.random.rand(8)

        with self.assertRaisesRegex(
            ValueError, "Argument `bins` should be a non-negative integer"
        ):
            hist_op(input_tensor, bins=-5)

    def test_histogram_range_validation(self):
        hist_op = knp.histogram
        input_tensor = np.random.rand(8)

        with self.assertRaisesRegex(
            ValueError, "Argument `range` must be a tuple of two elements"
        ):
            hist_op(input_tensor, range=(1,))

        with self.assertRaisesRegex(
            ValueError,
            "The second element of `range` must be greater than the first",
        ):
            hist_op(input_tensor, range=(5, 1))

    def test_histogram_large_values(self):
        hist_op = knp.histogram
        input_tensor = np.array([1e10, 2e10, 3e10, 4e10, 5e10])

        counts, edges = hist_op(input_tensor, bins=5)

        expected_counts, expected_edges = np.histogram(input_tensor, bins=5)

        self.assertAllClose(counts, expected_counts)
        self.assertAllClose(edges, expected_edges)

    def test_histogram_float_input(self):
        hist_op = knp.histogram
        input_tensor = np.random.rand(8)

        counts, edges = hist_op(input_tensor, bins=5)

        expected_counts, expected_edges = np.histogram(input_tensor, bins=5)

        self.assertAllClose(counts, expected_counts)
        self.assertAllClose(edges, expected_edges)

    def test_histogram_high_dimensional_input(self):
        hist_op = knp.histogram
        input_tensor = np.random.rand(3, 4, 5)

        with self.assertRaisesRegex(
            ValueError, "Input tensor must be 1-dimensional"
        ):
            hist_op(input_tensor)

    def test_histogram_values_on_edges(self):
        hist_op = knp.histogram
        input_tensor = np.array([0.0, 2.0, 4.0, 8.0, 10.0])
        bins = 5

        expected_counts, expected_edges = np.histogram(input_tensor, bins=bins)
        counts, edges = hist_op(input_tensor, bins=bins)

        self.assertAllClose(counts, expected_counts)
        self.assertAllClose(edges, expected_edges)

    # TODO: Fix predict for NumPy.
    @parameterized.named_parameters(
        ("jit_compile_false", False),
        ("jit_compile_true", True),
    )
    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason=(
            "`predict` errors out with 'autodetected range of [nan, nan] is "
            "not finite' on the NumPy backend. To be fixed."
        ),
    )
    def test_histogram_predict(self, jit_compile):
        class HistogramLayer(keras.layers.Layer):
            def call(self, x):
                shape = ops.shape(x)

                # Flatten, because the op does not work with >1-dim inputs.
                x = ops.reshape(x, (shape[0] * shape[1],))
                return knp.histogram(x, bins=5)

        inputs = keras.Input(shape=(8,))
        counts, edges = HistogramLayer()(inputs)
        model = keras.Model(inputs, (counts, edges))
        model.compile(jit_compile=jit_compile)

        model.predict(np.random.randn(1, 8))


class TileTest(testing.TestCase):
    def test_tile_shape_inference_in_layer(self):
        class TileLayer(keras.layers.Layer):
            def call(self, x):
                repeats = [1, 2, 1, 1]
                return knp.tile(x, repeats)

        inputs = keras.Input(shape=(3, 2, 2))
        output = TileLayer()(inputs)

        self.assertEqual(output.shape, (None, 6, 2, 2))
