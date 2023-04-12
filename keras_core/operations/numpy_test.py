import numpy as np

from keras_core import backend
from keras_core import testing
from keras_core.backend.keras_tensor import KerasTensor
from keras_core.operations import numpy as knp
from keras_core.operations import operation


class NumpyTwoInputOpsShapeTest(testing.TestCase):
    def test_add(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.add(x, y).shape, (2, 3))

        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.add(x, y).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.add(x, y)

    def test_subtract(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.subtract(x, y).shape, (2, 3))

        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.subtract(x, y).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.subtract(x, y)

    def test_multiply(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.multiply(x, y).shape, (2, 3))

        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.multiply(x, y).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.multiply(x, y)

    def test_matmul(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([3, 2])
        self.assertEqual(knp.matmul(x, y).shape, (2, 2))

        x = KerasTensor([None, 3, 4])
        y = KerasTensor([3, None, 4, 5])
        self.assertEqual(knp.matmul(x, y).shape, (3, None, 3, 5))

        with self.assertRaises(ValueError):
            x = KerasTensor([3, 4])
            y = KerasTensor([2, 3, 4])
            knp.matmul(x, y)

    def test_power(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.power(x, y).shape, (2, 3))

        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.power(x, y).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.power(x, y)

    def test_divide(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.divide(x, y).shape, (2, 3))

        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.divide(x, y).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.divide(x, y)

    def test_true_divide(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.true_divide(x, y).shape, (2, 3))

        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.true_divide(x, y).shape, (2, 3))

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

        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.append(x, y).shape, (None,))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.append(x, y, axis=2)


class NumpyOneInputOpsShapeTest(testing.TestCase):
    def test_mean(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.mean(x).shape, ())

        x = KerasTensor([None, 3])
        self.assertEqual(knp.mean(x).shape, ())

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.mean(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.mean(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_all(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.all(x).shape, ())

        x = KerasTensor([None, 3])
        self.assertEqual(knp.all(x).shape, ())

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.all(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.all(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_var(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.var(x).shape, ())

        x = KerasTensor([None, 3])
        self.assertEqual(knp.var(x).shape, ())

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.var(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.var(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_sum(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.sum(x).shape, ())

        x = KerasTensor([None, 3])
        self.assertEqual(knp.sum(x).shape, ())

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.sum(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.sum(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_amax(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.amax(x).shape, ())

        x = KerasTensor([None, 3])
        self.assertEqual(knp.amax(x).shape, ())

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.amax(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.amax(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_amin(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.amin(x).shape, ())

        x = KerasTensor([None, 3])
        self.assertEqual(knp.amin(x).shape, ())

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.amin(x, axis=1).shape, (None, 3))
        self.assertEqual(knp.amin(x, axis=1, keepdims=True).shape, (None, 1, 3))

    def test_square(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.square(x).shape, (2, 3))

        x = KerasTensor([None, 3])
        self.assertEqual(knp.square(x).shape, (None, 3))

    def test_negative(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.negative(x).shape, (2, 3))

        x = KerasTensor([None, 3])
        self.assertEqual(knp.negative(x).shape, (None, 3))

    def test_abs(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.abs(x).shape, (2, 3))

        x = KerasTensor([None, 3])
        self.assertEqual(knp.abs(x).shape, (None, 3))

    def test_absolute(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.absolute(x).shape, (2, 3))

        x = KerasTensor([None, 3])
        self.assertEqual(knp.absolute(x).shape, (None, 3))

    def test_squeeze(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.squeeze(x).shape, (2, 3))

        x = KerasTensor([None, 1])
        self.assertEqual(knp.squeeze(x).shape, (None,))
        self.assertEqual(knp.squeeze(x, axis=1).shape, (None,))

        with self.assertRaises(ValueError):
            x = KerasTensor([None, 1])
            knp.squeeze(x, axis=0)

    def test_transpose(self):
        x = KerasTensor([2, 3])
        self.assertEqual(knp.transpose(x).shape, (3, 2))

        x = KerasTensor([None, 3])
        self.assertEqual(knp.transpose(x).shape, (3, None))

        x = KerasTensor([None, 3, 3])
        self.assertEqual(knp.transpose(x, (2, 0, 1)).shape, (3, None, 3))


class NumpyTwoInputOpsCorretnessTest(testing.TestCase):
    def test_add(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        z = np.array([[[1, 2, 3], [3, 2, 1]]])
        np.testing.assert_array_equal(np.array(knp.add(x, y)), np.add(x, y))
        np.testing.assert_array_equal(np.array(knp.add(x, z)), np.add(x, z))

        np.testing.assert_array_equal(np.array(knp.Add()(x, y)), np.add(x, y))
        np.testing.assert_array_equal(np.array(knp.Add()(x, z)), np.add(x, z))

    def test_subtract(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        z = np.array([[[1, 2, 3], [3, 2, 1]]])
        np.testing.assert_array_equal(
            np.array(knp.subtract(x, y)), np.subtract(x, y)
        )
        np.testing.assert_array_equal(
            np.array(knp.subtract(x, z)), np.subtract(x, z)
        )

        np.testing.assert_array_equal(
            np.array(knp.Subtract()(x, y)), np.subtract(x, y)
        )
        np.testing.assert_array_equal(
            np.array(knp.Subtract()(x, z)), np.subtract(x, z)
        )

    def test_multiply(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        z = np.array([[[1, 2, 3], [3, 2, 1]]])
        np.testing.assert_array_equal(
            np.array(knp.multiply(x, y)), np.multiply(x, y)
        )
        np.testing.assert_array_equal(
            np.array(knp.multiply(x, z)), np.multiply(x, z)
        )

        np.testing.assert_array_equal(
            np.array(knp.Multiply()(x, y)), np.multiply(x, y)
        )
        np.testing.assert_array_equal(
            np.array(knp.Multiply()(x, z)), np.multiply(x, z)
        )

    def test_matmul(self):
        x = np.ones([2, 3, 4, 5])
        y = np.ones([2, 3, 5, 6])
        z = np.ones([5, 6])
        np.testing.assert_array_equal(
            np.array(knp.matmul(x, y)), np.matmul(x, y)
        )
        np.testing.assert_array_equal(
            np.array(knp.matmul(x, z)), np.matmul(x, z)
        )

        np.testing.assert_array_equal(
            np.array(knp.Matmul()(x, y)), np.matmul(x, y)
        )
        np.testing.assert_array_equal(
            np.array(knp.Matmul()(x, z)), np.matmul(x, z)
        )

    def test_power(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        z = np.array([[[1, 2, 3], [3, 2, 1]]])
        np.testing.assert_array_equal(np.array(knp.power(x, y)), np.power(x, y))
        np.testing.assert_array_equal(np.array(knp.power(x, z)), np.power(x, z))

        np.testing.assert_array_equal(
            np.array(knp.Power()(x, y)), np.power(x, y)
        )
        np.testing.assert_array_equal(
            np.array(knp.Power()(x, z)), np.power(x, z)
        )

    def test_divide(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        z = np.array([[[1, 2, 3], [3, 2, 1]]])
        np.testing.assert_array_equal(
            np.array(knp.divide(x, y)), np.divide(x, y)
        )
        np.testing.assert_array_equal(
            np.array(knp.divide(x, z)), np.divide(x, z)
        )

        np.testing.assert_array_equal(
            np.array(knp.Divide()(x, y)), np.divide(x, y)
        )
        np.testing.assert_array_equal(
            np.array(knp.Divide()(x, z)), np.divide(x, z)
        )

    def test_true_divide(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        z = np.array([[[1, 2, 3], [3, 2, 1]]])
        np.testing.assert_array_equal(
            np.array(knp.true_divide(x, y)), np.true_divide(x, y)
        )
        np.testing.assert_array_equal(
            np.array(knp.true_divide(x, z)), np.true_divide(x, z)
        )

        np.testing.assert_array_equal(
            np.array(knp.TrueDivide()(x, y)), np.true_divide(x, y)
        )
        np.testing.assert_array_equal(
            np.array(knp.TrueDivide()(x, z)), np.true_divide(x, z)
        )

    def test_append(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        y = np.array([[4, 5, 6], [3, 2, 1]])
        z = np.array([[[1, 2, 3], [3, 2, 1]], [[4, 5, 6], [3, 2, 1]]])
        np.testing.assert_array_equal(
            np.array(knp.append(x, y)), np.append(x, y)
        )
        np.testing.assert_array_equal(
            np.array(knp.append(x, y, axis=1)), np.append(x, y, axis=1)
        )
        np.testing.assert_array_equal(
            np.array(knp.append(x, z)), np.append(x, z)
        )

        np.testing.assert_array_equal(
            np.array(knp.Append()(x, y)), np.append(x, y)
        )
        np.testing.assert_array_equal(
            np.array(knp.Append(axis=1)(x, y)), np.append(x, y, axis=1)
        )
        np.testing.assert_array_equal(
            np.array(knp.Append()(x, z)), np.append(x, z)
        )


class NumpyOneInputOpsCorrectnessTest(testing.TestCase):
    def test_mean(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        np.testing.assert_array_equal(np.array(knp.mean(x)), np.mean(x))
        np.testing.assert_array_equal(
            np.array(knp.mean(x, axis=1)), np.mean(x, axis=1)
        )
        np.testing.assert_array_equal(
            np.array(knp.mean(x, axis=1, keepdims=True)),
            np.mean(x, axis=1, keepdims=True),
        )

        np.testing.assert_array_equal(np.array(knp.Mean()(x)), np.mean(x))
        np.testing.assert_array_equal(
            np.array(knp.Mean(axis=1)(x)), np.mean(x, axis=1)
        )
        np.testing.assert_array_equal(
            np.array(knp.Mean(axis=1, keepdims=True)(x)),
            np.mean(x, axis=1, keepdims=True),
        )

    def test_all(self):
        x = np.array([[True, False, True], [True, True, True]])
        np.testing.assert_array_equal(np.array(knp.all(x)), np.all(x))
        np.testing.assert_array_equal(
            np.array(knp.all(x, axis=1)), np.all(x, axis=1)
        )
        np.testing.assert_array_equal(
            np.array(knp.all(x, axis=1, keepdims=True)),
            np.all(x, axis=1, keepdims=True),
        )

        np.testing.assert_array_equal(np.array(knp.All()(x)), np.all(x))
        np.testing.assert_array_equal(
            np.array(knp.All(axis=1)(x)), np.all(x, axis=1)
        )
        np.testing.assert_array_equal(
            np.array(knp.All(axis=1, keepdims=True)(x)),
            np.all(x, axis=1, keepdims=True),
        )

    def test_var(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        np.testing.assert_array_equal(np.array(knp.var(x)), np.var(x))
        np.testing.assert_array_equal(
            np.array(knp.var(x, axis=1)), np.var(x, axis=1)
        )
        np.testing.assert_array_equal(
            np.array(knp.var(x, axis=1, keepdims=True)),
            np.var(x, axis=1, keepdims=True),
        )

        np.testing.assert_array_equal(np.array(knp.Var()(x)), np.var(x))
        np.testing.assert_array_equal(
            np.array(knp.Var(axis=1)(x)), np.var(x, axis=1)
        )
        np.testing.assert_array_equal(
            np.array(knp.Var(axis=1, keepdims=True)(x)),
            np.var(x, axis=1, keepdims=True),
        )

    def test_sum(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        np.testing.assert_array_equal(np.array(knp.sum(x)), np.sum(x))
        np.testing.assert_array_equal(
            np.array(knp.sum(x, axis=1)), np.sum(x, axis=1)
        )
        np.testing.assert_array_equal(
            np.array(knp.sum(x, axis=1, keepdims=True)),
            np.sum(x, axis=1, keepdims=True),
        )

        np.testing.assert_array_equal(np.array(knp.Sum()(x)), np.sum(x))
        np.testing.assert_array_equal(
            np.array(knp.Sum(axis=1)(x)), np.sum(x, axis=1)
        )
        np.testing.assert_array_equal(
            np.array(knp.Sum(axis=1, keepdims=True)(x)),
            np.sum(x, axis=1, keepdims=True),
        )

    def test_amax(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        np.testing.assert_array_equal(np.array(knp.amax(x)), np.amax(x))
        np.testing.assert_array_equal(
            np.array(knp.amax(x, axis=1)), np.amax(x, axis=1)
        )
        np.testing.assert_array_equal(
            np.array(knp.amax(x, axis=1, keepdims=True)),
            np.amax(x, axis=1, keepdims=True),
        )

        np.testing.assert_array_equal(np.array(knp.Amax()(x)), np.amax(x))
        np.testing.assert_array_equal(
            np.array(knp.Amax(axis=1)(x)), np.amax(x, axis=1)
        )
        np.testing.assert_array_equal(
            np.array(knp.Amax(axis=1, keepdims=True)(x)),
            np.amax(x, axis=1, keepdims=True),
        )

    def test_amin(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        np.testing.assert_array_equal(np.array(knp.amin(x)), np.amin(x))
        np.testing.assert_array_equal(
            np.array(knp.amin(x, axis=1)), np.amin(x, axis=1)
        )
        np.testing.assert_array_equal(
            np.array(knp.amin(x, axis=1, keepdims=True)),
            np.amin(x, axis=1, keepdims=True),
        )

        np.testing.assert_array_equal(np.array(knp.Amin()(x)), np.amin(x))
        np.testing.assert_array_equal(
            np.array(knp.Amin(axis=1)(x)), np.amin(x, axis=1)
        )
        np.testing.assert_array_equal(
            np.array(knp.Amin(axis=1, keepdims=True)(x)),
            np.amin(x, axis=1, keepdims=True),
        )

    def test_square(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        np.testing.assert_array_equal(np.array(knp.square(x)), np.square(x))

        np.testing.assert_array_equal(np.array(knp.Square()(x)), np.square(x))

    def test_negative(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        np.testing.assert_array_equal(np.array(knp.negative(x)), np.negative(x))

        np.testing.assert_array_equal(
            np.array(knp.Negative()(x)), np.negative(x)
        )

    def test_abs(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        np.testing.assert_array_equal(np.array(knp.abs(x)), np.abs(x))

        np.testing.assert_array_equal(np.array(knp.Abs()(x)), np.abs(x))

    def test_absolute(self):
        x = np.array([[1, 2, 3], [3, 2, 1]])
        np.testing.assert_array_equal(np.array(knp.absolute(x)), np.absolute(x))

        np.testing.assert_array_equal(
            np.array(knp.Absolute()(x)), np.absolute(x)
        )

    def test_squeeze(self):
        x = np.ones([1, 2, 3, 4, 5])
        np.testing.assert_array_equal(np.array(knp.squeeze(x)), np.squeeze(x))
        np.testing.assert_array_equal(
            np.array(knp.squeeze(x, axis=0)), np.squeeze(x, axis=0)
        )

        np.testing.assert_array_equal(np.array(knp.Squeeze()(x)), np.squeeze(x))
        np.testing.assert_array_equal(
            np.array(knp.Squeeze(axis=0)(x)), np.squeeze(x, axis=0)
        )

    def test_transpose(self):
        x = np.ones([1, 2, 3, 4, 5])
        np.testing.assert_array_equal(
            np.array(knp.transpose(x)), np.transpose(x)
        )
        np.testing.assert_array_equal(
            np.array(knp.transpose(x, axes=(1, 0, 3, 2, 4))),
            np.transpose(x, axes=(1, 0, 3, 2, 4)),
        )

        np.testing.assert_array_equal(
            np.array(knp.Transpose()(x)), np.transpose(x)
        )
        np.testing.assert_array_equal(
            np.array(knp.Transpose(axes=(1, 0, 3, 2, 4))(x)),
            np.transpose(x, axes=(1, 0, 3, 2, 4)),
        )


class NumpyArrayCreateOpsCorrectnessTest(testing.TestCase):
    def test_ones(self):
        np.testing.assert_array_equal(
            np.array(knp.ones([2, 3])), np.ones([2, 3])
        )
        np.testing.assert_array_equal(
            np.array(knp.Ones()([2, 3])), np.ones([2, 3])
        )

    def test_zeros(self):
        np.testing.assert_array_equal(
            np.array(knp.zeros([2, 3])), np.zeros([2, 3])
        )
        np.testing.assert_array_equal(
            np.array(knp.Zeros()([2, 3])), np.zeros([2, 3])
        )

    def test_eye(self):
        np.testing.assert_array_equal(np.array(knp.eye(3)), np.eye(3))
        np.testing.assert_array_equal(np.array(knp.eye(3, 4)), np.eye(3, 4))
        np.testing.assert_array_equal(
            np.array(knp.eye(3, 4, 1)), np.eye(3, 4, 1)
        )

        np.testing.assert_array_equal(np.array(knp.Eye()(3)), np.eye(3))
        np.testing.assert_array_equal(np.array(knp.Eye()(3, 4)), np.eye(3, 4))
        np.testing.assert_array_equal(
            np.array(knp.Eye()(3, 4, 1)), np.eye(3, 4, 1)
        )
