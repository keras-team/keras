import math

import numpy as np
import pytest
import scipy.signal
from absl.testing import parameterized

from keras import backend
from keras import testing
from keras.backend.common import standardize_dtype
from keras.backend.common.keras_tensor import KerasTensor
from keras.backend.common.variables import ALLOWED_DTYPES
from keras.ops import linalg
from keras.testing.test_utils import named_product


class LinalgOpsDynamicShapeTest(testing.TestCase):
    def test_cholesky(self):
        x = KerasTensor([None, 20, 20])
        out = linalg.cholesky(x)
        self.assertEqual(out.shape, (None, 20, 20))

        x = KerasTensor([None, None, 20])
        with self.assertRaises(ValueError):
            linalg.cholesky(x)

        x = KerasTensor([None, 20, 15])
        with self.assertRaises(ValueError):
            linalg.cholesky(x)

    def test_det(self):
        x = KerasTensor([None, 20, 20])
        out = linalg.det(x)
        self.assertEqual(out.shape, (None,))

        x = KerasTensor([None, None, 20])
        with self.assertRaises(ValueError):
            linalg.det(x)

        x = KerasTensor([None, 20, 15])
        with self.assertRaises(ValueError):
            linalg.det(x)

    def test_inv(self):
        x = KerasTensor([None, 20, 20])
        out = linalg.inv(x)
        self.assertEqual(out.shape, (None, 20, 20))

        x = KerasTensor([None, None, 20])
        with self.assertRaises(ValueError):
            linalg.inv(x)

        x = KerasTensor([None, 20, 15])
        with self.assertRaises(ValueError):
            linalg.inv(x)

    def test_solve(self):
        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, 20, 5])
        out = linalg.solve(a, b)
        self.assertEqual(out.shape, (None, 20, 5))

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, 20])
        out = linalg.solve(a, b)
        self.assertEqual(out.shape, (None, 20))

        a = KerasTensor([None, None, 20])
        b = KerasTensor([None, 20, 5])
        with self.assertRaises(ValueError):
            linalg.solve(a, b)

        a = KerasTensor([None, 20, 15])
        b = KerasTensor([None, 20, 5])
        with self.assertRaises(ValueError):
            linalg.solve(a, b)

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, None, 5])
        with self.assertRaises(ValueError):
            linalg.solve(a, b)


class LinalgOpsStaticShapeTest(testing.TestCase):
    def test_cholesky(self):
        x = KerasTensor([10, 20, 20])
        out = linalg.cholesky(x)
        self.assertEqual(out.shape, (10, 20, 20))

        x = KerasTensor([10, 20, 15])
        with self.assertRaises(ValueError):
            linalg.cholesky(x)

    def test_det(self):
        x = KerasTensor([10, 20, 20])
        out = linalg.det(x)
        self.assertEqual(out.shape, (10,))

        x = KerasTensor([10, 20, 15])
        with self.assertRaises(ValueError):
            linalg.det(x)

    def test_inv(self):
        x = KerasTensor([10, 20, 20])
        out = linalg.inv(x)
        self.assertEqual(out.shape, (10, 20, 20))

        x = KerasTensor([10, 20, 15])
        with self.assertRaises(ValueError):
            linalg.inv(x)

    def test_solve(self):
        a = KerasTensor([10, 20, 20])
        b = KerasTensor([10, 20, 5])
        out = linalg.solve(a, b)
        self.assertEqual(out.shape, (10, 20, 5))

        a = KerasTensor([10, 20, 20])
        b = KerasTensor([10, 20])
        out = linalg.solve(a, b)
        self.assertEqual(out.shape, (10, 20))

        a = KerasTensor([10, 20, 15])
        b = KerasTensor([10, 20, 5])
        with self.assertRaises(ValueError):
            linalg.solve(a, b)

        a = KerasTensor([20, 20])
        b = KerasTensor([])
        with self.assertRaises(ValueError):
            linalg.solve(a, b)


class LinalgOpsCorrectnessTest(testing.TestCase):

    def test_cholesky(self):
        x = np.random.rand(10, 20, 20)
        out = linalg.cholesky(x)
        np.testing.assert_allclose(out, np.linalg.cholesky(x))

    def test_det(self):
        x = np.random.rand(10, 20, 20)
        out = linalg.det(x)
        np.testing.assert_allclose(out, np.linalg.det(x))

    def test_inv(self):
        x = np.random.rand(10, 20, 20)
        out = linalg.inv(x)
        np.testing.assert_allclose(out, np.linalg.inv(x))

    def test_solve(self):
        a = np.random.rand(10, 20, 20)
        b = np.random.rand(10, 20, 5)
        out = linalg.solve(a, b)
        np.testing.assert_allclose(out, np.linalg.solve(a, b))

        a = np.random.rand(10, 20, 20)
        b = np.random.rand(10, 20)
        out = linalg.solve(a, b)
        np.testing.assert_allclose(out, np.linalg.solve(a, b))
