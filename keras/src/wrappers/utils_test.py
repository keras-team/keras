"""Tests for keras.src.wrappers.utils."""

import importlib.util

import numpy as np

from keras.src import testing
from keras.src.wrappers.utils import TargetReshaper
from keras.src.wrappers.utils import _check_model
from keras.src.wrappers.utils import assert_sklearn_installed

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


class AssertSklearnInstalledTest(testing.TestCase):
    def test_does_not_raise_when_installed(self):
        if not HAS_SKLEARN:
            self.skipTest("sklearn not installed")
        # Should not raise
        assert_sklearn_installed("TestSymbol")

    def test_error_message_includes_symbol_name(self):
        if HAS_SKLEARN:
            self.skipTest("sklearn is installed, cannot test ImportError")
        with self.assertRaisesRegex(ImportError, "TestSymbol"):
            assert_sklearn_installed("TestSymbol")


class CheckModelTest(testing.TestCase):
    def test_uncompiled_model_raises(self):
        from keras.src import layers
        from keras.src import models

        model = models.Sequential([layers.Dense(1, input_shape=(2,))])
        # Model is not compiled
        with self.assertRaisesRegex(RuntimeError, "compiled"):
            _check_model(model)

    def test_compiled_model_passes(self):
        from keras.src import layers
        from keras.src import models

        model = models.Sequential([layers.Dense(1, input_shape=(2,))])
        model.compile(optimizer="adam", loss="mse")
        # Should not raise
        _check_model(model)


class TargetReshaperTest(testing.TestCase):
    def test_fit_returns_self(self):
        if not HAS_SKLEARN:
            self.skipTest("sklearn not installed")
        reshaper = TargetReshaper()
        y = np.array([1, 2, 3])
        result = reshaper.fit(y)
        self.assertIs(result, reshaper)

    def test_fit_records_ndim(self):
        if not HAS_SKLEARN:
            self.skipTest("sklearn not installed")
        reshaper = TargetReshaper()
        y = np.array([1, 2, 3])
        reshaper.fit(y)
        self.assertEqual(reshaper.ndim_, 1)

        reshaper2 = TargetReshaper()
        y2 = np.array([[1], [2], [3]])
        reshaper2.fit(y2)
        self.assertEqual(reshaper2.ndim_, 2)

    def test_transform_1d_to_2d(self):
        if not HAS_SKLEARN:
            self.skipTest("sklearn not installed")
        reshaper = TargetReshaper()
        y = np.array([1, 2, 3])
        reshaper.fit(y)
        result = reshaper.transform(y)
        self.assertEqual(result.shape, (3, 1))

    def test_transform_2d_unchanged(self):
        if not HAS_SKLEARN:
            self.skipTest("sklearn not installed")
        reshaper = TargetReshaper()
        y = np.array([[1, 2], [3, 4]])
        reshaper.fit(y)
        result = reshaper.transform(y)
        self.assertEqual(result.shape, (2, 2))

    def test_inverse_transform_1d_squeeze(self):
        if not HAS_SKLEARN:
            self.skipTest("sklearn not installed")
        reshaper = TargetReshaper()
        y = np.array([1, 2, 3])
        reshaper.fit(y)
        transformed = reshaper.transform(y)
        # transformed is (3, 1), inverse should squeeze back to (3,)
        result = reshaper.inverse_transform(transformed)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.array_equal(result, y))

    def test_inverse_transform_2d_unchanged(self):
        if not HAS_SKLEARN:
            self.skipTest("sklearn not installed")
        reshaper = TargetReshaper()
        y = np.array([[1, 2], [3, 4]])
        reshaper.fit(y)
        result = reshaper.inverse_transform(y)
        self.assertEqual(result.shape, (2, 2))

    def test_inverse_without_fit_raises(self):
        if not HAS_SKLEARN:
            self.skipTest("sklearn not installed")
        reshaper = TargetReshaper()
        with self.assertRaises(Exception):
            reshaper.inverse_transform(np.array([[1], [2]]))

    def test_roundtrip_1d(self):
        """Transform then inverse should give back original."""
        if not HAS_SKLEARN:
            self.skipTest("sklearn not installed")
        reshaper = TargetReshaper()
        y = np.array([10, 20, 30])
        reshaper.fit(y)
        y_t = reshaper.transform(y)
        y_back = reshaper.inverse_transform(y_t)
        self.assertTrue(np.array_equal(y_back, y))


if __name__ == "__main__":
    testing.run_tests()
