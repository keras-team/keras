import unittest

import numpy as np


class TestCase(unittest.TestCase):
    def assertAllClose(self, x1, x2, atol=1e-7, rtol=1e-7):
        np.testing.assert_allclose(x1, x2, atol=atol, rtol=rtol)

    def assertAlmostEqual(self, x1, x2, decimal=3):
        np.testing.assert_almost_equal(x1, x2, decimal=decimal)

    def assertEqual(self, x1, x2):
        np.testing.assert_equal(x1, x2)

    def assertLen(self, iterable, expected_len):
        np.testing.assert_equal(len(iterable), expected_len)

    def assertRaisesRegex(
        self, exception_class, expected_regexp, *args, **kwargs
    ):
        return np.testing.assert_raises_regex(
            exception_class, expected_regexp, *args, **kwargs
        )
