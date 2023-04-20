import unittest

import numpy as np


class TestCase(unittest.TestCase):
    def assertAllClose(self, x1, x2, atol=1e-7, rtol=1e-7):
        np.testing.assert_allclose(x1, x2, atol=atol, rtol=rtol)

    def assertAlmostEqual(self, x1, x2, decimal=3):
        np.testing.assert_almost_equal(x1, x2, decimal=decimal)
