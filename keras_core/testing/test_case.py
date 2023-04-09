import numpy as np
import unittest


class TestCase(unittest.TestCase):
    def assertAllClose(self, x1, x2, atol=1e-7, rtol=1e-7):
        np.testing.assert_allclose(x1, x2, atol=atol, rtol=rtol)
