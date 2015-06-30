import unittest
import numpy as np
from theano import tensor as T


class TestConstraints(unittest.TestCase):
    def setUp(self):
        self.some_values = [0.1, 0.5, 3, 8, 1e-7]
        self.example_array = np.random.random((100, 100)) * 100. - 50.
        self.example_array[0, 0] = 0.  # 0 could possibly cause trouble

    def test_maxnorm(self):
        from keras.constraints import maxnorm

        for m in self.some_values:
            norm_instance = maxnorm(m)
            normed = norm_instance(self.example_array)
            assert (np.all(normed.eval() < m))

    def test_nonneg(self):
        from keras.constraints import nonneg

        normed = nonneg(self.example_array)
        assert (np.all(np.min(normed.eval(), axis=1) == 0.))

    def test_identity(self):
        from keras.constraints import identity

        normed = identity(self.example_array)
        assert (np.all(normed == self.example_array))

    def test_unitnorm(self):
        from keras.constraints import unitnorm

        normed = unitnorm(self.example_array)
        self.assertAlmostEqual(
            np.max(np.abs(np.sqrt(np.sum(normed.eval() ** 2, axis=1)) - 1.))
            , 0.)


if __name__ == '__main__':
    unittest.main()
