import unittest
import numpy as np
from numpy.testing import assert_allclose
from keras import backend as K


class TestConstraints(unittest.TestCase):
    def setUp(self):
        self.some_values = [0.1, 0.5, 3, 8, 1e-7]
        np.random.seed(3537)
        self.example_array = np.random.random((100, 100)) * 100. - 50.
        self.example_array[0, 0] = 0.  # 0 could possibly cause trouble

    def test_maxnorm(self):
        from keras.constraints import maxnorm

        for m in self.some_values:
            norm_instance = maxnorm(m)
            normed = norm_instance(K.variable(self.example_array))
            assert (np.all(K.eval(normed) < m))

        # a more explicit example
        norm_instance = maxnorm(2.0)
        x = np.array([[0, 0, 0], [1.0, 0, 0], [3, 0, 0], [3, 3, 3]]).T
        x_normed_target = np.array([[0, 0, 0], [1.0, 0, 0],
                                    [2.0, 0, 0],
                                    [2. / np.sqrt(3), 2. / np.sqrt(3), 2. / np.sqrt(3)]]).T
        x_normed_actual = K.eval(norm_instance(K.variable(x)))
        assert_allclose(x_normed_actual, x_normed_target, rtol=1e-05)

    def test_nonneg(self):
        from keras.constraints import nonneg

        nonneg_instance = nonneg()

        normed = nonneg_instance(K.variable(self.example_array))
        assert (np.all(np.min(K.eval(normed), axis=1) == 0.))

    def test_identity(self):
        from keras.constraints import identity

        identity_instance = identity()

        normed = identity_instance(self.example_array)
        assert (np.all(normed == self.example_array))

    def test_identity_oddballs(self):
        """
        test the identity constraint on some more exotic input.
        this does not need to pass for the desired real life behaviour,
        but it should in the current implementation.
        """
        from keras.constraints import identity
        identity_instance = identity()

        oddball_examples = ["Hello", [1], -1, None]
        assert(oddball_examples == identity_instance(oddball_examples))

    def test_unitnorm(self):
        from keras.constraints import unitnorm
        unitnorm_instance = unitnorm()

        normalized = unitnorm_instance(K.variable(self.example_array))

        norm_of_normalized = np.sqrt(np.sum(K.eval(normalized)**2, axis=1))
        # in the unit norm constraint, it should be equal to 1.
        difference = norm_of_normalized - 1.
        largest_difference = np.max(np.abs(difference))
        assert np.abs(largest_difference) < 10e-5

if __name__ == '__main__':
    unittest.main()
