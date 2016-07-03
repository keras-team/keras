import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras import constraints


test_values = [0.1, 0.5, 3, 8, 1e-7]
np.random.seed(3537)
example_array = np.random.random((100, 100)) * 100. - 50.
example_array[0, 0] = 0.  # 0 could possibly cause trouble


def test_maxnorm():
    for m in test_values:
        norm_instance = constraints.maxnorm(m)
        normed = norm_instance(K.variable(example_array))
        assert(np.all(K.eval(normed) < m))

    # a more explicit example
    norm_instance = constraints.maxnorm(2.0)
    x = np.array([[0, 0, 0], [1.0, 0, 0], [3, 0, 0], [3, 3, 3]]).T
    x_normed_target = np.array([[0, 0, 0], [1.0, 0, 0],
                                [2.0, 0, 0],
                                [2. / np.sqrt(3), 2. / np.sqrt(3), 2. / np.sqrt(3)]]).T
    x_normed_actual = K.eval(norm_instance(K.variable(x)))
    assert_allclose(x_normed_actual, x_normed_target, rtol=1e-05)


def test_nonneg():
    nonneg_instance = constraints.nonneg()
    normed = nonneg_instance(K.variable(example_array))
    assert(np.all(np.min(K.eval(normed), axis=1) == 0.))


def test_unitnorm():
    unitnorm_instance = constraints.unitnorm()
    normalized = unitnorm_instance(K.variable(example_array))
    norm_of_normalized = np.sqrt(np.sum(K.eval(normalized)**2, axis=0))
    # in the unit norm constraint, it should be equal to 1.
    difference = norm_of_normalized - 1.
    largest_difference = np.max(np.abs(difference))
    assert(np.abs(largest_difference) < 10e-5)


if __name__ == '__main__':
    pytest.main([__file__])
