import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras import constraints
from keras.utils.test_utils import keras_test


def get_test_values():
    return [0.1, 0.5, 3, 8, 1e-7]


def get_example_array():
    np.random.seed(3537)
    example_array = np.random.random((100, 100)) * 100. - 50.
    example_array[0, 0] = 0.  # 0 could possibly cause trouble
    return example_array


def test_serialization():
    all_activations = ['max_norm', 'non_neg',
                       'unit_norm', 'min_max_norm']
    for name in all_activations:
        fn = constraints.get(name)
        ref_fn = getattr(constraints, name)()
        assert fn.__class__ == ref_fn.__class__
        config = constraints.serialize(fn)
        fn = constraints.deserialize(config)
        assert fn.__class__ == ref_fn.__class__


@keras_test
def test_max_norm():
    array = get_example_array()
    for m in get_test_values():
        norm_instance = constraints.max_norm(m)
        normed = norm_instance(K.variable(array))
        assert(np.all(K.eval(normed) < m))

    # a more explicit example
    norm_instance = constraints.max_norm(2.0)
    x = np.array([[0, 0, 0], [1.0, 0, 0], [3, 0, 0], [3, 3, 3]]).T
    x_normed_target = np.array([[0, 0, 0], [1.0, 0, 0],
                                [2.0, 0, 0],
                                [2. / np.sqrt(3),
                                 2. / np.sqrt(3),
                                 2. / np.sqrt(3)]]).T
    x_normed_actual = K.eval(norm_instance(K.variable(x)))
    assert_allclose(x_normed_actual, x_normed_target, rtol=1e-05)


@keras_test
def test_non_neg():
    non_neg_instance = constraints.non_neg()
    normed = non_neg_instance(K.variable(get_example_array()))
    assert(np.all(np.min(K.eval(normed), axis=1) == 0.))


@keras_test
def test_unit_norm():
    unit_norm_instance = constraints.unit_norm()
    normalized = unit_norm_instance(K.variable(get_example_array()))
    norm_of_normalized = np.sqrt(np.sum(K.eval(normalized) ** 2, axis=0))
    # In the unit norm constraint, it should be equal to 1.
    difference = norm_of_normalized - 1.
    largest_difference = np.max(np.abs(difference))
    assert(np.abs(largest_difference) < 10e-5)


@keras_test
def test_min_max_norm():
    array = get_example_array()
    for m in get_test_values():
        norm_instance = constraints.min_max_norm(min_value=m, max_value=m * 2)
        normed = norm_instance(K.variable(array))
        value = K.eval(normed)
        l2 = np.sqrt(np.sum(np.square(value), axis=0))
        assert not l2[l2 < m]
        assert not l2[l2 > m * 2 + 1e-5]


if __name__ == '__main__':
    pytest.main([__file__])
