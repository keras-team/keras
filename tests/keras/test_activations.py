import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras import activations


def get_standard_values():
    '''
    These are just a set of floats used for testing the activation
    functions, and are useful in multiple tests.
    '''
    return np.array([[0, 0.1, 0.5, 0.9, 1.0]], dtype=K.floatx())


def test_softmax():
    # Test using a reference implementation of softmax
    def softmax(values):
        m = np.max(values)
        e = np.exp(values - m)
        return e / np.sum(e)

    x = K.placeholder(ndim=2)
    f = K.function([x], [activations.softmax(x)])
    test_values = get_standard_values()

    result = f([test_values])[0]
    expected = softmax(test_values)
    assert_allclose(result, expected, rtol=1e-05)


def test_relu():
    '''
    Relu implementation doesn't depend on the value being
    a theano variable. Testing ints, floats and theano tensors.
    '''
    x = K.placeholder(ndim=2)
    f = K.function([x], [activations.relu(x)])

    test_values = get_standard_values()
    result = f([test_values])[0]

    # because no negatives in test values
    assert_allclose(result, test_values, rtol=1e-05)


def test_tanh():
    test_values = get_standard_values()

    x = K.placeholder(ndim=2)
    exp = activations.tanh(x)
    f = K.function([x], [exp])

    result = f([test_values])[0]
    expected = np.tanh(test_values)
    assert_allclose(result, expected, rtol=1e-05)


def test_linear():
    '''
    This function does no input validation, it just returns the thing
    that was passed in.
    '''
    xs = [1, 5, True, None, 'foo']
    for x in xs:
        assert(x == activations.linear(x))


if __name__ == '__main__':
    pytest.main([__file__])
