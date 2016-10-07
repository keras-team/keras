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
    '''
    Test using a reference implementation of softmax
    '''
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


def test_time_distributed_softmax():
    x = K.placeholder(shape=(1, 1, 5))
    f = K.function([x], [activations.softmax(x)])
    test_values = get_standard_values()
    test_values = np.reshape(test_values, (1, 1, np.size(test_values)))
    f([test_values])[0]


def test_softplus():
    '''
    Test using a reference softplus implementation
    '''
    def softplus(x):
        return np.log(np.ones_like(x) + np.exp(x))

    x = K.placeholder(ndim=2)
    f = K.function([x], [activations.softplus(x)])
    test_values = get_standard_values()

    result = f([test_values])[0]
    expected = softplus(test_values)
    assert_allclose(result, expected, rtol=1e-05)


def test_softsign():
    '''
    Test using a reference softsign implementation
    '''
    def softsign(x):
        return np.divide(x, np.ones_like(x) + np.absolute(x))

    x = K.placeholder(ndim=2)
    f = K.function([x], [activations.softsign(x)])
    test_values = get_standard_values()

    result = f([test_values])[0]
    expected = softsign(test_values)
    assert_allclose(result, expected, rtol=1e-05)


def test_sigmoid():
    '''
    Test using a numerically stable reference sigmoid implementation
    '''
    def ref_sigmoid(x):
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        else:
            z = np.exp(x)
            return z / (1 + z)
    sigmoid = np.vectorize(ref_sigmoid)

    x = K.placeholder(ndim=2)
    f = K.function([x], [activations.sigmoid(x)])
    test_values = get_standard_values()

    result = f([test_values])[0]
    expected = sigmoid(test_values)
    assert_allclose(result, expected, rtol=1e-05)


def test_hard_sigmoid():
    '''
    Test using a reference hard sigmoid implementation
    '''
    def ref_hard_sigmoid(x):
        '''
        Reference hard sigmoid with slope and shift values from theano, see
        https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py
        '''
        x = (x * 0.2) + 0.5
        z = 0.0 if x <= 0 else (1.0 if x >= 1 else x)
        return z
    hard_sigmoid = np.vectorize(ref_hard_sigmoid)

    x = K.placeholder(ndim=2)
    f = K.function([x], [activations.hard_sigmoid(x)])
    test_values = get_standard_values()

    result = f([test_values])[0]
    expected = hard_sigmoid(test_values)
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


def test_elu():
    x = K.placeholder(ndim=2)
    f = K.function([x], [activations.elu(x, 0.5)])

    test_values = get_standard_values()
    result = f([test_values])[0]

    # because no negatives in test values
    assert_allclose(result, test_values, rtol=1e-05)

    negative_values = np.array([[-1, -2]], dtype=K.floatx())
    result = f([negative_values])[0]
    true_result = (np.exp(negative_values) - 1) / 2

    assert_allclose(result, true_result)


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
