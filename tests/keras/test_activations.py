import pytest
from keras import backend as K
import numpy as np
from numpy.testing import assert_allclose


def get_standard_values():
    '''
    These are just a set of floats used for testing the activation
    functions, and are useful in multiple tests.
    '''
    return np.array([[0, 0.1, 0.5, 0.9, 1.0]], dtype=K.floatx())


# Reference sigmoid, numerically stable
def ref_sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        z = np.exp(x)
        return z / (1 + z)
vec_sigmoid = np.vectorize(ref_sigmoid)


# Reference hard sigmoid with slope and shift values from theano, see
# https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py
def ref_hard_sigmoid(x):
    x = (x * 0.2) + 0.5
    z = 0.0 if x <= 0 else (1.0 if x >= 1 else x)
    return z
vec_hard_sigmoid = np.vectorize(ref_hard_sigmoid)


# Test using a reference implementation of softmax
def softmax(values):
    m = np.max(values)
    e = np.exp(values - m)
    return e / np.sum(e)


def test_softmax():
    from keras.activations import softmax as s

    x = K.placeholder(ndim=2)
    exp = s(x)
    f = K.function([x], [exp])
    test_values = get_standard_values()

    result = f([test_values])[0]
    expected = softmax(test_values)
    assert_allclose(result, expected, rtol=1e-05)


def test_sigmoid():
    from keras.activations import sigmoid

    x = K.placeholder(ndim=2)
    sig_out = sigmoid(x)
    f = K.function([x], [sig_out])
    test_values = get_standard_values()

    result = f([test_values])[0]
    expected = vec_sigmoid(test_values)
    assert_allclose(result, expected, rtol=1e-05)


def test_hard_sigmoid():
    from keras.activations import hard_sigmoid

    x = K.placeholder(ndim=2)
    hard_sig_out = hard_sigmoid(x)
    f = K.function([x], [hard_sig_out])
    test_values = get_standard_values()

    result = f([test_values])[0]
    expected = vec_hard_sigmoid(test_values)
    assert_allclose(result, expected, rtol=1e-05)


def test_relu():
    '''
    Relu implementation doesn't depend on the value being
    a theano variable. Testing ints, floats and theano tensors.
    '''
    from keras.activations import relu as r

    x = K.placeholder(ndim=2)
    exp = r(x)
    f = K.function([x], [exp])

    test_values = get_standard_values()
    result = f([test_values])[0]

    # because no negatives in test values
    assert_allclose(result, test_values, rtol=1e-05)


def test_tanh():
    from keras.activations import tanh as t
    test_values = get_standard_values()

    x = K.placeholder(ndim=2)
    exp = t(x)
    f = K.function([x], [exp])

    result = f([test_values])[0]
    expected = np.tanh(test_values)
    assert_allclose(result, expected, rtol=1e-05)


def test_linear():
    '''
    This function does no input validation, it just returns the thing
    that was passed in.
    '''
    from keras.activations import linear as l

    xs = [1, 5, True, None, 'foo']
    for x in xs:
        assert x == l(x)
