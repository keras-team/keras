import math

import keras
import theano
import theano.tensor as T

import numpy

def list_assert_equal(a, b, round_to=7):
    '''
    This will do a pairwise, rounded equality test across two lists of
    numbers.
    '''
    pairs = zip(a, b)
    for i, j in pairs:
        assert round(i, round_to) == round(j, round_to)

def get_standard_values():
    '''
    These are just a set of floats used for testing the activation
    functions, and are useful in multiple tests.
    '''

    return [0,0.1,0.5,0.9,1.0]

def test_softmax():

    from keras.activations import softmax as s

    # Test using a reference implementation of softmax
    def softmax(values):
        m = max(values)
        values = numpy.array(values)
        e = numpy.exp(values - m)
        dist = list(e / numpy.sum(e))

        return dist

    x = T.vector()
    exp = s(x)
    f = theano.function([x], exp)
    test_values=get_standard_values()

    result = f(test_values)
    expected = softmax(test_values)

    print(str(result))
    print(str(expected))

    list_assert_equal(result, expected)

def test_relu():
    '''
    Relu implementation doesn't depend on the value being
    a theano variable. Testing ints, floats and theano tensors.
    '''

    from keras.activations import relu as r

    assert r(5) == 5
    assert r(-5) == 0
    assert r(-0.1) == 0
    assert r(0.1) == 0.1

    x = T.vector()
    exp = r(x)
    f = theano.function([x], exp)

    test_values = get_standard_values()
    result = f(test_values)

    list_assert_equal(result, test_values) # because no negatives in test values


def test_tanh():

    from keras.activations import tanh as t
    test_values = get_standard_values()

    x = T.vector()
    exp = t(x)
    f = theano.function([x], exp)

    result = f(test_values)
    expected = [math.tanh(v) for v in test_values]

    print(result)
    print(expected)

    list_assert_equal(result, expected)


def test_linear():
    '''
    This function does no input validation, it just returns the thing
    that was passed in.
    '''

    from keras.activations import linear as l

    xs = [1, 5, True, None, 'foo']

    for x in xs:
        assert x == l(x)
