import unittest
import math
from keras import backend as K
import numpy as np
from numpy.testing import assert_allclose


def get_standard_values():
    '''
    These are just a set of floats used for testing the activation
    functions, and are useful in multiple tests.
    '''
    return [0, 0.1, 0.5, 0.9, 1.0]


class TestActivations(unittest.TestCase):

    def test_softmax(self):
        from keras.activations import softmax as s

        # Test using a reference implementation of softmax
        def softmax(values):
            m = max(values)
            values = np.array(values)
            e = np.exp(values - m)
            return e / np.sum(e)

        x = K.placeholder(ndim=1)
        exp = s(x)
        f = K.function([x], exp)
        test_values = get_standard_values()

        result = f(test_values)
        expected = softmax(test_values)
        assert_allclose(result.flatten(), expected)

    def test_relu(self):
        '''
        Relu implementation doesn't depend on the value being
        a theano variable. Testing ints, floats and theano tensors.
        '''
        from keras.activations import relu as r

        assert r(5) == 5
        assert r(-5) == 0
        assert r(-0.1) == 0
        assert r(0.1) == 0.1

        x = K.placeholder(ndim=1)
        exp = r(x)
        f = K.function([x], exp)

        test_values = get_standard_values()
        result = f(test_values)

        # because no negatives in test values
        assert_allclose(result.flatten(), test_values)

    def test_tanh(self):
        from keras.activations import tanh as t
        test_values = get_standard_values()

        x = K.placeholder(ndim=1)
        exp = t(x)
        f = K.function([x], exp)

        result = f(test_values)
        expected = [math.tanh(v) for v in test_values]
        assert_allclose(result.flatten(), expected)

    def test_linear(self):
        '''
        This function does no input validation, it just returns the thing
        that was passed in.
        '''
        from keras.activations import linear as l

        xs = [1, 5, True, None, 'foo']
        for x in xs:
            assert x == l(x)

if __name__ == '__main__':
    unittest.main()
