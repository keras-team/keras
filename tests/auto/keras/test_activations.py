import keras
import theano
import theano.tensor as T

import numpy

def list_assert_equal(a, b, round_to=7):
    pairs = zip(a, b)
    for i, j in pairs:
        assert round(i, round_to) == round(j, round_to)


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
    test_values=[0,0.1,0.5,0.9,1.0]

    result = f(test_values)
    expected = softmax(test_values)

    print(str(result))
    print(str(expected))

    list_assert_equal(result, expected)

def test_linear():

    from keras.activations import linear as l

    xs = [1, 5, True, None, 'foo']

    for x in xs:
        assert x == l(x)
