import unittest
import numpy as np
from numpy.testing import assert_allclose
import theano

from keras.layers import core


class TestLayerBase(unittest.TestCase):
    def test_input_output(self):
        nb_samples = 10
        input_dim = 5
        layer = core.Layer()

        # As long as there is no input, an error should be raised.
        for train in [True, False]:
            self.assertRaises(AttributeError, layer.get_input, train)
            self.assertRaises(AttributeError, layer.get_output, train)

        # Once an input is provided, it should be reachable through the
        # appropriate getters
        input = np.ones((nb_samples, input_dim))
        layer.input = theano.shared(value=input)
        for train in [True, False]:
            assert_allclose(layer.get_input(train).eval(), input)
            assert_allclose(layer.get_output(train).eval(), input)


if __name__ == '__main__':
    unittest.main()
