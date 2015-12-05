# encoding: utf-8
"""Test seya.layers.recurrent module"""

from __future__ import print_function

import unittest
import numpy as np

from numpy.testing import assert_allclose

from keras import backend as K
from keras.layers.core import Dense
from keras.models import Sequential
floatX = K.common._FLOATX


class TestCall(unittest.TestCase):
    """Test __call__ methods"""

    def test_layer_call(self):
        """Test keras.layers.core.Layer.__call__"""
        nb_samples, input_dim, output_dim = 3, 10, 5
        layer = Dense(output_dim, input_dim=input_dim)
        W = np.asarray(K.eval(layer.W))
        X = K.placeholder(ndim=2)
        Y = layer(X)
        F = K.function([X], [Y])

        x = np.random.randn(nb_samples, input_dim).astype(floatX)
        y = F([x])[0]
        assert_allclose(np.dot(x, W), y)

    def test_sequential_call(self):
        """Test keras.models.Sequential.__call__"""
        nb_samples, input_dim, output_dim = 3, 10, 5
        model = Sequential()
        model.add(Dense(output_dim=output_dim, input_dim=input_dim))
        model.compile('sgd', 'mse')

        X = K.placeholder(ndim=2)
        Y = model(X)
        F = K.function([X], [Y])

        x = np.random.randn(nb_samples, input_dim).astype(floatX)
        y1 = F([x])[0]
        y2 = model.predict(x)
        # results of __call__ should match model.predict
        assert_allclose(y1, y2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
