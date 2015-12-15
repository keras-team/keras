"""Test keras.layers.core.Layer.__call__"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras.layers.core import Dense
from keras.models import Sequential


def test_layer_call():
    """Test keras.layers.core.Layer.__call__"""
    nb_samples, input_dim, output_dim = 3, 10, 5
    layer = Dense(output_dim, input_dim=input_dim)
    W = np.asarray(K.eval(layer.W)).astype(K.floatx())
    X = K.placeholder(ndim=2)
    Y = layer(X)
    f = K.function([X], [Y])

    x = np.ones((nb_samples, input_dim)).astype(K.floatx())
    y = f([x])[0].astype(K.floatx())
    t = np.dot(x, W).astype(K.floatx())
    assert_allclose(t, y, rtol=.2)


def test_sequential_call():
    """Test keras.models.Sequential.__call__"""
    nb_samples, input_dim, output_dim = 3, 10, 5
    model = Sequential()
    model.add(Dense(output_dim=output_dim, input_dim=input_dim))
    model.compile('sgd', 'mse')

    # test flat model
    X = K.placeholder(ndim=2)
    Y = model(X)
    f = K.function([X], [Y])

    x = np.ones((nb_samples, input_dim)).astype(K.floatx())
    y1 = f([x])[0].astype(K.floatx())
    y2 = model.predict(x)
    # results of __call__ should match model.predict
    assert_allclose(y1, y2)

    # test nested model
    model2 = Sequential()
    model2.add(model)
    model2.compile('sgd', 'mse')

    Y2 = model2(X)
    f = K.function([X], [Y2])

    y1 = f([x])[0].astype(K.floatx())
    y2 = model2.predict(x)
    # results of __call__ should match model.predict
    assert_allclose(y1, y2)


if __name__ == '__main__':
    pytest.main([__file__])
