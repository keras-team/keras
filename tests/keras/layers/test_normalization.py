import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.layers.core import Dense, Activation
from keras.utils.test_utils import layer_test
from keras.layers import normalization
from keras.models import Sequential, Graph
from keras import backend as K


input_1 = np.arange(10)
input_2 = np.zeros(10)
input_3 = np.ones((10))
input_shapes = [np.ones((10, 10)), np.ones((10, 10, 10))]


def basic_batchnorm_test():
    layer_test(normalization.BatchNormalization,
               kwargs={'mode': 1},
               input_shape=(3, 4, 2))
    layer_test(normalization.BatchNormalization,
               kwargs={'mode': 0},
               input_shape=(3, 4, 2))


def test_batchnorm_mode_0_or_2():
    for mode in [0, 2]:
        model = Sequential()
        norm_m0 = normalization.BatchNormalization(mode=mode, input_shape=(10,))
        model.add(norm_m0)
        model.compile(loss='mse', optimizer='sgd')

        # centered on 5.0, variance 10.0
        X = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10))
        model.fit(X, X, nb_epoch=5, verbose=0)
        out = model.predict(X)
        out -= K.eval(norm_m0.beta)
        out /= K.eval(norm_m0.gamma)

        assert_allclose(out.mean(), 0.0, atol=1e-1)
        assert_allclose(out.std(), 1.0, atol=1e-1)


def test_batchnorm_mode_0_convnet():
    model = Sequential()
    norm_m0 = normalization.BatchNormalization(mode=0, axis=1, input_shape=(3, 4, 4))
    model.add(norm_m0)
    model.compile(loss='mse', optimizer='sgd')

    # centered on 5.0, variance 10.0
    X = np.random.normal(loc=5.0, scale=10.0, size=(1000, 3, 4, 4))
    model.fit(X, X, nb_epoch=5, verbose=0)
    out = model.predict(X)
    out -= np.reshape(K.eval(norm_m0.beta), (1, 3, 1, 1))
    out /= np.reshape(K.eval(norm_m0.gamma), (1, 3, 1, 1))

    assert_allclose(np.mean(out, axis=(0, 2, 3)), 0.0, atol=1e-1)
    assert_allclose(np.std(out, axis=(0, 2, 3)), 1.0, atol=1e-1)


def test_batchnorm_mode_1():
    norm_m1 = normalization.BatchNormalization(input_shape=(10,), mode=1)
    norm_m1.build(input_shape=(None, 10))

    for inp in [input_1, input_2, input_3]:
        out = (norm_m1.call(K.variable(inp)) - norm_m1.beta) / norm_m1.gamma
        assert_allclose(K.eval(K.mean(out)), 0.0, atol=1e-1)
        if inp.std() > 0.:
            assert_allclose(K.eval(K.std(out)), 1.0, atol=1e-1)
        else:
            assert_allclose(K.eval(K.std(out)), 0.0, atol=1e-1)


if __name__ == '__main__':
    pytest.main([__file__])
