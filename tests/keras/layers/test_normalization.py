import pytest
import numpy as np
from keras.layers.core import Dense, Activation
from numpy.testing import assert_allclose

from keras.layers import normalization
from keras.models import Sequential, Graph
from keras import backend as K


input_1 = np.arange(10)
input_2 = np.zeros(10)
input_3 = np.ones((10))
input_shapes = [np.ones((10, 10)), np.ones((10, 10, 10))]


def test_batchnorm_mode_0():
    np.random.seed(1337)
    model = Sequential()
    norm_m0 = normalization.BatchNormalization(input_shape=(10,))
    model.add(norm_m0)
    model.compile(loss='mse', optimizer='sgd')

    # centered on 5.0, variance 10.0
    X = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10))
    model.fit(X, X, nb_epoch=5, verbose=0)
    norm_m0.input = K.variable(X)
    out = (norm_m0.get_output(train=True) - norm_m0.beta) / norm_m0.gamma

    assert_allclose(K.eval(K.mean(out)), 0.0, atol=1e-1)
    assert_allclose(K.eval(K.std(out)), 1.0, atol=1e-1)


def test_batchnorm_mode_1():
    np.random.seed(1337)
    norm_m1 = normalization.BatchNormalization(input_shape=(10,), mode=1)

    for inp in [input_1, input_2, input_3]:
        norm_m1.input = K.variable(inp)
        out = (norm_m1.get_output(train=True) - norm_m1.beta) / norm_m1.gamma
        assert_allclose(K.eval(K.mean(out)), 0.0, atol=1e-1)
        if inp.std() > 0.:
            assert_allclose(K.eval(K.std(out)), 1.0, atol=1e-1)
        else:
            assert_allclose(K.eval(K.std(out)), 0.0, atol=1e-1)


def test_batchnorm_shapes():
    """
    Test batch normalization with various input shapes
    """
    for inp in input_shapes:
        norm_m0 = normalization.BatchNormalization(input_shape=inp.shape, mode=0)
        norm_m0.input = K.variable(inp)
        out = (norm_m0.get_output(train=True) - norm_m0.beta) / norm_m0.gamma

        norm_m1 = normalization.BatchNormalization(input_shape=inp.shape, mode=1)
        norm_m1.input = K.variable(inp)
        out = (norm_m1.get_output(train=True) - norm_m1.beta) / norm_m1.gamma


def test_batchnorm_weight_init():
    """
    Test weight initialization
    """
    np.random.seed(1337)
    norm_m1 = normalization.BatchNormalization(input_shape=(10,), mode=1,
                                               weights=[np.ones(10), np.ones(10), np.zeros(10), np.zeros(10)])

    for inp in [input_1, input_2, input_3]:
        norm_m1.input = K.variable(inp)
        out = (norm_m1.get_output(train=True) - np.ones(10)) / 1.
        assert_allclose(K.eval(K.mean(out)), 0.0, atol=1e-1)
        if inp.std() > 0.:
            assert_allclose(K.eval(K.std(out)), 1.0, atol=1e-1)
        else:
            assert_allclose(K.eval(K.std(out)), 0.0, atol=1e-1)

    assert_allclose(K.eval(norm_m1.gamma), np.ones(10), atol=1e-1)
    assert_allclose(K.eval(norm_m1.beta), np.ones(10), atol=1e-1)


def test_batchnorm_config():
    norm = normalization.BatchNormalization(input_shape=(10, 10), mode=1,
                                            epsilon=0.1, momentum=0.9)
    conf = norm.get_config()
    del conf['cache_enabled']
    del conf['trainable']
    del conf['custom_name']
    conf_target = {"input_shape": (10, 10),
                   "name": normalization.BatchNormalization.__name__,
                   "epsilon": 0.1, "mode": 1, "momentum": 0.9}
    assert(conf == conf_target)


def test_batchnorm_save_weights():
    norm = normalization.BatchNormalization(input_shape=(10, 10), mode=1,
                                            epsilon=0.1)
    weights = norm.get_weights()
    assert(len(weights) == 4)
    norm.set_weights(weights)


def test_batchnorm_nested():
    # regression test for issue #1386
    g = Graph()
    g.add_input("input", input_shape=[20])
    g.add_node(Dense(10), "dense", "input")
    g.add_node(normalization.BatchNormalization(), "bn", "dense")
    g.add_node(Activation('relu'), "activ", "bn")
    g.add_output("output", "activ")

    g2 = Graph()
    g2.add_input("input", input_shape=[10])
    g2.add_node(Dense(15), "dense", "input")
    g2.add_node(normalization.BatchNormalization(), "bn", "dense")
    g2.add_node(Activation('relu'), "activ", "bn")
    g2.add_output("output", "activ")

    model = Sequential()
    model.add(g)
    model.add(g2)
    model.compile(loss="mse", optimizer="adadelta")


if __name__ == '__main__':
    pytest.main([__file__])
