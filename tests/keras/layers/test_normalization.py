import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.layers import Dense, Activation, Input
from keras.utils.test_utils import layer_test, keras_test
from keras.layers import normalization
from keras.models import Sequential, Model
from keras import backend as K

input_1 = np.arange(10)
input_2 = np.zeros(10)
input_3 = np.ones((10))
input_shapes = [np.ones((10, 10)), np.ones((10, 10, 10))]


@keras_test
def basic_batchnorm_test():
    from keras import regularizers
    layer_test(normalization.BatchNormalization,
               kwargs={'mode': 1,
                       'gamma_regularizer': regularizers.l2(0.01),
                       'beta_regularizer': regularizers.l2(0.01)},
               input_shape=(3, 4, 2))
    layer_test(normalization.BatchNormalization,
               kwargs={'mode': 0},
               input_shape=(3, 4, 2))


@keras_test
def test_batchnorm_mode_0_or_2():
    for mode in [0, 2]:
        model = Sequential()
        norm_m0 = normalization.BatchNormalization(mode=mode, input_shape=(10,), momentum=0.8)
        model.add(norm_m0)
        model.compile(loss='mse', optimizer='sgd')

        # centered on 5.0, variance 10.0
        X = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10))
        model.fit(X, X, nb_epoch=4, verbose=0)
        out = model.predict(X)
        out -= K.eval(norm_m0.beta)
        out /= K.eval(norm_m0.gamma)

        assert_allclose(out.mean(), 0.0, atol=1e-1)
        assert_allclose(out.std(), 1.0, atol=1e-1)


@keras_test
def test_batchnorm_mode_0_or_2_twice():
    # This is a regression test for issue #4881 with the old
    # batch normalization functions in the Theano backend.
    model = Sequential()
    model.add(normalization.BatchNormalization(mode=0, input_shape=(10, 5, 5), axis=1))
    model.add(normalization.BatchNormalization(mode=0, input_shape=(10, 5, 5), axis=1))
    model.compile(loss='mse', optimizer='sgd')

    X = np.random.normal(loc=5.0, scale=10.0, size=(20, 10, 5, 5))
    model.fit(X, X, nb_epoch=1, verbose=0)
    model.predict(X)


@keras_test
def test_batchnorm_mode_0_convnet():
    model = Sequential()
    norm_m0 = normalization.BatchNormalization(mode=0, axis=1, input_shape=(3, 4, 4), momentum=0.8)
    model.add(norm_m0)
    model.compile(loss='mse', optimizer='sgd')

    # centered on 5.0, variance 10.0
    X = np.random.normal(loc=5.0, scale=10.0, size=(1000, 3, 4, 4))
    model.fit(X, X, nb_epoch=4, verbose=0)
    out = model.predict(X)
    out -= np.reshape(K.eval(norm_m0.beta), (1, 3, 1, 1))
    out /= np.reshape(K.eval(norm_m0.gamma), (1, 3, 1, 1))

    assert_allclose(np.mean(out, axis=(0, 2, 3)), 0.0, atol=1e-1)
    assert_allclose(np.std(out, axis=(0, 2, 3)), 1.0, atol=1e-1)


@keras_test
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


@keras_test
def test_shared_batchnorm():
    '''Test that a BN layer can be shared
    across different data streams.
    '''
    # Test single layer reuse
    bn = normalization.BatchNormalization(input_shape=(10,), mode=0)
    x1 = Input(shape=(10,))
    bn(x1)

    x2 = Input(shape=(10,))
    y2 = bn(x2)

    x = np.random.normal(loc=5.0, scale=10.0, size=(2, 10))
    model = Model(x2, y2)
    assert len(model.updates) == 2
    model.compile('sgd', 'mse')
    model.train_on_batch(x, x)

    # Test model-level reuse
    x3 = Input(shape=(10,))
    y3 = model(x3)
    new_model = Model(x3, y3)
    assert len(model.updates) == 2
    new_model.compile('sgd', 'mse')
    new_model.train_on_batch(x, x)


if __name__ == '__main__':
    pytest.main([__file__])
