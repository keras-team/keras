import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.layers import Input
from keras import regularizers
from keras.utils.test_utils import layer_test, keras_test
from keras.layers import normalization
from keras.models import Sequential, Model
from keras import backend as K

input_1 = np.arange(10)
input_2 = np.zeros(10)
input_3 = np.ones((10))
input_shapes = [np.ones((10, 10)), np.ones((10, 10, 10))]


@keras_test
def test_basic_batchnorm():
    layer_test(normalization.BatchNormalization,
               kwargs={'momentum': 0.9,
                       'epsilon': 0.1,
                       'gamma_regularizer': regularizers.l2(0.01),
                       'beta_regularizer': regularizers.l2(0.01)},
               input_shape=(3, 4, 2))
    layer_test(normalization.BatchNormalization,
               kwargs={'momentum': 0.9,
                       'epsilon': 0.1,
                       'axis': 1},
               input_shape=(3, 4, 2))
    layer_test(normalization.BatchNormalization,
               kwargs={'gamma_initializer': 'ones',
                       'beta_initializer': 'ones',
                       'moving_mean_initializer': 'zeros',
                       'moving_variance_initializer': 'ones'},
               input_shape=(3, 4, 2, 4))
    if K.backend() != 'theano':
        layer_test(normalization.BatchNormalization,
                   kwargs={'momentum': 0.9,
                           'epsilon': 0.1,
                           'axis': 1,
                           'scale': False,
                           'center': False},
                   input_shape=(3, 4, 2, 4))


@keras_test
def test_batchnorm_correctness_1d():
    model = Sequential()
    norm = normalization.BatchNormalization(input_shape=(10,), momentum=0.8)
    model.add(norm)
    model.compile(loss='mse', optimizer='rmsprop')

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10))
    model.fit(x, x, epochs=5, verbose=0)
    out = model.predict(x)
    out -= K.eval(norm.beta)
    out /= K.eval(norm.gamma)

    assert_allclose(out.mean(), 0.0, atol=1e-1)
    assert_allclose(out.std(), 1.0, atol=1e-1)


@keras_test
def test_batchnorm_correctness_2d():
    model = Sequential()
    norm = normalization.BatchNormalization(axis=1, input_shape=(10, 6), momentum=0.8)
    model.add(norm)
    model.compile(loss='mse', optimizer='rmsprop')

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10, 6))
    model.fit(x, x, epochs=5, verbose=0)
    out = model.predict(x)
    out -= np.reshape(K.eval(norm.beta), (1, 10, 1))
    out /= np.reshape(K.eval(norm.gamma), (1, 10, 1))

    assert_allclose(out.mean(axis=(0, 2)), 0.0, atol=1.1e-1)
    assert_allclose(out.std(axis=(0, 2)), 1.0, atol=1.1e-1)


@keras_test
def test_batchnorm_training_argument():
    bn1 = normalization.BatchNormalization(input_shape=(10,))
    x1 = Input(shape=(10,))
    y1 = bn1(x1, training=True)
    assert bn1.updates

    model1 = Model(x1, y1)
    np.random.seed(123)
    x = np.random.normal(loc=5.0, scale=10.0, size=(20, 10))
    output_a = model1.predict(x)

    model1.compile(loss='mse', optimizer='rmsprop')
    model1.fit(x, x, epochs=1, verbose=0)
    output_b = model1.predict(x)
    assert np.abs(np.sum(output_a - output_b)) > 0.1
    assert_allclose(output_b.mean(), 0.0, atol=1e-1)
    assert_allclose(output_b.std(), 1.0, atol=1e-1)

    bn2 = normalization.BatchNormalization(input_shape=(10,))
    x2 = Input(shape=(10,))
    bn2(x2, training=False)
    assert not bn2.updates


@keras_test
def test_batchnorm_mode_twice():
    # This is a regression test for issue #4881 with the old
    # batch normalization functions in the Theano backend.
    model = Sequential()
    model.add(normalization.BatchNormalization(input_shape=(10, 5, 5), axis=1))
    model.add(normalization.BatchNormalization(input_shape=(10, 5, 5), axis=1))
    model.compile(loss='mse', optimizer='sgd')

    x = np.random.normal(loc=5.0, scale=10.0, size=(20, 10, 5, 5))
    model.fit(x, x, epochs=1, verbose=0)
    model.predict(x)


@keras_test
def test_batchnorm_convnet():
    model = Sequential()
    norm = normalization.BatchNormalization(axis=1, input_shape=(3, 4, 4), momentum=0.8)
    model.add(norm)
    model.compile(loss='mse', optimizer='sgd')

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 3, 4, 4))
    model.fit(x, x, epochs=4, verbose=0)
    out = model.predict(x)
    out -= np.reshape(K.eval(norm.beta), (1, 3, 1, 1))
    out /= np.reshape(K.eval(norm.gamma), (1, 3, 1, 1))

    assert_allclose(np.mean(out, axis=(0, 2, 3)), 0.0, atol=1e-1)
    assert_allclose(np.std(out, axis=(0, 2, 3)), 1.0, atol=1e-1)


@keras_test
@pytest.mark.skipif((K.backend() == 'theano'),
                    reason='Bug with theano backend')
def test_batchnorm_convnet_no_center_no_scale():
    model = Sequential()
    norm = normalization.BatchNormalization(axis=-1, center=False, scale=False,
                                            input_shape=(3, 4, 4), momentum=0.8)
    model.add(norm)
    model.compile(loss='mse', optimizer='sgd')

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 3, 4, 4))
    model.fit(x, x, epochs=4, verbose=0)
    out = model.predict(x)

    assert_allclose(np.mean(out, axis=(0, 2, 3)), 0.0, atol=1e-1)
    assert_allclose(np.std(out, axis=(0, 2, 3)), 1.0, atol=1e-1)


@keras_test
def test_shared_batchnorm():
    '''Test that a BN layer can be shared
    across different data streams.
    '''
    # Test single layer reuse
    bn = normalization.BatchNormalization(input_shape=(10,))
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


@keras_test
def test_that_trainable_disables_updates():
    val_a = np.random.random((10, 4))
    val_out = np.random.random((10, 4))

    a = Input(shape=(4,))
    layer = normalization.BatchNormalization(input_shape=(4,))
    b = layer(a)
    model = Model(a, b)

    model.trainable = False
    assert not model.updates

    model.compile('sgd', 'mse')
    assert not model.updates

    x1 = model.predict(val_a)
    model.train_on_batch(val_a, val_out)
    x2 = model.predict(val_a)
    assert_allclose(x1, x2, atol=1e-7)

    model.trainable = True
    model.compile('sgd', 'mse')
    assert model.updates

    model.train_on_batch(val_a, val_out)
    x2 = model.predict(val_a)
    assert np.abs(np.sum(x1 - x2)) > 1e-5

    layer.trainable = False
    model.compile('sgd', 'mse')
    assert not model.updates

    x1 = model.predict(val_a)
    model.train_on_batch(val_a, val_out)
    x2 = model.predict(val_a)
    assert_allclose(x1, x2, atol=1e-7)


if __name__ == '__main__':
    pytest.main([__file__])
