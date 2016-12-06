from __future__ import absolute_import
import pytest
import itertools
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras.layers import crf
from keras.models import Sequential
from keras.utils.test_utils import layer_test, keras_test


@keras_test
def test_fn_sparse_chain_crf_loss():
    n_samples, n_steps, n_classes = 2, 5, 11

    x_np = np.random.uniform(size=(n_samples, n_steps, n_classes)).astype(K.floatx())
    y_np = np.random.randint(n_classes, size=(n_samples, n_steps))
    U_np = np.random.uniform(size=(n_classes, n_classes)).astype(K.floatx())
    b_np = np.random.uniform(size=(n_classes)).astype(K.floatx())

    x = K.variable(x_np)
    y = K.variable(y_np, dtype='int32')
    U = K.variable(U_np)
    b = K.variable(b_np)

    assert_allclose(K.eval(crf.sparse_chain_crf_loss(y, x, U, b)),
                    K.eval(-(K.crf_path_energy(y, x, U, b) - K.crf_free_energy(x, U, b))),
                    rtol=1e-06)


@keras_test
def test_fn_chain_crf_loss():
    n_samples, n_steps, n_classes = 2, 5, 11

    x_np = np.random.uniform(size=(n_samples, n_steps, n_classes)).astype(K.floatx())
    y_sparse_np = np.random.randint(n_classes, size=(n_samples, n_steps))
    y_np = np.eye(n_classes)[y_sparse_np]
    U_np = np.random.uniform(size=(n_classes, n_classes)).astype(K.floatx())
    b_np = np.random.uniform(size=(n_classes)).astype(K.floatx())

    x = K.variable(x_np)
    y_sparse = K.variable(y_sparse_np, dtype='int32')
    y = K.variable(y_np)
    U = K.variable(U_np)
    b = K.variable(b_np)

    assert_allclose(K.eval(crf.chain_crf_loss(y, x, U, b)),
                    K.eval(crf.sparse_chain_crf_loss(y_sparse, x, U, b)),
                    rtol=1e-06)


@keras_test
def test_chain_crf_layer():
    from keras import regularizers
    from keras import constraints

    batch_size, maxlen, n_classes = 2, 5, 11

    layer_test(crf.ChainCRF,
               input_shape=(batch_size, maxlen, n_classes))

    layer_test(crf.ChainCRF,
               kwargs={'U_regularizer': regularizers.WeightRegularizer(l1=0.01),
                       'b_regularizer': regularizers.l1(0.01),
                       'U_constraint': constraints.MaxNorm(1),
                       'b_constraint': constraints.MaxNorm(1)},
               input_shape=(batch_size, maxlen, n_classes))


@keras_test
def test_chain_crf_dynamic_behaviour():
    batch_size, maxlen, n_classes = 2, 5, 11
    model = Sequential()
    layer = crf.ChainCRF(input_shape=(maxlen, n_classes))
    model.add(layer)
    model.compile(loss=layer.loss, optimizer='sgd')
    x = np.random.random((batch_size, maxlen, n_classes))
    y = np.random.randint(n_classes, size=(batch_size, maxlen))
    y = np.eye(n_classes)[y]
    model.train_on_batch(x, y)


def test_sparse_chain_crf_dynamic_behaviour():
    batch_size, maxlen, n_classes = 2, 5, 11
    model = Sequential()
    layer = crf.ChainCRF(input_shape=(maxlen, n_classes))
    model.add(layer)
    model.compile(loss=layer.sparse_loss, optimizer='sgd')
    x = np.random.random((batch_size, maxlen, n_classes))
    y_sparse = np.expand_dims(np.random.randint(n_classes, size=(batch_size, maxlen)), -1)
    model.train_on_batch(x, y_sparse)


@keras_test
def test_chain_crf_call():
    batch_size, maxlen, n_classes = 2, 5, 11
    batch_input_shape = (None, maxlen, n_classes)
    layer = crf.ChainCRF(batch_input_shape=batch_input_shape)
    layer.build(batch_input_shape)
    x = K.placeholder(shape=batch_input_shape)
    fn = K.function([x, K.learning_phase()], [layer(x)])
    x_np = np.random.random((batch_size, maxlen, n_classes))

    # check that inputs pass through while training
    fn_outputs_train = fn([x_np, 1])
    assert_allclose(fn_outputs_train[0], x_np)
    fn_outputs_test = fn([x_np, 0])


@keras_test
def test_chain_crf_loss():
    batch_size, maxlen, n_classes = 2, 5, 11
    batch_input_shape = (None, maxlen, n_classes)
    layer = crf.ChainCRF(batch_input_shape=batch_input_shape)
    layer.build(batch_input_shape)
    y = K.placeholder(shape=(None, maxlen, n_classes))
    x = K.placeholder(shape=(None, maxlen, n_classes))
    fn = K.function([y, x, K.learning_phase()], [layer.loss(y, x)])

    x_np = np.random.random((batch_size, maxlen, n_classes))
    y_np = np.eye(n_classes)[np.random.randint(n_classes, size=(batch_size, maxlen))]

    fn_loss_train = fn([y_np, x_np, 1])[0]
    assert fn_loss_train.shape == (batch_size, )
    fn_loss_test = fn([y_np, x_np, 0])


if __name__ == '__main__':
    pytest.main([__file__])
