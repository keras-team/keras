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
                    K.eval(-(crf.path_energy(y, x, U, b) - crf.free_energy(x, U, b))),
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
def test_free_energy():

    def np_free_energy_single(x, U, b):
        maxlen = x.shape[0]
        n_classes = U.shape[0]
        all_paths = np.array(list(itertools.product(*([list(range(n_classes))] * maxlen))))
        t = np.arange(maxlen)

        scores = [x[t, y[t]].sum() + b[y[0]] + U[y[t[1:]-1], y[t[1:]]].sum() for y in all_paths]
        scores = np.asarray(scores)
        return np.log(np.sum(np.exp(scores), axis=0))

    batch_size, maxlen, n_classes = 2, 5, 3

    x_np = np.random.uniform(size=(batch_size, maxlen, n_classes))
    U_np = np.random.uniform(size=(n_classes, n_classes))
    b_np = np.random.uniform(size=(n_classes))

    expected = np.asarray([np_free_energy_single(x_np[k], U_np, b_np) for k in range(batch_size)])

    x = K.placeholder(shape=(None, None, n_classes))
    U = K.placeholder(shape=(n_classes, n_classes))
    b = K.placeholder(shape=(n_classes, ))
    fn = K.function([x, U, b], [crf.free_energy(x, U, b)])

    assert_allclose(fn([x_np, U_np, b_np])[0], expected, rtol=1e-05)

    # Check that gradients are computable
    df = K.gradients(K.mean(crf.free_energy(x, U, b), axis=0), [x, U, b])
    assert len(df) == 3


@keras_test
def test_path_energy():

    def np_path_energy_single(y, x, U, b):
        n_steps = x.shape[0]
        t = np.arange(n_steps)
        tag_path_energy = x[t, y[t]].sum()
        boundary_energy = b[y[0]]
        transition_energy = U[y[t[1:]-1], y[t[1:]]].sum()
        return tag_path_energy + boundary_energy + transition_energy

    batch_size, maxlen, n_classes = 2, 5, 3

    y_np = np.random.randint(n_classes, size=(batch_size, maxlen)).astype(np.int32)
    x_np = np.random.uniform(size=(batch_size, maxlen, n_classes))
    U_np = np.random.uniform(size=(n_classes, n_classes))
    b_np = np.random.uniform(size=(n_classes))

    expected = np.asarray([np_path_energy_single(y_np[k], x_np[k], U_np, b_np) for k in range(batch_size)])

    y = K.placeholder(shape=(None, maxlen), dtype='int32')
    x = K.placeholder(shape=(None, maxlen, n_classes))
    U = K.placeholder(shape=(n_classes, n_classes))
    b = K.placeholder(shape=(n_classes, ))
    fn = K.function([y, x, U, b], [crf.path_energy(y, x, U, b)])

    assert_allclose(fn([y_np, x_np, U_np, b_np])[0], expected, rtol=1e-05)

    # Check that gradients are computable
    df = K.gradients(K.mean(crf.path_energy(y, x, U, b), axis=0), [x, U, b])
    assert len(df) == 3


@keras_test
def test_viterbi_decode():

    def np_decode_single(x, U, b):
        maxlen = x.shape[0]
        n_classes = U.shape[0]
        all_paths = np.array(list(itertools.product(*([list(range(n_classes))] * maxlen))))
        t = np.arange(maxlen)

        scores = [x[t, y[t]].sum() + b[y[0]] + U[y[t[1:]-1], y[t[1:]]].sum() for y in all_paths]
        scores = np.asarray(scores)
        best_path_index = np.argmax(scores, axis=0)
        return all_paths[best_path_index]

    batch_size, maxlen, n_classes = 2, 7, 3

    x_np = np.random.uniform(size=(batch_size, maxlen, n_classes))
    A_np = np.random.uniform(size=(n_classes, n_classes))
    b_np = np.random.uniform(size=(n_classes))

    expected = np.asarray([np_decode_single(x_np[k], A_np, b_np) for k in range(batch_size)])

    x = K.variable(x_np)
    U = K.variable(A_np)
    b = K.variable(b_np)
    actual = K.eval(crf.viterbi_decode(x, U, b))
    assert_allclose(actual, expected, rtol=1e-05)


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
    assert fn_outputs_test[0].shape == x_np.shape


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
    fn_loss_test = fn([y_np, x_np, 0])[0]
    assert fn_loss_test.shape == (batch_size, )


if __name__ == '__main__':
    pytest.main([__file__])
