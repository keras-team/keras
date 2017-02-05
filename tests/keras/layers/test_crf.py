from __future__ import absolute_import
import pytest
import itertools
import numpy as np
np.random.seed(1337)
from numpy.testing import assert_allclose

from keras import backend as K
from keras.layers import crf, Embedding
from keras.models import Sequential
from keras.utils.test_utils import layer_test, keras_test


@keras_test
def test_fn_sparse_chain_crf_loss():
    '''Checks the equality loss = - (path_energy - free_energy)'''
    batch_size, maxlen, n_classes = 5, 7, 11

    x_np, y_np, mask_np, U_np, b_start_np, b_end_np = create_data(batch_size, maxlen, n_classes)

    x, y, mask, U, b_start, b_end = to_variables([x_np, y_np, mask_np, U_np, b_start_np, b_end_np])

    actual = K.eval(crf.sparse_chain_crf_loss(y, x, U, b_start, b_end, mask))
    path_energy = K.eval(crf.path_energy(y, x, U, b_start, b_end, mask))
    free_energy = K.eval(crf.free_energy(x, U, b_start, b_end, mask))
    expected = np.expand_dims(-(path_energy - free_energy), 1)

    assert_allclose(actual, expected, rtol=1e-06)


@keras_test
def test_fn_chain_crf_loss():
    '''Checks that the loss remains the same when using one-hot encoded y.'''
    batch_size, maxlen, n_classes = 5, 7, 11

    x_np, y_np, mask_np, U_np, b_start_np, b_end_np = create_data(batch_size, maxlen, n_classes)
    y_one_hot_np = np.eye(n_classes)[y_np]

    x, y, y_one_hot, mask, U, b_start, b_end = to_variables([x_np, y_np, y_one_hot_np, mask_np, U_np, b_start_np, b_end_np])

    assert_allclose(K.eval(crf.chain_crf_loss(y_one_hot, x, U, b_start, b_end)),
                    K.eval(crf.sparse_chain_crf_loss(y, x, U, b_start, b_end)),
                    rtol=1e-06)


def to_variables(params):
    '''Converts a list of numpy arrays to a list of initialized variables.'''
    return map(lambda param: K.variable(param, dtype=param.dtype), params)


@keras_test
def test_add_boundary_energy():
    '''Checks that the start boundary potential b_start (resp. end boundary
    potential b_end) is placed on the first (resp. last) masked element,
    and that all un-masked elements are zero.'''

    n_classes = 2

    x_np = np.asarray([[[0, 1], [2, 3], [4, 5]],
                       [[6, 7], [8, 9], [10, 11]],
                       [[12, 13], [14, 15], [16, 17]]])
    mask_np = np.array([[1, 1, 0],
                        [0, 1, 1],
                        [0, 0, 1]])
    b_start_np = np.array([.1, .2])
    b_end_np = np.array([.3, .4])

    expected = np.asarray([[[.1, 1.2], [2.3, 3.4], [0, 0]],
                          [[0, 0], [8.1, 9.2], [10.3, 11.4]],
                          [[0, 0], [0, 0], [16.4, 17.6]]])

    x = K.placeholder(shape=(None, None, n_classes))
    mask = K.placeholder(shape=(None, None))
    b_start = K.placeholder(shape=(n_classes, ))
    b_end = K.placeholder(shape=(n_classes, ))

    energies = crf.add_boundary_energy(x, b_start, b_end, mask)
    fn = K.function([x, b_start, b_end, mask], [energies])

    actual = fn([x_np, b_start_np, b_end_np, mask_np])[0]
    assert_allclose(actual, expected, rtol=1e-05)


def np_energy_simple(x, y, U, b_start=None, b_end=None):
    '''Computes the energy for a given array of observations x, labels y,
       transition energy U, start energy b_start and end energy b_end.

       E(x, y) = Sum_t=1^n x_{t, y_t} + b_start_{y_1} + Sum_t=2^n U_{y_{t-1}, y_t}

       Supports no masking (hence simple).
    '''
    maxlen = x.shape[0]
    t = np.arange(maxlen)
    path_energy = x[t, y[t]].sum()
    start_boundary_energy = b_start[y[0]] if b_start is not None else 0.0
    end_boundary_energy = b_end[y[-1]] if b_end is not None else 0.0
    transition_energy = U[y[t[1:] - 1], y[t[1:]]].sum()
    return path_energy + start_boundary_energy + transition_energy + end_boundary_energy


def np_energy_single(x, y, U, b_start=None, b_end=None, mask=None):
    '''Computes the energy with support for masking.'''
    maxlen = x.shape[0]
    if mask is None:
        if b_start is not None:
            x = np.concatenate([x[:1] + b_start, x[1:]], axis=0)
        if b_end is not None:
            x = np.concatenate([x[:-1], x[-1:] + b_end], axis=0)
    else:
        if b_start is not None:
            mask_r = np.concatenate([np.zeros((1)), mask[:-1]], axis=0)
            start_mask = np.greater(mask, mask_r)
            x = x + np.expand_dims(start_mask, 1) * b_start
        if b_end is not None:
            mask_l = np.concatenate([mask[1:], np.zeros((1))], axis=0)
            end_mask = np.greater(mask, mask_l)
            x = x + np.expand_dims(end_mask, 1) * b_end
        x = x * np.expand_dims(mask, 1)
    if mask is None:
        t = np.arange(maxlen)
        path_energy = x[t, y[t]].sum()
        transition_energy = U[y[t[1:] - 1], y[t[1:]]].sum()
        return path_energy + transition_energy
    else:
        energy = 0
        energy += x[0, y[0]]
        for t in range(1, maxlen):
            energy += x[t, y[t]] + U[y[t - 1], y[t]] * mask[t - 1] * mask[t]
        return energy


def test_np_energy_single():
    '''Checks that np_energy_single gives the same result as np_energy_simple
    when restricting to the masked subsequences of x and y.'''
    maxlen, n_classes = 5, 3

    x = np.random.uniform(size=(maxlen, n_classes))
    y = np.random.randint(n_classes, size=(maxlen)).astype(np.int32)
    U = np.random.uniform(size=(n_classes, n_classes))
    b_start = np.random.uniform(size=(n_classes))
    b_end = np.random.uniform(size=(n_classes))

    assert_allclose(np_energy_single(x, y, U, b_start, b_end, mask=None),
                    np_energy_simple(x, y, U, b_start, b_end))

    assert_allclose(np_energy_single(x, y, U, b_start, b_end, mask=np.array([0, 0, 0, 0, 0])),
                    0.0)

    for start in range(maxlen):
        for end in range(start + 1, maxlen):
            mask = np.ones(maxlen)
            mask[:start] = 0
            mask[end:] = 0
            assert_allclose(np_energy_single(x, y, U, b_start, b_end, mask=mask),
                            np_energy_simple(x[start:end], y[start:end], U, b_start, b_end))


def np_energy(x, y, U, b_start=None, b_end=None, mask=None):
    '''Batched variant of np_energy_single.'''
    batch_size = x.shape[0]
    if mask is None:
        mask = [None] * batch_size
    return np.asarray([np_energy_single(x[k], y[k], U, b_start, b_end, mask[k]) for k in range(batch_size)])


def np_free_energy_single(x, U, b_start=None, b_end=None, mask=None):
    '''Computes the free energy (naively) by summing over all possible tag
    sequences.'''
    all_paths = all_paths_for(x)
    energies = np.asarray([np_energy_single(x, y, U, b_start, b_end, mask) for y in all_paths])
    logZ = np.log(np.sum(np.exp(energies), axis=0))
    return logZ


def np_free_energy(x, U, b_start=None, b_end=None, mask=None):
    '''Batched variant of np_free_energy_single.'''
    batch_size = x.shape[0]
    if mask is None:
        mask = [None] * batch_size
    return np.asarray([np_free_energy_single(x[k], U, b_start, b_end, mask[k]) for k in range(batch_size)])


def np_decode_naively_single(x, U, b_start=None, b_end=None, mask=None):
    '''Finds the best tag path y (naively), which is the one that has
    maximal energy.'''
    all_paths = all_paths_for(x)
    energies = np.asarray([np_energy_single(x, y, U, b_start, b_end, mask) for y in all_paths])
    best_path_index = np.argmax(energies, axis=0)
    return all_paths[best_path_index]


def np_decode_naively(x, U, b_start=None, b_end=None, mask=None):
    '''Batched variant of np_decode_naively_single.'''
    batch_size = x.shape[0]
    if mask is None:
        mask = [None] * batch_size
    return np.asarray([np_decode_naively_single(x[k], U, b_start, b_end, mask[k]) for k in range(batch_size)])


@keras_test
def test_free_energy():

    batch_size, maxlen, n_classes = 4, 5, 3

    x_np, _, _, U_np, b_start_np, b_end_np = create_data(batch_size, maxlen, n_classes)
    expected = np_free_energy(x_np, U_np, b_start_np, b_end_np)

    x, _, _, U, b_start, b_end = create_placeholders(maxlen, n_classes)
    energy = crf.free_energy(x, U, b_start, b_end)

    fn = K.function([x, U, b_start, b_end], [energy])

    actual = fn([x_np, U_np, b_start_np, b_end_np])[0]
    assert_allclose(actual, expected, rtol=1e-05)

    # Check that gradients are computable
    df = K.gradients(K.mean(energy, axis=0), [x, U, b_start, b_end])
    assert len(df) == 4


@keras_test
def test_masking_free_energy():
    batch_size, maxlen, n_classes = 4, 5, 3

    x_np, _, mask_np, U_np, b_start_np, b_end_np = create_data(batch_size, maxlen, n_classes)
    expected = np_free_energy(x_np, U_np, b_start_np, b_end_np, mask_np)

    x, _, mask, U, b_start, b_end = create_placeholders(maxlen, n_classes)
    energy = crf.free_energy(x, U, b_start, b_end, mask)

    fn = K.function([x, U, b_start, b_end, mask], [energy])

    actual = fn([x_np, U_np, b_start_np, b_end_np, mask_np])[0]
    assert_allclose(actual, expected, rtol=1e-05)

    # Check that gradients are computable
    df = K.gradients(K.mean(energy, axis=0), [x, U, b_start, b_end])
    assert len(df) == 4


@keras_test
def test_path_energy():
    batch_size, maxlen, n_classes = 4, 7, 3

    x_np, y_np, _, U_np, b_start_np, b_end_np = create_data(batch_size, maxlen, n_classes)
    expected = np_energy(x_np, y_np, U_np, b_start_np, b_end_np)

    x, y, _, U, b_start, b_end = create_placeholders(maxlen, n_classes)
    energy = crf.path_energy(y, x, U, b_start, b_end)

    fn = K.function([y, x, U, b_start, b_end], [energy])

    actual = fn([y_np, x_np, U_np, b_start_np, b_end_np])[0]

    assert_allclose(actual, expected, rtol=1e-05)

    # Check that gradients are computable
    df = K.gradients(K.mean(energy, axis=0), [x, U, b_start, b_end])
    assert len(df) == 4


@keras_test
def test_masking_path_energy():
    batch_size, maxlen, n_classes = 4, 7, 3

    x_np, y_np, mask_np, U_np, b_start_np, b_end_np = create_data(batch_size, maxlen, n_classes)
    expected = np_energy(x_np, y_np, U_np, b_start_np, b_end_np, mask_np)

    x, y, mask, U, b_start, b_end = create_placeholders(maxlen, n_classes)
    energy = crf.path_energy(y, x, U, b_start, b_end, mask)

    fn = K.function([y, x, U, b_start, b_end, mask], [energy])

    actual = fn([y_np, x_np, U_np, b_start_np, b_end_np, mask_np])[0]

    assert_allclose(actual, expected, rtol=1e-05)

    # Check that gradients are computable
    df = K.gradients(K.mean(energy, axis=0), [x, U, b_start, b_end])
    assert len(df) == 4


@keras_test
def test_viterbi_decode():

    batch_size, maxlen, n_classes = 4, 7, 3

    x_np, _, _, U_np, b_start_np, b_end_np = create_data(batch_size, maxlen, n_classes)

    expected = np_decode_naively(x_np, U_np, b_start_np, b_end_np)

    x, _, _, U, b_start, b_end = create_placeholders(maxlen, n_classes)
    output = crf.viterbi_decode(x, U, b_start, b_end)

    fn = K.function([x, U, b_start, b_end], [output])
    actual = fn([x_np, U_np, b_start_np, b_end_np])[0]

    assert_allclose(actual, expected, rtol=1e-05)


@keras_test
def test_masking_viterbi_decode():

    batch_size, maxlen, n_classes = 4, 7, 3

    x_np, _, mask_np, U_np, b_start_np, b_end_np = create_data(batch_size, maxlen, n_classes)

    expected = np_decode_naively(x_np, U_np, b_start_np, b_end_np, mask_np)

    x, _, mask, U, b_start, b_end = create_placeholders(maxlen, n_classes)
    output = crf.viterbi_decode(x, U, b_start, b_end, mask)

    fn = K.function([x, U, b_start, b_end, mask], [output])
    actual = fn([x_np, U_np, b_start_np, b_end_np, mask_np])[0]
    assert_allclose(actual * mask_np, expected * mask_np, rtol=1e-05)


def create_data(batch_size, maxlen, n_classes):
    x = np.random.uniform(size=(batch_size, maxlen, n_classes)).astype(K.floatx())
    y = np.random.randint(n_classes, size=(batch_size, maxlen)).astype(np.int32)
    mask = np.ones((batch_size, maxlen))
    mask[1, :2] = 0
    mask[2, -4:] = 0
    mask[3, :1] = 0
    mask[3, -1:] = 0
    U = np.random.uniform(size=(n_classes, n_classes)).astype(K.floatx())
    b_start = np.random.uniform(size=(n_classes)).astype(K.floatx())
    b_end = np.random.uniform(size=(n_classes)).astype(K.floatx())
    return x, y, mask, U, b_start, b_end


def create_placeholders(maxlen, n_classes):
    x = K.placeholder(shape=(None, maxlen, n_classes))
    y = K.placeholder(shape=(None, maxlen), dtype='int32')
    mask = K.placeholder(shape=(None, None))
    U = K.placeholder(shape=(n_classes, n_classes))
    b_start = K.placeholder(shape=(n_classes, ))
    b_end = K.placeholder(shape=(n_classes, ))
    return x, y, mask, U, b_start, b_end


def all_paths_for(x):
    "Returns all possible tag paths for a given input"
    maxlen = x.shape[0]
    n_classes = x.shape[1]
    all_paths = np.array(list(itertools.product(*([list(range(n_classes))] * maxlen))))
    return all_paths


@keras_test
def test_chain_crf_layer():
    from keras import regularizers
    from keras import constraints

    batch_size, maxlen, n_classes = 2, 5, 11

    layer_test(crf.ChainCRF,
               input_shape=(batch_size, maxlen, n_classes))

    layer_test(crf.ChainCRF,
               kwargs={'U_regularizer': regularizers.WeightRegularizer(l1=0.01),
                       'b_start_regularizer': regularizers.l1(0.01),
                       'b_end_regularizer': regularizers.l1(0.01),
                       'U_constraint': constraints.MaxNorm(1),
                       'b_start_constraint': constraints.MaxNorm(1),
                       'b_end_constraint': constraints.MaxNorm(1)},
               input_shape=(batch_size, maxlen, n_classes))


@keras_test
def test_chain_crf_dynamic_behaviour_with_dense_labels():
    batch_size, maxlen, n_classes = 2, 5, 11
    model = Sequential()
    layer = crf.ChainCRF(input_shape=(maxlen, n_classes))
    model.add(layer)
    model.compile(loss=layer.loss, optimizer='sgd')
    x = np.random.random((batch_size, maxlen, n_classes))
    y = np.random.randint(n_classes, size=(batch_size, maxlen))
    y = np.eye(n_classes)[y]
    model.train_on_batch(x, y)


@keras_test
def test_chain_crf_dynamic_behaviour_with_masking_and_dense_labels():
    vocab_size = 20
    batch_size, maxlen, n_classes = 2, 5, 11
    model = Sequential()
    model.add(Embedding(vocab_size, n_classes, input_length=maxlen, mask_zero=True))
    layer = crf.ChainCRF()
    model.add(layer)
    model.compile(loss=layer.loss, optimizer='sgd')
    x = np.random.randint(1, vocab_size, size=(batch_size, maxlen))
    x[0, -4:] = 0  # right padding
    x[1, -2:] = 0  # left padding
    y = np.random.randint(n_classes, size=(batch_size, maxlen))
    y = np.eye(n_classes)[y]
    model.train_on_batch(x, y)


@keras_test
def test_chain_crf_dynamic_behaviour_with_sparse_labels():
    batch_size, maxlen, n_classes = 2, 5, 11
    model = Sequential()
    layer = crf.ChainCRF(input_shape=(maxlen, n_classes))
    model.add(layer)
    model.compile(loss=layer.sparse_loss, optimizer='sgd')
    x = np.random.random((batch_size, maxlen, n_classes))
    y = np.random.randint(n_classes, size=(batch_size, maxlen))
    y = np.expand_dims(y, -1)
    model.train_on_batch(x, y)


@keras_test
def test_chain_crf_dynamic_behaviour_with_masking_and_sparse_labels():
    vocab_size = 20
    batch_size, maxlen, n_classes = 2, 5, 11
    model = Sequential()
    model.add(Embedding(vocab_size, n_classes, input_length=maxlen, mask_zero=True))
    layer = crf.ChainCRF()
    model.add(layer)
    model.compile(loss=layer.sparse_loss, optimizer='sgd')
    x = np.random.randint(1, vocab_size, size=(batch_size, maxlen))
    x[0, -4:] = 0  # right padding
    x[1, -2:] = 0  # left padding
    y = np.random.randint(n_classes, size=(batch_size, maxlen))
    y = np.expand_dims(y, -1)
    model.train_on_batch(x, y)


@keras_test
def test_temporal_weights_with_sparse_loss():
    batch_size, maxlen, n_classes = 2, 5, 11
    model = Sequential()
    layer = crf.ChainCRF(input_shape=(maxlen, n_classes))
    model.add(layer)
    model.compile(loss=layer.sparse_loss, optimizer='sgd', sample_weight_mode='temporal')
    x = np.random.random((batch_size, maxlen, n_classes))
    y = np.random.randint(n_classes, size=(batch_size, maxlen))
    y = np.expand_dims(y, -1)
    sample_weight = np.random.randint(0, 2, size=(batch_size, maxlen))
    # Check compilation (even temporal weight mode doesn't make sense)
    model.train_on_batch(x, y, sample_weight=sample_weight)


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
    assert fn_loss_train.shape == (batch_size, 1)
    fn_loss_test = fn([y_np, x_np, 0])[0]
    assert fn_loss_test.shape == (batch_size, 1)


@keras_test
def test_chain_crf_sparse_loss():
    batch_size, maxlen, n_classes = 2, 5, 11
    batch_input_shape = (None, maxlen, n_classes)
    layer = crf.ChainCRF(batch_input_shape=batch_input_shape)
    layer.build(batch_input_shape)
    y = K.placeholder(shape=(None, maxlen, 1))
    x = K.placeholder(shape=(None, maxlen, n_classes))
    fn = K.function([y, x, K.learning_phase()], [layer.sparse_loss(y, x)])

    x_np = np.random.random((batch_size, maxlen, n_classes))
    y_np = np.random.randint(n_classes, size=(batch_size, maxlen))
    y_np = np.expand_dims(y_np, 2)

    fn_loss_train = fn([y_np, x_np, 1])[0]
    assert fn_loss_train.shape == (batch_size, 1)
    fn_loss_test = fn([y_np, x_np, 0])[0]
    assert fn_loss_test.shape == (batch_size, 1)


@keras_test
def test_persistence():
    import tempfile
    import os
    from keras.models import load_model

    batch_size, maxlen, n_classes = 10000, 5, 3
    model = Sequential()
    layer = crf.ChainCRF(input_shape=(maxlen, n_classes))
    model.add(layer)

    model.compile(loss=layer.loss, optimizer='sgd', metrics=['accuracy'])

    # Save model to temp file
    folder = tempfile.mkdtemp()
    filename = os.path.join(folder, 'model.h5')
    model.save(filename)
    del model

    # Load model using custom objects
    custom_objects = crf.create_custom_objects()
    model = load_model(filename, custom_objects=custom_objects)

    # Remove temp model
    os.remove(filename)
    os.rmdir(folder)

    x = np.random.random((batch_size, maxlen, n_classes))
    y = np.random.randint(n_classes, size=(batch_size, maxlen))
    y = np.eye(n_classes)[y]
    history = model.train_on_batch(x, y)

    assert history[0] >= 0


if __name__ == '__main__':
    pytest.main([__file__])
