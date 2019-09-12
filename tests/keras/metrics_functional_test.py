import pytest
import numpy as np
from numpy.testing import assert_allclose
from flaky import flaky

import keras
from keras import metrics
from keras import backend as K

all_metrics = [
    metrics.binary_accuracy,
    metrics.categorical_accuracy,
    metrics.mean_squared_error,
    metrics.mean_absolute_error,
    metrics.mean_absolute_percentage_error,
    metrics.mean_squared_logarithmic_error,
    metrics.squared_hinge,
    metrics.hinge,
    metrics.categorical_crossentropy,
    metrics.binary_crossentropy,
    metrics.poisson,
    metrics.cosine_proximity,
    metrics.logcosh,
]

all_sparse_metrics = [
    metrics.sparse_categorical_accuracy,
    metrics.sparse_categorical_crossentropy,
]


@pytest.mark.parametrize('metric', all_metrics)
def test_metrics(metric):
    y_a = K.variable(np.random.random((6, 7)))
    y_b = K.variable(np.random.random((6, 7)))
    output = metric(y_a, y_b)
    assert K.eval(output).shape == (6,)


@pytest.mark.parametrize('metric', all_sparse_metrics)
def test_sparse_metrics(metric):
    y_a = K.variable(np.random.randint(0, 7, (6,)), dtype=K.floatx())
    y_b = K.variable(np.random.random((6, 7)), dtype=K.floatx())
    assert K.eval(metric(y_a, y_b)).shape == (6,)


@pytest.mark.parametrize('shape', [(6,), (6, 3), (6, 3, 1)])
def test_sparse_categorical_accuracy_correctness(shape):
    y_a = K.variable(np.random.randint(0, 7, shape), dtype=K.floatx())
    y_b_shape = shape + (7,)
    y_b = K.variable(np.random.random(y_b_shape), dtype=K.floatx())
    # use one_hot embedding to convert sparse labels to equivalent dense labels
    y_a_dense_labels = K.cast(K.one_hot(K.cast(y_a, dtype='int32'), 7),
                              dtype=K.floatx())
    sparse_categorical_acc = metrics.sparse_categorical_accuracy(y_a, y_b)
    categorical_acc = metrics.categorical_accuracy(y_a_dense_labels, y_b)
    assert np.allclose(K.eval(sparse_categorical_acc), K.eval(categorical_acc))


def test_serialize():
    '''This is a mock 'round trip' of serialize and deserialize.
    '''

    class MockMetric:
        def __init__(self):
            self.__name__ = "mock_metric"

    mock = MockMetric()
    found = metrics.serialize(mock)
    assert found == "mock_metric"

    found = metrics.deserialize('mock_metric',
                                custom_objects={'mock_metric': True})
    assert found is True


def test_invalid_get():

    with pytest.raises(ValueError):
        metrics.get(5)


@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason='CNTK backend does not support top_k yet')
def test_top_k_categorical_accuracy():
    y_pred = K.variable(np.array([[0.3, 0.2, 0.1], [0.1, 0.2, 0.7]]))
    y_true = K.variable(np.array([[0, 1, 0], [1, 0, 0]]))
    success_result = K.eval(metrics.top_k_categorical_accuracy(y_true, y_pred,
                                                               k=3))
    assert np.mean(success_result) == 1
    partial_result = K.eval(metrics.top_k_categorical_accuracy(y_true, y_pred,
                                                               k=2))
    assert np.mean(partial_result) == 0.5
    failure_result = K.eval(metrics.top_k_categorical_accuracy(y_true, y_pred,
                                                               k=1))
    assert np.mean(failure_result) == 0


@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason='CNTK backend does not support top_k yet')
@pytest.mark.parametrize('y_pred, y_true', [
    # Test correctness if the shape of y_true is (num_samples, 1)
    (np.array([[0.3, 0.2, 0.1], [0.1, 0.2, 0.7]]), np.array([[1], [0]])),
    # Test correctness if the shape of y_true is (num_samples,)
    (np.array([[0.3, 0.2, 0.1], [0.1, 0.2, 0.7]]), np.array([1, 0])),
])
def test_sparse_top_k_categorical_accuracy(y_pred, y_true):
    y_pred = K.variable(y_pred)
    y_true = K.variable(y_true)
    success_result = K.eval(
        metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=3))

    assert np.mean(success_result) == 1
    partial_result = K.eval(
        metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=2))

    assert np.mean(partial_result) == 0.5
    failure_result = K.eval(
        metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=1))

    assert np.mean(failure_result) == 0


if __name__ == '__main__':
    pytest.main([__file__])
