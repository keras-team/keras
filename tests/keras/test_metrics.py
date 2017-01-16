import pytest
import numpy as np

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
    metrics.matthews_correlation,
]

all_sparse_metrics = [
    metrics.sparse_categorical_accuracy,
    metrics.sparse_categorical_crossentropy,
]


def test_metrics():
    y_a = K.variable(np.random.random((6, 7)))
    y_b = K.variable(np.random.random((6, 7)))
    for metric in all_metrics:
        output = metric(y_a, y_b)
        assert K.eval(output).shape == ()


def test_matthews_correlation():
    y_true = K.variable(np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0]))
    y_pred = K.variable(np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]))

    # Calculated using sklearn.metrics.matthews_corrcoef
    expected = -0.14907119849998601

    actual = K.eval(metrics.matthews_correlation(y_true, y_pred))
    epsilon = 1e-05
    assert expected - epsilon <= actual <= expected + epsilon


def test_precision():
    y_true = K.variable(np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0]))
    y_pred = K.variable(np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]))

    # Calculated using sklearn.metrics.precision_score
    expected = 0.40000000000000002

    actual = K.eval(metrics.precision(y_true, y_pred))
    epsilon = 1e-05
    assert expected - epsilon <= actual <= expected + epsilon


def test_recall():
    y_true = K.variable(np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0]))
    y_pred = K.variable(np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]))

    # Calculated using sklearn.metrics.recall_score
    expected = 0.2857142857142857

    actual = K.eval(metrics.recall(y_true, y_pred))
    epsilon = 1e-05
    assert expected - epsilon <= actual <= expected + epsilon


def test_fbeta_score():
    y_true = K.variable(np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0]))
    y_pred = K.variable(np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]))

    # Calculated using sklearn.metrics.fbeta_score
    expected = 0.30303030303030304

    actual = K.eval(metrics.fbeta_score(y_true, y_pred, beta=2))
    epsilon = 1e-05
    assert expected - epsilon <= actual <= expected + epsilon


def test_fmeasure():
    y_true = K.variable(np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0]))
    y_pred = K.variable(np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]))

    # Calculated using sklearn.metrics.f1_score
    expected = 0.33333333333333331

    actual = K.eval(metrics.fmeasure(y_true, y_pred))
    epsilon = 1e-05
    assert expected - epsilon <= actual <= expected + epsilon


def test_sparse_metrics():
    for metric in all_sparse_metrics:
        y_a = K.variable(np.random.randint(0, 7, (6,)), dtype=K.floatx())
        y_b = K.variable(np.random.random((6, 7)), dtype=K.floatx())
        assert K.eval(metric(y_a, y_b)).shape == ()


def test_top_k_categorical_accuracy():
    y_pred = K.variable(np.array([[0.3, 0.2, 0.1], [0.1, 0.2, 0.7]]))
    y_true = K.variable(np.array([[0, 1, 0], [1, 0, 0]]))
    success_result = K.eval(metrics.top_k_categorical_accuracy(y_true, y_pred,
                                                               k=3))
    assert success_result == 1
    partial_result = K.eval(metrics.top_k_categorical_accuracy(y_true, y_pred,
                                                               k=2))
    assert partial_result == 0.5
    failure_result = K.eval(metrics.top_k_categorical_accuracy(y_true, y_pred,
                                                               k=1))
    assert failure_result == 0


if __name__ == "__main__":
    pytest.main([__file__])
