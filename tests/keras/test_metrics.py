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


def test_fbetascore():
    y_true = K.variable(np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0]))
    y_pred = K.variable(np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]))

    # Calculated using sklearn.metrics.f1_score
    expected = 0.33333333333333331

    actual = K.eval(metrics.fbetascore(y_true, y_pred))
    epsilon = 1e-05
    assert expected - epsilon <= actual <= expected + epsilon


def test_sparse_metrics():
    for metric in all_sparse_metrics:
        y_a = K.variable(np.random.randint(0, 7, (6,)), dtype=K.floatx())
        y_b = K.variable(np.random.random((6, 7)), dtype=K.floatx())
        assert K.eval(metric(y_a, y_b)).shape == ()


if __name__ == "__main__":
    pytest.main([__file__])
