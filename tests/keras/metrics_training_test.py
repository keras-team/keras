"""Tests for metric objects training and evaluation."""
import pytest
import numpy as np

from keras import metrics
from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential


if K.backend() == 'cntk':
    pytestmark = pytest.mark.skip


METRICS = [
    metrics.Accuracy,
    metrics.MeanSquaredError,
    metrics.Hinge,
    metrics.CategoricalHinge,
    metrics.SquaredHinge,
    metrics.FalsePositives,
    metrics.TruePositives,
    metrics.FalseNegatives,
    metrics.TrueNegatives,
    metrics.BinaryAccuracy,
    metrics.CategoricalAccuracy,
    metrics.TopKCategoricalAccuracy,
    metrics.LogCoshError,
    metrics.Poisson,
    metrics.KLDivergence,
    metrics.CosineSimilarity,
    metrics.MeanAbsoluteError,
    metrics.MeanAbsolutePercentageError,
    metrics.MeanSquaredError,
    metrics.MeanSquaredLogarithmicError,
    metrics.RootMeanSquaredError,
    metrics.BinaryCrossentropy,
    metrics.CategoricalCrossentropy,
    metrics.Precision,
    metrics.Recall,
    metrics.AUC,
]
SPARSE_METRICS = [
    metrics.SparseCategoricalAccuracy,
    metrics.SparseTopKCategoricalAccuracy,
    metrics.SparseCategoricalCrossentropy
]


@pytest.mark.parametrize('metric_cls', METRICS)
def test_training_and_eval(metric_cls):
    model = Sequential([Dense(2, input_shape=(3,))])
    model.compile('rmsprop', 'mse', metrics=[metric_cls()])
    x = np.random.random((10, 3))
    y = np.random.random((10, 2))
    model.fit(x, y)
    model.evaluate(x, y)


@pytest.mark.parametrize('metric_cls', SPARSE_METRICS)
def test_sparse_metrics(metric_cls):
    model = Sequential([Dense(1, input_shape=(3,))])
    model.compile('rmsprop', 'mse', metrics=[metric_cls()])
    x = np.random.random((10, 3))
    y = np.random.random((10,))
    model.fit(x, y)
    model.evaluate(x, y)


def test_sensitivity_metrics():
    metrics_list = [
        metrics.SensitivityAtSpecificity(0.5),
        metrics.SpecificityAtSensitivity(0.5),
    ]
    model = Sequential([Dense(2, input_shape=(3,))])
    model.compile('rmsprop', 'mse', metrics=metrics_list)
    x = np.random.random((10, 3))
    y = np.random.random((10, 2))
    model.fit(x, y)
    model.evaluate(x, y)


@pytest.mark.skipif(True, reason='It is a flaky test, see #13477 for more context.')
def test_mean_iou():
    import tensorflow as tf
    if not tf.__version__.startswith('2.'):
        return

    model = Sequential([Dense(1, input_shape=(3,))])
    model.compile('rmsprop', 'mse', metrics=[metrics.MeanIoU(2)])
    x = np.random.random((10, 3))
    y = np.random.random((10,))
    model.fit(x, y)
    model.evaluate(x, y)
