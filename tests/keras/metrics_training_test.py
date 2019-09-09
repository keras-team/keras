"""Tests for metric objects training and evaluation."""
import pytest
import numpy as np

from keras import metrics
from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential


class TestMetricsTrainingEvaluation(object):

    def test_training_and_eval(self):
        model = Sequential([Dense(2, input_shape=(3,))])
        metrics_list = [
            metrics.Accuracy(),
            metrics.MeanSquaredError(),
            metrics.Hinge(),
            metrics.CategoricalHinge(),
            metrics.SquaredHinge(),
            metrics.FalsePositives(),
            metrics.TruePositives(),
            metrics.FalseNegatives(),
            metrics.TrueNegatives(),
            metrics.BinaryAccuracy(),
            metrics.CategoricalAccuracy(),
            metrics.TopKCategoricalAccuracy(),
            metrics.LogCoshError(),
            metrics.Poisson(),
            metrics.KLDivergence(),
            metrics.CosineSimilarity(),
            metrics.MeanAbsoluteError(),
            metrics.MeanAbsolutePercentageError(),
            metrics.MeanSquaredError(),
            metrics.MeanSquaredLogarithmicError(),
            metrics.RootMeanSquaredError(),
            metrics.BinaryCrossentropy(),
            metrics.CategoricalCrossentropy(),
            metrics.SensitivityAtSpecificity(0.5),
            metrics.SpecificityAtSensitivity(0.5),
            metrics.Precision(),
            metrics.Recall(),
            metrics.AUC()]
        model.compile('rmsprop', 'mse', metrics=metrics_list)
        x = np.random.random((10, 3))
        y = np.random.random((10, 2))
        model.fit(x, y)
        model.evaluate(x, y)

    def test_sparse_metrics(self):
        model = Sequential([Dense(1, input_shape=(3,))])
        metrics_list = [
            metrics.SparseCategoricalAccuracy(),
            metrics.SparseTopKCategoricalAccuracy(),
            metrics.SparseCategoricalCrossentropy()]
        model.compile('rmsprop', 'mse', metrics=metrics_list)
        x = np.random.random((10, 3))
        y = np.random.random((10,))
        model.fit(x, y)
        model.evaluate(x, y)

    @pytest.mark.skipif(K.backend() != 'tensorflow', reason="requires tensorflow")
    def test_mean_iou(self):
        model = Sequential([Dense(1, input_shape=(3,))])
        model.compile('rmsprop', 'mse', metrics=[metrics.MeanIoU(2)])
        x = np.random.random((10, 3))
        y = np.random.random((10,))
        model.fit(x, y)
        model.evaluate(x, y)
