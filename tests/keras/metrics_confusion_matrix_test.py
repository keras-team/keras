"""Tests for Keras confusion matrix metrics classes."""
import pytest
import numpy as np

from keras import metrics
from keras import backend as K

if K.backend() != 'tensorflow':
    # Need TensorFlow to use metric.__call__
    pytestmark = pytest.mark.skip


class TestFalsePositives(object):

    def test_config(self):
        fp_obj = metrics.FalsePositives(name='my_fp', thresholds=[0.4, 0.9])
        assert fp_obj.name == 'my_fp'
        assert len(fp_obj.weights) == 1
        assert fp_obj.thresholds == [0.4, 0.9]

        # Check save and restore config
        fp_obj2 = metrics.FalsePositives.from_config(fp_obj.get_config())
        assert fp_obj2.name == 'my_fp'
        assert len(fp_obj2.weights) == 1
        assert fp_obj2.thresholds == [.4, 0.9]

    def test_unweighted(self):
        fp_obj = metrics.FalsePositives()

        y_true = ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        y_pred = ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))

        result = fp_obj(y_true, y_pred)
        assert np.allclose(7., K.eval(result))

    def test_weighted(self):
        fp_obj = metrics.FalsePositives()
        y_true = ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        y_pred = ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        sample_weight = (1., 1.5, 2., 2.5)
        result = fp_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.allclose(14., K.eval(result))

    def test_unweighted_with_thresholds(self):
        fp_obj = metrics.FalsePositives(thresholds=[0.15, 0.5, 0.85])

        y_pred = ((0.9, 0.2, 0.8, 0.1), (0.2, 0.9, 0.7, 0.6),
                  (0.1, 0.2, 0.4, 0.3), (0, 1, 0.7, 0.3))
        y_true = ((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1))

        result = fp_obj(y_true, y_pred)
        assert np.allclose([7., 4., 2.], K.eval(result))

    def test_weighted_with_thresholds(self):
        fp_obj = metrics.FalsePositives(thresholds=[0.15, 0.5, 0.85])

        y_pred = ((0.9, 0.2, 0.8, 0.1), (0.2, 0.9, 0.7, 0.6),
                  (0.1, 0.2, 0.4, 0.3), (0, 1, 0.7, 0.3))
        y_true = ((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1))
        sample_weight = ((1.0, 2.0, 3.0, 5.0), (7.0, 11.0, 13.0, 17.0),
                         (19.0, 23.0, 29.0, 31.0), (5.0, 15.0, 10.0, 0))

        result = fp_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.allclose([125., 42., 12.], K.eval(result))

    def test_threshold_limit(self):
        with pytest.raises(Exception):
            metrics.FalsePositives(thresholds=[-1, 0.5, 2])

        with pytest.raises(Exception):
            metrics.FalsePositives(thresholds=[None])
