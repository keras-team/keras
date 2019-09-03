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


class TestTruePositives(object):

    def test_config(self):
        tp_obj = metrics.TruePositives(name='my_tp', thresholds=[0.4, 0.9])
        assert tp_obj.name == 'my_tp'
        assert len(tp_obj.weights) == 1
        assert tp_obj.thresholds == [0.4, 0.9]

        # Check save and restore config
        tp_obj2 = metrics.TruePositives.from_config(tp_obj.get_config())
        assert tp_obj2.name == 'my_tp'
        assert len(tp_obj2.weights) == 1
        assert tp_obj2.thresholds == [0.4, 0.9]

    def test_unweighted(self):
        tp_obj = metrics.TruePositives()

        y_true = ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                  (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        y_pred = ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                  (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))

        result = tp_obj(y_true, y_pred)
        assert np.allclose(7., K.eval(result))

    def test_weighted(self):
        tp_obj = metrics.TruePositives()
        y_true = ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                  (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        y_pred = ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                  (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        sample_weight = (1., 1.5, 2., 2.5)
        result = tp_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.allclose(12., K.eval(result))

    def test_unweighted_with_thresholds(self):
        tp_obj = metrics.TruePositives(thresholds=[0.15, 0.5, 0.85])

        y_pred = ((0.9, 0.2, 0.8, 0.1), (0.2, 0.9, 0.7, 0.6),
                  (0.1, 0.2, 0.4, 0.3), (0, 1, 0.7, 0.3))
        y_true = ((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0),
                  (1, 1, 1, 1))

        result = tp_obj(y_true, y_pred)
        assert np.allclose([6., 3., 1.], K.eval(result))

    def test_weighted_with_thresholds(self):
        tp_obj = metrics.TruePositives(thresholds=[0.15, 0.5, 0.85])

        y_pred = ((0.9, 0.2, 0.8, 0.1), (0.2, 0.9, 0.7, 0.6),
                  (0.1, 0.2, 0.4, 0.3), (0, 1, 0.7, 0.3))
        y_true = ((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0),
                  (1, 1, 1, 1))

        result = tp_obj(y_true, y_pred, sample_weight=37.)
        assert np.allclose([222., 111., 37.], K.eval(result))


class TestTrueNegatives(object):

    def test_config(self):
        tn_obj = metrics.TrueNegatives(name='my_tn', thresholds=[0.4, 0.9])
        assert tn_obj.name == 'my_tn'
        assert len(tn_obj.weights) == 1
        assert tn_obj.thresholds == [0.4, 0.9]

        # Check save and restore config
        tn_obj2 = metrics.TrueNegatives.from_config(tn_obj.get_config())
        assert tn_obj2.name == 'my_tn'
        assert len(tn_obj2.weights) == 1
        assert tn_obj2.thresholds == [0.4, 0.9]

    def test_unweighted(self):
        tn_obj = metrics.TrueNegatives()

        y_true = ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                  (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        y_pred = ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                  (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))

        result = tn_obj(y_true, y_pred)
        assert np.allclose(3., K.eval(result))

    def test_weighted(self):
        tn_obj = metrics.TrueNegatives()
        y_true = ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                  (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        y_pred = ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                  (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        sample_weight = (1., 1.5, 2., 2.5)
        result = tn_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.allclose(4., K.eval(result))

    def test_unweighted_with_thresholds(self):
        tn_obj = metrics.TrueNegatives(thresholds=[0.15, 0.5, 0.85])

        y_pred = ((0.9, 0.2, 0.8, 0.1), (0.2, 0.9, 0.7, 0.6),
                  (0.1, 0.2, 0.4, 0.3), (0, 1, 0.7, 0.3))
        y_true = ((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1))

        result = tn_obj(y_true, y_pred)
        assert np.allclose([2., 5., 7.], K.eval(result))

    def test_weighted_with_thresholds(self):
        tn_obj = metrics.TrueNegatives(thresholds=[0.15, 0.5, 0.85])

        y_pred = ((0.9, 0.2, 0.8, 0.1), (0.2, 0.9, 0.7, 0.6),
                  (0.1, 0.2, 0.4, 0.3), (0, 1, 0.7, 0.3))
        y_true = ((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1))
        sample_weight = ((0.0, 2.0, 3.0, 5.0),)

        result = tn_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.allclose([5., 15., 23.], K.eval(result))


class TestFalseNegatives(object):

    def test_config(self):
        fn_obj = metrics.FalseNegatives(name='my_fn', thresholds=[0.4, 0.9])
        assert fn_obj.name == 'my_fn'
        assert len(fn_obj.weights) == 1
        assert fn_obj.thresholds == [0.4, 0.9]

        # Check save and restore config
        fn_obj2 = metrics.FalseNegatives.from_config(fn_obj.get_config())
        assert fn_obj2.name == 'my_fn'
        assert len(fn_obj2.weights) == 1
        assert fn_obj2.thresholds == [0.4, 0.9]

    def test_unweighted(self):
        fn_obj = metrics.FalseNegatives()

        y_true = ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                  (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        y_pred = ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                  (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))

        result = fn_obj(y_true, y_pred)
        assert np.allclose(3., K.eval(result))

    def test_weighted(self):
        fn_obj = metrics.FalseNegatives()
        y_true = ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                  (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        y_pred = ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                  (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        sample_weight = (1., 1.5, 2., 2.5)
        result = fn_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.allclose(5., K.eval(result))

    def test_unweighted_with_thresholds(self):
        fn_obj = metrics.FalseNegatives(thresholds=[0.15, 0.5, 0.85])

        y_pred = ((0.9, 0.2, 0.8, 0.1), (0.2, 0.9, 0.7, 0.6),
                  (0.1, 0.2, 0.4, 0.3), (0, 1, 0.7, 0.3))
        y_true = ((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1))

        result = fn_obj(y_true, y_pred)
        assert np.allclose([1., 4., 6.], K.eval(result))

    def test_weighted_with_thresholds(self):
        fn_obj = metrics.FalseNegatives(thresholds=[0.15, 0.5, 0.85])

        y_pred = ((0.9, 0.2, 0.8, 0.1), (0.2, 0.9, 0.7, 0.6),
                  (0.1, 0.2, 0.4, 0.3), (0, 1, 0.7, 0.3))
        y_true = ((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1))
        sample_weight = ((3.0,), (5.0,), (7.0,), (4.0,))

        result = fn_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.allclose([4., 16., 23.], K.eval(result))
