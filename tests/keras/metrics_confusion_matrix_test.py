"""Tests for Keras confusion matrix metrics classes."""
import pytest
import numpy as np

from keras import metrics
from keras import backend as K
from keras.utils import metrics_utils

if K.backend() != 'tensorflow':
    # Need TensorFlow to use metric.__call__
    pytestmark = pytest.mark.skip

import tensorflow as tf


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


class TestSensitivityAtSpecificity(object):

    def test_config(self):
        s_obj = metrics.SensitivityAtSpecificity(
            0.4, num_thresholds=100, name='sensitivity_at_specificity_1')
        assert s_obj.name == 'sensitivity_at_specificity_1'
        assert len(s_obj.weights) == 4
        assert s_obj.specificity == 0.4
        assert s_obj.num_thresholds == 100

        # Check save and restore config
        s_obj2 = metrics.SensitivityAtSpecificity.from_config(s_obj.get_config())
        assert s_obj2.name == 'sensitivity_at_specificity_1'
        assert len(s_obj2.weights) == 4
        assert s_obj2.specificity == 0.4
        assert s_obj2.num_thresholds == 100

    def test_unweighted_all_correct(self):
        s_obj = metrics.SensitivityAtSpecificity(0.7, num_thresholds=1)
        inputs = np.random.randint(0, 2, size=(100, 1))
        y_pred = K.constant(inputs, dtype='float32')
        y_true = K.constant(inputs)
        result = s_obj(y_true, y_pred)
        assert np.isclose(1, K.eval(result))

    def test_unweighted_high_specificity(self):
        s_obj = metrics.SensitivityAtSpecificity(0.8)
        pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.1, 0.45, 0.5, 0.8, 0.9]
        label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        y_pred = K.constant(pred_values, dtype='float32')
        y_true = K.constant(label_values)
        result = s_obj(y_true, y_pred)
        assert np.isclose(0.8, K.eval(result))

    def test_unweighted_low_specificity(self):
        s_obj = metrics.SensitivityAtSpecificity(0.4)
        pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.25, 0.26, 0.26]
        label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        y_pred = K.constant(pred_values, dtype='float32')
        y_true = K.constant(label_values)
        result = s_obj(y_true, y_pred)
        assert np.isclose(0.6, K.eval(result))

    def test_weighted(self):
        s_obj = metrics.SensitivityAtSpecificity(0.4)
        pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.25, 0.26, 0.26]
        label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        weight_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        y_pred = K.constant(pred_values, dtype='float32')
        y_true = K.constant(label_values, dtype='float32')
        weights = K.constant(weight_values)
        result = s_obj(y_true, y_pred, sample_weight=weights)
        assert np.isclose(0.675, K.eval(result))

    def test_invalid_specificity(self):
        with pytest.raises(Exception):
            metrics.SensitivityAtSpecificity(-1)

    def test_invalid_num_thresholds(self):
        with pytest.raises(Exception):
            metrics.SensitivityAtSpecificity(0.4, num_thresholds=-1)


class TestSpecificityAtSensitivity(object):

    def test_config(self):
        s_obj = metrics.SpecificityAtSensitivity(
            0.4, num_thresholds=100, name='specificity_at_sensitivity_1')
        assert s_obj.name == 'specificity_at_sensitivity_1'
        assert len(s_obj.weights) == 4
        assert s_obj.sensitivity == 0.4
        assert s_obj.num_thresholds == 100

        # Check save and restore config
        s_obj2 = metrics.SpecificityAtSensitivity.from_config(s_obj.get_config())
        assert s_obj2.name == 'specificity_at_sensitivity_1'
        assert len(s_obj2.weights) == 4
        assert s_obj2.sensitivity == 0.4
        assert s_obj2.num_thresholds == 100

    def test_unweighted_all_correct(self):
        s_obj = metrics.SpecificityAtSensitivity(0.7, num_thresholds=1)
        inputs = np.random.randint(0, 2, size=(100, 1))
        y_pred = K.constant(inputs, dtype='float32')
        y_true = K.constant(inputs)
        result = s_obj(y_true, y_pred)
        assert np.isclose(1, K.eval(result))

    def test_unweighted_high_sensitivity(self):
        s_obj = metrics.SpecificityAtSensitivity(0.8)
        pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.1, 0.45, 0.5, 0.8, 0.9]
        label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        y_pred = K.constant(pred_values, dtype='float32')
        y_true = K.constant(label_values)
        result = s_obj(y_true, y_pred)
        assert np.isclose(0.4, K.eval(result))

    def test_unweighted_low_sensitivity(self):
        s_obj = metrics.SpecificityAtSensitivity(0.4)
        pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.25, 0.26, 0.26]
        label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        y_pred = K.constant(pred_values, dtype='float32')
        y_true = K.constant(label_values)
        result = s_obj(y_true, y_pred)
        assert np.isclose(0.6, K.eval(result))

    def test_weighted(self):
        s_obj = metrics.SpecificityAtSensitivity(0.4)
        pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.25, 0.26, 0.26]
        label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        weight_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        y_pred = K.constant(pred_values, dtype='float32')
        y_true = K.constant(label_values, dtype='float32')
        weights = K.constant(weight_values)
        result = s_obj(y_true, y_pred, sample_weight=weights)
        assert np.isclose(0.4, K.eval(result))

    def test_invalid_sensitivity(self):
        with pytest.raises(Exception):
            metrics.SpecificityAtSensitivity(-1)

    def test_invalid_num_thresholds(self):
        with pytest.raises(Exception):
            metrics.SpecificityAtSensitivity(0.4, num_thresholds=-1)


class TestAUC(object):

    def setup(self):
        self.num_thresholds = 3
        self.y_pred = K.constant([0, 0.5, 0.3, 0.9], dtype='float32')
        self.y_true = K.constant([0, 0, 1, 1])
        self.sample_weight = [1, 2, 3, 4]

        # threshold values are [0 - 1e-7, 0.5, 1 + 1e-7]
        # y_pred when threshold = 0 - 1e-7  : [1, 1, 1, 1]
        # y_pred when threshold = 0.5       : [0, 0, 0, 1]
        # y_pred when threshold = 1 + 1e-7  : [0, 0, 0, 0]

        # without sample_weight:
        # tp = np.sum([[0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]], axis=1)
        # fp = np.sum([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], axis=1)
        # fn = np.sum([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]], axis=1)
        # tn = np.sum([[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]], axis=1)

        # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]

        # with sample_weight:
        # tp = np.sum([[0, 0, 3, 4], [0, 0, 0, 4], [0, 0, 0, 0]], axis=1)
        # fp = np.sum([[1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], axis=1)
        # fn = np.sum([[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 3, 4]], axis=1)
        # tn = np.sum([[0, 0, 0, 0], [1, 2, 0, 0], [1, 2, 0, 0]], axis=1)

        # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]

    def test_config(self):
        auc_obj = metrics.AUC(
            num_thresholds=100,
            curve='PR',
            summation_method='majoring',
            name='auc_1')
        assert auc_obj.name == 'auc_1'
        assert len(auc_obj.weights) == 4
        assert auc_obj.num_thresholds == 100
        assert auc_obj.curve == metrics_utils.AUCCurve.PR
        assert auc_obj.summation_method == metrics_utils.AUCSummationMethod.MAJORING

        # Check save and restore config.
        auc_obj2 = metrics.AUC.from_config(auc_obj.get_config())
        assert auc_obj2.name == 'auc_1'
        assert len(auc_obj2.weights) == 4
        assert auc_obj2.num_thresholds == 100
        assert auc_obj2.curve == metrics_utils.AUCCurve.PR
        assert auc_obj2.summation_method == metrics_utils.AUCSummationMethod.MAJORING

    def test_config_manual_thresholds(self):
        auc_obj = metrics.AUC(
            num_thresholds=None,
            curve='PR',
            summation_method='majoring',
            name='auc_1',
            thresholds=[0.3, 0.5])
        assert auc_obj.name == 'auc_1'
        assert len(auc_obj.weights) == 4
        assert auc_obj.num_thresholds == 4
        assert np.allclose(auc_obj.thresholds, [0.0, 0.3, 0.5, 1.0], atol=1e-3)
        assert auc_obj.curve == metrics_utils.AUCCurve.PR
        assert auc_obj.summation_method == metrics_utils.AUCSummationMethod.MAJORING

        # Check save and restore config.
        auc_obj2 = metrics.AUC.from_config(auc_obj.get_config())
        assert auc_obj2.name == 'auc_1'
        assert len(auc_obj2.weights) == 4
        assert auc_obj2.num_thresholds == 4
        assert auc_obj2.curve == metrics_utils.AUCCurve.PR
        assert auc_obj2.summation_method == metrics_utils.AUCSummationMethod.MAJORING

    def test_unweighted_all_correct(self):
        self.setup()
        auc_obj = metrics.AUC()
        result = auc_obj(self.y_true, self.y_true)
        assert K.eval(result) == 1

    def test_unweighted(self):
        self.setup()
        auc_obj = metrics.AUC(num_thresholds=self.num_thresholds)
        result = auc_obj(self.y_true, self.y_pred)

        # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
        # recall = [2/2, 1/(1+1), 0] = [1, 0.5, 0]
        # fp_rate = [2/2, 0, 0] = [1, 0, 0]
        # heights = [(1 + 0.5)/2, (0.5 + 0)/2] = [0.75, 0.25]
        # widths = [(1 - 0), (0 - 0)] = [1, 0]
        expected_result = (0.75 * 1 + 0.25 * 0)
        assert np.allclose(K.eval(result), expected_result, atol=1e-3)

    def test_manual_thresholds(self):
        self.setup()
        # Verify that when specified, thresholds are used instead of num_thresholds.
        auc_obj = metrics.AUC(num_thresholds=2, thresholds=[0.5])
        assert auc_obj.num_thresholds == 3
        assert np.allclose(auc_obj.thresholds, [0.0, 0.5, 1.0], atol=1e-3)
        result = auc_obj(self.y_true, self.y_pred)

        # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
        # recall = [2/2, 1/(1+1), 0] = [1, 0.5, 0]
        # fp_rate = [2/2, 0, 0] = [1, 0, 0]
        # heights = [(1 + 0.5)/2, (0.5 + 0)/2] = [0.75, 0.25]
        # widths = [(1 - 0), (0 - 0)] = [1, 0]
        expected_result = (0.75 * 1 + 0.25 * 0)
        assert np.allclose(K.eval(result), expected_result, atol=1e-3)

    def test_weighted_roc_interpolation(self):
        self.setup()
        auc_obj = metrics.AUC(num_thresholds=self.num_thresholds)
        result = auc_obj(self.y_true, self.y_pred, sample_weight=self.sample_weight)

        # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
        # recall = [7/7, 4/(4+3), 0] = [1, 0.571, 0]
        # fp_rate = [3/3, 0, 0] = [1, 0, 0]
        # heights = [(1 + 0.571)/2, (0.571 + 0)/2] = [0.7855, 0.2855]
        # widths = [(1 - 0), (0 - 0)] = [1, 0]
        expected_result = (0.7855 * 1 + 0.2855 * 0)
        assert np.allclose(K.eval(result), expected_result, atol=1e-3)

    def test_weighted_roc_majoring(self):
        self.setup()
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds, summation_method='majoring')
        result = auc_obj(self.y_true, self.y_pred, sample_weight=self.sample_weight)

        # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
        # recall = [7/7, 4/(4+3), 0] = [1, 0.571, 0]
        # fp_rate = [3/3, 0, 0] = [1, 0, 0]
        # heights = [max(1, 0.571), max(0.571, 0)] = [1, 0.571]
        # widths = [(1 - 0), (0 - 0)] = [1, 0]
        expected_result = (1 * 1 + 0.571 * 0)
        assert np.allclose(K.eval(result), expected_result, atol=1e-3)

    def test_weighted_roc_minoring(self):
        self.setup()
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds, summation_method='minoring')
        result = auc_obj(self.y_true, self.y_pred, sample_weight=self.sample_weight)

        # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
        # recall = [7/7, 4/(4+3), 0] = [1, 0.571, 0]
        # fp_rate = [3/3, 0, 0] = [1, 0, 0]
        # heights = [min(1, 0.571), min(0.571, 0)] = [0.571, 0]
        # widths = [(1 - 0), (0 - 0)] = [1, 0]
        expected_result = (0.571 * 1 + 0 * 0)
        assert np.allclose(K.eval(result), expected_result, atol=1e-3)

    def test_weighted_pr_majoring(self):
        self.setup()
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds,
            curve='PR',
            summation_method='majoring')
        result = auc_obj(self.y_true, self.y_pred, sample_weight=self.sample_weight)

        # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
        # precision = [7/(7+3), 4/4, 0] = [0.7, 1, 0]
        # recall = [7/7, 4/(4+3), 0] = [1, 0.571, 0]
        # heights = [max(0.7, 1), max(1, 0)] = [1, 1]
        # widths = [(1 - 0.571), (0.571 - 0)] = [0.429, 0.571]
        expected_result = (1 * 0.429 + 1 * 0.571)
        assert np.allclose(K.eval(result), expected_result, atol=1e-3)

    def test_weighted_pr_minoring(self):
        self.setup()
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds,
            curve='PR',
            summation_method='minoring')
        result = auc_obj(self.y_true, self.y_pred, sample_weight=self.sample_weight)

        # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
        # precision = [7/(7+3), 4/4, 0] = [0.7, 1, 0]
        # recall = [7/7, 4/(4+3), 0] = [1, 0.571, 0]
        # heights = [min(0.7, 1), min(1, 0)] = [0.7, 0]
        # widths = [(1 - 0.571), (0.571 - 0)] = [0.429, 0.571]
        expected_result = (0.7 * 0.429 + 0 * 0.571)
        assert np.allclose(K.eval(result), expected_result, atol=1e-3)

    def test_weighted_pr_interpolation(self):
        self.setup()
        auc_obj = metrics.AUC(num_thresholds=self.num_thresholds, curve='PR')
        result = auc_obj(self.y_true, self.y_pred, sample_weight=self.sample_weight)

        # auc = (slope / Total Pos) * [dTP - intercept * log(Pb/Pa)]

        # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
        # P = tp + fp = [10, 4, 0]
        # dTP = [7-4, 4-0] = [3, 4]
        # dP = [10-4, 4-0] = [6, 4]
        # slope = dTP/dP = [0.5, 1]
        # intercept = (TPa+(slope*Pa) = [(4 - 0.5*4), (0 - 1*0)] = [2, 0]
        # (Pb/Pa) = (Pb/Pa) if Pb > 0 AND Pa > 0 else 1 = [10/4, 4/0] = [2.5, 1]
        # auc * TotalPos = [(0.5 * (3 + 2 * log(2.5))), (1 * (4 + 0))]
        #                = [2.416, 4]
        # auc = [2.416, 4]/(tp[1:]+fn[1:])
        # expected_result = (2.416 / 7 + 4 / 7)
        expected_result = 0.345 + 0.571
        assert np.allclose(K.eval(result), expected_result, atol=1e-3)

    def test_invalid_num_thresholds(self):
        with pytest.raises(Exception):
            metrics.AUC(num_thresholds=-1)

        with pytest.raises(Exception):
            metrics.AUC(num_thresholds=1)

    def test_invalid_curve(self):
        with pytest.raises(Exception):
            metrics.AUC(curve='Invalid')

    def test_invalid_summation_method(self):
        with pytest.raises(Exception):
            metrics.AUC(summation_method='Invalid')


class TestPrecisionTest(object):

    def test_config(self):
        p_obj = metrics.Precision(
            name='my_precision', thresholds=[0.4, 0.9], top_k=15, class_id=12)
        assert p_obj.name == 'my_precision'
        assert len(p_obj.weights) == 2
        assert ([v.name for v in p_obj.weights] ==
                ['true_positives:0', 'false_positives:0'])
        assert p_obj.thresholds == [0.4, 0.9]
        assert p_obj.top_k == 15
        assert p_obj.class_id == 12

        # Check save and restore config
        p_obj2 = metrics.Precision.from_config(p_obj.get_config())
        assert p_obj2.name == 'my_precision'
        assert len(p_obj2.weights) == 2
        assert p_obj2.thresholds == [0.4, 0.9]
        assert p_obj2.top_k == 15
        assert p_obj2.class_id == 12

    def test_unweighted(self):
        p_obj = metrics.Precision()
        y_pred = K.constant([1, 0, 1, 0], shape=(1, 4))
        y_true = K.constant([0, 1, 1, 0], shape=(1, 4))
        result = p_obj(y_true, y_pred)
        assert np.isclose(0.5, K.eval(result))

    def test_unweighted_all_incorrect(self):
        p_obj = metrics.Precision(thresholds=[0.5])
        inputs = np.random.randint(0, 2, size=(100, 1))
        y_pred = K.constant(inputs)
        y_true = K.constant(1 - inputs)
        result = p_obj(y_true, y_pred)
        assert np.isclose(0, K.eval(result))

    def test_weighted(self):
        p_obj = metrics.Precision()
        y_pred = K.constant([[1, 0, 1, 0], [1, 0, 1, 0]])
        y_true = K.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
        result = p_obj(
            y_true,
            y_pred,
            sample_weight=K.constant([[1, 2, 3, 4], [4, 3, 2, 1]]))
        weighted_tp = 3.0 + 4.0
        weighted_positives = (1.0 + 3.0) + (4.0 + 2.0)
        expected_precision = weighted_tp / weighted_positives
        assert np.isclose(expected_precision, K.eval(result))

    def test_unweighted_with_threshold(self):
        p_obj = metrics.Precision(thresholds=[0.5, 0.7])
        y_pred = K.constant([1, 0, 0.6, 0], shape=(1, 4))
        y_true = K.constant([0, 1, 1, 0], shape=(1, 4))
        result = p_obj(y_true, y_pred)
        assert np.allclose([0.5, 0.], K.eval(result), 0)

    def test_weighted_with_threshold(self):
        p_obj = metrics.Precision(thresholds=[0.5, 1.])
        y_true = K.constant([[0, 1], [1, 0]], shape=(2, 2))
        y_pred = K.constant([[1, 0], [0.6, 0]],
                            shape=(2, 2),
                            dtype='float32')
        weights = K.constant([[4, 0], [3, 1]],
                             shape=(2, 2),
                             dtype='float32')
        result = p_obj(y_true, y_pred, sample_weight=weights)
        weighted_tp = 0 + 3.
        weighted_positives = (0 + 3.) + (4. + 0.)
        expected_precision = weighted_tp / weighted_positives
        assert np.allclose([expected_precision, 0], K.eval(result), 1e-3)

    def test_unweighted_top_k(self):
        p_obj = metrics.Precision(top_k=3)
        y_pred = K.constant([0.2, 0.1, 0.5, 0, 0.2], shape=(1, 5))
        y_true = K.constant([0, 1, 1, 0, 0], shape=(1, 5))
        result = p_obj(y_true, y_pred)
        assert np.isclose(1. / 3, K.eval(result))

    def test_weighted_top_k(self):
        p_obj = metrics.Precision(top_k=3)
        y_pred1 = K.constant([0.2, 0.1, 0.4, 0, 0.2], shape=(1, 5))
        y_true1 = K.constant([0, 1, 1, 0, 1], shape=(1, 5))
        K.eval(
            p_obj(
                y_true1,
                y_pred1,
                sample_weight=K.constant([[1, 4, 2, 3, 5]])))

        y_pred2 = K.constant([0.2, 0.6, 0.4, 0.2, 0.2], shape=(1, 5))
        y_true2 = K.constant([1, 0, 1, 1, 1], shape=(1, 5))
        result = p_obj(y_true2, y_pred2, sample_weight=K.constant(3))

        tp = (2 + 5) + (3 + 3)
        predicted_positives = (1 + 2 + 5) + (3 + 3 + 3)
        expected_precision = float(tp) / predicted_positives
        assert np.isclose(expected_precision, K.eval(result))

    def test_unweighted_class_id(self):
        p_obj = metrics.Precision(class_id=2)

        y_pred = K.constant([0.2, 0.1, 0.6, 0, 0.2], shape=(1, 5))
        y_true = K.constant([0, 1, 1, 0, 0], shape=(1, 5))
        result = p_obj(y_true, y_pred)
        assert np.isclose(1, K.eval(result))
        assert np.isclose(1, K.eval(p_obj.true_positives))
        assert np.isclose(0, K.eval(p_obj.false_positives))

        y_pred = K.constant([0.2, 0.1, 0, 0, 0.2], shape=(1, 5))
        y_true = K.constant([0, 1, 1, 0, 0], shape=(1, 5))
        result = p_obj(y_true, y_pred)
        assert np.isclose(1, K.eval(result))
        assert np.isclose(1, K.eval(p_obj.true_positives))
        assert np.isclose(0, K.eval(p_obj.false_positives))

        y_pred = K.constant([0.2, 0.1, 0.6, 0, 0.2], shape=(1, 5))
        y_true = K.constant([0, 1, 0, 0, 0], shape=(1, 5))
        result = p_obj(y_true, y_pred)
        assert np.isclose(0.5, K.eval(result))
        assert np.isclose(1, K.eval(p_obj.true_positives))
        assert np.isclose(1, K.eval(p_obj.false_positives))

    def test_unweighted_top_k_and_class_id(self):
        p_obj = metrics.Precision(class_id=2, top_k=2)

        y_pred = K.constant([0.2, 0.6, 0.3, 0, 0.2], shape=(1, 5))
        y_true = K.constant([0, 1, 1, 0, 0], shape=(1, 5))
        result = p_obj(y_true, y_pred)
        assert np.isclose(1, K.eval(result))
        assert np.isclose(1, K.eval(p_obj.true_positives))
        assert np.isclose(0, K.eval(p_obj.false_positives))

        y_pred = K.constant([1, 1, 0.9, 1, 1], shape=(1, 5))
        y_true = K.constant([0, 1, 1, 0, 0], shape=(1, 5))
        result = p_obj(y_true, y_pred)
        assert np.isclose(1, K.eval(result))
        assert np.isclose(1, K.eval(p_obj.true_positives))
        assert np.isclose(0, K.eval(p_obj.false_positives))

    def test_unweighted_top_k_and_threshold(self):
        p_obj = metrics.Precision(thresholds=.7, top_k=2)

        y_pred = K.constant([0.2, 0.8, 0.6, 0, 0.2], shape=(1, 5))
        y_true = K.constant([0, 1, 1, 0, 1], shape=(1, 5))
        result = p_obj(y_true, y_pred)
        assert np.isclose(1, K.eval(result))
        assert np.isclose(1, K.eval(p_obj.true_positives))
        assert np.isclose(0, K.eval(p_obj.false_positives))


class TestRecall(object):

    def test_config(self):
        r_obj = metrics.Recall(
            name='my_recall', thresholds=[0.4, 0.9], top_k=15, class_id=12)
        assert r_obj.name == 'my_recall'
        assert len(r_obj.weights) == 2
        assert ([v.name for v in r_obj.weights] ==
                ['true_positives:0', 'false_negatives:0'])
        assert r_obj.thresholds == [0.4, 0.9]
        assert r_obj.top_k == 15
        assert r_obj.class_id == 12

        # Check save and restore config
        r_obj2 = metrics.Recall.from_config(r_obj.get_config())
        assert r_obj2.name == 'my_recall'
        assert len(r_obj2.weights) == 2
        assert r_obj2.thresholds == [0.4, 0.9]
        assert r_obj2.top_k == 15
        assert r_obj2.class_id == 12

    def test_unweighted(self):
        r_obj = metrics.Recall()
        y_pred = K.constant([1, 0, 1, 0], shape=(1, 4))
        y_true = K.constant([0, 1, 1, 0], shape=(1, 4))
        result = r_obj(y_true, y_pred)
        assert np.isclose(0.5, K.eval(result))

    def test_unweighted_all_incorrect(self):
        r_obj = metrics.Recall(thresholds=[0.5])
        inputs = np.random.randint(0, 2, size=(100, 1))
        y_pred = K.constant(inputs)
        y_true = K.constant(1 - inputs)
        result = r_obj(y_true, y_pred)
        assert np.isclose(0, K.eval(result))

    def test_weighted(self):
        r_obj = metrics.Recall()
        y_pred = K.constant([[1, 0, 1, 0], [0, 1, 0, 1]])
        y_true = K.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
        result = r_obj(
            y_true,
            y_pred,
            sample_weight=K.constant([[1, 2, 3, 4], [4, 3, 2, 1]]))
        weighted_tp = 3.0 + 1.0
        weighted_t = (2.0 + 3.0) + (4.0 + 1.0)
        expected_recall = weighted_tp / weighted_t
        assert np.isclose(expected_recall, K.eval(result))

    def test_unweighted_with_threshold(self):
        r_obj = metrics.Recall(thresholds=[0.5, 0.7])
        y_pred = K.constant([1, 0, 0.6, 0], shape=(1, 4))
        y_true = K.constant([0, 1, 1, 0], shape=(1, 4))
        result = r_obj(y_true, y_pred)
        assert np.allclose([0.5, 0.], K.eval(result), 0)

    def test_weighted_with_threshold(self):
        r_obj = metrics.Recall(thresholds=[0.5, 1.])
        y_true = K.constant([[0, 1], [1, 0]], shape=(2, 2))
        y_pred = K.constant([[1, 0], [0.6, 0]],
                            shape=(2, 2),
                            dtype='float32')
        weights = K.constant([[1, 4], [3, 2]],
                             shape=(2, 2),
                             dtype='float32')
        result = r_obj(y_true, y_pred, sample_weight=weights)
        weighted_tp = 0 + 3.
        weighted_positives = (0 + 3.) + (4. + 0.)
        expected_recall = weighted_tp / weighted_positives
        assert np.allclose([expected_recall, 0], K.eval(result), 1e-3)

    def test_unweighted_top_k(self):
        r_obj = metrics.Recall(top_k=3)
        y_pred = K.constant([0.2, 0.1, 0.5, 0, 0.2], shape=(1, 5))
        y_true = K.constant([0, 1, 1, 0, 0], shape=(1, 5))
        result = r_obj(y_true, y_pred)
        assert np.isclose(0.5, K.eval(result))

    def test_weighted_top_k(self):
        r_obj = metrics.Recall(top_k=3)
        y_pred1 = K.constant([0.2, 0.1, 0.4, 0, 0.2], shape=(1, 5))
        y_true1 = K.constant([0, 1, 1, 0, 1], shape=(1, 5))
        K.eval(
            r_obj(
                y_true1,
                y_pred1,
                sample_weight=K.constant([[1, 4, 2, 3, 5]])))

        y_pred2 = K.constant([0.2, 0.6, 0.4, 0.2, 0.2], shape=(1, 5))
        y_true2 = K.constant([1, 0, 1, 1, 1], shape=(1, 5))
        result = r_obj(y_true2, y_pred2, sample_weight=K.constant(3))

        tp = (2 + 5) + (3 + 3)
        positives = (4 + 2 + 5) + (3 + 3 + 3 + 3)
        expected_recall = float(tp) / positives
        assert np.isclose(expected_recall, K.eval(result))

    def test_unweighted_class_id(self):
        r_obj = metrics.Recall(class_id=2)

        y_pred = K.constant([0.2, 0.1, 0.6, 0, 0.2], shape=(1, 5))
        y_true = K.constant([0, 1, 1, 0, 0], shape=(1, 5))
        result = r_obj(y_true, y_pred)
        assert np.isclose(1, K.eval(result))
        assert np.isclose(1, K.eval(r_obj.true_positives))
        assert np.isclose(0, K.eval(r_obj.false_negatives))

        y_pred = K.constant([0.2, 0.1, 0, 0, 0.2], shape=(1, 5))
        y_true = K.constant([0, 1, 1, 0, 0], shape=(1, 5))
        result = r_obj(y_true, y_pred)
        assert np.isclose(0.5, K.eval(result))
        assert np.isclose(1, K.eval(r_obj.true_positives))
        assert np.isclose(1, K.eval(r_obj.false_negatives))

        y_pred = K.constant([0.2, 0.1, 0.6, 0, 0.2], shape=(1, 5))
        y_true = K.constant([0, 1, 0, 0, 0], shape=(1, 5))
        result = r_obj(y_true, y_pred)
        assert np.isclose(0.5, K.eval(result))
        assert np.isclose(1, K.eval(r_obj.true_positives))
        assert np.isclose(1, K.eval(r_obj.false_negatives))

    def test_unweighted_top_k_and_class_id(self):
        r_obj = metrics.Recall(class_id=2, top_k=2)

        y_pred = K.constant([0.2, 0.6, 0.3, 0, 0.2], shape=(1, 5))
        y_true = K.constant([0, 1, 1, 0, 0], shape=(1, 5))
        result = r_obj(y_true, y_pred)
        assert np.isclose(1, K.eval(result))
        assert np.isclose(1, K.eval(r_obj.true_positives))
        assert np.isclose(0, K.eval(r_obj.false_negatives))

        y_pred = K.constant([1, 1, 0.9, 1, 1], shape=(1, 5))
        y_true = K.constant([0, 1, 1, 0, 0], shape=(1, 5))
        result = r_obj(y_true, y_pred)
        assert np.isclose(0.5, K.eval(result))
        assert np.isclose(1, K.eval(r_obj.true_positives))
        assert np.isclose(1, K.eval(r_obj.false_negatives))

    def test_unweighted_top_k_and_threshold(self):
        r_obj = metrics.Recall(thresholds=.7, top_k=2)

        y_pred = K.constant([0.2, 0.8, 0.6, 0, 0.2], shape=(1, 5))
        y_true = K.constant([1, 1, 1, 0, 1], shape=(1, 5))
        result = r_obj(y_true, y_pred)
        assert np.isclose(0.25, K.eval(result))
        assert np.isclose(1, K.eval(r_obj.true_positives))
        assert np.isclose(3, K.eval(r_obj.false_negatives))


@pytest.mark.skipif(not tf.__version__.startswith('2.'),
                    reason='Requires TF 2')
class TestMeanIoU(object):

    def test_config(self):
        m_obj = metrics.MeanIoU(num_classes=2, name='mean_iou')
        assert m_obj.name == 'mean_iou'
        assert m_obj.num_classes == 2

        m_obj2 = metrics.MeanIoU.from_config(m_obj.get_config())
        assert m_obj2.name == 'mean_iou'
        assert m_obj2.num_classes == 2

    def test_unweighted(self):
        y_pred = K.constant([0, 1, 0, 1], shape=(1, 4))
        y_true = K.constant([0, 0, 1, 1], shape=(1, 4))

        m_obj = metrics.MeanIoU(num_classes=2)
        result = m_obj(y_true, y_pred)

        # cm = [[1, 1],
        #       [1, 1]]
        # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (1. / (2 + 2 - 1) + 1. / (2 + 2 - 1)) / 2
        assert np.allclose(K.eval(result), expected_result, atol=1e-3)

    def test_weighted(self):
        y_pred = K.constant([0, 1, 0, 1], dtype='float32')
        y_true = K.constant([0, 0, 1, 1])
        sample_weight = K.constant([0.2, 0.3, 0.4, 0.1])

        m_obj = metrics.MeanIoU(num_classes=2)
        result = m_obj(y_true, y_pred, sample_weight=sample_weight)

        # cm = [[0.2, 0.3],
        #       [0.4, 0.1]]
        # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5], true_positives = [0.2, 0.1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (0.2 / (0.6 + 0.5 - 0.2) + 0.1 / (0.4 + 0.5 - 0.1)) / 2
        assert np.allclose(K.eval(result), expected_result, atol=1e-3)

    def test_multi_dim_input(self):
        y_pred = K.constant([[0, 1], [0, 1]], dtype='float32')
        y_true = K.constant([[0, 0], [1, 1]])
        sample_weight = K.constant([[0.2, 0.3], [0.4, 0.1]])

        m_obj = metrics.MeanIoU(num_classes=2)
        result = m_obj(y_true, y_pred, sample_weight=sample_weight)

        # cm = [[0.2, 0.3],
        #       [0.4, 0.1]]
        # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5], true_positives = [0.2, 0.1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (0.2 / (0.6 + 0.5 - 0.2) + 0.1 / (0.4 + 0.5 - 0.1)) / 2
        assert np.allclose(K.eval(result), expected_result, atol=1e-3)

    def test_zero_valid_entries(self):
        m_obj = metrics.MeanIoU(num_classes=2)
        assert np.allclose(K.eval(m_obj.result()), 0, atol=1e-3)

    def test_zero_and_non_zero_entries(self):
        y_pred = K.constant([1], dtype='float32')
        y_true = K.constant([1])

        m_obj = metrics.MeanIoU(num_classes=2)
        result = m_obj(y_true, y_pred)

        # cm = [[0, 0],
        #       [0, 1]]
        # sum_row = [0, 1], sum_col = [0, 1], true_positives = [0, 1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (0. + 1. / (1 + 1 - 1)) / 1
        assert np.allclose(K.eval(result), expected_result, atol=1e-3)
