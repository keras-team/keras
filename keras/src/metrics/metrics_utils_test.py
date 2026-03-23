import numpy as np

from keras.src import ops
from keras.src import testing
from keras.src.metrics.metrics_utils import AUCCurve
from keras.src.metrics.metrics_utils import AUCSummationMethod
from keras.src.metrics.metrics_utils import ConfusionMatrix
from keras.src.metrics.metrics_utils import assert_thresholds_range
from keras.src.metrics.metrics_utils import confusion_matrix
from keras.src.metrics.metrics_utils import is_evenly_distributed_thresholds
from keras.src.metrics.metrics_utils import parse_init_thresholds
from keras.src.metrics.metrics_utils import update_confusion_matrix_variables


class AssertThresholdsRangeTest(testing.TestCase):
    def test_valid_thresholds(self):
        # Should not raise
        assert_thresholds_range([0.0, 0.5, 1.0])

    def test_none_thresholds(self):
        # Should not raise
        assert_thresholds_range(None)

    def test_threshold_below_zero_raises(self):
        with self.assertRaises(ValueError):
            assert_thresholds_range([-0.1, 0.5])

    def test_threshold_above_one_raises(self):
        with self.assertRaises(ValueError):
            assert_thresholds_range([0.5, 1.1])

    def test_none_in_thresholds_raises(self):
        with self.assertRaises(ValueError):
            assert_thresholds_range([None, 0.5])

    def test_boundary_values(self):
        assert_thresholds_range([0.0, 1.0])

    def test_single_threshold(self):
        assert_thresholds_range([0.5])

    def test_empty_list(self):
        assert_thresholds_range([])


class ParseInitThresholdsTest(testing.TestCase):
    def test_none_returns_default(self):
        result = parse_init_thresholds(None)
        self.assertEqual(result, [0.5])

    def test_custom_default(self):
        result = parse_init_thresholds(None, default_threshold=0.7)
        self.assertEqual(result, [0.7])

    def test_single_threshold(self):
        result = parse_init_thresholds(0.3)
        self.assertEqual(result, [0.3])

    def test_list_of_thresholds(self):
        result = parse_init_thresholds([0.1, 0.5, 0.9])
        self.assertEqual(result, [0.1, 0.5, 0.9])

    def test_invalid_threshold_raises(self):
        with self.assertRaises(ValueError):
            parse_init_thresholds(1.5)

    def test_negative_threshold_raises(self):
        with self.assertRaises(ValueError):
            parse_init_thresholds(-0.1)


class ConfusionMatrixTest(testing.TestCase):
    def test_values(self):
        self.assertEqual(ConfusionMatrix.TRUE_POSITIVES.value, "tp")
        self.assertEqual(ConfusionMatrix.FALSE_POSITIVES.value, "fp")
        self.assertEqual(ConfusionMatrix.TRUE_NEGATIVES.value, "tn")
        self.assertEqual(ConfusionMatrix.FALSE_NEGATIVES.value, "fn")


class AUCCurveTest(testing.TestCase):
    def test_roc_from_str(self):
        self.assertEqual(AUCCurve.from_str("ROC"), AUCCurve.ROC)
        self.assertEqual(AUCCurve.from_str("roc"), AUCCurve.ROC)

    def test_pr_from_str(self):
        self.assertEqual(AUCCurve.from_str("PR"), AUCCurve.PR)
        self.assertEqual(AUCCurve.from_str("pr"), AUCCurve.PR)

    def test_prgain_from_str(self):
        self.assertEqual(AUCCurve.from_str("PRGAIN"), AUCCurve.PRGAIN)
        self.assertEqual(AUCCurve.from_str("prgain"), AUCCurve.PRGAIN)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            AUCCurve.from_str("invalid")

    def test_values(self):
        self.assertEqual(AUCCurve.ROC.value, "ROC")
        self.assertEqual(AUCCurve.PR.value, "PR")
        self.assertEqual(AUCCurve.PRGAIN.value, "PRGAIN")


class AUCSummationMethodTest(testing.TestCase):
    def test_interpolation_from_str(self):
        result = AUCSummationMethod.from_str("interpolation")
        self.assertEqual(result, AUCSummationMethod.INTERPOLATION)

    def test_majoring_from_str(self):
        result = AUCSummationMethod.from_str("majoring")
        self.assertEqual(result, AUCSummationMethod.MAJORING)

    def test_minoring_from_str(self):
        result = AUCSummationMethod.from_str("minoring")
        self.assertEqual(result, AUCSummationMethod.MINORING)

    def test_capitalized_from_str(self):
        self.assertEqual(
            AUCSummationMethod.from_str("Interpolation"),
            AUCSummationMethod.INTERPOLATION,
        )
        self.assertEqual(
            AUCSummationMethod.from_str("Majoring"),
            AUCSummationMethod.MAJORING,
        )
        self.assertEqual(
            AUCSummationMethod.from_str("Minoring"),
            AUCSummationMethod.MINORING,
        )

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            AUCSummationMethod.from_str("invalid")

    def test_values(self):
        self.assertEqual(
            AUCSummationMethod.INTERPOLATION.value, "interpolation"
        )
        self.assertEqual(AUCSummationMethod.MAJORING.value, "majoring")
        self.assertEqual(AUCSummationMethod.MINORING.value, "minoring")


class IsEvenlyDistributedThresholdsTest(testing.TestCase):
    def test_returns_false_for_less_than_3_thresholds(self):
        self.assertFalse(is_evenly_distributed_thresholds([]))
        self.assertFalse(is_evenly_distributed_thresholds([0.5]))
        self.assertFalse(is_evenly_distributed_thresholds([0.0, 1.0]))

    def test_returns_true_for_evenly_distributed(self):
        thresholds = list(np.arange(5, dtype=np.float32) / 4)
        self.assertTrue(is_evenly_distributed_thresholds(thresholds))

    def test_returns_false_for_unevenly_distributed(self):
        self.assertFalse(is_evenly_distributed_thresholds([0.0, 0.3, 0.7, 1.0]))

    def test_returns_true_for_3_even_thresholds(self):
        self.assertTrue(is_evenly_distributed_thresholds([0.0, 0.5, 1.0]))

    def test_returns_true_for_11_thresholds(self):
        thresholds = list(np.arange(11, dtype=np.float32) / 10)
        self.assertTrue(is_evenly_distributed_thresholds(thresholds))

    def test_returns_false_for_non_starting_at_zero(self):
        self.assertFalse(
            is_evenly_distributed_thresholds([0.1, 0.35, 0.6, 0.85])
        )


class ConfusionMatrixFunctionTest(testing.TestCase):
    def test_perfect_predictions(self):
        cm = ops.convert_to_numpy(
            confusion_matrix([0, 1, 2], [0, 1, 2], num_classes=3)
        )
        expected = np.eye(3, dtype=np.int32)
        np.testing.assert_array_equal(cm, expected)

    def test_all_misclassified(self):
        cm = ops.convert_to_numpy(
            confusion_matrix([0, 0, 1, 1], [1, 1, 0, 0], num_classes=2)
        )
        expected = np.array([[0, 2], [2, 0]], dtype=np.int32)
        np.testing.assert_array_equal(cm, expected)

    def test_mixed_predictions(self):
        # from docstring: labels=[1,2,4], predictions=[2,2,4]
        cm = ops.convert_to_numpy(
            confusion_matrix([1, 2, 4], [2, 2, 4], num_classes=5)
        )
        self.assertEqual(cm[1, 2], 1)
        self.assertEqual(cm[2, 2], 1)
        self.assertEqual(cm[4, 4], 1)
        self.assertEqual(cm.sum(), 3)

    def test_num_classes_larger_than_max_label(self):
        cm = ops.convert_to_numpy(
            confusion_matrix([0, 1], [0, 1], num_classes=4)
        )
        self.assertEqual(cm.shape, (4, 4))
        self.assertEqual(int(cm[0, 0]), 1)
        self.assertEqual(int(cm[1, 1]), 1)

    def test_with_weights(self):
        # labels=[0,0,1], predictions=[0,0,1], weights=[2,3,5]
        cm = ops.convert_to_numpy(
            confusion_matrix(
                [0, 0, 1], [0, 0, 1], num_classes=2, weights=[2, 3, 5]
            )
        )
        self.assertEqual(int(cm[0, 0]), 5)  # 2+3
        self.assertEqual(int(cm[1, 1]), 5)

    def test_single_class(self):
        cm = ops.convert_to_numpy(
            confusion_matrix([0, 0, 0], [0, 0, 0], num_classes=1)
        )
        self.assertEqual(int(cm[0, 0]), 3)

    def test_output_shape(self):
        cm = ops.convert_to_numpy(
            confusion_matrix([0, 1, 2], [2, 1, 0], num_classes=3)
        )
        self.assertEqual(cm.shape, (3, 3))


class UpdateConfusionMatrixVariablesTest(testing.TestCase):
    def _make_vars(self, num_thresholds):
        from keras.src.backend import Variable

        shape = (num_thresholds,)
        return {
            ConfusionMatrix.TRUE_POSITIVES: Variable(
                np.zeros(shape, dtype="float32"), trainable=False
            ),
            ConfusionMatrix.FALSE_POSITIVES: Variable(
                np.zeros(shape, dtype="float32"), trainable=False
            ),
            ConfusionMatrix.TRUE_NEGATIVES: Variable(
                np.zeros(shape, dtype="float32"), trainable=False
            ),
            ConfusionMatrix.FALSE_NEGATIVES: Variable(
                np.zeros(shape, dtype="float32"), trainable=False
            ),
        }

    def test_all_true_positive_at_low_threshold(self):
        y_true = np.ones(4, dtype="float32")
        y_pred = np.full(4, 0.9, dtype="float32")
        vars_ = self._make_vars(1)
        update_confusion_matrix_variables(vars_, y_true, y_pred, [0.5])
        tp = float(vars_[ConfusionMatrix.TRUE_POSITIVES].numpy()[0])
        fp = float(vars_[ConfusionMatrix.FALSE_POSITIVES].numpy()[0])
        fn = float(vars_[ConfusionMatrix.FALSE_NEGATIVES].numpy()[0])
        tn = float(vars_[ConfusionMatrix.TRUE_NEGATIVES].numpy()[0])
        self.assertAlmostEqual(tp, 4.0)
        self.assertAlmostEqual(fp, 0.0)
        self.assertAlmostEqual(fn, 0.0)
        self.assertAlmostEqual(tn, 0.0)

    def test_all_false_positive_at_low_threshold(self):
        y_true = np.zeros(4, dtype="float32")
        y_pred = np.full(4, 0.9, dtype="float32")
        vars_ = self._make_vars(1)
        update_confusion_matrix_variables(vars_, y_true, y_pred, [0.5])
        tp = float(vars_[ConfusionMatrix.TRUE_POSITIVES].numpy()[0])
        fp = float(vars_[ConfusionMatrix.FALSE_POSITIVES].numpy()[0])
        fn = float(vars_[ConfusionMatrix.FALSE_NEGATIVES].numpy()[0])
        tn = float(vars_[ConfusionMatrix.TRUE_NEGATIVES].numpy()[0])
        self.assertAlmostEqual(tp, 0.0)
        self.assertAlmostEqual(fp, 4.0)
        self.assertAlmostEqual(fn, 0.0)
        self.assertAlmostEqual(tn, 0.0)

    def test_all_true_negative_at_high_threshold(self):
        y_true = np.zeros(4, dtype="float32")
        y_pred = np.full(4, 0.1, dtype="float32")
        vars_ = self._make_vars(1)
        update_confusion_matrix_variables(vars_, y_true, y_pred, [0.5])
        tn = float(vars_[ConfusionMatrix.TRUE_NEGATIVES].numpy()[0])
        self.assertAlmostEqual(tn, 4.0)

    def test_all_false_negative_at_high_threshold(self):
        y_true = np.ones(4, dtype="float32")
        y_pred = np.full(4, 0.1, dtype="float32")
        vars_ = self._make_vars(1)
        update_confusion_matrix_variables(vars_, y_true, y_pred, [0.5])
        fn = float(vars_[ConfusionMatrix.FALSE_NEGATIVES].numpy()[0])
        self.assertAlmostEqual(fn, 4.0)

    def test_mixed_tp_fp_fn_tn(self):
        # y_true=[1,1,0,0], y_pred=[0.9,0.1,0.9,0.1], threshold=0.5
        # TP=1, FN=1, FP=1, TN=1
        y_true = np.array([1, 1, 0, 0], dtype="float32")
        y_pred = np.array([0.9, 0.1, 0.9, 0.1], dtype="float32")
        vars_ = self._make_vars(1)
        update_confusion_matrix_variables(vars_, y_true, y_pred, [0.5])
        tp = float(vars_[ConfusionMatrix.TRUE_POSITIVES].numpy()[0])
        fp = float(vars_[ConfusionMatrix.FALSE_POSITIVES].numpy()[0])
        fn = float(vars_[ConfusionMatrix.FALSE_NEGATIVES].numpy()[0])
        tn = float(vars_[ConfusionMatrix.TRUE_NEGATIVES].numpy()[0])
        self.assertAlmostEqual(tp, 1.0)
        self.assertAlmostEqual(fp, 1.0)
        self.assertAlmostEqual(fn, 1.0)
        self.assertAlmostEqual(tn, 1.0)

    def test_multiple_thresholds(self):
        # y_true=[1,0], y_pred=[0.8,0.3], thresholds=[0.2, 0.5, 0.9]
        # t=0.2: TP=1, FP=1; t=0.5: TP=1, TN=1; t=0.9: FN=1, TN=1
        y_true = np.array([1.0, 0.0], dtype="float32")
        y_pred = np.array([0.8, 0.3], dtype="float32")
        vars_ = self._make_vars(3)
        update_confusion_matrix_variables(
            vars_, y_true, y_pred, [0.2, 0.5, 0.9]
        )
        tp = vars_[ConfusionMatrix.TRUE_POSITIVES].numpy()
        tn = vars_[ConfusionMatrix.TRUE_NEGATIVES].numpy()
        self.assertAlmostEqual(float(tp[0]), 1.0)
        self.assertAlmostEqual(float(tp[1]), 1.0)
        self.assertAlmostEqual(float(tp[2]), 0.0)
        self.assertAlmostEqual(float(tn[0]), 0.0)
        self.assertAlmostEqual(float(tn[1]), 1.0)
        self.assertAlmostEqual(float(tn[2]), 1.0)

    def test_tp_plus_fn_equals_total_positives(self):
        # Conservation: TP + FN = total positives
        y_true = np.array([1, 1, 0, 1, 0], dtype="float32")
        y_pred = np.array([0.9, 0.4, 0.6, 0.3, 0.1], dtype="float32")
        vars_ = self._make_vars(1)
        update_confusion_matrix_variables(vars_, y_true, y_pred, [0.5])
        tp = float(vars_[ConfusionMatrix.TRUE_POSITIVES].numpy()[0])
        fn = float(vars_[ConfusionMatrix.FALSE_NEGATIVES].numpy()[0])
        self.assertAlmostEqual(tp + fn, float(y_true.sum()))

    def test_fp_plus_tn_equals_total_negatives(self):
        # Conservation: FP + TN = total negatives
        y_true = np.array([1, 1, 0, 1, 0], dtype="float32")
        y_pred = np.array([0.9, 0.4, 0.6, 0.3, 0.1], dtype="float32")
        vars_ = self._make_vars(1)
        update_confusion_matrix_variables(vars_, y_true, y_pred, [0.5])
        fp = float(vars_[ConfusionMatrix.FALSE_POSITIVES].numpy()[0])
        tn = float(vars_[ConfusionMatrix.TRUE_NEGATIVES].numpy()[0])
        self.assertAlmostEqual(fp + tn, float((1.0 - y_true).sum()))

    def test_with_sample_weights(self):
        # y_true=[1,0], y_pred=[0.9,0.9], weights=[2.0,3.0], threshold 0.5
        # TP = 2.0; FP = 3.0
        y_true = np.array([1.0, 0.0], dtype="float32")
        y_pred = np.array([0.9, 0.9], dtype="float32")
        weights = np.array([2.0, 3.0], dtype="float32")
        vars_ = self._make_vars(1)
        update_confusion_matrix_variables(
            vars_, y_true, y_pred, [0.5], sample_weight=weights
        )
        tp = float(vars_[ConfusionMatrix.TRUE_POSITIVES].numpy()[0])
        fp = float(vars_[ConfusionMatrix.FALSE_POSITIVES].numpy()[0])
        self.assertAlmostEqual(tp, 2.0)
        self.assertAlmostEqual(fp, 3.0)

    def test_evenly_distributed_thresholds_path(self):
        # Optimized path with thresholds_distributed_evenly=True
        thresholds = list(np.arange(11, dtype=np.float64) / 10)
        y_true = np.array([1, 1, 0, 0], dtype="float32")
        y_pred = np.array([0.9, 0.4, 0.6, 0.1], dtype="float32")
        vars_ = self._make_vars(11)
        update_confusion_matrix_variables(
            vars_,
            y_true,
            y_pred,
            thresholds,
            thresholds_distributed_evenly=True,
        )
        # At threshold 0.5 (index 5): TP=1, FP=1
        tp = vars_[ConfusionMatrix.TRUE_POSITIVES].numpy()
        fp = vars_[ConfusionMatrix.FALSE_POSITIVES].numpy()
        self.assertAlmostEqual(float(tp[5]), 1.0)
        self.assertAlmostEqual(float(fp[5]), 1.0)


class AdditionalEdgeCaseTest(testing.TestCase):
    def test_parse_init_thresholds_boundary_zero(self):
        self.assertEqual(parse_init_thresholds(0.0), [0.0])

    def test_parse_init_thresholds_boundary_one(self):
        self.assertEqual(parse_init_thresholds(1.0), [1.0])

    def test_parse_init_thresholds_with_python_list(self):
        result = parse_init_thresholds([0.2, 0.5, 0.8])
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 0.2, decimal=5)
        self.assertAlmostEqual(result[1], 0.5, decimal=5)
        self.assertAlmostEqual(result[2], 0.8, decimal=5)

    def test_auc_curve_from_str_invalid(self):
        with self.assertRaises(ValueError):
            AUCCurve.from_str("invalid_curve")

    def test_auc_summation_method_from_str_empty_string_raises(self):
        with self.assertRaises(ValueError):
            AUCSummationMethod.from_str("")

    def test_confusion_matrix_int_dtype(self):
        cm = ops.convert_to_numpy(
            confusion_matrix([0, 1], [0, 1], num_classes=2, dtype="int32")
        )
        self.assertEqual(cm.dtype, np.int32)

    def test_confusion_matrix_anti_diagonal(self):
        # All misclassified: [0→1, 1→0]
        cm = ops.convert_to_numpy(
            confusion_matrix([0, 1], [1, 0], num_classes=2, dtype="int32")
        )
        self.assertEqual(int(cm[0, 1]), 1)
        self.assertEqual(int(cm[1, 0]), 1)
        self.assertEqual(int(cm[0, 0]), 0)
        self.assertEqual(int(cm[1, 1]), 0)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
