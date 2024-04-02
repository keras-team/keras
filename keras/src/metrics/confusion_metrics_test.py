import json

import numpy as np
import pytest
from absl import logging
from absl.testing import parameterized

from keras.src import layers
from keras.src import metrics
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.metrics import metrics_utils


class FalsePositivesTest(testing.TestCase):
    def test_config(self):
        fp_obj = metrics.FalsePositives(name="my_fp", thresholds=[0.4, 0.9])
        self.assertEqual(fp_obj.name, "my_fp")
        self.assertLen(fp_obj.variables, 1)
        self.assertEqual(fp_obj.thresholds, [0.4, 0.9])

        # Check save and restore config
        fp_obj2 = metrics.FalsePositives.from_config(fp_obj.get_config())
        self.assertEqual(fp_obj2.name, "my_fp")
        self.assertLen(fp_obj2.variables, 1)
        self.assertEqual(fp_obj2.thresholds, [0.4, 0.9])

    def test_unweighted(self):
        fp_obj = metrics.FalsePositives()

        y_true = np.array(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = np.array(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )

        fp_obj.update_state(y_true, y_pred)
        self.assertAllClose(7.0, fp_obj.result())

    def test_weighted(self):
        fp_obj = metrics.FalsePositives()
        y_true = np.array(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = np.array(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )
        sample_weight = np.array((1.0, 1.5, 2.0, 2.5))
        result = fp_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(14.0, result)

    def test_unweighted_with_thresholds(self):
        fp_obj = metrics.FalsePositives(thresholds=[0.15, 0.5, 0.85])

        y_pred = np.array(
            (
                (0.9, 0.2, 0.8, 0.1),
                (0.2, 0.9, 0.7, 0.6),
                (0.1, 0.2, 0.4, 0.3),
                (0, 1, 0.7, 0.3),
            )
        )
        y_true = np.array(
            ((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1))
        )

        fp_obj.update_state(y_true, y_pred)
        self.assertAllClose([7.0, 4.0, 2.0], fp_obj.result())

    def test_weighted_with_thresholds(self):
        fp_obj = metrics.FalsePositives(thresholds=[0.15, 0.5, 0.85])

        y_pred = np.array(
            (
                (0.9, 0.2, 0.8, 0.1),
                (0.2, 0.9, 0.7, 0.6),
                (0.1, 0.2, 0.4, 0.3),
                (0, 1, 0.7, 0.3),
            )
        )
        y_true = np.array(
            ((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1))
        )
        sample_weight = (
            (1.0, 2.0, 3.0, 5.0),
            (7.0, 11.0, 13.0, 17.0),
            (19.0, 23.0, 29.0, 31.0),
            (5.0, 15.0, 10.0, 0),
        )

        result = fp_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose([125.0, 42.0, 12.0], result)

    def test_threshold_limit(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Threshold values must be in \[0, 1\]. Received: \[-1, 2\]",
        ):
            metrics.FalsePositives(thresholds=[-1, 0.5, 2])

        with self.assertRaisesRegex(
            ValueError,
            r"Threshold values must be in \[0, 1\]. Received: \[None\]",
        ):
            metrics.FalsePositives(thresholds=[None])


class FalseNegativesTest(testing.TestCase):
    def test_config(self):
        fn_obj = metrics.FalseNegatives(name="my_fn", thresholds=[0.4, 0.9])
        self.assertEqual(fn_obj.name, "my_fn")
        self.assertLen(fn_obj.variables, 1)
        self.assertEqual(fn_obj.thresholds, [0.4, 0.9])

        # Check save and restore config
        fn_obj2 = metrics.FalseNegatives.from_config(fn_obj.get_config())
        self.assertEqual(fn_obj2.name, "my_fn")
        self.assertLen(fn_obj2.variables, 1)
        self.assertEqual(fn_obj2.thresholds, [0.4, 0.9])

    def test_unweighted(self):
        fn_obj = metrics.FalseNegatives()

        y_true = np.array(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = np.array(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )

        fn_obj.update_state(y_true, y_pred)
        self.assertAllClose(3.0, fn_obj.result())

    def test_weighted(self):
        fn_obj = metrics.FalseNegatives()
        y_true = np.array(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = np.array(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )
        sample_weight = np.array((1.0, 1.5, 2.0, 2.5))
        result = fn_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(5.0, result)

    def test_unweighted_with_thresholds(self):
        fn_obj = metrics.FalseNegatives(thresholds=[0.15, 0.5, 0.85])

        y_pred = np.array(
            (
                (0.9, 0.2, 0.8, 0.1),
                (0.2, 0.9, 0.7, 0.6),
                (0.1, 0.2, 0.4, 0.3),
                (0, 1, 0.7, 0.3),
            )
        )
        y_true = np.array(
            ((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1))
        )

        fn_obj.update_state(y_true, y_pred)
        self.assertAllClose([1.0, 4.0, 6.0], fn_obj.result())

    def test_weighted_with_thresholds(self):
        fn_obj = metrics.FalseNegatives(thresholds=[0.15, 0.5, 0.85])

        y_pred = np.array(
            (
                (0.9, 0.2, 0.8, 0.1),
                (0.2, 0.9, 0.7, 0.6),
                (0.1, 0.2, 0.4, 0.3),
                (0, 1, 0.7, 0.3),
            )
        )
        y_true = np.array(
            ((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1))
        )
        sample_weight = ((3.0,), (5.0,), (7.0,), (4.0,))

        result = fn_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose([4.0, 16.0, 23.0], result)

    def test_threshold_limit(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Threshold values must be in \[0, 1\]. Received: \[-1, 2\]",
        ):
            metrics.FalseNegatives(thresholds=[-1, 0.5, 2])

        with self.assertRaisesRegex(
            ValueError,
            r"Threshold values must be in \[0, 1\]. Received: \[None\]",
        ):
            metrics.FalseNegatives(thresholds=[None])


class TrueNegativesTest(testing.TestCase):
    def test_config(self):
        tn_obj = metrics.TrueNegatives(name="my_tn", thresholds=[0.4, 0.9])
        self.assertEqual(tn_obj.name, "my_tn")
        self.assertLen(tn_obj.variables, 1)
        self.assertEqual(tn_obj.thresholds, [0.4, 0.9])

        # Check save and restore config
        tn_obj2 = metrics.TrueNegatives.from_config(tn_obj.get_config())
        self.assertEqual(tn_obj2.name, "my_tn")
        self.assertLen(tn_obj2.variables, 1)
        self.assertEqual(tn_obj2.thresholds, [0.4, 0.9])

    def test_unweighted(self):
        tn_obj = metrics.TrueNegatives()

        y_true = np.array(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = np.array(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )

        tn_obj.update_state(y_true, y_pred)
        self.assertAllClose(3.0, tn_obj.result())

    def test_weighted(self):
        tn_obj = metrics.TrueNegatives()
        y_true = np.array(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = np.array(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )
        sample_weight = np.array((1.0, 1.5, 2.0, 2.5))
        result = tn_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(4.0, result)

    def test_unweighted_with_thresholds(self):
        tn_obj = metrics.TrueNegatives(thresholds=[0.15, 0.5, 0.85])

        y_pred = np.array(
            (
                (0.9, 0.2, 0.8, 0.1),
                (0.2, 0.9, 0.7, 0.6),
                (0.1, 0.2, 0.4, 0.3),
                (0, 1, 0.7, 0.3),
            )
        )
        y_true = np.array(
            ((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1))
        )

        tn_obj.update_state(y_true, y_pred)
        self.assertAllClose([2.0, 5.0, 7.0], tn_obj.result())

    def test_weighted_with_thresholds(self):
        tn_obj = metrics.TrueNegatives(thresholds=[0.15, 0.5, 0.85])

        y_pred = np.array(
            (
                (0.9, 0.2, 0.8, 0.1),
                (0.2, 0.9, 0.7, 0.6),
                (0.1, 0.2, 0.4, 0.3),
                (0, 1, 0.7, 0.3),
            )
        )
        y_true = np.array(
            ((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1))
        )
        sample_weight = ((0.0, 2.0, 3.0, 5.0),)

        result = tn_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose([5.0, 15.0, 23.0], result)

    def test_threshold_limit(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Threshold values must be in \[0, 1\]. Received: \[-1, 2\]",
        ):
            metrics.TrueNegatives(thresholds=[-1, 0.5, 2])

        with self.assertRaisesRegex(
            ValueError,
            r"Threshold values must be in \[0, 1\]. Received: \[None\]",
        ):
            metrics.TrueNegatives(thresholds=[None])


class TruePositiveTest(testing.TestCase):
    def test_config(self):
        tp_obj = metrics.TruePositives(name="my_tp", thresholds=[0.4, 0.9])
        self.assertEqual(tp_obj.name, "my_tp")
        self.assertLen(tp_obj.variables, 1)
        self.assertEqual(tp_obj.thresholds, [0.4, 0.9])

        # Check save and restore config
        tp_obj2 = metrics.TruePositives.from_config(tp_obj.get_config())
        self.assertEqual(tp_obj2.name, "my_tp")
        self.assertLen(tp_obj2.variables, 1)
        self.assertEqual(tp_obj2.thresholds, [0.4, 0.9])

    def test_unweighted(self):
        tp_obj = metrics.TruePositives()

        y_true = np.array(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = np.array(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )

        tp_obj.update_state(y_true, y_pred)
        self.assertAllClose(7.0, tp_obj.result())

    def test_weighted(self):
        tp_obj = metrics.TruePositives()
        y_true = np.array(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = np.array(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )
        sample_weight = np.array((1.0, 1.5, 2.0, 2.5))
        result = tp_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(12.0, result)

    def test_unweighted_with_thresholds(self):
        tp_obj = metrics.TruePositives(thresholds=[0.15, 0.5, 0.85])

        y_pred = np.array(
            (
                (0.9, 0.2, 0.8, 0.1),
                (0.2, 0.9, 0.7, 0.6),
                (0.1, 0.2, 0.4, 0.3),
                (0, 1, 0.7, 0.3),
            )
        )
        y_true = np.array(
            ((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1))
        )

        tp_obj.update_state(y_true, y_pred)
        self.assertAllClose([6.0, 3.0, 1.0], tp_obj.result())

    def test_weighted_with_thresholds(self):
        tp_obj = metrics.TruePositives(thresholds=[0.15, 0.5, 0.85])

        y_pred = np.array(
            (
                (0.9, 0.2, 0.8, 0.1),
                (0.2, 0.9, 0.7, 0.6),
                (0.1, 0.2, 0.4, 0.3),
                (0, 1, 0.7, 0.3),
            )
        )
        y_true = np.array(
            ((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1))
        )
        sample_weight = 37.0

        result = tp_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose([222.0, 111.0, 37.0], result)

    def test_threshold_limit(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Threshold values must be in \[0, 1\]. Received: \[-1, 2\]",
        ):
            metrics.TruePositives(thresholds=[-1, 0.5, 2])

        with self.assertRaisesRegex(
            ValueError,
            r"Threshold values must be in \[0, 1\]. Received: \[None\]",
        ):
            metrics.TruePositives(thresholds=[None])


class PrecisionTest(testing.TestCase):
    def test_config(self):
        p_obj = metrics.Precision(
            name="my_precision", thresholds=[0.4, 0.9], top_k=15, class_id=12
        )
        self.assertEqual(p_obj.name, "my_precision")
        self.assertLen(p_obj.variables, 2)
        self.assertEqual(
            [v.name for v in p_obj.variables],
            ["true_positives", "false_positives"],
        )
        self.assertEqual(p_obj.thresholds, [0.4, 0.9])
        self.assertEqual(p_obj.top_k, 15)
        self.assertEqual(p_obj.class_id, 12)

        # Check save and restore config
        p_obj2 = metrics.Precision.from_config(p_obj.get_config())
        self.assertEqual(p_obj2.name, "my_precision")
        self.assertLen(p_obj2.variables, 2)
        self.assertEqual(p_obj2.thresholds, [0.4, 0.9])
        self.assertEqual(p_obj2.top_k, 15)
        self.assertEqual(p_obj2.class_id, 12)

    def test_unweighted(self):
        p_obj = metrics.Precision()
        y_pred = np.array([1, 0, 1, 0])
        y_true = np.array([0, 1, 1, 0])
        result = p_obj(y_true, y_pred)
        self.assertAlmostEqual(0.5, result)

    def test_unweighted_all_incorrect(self):
        p_obj = metrics.Precision(thresholds=[0.5])
        inputs = np.random.randint(0, 2, size=(100, 1))
        y_pred = np.array(inputs)
        y_true = np.array(1 - inputs)
        result = p_obj(y_true, y_pred)
        self.assertAlmostEqual(0, result)

    def test_weighted(self):
        p_obj = metrics.Precision()
        y_pred = np.array([[1, 0, 1, 0], [1, 0, 1, 0]])
        y_true = np.array([[0, 1, 1, 0], [1, 0, 0, 1]])
        result = p_obj(
            y_true,
            y_pred,
            sample_weight=np.array([[1, 2, 3, 4], [4, 3, 2, 1]]),
        )
        weighted_tp = 3.0 + 4.0
        weighted_positives = (1.0 + 3.0) + (4.0 + 2.0)
        expected_precision = weighted_tp / weighted_positives
        self.assertAlmostEqual(expected_precision, result)

    def test_div_by_zero(self):
        p_obj = metrics.Precision()
        y_pred = np.array([0, 0, 0, 0])
        y_true = np.array([0, 0, 0, 0])
        result = p_obj(y_true, y_pred)
        self.assertEqual(0, result)

    def test_unweighted_with_threshold(self):
        p_obj = metrics.Precision(thresholds=[0.5, 0.7])
        y_pred = np.array([1, 0, 0.6, 0])
        y_true = np.array([0, 1, 1, 0])
        result = p_obj(y_true, y_pred)
        self.assertAlmostEqual([0.5, 0.0], result, 0)

    def test_weighted_with_threshold(self):
        p_obj = metrics.Precision(thresholds=[0.5, 1.0])
        y_true = np.array([[0, 1], [1, 0]])
        y_pred = np.array([[1, 0], [0.6, 0]], dtype="float32")
        weights = np.array([[4, 0], [3, 1]], dtype="float32")
        result = p_obj(y_true, y_pred, sample_weight=weights)
        weighted_tp = 0 + 3.0
        weighted_positives = (0 + 3.0) + (4.0 + 0.0)
        expected_precision = weighted_tp / weighted_positives
        self.assertAlmostEqual([expected_precision, 0], result, 1e-3)

    def test_multiple_updates(self):
        p_obj = metrics.Precision(thresholds=[0.5, 1.0])
        y_true = np.array([[0, 1], [1, 0]])
        y_pred = np.array([[1, 0], [0.6, 0]], dtype="float32")
        weights = np.array([[4, 0], [3, 1]], dtype="float32")
        for _ in range(2):
            p_obj.update_state(y_true, y_pred, sample_weight=weights)

        weighted_tp = (0 + 3.0) + (0 + 3.0)
        weighted_positives = ((0 + 3.0) + (4.0 + 0.0)) + (
            (0 + 3.0) + (4.0 + 0.0)
        )
        expected_precision = weighted_tp / weighted_positives
        self.assertAlmostEqual([expected_precision, 0], p_obj.result(), 1e-3)

    def test_unweighted_top_k(self):
        p_obj = metrics.Precision(top_k=3)
        y_pred = np.array([0.2, 0.1, 0.5, 0, 0.2])
        y_true = np.array([0, 1, 1, 0, 0])
        result = p_obj(y_true, y_pred)
        self.assertAlmostEqual(1.0 / 3, result)

    def test_weighted_top_k(self):
        p_obj = metrics.Precision(top_k=3)
        y_pred1 = np.array([[0.2, 0.1, 0.4, 0, 0.2]])
        y_true1 = np.array([[0, 1, 1, 0, 1]])
        p_obj(y_true1, y_pred1, sample_weight=np.array([[1, 4, 2, 3, 5]]))

        y_pred2 = np.array([0.2, 0.6, 0.4, 0.2, 0.2])
        y_true2 = np.array([1, 0, 1, 1, 1])
        result = p_obj(y_true2, y_pred2, sample_weight=np.array(3))

        tp = (2 + 5) + (3 + 3)
        predicted_positives = (1 + 2 + 5) + (3 + 3 + 3)
        expected_precision = tp / predicted_positives
        self.assertAlmostEqual(expected_precision, result)

    def test_unweighted_class_id_should_throw_error_1d(self):
        p_obj = metrics.Precision(class_id=2)

        y_pred = np.array([0.2, 0.1, 0.6, 0, 0.2])
        y_true = np.array([0, 1, 1, 0, 0])

        with self.assertRaisesRegex(
            ValueError,
            r"When class_id is provided, y_pred must be a 2D array "
            r"with shape \(num_samples, num_classes\), found shape:.*",
        ):
            p_obj(y_true, y_pred)

    def test_unweighted_class_id_multiclass(self):
        p_obj = metrics.Precision(class_id=1)

        y_pred = np.array(
            [
                [0.1, 0.2, 0.7],
                [0.5, 0.3, 0.2],
                [0.2, 0.6, 0.2],
                [0.7, 0.2, 0.1],
                [0.1, 0.1, 0.8],
            ]
        )

        y_true = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        result = p_obj(y_true, y_pred)
        self.assertAlmostEqual(1.0, result)
        self.assertAlmostEqual(1.0, p_obj.true_positives)
        self.assertAlmostEqual(0.0, p_obj.false_positives)

    def test_unweighted_top_k_and_threshold(self):
        p_obj = metrics.Precision(thresholds=0.7, top_k=2)

        y_pred = np.array([0.2, 0.8, 0.6, 0, 0.2])
        y_true = np.array([0, 1, 1, 0, 1])
        result = p_obj(y_true, y_pred)
        self.assertAlmostEqual(1, result)
        self.assertAlmostEqual(1, p_obj.true_positives)
        self.assertAlmostEqual(0, p_obj.false_positives)


class RecallTest(testing.TestCase):
    def test_config(self):
        r_obj = metrics.Recall(
            name="my_recall", thresholds=[0.4, 0.9], top_k=15, class_id=12
        )
        self.assertEqual(r_obj.name, "my_recall")
        self.assertLen(r_obj.variables, 2)
        self.assertEqual(
            [v.name for v in r_obj.variables],
            ["true_positives", "false_negatives"],
        )
        self.assertEqual(r_obj.thresholds, [0.4, 0.9])
        self.assertEqual(r_obj.top_k, 15)
        self.assertEqual(r_obj.class_id, 12)

        # Check save and restore config
        r_obj2 = metrics.Recall.from_config(r_obj.get_config())
        self.assertEqual(r_obj2.name, "my_recall")
        self.assertLen(r_obj2.variables, 2)
        self.assertEqual(r_obj2.thresholds, [0.4, 0.9])
        self.assertEqual(r_obj2.top_k, 15)
        self.assertEqual(r_obj2.class_id, 12)

    def test_unweighted(self):
        r_obj = metrics.Recall()
        y_pred = np.array([1, 0, 1, 0])
        y_true = np.array([0, 1, 1, 0])
        self.assertAlmostEqual(0.5, r_obj(y_true, y_pred))

    def test_unweighted_all_incorrect(self):
        r_obj = metrics.Recall(thresholds=[0.5])
        inputs = np.random.randint(0, 2, size=(100, 1))
        y_pred = np.array(inputs)
        y_true = np.array(1 - inputs)
        self.assertAlmostEqual(0, r_obj(y_true, y_pred))

    def test_weighted(self):
        r_obj = metrics.Recall()
        y_pred = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
        y_true = np.array([[0, 1, 1, 0], [1, 0, 0, 1]])
        result = r_obj(
            y_true,
            y_pred,
            sample_weight=np.array([[1, 2, 3, 4], [4, 3, 2, 1]]),
        )
        weighted_tp = 3.0 + 1.0
        weighted_t = (2.0 + 3.0) + (4.0 + 1.0)
        expected_recall = weighted_tp / weighted_t
        self.assertAlmostEqual(expected_recall, result)

    def test_div_by_zero(self):
        r_obj = metrics.Recall()
        y_pred = np.array([0, 0, 0, 0])
        y_true = np.array([0, 0, 0, 0])
        self.assertEqual(0, r_obj(y_true, y_pred))

    def test_unweighted_with_threshold(self):
        r_obj = metrics.Recall(thresholds=[0.5, 0.7])
        y_pred = np.array([1, 0, 0.6, 0])
        y_true = np.array([0, 1, 1, 0])
        self.assertAllClose([0.5, 0.0], r_obj(y_true, y_pred), 0)

    def test_weighted_with_threshold(self):
        r_obj = metrics.Recall(thresholds=[0.5, 1.0])
        y_true = np.array([[0, 1], [1, 0]])
        y_pred = np.array([[1, 0], [0.6, 0]], dtype="float32")
        weights = np.array([[1, 4], [3, 2]], dtype="float32")
        result = r_obj(y_true, y_pred, sample_weight=weights)
        weighted_tp = 0 + 3.0
        weighted_positives = (0 + 3.0) + (4.0 + 0.0)
        expected_recall = weighted_tp / weighted_positives
        self.assertAllClose([expected_recall, 0], result, 1e-3)

    def test_multiple_updates(self):
        r_obj = metrics.Recall(thresholds=[0.5, 1.0])
        y_true = np.array([[0, 1], [1, 0]])
        y_pred = np.array([[1, 0], [0.6, 0]], dtype="float32")
        weights = np.array([[1, 4], [3, 2]], dtype="float32")
        for _ in range(2):
            r_obj.update_state(y_true, y_pred, sample_weight=weights)

        weighted_tp = (0 + 3.0) + (0 + 3.0)
        weighted_positives = ((0 + 3.0) + (4.0 + 0.0)) + (
            (0 + 3.0) + (4.0 + 0.0)
        )
        expected_recall = weighted_tp / weighted_positives
        self.assertAllClose([expected_recall, 0], r_obj.result(), 1e-3)

    def test_unweighted_top_k(self):
        r_obj = metrics.Recall(top_k=3)
        y_pred = np.array([0.2, 0.1, 0.5, 0, 0.2])
        y_true = np.array([0, 1, 1, 0, 0])
        self.assertAlmostEqual(0.5, r_obj(y_true, y_pred))

    def test_weighted_top_k(self):
        r_obj = metrics.Recall(top_k=3)
        y_pred1 = np.array([[0.2, 0.1, 0.4, 0, 0.2]])
        y_true1 = np.array([[0, 1, 1, 0, 1]])
        r_obj(y_true1, y_pred1, sample_weight=np.array([[1, 4, 2, 3, 5]]))

        y_pred2 = np.array([0.2, 0.6, 0.4, 0.2, 0.2])
        y_true2 = np.array([1, 0, 1, 1, 1])
        result = r_obj(y_true2, y_pred2, sample_weight=np.array(3))

        tp = (2 + 5) + (3 + 3)
        positives = (4 + 2 + 5) + (3 + 3 + 3 + 3)
        expected_recall = tp / positives
        self.assertAlmostEqual(expected_recall, result)

    def test_unweighted_class_id_should_throw_error_1d(self):
        r_obj = metrics.Recall(class_id=2)

        y_pred = np.array([0.2, 0.1, 0.6, 0, 0.2])
        y_true = np.array([0, 1, 1, 0, 0])

        with self.assertRaisesRegex(
            ValueError,
            r"When class_id is provided, y_pred must be a 2D array "
            r"with shape \(num_samples, num_classes\), found shape:.*",
        ):
            r_obj(y_true, y_pred)

    def test_unweighted_class_id_multiclass(self):
        r_obj = metrics.Recall(class_id=1)

        y_pred = np.array(
            [
                [0.1, 0.2, 0.7],
                [0.5, 0.3, 0.2],
                [0.2, 0.6, 0.2],
                [0.7, 0.2, 0.1],
                [0.1, 0.1, 0.8],
            ]
        )

        y_true = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        result = r_obj(y_true, y_pred)
        self.assertAlmostEqual(1.0, result)
        self.assertAlmostEqual(1.0, r_obj.true_positives)
        self.assertAlmostEqual(0.0, r_obj.false_negatives)

    def test_unweighted_top_k_and_threshold(self):
        r_obj = metrics.Recall(thresholds=0.7, top_k=2)

        y_pred = np.array([0.2, 0.8, 0.6, 0, 0.2])
        y_true = np.array([1, 1, 1, 0, 1])
        self.assertAlmostEqual(0.25, r_obj(y_true, y_pred))
        self.assertAlmostEqual(1, r_obj.true_positives)
        self.assertAlmostEqual(3, r_obj.false_negatives)


class SensitivityAtSpecificityTest(testing.TestCase, parameterized.TestCase):
    def test_config(self):
        s_obj = metrics.SensitivityAtSpecificity(
            0.4,
            num_thresholds=100,
            class_id=12,
            name="sensitivity_at_specificity_1",
        )
        self.assertEqual(s_obj.name, "sensitivity_at_specificity_1")
        self.assertLen(s_obj.variables, 4)
        self.assertEqual(s_obj.specificity, 0.4)
        self.assertEqual(s_obj.num_thresholds, 100)
        self.assertEqual(s_obj.class_id, 12)

        # Check save and restore config
        s_obj2 = metrics.SensitivityAtSpecificity.from_config(
            s_obj.get_config()
        )
        self.assertEqual(s_obj2.name, "sensitivity_at_specificity_1")
        self.assertLen(s_obj2.variables, 4)
        self.assertEqual(s_obj2.specificity, 0.4)
        self.assertEqual(s_obj2.num_thresholds, 100)
        self.assertEqual(s_obj.class_id, 12)

    def test_unweighted_all_correct(self):
        s_obj = metrics.SensitivityAtSpecificity(0.7)
        inputs = np.random.randint(0, 2, size=(100, 1))
        y_pred = np.array(inputs, dtype="float32")
        y_true = np.array(inputs)
        self.assertAlmostEqual(1, s_obj(y_true, y_pred))

    def test_unweighted_high_specificity(self):
        s_obj = metrics.SensitivityAtSpecificity(0.8)
        pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.1, 0.45, 0.5, 0.8, 0.9]
        label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        y_pred = np.array(pred_values, dtype="float32")
        y_true = np.array(label_values)

        self.assertAlmostEqual(0.8, s_obj(y_true, y_pred))

    def test_unweighted_low_specificity(self):
        s_obj = metrics.SensitivityAtSpecificity(0.4)
        pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.25, 0.26, 0.26]
        label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        y_pred = np.array(pred_values, dtype="float32")
        y_true = np.array(label_values)

        self.assertAlmostEqual(0.6, s_obj(y_true, y_pred))

    def test_unweighted_class_id(self):
        s_obj = metrics.SpecificityAtSensitivity(0.4, class_id=2)
        pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.25, 0.26, 0.26]
        label_values = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2]

        y_pred = ops.transpose(np.array([pred_values] * 3))
        y_true = ops.one_hot(np.array(label_values), num_classes=3)

        self.assertAlmostEqual(0.6, s_obj(y_true, y_pred))

    @parameterized.parameters(["bool", "int32", "float32"])
    def test_weighted(self, label_dtype):
        s_obj = metrics.SensitivityAtSpecificity(0.4)
        pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.25, 0.26, 0.26]
        label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        weight_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        y_pred = np.array(pred_values, dtype="float32")
        y_true = ops.cast(label_values, dtype=label_dtype)
        weights = np.array(weight_values)

        result = s_obj(y_true, y_pred, sample_weight=weights)
        self.assertAlmostEqual(0.675, result)

    def test_invalid_specificity(self):
        with self.assertRaisesRegex(
            ValueError, r"`specificity` must be in the range \[0, 1\]."
        ):
            metrics.SensitivityAtSpecificity(-1)

    def test_invalid_num_thresholds(self):
        with self.assertRaisesRegex(
            ValueError, "Argument `num_thresholds` must be an integer > 0"
        ):
            metrics.SensitivityAtSpecificity(0.4, num_thresholds=-1)


class SpecificityAtSensitivityTest(testing.TestCase, parameterized.TestCase):
    def test_config(self):
        s_obj = metrics.SpecificityAtSensitivity(
            0.4,
            num_thresholds=100,
            class_id=12,
            name="specificity_at_sensitivity_1",
        )
        self.assertEqual(s_obj.name, "specificity_at_sensitivity_1")
        self.assertLen(s_obj.variables, 4)
        self.assertEqual(s_obj.sensitivity, 0.4)
        self.assertEqual(s_obj.num_thresholds, 100)
        self.assertEqual(s_obj.class_id, 12)

        # Check save and restore config
        s_obj2 = metrics.SpecificityAtSensitivity.from_config(
            s_obj.get_config()
        )
        self.assertEqual(s_obj2.name, "specificity_at_sensitivity_1")
        self.assertLen(s_obj2.variables, 4)
        self.assertEqual(s_obj2.sensitivity, 0.4)
        self.assertEqual(s_obj2.num_thresholds, 100)
        self.assertEqual(s_obj.class_id, 12)

    def test_unweighted_all_correct(self):
        s_obj = metrics.SpecificityAtSensitivity(0.7)
        inputs = np.random.randint(0, 2, size=(100, 1))
        y_pred = np.array(inputs, dtype="float32")
        y_true = np.array(inputs)

        self.assertAlmostEqual(1, s_obj(y_true, y_pred))

    def test_unweighted_high_sensitivity(self):
        s_obj = metrics.SpecificityAtSensitivity(1.0)
        pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.25, 0.26, 0.26]
        label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        y_pred = np.array(pred_values, dtype="float32")
        y_true = np.array(label_values)

        self.assertAlmostEqual(0.2, s_obj(y_true, y_pred))

    def test_unweighted_low_sensitivity(self):
        s_obj = metrics.SpecificityAtSensitivity(0.4)
        pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.25, 0.26, 0.26]
        label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        y_pred = np.array(pred_values, dtype="float32")
        y_true = np.array(label_values)

        self.assertAlmostEqual(0.6, s_obj(y_true, y_pred))

    def test_unweighted_class_id(self):
        s_obj = metrics.SpecificityAtSensitivity(0.4, class_id=2)
        pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.25, 0.26, 0.26]
        label_values = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2]

        y_pred = ops.transpose(np.array([pred_values] * 3))
        y_true = ops.one_hot(np.array(label_values), num_classes=3)

        self.assertAlmostEqual(0.6, s_obj(y_true, y_pred))

    @parameterized.parameters(["bool", "int32", "float32"])
    def test_weighted(self, label_dtype):
        s_obj = metrics.SpecificityAtSensitivity(0.4)
        pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.25, 0.26, 0.26]
        label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        weight_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        y_pred = np.array(pred_values, dtype="float32")
        y_true = ops.cast(label_values, dtype=label_dtype)
        weights = np.array(weight_values)

        result = s_obj(y_true, y_pred, sample_weight=weights)
        self.assertAlmostEqual(0.4, result)

    def test_invalid_sensitivity(self):
        with self.assertRaisesRegex(
            ValueError, r"`sensitivity` must be in the range \[0, 1\]."
        ):
            metrics.SpecificityAtSensitivity(-1)

    def test_invalid_num_thresholds(self):
        with self.assertRaisesRegex(
            ValueError, "Argument `num_thresholds` must be an integer > 0"
        ):
            metrics.SpecificityAtSensitivity(0.4, num_thresholds=-1)


class PrecisionAtRecallTest(testing.TestCase, parameterized.TestCase):
    def test_config(self):
        s_obj = metrics.PrecisionAtRecall(
            0.4, num_thresholds=100, class_id=12, name="precision_at_recall_1"
        )
        self.assertEqual(s_obj.name, "precision_at_recall_1")
        self.assertLen(s_obj.variables, 4)
        self.assertEqual(s_obj.recall, 0.4)
        self.assertEqual(s_obj.num_thresholds, 100)
        self.assertEqual(s_obj.class_id, 12)

        # Check save and restore config
        s_obj2 = metrics.PrecisionAtRecall.from_config(s_obj.get_config())
        self.assertEqual(s_obj2.name, "precision_at_recall_1")
        self.assertLen(s_obj2.variables, 4)
        self.assertEqual(s_obj2.recall, 0.4)
        self.assertEqual(s_obj2.num_thresholds, 100)
        self.assertEqual(s_obj.class_id, 12)

    def test_unweighted_all_correct(self):
        s_obj = metrics.PrecisionAtRecall(0.7)
        inputs = np.random.randint(0, 2, size=(100, 1))
        y_pred = np.array(inputs, dtype="float32")
        y_true = np.array(inputs)

        self.assertAlmostEqual(1, s_obj(y_true, y_pred))

    def test_unweighted_high_recall(self):
        s_obj = metrics.PrecisionAtRecall(0.8)
        pred_values = [0.0, 0.1, 0.2, 0.5, 0.6, 0.2, 0.5, 0.6, 0.8, 0.9]
        label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        y_pred = np.array(pred_values, dtype="float32")
        y_true = np.array(label_values)

        # For 0.5 < decision threshold < 0.6.
        self.assertAlmostEqual(2.0 / 3, s_obj(y_true, y_pred))

    def test_unweighted_low_recall(self):
        s_obj = metrics.PrecisionAtRecall(0.6)
        pred_values = [0.0, 0.1, 0.2, 0.5, 0.6, 0.2, 0.5, 0.6, 0.8, 0.9]
        label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        y_pred = np.array(pred_values, dtype="float32")
        y_true = np.array(label_values)

        # For 0.2 < decision threshold < 0.5.
        self.assertAlmostEqual(0.75, s_obj(y_true, y_pred))

    def test_unweighted_class_id(self):
        s_obj = metrics.PrecisionAtRecall(0.6, class_id=2)
        pred_values = [0.0, 0.1, 0.2, 0.5, 0.6, 0.2, 0.5, 0.6, 0.8, 0.9]
        label_values = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2]

        y_pred = ops.transpose(np.array([pred_values] * 3))
        y_true = ops.one_hot(np.array(label_values), num_classes=3)

        # For 0.2 < decision threshold < 0.5.
        self.assertAlmostEqual(0.75, s_obj(y_true, y_pred))

    @parameterized.parameters(["bool", "int32", "float32"])
    def test_weighted(self, label_dtype):
        s_obj = metrics.PrecisionAtRecall(7.0 / 8)
        pred_values = [0.0, 0.1, 0.2, 0.5, 0.6, 0.2, 0.5, 0.6, 0.8, 0.9]
        label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        weight_values = [2, 1, 2, 1, 2, 1, 2, 2, 1, 2]

        y_pred = np.array(pred_values, dtype="float32")
        y_true = ops.cast(label_values, dtype=label_dtype)
        weights = np.array(weight_values)

        result = s_obj(y_true, y_pred, sample_weight=weights)
        # For 0.0 < decision threshold < 0.2.
        self.assertAlmostEqual(0.7, result)

    def test_invalid_sensitivity(self):
        with self.assertRaisesRegex(
            ValueError, r"`recall` must be in the range \[0, 1\]."
        ):
            metrics.PrecisionAtRecall(-1)

    def test_invalid_num_thresholds(self):
        with self.assertRaisesRegex(
            ValueError, "Argument `num_thresholds` must be an integer > 0"
        ):
            metrics.PrecisionAtRecall(0.4, num_thresholds=-1)


class RecallAtPrecisionTest(testing.TestCase, parameterized.TestCase):
    def test_config(self):
        s_obj = metrics.RecallAtPrecision(
            0.4, num_thresholds=100, class_id=12, name="recall_at_precision_1"
        )
        self.assertEqual(s_obj.name, "recall_at_precision_1")
        self.assertLen(s_obj.variables, 4)
        self.assertEqual(s_obj.precision, 0.4)
        self.assertEqual(s_obj.num_thresholds, 100)
        self.assertEqual(s_obj.class_id, 12)

        # Check save and restore config
        s_obj2 = metrics.RecallAtPrecision.from_config(s_obj.get_config())
        self.assertEqual(s_obj2.name, "recall_at_precision_1")
        self.assertLen(s_obj2.variables, 4)
        self.assertEqual(s_obj2.precision, 0.4)
        self.assertEqual(s_obj2.num_thresholds, 100)
        self.assertEqual(s_obj.class_id, 12)

    def test_unweighted_all_correct(self):
        s_obj = metrics.RecallAtPrecision(0.7)
        inputs = np.random.randint(0, 2, size=(100, 1))
        y_pred = np.array(inputs, dtype="float32")
        y_true = np.array(inputs)

        self.assertAlmostEqual(1, s_obj(y_true, y_pred))

    def test_unweighted_high_precision(self):
        s_obj = metrics.RecallAtPrecision(0.75)
        pred_values = [
            0.05,
            0.1,
            0.2,
            0.3,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.6,
            0.9,
            0.95,
        ]
        label_values = [0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1]
        # precisions: [1/2, 6/11, 1/2, 5/9, 5/8, 5/7, 2/3, 3/5, 3/5, 2/3, 1/2,
        # 1].
        # recalls:    [1,   1,    5/6, 5/6, 5/6, 5/6, 2/3, 1/2, 1/2, 1/3, 1/6,
        # 1/6].
        y_pred = np.array(pred_values, dtype="float32")
        y_true = np.array(label_values)

        # The precision 0.75 can be reached at thresholds 0.4<=t<0.45.
        self.assertAlmostEqual(0.5, s_obj(y_true, y_pred))

    def test_unweighted_low_precision(self):
        s_obj = metrics.RecallAtPrecision(2.0 / 3)
        pred_values = [
            0.05,
            0.1,
            0.2,
            0.3,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.6,
            0.9,
            0.95,
        ]
        label_values = [0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1]
        # precisions: [1/2, 6/11, 1/2, 5/9, 5/8, 5/7, 2/3, 3/5, 3/5, 2/3, 1/2,
        # 1].
        # recalls:    [1,   1,    5/6, 5/6, 5/6, 5/6, 2/3, 1/2, 1/2, 1/3, 1/6,
        # 1/6].
        y_pred = np.array(pred_values, dtype="float32")
        y_true = np.array(label_values)

        # The precision 5/7 can be reached at thresholds 00.3<=t<0.35.
        self.assertAlmostEqual(5.0 / 6, s_obj(y_true, y_pred))

    def test_unweighted_class_id(self):
        s_obj = metrics.RecallAtPrecision(2.0 / 3, class_id=2)
        pred_values = [
            0.05,
            0.1,
            0.2,
            0.3,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.6,
            0.9,
            0.95,
        ]
        label_values = [0, 2, 0, 0, 0, 2, 2, 0, 2, 2, 0, 2]
        # precisions: [1/2, 6/11, 1/2, 5/9, 5/8, 5/7, 2/3, 3/5, 3/5, 2/3, 1/2,
        # 1].
        # recalls:    [1,   1,    5/6, 5/6, 5/6, 5/6, 2/3, 1/2, 1/2, 1/3, 1/6,
        # 1/6].
        y_pred = ops.transpose(np.array([pred_values] * 3))
        y_true = ops.one_hot(np.array(label_values), num_classes=3)

        # The precision 5/7 can be reached at thresholds 00.3<=t<0.35.
        self.assertAlmostEqual(5.0 / 6, s_obj(y_true, y_pred))

    @parameterized.parameters(["bool", "int32", "float32"])
    def test_weighted(self, label_dtype):
        s_obj = metrics.RecallAtPrecision(0.75)
        pred_values = [0.1, 0.2, 0.3, 0.5, 0.6, 0.9, 0.9]
        label_values = [0, 1, 0, 0, 0, 1, 1]
        weight_values = [1, 2, 1, 2, 1, 2, 1]
        y_pred = np.array(pred_values, dtype="float32")
        y_true = ops.cast(label_values, dtype=label_dtype)
        weights = np.array(weight_values)

        result = s_obj(y_true, y_pred, sample_weight=weights)
        self.assertAlmostEqual(0.6, result)

    def test_unachievable_precision(self):
        s_obj = metrics.RecallAtPrecision(2.0 / 3)
        pred_values = [0.1, 0.2, 0.3, 0.9]
        label_values = [1, 1, 0, 0]
        y_pred = np.array(pred_values, dtype="float32")
        y_true = np.array(label_values)

        # The highest possible precision is 1/2 which is below the required
        # value, expect 0 recall.
        self.assertAlmostEqual(0, s_obj(y_true, y_pred))

    def test_invalid_sensitivity(self):
        with self.assertRaisesRegex(
            ValueError, r"`precision` must be in the range \[0, 1\]."
        ):
            metrics.RecallAtPrecision(-1)

    def test_invalid_num_thresholds(self):
        with self.assertRaisesRegex(
            ValueError, "Argument `num_thresholds` must be an integer > 0"
        ):
            metrics.RecallAtPrecision(0.4, num_thresholds=-1)

    @pytest.mark.requires_trainable_backend
    def test_end_to_end(self):
        # Test for https://github.com/keras-team/keras/issues/718
        model = models.Sequential(
            [
                layers.Input((1,)),
                layers.Dense(1),
            ]
        )
        model.compile(
            optimizer="rmsprop", loss="mse", metrics=[metrics.Precision()]
        )
        model.fit(np.ones((5, 1)), np.ones((5, 1)))


class AUCTest(testing.TestCase):
    def setUp(self):
        self.num_thresholds = 3
        self.y_pred = np.array([0, 0.5, 0.3, 0.9], dtype="float32")
        self.y_pred_multi_label = np.array(
            [[0.0, 0.4], [0.5, 0.7], [0.3, 0.2], [0.9, 0.3]], dtype="float32"
        )
        epsilon = 1e-12
        self.y_pred_logits = -ops.log(1.0 / (self.y_pred + epsilon) - 1.0)
        self.y_true = np.array([0, 0, 1, 1])
        self.y_true_multi_label = np.array([[0, 0], [1, 1], [1, 1], [1, 0]])
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
            curve="PR",
            summation_method="majoring",
            name="auc_1",
            dtype="float64",
            multi_label=True,
            num_labels=2,
            from_logits=True,
        )
        auc_obj.update_state(self.y_true_multi_label, self.y_pred_multi_label)
        self.assertEqual(auc_obj.name, "auc_1")
        self.assertEqual(auc_obj._dtype, "float64")
        self.assertLen(auc_obj.variables, 4)
        self.assertEqual(auc_obj.num_thresholds, 100)
        self.assertEqual(auc_obj.curve, metrics_utils.AUCCurve.PR)
        self.assertEqual(
            auc_obj.summation_method, metrics_utils.AUCSummationMethod.MAJORING
        )
        self.assertTrue(auc_obj.multi_label)
        self.assertEqual(auc_obj.num_labels, 2)
        self.assertTrue(auc_obj._from_logits)
        old_config = auc_obj.get_config()
        self.assertNotIn("thresholds", old_config)
        self.assertDictEqual(old_config, json.loads(json.dumps(old_config)))

        # Check save and restore config.
        auc_obj2 = metrics.AUC.from_config(auc_obj.get_config())
        auc_obj2.update_state(self.y_true_multi_label, self.y_pred_multi_label)
        self.assertEqual(auc_obj2.name, "auc_1")
        self.assertLen(auc_obj2.variables, 4)
        self.assertEqual(auc_obj2.num_thresholds, 100)
        self.assertEqual(auc_obj2.curve, metrics_utils.AUCCurve.PR)
        self.assertEqual(
            auc_obj2.summation_method, metrics_utils.AUCSummationMethod.MAJORING
        )
        self.assertTrue(auc_obj2.multi_label)
        self.assertEqual(auc_obj2.num_labels, 2)
        self.assertTrue(auc_obj2._from_logits)
        new_config = auc_obj2.get_config()
        self.assertNotIn("thresholds", new_config)
        self.assertDictEqual(old_config, new_config)
        self.assertAllClose(auc_obj.thresholds, auc_obj2.thresholds)

    def test_config_manual_thresholds(self):
        auc_obj = metrics.AUC(
            num_thresholds=None,
            curve="PR",
            summation_method="majoring",
            name="auc_1",
            thresholds=[0.3, 0.5],
        )
        auc_obj.update_state(self.y_true, self.y_pred)
        self.assertEqual(auc_obj.name, "auc_1")
        self.assertLen(auc_obj.variables, 4)
        self.assertEqual(auc_obj.num_thresholds, 4)
        self.assertAllClose(auc_obj.thresholds, [0.0, 0.3, 0.5, 1.0])
        self.assertEqual(auc_obj.curve, metrics_utils.AUCCurve.PR)
        self.assertEqual(
            auc_obj.summation_method, metrics_utils.AUCSummationMethod.MAJORING
        )
        old_config = auc_obj.get_config()
        self.assertDictEqual(old_config, json.loads(json.dumps(old_config)))

        # Check save and restore config.
        auc_obj2 = metrics.AUC.from_config(auc_obj.get_config())
        auc_obj2.update_state(self.y_true, self.y_pred)
        self.assertEqual(auc_obj2.name, "auc_1")
        self.assertLen(auc_obj2.variables, 4)
        self.assertEqual(auc_obj2.num_thresholds, 4)
        self.assertEqual(auc_obj2.curve, metrics_utils.AUCCurve.PR)
        self.assertEqual(
            auc_obj2.summation_method, metrics_utils.AUCSummationMethod.MAJORING
        )
        new_config = auc_obj2.get_config()
        self.assertDictEqual(old_config, new_config)
        self.assertAllClose(auc_obj.thresholds, auc_obj2.thresholds)

    def test_unweighted_all_correct(self):
        auc_obj = metrics.AUC()
        self.assertEqual(auc_obj(self.y_true, self.y_true), 1)

    def test_unweighted(self):
        auc_obj = metrics.AUC(num_thresholds=self.num_thresholds)
        result = auc_obj(self.y_true, self.y_pred)

        # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
        # recall = [2/2, 1/(1+1), 0] = [1, 0.5, 0]
        # fp_rate = [2/2, 0, 0] = [1, 0, 0]
        # heights = [(1 + 0.5)/2, (0.5 + 0)/2] = [0.75, 0.25]
        # widths = [(1 - 0), (0 - 0)] = [1, 0]
        expected_result = 0.75 * 1 + 0.25 * 0
        self.assertAllClose(result, expected_result, 1e-3)

    def test_unweighted_from_logits(self):
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds, from_logits=True
        )
        result = auc_obj(self.y_true, self.y_pred_logits)

        # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
        # recall = [2/2, 1/(1+1), 0] = [1, 0.5, 0]
        # fp_rate = [2/2, 0, 0] = [1, 0, 0]
        # heights = [(1 + 0.5)/2, (0.5 + 0)/2] = [0.75, 0.25]
        # widths = [(1 - 0), (0 - 0)] = [1, 0]
        expected_result = 0.75 * 1 + 0.25 * 0
        self.assertAllClose(result, expected_result, 1e-3)

    def test_manual_thresholds(self):
        # Verify that when specified, thresholds are used instead of
        # num_thresholds.
        auc_obj = metrics.AUC(num_thresholds=2, thresholds=[0.5])
        self.assertEqual(auc_obj.num_thresholds, 3)
        self.assertAllClose(auc_obj.thresholds, [0.0, 0.5, 1.0])
        result = auc_obj(self.y_true, self.y_pred)

        # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
        # recall = [2/2, 1/(1+1), 0] = [1, 0.5, 0]
        # fp_rate = [2/2, 0, 0] = [1, 0, 0]
        # heights = [(1 + 0.5)/2, (0.5 + 0)/2] = [0.75, 0.25]
        # widths = [(1 - 0), (0 - 0)] = [1, 0]
        expected_result = 0.75 * 1 + 0.25 * 0
        self.assertAllClose(result, expected_result, 1e-3)

    def test_weighted_roc_interpolation(self):
        auc_obj = metrics.AUC(num_thresholds=self.num_thresholds)
        result = auc_obj(
            self.y_true, self.y_pred, sample_weight=self.sample_weight
        )

        # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
        # recall = [7/7, 4/(4+3), 0] = [1, 0.571, 0]
        # fp_rate = [3/3, 0, 0] = [1, 0, 0]
        # heights = [(1 + 0.571)/2, (0.571 + 0)/2] = [0.7855, 0.2855]
        # widths = [(1 - 0), (0 - 0)] = [1, 0]
        expected_result = 0.7855 * 1 + 0.2855 * 0
        self.assertAllClose(result, expected_result, 1e-3)

    def test_weighted_roc_majoring(self):
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds, summation_method="majoring"
        )
        result = auc_obj(
            self.y_true, self.y_pred, sample_weight=self.sample_weight
        )

        # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
        # recall = [7/7, 4/(4+3), 0] = [1, 0.571, 0]
        # fp_rate = [3/3, 0, 0] = [1, 0, 0]
        # heights = [max(1, 0.571), max(0.571, 0)] = [1, 0.571]
        # widths = [(1 - 0), (0 - 0)] = [1, 0]
        expected_result = 1 * 1 + 0.571 * 0
        self.assertAllClose(result, expected_result, 1e-3)

    def test_weighted_roc_minoring(self):
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds, summation_method="minoring"
        )
        result = auc_obj(
            self.y_true, self.y_pred, sample_weight=self.sample_weight
        )

        # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
        # recall = [7/7, 4/(4+3), 0] = [1, 0.571, 0]
        # fp_rate = [3/3, 0, 0] = [1, 0, 0]
        # heights = [min(1, 0.571), min(0.571, 0)] = [0.571, 0]
        # widths = [(1 - 0), (0 - 0)] = [1, 0]
        expected_result = 0.571 * 1 + 0 * 0
        self.assertAllClose(result, expected_result, 1e-3)

    def test_weighted_pr_majoring(self):
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds,
            curve="PR",
            summation_method="majoring",
        )
        result = auc_obj(
            self.y_true, self.y_pred, sample_weight=self.sample_weight
        )

        # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
        # precision = [7/(7+3), 4/4, 0] = [0.7, 1, 0]
        # recall = [7/7, 4/(4+3), 0] = [1, 0.571, 0]
        # heights = [max(0.7, 1), max(1, 0)] = [1, 1]
        # widths = [(1 - 0.571), (0.571 - 0)] = [0.429, 0.571]
        expected_result = 1 * 0.429 + 1 * 0.571
        self.assertAllClose(result, expected_result, 1e-3)

    def test_weighted_pr_minoring(self):
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds,
            curve="PR",
            summation_method="minoring",
        )
        result = auc_obj(
            self.y_true, self.y_pred, sample_weight=self.sample_weight
        )

        # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
        # precision = [7/(7+3), 4/4, 0] = [0.7, 1, 0]
        # recall = [7/7, 4/(4+3), 0] = [1, 0.571, 0]
        # heights = [min(0.7, 1), min(1, 0)] = [0.7, 0]
        # widths = [(1 - 0.571), (0.571 - 0)] = [0.429, 0.571]
        expected_result = 0.7 * 0.429 + 0 * 0.571
        self.assertAllClose(result, expected_result, 1e-3)

    def test_weighted_pr_interpolation(self):
        auc_obj = metrics.AUC(num_thresholds=self.num_thresholds, curve="PR")
        result = auc_obj(
            self.y_true, self.y_pred, sample_weight=self.sample_weight
        )

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
        expected_result = 2.416 / 7 + 4 / 7
        self.assertAllClose(result, expected_result, 1e-3)

    def test_weighted_pr_interpolation_negative_weights(self):
        auc_obj = metrics.AUC(num_thresholds=self.num_thresholds, curve="PR")
        sample_weight = [-1, -2, -3, -4]
        result = auc_obj(self.y_true, self.y_pred, sample_weight=sample_weight)

        # Divisor in auc formula is max(tp[1:]+fn[1:], 0), which is all zeros
        # because the all values in tp and fn are negative, divide_no_nan will
        # produce all zeros.
        self.assertAllClose(result, 0.0, 1e-3)

    def test_invalid_num_thresholds(self):
        with self.assertRaisesRegex(
            ValueError, "Argument `num_thresholds` must be an integer > 1"
        ):
            metrics.AUC(num_thresholds=-1)

        with self.assertRaisesRegex(
            ValueError, "Argument `num_thresholds` must be an integer > 1."
        ):
            metrics.AUC(num_thresholds=1)

    def test_invalid_curve(self):
        with self.assertRaisesRegex(
            ValueError, 'Invalid AUC curve value: "Invalid".'
        ):
            metrics.AUC(curve="Invalid")

    def test_invalid_summation_method(self):
        with self.assertRaisesRegex(
            ValueError, 'Invalid AUC summation method value: "Invalid".'
        ):
            metrics.AUC(summation_method="Invalid")

    def test_extra_dims(self):
        try:
            from scipy import special

            logits = special.expit(
                -np.array(
                    [
                        [[-10.0, 10.0, -10.0], [10.0, -10.0, 10.0]],
                        [[-12.0, 12.0, -12.0], [12.0, -12.0, 12.0]],
                    ],
                    dtype=np.float32,
                )
            )
            labels = np.array(
                [[[1, 0, 0], [1, 0, 0]], [[0, 1, 1], [0, 1, 1]]], dtype=np.int64
            )
            auc_obj = metrics.AUC()
            result = auc_obj(labels, logits)
            self.assertEqual(result, 0.5)
        except ImportError as e:
            logging.warning(f"Cannot test special functions: {str(e)}")


class MultiAUCTest(testing.TestCase):
    def setUp(self):
        self.num_thresholds = 5
        self.y_pred = np.array(
            [[0, 0.5, 0.3, 0.9], [0.1, 0.2, 0.3, 0.4]], dtype="float32"
        ).T

        epsilon = 1e-12
        self.y_pred_logits = -ops.log(1.0 / (self.y_pred + epsilon) - 1.0)

        self.y_true_good = np.array([[0, 0, 1, 1], [0, 0, 1, 1]]).T
        self.y_true_bad = np.array([[0, 0, 1, 1], [1, 1, 0, 0]]).T
        self.sample_weight = [1, 2, 3, 4]

        # threshold values are [0 - 1e-7, 0.25, 0.5, 0.75, 1 + 1e-7]
        # y_pred when threshold = 0 - 1e-7   : [[1, 1, 1, 1], [1, 1, 1, 1]]
        # y_pred when threshold = 0.25       : [[0, 1, 1, 1], [0, 0, 1, 1]]
        # y_pred when threshold = 0.5        : [[0, 0, 0, 1], [0, 0, 0, 0]]
        # y_pred when threshold = 0.75       : [[0, 0, 0, 1], [0, 0, 0, 0]]
        # y_pred when threshold = 1 + 1e-7   : [[0, 0, 0, 0], [0, 0, 0, 0]]

        # for y_true_good, over thresholds:
        # tp = [[2, 2, 1, 1, 0], [2, 2, 0, 0, 0]]
        # fp = [[2, 1, 0, 0 , 0], [2, 0, 0 ,0, 0]]
        # fn = [[0, 0, 1, 1, 2], [0, 0, 2, 2, 2]]
        # tn = [[0, 1, 2, 2, 2], [0, 2, 2, 2, 2]]

        # tpr = [[1, 1, 0.5, 0.5, 0], [1, 1, 0, 0, 0]]
        # fpr = [[1, 0.5, 0, 0, 0], [1, 0, 0, 0, 0]]

        # for y_true_bad:
        # tp = [[2, 2, 1, 1, 0], [2, 0, 0, 0, 0]]
        # fp = [[2, 1, 0, 0 , 0], [2, 2, 0 ,0, 0]]
        # fn = [[0, 0, 1, 1, 2], [0, 2, 2, 2, 2]]
        # tn = [[0, 1, 2, 2, 2], [0, 0, 2, 2, 2]]

        # tpr = [[1, 1, 0.5, 0.5, 0], [1, 0, 0, 0, 0]]
        # fpr = [[1, 0.5, 0, 0, 0], [1, 1, 0, 0, 0]]

        # for y_true_good with sample_weights:

        # tp = [[7, 7, 4, 4, 0], [7, 7, 0, 0, 0]]
        # fp = [[3, 2, 0, 0, 0], [3, 0, 0, 0, 0]]
        # fn = [[0, 0, 3, 3, 7], [0, 0, 7, 7, 7]]
        # tn = [[0, 1, 3, 3, 3], [0, 3, 3, 3, 3]]

        # tpr = [[1, 1,    0.57, 0.57, 0], [1, 1, 0, 0, 0]]
        # fpr = [[1, 0.67, 0,    0,    0], [1, 0, 0, 0, 0]]

    def test_unweighted_all_correct(self):
        auc_obj = metrics.AUC(multi_label=True)
        result = auc_obj(self.y_true_good, self.y_true_good)
        self.assertEqual(result, 1)

    def test_unweighted_all_correct_flat(self):
        auc_obj = metrics.AUC(multi_label=False)
        result = auc_obj(self.y_true_good, self.y_true_good)
        self.assertEqual(result, 1)

    def test_unweighted(self):
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds, multi_label=True
        )
        result = auc_obj(self.y_true_good, self.y_pred)

        # tpr = [[1, 1, 0.5, 0.5, 0], [1, 1, 0, 0, 0]]
        # fpr = [[1, 0.5, 0, 0, 0], [1, 0, 0, 0, 0]]
        expected_result = (0.875 + 1.0) / 2.0
        self.assertAllClose(result, expected_result, 1e-3)

    def test_unweighted_from_logits(self):
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds,
            multi_label=True,
            from_logits=True,
        )
        result = auc_obj(self.y_true_good, self.y_pred_logits)

        # tpr = [[1, 1, 0.5, 0.5, 0], [1, 1, 0, 0, 0]]
        # fpr = [[1, 0.5, 0, 0, 0], [1, 0, 0, 0, 0]]
        expected_result = (0.875 + 1.0) / 2.0
        self.assertAllClose(result, expected_result, 1e-3)

    def test_sample_weight_flat(self):
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds, multi_label=False
        )
        result = auc_obj(
            self.y_true_good, self.y_pred, sample_weight=[1, 2, 3, 4]
        )

        # tpr = [1, 1, 0.2857, 0.2857, 0]
        # fpr = [1, 0.3333, 0, 0, 0]
        expected_result = 1.0 - (0.3333 * (1.0 - 0.2857) / 2.0)
        self.assertAllClose(result, expected_result, 1e-3)

    def test_full_sample_weight_flat(self):
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds, multi_label=False
        )
        sw = np.arange(4 * 2)
        sw = sw.reshape(4, 2)
        result = auc_obj(self.y_true_good, self.y_pred, sample_weight=sw)

        # tpr = [1, 1, 0.2727, 0.2727, 0]
        # fpr = [1, 0.3333, 0, 0, 0]
        expected_result = 1.0 - (0.3333 * (1.0 - 0.2727) / 2.0)
        self.assertAllClose(result, expected_result, 1e-3)

    def test_label_weights(self):
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds,
            multi_label=True,
            label_weights=[0.75, 0.25],
        )
        result = auc_obj(self.y_true_good, self.y_pred)

        # tpr = [[1, 1, 0.5, 0.5, 0], [1, 1, 0, 0, 0]]
        # fpr = [[1, 0.5, 0, 0, 0], [1, 0, 0, 0, 0]]
        expected_result = (0.875 * 0.75 + 1.0 * 0.25) / (0.75 + 0.25)
        self.assertAllClose(result, expected_result, 1e-3)

    def test_label_weights_flat(self):
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds,
            multi_label=False,
            label_weights=[0.75, 0.25],
        )
        result = auc_obj(self.y_true_good, self.y_pred)

        # tpr = [1, 1, 0.375, 0.375, 0]
        # fpr = [1, 0.375, 0, 0, 0]
        expected_result = 1.0 - ((1.0 - 0.375) * 0.375 / 2.0)
        self.assertAllClose(result, expected_result, 1e-2)

    def test_unweighted_flat(self):
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds, multi_label=False
        )
        result = auc_obj(self.y_true_good, self.y_pred)

        # tp = [4, 4, 1, 1, 0]
        # fp = [4, 1, 0, 0, 0]
        # fn = [0, 0, 3, 3, 4]
        # tn = [0, 3, 4, 4, 4]

        # tpr = [1, 1, 0.25, 0.25, 0]
        # fpr = [1, 0.25, 0, 0, 0]
        expected_result = 1.0 - (3.0 / 32.0)
        self.assertAllClose(result, expected_result, 1e-3)

    def test_unweighted_flat_from_logits(self):
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds,
            multi_label=False,
            from_logits=True,
        )
        result = auc_obj(self.y_true_good, self.y_pred_logits)

        # tp = [4, 4, 1, 1, 0]
        # fp = [4, 1, 0, 0, 0]
        # fn = [0, 0, 3, 3, 4]
        # tn = [0, 3, 4, 4, 4]

        # tpr = [1, 1, 0.25, 0.25, 0]
        # fpr = [1, 0.25, 0, 0, 0]
        expected_result = 1.0 - (3.0 / 32.0)
        self.assertAllClose(result, expected_result, 1e-3)

    def test_manual_thresholds(self):
        # Verify that when specified, thresholds are used instead of
        # num_thresholds.
        auc_obj = metrics.AUC(
            num_thresholds=2, thresholds=[0.5], multi_label=True
        )
        self.assertEqual(auc_obj.num_thresholds, 3)
        self.assertAllClose(auc_obj.thresholds, [0.0, 0.5, 1.0])
        result = auc_obj(self.y_true_good, self.y_pred)

        # tp = [[2, 1, 0], [2, 0, 0]]
        # fp = [2, 0, 0], [2, 0, 0]]
        # fn = [[0, 1, 2], [0, 2, 2]]
        # tn = [[0, 2, 2], [0, 2, 2]]

        # tpr = [[1, 0.5, 0], [1, 0, 0]]
        # fpr = [[1, 0, 0], [1, 0, 0]]

        # auc by slice = [0.75, 0.5]
        expected_result = (0.75 + 0.5) / 2.0

        self.assertAllClose(result, expected_result, 1e-3)

    def test_weighted_roc_interpolation(self):
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds, multi_label=True
        )
        result = auc_obj(
            self.y_true_good, self.y_pred, sample_weight=self.sample_weight
        )

        # tpr = [[1, 1,    0.57, 0.57, 0], [1, 1, 0, 0, 0]]
        # fpr = [[1, 0.67, 0,    0,    0], [1, 0, 0, 0, 0]]
        expected_result = 1.0 - 0.5 * 0.43 * 0.67
        self.assertAllClose(result, expected_result, 1e-1)

    def test_pr_interpolation_unweighted(self):
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds, curve="PR", multi_label=True
        )
        good_result = auc_obj(self.y_true_good, self.y_pred)
        with self.subTest(name="good"):
            # PR AUCs are 0.917 and 1.0 respectively
            self.assertAllClose(good_result, (0.91667 + 1.0) / 2.0, 1e-1)
        bad_result = auc_obj(self.y_true_bad, self.y_pred)
        with self.subTest(name="bad"):
            # PR AUCs are 0.917 and 0.5 respectively
            self.assertAllClose(bad_result, (0.91667 + 0.5) / 2.0, 1e-1)

    def test_pr_interpolation(self):
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds, curve="PR", multi_label=True
        )
        good_result = auc_obj(
            self.y_true_good, self.y_pred, sample_weight=self.sample_weight
        )
        # PR AUCs are 0.939 and 1.0 respectively
        self.assertAllClose(good_result, (0.939 + 1.0) / 2.0, 1e-1)

    @pytest.mark.requires_trainable_backend
    def test_keras_model_compiles(self):
        inputs = layers.Input(shape=(10,), batch_size=1)
        output = layers.Dense(3, activation="sigmoid")(inputs)
        model = models.Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[metrics.AUC(multi_label=True)],
        )

    def test_reset_state(self):
        auc_obj = metrics.AUC(
            num_thresholds=self.num_thresholds, multi_label=True
        )
        auc_obj(self.y_true_good, self.y_pred)
        auc_obj.reset_state()
        self.assertAllClose(auc_obj.true_positives, np.zeros((5, 2)))
