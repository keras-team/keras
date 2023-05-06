import numpy as np
from absl.testing import parameterized
from tensorflow.python.ops.numpy_ops import np_config

from keras_core import metrics
from keras_core import operations as ops
from keras_core import testing

# TODO: remove reliance on this (or alternatively, turn it on by default).
# This is no longer needed with tf-nightly.
np_config.enable_numpy_behavior()


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

    def test_unweighted_class_id(self):
        p_obj = metrics.Precision(class_id=2)

        y_pred = np.array([0.2, 0.1, 0.6, 0, 0.2])
        y_true = np.array([0, 1, 1, 0, 0])
        result = p_obj(y_true, y_pred)
        self.assertAlmostEqual(1, result)
        self.assertAlmostEqual(1, p_obj.true_positives)
        self.assertAlmostEqual(0, p_obj.false_positives)

        y_pred = np.array([0.2, 0.1, 0, 0, 0.2])
        y_true = np.array([0, 1, 1, 0, 0])
        result = p_obj(y_true, y_pred)
        self.assertAlmostEqual(1, result)
        self.assertAlmostEqual(1, p_obj.true_positives)
        self.assertAlmostEqual(0, p_obj.false_positives)

        y_pred = np.array([0.2, 0.1, 0.6, 0, 0.2])
        y_true = np.array([0, 1, 0, 0, 0])
        result = p_obj(y_true, y_pred)
        self.assertAlmostEqual(0.5, result)
        self.assertAlmostEqual(1, p_obj.true_positives)
        self.assertAlmostEqual(1, p_obj.false_positives)

    def test_unweighted_top_k_and_class_id(self):
        p_obj = metrics.Precision(class_id=2, top_k=2)

        y_pred = np.array([0.2, 0.6, 0.3, 0, 0.2])
        y_true = np.array([0, 1, 1, 0, 0])
        result = p_obj(y_true, y_pred)
        self.assertAlmostEqual(1, result)
        self.assertAlmostEqual(1, p_obj.true_positives)
        self.assertAlmostEqual(0, p_obj.false_positives)

        y_pred = np.array([1, 1, 0.9, 1, 1])
        y_true = np.array([0, 1, 1, 0, 0])
        result = p_obj(y_true, y_pred)
        self.assertAlmostEqual(1, result)
        self.assertAlmostEqual(1, p_obj.true_positives)
        self.assertAlmostEqual(0, p_obj.false_positives)

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

    def test_unweighted_class_id(self):
        r_obj = metrics.Recall(class_id=2)

        y_pred = np.array([0.2, 0.1, 0.6, 0, 0.2])
        y_true = np.array([0, 1, 1, 0, 0])
        self.assertAlmostEqual(1, r_obj(y_true, y_pred))
        self.assertAlmostEqual(1, r_obj.true_positives)
        self.assertAlmostEqual(0, r_obj.false_negatives)

        y_pred = np.array([0.2, 0.1, 0, 0, 0.2])
        y_true = np.array([0, 1, 1, 0, 0])
        self.assertAlmostEqual(0.5, r_obj(y_true, y_pred))
        self.assertAlmostEqual(1, r_obj.true_positives)
        self.assertAlmostEqual(1, r_obj.false_negatives)

        y_pred = np.array([0.2, 0.1, 0.6, 0, 0.2])
        y_true = np.array([0, 1, 0, 0, 0])
        self.assertAlmostEqual(0.5, r_obj(y_true, y_pred))
        self.assertAlmostEqual(1, r_obj.true_positives)
        self.assertAlmostEqual(1, r_obj.false_negatives)

    def test_unweighted_top_k_and_class_id(self):
        r_obj = metrics.Recall(class_id=2, top_k=2)

        y_pred = np.array([0.2, 0.6, 0.3, 0, 0.2])
        y_true = np.array([0, 1, 1, 0, 0])
        self.assertAlmostEqual(1, r_obj(y_true, y_pred))
        self.assertAlmostEqual(1, r_obj.true_positives)
        self.assertAlmostEqual(0, r_obj.false_negatives)

        y_pred = np.array([1, 1, 0.9, 1, 1])
        y_true = np.array([0, 1, 1, 0, 0])
        self.assertAlmostEqual(0.5, r_obj(y_true, y_pred))
        self.assertAlmostEqual(1, r_obj.true_positives)
        self.assertAlmostEqual(1, r_obj.false_negatives)

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
        y_true = ops.one_hot(label_values, num_classes=3)

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
        y_true = ops.one_hot(label_values, num_classes=3)

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
        y_true = ops.one_hot(label_values, num_classes=3)

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
        y_true = ops.one_hot(label_values, num_classes=3)

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
