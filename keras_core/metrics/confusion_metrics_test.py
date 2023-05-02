import numpy as np
from tensorflow.python.ops.numpy_ops import np_config

from keras_core import metrics
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
