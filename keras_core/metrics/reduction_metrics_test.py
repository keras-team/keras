import numpy as np

from keras_core import testing
from keras_core.metrics import reduction_metrics


class SumTest(testing.TestCase):
    def test_config(self):
        sum_obj = reduction_metrics.Sum(name="sum", dtype="float32")
        self.assertEqual(sum_obj.name, "sum")
        self.assertEqual(len(sum_obj.variables), 1)
        self.assertEqual(sum_obj._dtype, "float32")
        # TODO: Check save and restore config

    def test_unweighted(self):
        sum_obj = reduction_metrics.Sum(name="sum", dtype="float32")
        sum_obj.update_state([1, 3, 5, 7])
        result = sum_obj.result()
        self.assertAllClose(result, 16.0, atol=1e-3)

    def test_weighted(self):
        sum_obj = reduction_metrics.Sum(name="sum", dtype="float32")
        sum_obj.update_state([1, 3, 5, 7], sample_weight=[1, 1, 0, 0])
        result = sum_obj.result()
        self.assertAllClose(result, 4.0, atol=1e-3)


class MeanTest(testing.TestCase):
    def test_config(self):
        mean_obj = reduction_metrics.Mean(name="mean", dtype="float32")
        self.assertEqual(mean_obj.name, "mean")
        self.assertEqual(len(mean_obj.variables), 2)
        self.assertEqual(mean_obj._dtype, "float32")
        # TODO: Check save and restore config

    def test_unweighted(self):
        mean_obj = reduction_metrics.Mean(name="mean", dtype="float32")
        mean_obj.update_state([1, 3, 5, 7])
        result = mean_obj.result()
        self.assertAllClose(result, 4.0, atol=1e-3)

    def test_weighted(self):
        mean_obj = reduction_metrics.Mean(name="mean", dtype="float32")
        mean_obj.update_state([1, 3, 5, 7], sample_weight=[1, 1, 0, 0])
        result = mean_obj.result()
        self.assertAllClose(result, 2.0, atol=1e-3)


def mse(y_true, y_pred):
    return (y_true - y_pred) ** 2


class MetricWrapperTest(testing.TestCase):
    def test_config(self):
        mse_obj = reduction_metrics.MeanMetricWrapper(
            fn=mse, name="mse", dtype="float32"
        )
        self.assertEqual(mse_obj.name, "mse")
        self.assertEqual(len(mse_obj.variables), 2)
        self.assertEqual(mse_obj._dtype, "float32")
        # TODO: Check save and restore config

    def test_unweighted(self):
        mse_obj = reduction_metrics.MeanMetricWrapper(
            fn=mse, name="mse", dtype="float32"
        )
        y_true = np.array(
            [[0, 1, 0, 1, 0], [0, 0, 1, 1, 1], [1, 1, 1, 1, 0], [0, 0, 0, 0, 1]]
        )
        y_pred = np.array(
            [[0, 0, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1]]
        )

        mse_obj.update_state(y_true, y_pred)
        result = mse_obj.result()
        self.assertAllClose(0.5, result, atol=1e-5)

    def test_weighted(self):
        mse_obj = reduction_metrics.MeanMetricWrapper(
            fn=mse, name="mse", dtype="float32"
        )
        y_true = np.array(
            [[0, 1, 0, 1, 0], [0, 0, 1, 1, 1], [1, 1, 1, 1, 0], [0, 0, 0, 0, 1]]
        )
        y_pred = np.array(
            [[0, 0, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1]]
        )
        sample_weight = np.array([1.0, 1.5, 2.0, 2.5])
        result = mse_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(0.54285, result, atol=1e-5)
