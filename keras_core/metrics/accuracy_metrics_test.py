import numpy as np

from keras_core import testing
from keras_core.metrics import accuracy_metrics


class AccuracyTest(testing.TestCase):
    def test_config(self):
        acc_obj = accuracy_metrics.Accuracy(name="accuracy", dtype="float32")
        self.assertEqual(acc_obj.name, "accuracy")
        self.assertEqual(len(acc_obj.variables), 2)
        self.assertEqual(acc_obj._dtype, "float32")
        # TODO: Check save and restore config

    def test_unweighted(self):
        acc_obj = accuracy_metrics.Accuracy(name="accuracy", dtype="float32")
        y_true = np.array([[1], [2], [3], [4]])
        y_pred = np.array([[0], [2], [3], [4]])
        acc_obj.update_state(y_true, y_pred)
        result = acc_obj.result()
        self.assertAllClose(result, 0.75, atol=1e-3)

    def test_weighted(self):
        acc_obj = accuracy_metrics.Accuracy(name="accuracy", dtype="float32")
        y_true = np.array([[1], [2], [3], [4]])
        y_pred = np.array([[0], [2], [3], [4]])
        sample_weight = np.array([1, 1, 0, 0])
        acc_obj.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = acc_obj.result()
        self.assertAllClose(result, 0.5, atol=1e-3)
