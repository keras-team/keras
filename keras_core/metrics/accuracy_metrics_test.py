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


class BinaryAccuracyTest(testing.TestCase):
    def test_config(self):
        bin_acc_obj = accuracy_metrics.BinaryAccuracy(
            name="binary_accuracy", dtype="float32"
        )
        self.assertEqual(bin_acc_obj.name, "binary_accuracy")
        self.assertEqual(len(bin_acc_obj.variables), 2)
        self.assertEqual(bin_acc_obj._dtype, "float32")
        # TODO: Check save and restore config

    def test_unweighted(self):
        bin_acc_obj = accuracy_metrics.BinaryAccuracy(
            name="binary_accuracy", dtype="float32"
        )
        y_true = np.array([[1], [1], [0], [0]])
        y_pred = np.array([[0.98], [1], [0], [0.6]])
        bin_acc_obj.update_state(y_true, y_pred)
        result = bin_acc_obj.result()
        self.assertAllClose(result, 0.75, atol=1e-3)

    def test_weighted(self):
        bin_acc_obj = accuracy_metrics.BinaryAccuracy(
            name="binary_accuracy", dtype="float32"
        )
        y_true = np.array([[1], [1], [0], [0]])
        y_pred = np.array([[0.98], [1], [0], [0.6]])
        sample_weight = np.array([1, 0, 0, 1])
        bin_acc_obj.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = bin_acc_obj.result()
        self.assertAllClose(result, 0.5, atol=1e-3)


class CategoricalAccuracyTest(testing.TestCase):
    def test_config(self):
        cat_acc_obj = accuracy_metrics.CategoricalAccuracy(
            name="categorical_accuracy", dtype="float32"
        )
        self.assertEqual(cat_acc_obj.name, "categorical_accuracy")
        self.assertEqual(len(cat_acc_obj.variables), 2)
        self.assertEqual(cat_acc_obj._dtype, "float32")
        # TODO: Check save and restore config

    def test_unweighted(self):
        cat_acc_obj = accuracy_metrics.CategoricalAccuracy(
            name="categorical_accuracy", dtype="float32"
        )
        y_true = np.array([[0, 0, 1], [0, 1, 0]])
        y_pred = np.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
        cat_acc_obj.update_state(y_true, y_pred)
        result = cat_acc_obj.result()
        self.assertAllClose(result, 0.5, atol=1e-3)

    def test_weighted(self):
        cat_acc_obj = accuracy_metrics.CategoricalAccuracy(
            name="categorical_accuracy", dtype="float32"
        )
        y_true = np.array([[0, 0, 1], [0, 1, 0]])
        y_pred = np.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
        sample_weight = np.array([0.7, 0.3])
        cat_acc_obj.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = cat_acc_obj.result()
        self.assertAllClose(result, 0.3, atol=1e-3)


class SparseCategoricalAccuracyTest(testing.TestCase):
    def test_config(self):
        sp_cat_acc_obj = accuracy_metrics.SparseCategoricalAccuracy(
            name="sparse_categorical_accuracy", dtype="float32"
        )
        self.assertEqual(sp_cat_acc_obj.name, "sparse_categorical_accuracy")
        self.assertEqual(len(sp_cat_acc_obj.variables), 2)
        self.assertEqual(sp_cat_acc_obj._dtype, "float32")
        # TODO: Check save and restore config

    def test_unweighted(self):
        sp_cat_acc_obj = accuracy_metrics.SparseCategoricalAccuracy(
            name="sparse_categorical_accuracy", dtype="float32"
        )
        y_true = np.array([[2], [1]])
        y_pred = np.array([[0.1, 0.6, 0.3], [0.05, 0.95, 0]])
        sp_cat_acc_obj.update_state(y_true, y_pred)
        result = sp_cat_acc_obj.result()
        self.assertAllClose(result, 0.5, atol=1e-3)

    def test_weighted(self):
        sp_cat_acc_obj = accuracy_metrics.SparseCategoricalAccuracy(
            name="sparse_categorical_accuracy", dtype="float32"
        )
        y_true = np.array([[2], [1]])
        y_pred = np.array([[0.1, 0.6, 0.3], [0.05, 0.95, 0]])
        sample_weight = np.array([0.7, 0.3])
        sp_cat_acc_obj.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = sp_cat_acc_obj.result()
        self.assertAllClose(result, 0.3, atol=1e-3)
