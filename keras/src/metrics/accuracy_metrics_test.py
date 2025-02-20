import numpy as np

from keras.src import testing
from keras.src.metrics import accuracy_metrics


class AccuracyTest(testing.TestCase):
    def test_config(self):
        acc_obj = accuracy_metrics.Accuracy(name="accuracy", dtype="float32")
        self.assertEqual(acc_obj.name, "accuracy")
        self.assertEqual(len(acc_obj.variables), 2)
        self.assertEqual(acc_obj._dtype, "float32")

        # Test get_config
        acc_obj_config = acc_obj.get_config()
        self.assertEqual(acc_obj_config["name"], "accuracy")
        self.assertEqual(acc_obj_config["dtype"], "float32")

        # Check save and restore config
        acc_obj2 = accuracy_metrics.Accuracy.from_config(acc_obj_config)
        self.assertEqual(acc_obj2.name, "accuracy")
        self.assertEqual(len(acc_obj2.variables), 2)
        self.assertEqual(acc_obj2._dtype, "float32")

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

    def test_weighted_rank_1(self):
        acc_obj = accuracy_metrics.Accuracy(name="accuracy", dtype="float32")
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([0, 2, 3, 4])
        sample_weight = np.array([1, 1, 0, 0])
        acc_obj.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = acc_obj.result()
        self.assertAllClose(result, 0.5, atol=1e-3)

    def test_weighted_nd_weights(self):
        acc_obj = accuracy_metrics.Accuracy(name="accuracy", dtype="float32")
        y_true = np.array([[1, 2], [3, 4]])
        y_pred = np.array([[0, 2], [3, 4]])
        sample_weight = np.array([[1, 0], [0, 1]])
        acc_obj.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = acc_obj.result()
        self.assertAllClose(result, 0.5, atol=1e-3)

    def test_weighted_nd_broadcast_weights(self):
        acc_obj = accuracy_metrics.Accuracy(name="accuracy", dtype="float32")
        y_true = np.array([[1, 2], [3, 4]])
        y_pred = np.array([[0, 2], [3, 4]])
        sample_weight = np.array([[1, 0]])
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

        # Test get_config
        bin_acc_obj_config = bin_acc_obj.get_config()
        self.assertEqual(bin_acc_obj_config["name"], "binary_accuracy")
        self.assertEqual(bin_acc_obj_config["dtype"], "float32")

        # Check save and restore config
        bin_acc_obj2 = accuracy_metrics.BinaryAccuracy.from_config(
            bin_acc_obj_config
        )
        self.assertEqual(bin_acc_obj2.name, "binary_accuracy")
        self.assertEqual(len(bin_acc_obj2.variables), 2)
        self.assertEqual(bin_acc_obj2._dtype, "float32")

    def test_unweighted(self):
        bin_acc_obj = accuracy_metrics.BinaryAccuracy(
            name="binary_accuracy", dtype="float32"
        )
        y_true = np.array([[1], [1], [0], [0]])
        y_pred = np.array([[0.98], [1], [0], [0.6]])
        bin_acc_obj.update_state(y_true, y_pred)
        result = bin_acc_obj.result()
        self.assertAllClose(result, 0.75, atol=1e-3)

        # Test broadcasting case
        bin_acc_obj = accuracy_metrics.BinaryAccuracy(
            name="binary_accuracy", dtype="float32"
        )
        y_true = np.array([1, 1, 0, 0])
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

    def test_weighted_rank_1(self):
        bin_acc_obj = accuracy_metrics.BinaryAccuracy(
            name="binary_accuracy", dtype="float32"
        )
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.98, 1, 0, 0.6])
        sample_weight = np.array([1, 0, 0, 1])
        bin_acc_obj.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = bin_acc_obj.result()
        self.assertAllClose(result, 0.5, atol=1e-3)

    def test_weighted_nd_weights(self):
        bin_acc_obj = accuracy_metrics.BinaryAccuracy(
            name="binary_accuracy", dtype="float32"
        )
        y_true = np.array([[1, 1], [0, 0]])
        y_pred = np.array([[0.98, 1], [0, 0.6]])
        sample_weight = np.array([[1, 0], [0, 1]])
        bin_acc_obj.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = bin_acc_obj.result()
        self.assertAllClose(result, 0.5, atol=1e-3)

    def test_weighted_nd_broadcast_weights(self):
        bin_acc_obj = accuracy_metrics.BinaryAccuracy(
            name="binary_accuracy", dtype="float32"
        )
        y_true = np.array([[1, 1], [0, 0]])
        y_pred = np.array([[0.98, 1], [0, 0.6]])
        sample_weight = np.array([[1, 0]])
        bin_acc_obj.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = bin_acc_obj.result()
        self.assertAllClose(result, 1.0, atol=1e-3)

    def test_threshold(self):
        bin_acc_obj_1 = accuracy_metrics.BinaryAccuracy(
            name="binary_accuracy", dtype="float32", threshold=0.3
        )
        bin_acc_obj_2 = accuracy_metrics.BinaryAccuracy(
            name="binary_accuracy", dtype="float32", threshold=0.9
        )
        y_true = np.array([[1], [1], [0], [0]])
        y_pred = np.array([[0.98], [0.5], [0.1], [0.2]])

        bin_acc_obj_1.update_state(y_true, y_pred)
        bin_acc_obj_2.update_state(y_true, y_pred)
        result_1 = bin_acc_obj_1.result()
        result_2 = bin_acc_obj_2.result()

        # Higher threshold must result in lower measured accuracy.
        self.assertAllClose(result_1, 1.0)
        self.assertAllClose(result_2, 0.75)


class CategoricalAccuracyTest(testing.TestCase):
    def test_config(self):
        cat_acc_obj = accuracy_metrics.CategoricalAccuracy(
            name="categorical_accuracy", dtype="float32"
        )
        self.assertEqual(cat_acc_obj.name, "categorical_accuracy")
        self.assertEqual(len(cat_acc_obj.variables), 2)
        self.assertEqual(cat_acc_obj._dtype, "float32")

        # Test get_config
        cat_acc_obj_config = cat_acc_obj.get_config()
        self.assertEqual(cat_acc_obj_config["name"], "categorical_accuracy")
        self.assertEqual(cat_acc_obj_config["dtype"], "float32")

        # Check save and restore config
        cat_acc_obj2 = accuracy_metrics.CategoricalAccuracy.from_config(
            cat_acc_obj_config
        )
        self.assertEqual(cat_acc_obj2.name, "categorical_accuracy")
        self.assertEqual(len(cat_acc_obj2.variables), 2)
        self.assertEqual(cat_acc_obj2._dtype, "float32")

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

        # Test get_config
        sp_cat_acc_obj_config = sp_cat_acc_obj.get_config()
        self.assertEqual(
            sp_cat_acc_obj_config["name"], "sparse_categorical_accuracy"
        )
        self.assertEqual(sp_cat_acc_obj_config["dtype"], "float32")

        # Check save and restore config
        sp_cat_acc_obj2 = (
            accuracy_metrics.SparseCategoricalAccuracy.from_config(
                sp_cat_acc_obj_config
            )
        )
        self.assertEqual(sp_cat_acc_obj2.name, "sparse_categorical_accuracy")
        self.assertEqual(len(sp_cat_acc_obj2.variables), 2)
        self.assertEqual(sp_cat_acc_obj2._dtype, "float32")

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

    def test_squeeze_y_true(self):
        sp_cat_acc_obj = accuracy_metrics.SparseCategoricalAccuracy(
            name="sparse_categorical_accuracy", dtype="float32"
        )
        # Scenario with 100% accuracy for simplicity.
        # y_true is a 2D tensor with shape (3, 1) to test squeeze.
        y_true = np.array([[0], [1], [2]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
        )
        sp_cat_acc_obj.update_state(y_true, y_pred)
        result = sp_cat_acc_obj.result()
        self.assertAllClose(result, 1.0, atol=1e-4)

    def test_cast_y_pred_dtype(self):
        sp_cat_acc_obj = accuracy_metrics.SparseCategoricalAccuracy(
            name="sparse_categorical_accuracy", dtype="float32"
        )
        # Scenario with 100% accuracy for simplicity.
        # y_true is a 1D tensor with shape (2,) to test cast.
        y_true = np.array([0, 1], dtype=np.int64)
        y_pred = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
        sp_cat_acc_obj.update_state(y_true, y_pred)
        result = sp_cat_acc_obj.result()
        self.assertAllClose(result, 1.0, atol=1e-4)

    def test_reshape_matches(self):
        sp_cat_acc_obj = accuracy_metrics.SparseCategoricalAccuracy(
            name="sparse_categorical_accuracy", dtype="float32"
        )
        # Scenario with 100% accuracy for simplicity.
        # y_true is a 2D tensor with shape (2, 1) to test reshape.
        y_true = np.array([[0], [0]], dtype=np.int64)
        y_pred = np.array(
            [[[0.9, 0.1, 0.0], [0.8, 0.15, 0.05]]], dtype=np.float32
        )
        sp_cat_acc_obj.update_state(y_true, y_pred)
        result = sp_cat_acc_obj.result()
        self.assertAllClose(result, np.array([1.0, 1.0]))

    def test_squeeze_y_true_shape(self):
        sp_cat_acc_obj = accuracy_metrics.SparseCategoricalAccuracy(
            name="sparse_categorical_accuracy", dtype="float32"
        )
        # True labels are in the shape (num_samples, 1) should be squeezed.
        y_true = np.array([[0], [1], [2]])
        y_pred = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        sp_cat_acc_obj.update_state(y_true, y_pred)
        result = sp_cat_acc_obj.result()
        self.assertAllClose(result, 1.0, atol=1e-4)

    def test_cast_y_pred_to_match_y_true_dtype(self):
        sp_cat_acc_obj = accuracy_metrics.SparseCategoricalAccuracy(
            name="sparse_categorical_accuracy", dtype="float32"
        )
        # True labels are integers, while predictions are floats.
        y_true = np.array([0, 1, 2], dtype=np.int32)
        y_pred = np.array(
            [[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 0.9]],
            dtype=np.float64,
        )
        sp_cat_acc_obj.update_state(y_true, y_pred)
        result = sp_cat_acc_obj.result()
        self.assertAllClose(result, 1.0, atol=1e-4)

    def test_reshape_matches_to_original_y_true_shape(self):
        sp_cat_acc_obj = accuracy_metrics.SparseCategoricalAccuracy(
            name="sparse_categorical_accuracy", dtype="float32"
        )
        # True labels have an additional dimension that needs to be squeezed.
        y_true = np.array([[0], [1]])
        # Predictions must trigger a reshape of matches.
        y_pred = np.array([[0.9, 0.1], [0.1, 0.9]])
        sp_cat_acc_obj.update_state(y_true, y_pred)
        result = sp_cat_acc_obj.result()
        self.assertAllClose(result, 1.0, atol=1e-4)

    def test_matching_shapes_without_squeeze(self):
        sp_cat_acc_obj = accuracy_metrics.SparseCategoricalAccuracy(
            name="sparse_categorical_accuracy", dtype="float32"
        )
        y_true = np.array([2, 1, 0], dtype=np.int32)
        y_pred = np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        # No need to squeeze or reshape.
        sp_cat_acc_obj.update_state(y_true, y_pred)
        result = sp_cat_acc_obj.result()
        self.assertAllClose(result, 1.0, atol=1e-4)


class TopKCategoricalAccuracyTest(testing.TestCase):
    def test_config(self):
        top_k_cat_acc_obj = accuracy_metrics.TopKCategoricalAccuracy(
            k=1, name="top_k_categorical_accuracy", dtype="float32"
        )
        self.assertEqual(top_k_cat_acc_obj.name, "top_k_categorical_accuracy")
        self.assertEqual(len(top_k_cat_acc_obj.variables), 2)
        self.assertEqual(top_k_cat_acc_obj._dtype, "float32")

        # Test get_config
        top_k_cat_acc_obj_config = top_k_cat_acc_obj.get_config()
        self.assertEqual(
            top_k_cat_acc_obj_config["name"], "top_k_categorical_accuracy"
        )
        self.assertEqual(top_k_cat_acc_obj_config["dtype"], "float32")
        self.assertEqual(top_k_cat_acc_obj_config["k"], 1)

        # Check save and restore config
        top_k_cat_acc_obj2 = (
            accuracy_metrics.TopKCategoricalAccuracy.from_config(
                top_k_cat_acc_obj_config
            )
        )
        self.assertEqual(top_k_cat_acc_obj2.name, "top_k_categorical_accuracy")
        self.assertEqual(len(top_k_cat_acc_obj2.variables), 2)
        self.assertEqual(top_k_cat_acc_obj2._dtype, "float32")
        self.assertEqual(top_k_cat_acc_obj2.k, 1)

    def test_unweighted(self):
        top_k_cat_acc_obj = accuracy_metrics.TopKCategoricalAccuracy(
            k=1, name="top_k_categorical_accuracy", dtype="float32"
        )
        y_true = np.array([[0, 0, 1], [0, 1, 0]])
        y_pred = np.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]], dtype="float32")
        top_k_cat_acc_obj.update_state(y_true, y_pred)
        result = top_k_cat_acc_obj.result()
        self.assertAllClose(result, 0.5, atol=1e-3)

    def test_weighted(self):
        top_k_cat_acc_obj = accuracy_metrics.TopKCategoricalAccuracy(
            k=1, name="top_k_categorical_accuracy", dtype="float32"
        )
        y_true = np.array([[0, 0, 1], [0, 1, 0]])
        y_pred = np.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]], dtype="float32")
        sample_weight = np.array([0.7, 0.3])
        top_k_cat_acc_obj.update_state(
            y_true, y_pred, sample_weight=sample_weight
        )
        result = top_k_cat_acc_obj.result()
        self.assertAllClose(result, 0.3, atol=1e-3)


class SparseTopKCategoricalAccuracyTest(testing.TestCase):
    def test_config(self):
        sp_top_k_cat_acc_obj = accuracy_metrics.SparseTopKCategoricalAccuracy(
            k=1, name="sparse_top_k_categorical_accuracy", dtype="float32"
        )
        self.assertEqual(
            sp_top_k_cat_acc_obj.name, "sparse_top_k_categorical_accuracy"
        )
        self.assertEqual(len(sp_top_k_cat_acc_obj.variables), 2)
        self.assertEqual(sp_top_k_cat_acc_obj._dtype, "float32")

        # Test get_config
        sp_top_k_cat_acc_obj_config = sp_top_k_cat_acc_obj.get_config()
        self.assertEqual(
            sp_top_k_cat_acc_obj_config["name"],
            "sparse_top_k_categorical_accuracy",
        )
        self.assertEqual(sp_top_k_cat_acc_obj_config["dtype"], "float32")
        self.assertEqual(sp_top_k_cat_acc_obj_config["k"], 1)

        # Check save and restore config
        sp_top_k_cat_acc_obj2 = (
            accuracy_metrics.SparseTopKCategoricalAccuracy.from_config(
                sp_top_k_cat_acc_obj_config
            )
        )
        self.assertEqual(
            sp_top_k_cat_acc_obj2.name, "sparse_top_k_categorical_accuracy"
        )
        self.assertEqual(len(sp_top_k_cat_acc_obj2.variables), 2)
        self.assertEqual(sp_top_k_cat_acc_obj2._dtype, "float32")
        self.assertEqual(sp_top_k_cat_acc_obj2.k, 1)
        self.assertFalse(sp_top_k_cat_acc_obj2.from_sorted_ids)

    def test_config_from_sorted_ids(self):
        sp_top_k_cat_acc_obj = accuracy_metrics.SparseTopKCategoricalAccuracy(
            k=1,
            name="sparse_top_k_categorical_accuracy",
            dtype="float32",
            from_sorted_ids=True,
        )

        # Test get_config
        sp_top_k_cat_acc_obj_config = sp_top_k_cat_acc_obj.get_config()
        self.assertTrue(sp_top_k_cat_acc_obj_config["from_sorted_ids"])

        # Check save and restore config
        sp_top_k_cat_acc_obj2 = (
            accuracy_metrics.SparseTopKCategoricalAccuracy.from_config(
                sp_top_k_cat_acc_obj_config
            )
        )
        self.assertTrue(sp_top_k_cat_acc_obj2.from_sorted_ids)

    def test_unweighted(self):
        sp_top_k_cat_acc_obj = accuracy_metrics.SparseTopKCategoricalAccuracy(
            k=1, name="sparse_top_k_categorical_accuracy", dtype="float32"
        )
        y_true = np.array([2, 1])
        y_pred = np.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]], dtype="float32")
        sp_top_k_cat_acc_obj.update_state(y_true, y_pred)
        result = sp_top_k_cat_acc_obj.result()
        self.assertAllClose(result, 0.5, atol=1e-3)

    def test_weighted(self):
        sp_top_k_cat_acc_obj = accuracy_metrics.SparseTopKCategoricalAccuracy(
            k=1, name="sparse_top_k_categorical_accuracy", dtype="float32"
        )
        y_true = np.array([2, 1])
        y_pred = np.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]], dtype="float32")
        sample_weight = np.array([0.7, 0.3])
        sp_top_k_cat_acc_obj.update_state(
            y_true, y_pred, sample_weight=sample_weight
        )
        result = sp_top_k_cat_acc_obj.result()
        self.assertAllClose(result, 0.3, atol=1e-3)

    def test_from_sorted_ids_unweighted(self):
        sp_top_k_cat_acc_obj = accuracy_metrics.SparseTopKCategoricalAccuracy(
            k=1,
            name="sparse_top_k_categorical_accuracy",
            dtype="float32",
            from_sorted_ids=True,
        )
        y_true = np.array([2, 1])
        y_pred = np.array([[1, 0, 3], [1, 2, 3]])
        sp_top_k_cat_acc_obj.update_state(y_true, y_pred)
        result = sp_top_k_cat_acc_obj.result()
        self.assertAllClose(result, 0.5, atol=1e-3)

    def test_from_sorted_ids_weighted(self):
        sp_top_k_cat_acc_obj = accuracy_metrics.SparseTopKCategoricalAccuracy(
            k=1,
            name="sparse_top_k_categorical_accuracy",
            dtype="float32",
            from_sorted_ids=True,
        )
        y_true = np.array([2, 1])
        y_pred = np.array([[1, 0, 3], [1, 2, 3]])
        sample_weight = np.array([0.7, 0.3])
        sp_top_k_cat_acc_obj.update_state(
            y_true, y_pred, sample_weight=sample_weight
        )
        result = sp_top_k_cat_acc_obj.result()
        self.assertAllClose(result, 0.3, atol=1e-3)
