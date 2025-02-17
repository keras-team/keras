import numpy as np
import pytest

from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.metrics import iou_metrics as metrics
from keras.src.ops import convert_to_tensor


class IoUTest(testing.TestCase):
    def test_config(self):
        obj = metrics.IoU(
            num_classes=2, target_class_ids=[1, 0], name="iou_class_1_0"
        )
        self.assertEqual(obj.name, "iou_class_1_0")
        self.assertEqual(obj.num_classes, 2)
        self.assertEqual(obj.target_class_ids, [1, 0])

        obj2 = metrics.IoU.from_config(obj.get_config())
        self.assertEqual(obj2.name, "iou_class_1_0")
        self.assertEqual(obj2.num_classes, 2)
        self.assertEqual(obj2.target_class_ids, [1, 0])

    def test_unweighted(self):
        y_pred = [0, 1, 0, 1]
        y_true = [0, 0, 1, 1]

        obj = metrics.IoU(num_classes=2, target_class_ids=[0, 1])

        result = obj(y_true, y_pred)

        # cm = [[1, 1],
        #       [1, 1]]
        # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_weighted(self):
        y_pred = np.array([0, 1, 0, 1], dtype=np.float32)
        y_true = np.array([0, 0, 1, 1])
        sample_weight = np.array([0.2, 0.3, 0.4, 0.1])

        obj = metrics.IoU(
            num_classes=2, target_class_ids=[1, 0], dtype="float32"
        )

        result = obj(y_true, y_pred, sample_weight=sample_weight)

        # cm = [[0.2, 0.3],
        #       [0.4, 0.1]]
        # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5], true_positives = [0.2,
        # 0.1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            0.1 / (0.4 + 0.5 - 0.1) + 0.2 / (0.6 + 0.5 - 0.2)
        ) / 2
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_multi_dim_input(self):
        y_pred = np.array([[0, 1], [0, 1]], dtype=np.float32)
        y_true = np.array([[0, 0], [1, 1]])
        sample_weight = np.array([[0.2, 0.3], [0.4, 0.1]])

        obj = metrics.IoU(
            num_classes=2, target_class_ids=[0, 1], dtype="float32"
        )

        result = obj(y_true, y_pred, sample_weight=sample_weight)

        # cm = [[0.2, 0.3],
        #       [0.4, 0.1]]
        # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5], true_positives = [0.2,
        # 0.1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            0.2 / (0.6 + 0.5 - 0.2) + 0.1 / (0.4 + 0.5 - 0.1)
        ) / 2
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_zero_valid_entries(self):
        obj = metrics.IoU(num_classes=2, target_class_ids=[0, 1])
        self.assertAllClose(obj.result(), 0, atol=1e-3)

    def test_zero_and_non_zero_entries(self):
        y_pred = np.array([1], dtype=np.float32)
        y_true = np.array([1])

        obj = metrics.IoU(num_classes=2, target_class_ids=[0, 1])
        result = obj(y_true, y_pred)

        # cm = [[0, 0],
        #       [0, 1]]
        # sum_row = [0, 1], sum_col = [0, 1], true_positives = [0, 1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (1 / (1 + 1 - 1)) / 1
        self.assertAllClose(result, expected_result, atol=1e-3)

    @pytest.mark.requires_trainable_backend
    def test_compilation(self):
        m_obj = metrics.MeanIoU(num_classes=2, ignore_class=0)
        model = models.Sequential(
            [
                layers.Dense(2, activation="softmax"),
            ]
        )
        model.compile(optimizer="rmsprop", loss="mse", metrics=[m_obj])
        model.fit(np.array([[1.0, 1.0]]), np.array([[1.0, 0.0]]))


class BinaryIoUTest(testing.TestCase):
    def test_config(self):
        obj = metrics.BinaryIoU(
            target_class_ids=[1, 0], threshold=0.1, name="iou_class_1_0"
        )
        self.assertEqual(obj.name, "iou_class_1_0")
        self.assertAlmostEqual(obj.threshold, 0.1)
        self.assertEqual(obj.target_class_ids, [1, 0])

        obj2 = metrics.BinaryIoU.from_config(obj.get_config())
        self.assertEqual(obj.name, "iou_class_1_0")
        self.assertAlmostEqual(obj2.threshold, 0.1)
        self.assertEqual(obj.target_class_ids, [1, 0])

    def test_different_thresholds_weighted(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0.1, 0.2, 0.4, 0.7]

        sample_weight = np.array([0.2, 0.3, 0.4, 0.1])
        # with threshold = 0.3, y_pred will be converted to [0, 0, 1, 1]
        # cm = [[0.2, 0.4],
        #       [0.3, 0.1]]
        # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5], true_positives = [0.2,
        # 0.1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            0.2 / (0.6 + 0.5 - 0.2) + 0.1 / (0.4 + 0.5 - 0.1)
        ) / 2
        obj = metrics.BinaryIoU(
            target_class_ids=[0, 1], threshold=0.3, dtype="float32"
        )
        result = obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(result, expected_result, atol=1e-3)

        sample_weight = np.array([0.1, 0.2, 0.4, 0.3])
        # with threshold = 0.5, y_pred will be converted to [0, 0, 0, 1]
        # cm = [[0.1+0.4, 0],
        #       [0.2, 0.3]]
        # sum_row = [0.5, 0.5], sum_col = [0.7, 0.3], true_positives = [0.5,
        # 0.3]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            0.5 / (0.5 + 0.7 - 0.5) + 0.3 / (0.5 + 0.3 - 0.3)
        ) / 2
        obj = metrics.BinaryIoU(
            target_class_ids=[0, 1], threshold=0.5, dtype="float32"
        )
        result = obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_different_thresholds_unweighted(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0.1, 0.2, 0.4, 0.7]

        # with threshold = 0.3, y_pred will be converted to [0, 0, 1, 1]
        # cm = [[1, 1],
        #       [1, 1]]
        # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2
        obj = metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.3)
        result = obj(y_true, y_pred)
        self.assertAllClose(result, expected_result, atol=1e-3)

        # with threshold = 0.5, y_pred will be converted to [0, 0, 0, 1]
        # cm = [[2, 0],
        #       [1, 1]]
        # sum_row = [2, 2], sum_col = [3, 1], true_positives = [2, 1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (2 / (2 + 3 - 2) + 1 / (2 + 1 - 1)) / 2
        obj = metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5)
        result = obj(y_true, y_pred)
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_multi_dim_input(self):
        y_true = np.array([[0, 1], [0, 1]], dtype=np.float32)
        y_pred = np.array([[0.1, 0.7], [0.9, 0.3]])
        threshold = 0.4  # y_pred will become [[0, 1], [1, 0]]
        sample_weight = np.array([[0.2, 0.3], [0.4, 0.1]])
        # cm = [[0.2, 0.4],
        #       [0.1, 0.3]]
        # sum_row = [0.6, 0.4], sum_col = [0.3, 0.7], true_positives = [0.2,
        # 0.3]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            0.2 / (0.6 + 0.3 - 0.2) + 0.3 / (0.4 + 0.7 - 0.3)
        ) / 2
        obj = metrics.BinaryIoU(
            target_class_ids=[0, 1], threshold=threshold, dtype="float32"
        )
        result = obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_zero_valid_entries(self):
        obj = metrics.BinaryIoU(target_class_ids=[0, 1])
        self.assertAllClose(obj.result(), 0, atol=1e-3)

    def test_zero_and_non_zero_entries(self):
        y_pred = np.array([0.6], dtype=np.float32)
        threshold = 0.5
        y_true = np.array([1])

        obj = metrics.BinaryIoU(target_class_ids=[0, 1], threshold=threshold)
        result = obj(y_true, y_pred)

        # cm = [[0, 0],
        #       [0, 1]]
        # sum_row = [0, 1], sum_col = [0, 1], true_positives = [0, 1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = 1 / (1 + 1 - 1)
        self.assertAllClose(result, expected_result, atol=1e-3)


class MeanIoUTest(testing.TestCase):
    def test_config(self):
        m_obj = metrics.MeanIoU(num_classes=2, name="mean_iou")
        self.assertEqual(m_obj.name, "mean_iou")
        self.assertEqual(m_obj.num_classes, 2)

        m_obj2 = metrics.MeanIoU.from_config(m_obj.get_config())
        self.assertEqual(m_obj2.name, "mean_iou")
        self.assertEqual(m_obj2.num_classes, 2)

    def test_unweighted(self):
        y_pred = [0, 1, 0, 1]
        y_true = [0, 0, 1, 1]

        m_obj = metrics.MeanIoU(num_classes=2)

        result = m_obj(y_true, y_pred)

        # cm = [[1, 1],
        #       [1, 1]]
        # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_unweighted_ignore_class_255(self):
        y_pred = [0, 1, 1, 1]
        y_true = [0, 1, 2, 255]

        m_obj = metrics.MeanIoU(num_classes=3, ignore_class=255)

        result = m_obj(y_true, y_pred)

        # cm = [[1, 0, 0],
        #       [0, 1, 0],
        #       [0, 1, 0]]
        # sum_row = [1, 1, 1], sum_col = [1, 2, 0], true_positives = [1, 1, 0]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            1 / (1 + 1 - 1) + 1 / (2 + 1 - 1) + 0 / (0 + 1 - 0)
        ) / 3
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_unweighted_ignore_class_1(self):
        y_pred = [0, 1, 1, 1]
        y_true = [0, 1, 2, -1]

        m_obj = metrics.MeanIoU(num_classes=3, ignore_class=-1)

        result = m_obj(y_true, y_pred)

        # cm = [[1, 0, 0],
        #       [0, 1, 0],
        #       [0, 1, 0]]
        # sum_row = [1, 1, 1], sum_col = [1, 2, 0], true_positives = [1, 1, 0]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            1 / (1 + 1 - 1) + 1 / (2 + 1 - 1) + 0 / (0 + 1 - 0)
        ) / 3
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_weighted(self):
        y_pred = np.array([0, 1, 0, 1], dtype=np.float32)
        y_true = np.array([0, 0, 1, 1])
        sample_weight = np.array([0.2, 0.3, 0.4, 0.1])

        m_obj = metrics.MeanIoU(num_classes=2, dtype="float32")

        result = m_obj(y_true, y_pred, sample_weight=sample_weight)

        # cm = [[0.2, 0.3],
        #       [0.4, 0.1]]
        # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5], true_positives = [0.2,
        # 0.1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            0.2 / (0.6 + 0.5 - 0.2) + 0.1 / (0.4 + 0.5 - 0.1)
        ) / 2
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_weighted_ignore_class_1(self):
        y_pred = np.array([0, 1, 0, 1], dtype=np.float32)
        y_true = np.array([0, 0, 1, -1])
        sample_weight = np.array([0.2, 0.3, 0.4, 0.1])

        m_obj = metrics.MeanIoU(num_classes=2, ignore_class=-1, dtype="float32")

        result = m_obj(y_true, y_pred, sample_weight=sample_weight)

        # cm = [[0.2, 0.3],
        #       [0.4, 0.0]]
        # sum_row = [0.6, 0.3], sum_col = [0.5, 0.4], true_positives = [0.2,
        # 0.0]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            0.2 / (0.6 + 0.5 - 0.2) + 0.0 / (0.3 + 0.4 - 0.0)
        ) / 2
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_multi_dim_input(self):
        y_pred = np.array([[0, 1], [0, 1]], dtype=np.float32)
        y_true = np.array([[0, 0], [1, 1]])
        sample_weight = np.array([[0.2, 0.3], [0.4, 0.1]])

        m_obj = metrics.MeanIoU(num_classes=2, dtype="float32")

        result = m_obj(y_true, y_pred, sample_weight=sample_weight)

        # cm = [[0.2, 0.3],
        #       [0.4, 0.1]]
        # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5], true_positives = [0.2,
        # 0.1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            0.2 / (0.6 + 0.5 - 0.2) + 0.1 / (0.4 + 0.5 - 0.1)
        ) / 2
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_zero_valid_entries(self):
        m_obj = metrics.MeanIoU(num_classes=2)
        self.assertAllClose(m_obj.result(), 0, atol=1e-3)

    def test_zero_and_non_zero_entries(self):
        y_pred = np.array([1], dtype=np.float32)
        y_true = np.array([1])

        m_obj = metrics.MeanIoU(num_classes=2)
        result = m_obj(y_true, y_pred)

        # cm = [[0, 0],
        #       [0, 1]]
        # sum_row = [0, 1], sum_col = [0, 1], true_positives = [0, 1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (0 + 1 / (1 + 1 - 1)) / 1
        self.assertAllClose(result, expected_result, atol=1e-3)

    @staticmethod
    def _confusion_matrix(y_true, y_pred, num_classes):
        """
        Creates a confusion matrix as a numpy array using vectorized operations.

        Parameters:
        - y_true: array-like, true class labels.
        - y_pred: array-like, predicted class labels.
        - num_classes: int, number of classes.

        Returns:
        - conf_matrix: np.ndarray, confusion matrix of shape (num_classes,
                                                              num_classes).
        """
        # Map pairs of (y_true, y_pred) to indices in the confusion matrix
        indices = y_true * num_classes + y_pred
        # Count occurrences of each index
        conf_matrix = np.bincount(indices, minlength=num_classes * num_classes)
        # Reshape the flat array into a 2D confusion matrix
        conf_matrix = conf_matrix.reshape((num_classes, num_classes))
        return conf_matrix

    @staticmethod
    def _get_big_chunk(dtype):
        np.random.seed(14)
        all_y_true = np.random.choice([0, 1, 2], size=(10, 530, 530))
        # Generate random probabilities for each channel
        random_probs = np.random.rand(10, 530, 530, 3)
        # Normalize to ensure the last dimension sums to 1
        all_y_pred = random_probs / random_probs.sum(axis=-1, keepdims=True)
        # Convert predictions to class indices
        all_y_pred_arg = np.argmax(all_y_pred, axis=-1)
        mean_iou_metric = metrics.MeanIoU(num_classes=3, dtype=dtype)
        conf_matrix_start_point = np.array(
            [
                [18729664, 18728760, 18731196],
                [18727297, 18726105, 18728071],
                [18727917, 18717835, 18723155],
            ]
        )
        mean_iou_metric.total_cm = mean_iou_metric.add_variable(
            name="total_confusion_matrix",
            shape=(3, 3),
            initializer=convert_to_tensor(conf_matrix_start_point),
            dtype=dtype or "int",
        )
        mean_iou_metric.update_state(all_y_true, all_y_pred_arg)
        tmp_true = np.reshape(all_y_true, -1)
        tmp_pred = np.reshape(all_y_pred_arg, -1)
        return (
            all_y_true,
            all_y_pred_arg,
            mean_iou_metric,
            tmp_true,
            tmp_pred,
            conf_matrix_start_point,
        )

    def test_big_chunk(self):
        # Init. process with dtype=None which will default to int
        (
            all_y_true,
            all_y_pred_arg,
            mean_iou_metric_all,
            tmp_true,
            tmp_pred,
            conf_matrix_start_point,
        ) = self._get_big_chunk(dtype=None)
        conf_matrix_from_keras = np.array(mean_iou_metric_all.total_cm)
        # Validate confusion matrices and results
        conf_matrix_manual = (
            self._confusion_matrix(tmp_true, tmp_pred, 3)
            + conf_matrix_start_point
        )
        self.assertTrue(
            np.array_equal(conf_matrix_from_keras, conf_matrix_manual),
            msg="Confusion matrices do not match!",
        )
        # Now same but with float32 dtype, in here the confusion matrix
        # should not match. Likely this can be removed
        (
            all_y_true,
            all_y_pred_arg,
            mean_iou_metric_all,
            tmp_true,
            tmp_pred,
            conf_matrix_start_point,
        ) = self._get_big_chunk(dtype="float32")
        conf_matrix_from_keras = np.array(mean_iou_metric_all.total_cm)
        # Validate confusion matrices and results
        conf_matrix_manual = (
            self._confusion_matrix(tmp_true, tmp_pred, 3)
            + conf_matrix_start_point
        )
        self.assertFalse(
            np.array_equal(conf_matrix_from_keras, conf_matrix_manual),
            msg="Confusion matrices match, but they should not!",
        )

    def test_user_warning_float_weight(self):
        y_pred = [0, 1, 1, 1]
        y_true = [0, 1, 1, 0]
        m_obj = metrics.MeanIoU(num_classes=3)
        with pytest.warns(Warning, match=r"weight.*float.*int.*casting"):
            m_obj(y_true, y_pred, sample_weight=np.array([0.2, 0.3, 0.4, 0.1]))


class OneHotIoUTest(testing.TestCase):
    def test_unweighted(self):
        y_true = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
        # y_true will be converted to [2, 0, 1, 0]
        y_pred = np.array(
            [[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.5, 0.3, 0.1], [0.1, 0.4, 0.5]]
        )
        # y_pred will be converted to [2, 2, 0, 2]
        # cm = [[0, 0, 2],
        #       [1, 0, 0],
        #       [0, 0, 1]
        # sum_row = [1, 0, 3], sum_col = [2, 1, 1], true_positives = [0, 0, 1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (0 / (1 + 2 - 0) + 1 / (3 + 1 - 1)) / 2
        obj = metrics.OneHotIoU(num_classes=3, target_class_ids=[0, 2])
        result = obj(y_true, y_pred)
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_weighted(self):
        y_true = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
        # y_true will be converted to [2, 0, 1, 0]
        y_pred = np.array(
            [[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.5, 0.3, 0.1], [0.1, 0.4, 0.5]]
        )
        # y_pred will be converted to [2, 2, 0, 2]
        sample_weight = [0.1, 0.2, 0.3, 0.4]
        # cm = [[0, 0, 0.2+0.4],
        #       [0.3, 0, 0],
        #       [0, 0, 0.1]]
        # sum_row = [0.3, 0, 0.7], sum_col = [0.6, 0.3, 0.1]
        # true_positives = [0, 0, 0.1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (0 / (0.3 + 0.6 - 0) + 0.1 / (0.7 + 0.1 - 0.1)) / 2
        obj = metrics.OneHotIoU(
            num_classes=3, target_class_ids=[0, 2], dtype="float32"
        )
        result = obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(result, expected_result, atol=1e-3)


class OneHotMeanIoUTest(testing.TestCase):
    def test_unweighted(self):
        y_true = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
        # y_true will be converted to [2, 0, 1, 0]
        y_pred = np.array(
            [[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.5, 0.3, 0.1], [0.1, 0.4, 0.5]]
        )
        # y_pred will be converted to [2, 2, 0, 2]
        # cm = [[0, 0, 2],
        #       [1, 0, 0],
        #       [0, 0, 1]
        # sum_row = [1, 0, 3], sum_col = [2, 1, 1], true_positives = [0, 0, 1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (0 + 0 + 1 / (3 + 1 - 1)) / 3
        obj = metrics.OneHotMeanIoU(num_classes=3)
        result = obj(y_true, y_pred)
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_weighted(self):
        y_true = np.array(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 0, 0],
            ]
        )
        # y_true will be converted to [2, 0, 1, 0, 0]
        y_pred = np.array(
            [
                [0.2, 0.3, 0.5],
                [0.1, 0.2, 0.7],
                [0.5, 0.3, 0.1],
                [0.1, 0.4, 0.5],
                [0.6, 0.2, 0.2],
            ]
        )
        # y_pred will be converted to [2, 2, 0, 2, 0]
        sample_weight = [0.1, 0.2, 0.3, 0.3, 0.1]
        # cm = [[0.1, 0, 0.2+0.3],
        #       [0.3, 0, 0],
        #       [0, 0, 0.1]]
        # sum_row = [0.4, 0, 0.6], sum_col = [0.6, 0.3, 0.1]
        # true_positives = [0.1, 0, 0.1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            0.1 / (0.4 + 0.6 - 0.1) + 0 + 0.1 / (0.6 + 0.1 - 0.1)
        ) / 3
        obj = metrics.OneHotMeanIoU(num_classes=3, dtype="float32")
        result = obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(result, expected_result, atol=1e-3)

        # Check same result with int weights
        sample_weight_int = [1, 2, 3, 3, 1]
        obj_int = metrics.OneHotMeanIoU(num_classes=3)
        result_int = obj_int(y_true, y_pred, sample_weight=sample_weight_int)
        self.assertAllClose(result_int, expected_result, atol=1e-3)
