import numpy as np
import pytest

from keras import layers
from keras import models
from keras import testing
from keras.metrics import iou_metrics as metrics


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

        obj = metrics.IoU(
            num_classes=2, target_class_ids=[0, 1], dtype="float32"
        )

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

        obj = metrics.IoU(num_classes=2, target_class_ids=[0, 1])

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
        m_obj = metrics.MeanIoU(num_classes=2)
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
        obj = metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.3)
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
        obj = metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5)
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
        obj = metrics.BinaryIoU(target_class_ids=[0, 1], threshold=threshold)
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

        m_obj = metrics.MeanIoU(num_classes=2)

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

        m_obj = metrics.MeanIoU(num_classes=2, ignore_class=-1)

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

        m_obj = metrics.MeanIoU(num_classes=2)

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
        obj = metrics.OneHotIoU(num_classes=3, target_class_ids=[0, 2])
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
        obj = metrics.OneHotMeanIoU(num_classes=3)
        result = obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(result, expected_result, atol=1e-3)
