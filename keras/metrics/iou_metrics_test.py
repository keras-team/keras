# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Keras metrics."""

import tensorflow.compat.v2 as tf

from keras import metrics
from keras.testing_infra import test_combinations


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class IoUTest(tf.test.TestCase):
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
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))

        result = obj(y_true, y_pred)

        # cm = [[1, 1],
        #       [1, 1]]
        # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

    def test_weighted(self):
        y_pred = tf.constant([0, 1, 0, 1], dtype=tf.float32)
        y_true = tf.constant([0, 0, 1, 1])
        sample_weight = tf.constant([0.2, 0.3, 0.4, 0.1])

        obj = metrics.IoU(num_classes=2, target_class_ids=[1, 0])
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))

        result = obj(y_true, y_pred, sample_weight=sample_weight)

        # cm = [[0.2, 0.3],
        #       [0.4, 0.1]]
        # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5], true_positives = [0.2,
        # 0.1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            0.1 / (0.4 + 0.5 - 0.1) + 0.2 / (0.6 + 0.5 - 0.2)
        ) / 2
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

    def test_multi_dim_input(self):
        y_pred = tf.constant([[0, 1], [0, 1]], dtype=tf.float32)
        y_true = tf.constant([[0, 0], [1, 1]])
        sample_weight = tf.constant([[0.2, 0.3], [0.4, 0.1]])

        obj = metrics.IoU(num_classes=2, target_class_ids=[0, 1])
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))

        result = obj(y_true, y_pred, sample_weight=sample_weight)

        # cm = [[0.2, 0.3],
        #       [0.4, 0.1]]
        # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5], true_positives = [0.2,
        # 0.1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            0.2 / (0.6 + 0.5 - 0.2) + 0.1 / (0.4 + 0.5 - 0.1)
        ) / 2
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

    def test_zero_valid_entries(self):
        obj = metrics.IoU(num_classes=2, target_class_ids=[0, 1])
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))
        self.assertAllClose(self.evaluate(obj.result()), 0, atol=1e-3)

    def test_zero_and_non_zero_entries(self):
        y_pred = tf.constant([1], dtype=tf.float32)
        y_true = tf.constant([1])

        obj = metrics.IoU(num_classes=2, target_class_ids=[0, 1])
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))
        result = obj(y_true, y_pred)

        # cm = [[0, 0],
        #       [0, 1]]
        # sum_row = [0, 1], sum_col = [0, 1], true_positives = [0, 1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (1 / (1 + 1 - 1)) / 1
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class BinaryIoUTest(tf.test.TestCase):
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

        sample_weight = tf.constant([0.2, 0.3, 0.4, 0.1])
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
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))
        result = obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

        sample_weight = tf.constant([0.1, 0.2, 0.4, 0.3])
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
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))
        result = obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

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
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))
        result = obj(y_true, y_pred)
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

        # with threshold = 0.5, y_pred will be converted to [0, 0, 0, 1]
        # cm = [[2, 0],
        #       [1, 1]]
        # sum_row = [2, 2], sum_col = [3, 1], true_positives = [2, 1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (2 / (2 + 3 - 2) + 1 / (2 + 1 - 1)) / 2
        obj = metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5)
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))
        result = obj(y_true, y_pred)
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

    def test_multi_dim_input(self):
        y_true = tf.constant([[0, 1], [0, 1]], dtype=tf.float32)
        y_pred = tf.constant([[0.1, 0.7], [0.9, 0.3]])
        threshold = 0.4  # y_pred will become [[0, 1], [1, 0]]
        sample_weight = tf.constant([[0.2, 0.3], [0.4, 0.1]])
        # cm = [[0.2, 0.4],
        #       [0.1, 0.3]]
        # sum_row = [0.6, 0.4], sum_col = [0.3, 0.7], true_positives = [0.2,
        # 0.3]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            0.2 / (0.6 + 0.3 - 0.2) + 0.3 / (0.4 + 0.7 - 0.3)
        ) / 2
        obj = metrics.BinaryIoU(target_class_ids=[0, 1], threshold=threshold)
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))
        result = obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

    def test_zero_valid_entries(self):
        obj = metrics.BinaryIoU(target_class_ids=[0, 1])
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))
        self.assertAllClose(self.evaluate(obj.result()), 0, atol=1e-3)

    def test_zero_and_non_zero_entries(self):
        y_pred = tf.constant([0.6], dtype=tf.float32)
        threshold = 0.5
        y_true = tf.constant([1])

        obj = metrics.BinaryIoU(target_class_ids=[0, 1], threshold=threshold)
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))
        result = obj(y_true, y_pred)

        # cm = [[0, 0],
        #       [0, 1]]
        # sum_row = [0, 1], sum_col = [0, 1], true_positives = [0, 1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = 1 / (1 + 1 - 1)
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class MeanIoUTest(tf.test.TestCase):
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
        self.evaluate(tf.compat.v1.variables_initializer(m_obj.variables))

        result = m_obj(y_true, y_pred)

        # cm = [[1, 1],
        #       [1, 1]]
        # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

    def test_unweighted_ignore_class_255(self):
        y_pred = [0, 1, 1, 1]
        y_true = [0, 1, 2, 255]

        m_obj = metrics.MeanIoU(num_classes=3, ignore_class=255)
        self.evaluate(tf.compat.v1.variables_initializer(m_obj.variables))

        result = m_obj(y_true, y_pred)

        # cm = [[1, 0, 0],
        #       [0, 1, 0],
        #       [0, 1, 0]]
        # sum_row = [1, 1, 1], sum_col = [1, 2, 0], true_positives = [1, 1, 0]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            1 / (1 + 1 - 1) + 1 / (2 + 1 - 1) + 0 / (0 + 1 - 0)
        ) / 3
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

    def test_unweighted_ignore_class_1(self):
        y_pred = [0, 1, 1, 1]
        y_true = [0, 1, 2, -1]

        m_obj = metrics.MeanIoU(num_classes=3, ignore_class=-1)
        self.evaluate(tf.compat.v1.variables_initializer(m_obj.variables))

        result = m_obj(y_true, y_pred)

        # cm = [[1, 0, 0],
        #       [0, 1, 0],
        #       [0, 1, 0]]
        # sum_row = [1, 1, 1], sum_col = [1, 2, 0], true_positives = [1, 1, 0]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            1 / (1 + 1 - 1) + 1 / (2 + 1 - 1) + 0 / (0 + 1 - 0)
        ) / 3
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

    def test_weighted(self):
        y_pred = tf.constant([0, 1, 0, 1], dtype=tf.float32)
        y_true = tf.constant([0, 0, 1, 1])
        sample_weight = tf.constant([0.2, 0.3, 0.4, 0.1])

        m_obj = metrics.MeanIoU(num_classes=2)
        self.evaluate(tf.compat.v1.variables_initializer(m_obj.variables))

        result = m_obj(y_true, y_pred, sample_weight=sample_weight)

        # cm = [[0.2, 0.3],
        #       [0.4, 0.1]]
        # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5], true_positives = [0.2,
        # 0.1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            0.2 / (0.6 + 0.5 - 0.2) + 0.1 / (0.4 + 0.5 - 0.1)
        ) / 2
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

    def test_weighted_ignore_class_1(self):
        y_pred = tf.constant([0, 1, 0, 1], dtype=tf.float32)
        y_true = tf.constant([0, 0, 1, -1])
        sample_weight = tf.constant([0.2, 0.3, 0.4, 0.1])

        m_obj = metrics.MeanIoU(num_classes=2, ignore_class=-1)
        self.evaluate(tf.compat.v1.variables_initializer(m_obj.variables))

        result = m_obj(y_true, y_pred, sample_weight=sample_weight)

        # cm = [[0.2, 0.3],
        #       [0.4, 0.0]]
        # sum_row = [0.6, 0.3], sum_col = [0.5, 0.4], true_positives = [0.2,
        # 0.0]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            0.2 / (0.6 + 0.5 - 0.2) + 0.0 / (0.3 + 0.4 - 0.0)
        ) / 2
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

    def test_multi_dim_input(self):
        y_pred = tf.constant([[0, 1], [0, 1]], dtype=tf.float32)
        y_true = tf.constant([[0, 0], [1, 1]])
        sample_weight = tf.constant([[0.2, 0.3], [0.4, 0.1]])

        m_obj = metrics.MeanIoU(num_classes=2)
        self.evaluate(tf.compat.v1.variables_initializer(m_obj.variables))

        result = m_obj(y_true, y_pred, sample_weight=sample_weight)

        # cm = [[0.2, 0.3],
        #       [0.4, 0.1]]
        # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5], true_positives = [0.2,
        # 0.1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            0.2 / (0.6 + 0.5 - 0.2) + 0.1 / (0.4 + 0.5 - 0.1)
        ) / 2
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

    def test_zero_valid_entries(self):
        m_obj = metrics.MeanIoU(num_classes=2)
        self.evaluate(tf.compat.v1.variables_initializer(m_obj.variables))
        self.assertAllClose(self.evaluate(m_obj.result()), 0, atol=1e-3)

    def test_zero_and_non_zero_entries(self):
        y_pred = tf.constant([1], dtype=tf.float32)
        y_true = tf.constant([1])

        m_obj = metrics.MeanIoU(num_classes=2)
        self.evaluate(tf.compat.v1.variables_initializer(m_obj.variables))
        result = m_obj(y_true, y_pred)

        # cm = [[0, 0],
        #       [0, 1]]
        # sum_row = [0, 1], sum_col = [0, 1], true_positives = [0, 1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (0 + 1 / (1 + 1 - 1)) / 1
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class OneHotIoUTest(tf.test.TestCase):
    def test_unweighted(self):
        y_true = tf.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
        # y_true will be converted to [2, 0, 1, 0]
        y_pred = tf.constant(
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
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))
        result = obj(y_true, y_pred)
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

    def test_weighted(self):
        y_true = tf.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
        # y_true will be converted to [2, 0, 1, 0]
        y_pred = tf.constant(
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
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))
        result = obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class OneHotMeanIoUTest(tf.test.TestCase):
    def test_unweighted(self):
        y_true = tf.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
        # y_true will be converted to [2, 0, 1, 0]
        y_pred = tf.constant(
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
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))
        result = obj(y_true, y_pred)
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

    def test_weighted(self):
        y_true = tf.constant(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 0, 0],
            ]
        )
        # y_true will be converted to [2, 0, 1, 0, 0]
        y_pred = tf.constant(
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
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))
        result = obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)


if __name__ == "__main__":
    tf.test.main()
