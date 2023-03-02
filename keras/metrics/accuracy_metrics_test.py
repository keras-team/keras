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
"""Tests for accuracy metrics."""

import tensorflow.compat.v2 as tf

from keras import Model
from keras import layers
from keras import metrics
from keras.testing_infra import test_combinations


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class AccuracyTest(tf.test.TestCase):
    def test_accuracy(self):
        acc_obj = metrics.Accuracy(name="my_acc")

        # check config
        self.assertEqual(acc_obj.name, "my_acc")
        self.assertTrue(acc_obj.stateful)
        self.assertEqual(len(acc_obj.variables), 2)
        self.assertEqual(acc_obj.dtype, tf.float32)
        self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))

        # verify that correct value is returned
        update_op = acc_obj.update_state(
            [[1], [2], [3], [4]], [[1], [2], [3], [4]]
        )
        self.evaluate(update_op)
        result = self.evaluate(acc_obj.result())
        self.assertEqual(result, 1)  # 2/2

        # Check save and restore config
        a2 = metrics.Accuracy.from_config(acc_obj.get_config())
        self.assertEqual(a2.name, "my_acc")
        self.assertTrue(a2.stateful)
        self.assertEqual(len(a2.variables), 2)
        self.assertEqual(a2.dtype, tf.float32)

        # check with sample_weight
        result_t = acc_obj([[2], [1]], [[2], [0]], sample_weight=[[0.5], [0.2]])
        result = self.evaluate(result_t)
        self.assertAlmostEqual(result, 0.96, 2)  # 4.5/4.7

    def test_accuracy_ragged(self):
        acc_obj = metrics.Accuracy(name="my_acc")
        self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))

        # verify that correct value is returned
        rt1 = tf.ragged.constant([[1], [2], [3], [4]])
        rt2 = tf.ragged.constant([[1], [2], [3], [4]])
        update_op = acc_obj.update_state(rt1, rt2)
        self.evaluate(update_op)
        result = self.evaluate(acc_obj.result())
        self.assertEqual(result, 1)  # 2/2

        # check with sample_weight
        rt1 = tf.ragged.constant([[2], [1]])
        rt2 = tf.ragged.constant([[2], [0]])
        sw_ragged = tf.ragged.constant([[0.5], [0.2]])
        result_t = acc_obj(rt1, rt2, sample_weight=sw_ragged)
        result = self.evaluate(result_t)
        self.assertAlmostEqual(result, 0.96, 2)  # 4.5/4.7

    def test_binary_accuracy(self):
        acc_obj = metrics.BinaryAccuracy(name="my_acc")

        # check config
        self.assertEqual(acc_obj.name, "my_acc")
        self.assertTrue(acc_obj.stateful)
        self.assertEqual(len(acc_obj.variables), 2)
        self.assertEqual(acc_obj.dtype, tf.float32)
        self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))

        # verify that correct value is returned
        update_op = acc_obj.update_state([[1], [0]], [[1], [0]])
        self.evaluate(update_op)
        result = self.evaluate(acc_obj.result())
        self.assertEqual(result, 1)  # 2/2

        # check y_pred squeeze
        update_op = acc_obj.update_state([[1], [1]], [[[1]], [[0]]])
        self.evaluate(update_op)
        result = self.evaluate(acc_obj.result())
        self.assertAlmostEqual(result, 0.75, 2)  # 3/4

        # check y_true squeeze
        result_t = acc_obj([[[1]], [[1]]], [[1], [0]])
        result = self.evaluate(result_t)
        self.assertAlmostEqual(result, 0.67, 2)  # 4/6

        # check with sample_weight
        result_t = acc_obj([[1], [1]], [[1], [0]], [[0.5], [0.2]])
        result = self.evaluate(result_t)
        self.assertAlmostEqual(result, 0.67, 2)  # 4.5/6.7

    def test_binary_accuracy_ragged(self):
        acc_obj = metrics.BinaryAccuracy(name="my_acc")
        self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))

        # verify that correct value is returned
        rt1 = tf.ragged.constant([[1], [0]])
        rt2 = tf.ragged.constant([[1], [0]])
        update_op = acc_obj.update_state(rt1, rt2)
        self.evaluate(update_op)
        result = self.evaluate(acc_obj.result())
        self.assertEqual(result, 1)  # 2/2

        # check y_true squeeze only supported for dense tensors and is
        # not supported by ragged tensor (different ranks). --> error
        rt1 = tf.ragged.constant([[[1], [1]]])
        rt2 = tf.ragged.constant([[1], [0]])
        with self.assertRaises(ValueError):
            result_t = acc_obj(rt1, rt2)
            result = self.evaluate(result_t)

    def test_binary_accuracy_threshold(self):
        acc_obj = metrics.BinaryAccuracy(threshold=0.7)
        self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))
        result_t = acc_obj([[1], [1], [0], [0]], [[0.9], [0.6], [0.4], [0.8]])
        result = self.evaluate(result_t)
        self.assertAlmostEqual(result, 0.5, 2)

    def test_binary_accuracy_threshold_ragged(self):
        acc_obj = metrics.BinaryAccuracy(threshold=0.7)
        self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))
        rt1 = tf.ragged.constant([[1], [1], [0], [0]])
        rt2 = tf.ragged.constant([[0.9], [0.6], [0.4], [0.8]])
        result_t = acc_obj(rt1, rt2)
        result = self.evaluate(result_t)
        self.assertAlmostEqual(result, 0.5, 2)

    def test_categorical_accuracy(self):
        acc_obj = metrics.CategoricalAccuracy(name="my_acc")

        # check config
        self.assertEqual(acc_obj.name, "my_acc")
        self.assertTrue(acc_obj.stateful)
        self.assertEqual(len(acc_obj.variables), 2)
        self.assertEqual(acc_obj.dtype, tf.float32)
        self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))

        # verify that correct value is returned
        update_op = acc_obj.update_state(
            [[0, 0, 1], [0, 1, 0]], [[0.1, 0.1, 0.8], [0.05, 0.95, 0]]
        )
        self.evaluate(update_op)
        result = self.evaluate(acc_obj.result())
        self.assertEqual(result, 1)  # 2/2

        # check with sample_weight
        result_t = acc_obj(
            [[0, 0, 1], [0, 1, 0]],
            [[0.1, 0.1, 0.8], [0.05, 0, 0.95]],
            [[0.5], [0.2]],
        )
        result = self.evaluate(result_t)
        self.assertAlmostEqual(result, 0.93, 2)  # 2.5/2.7

    def test_categorical_accuracy_ragged(self):
        acc_obj = metrics.CategoricalAccuracy(name="my_acc")
        self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))

        # verify that correct value is returned
        rt1 = tf.ragged.constant([[0, 0, 1], [0, 1, 0]])
        rt2 = tf.ragged.constant([[0.1, 0.1, 0.8], [0.05, 0.95, 0]])
        update_op = acc_obj.update_state(rt1, rt2)
        self.evaluate(update_op)
        result = self.evaluate(acc_obj.result())
        self.assertEqual(result, 1)  # 2/2

        # check with sample_weight
        rt1 = tf.ragged.constant([[0, 0, 1], [0, 1, 0]])
        rt2 = tf.ragged.constant([[0.1, 0.1, 0.8], [0.05, 0, 0.95]])
        sample_weight = tf.ragged.constant([[0.5], [0.2]])
        with self.assertRaises(tf.errors.InvalidArgumentError):
            result_t = acc_obj(rt1, rt2, sample_weight)
            result = self.evaluate(result_t)

    def test_sparse_categorical_accuracy(self):
        acc_obj = metrics.SparseCategoricalAccuracy(name="my_acc")

        # check config
        self.assertEqual(acc_obj.name, "my_acc")
        self.assertTrue(acc_obj.stateful)
        self.assertEqual(len(acc_obj.variables), 2)
        self.assertEqual(acc_obj.dtype, tf.float32)
        self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))

        # verify that correct value is returned
        update_op = acc_obj.update_state(
            [[2], [1]], [[0.1, 0.1, 0.8], [0.05, 0.95, 0]]
        )
        self.evaluate(update_op)
        result = self.evaluate(acc_obj.result())
        self.assertEqual(result, 1)  # 2/2

        # check with sample_weight
        result_t = acc_obj(
            [[2], [1]], [[0.1, 0.1, 0.8], [0.05, 0, 0.95]], [[0.5], [0.2]]
        )
        result = self.evaluate(result_t)
        self.assertAlmostEqual(result, 0.93, 2)  # 2.5/2.7

    def test_sparse_categorical_accuracy_ragged(self):
        acc_obj = metrics.SparseCategoricalAccuracy(name="my_acc")

        # verify that correct value is returned
        rt1 = tf.ragged.constant([[2], [1]])
        rt2 = tf.ragged.constant([[0.1, 0.1, 0.8], [0.05, 0.95, 0]])

        with self.assertRaises(tf.errors.InvalidArgumentError):
            # sparse_categorical_accuracy is not supported for composite/ragged
            # tensors.
            update_op = acc_obj.update_state(rt1, rt2)
            self.evaluate(update_op)

    def test_sparse_categorical_accuracy_mismatched_dims(self):
        acc_obj = metrics.SparseCategoricalAccuracy(name="my_acc")

        # check config
        self.assertEqual(acc_obj.name, "my_acc")
        self.assertTrue(acc_obj.stateful)
        self.assertEqual(len(acc_obj.variables), 2)
        self.assertEqual(acc_obj.dtype, tf.float32)
        self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))

        # verify that correct value is returned
        update_op = acc_obj.update_state(
            [2, 1], [[0.1, 0.1, 0.8], [0.05, 0.95, 0]]
        )
        self.evaluate(update_op)
        result = self.evaluate(acc_obj.result())
        self.assertEqual(result, 1)  # 2/2

        # check with sample_weight
        result_t = acc_obj(
            [2, 1], [[0.1, 0.1, 0.8], [0.05, 0, 0.95]], [[0.5], [0.2]]
        )
        result = self.evaluate(result_t)
        self.assertAlmostEqual(result, 0.93, 2)  # 2.5/2.7

    def test_sparse_categorical_accuracy_mismatched_dims_dynamic(self):
        with tf.compat.v1.get_default_graph().as_default(), self.cached_session() as sess:  # noqa: E501
            acc_obj = metrics.SparseCategoricalAccuracy(name="my_acc")
            self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))

            t = tf.compat.v1.placeholder(tf.float32)
            p = tf.compat.v1.placeholder(tf.float32)
            w = tf.compat.v1.placeholder(tf.float32)

            result_t = acc_obj(t, p, w)
            result = sess.run(
                result_t,
                feed_dict=(
                    {
                        t: [2, 1],
                        p: [[0.1, 0.1, 0.8], [0.05, 0, 0.95]],
                        w: [[0.5], [0.2]],
                    }
                ),
            )
            self.assertAlmostEqual(result, 0.71, 2)  # 2.5/2.7

    def test_get_acc(self):
        acc_fn = metrics.get("acc")
        self.assertEqual(acc_fn, metrics.accuracy)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class TopKCategoricalAccuracyTest(tf.test.TestCase):
    def test_config(self):
        a_obj = metrics.TopKCategoricalAccuracy(name="topkca", dtype=tf.int32)
        self.assertEqual(a_obj.name, "topkca")
        self.assertEqual(a_obj._dtype, tf.int32)

        a_obj2 = metrics.TopKCategoricalAccuracy.from_config(a_obj.get_config())
        self.assertEqual(a_obj2.name, "topkca")
        self.assertEqual(a_obj2._dtype, tf.int32)

    def test_correctness(self):
        a_obj = metrics.TopKCategoricalAccuracy()
        self.evaluate(tf.compat.v1.variables_initializer(a_obj.variables))
        y_true = tf.constant([[0, 0, 1], [0, 1, 0]])
        y_pred = tf.constant([[0.1, 0.9, 0.8], [0.05, 0.95, 0]])

        result = a_obj(y_true, y_pred)
        self.assertEqual(1, self.evaluate(result))  # both the samples match

        # With `k` < 5.
        a_obj = metrics.TopKCategoricalAccuracy(k=1)
        self.evaluate(tf.compat.v1.variables_initializer(a_obj.variables))
        result = a_obj(y_true, y_pred)
        self.assertEqual(0.5, self.evaluate(result))  # only sample #2 matches

        # With `k` > 5.
        y_true = tf.constant([[0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]])
        y_pred = tf.constant(
            [[0.5, 0.9, 0.1, 0.7, 0.6, 0.5, 0.4], [0.05, 0.95, 0, 0, 0, 0, 0]]
        )
        a_obj = metrics.TopKCategoricalAccuracy(k=6)
        self.evaluate(tf.compat.v1.variables_initializer(a_obj.variables))
        result = a_obj(y_true, y_pred)
        self.assertEqual(0.5, self.evaluate(result))  # only 1 sample matches.

    def test_weighted(self):
        a_obj = metrics.TopKCategoricalAccuracy(k=2)
        self.evaluate(tf.compat.v1.variables_initializer(a_obj.variables))
        y_true = tf.constant([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        y_pred = tf.constant([[0, 0.9, 0.1], [0, 0.9, 0.1], [0, 0.9, 0.1]])
        sample_weight = tf.constant((1.0, 0.0, 1.0))
        result = a_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(1.0, self.evaluate(result), atol=1e-5)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class SparseTopKCategoricalAccuracyTest(tf.test.TestCase):
    def test_config(self):
        a_obj = metrics.SparseTopKCategoricalAccuracy(
            name="stopkca", dtype=tf.int32
        )
        self.assertEqual(a_obj.name, "stopkca")
        self.assertEqual(a_obj._dtype, tf.int32)

        a_obj2 = metrics.SparseTopKCategoricalAccuracy.from_config(
            a_obj.get_config()
        )
        self.assertEqual(a_obj2.name, "stopkca")
        self.assertEqual(a_obj2._dtype, tf.int32)

    def test_correctness(self):
        a_obj = metrics.SparseTopKCategoricalAccuracy()
        self.evaluate(tf.compat.v1.variables_initializer(a_obj.variables))
        y_true = tf.constant([2, 1])
        y_pred = tf.constant([[0.1, 0.9, 0.8], [0.05, 0.95, 0]])

        result = a_obj(y_true, y_pred)
        self.assertEqual(1, self.evaluate(result))  # both the samples match

        # With `k` < 5.
        a_obj = metrics.SparseTopKCategoricalAccuracy(k=1)
        self.evaluate(tf.compat.v1.variables_initializer(a_obj.variables))
        result = a_obj(y_true, y_pred)
        self.assertEqual(0.5, self.evaluate(result))  # only sample #2 matches

        # With `k` > 5.
        y_pred = tf.constant(
            [[0.5, 0.9, 0.1, 0.7, 0.6, 0.5, 0.4], [0.05, 0.95, 0, 0, 0, 0, 0]]
        )
        a_obj = metrics.SparseTopKCategoricalAccuracy(k=6)
        self.evaluate(tf.compat.v1.variables_initializer(a_obj.variables))
        result = a_obj(y_true, y_pred)
        self.assertEqual(0.5, self.evaluate(result))  # only 1 sample matches.

    def test_weighted(self):
        a_obj = metrics.SparseTopKCategoricalAccuracy(k=2)
        self.evaluate(tf.compat.v1.variables_initializer(a_obj.variables))
        y_true = tf.constant([1, 0, 2])
        y_pred = tf.constant([[0, 0.9, 0.1], [0, 0.9, 0.1], [0, 0.9, 0.1]])
        sample_weight = tf.constant((1.0, 0.0, 1.0))
        result = a_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(1.0, self.evaluate(result), atol=1e-5)

    def test_sparse_top_k_categorical_accuracy_mismatched_dims_dynamic(self):

        if not tf.compat.v1.executing_eagerly():
            # Test will fail in v1 graph mode since the metric is not a normal
            # layer.  It will aggregate the output by batch dim, which failed on
            # v1 code.
            self.skipTest("v2 eager mode only")

        class AccLayer(layers.Layer):
            def build(self, _):
                self.acc = metrics.SparseTopKCategoricalAccuracy(k=1)

            def call(self, y_true, y_pred):
                return self.acc(y_true, y_pred)

        label = layers.Input(shape=[1])
        predict = layers.Input(shape=[3])
        metric_result = AccLayer()(label, predict)
        model = Model([label, predict], metric_result)

        result = model.predict(
            [
                tf.constant([[2], [1]]),
                tf.constant([[0.1, 0.1, 0.8], [0.05, 0, 0.95]]),
            ],
            steps=1,
        )
        self.assertAllClose(result, 0.5)


if __name__ == "__main__":
    tf.test.main()
