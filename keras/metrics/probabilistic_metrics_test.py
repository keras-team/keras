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

import json

import numpy as np
import tensorflow.compat.v2 as tf

from keras import metrics
from keras.testing_infra import test_combinations


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class PoissonTest(tf.test.TestCase):
    def setup(self):
        y_pred = np.asarray([1, 9, 2, 5, 2, 6]).reshape((2, 3))
        y_true = np.asarray([4, 8, 12, 8, 1, 3]).reshape((2, 3))

        self.batch_size = 6
        self.expected_results = y_pred - np.multiply(y_true, np.log(y_pred))

        self.y_pred = tf.constant(y_pred, dtype=tf.float32)
        self.y_true = tf.constant(y_true)

    def test_config(self):
        poisson_obj = metrics.Poisson(name="poisson", dtype=tf.int32)
        self.assertEqual(poisson_obj.name, "poisson")
        self.assertEqual(poisson_obj._dtype, tf.int32)

        poisson_obj2 = metrics.Poisson.from_config(poisson_obj.get_config())
        self.assertEqual(poisson_obj2.name, "poisson")
        self.assertEqual(poisson_obj2._dtype, tf.int32)

    def test_unweighted(self):
        self.setup()
        poisson_obj = metrics.Poisson()
        self.evaluate(tf.compat.v1.variables_initializer(poisson_obj.variables))

        update_op = poisson_obj.update_state(self.y_true, self.y_pred)
        self.evaluate(update_op)
        result = poisson_obj.result()
        expected_result = np.sum(self.expected_results) / self.batch_size
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_weighted(self):
        self.setup()
        poisson_obj = metrics.Poisson()
        self.evaluate(tf.compat.v1.variables_initializer(poisson_obj.variables))
        sample_weight = tf.constant([1.2, 3.4], shape=(2, 1))

        result = poisson_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )
        sample_weight = np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape(
            (2, 3)
        )
        expected_result = np.multiply(self.expected_results, sample_weight)
        expected_result = np.sum(expected_result) / np.sum(sample_weight)
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class KLDivergenceTest(tf.test.TestCase):
    def setup(self):
        y_pred = np.asarray([0.4, 0.9, 0.12, 0.36, 0.3, 0.4]).reshape((2, 3))
        y_true = np.asarray([0.5, 0.8, 0.12, 0.7, 0.43, 0.8]).reshape((2, 3))

        self.batch_size = 2
        self.expected_results = np.multiply(y_true, np.log(y_true / y_pred))

        self.y_pred = tf.constant(y_pred, dtype=tf.float32)
        self.y_true = tf.constant(y_true)

    def test_config(self):
        k_obj = metrics.KLDivergence(name="kld", dtype=tf.int32)
        self.assertEqual(k_obj.name, "kld")
        self.assertEqual(k_obj._dtype, tf.int32)

        k_obj2 = metrics.KLDivergence.from_config(k_obj.get_config())
        self.assertEqual(k_obj2.name, "kld")
        self.assertEqual(k_obj2._dtype, tf.int32)

    def test_unweighted(self):
        self.setup()
        k_obj = metrics.KLDivergence()
        self.evaluate(tf.compat.v1.variables_initializer(k_obj.variables))

        update_op = k_obj.update_state(self.y_true, self.y_pred)
        self.evaluate(update_op)
        result = k_obj.result()
        expected_result = np.sum(self.expected_results) / self.batch_size
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_weighted(self):
        self.setup()
        k_obj = metrics.KLDivergence()
        self.evaluate(tf.compat.v1.variables_initializer(k_obj.variables))

        sample_weight = tf.constant([1.2, 3.4], shape=(2, 1))
        result = k_obj(self.y_true, self.y_pred, sample_weight=sample_weight)

        sample_weight = np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape(
            (2, 3)
        )
        expected_result = np.multiply(self.expected_results, sample_weight)
        expected_result = np.sum(expected_result) / (1.2 + 3.4)
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class BinaryCrossentropyTest(tf.test.TestCase):
    def test_config(self):
        bce_obj = metrics.BinaryCrossentropy(
            name="bce", dtype=tf.int32, label_smoothing=0.2
        )
        self.assertEqual(bce_obj.name, "bce")
        self.assertEqual(bce_obj._dtype, tf.int32)

        old_config = bce_obj.get_config()
        self.assertAllClose(old_config["label_smoothing"], 0.2, 1e-3)

        # Check save and restore config
        bce_obj2 = metrics.BinaryCrossentropy.from_config(old_config)
        self.assertEqual(bce_obj2.name, "bce")
        self.assertEqual(bce_obj2._dtype, tf.int32)
        new_config = bce_obj2.get_config()
        self.assertDictEqual(old_config, new_config)

    def test_unweighted(self):
        bce_obj = metrics.BinaryCrossentropy()
        self.evaluate(tf.compat.v1.variables_initializer(bce_obj.variables))
        y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.asarray([1, 1, 1, 0], dtype=np.float32).reshape([2, 2])
        result = bce_obj(y_true, y_pred)

        # EPSILON = 1e-7, y = y_true, y` = y_pred, Y_MAX = 0.9999999
        # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
        # y` = [Y_MAX, Y_MAX, Y_MAX, EPSILON]

        # Metric = -(y log(y` + EPSILON) + (1 - y) log(1 - y` + EPSILON))
        #        = [-log(Y_MAX + EPSILON), -log(1 - Y_MAX + EPSILON),
        #           -log(Y_MAX + EPSILON), -log(1)]
        #        = [(0 + 15.33) / 2, (0 + 0) / 2]
        # Reduced metric = 7.665 / 2

        self.assertAllClose(self.evaluate(result), 3.833, atol=1e-3)

    def test_unweighted_with_logits(self):
        bce_obj = metrics.BinaryCrossentropy(from_logits=True)
        self.evaluate(tf.compat.v1.variables_initializer(bce_obj.variables))

        y_true = tf.constant([[1, 0, 1], [0, 1, 1]])
        y_pred = tf.constant([[100.0, -100.0, 100.0], [100.0, 100.0, -100.0]])
        result = bce_obj(y_true, y_pred)

        # Metric = max(x, 0) - x * z + log(1 + exp(-abs(x)))
        #              (where x = logits and z = y_true)
        #        = [((100 - 100 * 1 + log(1 + exp(-100))) +
        #            (0 + 100 * 0 + log(1 + exp(-100))) +
        #            (100 - 100 * 1 + log(1 + exp(-100))),
        #           ((100 - 100 * 0 + log(1 + exp(-100))) +
        #            (100 - 100 * 1 + log(1 + exp(-100))) +
        #            (0 + 100 * 1 + log(1 + exp(-100))))]
        #        = [(0 + 0 + 0) / 3, 200 / 3]
        # Reduced metric = (0 + 66.666) / 2

        self.assertAllClose(self.evaluate(result), 33.333, atol=1e-3)

    def test_weighted(self):
        bce_obj = metrics.BinaryCrossentropy()
        self.evaluate(tf.compat.v1.variables_initializer(bce_obj.variables))
        y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.asarray([1, 1, 1, 0], dtype=np.float32).reshape([2, 2])
        sample_weight = tf.constant([1.5, 2.0])
        result = bce_obj(y_true, y_pred, sample_weight=sample_weight)

        # EPSILON = 1e-7, y = y_true, y` = y_pred, Y_MAX = 0.9999999
        # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
        # y` = [Y_MAX, Y_MAX, Y_MAX, EPSILON]

        # Metric = -(y log(y` + EPSILON) + (1 - y) log(1 - y` + EPSILON))
        #        = [-log(Y_MAX + EPSILON), -log(1 - Y_MAX + EPSILON),
        #           -log(Y_MAX + EPSILON), -log(1)]
        #        = [(0 + 15.33) / 2, (0 + 0) / 2]
        # Weighted metric = [7.665 * 1.5, 0]
        # Reduced metric = 7.665 * 1.5 / (1.5 + 2)

        self.assertAllClose(self.evaluate(result), 3.285, atol=1e-3)

    def test_weighted_from_logits(self):
        bce_obj = metrics.BinaryCrossentropy(from_logits=True)
        self.evaluate(tf.compat.v1.variables_initializer(bce_obj.variables))
        y_true = tf.constant([[1, 0, 1], [0, 1, 1]])
        y_pred = tf.constant([[100.0, -100.0, 100.0], [100.0, 100.0, -100.0]])
        sample_weight = tf.constant([2.0, 2.5])
        result = bce_obj(y_true, y_pred, sample_weight=sample_weight)

        # Metric = max(x, 0) - x * z + log(1 + exp(-abs(x)))
        #              (where x = logits and z = y_true)
        #        = [(0 + 0 + 0) / 3, 200 / 3]
        # Weighted metric = [0, 66.666 * 2.5]
        # Reduced metric = 66.666 * 2.5 / (2 + 2.5)

        self.assertAllClose(self.evaluate(result), 37.037, atol=1e-3)

    def test_label_smoothing(self):
        logits = tf.constant(((100.0, -100.0, -100.0)))
        y_true = tf.constant(((1, 0, 1)))
        label_smoothing = 0.1
        # Metric: max(x, 0) - x * z + log(1 + exp(-abs(x)))
        #             (where x = logits and z = y_true)
        # Label smoothing: z' = z * (1 - L) + 0.5L
        # After label smoothing, label 1 becomes 1 - 0.5L
        #                        label 0 becomes 0.5L
        # Applying the above two fns to the given input:
        # (100 - 100 * (1 - 0.5 L)  + 0 +
        #  0   + 100 * (0.5 L)      + 0 +
        #  0   + 100 * (1 - 0.5 L)  + 0) * (1/3)
        #  = (100 + 50L) * 1/3
        bce_obj = metrics.BinaryCrossentropy(
            from_logits=True, label_smoothing=label_smoothing
        )
        self.evaluate(tf.compat.v1.variables_initializer(bce_obj.variables))
        result = bce_obj(y_true, logits)
        expected_value = (100.0 + 50.0 * label_smoothing) / 3.0
        self.assertAllClose(expected_value, self.evaluate(result), atol=1e-3)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class CategoricalCrossentropyTest(tf.test.TestCase):
    def test_config(self):
        cce_obj = metrics.CategoricalCrossentropy(
            name="cce", dtype=tf.int32, label_smoothing=0.2
        )
        self.assertEqual(cce_obj.name, "cce")
        self.assertEqual(cce_obj._dtype, tf.int32)

        old_config = cce_obj.get_config()
        self.assertAllClose(old_config["label_smoothing"], 0.2, 1e-3)

        # Check save and restore config
        cce_obj2 = metrics.CategoricalCrossentropy.from_config(old_config)
        self.assertEqual(cce_obj2.name, "cce")
        self.assertEqual(cce_obj2._dtype, tf.int32)
        new_config = cce_obj2.get_config()
        self.assertDictEqual(old_config, new_config)

    def test_unweighted(self):
        cce_obj = metrics.CategoricalCrossentropy()
        self.evaluate(tf.compat.v1.variables_initializer(cce_obj.variables))

        y_true = np.asarray([[0, 1, 0], [0, 0, 1]])
        y_pred = np.asarray([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
        result = cce_obj(y_true, y_pred)

        # EPSILON = 1e-7, y = y_true, y` = y_pred
        # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
        # y` = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]

        # Metric = -sum(y * log(y'), axis = -1)
        #        = -((log 0.95), (log 0.1))
        #        = [0.051, 2.302]
        # Reduced metric = (0.051 + 2.302) / 2

        self.assertAllClose(self.evaluate(result), 1.176, atol=1e-3)

    def test_unweighted_from_logits(self):
        cce_obj = metrics.CategoricalCrossentropy(from_logits=True)
        self.evaluate(tf.compat.v1.variables_initializer(cce_obj.variables))

        y_true = np.asarray([[0, 1, 0], [0, 0, 1]])
        logits = np.asarray([[1, 9, 0], [1, 8, 1]], dtype=np.float32)
        result = cce_obj(y_true, logits)

        # softmax = exp(logits) / sum(exp(logits), axis=-1)
        # xent = -sum(labels * log(softmax), 1)

        # exp(logits) = [[2.718, 8103.084, 1], [2.718, 2980.958, 2.718]]
        # sum(exp(logits), axis=-1) = [8106.802, 2986.394]
        # softmax = [[0.00033, 0.99954, 0.00012], [0.00091, 0.99817, 0.00091]]
        # log(softmax) = [[-8.00045, -0.00045, -9.00045],
        #                 [-7.00182, -0.00182, -7.00182]]
        # labels * log(softmax) = [[0, -0.00045, 0], [0, 0, -7.00182]]
        # xent = [0.00045, 7.00182]
        # Reduced xent = (0.00045 + 7.00182) / 2

        self.assertAllClose(self.evaluate(result), 3.5011, atol=1e-3)

    def test_weighted(self):
        cce_obj = metrics.CategoricalCrossentropy()
        self.evaluate(tf.compat.v1.variables_initializer(cce_obj.variables))

        y_true = np.asarray([[0, 1, 0], [0, 0, 1]])
        y_pred = np.asarray([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
        sample_weight = tf.constant([1.5, 2.0])
        result = cce_obj(y_true, y_pred, sample_weight=sample_weight)

        # EPSILON = 1e-7, y = y_true, y` = y_pred
        # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
        # y` = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]

        # Metric = -sum(y * log(y'), axis = -1)
        #        = -((log 0.95), (log 0.1))
        #        = [0.051, 2.302]
        # Weighted metric = [0.051 * 1.5, 2.302 * 2.]
        # Reduced metric = (0.051 * 1.5 + 2.302 * 2.) / 3.5

        self.assertAllClose(self.evaluate(result), 1.338, atol=1e-3)

    def test_weighted_from_logits(self):
        cce_obj = metrics.CategoricalCrossentropy(from_logits=True)
        self.evaluate(tf.compat.v1.variables_initializer(cce_obj.variables))

        y_true = np.asarray([[0, 1, 0], [0, 0, 1]])
        logits = np.asarray([[1, 9, 0], [1, 8, 1]], dtype=np.float32)
        sample_weight = tf.constant([1.5, 2.0])
        result = cce_obj(y_true, logits, sample_weight=sample_weight)

        # softmax = exp(logits) / sum(exp(logits), axis=-1)
        # xent = -sum(labels * log(softmax), 1)
        # xent = [0.00045, 7.00182]
        # weighted xent = [0.000675, 14.00364]
        # Reduced xent = (0.000675 + 14.00364) / (1.5 + 2)

        self.assertAllClose(self.evaluate(result), 4.0012, atol=1e-3)

    def test_label_smoothing(self):
        y_true = np.asarray([[0, 1, 0], [0, 0, 1]])
        logits = np.asarray([[1, 9, 0], [1, 8, 1]], dtype=np.float32)
        label_smoothing = 0.1

        # Label smoothing: z' = z * (1 - L) + L/n,
        #     where L = label smoothing value and n = num classes
        # Label value 1 becomes: 1 - L + L/n
        # Label value 0 becomes: L/n
        # y_true with label_smoothing = [[0.0333, 0.9333, 0.0333],
        #                               [0.0333, 0.0333, 0.9333]]

        # softmax = exp(logits) / sum(exp(logits), axis=-1)
        # xent = -sum(labels * log(softmax), 1)
        # log(softmax) = [[-8.00045, -0.00045, -9.00045],
        #                 [-7.00182, -0.00182, -7.00182]]
        # labels * log(softmax) = [[-0.26641, -0.00042, -0.29971],
        #                          [-0.23316, -0.00006, -6.53479]]
        # xent = [0.56654, 6.76801]
        # Reduced xent = (0.56654 + 6.76801) / 2

        cce_obj = metrics.CategoricalCrossentropy(
            from_logits=True, label_smoothing=label_smoothing
        )
        self.evaluate(tf.compat.v1.variables_initializer(cce_obj.variables))
        loss = cce_obj(y_true, logits)
        self.assertAllClose(self.evaluate(loss), 3.667, atol=1e-3)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class SparseCategoricalCrossentropyTest(tf.test.TestCase):
    def test_config(self):
        scce_obj = metrics.SparseCategoricalCrossentropy(
            name="scce", dtype=tf.int32
        )
        self.assertEqual(scce_obj.name, "scce")
        self.assertEqual(scce_obj.dtype, tf.int32)
        old_config = scce_obj.get_config()
        self.assertDictEqual(old_config, json.loads(json.dumps(old_config)))

        # Check save and restore config
        scce_obj2 = metrics.SparseCategoricalCrossentropy.from_config(
            old_config
        )
        self.assertEqual(scce_obj2.name, "scce")
        self.assertEqual(scce_obj2.dtype, tf.int32)
        new_config = scce_obj2.get_config()
        self.assertDictEqual(old_config, new_config)

    def test_unweighted(self):
        scce_obj = metrics.SparseCategoricalCrossentropy()
        self.evaluate(tf.compat.v1.variables_initializer(scce_obj.variables))

        y_true = np.asarray([1, 2])
        y_pred = np.asarray([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
        result = scce_obj(y_true, y_pred)

        # EPSILON = 1e-7, y = y_true, y` = y_pred
        # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
        # y` = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
        # logits = log(y`) =  [[-2.9957, -0.0513, -16.1181],
        #                      [-2.3026, -0.2231, -2.3026]]

        # softmax = exp(logits) / sum(exp(logits), axis=-1)
        # y = one_hot(y) = [[0, 1, 0], [0, 0, 1]]
        # xent = -sum(y * log(softmax), 1)

        # exp(logits) = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
        # sum(exp(logits), axis=-1) = [1, 1]
        # softmax = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
        # log(softmax) = [[-2.9957, -0.0513, -16.1181],
        #                 [-2.3026, -0.2231, -2.3026]]
        # y * log(softmax) = [[0, -0.0513, 0], [0, 0, -2.3026]]
        # xent = [0.0513, 2.3026]
        # Reduced xent = (0.0513 + 2.3026) / 2

        self.assertAllClose(self.evaluate(result), 1.176, atol=1e-3)

    def test_unweighted_ignore_class(self):
        scce_obj = metrics.SparseCategoricalCrossentropy(ignore_class=-1)
        self.evaluate(tf.compat.v1.variables_initializer(scce_obj.variables))

        y_true = np.asarray([-1, 2])
        y_pred = np.asarray([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
        result = scce_obj(y_true, y_pred)

        self.assertAllClose(self.evaluate(result), 2.3026, atol=1e-3)

    def test_unweighted_from_logits(self):
        scce_obj = metrics.SparseCategoricalCrossentropy(from_logits=True)
        self.evaluate(tf.compat.v1.variables_initializer(scce_obj.variables))

        y_true = np.asarray([1, 2])
        logits = np.asarray([[1, 9, 0], [1, 8, 1]], dtype=np.float32)
        result = scce_obj(y_true, logits)

        # softmax = exp(logits) / sum(exp(logits), axis=-1)
        # y_true = one_hot(y_true) = [[0, 1, 0], [0, 0, 1]]
        # xent = -sum(y_true * log(softmax), 1)

        # exp(logits) = [[2.718, 8103.084, 1], [2.718, 2980.958, 2.718]]
        # sum(exp(logits), axis=-1) = [8106.802, 2986.394]
        # softmax = [[0.00033, 0.99954, 0.00012], [0.00091, 0.99817, 0.00091]]
        # log(softmax) = [[-8.00045, -0.00045, -9.00045],
        #                 [-7.00182, -0.00182, -7.00182]]
        # y_true * log(softmax) = [[0, -0.00045, 0], [0, 0, -7.00182]]
        # xent = [0.00045, 7.00182]
        # Reduced xent = (0.00045 + 7.00182) / 2

        self.assertAllClose(self.evaluate(result), 3.5011, atol=1e-3)

    def test_weighted(self):
        scce_obj = metrics.SparseCategoricalCrossentropy()
        self.evaluate(tf.compat.v1.variables_initializer(scce_obj.variables))

        y_true = np.asarray([1, 2])
        y_pred = np.asarray([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
        sample_weight = tf.constant([1.5, 2.0])
        result = scce_obj(y_true, y_pred, sample_weight=sample_weight)

        # EPSILON = 1e-7, y = y_true, y` = y_pred
        # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
        # y` = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
        # logits = log(y`) =  [[-2.9957, -0.0513, -16.1181],
        #                      [-2.3026, -0.2231, -2.3026]]

        # softmax = exp(logits) / sum(exp(logits), axis=-1)
        # y = one_hot(y) = [[0, 1, 0], [0, 0, 1]]
        # xent = -sum(y * log(softmax), 1)

        # exp(logits) = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
        # sum(exp(logits), axis=-1) = [1, 1]
        # softmax = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
        # log(softmax) = [[-2.9957, -0.0513, -16.1181],
        #                 [-2.3026, -0.2231, -2.3026]]
        # y * log(softmax) = [[0, -0.0513, 0], [0, 0, -2.3026]]
        # xent = [0.0513, 2.3026]
        # Weighted xent = [0.051 * 1.5, 2.302 * 2.]
        # Reduced xent = (0.051 * 1.5 + 2.302 * 2.) / 3.5

        self.assertAllClose(self.evaluate(result), 1.338, atol=1e-3)

    def test_weighted_ignore_class(self):
        scce_obj = metrics.SparseCategoricalCrossentropy(ignore_class=-1)
        self.evaluate(tf.compat.v1.variables_initializer(scce_obj.variables))

        y_true = np.asarray([1, 2, -1])
        y_pred = np.asarray([[0.05, 0.95, 0], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]])
        sample_weight = tf.constant([1.5, 2.0, 1.5])
        result = scce_obj(y_true, y_pred, sample_weight=sample_weight)

        self.assertAllClose(self.evaluate(result), 1.338, atol=1e-3)

    def test_weighted_from_logits(self):
        scce_obj = metrics.SparseCategoricalCrossentropy(from_logits=True)
        self.evaluate(tf.compat.v1.variables_initializer(scce_obj.variables))

        y_true = np.asarray([1, 2])
        logits = np.asarray([[1, 9, 0], [1, 8, 1]], dtype=np.float32)
        sample_weight = tf.constant([1.5, 2.0])
        result = scce_obj(y_true, logits, sample_weight=sample_weight)

        # softmax = exp(logits) / sum(exp(logits), axis=-1)
        # y_true = one_hot(y_true) = [[0, 1, 0], [0, 0, 1]]
        # xent = -sum(y_true * log(softmax), 1)
        # xent = [0.00045, 7.00182]
        # weighted xent = [0.000675, 14.00364]
        # Reduced xent = (0.000675 + 14.00364) / (1.5 + 2)

        self.assertAllClose(self.evaluate(result), 4.0012, atol=1e-3)

    def test_axis(self):
        scce_obj = metrics.SparseCategoricalCrossentropy(axis=0)
        self.evaluate(tf.compat.v1.variables_initializer(scce_obj.variables))

        y_true = np.asarray([1, 2])
        y_pred = np.asarray([[0.05, 0.1], [0.95, 0.8], [0, 0.1]])
        result = scce_obj(y_true, y_pred)

        # EPSILON = 1e-7, y = y_true, y` = y_pred
        # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
        # y` = [[0.05, 0.1], [0.95, 0.8], [EPSILON, 0.1]]
        # logits = log(y`) =  [[-2.9957, -2.3026],
        #                      [-0.0513, -0.2231],
        #                      [-16.1181, -2.3026]]

        # softmax = exp(logits) / sum(exp(logits), axis=-1)
        # y = one_hot(y) = [[0, 0], [1, 0], [0, 1]]
        # xent = -sum(y * log(softmax), 1)

        # exp(logits) = [[0.05, 0.1], [0.95, 0.8], [EPSILON, 0.1]]
        # sum(exp(logits)) = [1, 1]
        # softmax = [[0.05, 0.1], [0.95, 0.8], [EPSILON, 0.1]]
        # log(softmax) = [[-2.9957, -2.3026],
        #                 [-0.0513, -0.2231],
        #                 [-16.1181, -2.3026]]
        # y * log(softmax) = [[0, 0], [-0.0513, 0], [0, -2.3026]]
        # xent = [0.0513, 2.3026]
        # Reduced xent = (0.0513 + 2.3026) / 2

        self.assertAllClose(self.evaluate(result), 1.176, atol=1e-3)


class BinaryTruePositives(metrics.Metric):
    def __init__(self, name="binary_true_positives", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, dtype=self.dtype)
            sample_weight = tf.__internal__.ops.broadcast_weights(
                sample_weight, values
            )
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives


if __name__ == "__main__":
    tf.test.main()
