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
class HingeTest(tf.test.TestCase):
    def test_config(self):
        hinge_obj = metrics.Hinge(name="hinge", dtype=tf.int32)
        self.assertEqual(hinge_obj.name, "hinge")
        self.assertEqual(hinge_obj._dtype, tf.int32)

        # Check save and restore config
        hinge_obj2 = metrics.Hinge.from_config(hinge_obj.get_config())
        self.assertEqual(hinge_obj2.name, "hinge")
        self.assertEqual(hinge_obj2._dtype, tf.int32)

    def test_unweighted(self):
        hinge_obj = metrics.Hinge()
        self.evaluate(tf.compat.v1.variables_initializer(hinge_obj.variables))
        y_true = tf.constant([[0, 1, 0, 1], [0, 0, 1, 1]])
        y_pred = tf.constant([[-0.3, 0.2, -0.1, 1.6], [-0.25, -1.0, 0.5, 0.6]])

        # metric = max(0, 1-y_true * y_pred), where y_true is -1/1

        # y_true = [[-1, 1, -1, 1], [-1, -1, 1, 1]]
        # y_true * y_pred = [[0.3, 0.2, 0.1, 1.6], [0.25, 1, 0.5, 0.6]]
        # 1 - y_true * y_pred = [[0.7, 0.8, 0.9, -0.6], [0.75, 0, 0.5, 0.4]]
        # metric = [(0.7 + 0.8 + 0.9 + 0) / 4, (0.75 + 0 + 0.5 + 0.4) / 4]
        #        = [0.6, 0.4125]
        # reduced metric = (0.6 + 0.4125) / 2

        update_op = hinge_obj.update_state(y_true, y_pred)
        self.evaluate(update_op)
        result = hinge_obj.result()
        self.assertAllClose(0.506, result, atol=1e-3)

    def test_weighted(self):
        hinge_obj = metrics.Hinge()
        self.evaluate(tf.compat.v1.variables_initializer(hinge_obj.variables))
        y_true = tf.constant([[-1, 1, -1, 1], [-1, -1, 1, 1]])
        y_pred = tf.constant([[-0.3, 0.2, -0.1, 1.6], [-0.25, -1.0, 0.5, 0.6]])
        sample_weight = tf.constant([1.5, 2.0])

        # metric = max(0, 1-y_true * y_pred), where y_true is -1/1

        # y_true * y_pred = [[0.3, 0.2, 0.1, 1.6], [0.25, 1, 0.5, 0.6]]
        # 1 - y_true * y_pred = [[0.7, 0.8, 0.9, -0.6], [0.75, 0, 0.5, 0.4]]
        # metric = [(0.7 + 0.8 + 0.9 + 0) / 4, (0.75 + 0 + 0.5 + 0.4) / 4]
        #        = [0.6, 0.4125]
        # weighted metric = [0.6 * 1.5, 0.4125 * 2]
        # reduced metric = (0.6 * 1.5 + 0.4125 * 2) / (1.5 + 2)

        result = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(0.493, self.evaluate(result), atol=1e-3)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class SquaredHingeTest(tf.test.TestCase):
    def test_config(self):
        sq_hinge_obj = metrics.SquaredHinge(name="sq_hinge", dtype=tf.int32)
        self.assertEqual(sq_hinge_obj.name, "sq_hinge")
        self.assertEqual(sq_hinge_obj._dtype, tf.int32)

        # Check save and restore config
        sq_hinge_obj2 = metrics.SquaredHinge.from_config(
            sq_hinge_obj.get_config()
        )
        self.assertEqual(sq_hinge_obj2.name, "sq_hinge")
        self.assertEqual(sq_hinge_obj2._dtype, tf.int32)

    def test_unweighted(self):
        sq_hinge_obj = metrics.SquaredHinge()
        self.evaluate(
            tf.compat.v1.variables_initializer(sq_hinge_obj.variables)
        )
        y_true = tf.constant([[0, 1, 0, 1], [0, 0, 1, 1]])
        y_pred = tf.constant([[-0.3, 0.2, -0.1, 1.6], [-0.25, -1.0, 0.5, 0.6]])

        # metric = max(0, 1-y_true * y_pred), where y_true is -1/1

        # y_true = [[-1, 1, -1, 1], [-1, -1, 1, 1]]
        # y_true * y_pred = [[0.3, 0.2, 0.1, 1.6], [0.25, 1, 0.5, 0.6]]
        # 1 - y_true * y_pred = [[0.7, 0.8, 0.9, -0.6], [0.75, 0, 0.5, 0.4]]
        # max(0, 1 - y_true * y_pred) = [[0.7, 0.8, 0.9, 0], [0.75, 0, 0.5,
        # 0.4]]
        # squared(max(0, 1 - y_true * y_pred)) = [[0.49, 0.64, 0.81, 0],
        #                                         [0.5625, 0, 0.25, 0.16]]
        # metric = [(0.49 + 0.64 + 0.81 + 0) / 4, (0.5625 + 0 + 0.25 + 0.16) /
        # 4]
        #        = [0.485, 0.2431]
        # reduced metric = (0.485 + 0.2431) / 2

        update_op = sq_hinge_obj.update_state(y_true, y_pred)
        self.evaluate(update_op)
        result = sq_hinge_obj.result()
        self.assertAllClose(0.364, result, atol=1e-3)

    def test_weighted(self):
        sq_hinge_obj = metrics.SquaredHinge()
        self.evaluate(
            tf.compat.v1.variables_initializer(sq_hinge_obj.variables)
        )
        y_true = tf.constant([[-1, 1, -1, 1], [-1, -1, 1, 1]])
        y_pred = tf.constant([[-0.3, 0.2, -0.1, 1.6], [-0.25, -1.0, 0.5, 0.6]])
        sample_weight = tf.constant([1.5, 2.0])

        # metric = max(0, 1-y_true * y_pred), where y_true is -1/1

        # y_true * y_pred = [[0.3, 0.2, 0.1, 1.6], [0.25, 1, 0.5, 0.6]]
        # 1 - y_true * y_pred = [[0.7, 0.8, 0.9, -0.6], [0.75, 0, 0.5, 0.4]]
        # max(0, 1 - y_true * y_pred) = [[0.7, 0.8, 0.9, 0], [0.75, 0, 0.5,
        # 0.4]]
        # squared(max(0, 1 - y_true * y_pred)) = [[0.49, 0.64, 0.81, 0],
        #                                         [0.5625, 0, 0.25, 0.16]]
        # metric = [(0.49 + 0.64 + 0.81 + 0) / 4, (0.5625 + 0 + 0.25 + 0.16) /
        # 4]
        #        = [0.485, 0.2431]
        # weighted metric = [0.485 * 1.5, 0.2431 * 2]
        # reduced metric = (0.485 * 1.5 + 0.2431 * 2) / (1.5 + 2)

        result = sq_hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(0.347, self.evaluate(result), atol=1e-3)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class CategoricalHingeTest(tf.test.TestCase):
    def test_config(self):
        cat_hinge_obj = metrics.CategoricalHinge(
            name="cat_hinge", dtype=tf.int32
        )
        self.assertEqual(cat_hinge_obj.name, "cat_hinge")
        self.assertEqual(cat_hinge_obj._dtype, tf.int32)

        # Check save and restore config
        cat_hinge_obj2 = metrics.CategoricalHinge.from_config(
            cat_hinge_obj.get_config()
        )
        self.assertEqual(cat_hinge_obj2.name, "cat_hinge")
        self.assertEqual(cat_hinge_obj2._dtype, tf.int32)

    def test_unweighted(self):
        cat_hinge_obj = metrics.CategoricalHinge()
        self.evaluate(
            tf.compat.v1.variables_initializer(cat_hinge_obj.variables)
        )
        y_true = tf.constant(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = tf.constant(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )

        update_op = cat_hinge_obj.update_state(y_true, y_pred)
        self.evaluate(update_op)
        result = cat_hinge_obj.result()
        self.assertAllClose(0.5, result, atol=1e-5)

    def test_weighted(self):
        cat_hinge_obj = metrics.CategoricalHinge()
        self.evaluate(
            tf.compat.v1.variables_initializer(cat_hinge_obj.variables)
        )
        y_true = tf.constant(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = tf.constant(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )
        sample_weight = tf.constant((1.0, 1.5, 2.0, 2.5))
        result = cat_hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(0.5, self.evaluate(result), atol=1e-5)


if __name__ == "__main__":
    tf.test.main()
