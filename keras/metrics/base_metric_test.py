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
"""Tests for Keras base Metric classes."""

import copy
import os

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras import Model
from keras import layers
from keras import metrics
from keras.engine import base_layer
from keras.engine import training as training_module
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class KerasSumTest(tf.test.TestCase, parameterized.TestCase):
    def test_sum(self):
        with self.test_session():
            m = metrics.Sum(name="my_sum")

            # check config
            self.assertEqual(m.name, "my_sum")
            self.assertTrue(m.stateful)
            self.assertEqual(m.dtype, tf.float32)
            self.assertLen(m.variables, 1)
            self.evaluate(tf.compat.v1.variables_initializer(m.variables))

            # check initial state
            self.assertEqual(self.evaluate(m.total), 0)

            # check __call__()
            self.assertEqual(self.evaluate(m(100)), 100)
            self.assertEqual(self.evaluate(m.total), 100)

            # check update_state() and result() + state accumulation + tensor
            # input
            update_op = m.update_state(tf.convert_to_tensor([1, 5]))
            self.evaluate(update_op)
            self.assertAlmostEqual(self.evaluate(m.result()), 106)
            self.assertEqual(self.evaluate(m.total), 106)  # 100 + 1 + 5

            # check reset_state()
            m.reset_state()
            self.assertEqual(self.evaluate(m.total), 0)

    def test_sum_with_sample_weight(self):
        m = metrics.Sum(dtype=tf.float64)
        self.assertEqual(m.dtype, tf.float64)
        self.evaluate(tf.compat.v1.variables_initializer(m.variables))

        # check scalar weight
        result_t = m(100, sample_weight=0.5)
        self.assertEqual(self.evaluate(result_t), 50)
        self.assertEqual(self.evaluate(m.total), 50)

        # check weights not scalar and weights rank matches values rank
        result_t = m([1, 5], sample_weight=[1, 0.2])
        result = self.evaluate(result_t)
        self.assertAlmostEqual(result, 52.0, 4)  # 50 + 1 + 5 * 0.2
        self.assertAlmostEqual(self.evaluate(m.total), 52.0, 4)

        # check weights broadcast
        result_t = m([1, 2], sample_weight=0.5)
        self.assertAlmostEqual(self.evaluate(result_t), 53.5, 1)  # 52 + 0.5 + 1
        self.assertAlmostEqual(self.evaluate(m.total), 53.5, 1)

        # check weights squeeze
        result_t = m([1, 5], sample_weight=[[1], [0.2]])
        self.assertAlmostEqual(self.evaluate(result_t), 55.5, 1)  # 53.5 + 1 + 1
        self.assertAlmostEqual(self.evaluate(m.total), 55.5, 1)

        # check weights expand
        result_t = m([[1], [5]], sample_weight=[1, 0.2])
        self.assertAlmostEqual(self.evaluate(result_t), 57.5, 2)  # 55.5 + 1 + 1
        self.assertAlmostEqual(self.evaluate(m.total), 57.5, 1)

        # check values reduced to the dimensions of weight
        result_t = m(
            [[[1.0, 2.0], [3.0, 2.0], [0.5, 4.0]]], sample_weight=[0.5]
        )
        result = np.round(self.evaluate(result_t), decimals=2)
        # result = (prev: 57.5) + 0.5 + 1 + 1.5 + 1 + 0.25 + 2
        self.assertAlmostEqual(result, 63.75, 2)
        self.assertAlmostEqual(self.evaluate(m.total), 63.75, 2)

    def test_sum_graph_with_placeholder(self):
        with tf.compat.v1.get_default_graph().as_default(), self.cached_session() as sess:  # noqa: E501
            m = metrics.Sum()
            v = tf.compat.v1.placeholder(tf.float32)
            w = tf.compat.v1.placeholder(tf.float32)
            self.evaluate(tf.compat.v1.variables_initializer(m.variables))

            # check __call__()
            result_t = m(v, sample_weight=w)
            result = sess.run(result_t, feed_dict=({v: 100, w: 0.5}))
            self.assertEqual(result, 50)
            self.assertEqual(self.evaluate(m.total), 50)

            # check update_state() and result()
            result = sess.run(result_t, feed_dict=({v: [1, 5], w: [1, 0.2]}))
            self.assertAlmostEqual(result, 52.0, 2)  # 50 + 1 + 5 * 0.2
            self.assertAlmostEqual(self.evaluate(m.total), 52.0, 2)

    def test_save_restore(self):
        with self.test_session():
            checkpoint_directory = self.get_temp_dir()
            checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
            m = metrics.Sum()
            checkpoint = tf.train.Checkpoint(sum=m)
            self.evaluate(tf.compat.v1.variables_initializer(m.variables))

            # update state
            self.evaluate(m(100.0))
            self.evaluate(m(200.0))

            # save checkpoint and then add an update
            save_path = checkpoint.save(checkpoint_prefix)
            self.evaluate(m(1000.0))

            # restore to the same checkpoint sum object (= 300)
            checkpoint.restore(save_path).assert_consumed().run_restore_ops()
            self.evaluate(m(300.0))
            self.assertEqual(600.0, self.evaluate(m.result()))

            # restore to a different checkpoint sum object
            restore_sum = metrics.Sum()
            restore_checkpoint = tf.train.Checkpoint(sum=restore_sum)
            status = restore_checkpoint.restore(save_path)
            restore_update = restore_sum(300.0)
            status.assert_consumed().run_restore_ops()
            self.evaluate(restore_update)
            self.assertEqual(600.0, self.evaluate(restore_sum.result()))

    def test_init_scope_during_add_weight(self):
        seen_variables = 0

        def capture_variable_creation(next_creator_fn, **kwargs) -> tf.Variable:
            nonlocal seen_variables
            seen_variables += 1
            return tf.constant(seen_variables)

        @tf.function
        def create_variables():
            # When this method is called in a graph context, any usage of
            # `tf.init_scope` will bypass this variable creator scope, resulting
            # in different behavior.
            with tf.variable_creator_scope(capture_variable_creation):
                return metrics.Sum().variables

        metric_variables = self.evaluate(create_variables())
        # The Sum metric contains a single `total` variable, which the creation
        # scope has changed to a `1` tensor.
        self.assertAllEqual([1], metric_variables)


class MeanTest(test_combinations.TestCase):

    # TODO(b/120949004): Re-enable garbage collection check
    # @tf_test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
    @test_combinations.run_all_keras_modes
    def test_mean(self):
        m = metrics.Mean(name="my_mean")

        # check config
        self.assertEqual(m.name, "my_mean")
        self.assertTrue(m.stateful)
        self.assertEqual(m.dtype, tf.float32)
        self.assertEqual(len(m.variables), 2)
        self.evaluate(tf.compat.v1.variables_initializer(m.variables))

        # check initial state
        self.assertEqual(self.evaluate(m.total), 0)
        self.assertEqual(self.evaluate(m.count), 0)

        # check __call__()
        self.assertEqual(self.evaluate(m(100)), 100)
        self.assertEqual(self.evaluate(m.total), 100)
        self.assertEqual(self.evaluate(m.count), 1)

        # check update_state() and result() + state accumulation + tensor input
        update_op = m.update_state(
            [tf.convert_to_tensor(1), tf.convert_to_tensor(5)]
        )
        self.evaluate(update_op)
        self.assertAlmostEqual(self.evaluate(m.result()), 106 / 3, 2)
        self.assertEqual(self.evaluate(m.total), 106)  # 100 + 1 + 5
        self.assertEqual(self.evaluate(m.count), 3)

        # check reset_state()
        m.reset_state()
        self.assertEqual(self.evaluate(m.total), 0)
        self.assertEqual(self.evaluate(m.count), 0)

        # Check save and restore config
        m2 = metrics.Mean.from_config(m.get_config())
        self.assertEqual(m2.name, "my_mean")
        self.assertTrue(m2.stateful)
        self.assertEqual(m2.dtype, tf.float32)
        self.assertEqual(len(m2.variables), 2)

    @test_utils.run_v2_only
    def test_function_wrapped_reset_state(self):
        m = metrics.Mean(name="my_mean")

        # check reset_state in function.
        @tf.function
        def reset_in_fn():
            m.reset_state()
            return m.update_state(100)

        for _ in range(5):
            self.evaluate(reset_in_fn())
        self.assertEqual(self.evaluate(m.count), 1)

    @test_combinations.run_all_keras_modes
    def test_mean_with_sample_weight(self):
        m = metrics.Mean(dtype=tf.float64)
        self.assertEqual(m.dtype, tf.float64)
        self.evaluate(tf.compat.v1.variables_initializer(m.variables))

        # check scalar weight
        result_t = m(100, sample_weight=0.5)
        self.assertEqual(self.evaluate(result_t), 50 / 0.5)
        self.assertEqual(self.evaluate(m.total), 50)
        self.assertEqual(self.evaluate(m.count), 0.5)

        # check weights not scalar and weights rank matches values rank
        result_t = m([1, 5], sample_weight=[1, 0.2])
        result = self.evaluate(result_t)
        self.assertAlmostEqual(result, 52 / 1.7, 2)
        self.assertAlmostEqual(
            self.evaluate(m.total), 52, 2
        )  # 50 + 1 + 5 * 0.2
        self.assertAlmostEqual(self.evaluate(m.count), 1.7, 2)  # 0.5 + 1.2

        # check weights broadcast
        result_t = m([1, 2], sample_weight=0.5)
        self.assertAlmostEqual(self.evaluate(result_t), 53.5 / 2.7, 2)
        self.assertAlmostEqual(self.evaluate(m.total), 53.5, 2)  # 52 + 0.5 + 1
        self.assertAlmostEqual(
            self.evaluate(m.count), 2.7, 2
        )  # 1.7 + 0.5 + 0.5

        # check weights squeeze
        result_t = m([1, 5], sample_weight=[[1], [0.2]])
        self.assertAlmostEqual(self.evaluate(result_t), 55.5 / 3.9, 2)
        self.assertAlmostEqual(self.evaluate(m.total), 55.5, 2)  # 53.5 + 1 + 1
        self.assertAlmostEqual(self.evaluate(m.count), 3.9, 2)  # 2.7 + 1.2

        # check weights expand
        result_t = m([[1], [5]], sample_weight=[1, 0.2])
        self.assertAlmostEqual(self.evaluate(result_t), 57.5 / 5.1, 2)
        self.assertAlmostEqual(self.evaluate(m.total), 57.5, 2)  # 55.5 + 1 + 1
        self.assertAlmostEqual(self.evaluate(m.count), 5.1, 2)  # 3.9 + 1.2

        # check values reduced to the dimensions of weight
        result_t = m(
            [[[1.0, 2.0], [3.0, 2.0], [0.5, 4.0]]], sample_weight=[0.5]
        )
        result = np.round(self.evaluate(result_t), decimals=2)  # 58.5 / 5.6
        self.assertEqual(result, 10.45)
        self.assertEqual(np.round(self.evaluate(m.total), decimals=2), 58.54)
        self.assertEqual(np.round(self.evaluate(m.count), decimals=2), 5.6)

    @test_combinations.run_all_keras_modes
    def test_mean_graph_with_placeholder(self):
        with tf.compat.v1.get_default_graph().as_default(), self.cached_session() as sess:  # noqa: E501
            m = metrics.Mean()
            v = tf.compat.v1.placeholder(tf.float32)
            w = tf.compat.v1.placeholder(tf.float32)
            self.evaluate(tf.compat.v1.variables_initializer(m.variables))

            # check __call__()
            result_t = m(v, sample_weight=w)
            result = sess.run(result_t, feed_dict=({v: 100, w: 0.5}))
            self.assertEqual(self.evaluate(m.total), 50)
            self.assertEqual(self.evaluate(m.count), 0.5)
            self.assertEqual(result, 50 / 0.5)

            # check update_state() and result()
            result = sess.run(result_t, feed_dict=({v: [1, 5], w: [1, 0.2]}))
            self.assertAlmostEqual(
                self.evaluate(m.total), 52, 2
            )  # 50 + 1 + 5 * 0.2
            self.assertAlmostEqual(self.evaluate(m.count), 1.7, 2)  # 0.5 + 1.2
            self.assertAlmostEqual(result, 52 / 1.7, 2)

    @test_combinations.run_all_keras_modes
    def test_save_restore(self):
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        m = metrics.Mean()
        checkpoint = tf.train.Checkpoint(mean=m)
        self.evaluate(tf.compat.v1.variables_initializer(m.variables))

        # update state
        self.evaluate(m(100.0))
        self.evaluate(m(200.0))

        # save checkpoint and then add an update
        save_path = checkpoint.save(checkpoint_prefix)
        self.evaluate(m(1000.0))

        # restore to the same checkpoint mean object
        checkpoint.restore(save_path).assert_consumed().run_restore_ops()
        self.evaluate(m(300.0))
        self.assertEqual(200.0, self.evaluate(m.result()))

        # restore to a different checkpoint mean object
        restore_mean = metrics.Mean()
        restore_checkpoint = tf.train.Checkpoint(mean=restore_mean)
        status = restore_checkpoint.restore(save_path)
        restore_update = restore_mean(300.0)
        status.assert_consumed().run_restore_ops()
        self.evaluate(restore_update)
        self.assertEqual(200.0, self.evaluate(restore_mean.result()))
        self.assertEqual(3, self.evaluate(restore_mean.count))

    @test_combinations.run_all_keras_modes
    def test_multiple_instances(self):
        m = metrics.Mean()
        m2 = metrics.Mean()

        self.assertEqual(m.name, "mean")
        self.assertEqual(m2.name, "mean")

        self.assertEqual(
            [v.name for v in m.variables],
            test_utils.get_expected_metric_variable_names(["total", "count"]),
        )
        self.assertEqual(
            [v.name for v in m2.variables],
            test_utils.get_expected_metric_variable_names(
                ["total", "count"], name_suffix="_1"
            ),
        )

        self.evaluate(tf.compat.v1.variables_initializer(m.variables))
        self.evaluate(tf.compat.v1.variables_initializer(m2.variables))

        # check initial state
        self.assertEqual(self.evaluate(m.total), 0)
        self.assertEqual(self.evaluate(m.count), 0)
        self.assertEqual(self.evaluate(m2.total), 0)
        self.assertEqual(self.evaluate(m2.count), 0)

        # check __call__()
        self.assertEqual(self.evaluate(m(100)), 100)
        self.assertEqual(self.evaluate(m.total), 100)
        self.assertEqual(self.evaluate(m.count), 1)
        self.assertEqual(self.evaluate(m2.total), 0)
        self.assertEqual(self.evaluate(m2.count), 0)

        self.assertEqual(self.evaluate(m2([63, 10])), 36.5)
        self.assertEqual(self.evaluate(m2.total), 73)
        self.assertEqual(self.evaluate(m2.count), 2)
        self.assertEqual(self.evaluate(m.result()), 100)
        self.assertEqual(self.evaluate(m.total), 100)
        self.assertEqual(self.evaluate(m.count), 1)

    @test_utils.run_v2_only
    def test_deepcopy_of_metrics(self):
        m = metrics.Mean(name="my_mean")

        m.reset_state()
        m.update_state(100)
        m_copied = copy.deepcopy(m)
        m_copied.update_state(200)

        self.assertEqual(self.evaluate(m.result()), 100)
        self.assertEqual(self.evaluate(m_copied.result()), 150)

        m.reset_state()

        self.assertEqual(self.evaluate(m.result()), 0)
        self.assertEqual(self.evaluate(m_copied.result()), 150)


class MeanTensorTest(tf.test.TestCase, parameterized.TestCase):
    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_config(self):
        with self.test_session():
            m = metrics.MeanTensor(name="mean_by_element")

            # check config
            self.assertEqual(m.name, "mean_by_element")
            self.assertTrue(m.stateful)
            self.assertEqual(m.dtype, tf.float32)
            self.assertEmpty(m.variables)

            with self.assertRaisesRegex(
                ValueError, "does not have any value yet"
            ):
                m.result()

            self.evaluate(m([[3], [5], [3]]))
            self.assertAllEqual(m._shape, [3, 1])

            m2 = metrics.MeanTensor.from_config(m.get_config())
            self.assertEqual(m2.name, "mean_by_element")
            self.assertTrue(m2.stateful)
            self.assertEqual(m2.dtype, tf.float32)
            self.assertEmpty(m2.variables)

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_unweighted(self):
        with self.test_session():
            m = metrics.MeanTensor(dtype=tf.float64)

            # check __call__()
            self.assertAllClose(self.evaluate(m([100, 40])), [100, 40])
            self.assertAllClose(self.evaluate(m.total), [100, 40])
            self.assertAllClose(self.evaluate(m.count), [1, 1])

            # check update_state() and result() + state accumulation + tensor
            # input
            update_op = m.update_state(
                [tf.convert_to_tensor(1), tf.convert_to_tensor(5)]
            )
            self.evaluate(update_op)
            self.assertAllClose(self.evaluate(m.result()), [50.5, 22.5])
            self.assertAllClose(self.evaluate(m.total), [101, 45])
            self.assertAllClose(self.evaluate(m.count), [2, 2])

            # check reset_state()
            m.reset_state()
            self.assertAllClose(self.evaluate(m.total), [0, 0])
            self.assertAllClose(self.evaluate(m.count), [0, 0])

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_weighted(self):
        with self.test_session():
            m = metrics.MeanTensor(dtype=tf.float64)
            self.assertEqual(m.dtype, tf.float64)

            # check scalar weight
            result_t = m([100, 30], sample_weight=0.5)
            self.assertAllClose(self.evaluate(result_t), [100, 30])
            self.assertAllClose(self.evaluate(m.total), [50, 15])
            self.assertAllClose(self.evaluate(m.count), [0.5, 0.5])

            # check weights not scalar and weights rank matches values rank
            result_t = m([1, 5], sample_weight=[1, 0.2])
            result = self.evaluate(result_t)
            self.assertAllClose(result, [51 / 1.5, 16 / 0.7], 2)
            self.assertAllClose(self.evaluate(m.total), [51, 16])
            self.assertAllClose(self.evaluate(m.count), [1.5, 0.7])

            # check weights broadcast
            result_t = m([1, 2], sample_weight=0.5)
            self.assertAllClose(self.evaluate(result_t), [51.5 / 2, 17 / 1.2])
            self.assertAllClose(self.evaluate(m.total), [51.5, 17])
            self.assertAllClose(self.evaluate(m.count), [2, 1.2])

            # check weights squeeze
            result_t = m([1, 5], sample_weight=[[1], [0.2]])
            self.assertAllClose(self.evaluate(result_t), [52.5 / 3, 18 / 1.4])
            self.assertAllClose(self.evaluate(m.total), [52.5, 18])
            self.assertAllClose(self.evaluate(m.count), [3, 1.4])

            # check weights expand
            m = metrics.MeanTensor(dtype=tf.float64)
            self.evaluate(tf.compat.v1.variables_initializer(m.variables))
            result_t = m([[1], [5]], sample_weight=[1, 0.2])
            self.assertAllClose(self.evaluate(result_t), [[1], [5]])
            self.assertAllClose(self.evaluate(m.total), [[1], [1]])
            self.assertAllClose(self.evaluate(m.count), [[1], [0.2]])

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_invalid_value_shape(self):
        m = metrics.MeanTensor(dtype=tf.float64)
        m([1])
        with self.assertRaisesRegex(
            ValueError,
            "MeanTensor input values must always have the same shape",
        ):
            m([1, 5])

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_build_in_tf_function(self):
        """Ensure that variables are created correctly in a tf function."""
        m = metrics.MeanTensor(dtype=tf.float64)

        @tf.function
        def call_metric(x):
            return m(x)

        with self.test_session():
            self.assertAllClose(
                self.evaluate(call_metric([100, 40])), [100, 40]
            )
            self.assertAllClose(self.evaluate(m.total), [100, 40])
            self.assertAllClose(self.evaluate(m.count), [1, 1])
            self.assertAllClose(self.evaluate(call_metric([20, 2])), [60, 21])

    @test_combinations.generate(test_combinations.combine(mode=["eager"]))
    def test_in_keras_model(self):
        class ModelWithMetric(Model):
            def __init__(self):
                super().__init__()
                self.dense1 = layers.Dense(
                    3, activation="relu", kernel_initializer="ones"
                )
                self.dense2 = layers.Dense(
                    1, activation="sigmoid", kernel_initializer="ones"
                )
                self.mean_tensor = metrics.MeanTensor()

            def call(self, x):
                x = self.dense1(x)
                x = self.dense2(x)
                self.mean_tensor(self.dense1.kernel)
                return x

        model = ModelWithMetric()
        model.compile(loss="mae", optimizer="rmsprop", run_eagerly=True)

        x = np.ones((100, 4))
        y = np.zeros((100, 1))
        model.evaluate(x, y, batch_size=50)
        self.assertAllClose(
            self.evaluate(model.mean_tensor.result()), np.ones((4, 3))
        )
        self.assertAllClose(
            self.evaluate(model.mean_tensor.total), np.full((4, 3), 2)
        )
        self.assertAllClose(
            self.evaluate(model.mean_tensor.count), np.full((4, 3), 2)
        )

        model.evaluate(x, y, batch_size=25)
        self.assertAllClose(
            self.evaluate(model.mean_tensor.result()), np.ones((4, 3))
        )
        self.assertAllClose(
            self.evaluate(model.mean_tensor.total), np.full((4, 3), 4)
        )
        self.assertAllClose(
            self.evaluate(model.mean_tensor.count), np.full((4, 3), 4)
        )


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


class BinaryTruePositivesViaControlFlow(metrics.Metric):
    def __init__(self, name="binary_true_positives", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                if y_true[i][j] and y_pred[i][j]:
                    if sample_weight is None:
                        self.true_positives.assign_add(1)
                    else:
                        self.true_positives.assign_add(sample_weight[i][0])

    def result(self):
        if tf.constant(True):
            return self.true_positives
        return 0.0


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class CustomMetricsTest(tf.test.TestCase):
    def test_config(self):
        btp_obj = BinaryTruePositives(name="btp", dtype=tf.int32)
        self.assertEqual(btp_obj.name, "btp")
        self.assertEqual(btp_obj.dtype, tf.int32)

        # Check save and restore config
        btp_obj2 = BinaryTruePositives.from_config(btp_obj.get_config())
        self.assertEqual(btp_obj2.name, "btp")
        self.assertEqual(btp_obj2.dtype, tf.int32)

    def test_unweighted(self):
        btp_obj = BinaryTruePositives()
        self.evaluate(tf.compat.v1.variables_initializer(btp_obj.variables))
        y_true = tf.constant(
            [
                [0, 0.9, 0, 1, 0],
                [0, 0, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [0, 0, 0, 0, 1.5],
            ]
        )
        y_pred = tf.constant(
            [
                [0, 0, 1, 5, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 0, 1, 0],
                [1, 10, 1, 1, 1],
            ]
        )

        update_op = btp_obj.update_state(y_true, y_pred)
        self.evaluate(update_op)
        result = btp_obj.result()
        self.assertEqual(7, self.evaluate(result))

    def test_weighted(self):
        btp_obj = BinaryTruePositives()
        self.evaluate(tf.compat.v1.variables_initializer(btp_obj.variables))
        y_true = tf.constant(
            [
                [0, 0.9, 0, 1, 0],
                [0, 0, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [0, 0, 0, 0, 1.5],
            ]
        )
        y_pred = tf.constant(
            [
                [0, 0, 1, 5, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 0, 1, 0],
                [1, 10, 1, 1, 1],
            ]
        )
        sample_weight = tf.constant([[1.0], [1.5], [2.0], [2.5]])
        result = btp_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(12, self.evaluate(result))

    def test_autograph(self):
        metric = BinaryTruePositivesViaControlFlow()
        self.evaluate(tf.compat.v1.variables_initializer(metric.variables))
        y_true = tf.constant(
            [
                [0, 0.9, 0, 1, 0],
                [0, 0, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [0, 0, 0, 0, 1.5],
            ]
        )
        y_pred = tf.constant(
            [
                [0, 0, 1, 5, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 0, 1, 0],
                [1, 10, 1, 1, 1],
            ]
        )
        sample_weight = tf.constant([[1.0], [1.5], [2.0], [2.5]])

        @tf.function
        def compute_metric(y_true, y_pred, sample_weight):
            metric(y_true, y_pred, sample_weight)
            return metric.result()

        result = compute_metric(y_true, y_pred, sample_weight)
        self.assertEqual(12, self.evaluate(result))

    def test_metric_wrappers_autograph(self):
        def metric_fn(y_true, y_pred):
            x = tf.constant(0.0)
            for i in range(len(y_true)):
                for j in range(len(y_true[i])):
                    if (
                        tf.equal(y_true[i][j], y_pred[i][j])
                        and y_true[i][j] > 0
                    ):
                        x += 1.0
            return x

        mean_metric = metrics.MeanMetricWrapper(metric_fn)
        sum_metric = metrics.SumOverBatchSizeMetricWrapper(metric_fn)
        self.evaluate(tf.compat.v1.variables_initializer(mean_metric.variables))
        self.evaluate(tf.compat.v1.variables_initializer(sum_metric.variables))

        y_true = tf.constant(
            [[0, 0, 0, 1, 0], [0, 0, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 0, 1]]
        )
        y_pred = tf.constant(
            [[0, 0, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1]]
        )

        @tf.function
        def tf_functioned_metric_fn(metric, y_true, y_pred):
            return metric(y_true, y_pred)

        metric_result = tf_functioned_metric_fn(mean_metric, y_true, y_pred)
        self.assertAllClose(self.evaluate(metric_result), 10, 1e-2)
        metric_result = tf_functioned_metric_fn(sum_metric, y_true, y_pred)
        self.assertAllClose(self.evaluate(metric_result), 10, 1e-2)

    def test_metric_not_tracked_as_sublayer_in_layer(self):
        class MyLayer(base_layer.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.mean_obj = metrics.Mean(name="my_mean_obj")

            def call(self, x):
                self.add_metric(
                    tf.reduce_sum(x), aggregation="mean", name="my_mean_tensor"
                )
                self.add_metric(self.mean_obj(x))
                return x

        layer = MyLayer()
        x = np.ones((1, 1))
        layer(x)
        self.assertLen(list(layer._flatten_layers(include_self=False)), 0)
        self.assertLen(layer.metrics, 2)

    def test_metric_not_tracked_as_sublayer_in_model(self):
        class MyModel(training_module.Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.mean_obj = metrics.Mean(name="my_mean_obj")

            def call(self, x):
                self.add_metric(
                    tf.reduce_sum(x), aggregation="mean", name="my_mean_tensor"
                )
                self.add_metric(self.mean_obj(x))
                return x

        model = MyModel()
        x = np.ones((1, 1))
        model(x)
        self.assertLen(list(model._flatten_layers(include_self=False)), 0)
        self.assertLen(model.layers, 0)
        self.assertLen(model.metrics, 2)

    def test_invalid_custom_metric_class_error_msg(self):
        x = layers.Input(shape=(2,))
        y = layers.Dense(3)(x)
        model = training_module.Model(x, y)

        class BadMetric(metrics.Metric):
            def update_state(self, y_true, y_pred, sample_weight=None):
                return

            def result(self):
                return

        with self.assertRaisesRegex(RuntimeError, "can only be a single"):
            model.compile("sgd", "mse", metrics=[BadMetric()])
            model.fit(np.ones((10, 2)), np.ones((10, 3)))

    def test_invalid_custom_metric_fn_error_msg(self):
        x = layers.Input(shape=(2,))
        y = layers.Dense(3)(x)
        model = training_module.Model(x, y)

        def bad_metric(y_true, y_pred, sample_weight=None):
            return None

        def dict_metric(y_true, y_pred, sample_weight=None):
            return {"value": 0.0}

        with self.assertRaisesRegex(
            RuntimeError, "The output of a metric function can only be"
        ):
            model.compile("sgd", "mse", metrics=[bad_metric])
            model.fit(np.ones((10, 2)), np.ones((10, 3)))
        with self.assertRaisesRegex(
            RuntimeError, "To return a dict of values, implement"
        ):
            model.compile("sgd", "mse", metrics=[dict_metric])
            model.fit(np.ones((10, 2)), np.ones((10, 3)))


if __name__ == "__main__":
    tf.test.main()
