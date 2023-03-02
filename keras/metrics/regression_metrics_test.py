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

import math

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras import Input
from keras import metrics
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class CosineSimilarityTest(tf.test.TestCase):
    def l2_norm(self, x, axis):
        epsilon = 1e-12
        square_sum = np.sum(np.square(x), axis=axis, keepdims=True)
        x_inv_norm = 1 / np.sqrt(np.maximum(square_sum, epsilon))
        return np.multiply(x, x_inv_norm)

    def setup(self, axis=1):
        self.np_y_true = np.asarray([[1, 9, 2], [-5, -2, 6]], dtype=np.float32)
        self.np_y_pred = np.asarray([[4, 8, 12], [8, 1, 3]], dtype=np.float32)

        y_true = self.l2_norm(self.np_y_true, axis)
        y_pred = self.l2_norm(self.np_y_pred, axis)
        self.expected_loss = np.sum(np.multiply(y_true, y_pred), axis=(axis,))

        self.y_true = tf.constant(self.np_y_true)
        self.y_pred = tf.constant(self.np_y_pred)

    def test_config(self):
        cosine_obj = metrics.CosineSimilarity(
            axis=2, name="my_cos", dtype=tf.int32
        )
        self.assertEqual(cosine_obj.name, "my_cos")
        self.assertEqual(cosine_obj._dtype, tf.int32)

        # Check save and restore config
        cosine_obj2 = metrics.CosineSimilarity.from_config(
            cosine_obj.get_config()
        )
        self.assertEqual(cosine_obj2.name, "my_cos")
        self.assertEqual(cosine_obj2._dtype, tf.int32)

    def test_unweighted(self):
        self.setup()
        cosine_obj = metrics.CosineSimilarity()
        self.evaluate(tf.compat.v1.variables_initializer(cosine_obj.variables))
        loss = cosine_obj(self.y_true, self.y_pred)
        expected_loss = np.mean(self.expected_loss)
        self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

    def test_weighted(self):
        self.setup()
        cosine_obj = metrics.CosineSimilarity()
        self.evaluate(tf.compat.v1.variables_initializer(cosine_obj.variables))
        sample_weight = np.asarray([1.2, 3.4])
        loss = cosine_obj(
            self.y_true, self.y_pred, sample_weight=tf.constant(sample_weight)
        )
        expected_loss = np.sum(self.expected_loss * sample_weight) / np.sum(
            sample_weight
        )
        self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

    def test_axis(self):
        self.setup(axis=1)
        cosine_obj = metrics.CosineSimilarity(axis=1)
        self.evaluate(tf.compat.v1.variables_initializer(cosine_obj.variables))
        loss = cosine_obj(self.y_true, self.y_pred)
        expected_loss = np.mean(self.expected_loss)
        self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class MeanAbsoluteErrorTest(tf.test.TestCase):
    def test_config(self):
        mae_obj = metrics.MeanAbsoluteError(name="my_mae", dtype=tf.int32)
        self.assertEqual(mae_obj.name, "my_mae")
        self.assertEqual(mae_obj._dtype, tf.int32)

        # Check save and restore config
        mae_obj2 = metrics.MeanAbsoluteError.from_config(mae_obj.get_config())
        self.assertEqual(mae_obj2.name, "my_mae")
        self.assertEqual(mae_obj2._dtype, tf.int32)

    def test_unweighted(self):
        mae_obj = metrics.MeanAbsoluteError()
        self.evaluate(tf.compat.v1.variables_initializer(mae_obj.variables))
        y_true = tf.constant(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = tf.constant(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )

        update_op = mae_obj.update_state(y_true, y_pred)
        self.evaluate(update_op)
        result = mae_obj.result()
        self.assertAllClose(0.5, result, atol=1e-5)

    def test_weighted(self):
        mae_obj = metrics.MeanAbsoluteError()
        self.evaluate(tf.compat.v1.variables_initializer(mae_obj.variables))
        y_true = tf.constant(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = tf.constant(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )
        sample_weight = tf.constant((1.0, 1.5, 2.0, 2.5))
        result = mae_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(0.54285, self.evaluate(result), atol=1e-5)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class MeanAbsolutePercentageErrorTest(tf.test.TestCase):
    def test_config(self):
        mape_obj = metrics.MeanAbsolutePercentageError(
            name="my_mape", dtype=tf.int32
        )
        self.assertEqual(mape_obj.name, "my_mape")
        self.assertEqual(mape_obj._dtype, tf.int32)

        # Check save and restore config
        mape_obj2 = metrics.MeanAbsolutePercentageError.from_config(
            mape_obj.get_config()
        )
        self.assertEqual(mape_obj2.name, "my_mape")
        self.assertEqual(mape_obj2._dtype, tf.int32)

    def test_unweighted(self):
        mape_obj = metrics.MeanAbsolutePercentageError()
        self.evaluate(tf.compat.v1.variables_initializer(mape_obj.variables))
        y_true = tf.constant(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = tf.constant(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )

        update_op = mape_obj.update_state(y_true, y_pred)
        self.evaluate(update_op)
        result = mape_obj.result()
        self.assertAllClose(35e7, result, atol=1e-5)

    def test_weighted(self):
        mape_obj = metrics.MeanAbsolutePercentageError()
        self.evaluate(tf.compat.v1.variables_initializer(mape_obj.variables))
        y_true = tf.constant(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = tf.constant(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )
        sample_weight = tf.constant((1.0, 1.5, 2.0, 2.5))
        result = mape_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(40e7, self.evaluate(result), atol=1e-5)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class MeanSquaredErrorTest(tf.test.TestCase):
    def test_config(self):
        mse_obj = metrics.MeanSquaredError(name="my_mse", dtype=tf.int32)
        self.assertEqual(mse_obj.name, "my_mse")
        self.assertEqual(mse_obj._dtype, tf.int32)

        # Check save and restore config
        mse_obj2 = metrics.MeanSquaredError.from_config(mse_obj.get_config())
        self.assertEqual(mse_obj2.name, "my_mse")
        self.assertEqual(mse_obj2._dtype, tf.int32)

    def test_unweighted(self):
        mse_obj = metrics.MeanSquaredError()
        self.evaluate(tf.compat.v1.variables_initializer(mse_obj.variables))
        y_true = tf.constant(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = tf.constant(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )

        update_op = mse_obj.update_state(y_true, y_pred)
        self.evaluate(update_op)
        result = mse_obj.result()
        self.assertAllClose(0.5, result, atol=1e-5)

    def test_weighted(self):
        mse_obj = metrics.MeanSquaredError()
        self.evaluate(tf.compat.v1.variables_initializer(mse_obj.variables))
        y_true = tf.constant(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = tf.constant(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )
        sample_weight = tf.constant((1.0, 1.5, 2.0, 2.5))
        result = mse_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(0.54285, self.evaluate(result), atol=1e-5)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class MeanSquaredLogarithmicErrorTest(tf.test.TestCase):
    def test_config(self):
        msle_obj = metrics.MeanSquaredLogarithmicError(
            name="my_msle", dtype=tf.int32
        )
        self.assertEqual(msle_obj.name, "my_msle")
        self.assertEqual(msle_obj._dtype, tf.int32)

        # Check save and restore config
        msle_obj2 = metrics.MeanSquaredLogarithmicError.from_config(
            msle_obj.get_config()
        )
        self.assertEqual(msle_obj2.name, "my_msle")
        self.assertEqual(msle_obj2._dtype, tf.int32)

    def test_unweighted(self):
        msle_obj = metrics.MeanSquaredLogarithmicError()
        self.evaluate(tf.compat.v1.variables_initializer(msle_obj.variables))
        y_true = tf.constant(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = tf.constant(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )

        update_op = msle_obj.update_state(y_true, y_pred)
        self.evaluate(update_op)
        result = msle_obj.result()
        self.assertAllClose(0.24022, result, atol=1e-5)

    def test_weighted(self):
        msle_obj = metrics.MeanSquaredLogarithmicError()
        self.evaluate(tf.compat.v1.variables_initializer(msle_obj.variables))
        y_true = tf.constant(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = tf.constant(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )
        sample_weight = tf.constant((1.0, 1.5, 2.0, 2.5))
        result = msle_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(0.26082, self.evaluate(result), atol=1e-5)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class RootMeanSquaredErrorTest(tf.test.TestCase):
    def test_config(self):
        rmse_obj = metrics.RootMeanSquaredError(name="rmse", dtype=tf.int32)
        self.assertEqual(rmse_obj.name, "rmse")
        self.assertEqual(rmse_obj._dtype, tf.int32)

        rmse_obj2 = metrics.RootMeanSquaredError.from_config(
            rmse_obj.get_config()
        )
        self.assertEqual(rmse_obj2.name, "rmse")
        self.assertEqual(rmse_obj2._dtype, tf.int32)

    def test_unweighted(self):
        rmse_obj = metrics.RootMeanSquaredError()
        self.evaluate(tf.compat.v1.variables_initializer(rmse_obj.variables))
        y_true = tf.constant((2, 4, 6))
        y_pred = tf.constant((1, 3, 2))

        update_op = rmse_obj.update_state(y_true, y_pred)
        self.evaluate(update_op)
        result = rmse_obj.result()
        # error = [-1, -1, -4], square(error) = [1, 1, 16], mean = 18/3 = 6
        self.assertAllClose(math.sqrt(6), result, atol=1e-3)

    def test_weighted(self):
        rmse_obj = metrics.RootMeanSquaredError()
        self.evaluate(tf.compat.v1.variables_initializer(rmse_obj.variables))
        y_true = tf.constant((2, 4, 6, 8))
        y_pred = tf.constant((1, 3, 2, 3))
        sample_weight = tf.constant((0, 1, 0, 1))
        result = rmse_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(math.sqrt(13), self.evaluate(result), atol=1e-3)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class LogCoshErrorTest(tf.test.TestCase):
    def setup(self):
        y_pred = np.asarray([1, 9, 2, -5, -2, 6]).reshape((2, 3))
        y_true = np.asarray([4, 8, 12, 8, 1, 3]).reshape((2, 3))

        self.batch_size = 6
        error = y_pred - y_true
        self.expected_results = np.log((np.exp(error) + np.exp(-error)) / 2)

        self.y_pred = tf.constant(y_pred, dtype=tf.float32)
        self.y_true = tf.constant(y_true)

    def test_config(self):
        logcosh_obj = metrics.LogCoshError(name="logcosh", dtype=tf.int32)
        self.assertEqual(logcosh_obj.name, "logcosh")
        self.assertEqual(logcosh_obj._dtype, tf.int32)

    def test_unweighted(self):
        self.setup()
        logcosh_obj = metrics.LogCoshError()
        self.evaluate(tf.compat.v1.variables_initializer(logcosh_obj.variables))

        update_op = logcosh_obj.update_state(self.y_true, self.y_pred)
        self.evaluate(update_op)
        result = logcosh_obj.result()
        expected_result = np.sum(self.expected_results) / self.batch_size
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_weighted(self):
        self.setup()
        logcosh_obj = metrics.LogCoshError()
        self.evaluate(tf.compat.v1.variables_initializer(logcosh_obj.variables))
        sample_weight = tf.constant([1.2, 3.4], shape=(2, 1))
        result = logcosh_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )

        sample_weight = np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape(
            (2, 3)
        )
        expected_result = np.multiply(self.expected_results, sample_weight)
        expected_result = np.sum(expected_result) / np.sum(sample_weight)
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class MeanRelativeErrorTest(tf.test.TestCase):
    def test_config(self):
        normalizer = tf.constant([1, 3], dtype=tf.float32)
        mre_obj = metrics.MeanRelativeError(normalizer=normalizer, name="mre")
        self.assertEqual(mre_obj.name, "mre")
        self.assertArrayNear(self.evaluate(mre_obj.normalizer), [1, 3], 1e-1)

        mre_obj2 = metrics.MeanRelativeError.from_config(mre_obj.get_config())
        self.assertEqual(mre_obj2.name, "mre")
        self.assertArrayNear(self.evaluate(mre_obj2.normalizer), [1, 3], 1e-1)

    def test_unweighted(self):
        np_y_pred = np.asarray([2, 4, 6, 8], dtype=np.float32)
        np_y_true = np.asarray([1, 3, 2, 3], dtype=np.float32)
        expected_error = np.mean(
            np.divide(np.absolute(np_y_pred - np_y_true), np_y_true)
        )

        y_pred = tf.constant(np_y_pred, shape=(1, 4), dtype=tf.float32)
        y_true = tf.constant(np_y_true, shape=(1, 4))

        mre_obj = metrics.MeanRelativeError(normalizer=y_true)
        self.evaluate(tf.compat.v1.variables_initializer(mre_obj.variables))

        result = mre_obj(y_true, y_pred)
        self.assertAllClose(self.evaluate(result), expected_error, atol=1e-3)

    def test_weighted(self):
        np_y_pred = np.asarray([2, 4, 6, 8], dtype=np.float32)
        np_y_true = np.asarray([1, 3, 2, 3], dtype=np.float32)
        sample_weight = np.asarray([0.2, 0.3, 0.5, 0], dtype=np.float32)
        rel_errors = np.divide(np.absolute(np_y_pred - np_y_true), np_y_true)
        expected_error = np.sum(rel_errors * sample_weight)

        y_pred = tf.constant(np_y_pred, dtype=tf.float32)
        y_true = tf.constant(np_y_true)

        mre_obj = metrics.MeanRelativeError(normalizer=y_true)
        self.evaluate(tf.compat.v1.variables_initializer(mre_obj.variables))

        result = mre_obj(
            y_true, y_pred, sample_weight=tf.constant(sample_weight)
        )
        self.assertAllClose(self.evaluate(result), expected_error, atol=1e-3)

    def test_zero_normalizer(self):
        y_pred = tf.constant([2, 4], dtype=tf.float32)
        y_true = tf.constant([1, 3])

        mre_obj = metrics.MeanRelativeError(normalizer=tf.zeros_like(y_true))
        self.evaluate(tf.compat.v1.variables_initializer(mre_obj.variables))

        result = mre_obj(y_true, y_pred)
        self.assertEqual(self.evaluate(result), 0)


@test_utils.run_v2_only
class R2ScoreTest(parameterized.TestCase, tf.test.TestCase):
    def _run_test(
        self,
        y_true,
        y_pred,
        sample_weights,
        class_aggregation,
        num_regressors,
        reference_result,
    ):
        y_true = tf.constant(y_true, dtype="float32")
        y_pred = tf.constant(y_pred, dtype="float32")
        r2 = metrics.R2Score(class_aggregation, num_regressors)
        r2.update_state(y_true, y_pred, sample_weights)
        result = r2.result().numpy()
        self.assertAllClose(result, reference_result, atol=1e-6)

    def test_config(self):
        r2_obj = metrics.R2Score(
            class_aggregation=None,
            num_regressors=2,
        )
        self.assertEqual(r2_obj.class_aggregation, None)
        self.assertEqual(r2_obj.num_regressors, 2)
        self.assertEqual(r2_obj.dtype, tf.float32)

        # Check save and restore config
        r2_obj2 = metrics.R2Score.from_config(r2_obj.get_config())
        self.assertEqual(r2_obj2.class_aggregation, None)
        self.assertEqual(r2_obj2.num_regressors, 2)
        self.assertEqual(r2_obj2.dtype, tf.float32)

    @parameterized.parameters(
        # class_aggregation, num_regressors, result
        (None, 0, [0.37, -1.295, 0.565]),
        ("uniform_average", 0, -0.12),
        ("variance_weighted_average", 0, -0.12),
    )
    def test_r2_sklearn_comparison(
        self, class_aggregation, num_regressors, result
    ):
        y_true = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        y_pred = [[0.4, 0.5, 0.6], [0.1, 0.2, 0.3], [0.5, 0.8, 0.2]]
        self._run_test(
            y_true,
            y_pred,
            None,
            class_aggregation=class_aggregation,
            num_regressors=num_regressors,
            reference_result=result,
        )

    @parameterized.parameters(
        # class_aggregation, num_regressors, result
        (None, 0, [0.17305559, -8.836666, -0.521]),
        (None, 1, [0.054920673, -10.241904, -0.7382858]),
        (None, 2, [-0.10259259, -12.115555, -1.0280001]),
        ("uniform_average", 0, -3.0615367889404297),
        ("uniform_average", 1, -3.641756534576416),
        ("uniform_average", 2, -4.415382385253906),
        ("variance_weighted_average", 0, -1.3710224628448486),
        ("variance_weighted_average", 1, -1.7097399234771729),
        ("variance_weighted_average", 2, -2.161363363265991),
    )
    def test_r2_tfa_comparison(self, class_aggregation, num_regressors, result):
        y_true = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        y_pred = [[0.4, 0.9, 1.6], [0.1, 1.2, 0.6], [1.5, 0.8, 0.6]]
        sample_weights = [0.8, 0.1, 0.4]
        self._run_test(
            y_true,
            y_pred,
            sample_weights,
            class_aggregation=class_aggregation,
            num_regressors=num_regressors,
            reference_result=result,
        )

    def test_errors(self):
        # Bad class_aggregation value
        with self.assertRaisesRegex(
            ValueError, "Invalid value for argument `class_aggregation`"
        ):
            metrics.R2Score(class_aggregation="wrong")

        # Bad num_regressors value
        with self.assertRaisesRegex(
            ValueError, "Invalid value for argument `num_regressors`"
        ):
            metrics.R2Score(num_regressors=-1)

        # Bad input shape
        with self.assertRaisesRegex(ValueError, "expects 2D inputs with shape"):
            r2 = metrics.R2Score()
            r2.update_state(tf.constant([0.0, 1.0]), tf.constant([0.0, 1.0]))

        with self.assertRaisesRegex(
            ValueError, "with output_dim fully defined"
        ):
            r2 = metrics.R2Score()
            r2.update_state(Input(shape=(None,)), tf.constant([[0.0], [1.0]]))


if __name__ == "__main__":
    tf.test.main()
