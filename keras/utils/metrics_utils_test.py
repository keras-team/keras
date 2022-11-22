# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for metrics_utils."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras import backend
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras.utils import metrics_utils


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class RaggedSizeOpTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        [
            {"x_list": [1], "y_list": [2]},
            {"x_list": [1, 2], "y_list": [2, 3]},
            {"x_list": [1, 2, 4], "y_list": [2, 3, 5]},
            {"x_list": [[1, 2], [3, 4]], "y_list": [[2, 3], [5, 6]]},
        ]
    )
    def test_passing_dense_tensors(self, x_list, y_list):
        x = tf.constant(x_list)
        y = tf.constant(y_list)
        [x, y], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values(
            [x, y]
        )
        x.shape.assert_is_compatible_with(y.shape)

    @parameterized.parameters(
        [
            {
                "x_list": [1],
            },
            {
                "x_list": [1, 2],
            },
            {
                "x_list": [1, 2, 4],
            },
            {
                "x_list": [[1, 2], [3, 4]],
            },
        ]
    )
    def test_passing_one_dense_tensor(self, x_list):
        x = tf.constant(x_list)
        [x], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values([x])

    @parameterized.parameters(
        [
            {"x_list": [1], "y_list": [2]},
            {"x_list": [1, 2], "y_list": [2, 3]},
            {"x_list": [1, 2, 4], "y_list": [2, 3, 5]},
            {"x_list": [[1, 2], [3, 4]], "y_list": [[2, 3], [5, 6]]},
            {"x_list": [[1, 2], [3, 4], [1]], "y_list": [[2, 3], [5, 6], [3]]},
            {"x_list": [[1, 2], [], [1]], "y_list": [[2, 3], [], [3]]},
        ]
    )
    def test_passing_both_ragged(self, x_list, y_list):
        x = tf.ragged.constant(x_list)
        y = tf.ragged.constant(y_list)
        [x, y], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values(
            [x, y]
        )
        x.shape.assert_is_compatible_with(y.shape)

    @parameterized.parameters(
        [
            {
                "x_list": [1],
            },
            {
                "x_list": [1, 2],
            },
            {
                "x_list": [1, 2, 4],
            },
            {
                "x_list": [[1, 2], [3, 4]],
            },
            {
                "x_list": [[1, 2], [3, 4], [1]],
            },
            {
                "x_list": [[1, 2], [], [1]],
            },
        ]
    )
    def test_passing_one_ragged(self, x_list):
        x = tf.ragged.constant(x_list)
        [x], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values([x])

    @parameterized.parameters(
        [
            {"x_list": [1], "y_list": [2], "mask_list": [0]},
            {"x_list": [1, 2], "y_list": [2, 3], "mask_list": [0, 1]},
            {"x_list": [1, 2, 4], "y_list": [2, 3, 5], "mask_list": [1, 1, 1]},
            {
                "x_list": [[1, 2], [3, 4]],
                "y_list": [[2, 3], [5, 6]],
                "mask_list": [[1, 1], [0, 1]],
            },
            {
                "x_list": [[1, 2], [3, 4], [1]],
                "y_list": [[2, 3], [5, 6], [3]],
                "mask_list": [[1, 1], [0, 0], [1]],
            },
            {
                "x_list": [[1, 2], [], [1]],
                "y_list": [[2, 3], [], [3]],
                "mask_list": [[1, 1], [], [0]],
            },
        ]
    )
    def test_passing_both_ragged_with_mask(self, x_list, y_list, mask_list):
        x = tf.ragged.constant(x_list)
        y = tf.ragged.constant(y_list)
        mask = tf.ragged.constant(mask_list)
        [
            x,
            y,
        ], mask = metrics_utils.ragged_assert_compatible_and_get_flat_values(
            [x, y], mask
        )
        x.shape.assert_is_compatible_with(y.shape)
        y.shape.assert_is_compatible_with(mask.shape)

    @parameterized.parameters(
        [
            {"x_list": [1], "mask_list": [0]},
            {"x_list": [1, 2], "mask_list": [0, 1]},
            {"x_list": [1, 2, 4], "mask_list": [1, 1, 1]},
            {"x_list": [[1, 2], [3, 4]], "mask_list": [[1, 1], [0, 1]]},
            {
                "x_list": [[1, 2], [3, 4], [1]],
                "mask_list": [[1, 1], [0, 0], [1]],
            },
            {"x_list": [[1, 2], [], [1]], "mask_list": [[1, 1], [], [0]]},
        ]
    )
    def test_passing_one_ragged_with_mask(self, x_list, mask_list):
        x = tf.ragged.constant(x_list)
        mask = tf.ragged.constant(mask_list)
        [x], mask = metrics_utils.ragged_assert_compatible_and_get_flat_values(
            [x], mask
        )
        x.shape.assert_is_compatible_with(mask.shape)

    @parameterized.parameters(
        [
            {"x_list": [[[1, 3]]], "y_list": [[2, 3]]},
        ]
    )
    def test_failing_different_ragged_and_dense_ranks(self, x_list, y_list):
        x = tf.ragged.constant(x_list)
        y = tf.ragged.constant(y_list)
        with self.assertRaises(ValueError):
            [
                x,
                y,
            ], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values(
                [x, y]
            )

    @parameterized.parameters(
        [
            {"x_list": [[[1, 3]]], "y_list": [[[2, 3]]], "mask_list": [[0, 1]]},
        ]
    )
    def test_failing_different_mask_ranks(self, x_list, y_list, mask_list):
        x = tf.ragged.constant(x_list)
        y = tf.ragged.constant(y_list)
        mask = tf.ragged.constant(mask_list)
        with self.assertRaises(ValueError):
            [
                x,
                y,
            ], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values(
                [x, y], mask
            )

    # we do not support such cases that ragged_ranks are different but overall
    # dimension shapes and sizes are identical due to adding too much
    # performance overheads to the overall use cases.
    def test_failing_different_ragged_ranks(self):
        dt = tf.constant([[[1, 2]]])
        # adding a ragged dimension
        x = tf.RaggedTensor.from_row_splits(dt, row_splits=[0, 1])
        y = tf.ragged.constant([[[[1, 2]]]])
        with self.assertRaises(ValueError):
            [
                x,
                y,
            ], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values(
                [x, y]
            )


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class FilterTopKTest(tf.test.TestCase, parameterized.TestCase):
    def test_one_dimensional(self):
        x = tf.constant([0.3, 0.1, 0.2, -0.5, 42.0])
        top_1 = self.evaluate(metrics_utils._filter_top_k(x=x, k=1))
        top_2 = self.evaluate(metrics_utils._filter_top_k(x=x, k=2))
        top_3 = self.evaluate(metrics_utils._filter_top_k(x=x, k=3))

        self.assertAllClose(
            top_1,
            [
                metrics_utils.NEG_INF,
                metrics_utils.NEG_INF,
                metrics_utils.NEG_INF,
                metrics_utils.NEG_INF,
                42.0,
            ],
        )
        self.assertAllClose(
            top_2,
            [
                0.3,
                metrics_utils.NEG_INF,
                metrics_utils.NEG_INF,
                metrics_utils.NEG_INF,
                42.0,
            ],
        )
        self.assertAllClose(
            top_3,
            [0.3, metrics_utils.NEG_INF, 0.2, metrics_utils.NEG_INF, 42.0],
        )

    def test_three_dimensional(self):
        x = tf.constant(
            [
                [[0.3, 0.1, 0.2], [-0.3, -0.2, -0.1]],
                [[5.0, 0.2, 42.0], [-0.3, -0.6, -0.99]],
            ]
        )
        top_2 = self.evaluate(metrics_utils._filter_top_k(x=x, k=2))

        self.assertAllClose(
            top_2,
            [
                [
                    [0.3, metrics_utils.NEG_INF, 0.2],
                    [metrics_utils.NEG_INF, -0.2, -0.1],
                ],
                [
                    [5.0, metrics_utils.NEG_INF, 42.0],
                    [-0.3, -0.6, metrics_utils.NEG_INF],
                ],
            ],
        )

    def test_handles_dynamic_shapes(self):
        # See b/150281686.  # GOOGLE_INTERNAL

        def _identity(x):
            return x

        def _filter_top_k(x):
            # This loses the static shape.
            x = tf.numpy_function(_identity, (x,), tf.float32)

            return metrics_utils._filter_top_k(x=x, k=2)

        x = tf.constant([0.3, 0.1, 0.2, -0.5, 42.0])
        top_2 = self.evaluate(_filter_top_k(x))
        self.assertAllClose(
            top_2,
            [
                0.3,
                metrics_utils.NEG_INF,
                metrics_utils.NEG_INF,
                metrics_utils.NEG_INF,
                42.0,
            ],
        )


class MatchesMethodsTest(tf.test.TestCase, parameterized.TestCase):
    def test_sparse_categorical_matches(self):
        matches_method = metrics_utils.sparse_categorical_matches

        # Test return tensor is type float
        y_true = tf.constant(np.random.randint(0, 7, (6,)))
        y_pred = tf.constant(np.random.random((6, 7)))
        self.assertEqual(matches_method(y_true, y_pred).dtype, backend.floatx())

        # Tests that resulting Tensor always has same shape as y_true. Tests
        # from 1 dim to 4 dims
        dims = []
        for _ in range(4):
            dims.append(np.random.randint(1, 7))
            y_true = tf.constant(np.random.randint(0, 7, dims))
            y_pred = tf.constant(np.random.random(dims + [3]))
            self.assertEqual(matches_method(y_true, y_pred).shape, y_true.shape)

        # Test correctness if the shape of y_true is (num_samples,)
        y_true = tf.constant([1.0, 0.0, 0.0, 0.0])
        y_pred = tf.constant([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3], [0.9, 0.1]])
        self.assertAllEqual(
            matches_method(y_true, y_pred), [0.0, 1.0, 1.0, 1.0]
        )

        # Test correctness if the shape of y_true is (num_samples, 1)
        y_true = tf.constant([[1.0], [0.0], [0.0], [0.0]])
        y_pred = tf.constant([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3], [0.9, 0.1]])
        self.assertAllEqual(
            matches_method(y_true, y_pred), [[0.0], [1.0], [1.0], [1.0]]
        )

        # Test correctness if the shape of y_true is (batch_size, seq_length)
        # and y_pred is (batch_size, seq_length, num_classes)
        y_pred = tf.constant(
            [
                [[0.2, 0.3, 0.1], [0.1, 0.2, 0.7]],
                [[0.3, 0.2, 0.1], [0.7, 0.2, 0.1]],
            ]
        )
        y_true = tf.constant([[1, 0], [1, 0]])
        self.assertAllEqual(
            matches_method(y_true, y_pred), [[1.0, 0.0], [0.0, 1.0]]
        )

    def test_sparse_top_k_categorical_matches(self):
        matches_method = metrics_utils.sparse_top_k_categorical_matches

        # Test return tensor is type float
        y_true = tf.constant(np.random.randint(0, 7, (6,)))
        y_pred = tf.constant(np.random.random((6, 7)), dtype=tf.float32)
        self.assertEqual(
            matches_method(y_true, y_pred, 1).dtype, backend.floatx()
        )

        # Tests that resulting Tensor always has same shape as y_true. Tests
        # from 1 dim to 4 dims
        dims = []
        for _ in range(4):
            dims.append(np.random.randint(1, 7))
            y_true = tf.constant(np.random.randint(0, 7, dims))
            y_pred = tf.constant(np.random.random(dims + [3]), dtype=tf.float32)
            self.assertEqual(
                matches_method(y_true, y_pred, 1).shape, y_true.shape
            )

        # Test correctness if the shape of y_true is (num_samples,) for k =
        # 1,2,3
        y_true = tf.constant([1.0, 0.0, 0.0, 0.0])
        y_pred = tf.constant(
            [[0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.6, 0.3, 0.1], [0.0, 0.1, 0.9]]
        )
        self.assertAllEqual(
            matches_method(y_true, y_pred, 1), [0.0, 1.0, 1.0, 0.0]
        )
        self.assertAllEqual(
            matches_method(y_true, y_pred, 2), [1.0, 1.0, 1.0, 0.0]
        )
        self.assertAllEqual(
            matches_method(y_true, y_pred, 3), [1.0, 1.0, 1.0, 1.0]
        )

        # Test correctness if the shape of y_true is (num_samples, 1)
        # for k = 1,2,3
        y_true = tf.constant([[1.0], [0.0], [0.0], [0.0]])
        y_pred = tf.constant(
            [[0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.6, 0.3, 0.1], [0.0, 0.1, 0.9]]
        )
        self.assertAllEqual(
            matches_method(y_true, y_pred, 1), [[0.0], [1.0], [1.0], [0.0]]
        )
        self.assertAllEqual(
            matches_method(y_true, y_pred, 2), [[1.0], [1.0], [1.0], [0.0]]
        )
        self.assertAllEqual(
            matches_method(y_true, y_pred, 3), [[1.0], [1.0], [1.0], [1.0]]
        )

        # Test correctness if the shape of y_true is (batch_size, seq_length)
        # and y_pred is (batch_size, seq_length, num_classes) for k = 1,2,3
        y_pred = tf.constant(
            [
                [[0.2, 0.3, 0.1], [0.1, 0.2, 0.7]],
                [[0.3, 0.2, 0.1], [0.7, 0.2, 0.1]],
            ]
        )
        y_true = tf.constant([[1, 0], [1, 0]])
        self.assertAllEqual(
            matches_method(y_true, y_pred, 1), [[1.0, 0.0], [0.0, 1.0]]
        )
        self.assertAllEqual(
            matches_method(y_true, y_pred, 2), [[1.0, 0.0], [1.0, 1.0]]
        )
        self.assertAllEqual(
            matches_method(y_true, y_pred, 3), [[1.0, 1.0], [1.0, 1.0]]
        )

    def test_binary_matches(self):
        matches_method = metrics_utils.binary_matches

        # Test return tensor is type float
        y_true = tf.constant(np.random.random((6, 7)))
        y_pred = tf.constant(np.random.random((6, 7)))
        self.assertEqual(
            matches_method(y_true, y_pred, 0.5).dtype, backend.floatx()
        )

        # Tests that resulting Tensor always has same shape as y_true. Tests
        # from 1 dim to 4 dims.
        dims = []
        for _ in range(4):
            dims.append(np.random.randint(1, 7))
            y_true = y_pred = tf.constant(np.random.random(dims))
            self.assertEqual(
                matches_method(y_true, y_pred, 0.0).shape, y_true.shape
            )

        # Testing for correctness shape (num_samples, 1)
        y_true = tf.constant([[1.0], [0.0], [1.0], [1.0]])
        y_pred = tf.constant([[0.75], [0.2], [0.2], [0.75]])
        self.assertAllEqual(
            matches_method(y_true, y_pred, 0.5), [[1.0], [1.0], [0.0], [1.0]]
        )

        # Testing for correctness shape (num_samples,)
        y_true = tf.constant([1.0, 0.0, 1.0, 1.0])
        y_pred = tf.constant([0.75, 0.2, 0.2, 0.75])
        self.assertAllEqual(
            matches_method(y_true, y_pred, 0.5), [1.0, 1.0, 0.0, 1.0]
        )

        # Testing for correctness batches of sequences
        # shape (num_samples, seq_len)
        y_true = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
        y_pred = tf.constant(
            [[0.75, 0.2], [0.2, 0.75], [0.2, 0.75], [0.75, 0.2]]
        )
        self.assertAllEqual(
            matches_method(y_true, y_pred, 0.5),
            [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]],
        )


@test_utils.run_v2_only
class UpdateConfusionMatrixVarTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.tp = metrics_utils.ConfusionMatrix.TRUE_POSITIVES
        self.tn = metrics_utils.ConfusionMatrix.TRUE_NEGATIVES
        self.fp = metrics_utils.ConfusionMatrix.FALSE_POSITIVES
        self.fn = metrics_utils.ConfusionMatrix.FALSE_NEGATIVES
        self.variables_to_update = {
            self.tp: tf.Variable([0], dtype=tf.float32),
            self.tn: tf.Variable([0], dtype=tf.float32),
            self.fp: tf.Variable([0], dtype=tf.float32),
            self.fn: tf.Variable([0], dtype=tf.float32),
        }

    def test_without_sample_weight(self):
        y_true = tf.constant([[1, 1, 0], [0, 0, 1]])
        y_pred = tf.constant([[0.8, 0.7, 0.1], [0.1, 0.6, 0.4]])
        thresholds = [0.5]

        metrics_utils.update_confusion_matrix_variables(
            variables_to_update=self.variables_to_update,
            y_true=y_true,
            y_pred=y_pred,
            thresholds=thresholds,
        )
        self.assertEqual(self.variables_to_update[self.tp].numpy()[0], 2)
        self.assertEqual(self.variables_to_update[self.tn].numpy()[0], 2)
        self.assertEqual(self.variables_to_update[self.fp].numpy()[0], 1)
        self.assertEqual(self.variables_to_update[self.fn].numpy()[0], 1)

    def test_with_sample_weight(self):
        y_true = tf.constant([[1, 1, 0], [0, 0, 1]])
        y_pred = tf.constant([[0.8, 0.7, 0.1], [0.1, 0.6, 0.4]])
        thresholds = [0.5]
        sample_weight = [2, 1]

        metrics_utils.update_confusion_matrix_variables(
            variables_to_update=self.variables_to_update,
            y_true=y_true,
            y_pred=y_pred,
            thresholds=thresholds,
            sample_weight=sample_weight,
        )
        self.assertEqual(self.variables_to_update[self.tp].numpy()[0], 4)
        self.assertEqual(self.variables_to_update[self.tn].numpy()[0], 3)
        self.assertEqual(self.variables_to_update[self.fp].numpy()[0], 1)
        self.assertEqual(self.variables_to_update[self.fn].numpy()[0], 1)

    def test_with_class_id(self):
        y_true = tf.constant([[1, 1, 0], [0, 0, 1]])
        y_pred = tf.constant([[0.8, 0.7, 0.1], [0.1, 0.6, 0.4]])
        thresholds = [0.5]
        class_id = 2

        metrics_utils.update_confusion_matrix_variables(
            variables_to_update=self.variables_to_update,
            y_true=y_true,
            y_pred=y_pred,
            thresholds=thresholds,
            class_id=class_id,
        )
        self.assertEqual(self.variables_to_update[self.tp].numpy()[0], 0)
        self.assertEqual(self.variables_to_update[self.tn].numpy()[0], 1)
        self.assertEqual(self.variables_to_update[self.fp].numpy()[0], 0)
        self.assertEqual(self.variables_to_update[self.fn].numpy()[0], 1)

    def test_with_sample_weight_and_classid(self):
        y_true = tf.constant([[1, 1, 0], [0, 0, 1]])
        y_pred = tf.constant([[0.8, 0.7, 0.1], [0.1, 0.6, 0.4]])
        thresholds = [0.5]
        sample_weight = [2, 1]
        class_id = 2

        metrics_utils.update_confusion_matrix_variables(
            variables_to_update=self.variables_to_update,
            y_true=y_true,
            y_pred=y_pred,
            thresholds=thresholds,
            sample_weight=sample_weight,
            class_id=class_id,
        )
        self.assertEqual(self.variables_to_update[self.tp].numpy()[0], 0)
        self.assertEqual(self.variables_to_update[self.tn].numpy()[0], 2)
        self.assertEqual(self.variables_to_update[self.fp].numpy()[0], 0)
        self.assertEqual(self.variables_to_update[self.fn].numpy()[0], 1)


if __name__ == "__main__":
    tf.test.main()
