# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests BaseDenseAttention layer."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras.layers.attention.base_dense_attention import BaseDenseAttention
from keras.layers.attention.base_dense_attention import _lower_triangular_mask
from keras.testing_infra import test_combinations


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class BaseDenseAttentionTest(tf.test.TestCase, parameterized.TestCase):
    def test_one_dim_with_mask(self):
        # Scores tensor of shape [1, 1, 1]
        scores = np.array([[[1.1]]], dtype=np.float32)
        # Value tensor of shape [1, 1, 1]
        v = np.array([[[1.6]]], dtype=np.float32)
        # Scores mask tensor of shape [1, 1, 1]
        scores_mask = np.array([[[True]]], dtype=np.bool_)
        actual, actual_scores = BaseDenseAttention()._apply_scores(
            scores=scores, value=v, scores_mask=scores_mask
        )

        # Expected softmax_scores = [[[1]]]
        expected_scores = np.array([[[1.0]]], dtype=np.float32)
        self.assertAllClose(expected_scores, actual_scores)
        # Expected tensor of shape [1, 1, 1].
        # expected000 = softmax_scores[0, 0] * 1.6 = 1.6
        expected = np.array([[[1.6]]], dtype=np.float32)
        self.assertAllClose(expected, actual)

    def test_one_dim_no_mask(self):
        # Scores tensor of shape [1, 1, 1]
        scores = np.array([[[1.1]]], dtype=np.float32)
        # Value tensor of shape [1, 1, 1]
        v = np.array([[[1.6]]], dtype=np.float32)
        actual, actual_scores = BaseDenseAttention()._apply_scores(
            scores=scores, value=v
        )

        # Expected softmax_scores = [[[1]]]
        expected_scores = np.array([[[1.0]]], dtype=np.float32)
        self.assertAllClose(expected_scores, actual_scores)
        # Expected tensor of shape [1, 1, 1].
        # expected000 = softmax_scores[0, 0] * 1.6 = 1.6
        expected = np.array([[[1.6]]], dtype=np.float32)
        self.assertAllClose(expected, actual)

    def test_multi_dim_with_mask(self):
        # Scores tensor of shape [1, 1, 3]
        scores = np.array([[[1.0, 0.0, 1.0]]], dtype=np.float32)
        # Value tensor of shape [1, 3, 1]
        v = np.array([[[1.6], [0.7], [-0.8]]], dtype=np.float32)
        # Scores mask tensor of shape [1, 1, 3]
        scores_mask = np.array([[[True, True, False]]], dtype=np.bool_)
        actual, actual_scores = BaseDenseAttention()._apply_scores(
            scores=scores, value=v, scores_mask=scores_mask
        )

        # Expected softmax scores = softmax(scores) with zeros in positions
        # where v_mask == False.
        # => softmax_scores000 = exp(1)/(exp(1) + exp(0)) = 0.73105857863
        #    softmax_scores001 = exp(0)/(exp(1) + exp(0)) = 0.26894142137
        #    softmax_scores002 = 0
        expected_scores = np.array(
            [[[0.73105857863, 0.26894142137, 0.0]]], dtype=np.float32
        )
        self.assertAllClose(expected_scores, actual_scores)
        # Expected tensor of shape [1, 1, 1].
        # expected000 = 0.73105857863 * 1.6 + 0.26894142137 * 0.7 - 0 * 0.8
        #             = 1.35795272077
        expected = np.array([[[1.35795272077]]], dtype=np.float32)
        self.assertAllClose(expected, actual)

    def test_multi_dim_no_mask(self):
        # Scores tensor of shape [1, 1, 3]
        scores = np.array([[[1.0, 0.0, 1.0]]], dtype=np.float32)
        # Value tensor of shape [1, 3, 1]
        v = np.array([[[1.6], [0.7], [-0.8]]], dtype=np.float32)
        actual, actual_scores = BaseDenseAttention()._apply_scores(
            scores=scores, value=v
        )

        # Expected softmax_scores = softmax(scores).
        # => softmax_scores000 = exp(1)/(exp(1) + exp(0) + exp(1))
        #                      = 0.42231879825
        #    softmax_scores001 = exp(0)/(exp(1) + exp(0) + exp(1))
        #                      = 0.15536240349
        #    softmax_scores002 = exp(1)/(exp(1) + exp(0) + exp(1))
        #                      = 0.42231879825
        expected_scores = np.array(
            [[[0.42231879825, 0.15536240349, 0.42231879825]]], dtype=np.float32
        )
        self.assertAllClose(expected_scores, actual_scores)
        # Expected tensor of shape [1, 1, 1].
        # expected000 = 0.42231879825 * 1.6 + 0.15536240349 * 0.7
        #               - 0.42231879825 * 0.8
        #             = 0.44660872104
        expected = np.array([[[0.44660872104]]], dtype=np.float32)
        self.assertAllClose(expected, actual)

    def test_one_dim_batch_size_two(self):
        # Scores tensor of shape [2, 1, 1]
        scores = np.array([[[1.1]], [[2.1]]], dtype=np.float32)
        # Value tensor of shape [2, 1, 1]
        v = np.array([[[1.6]], [[2.6]]], dtype=np.float32)
        # Scpres mask tensor of shape [2, 1, 1]
        scores_mask = np.array([[[True]], [[True]]], dtype=np.bool_)
        actual, actual_scores = BaseDenseAttention()._apply_scores(
            scores=scores, value=v, scores_mask=scores_mask
        )

        # Expected softmax_scores = [[[1]], [[1]]]
        expected_scores = np.array([[[1.0]], [[1.0]]], dtype=np.float32)
        self.assertAllClose(expected_scores, actual_scores)
        # Expected tensor of shape [2, 1, 1].
        # expected000 = softmax_scores[0, 0] * 1.6 = 1.6
        # expected100 = softmax_scores[1, 0] * 2.6 = 2.6
        expected = np.array([[[1.6]], [[2.6]]], dtype=np.float32)
        self.assertAllClose(expected, actual)

    def test_shape_with_dropout(self):
        # scores: Scores float tensor of shape `[batch_size, tq, tv]`.
        # value: Value tensor of shape `[batch_size, tv, dim]`.
        batch_size = 4
        tq = 5
        tv = 6
        dim = 7
        scores = np.ones((batch_size, tq, tv))
        value = np.ones((batch_size, tv, dim))
        actual, actual_scores = BaseDenseAttention(dropout=0.1)._apply_scores(
            scores=scores, value=value, training=False
        )

        # Expected Tensor of shape `[batch_size, tq, tv]`.
        expected_scores_shape = [batch_size, tq, tv]
        self.assertAllEqual(expected_scores_shape, tf.shape(actual_scores))
        # Expected Tensor of shape `[batch_size, tq, dim]`.
        expected_shape = [batch_size, tq, dim]
        self.assertAllEqual(expected_shape, tf.shape(actual))

    def test_skip_rng_init_when_no_dropout(self):
        batch_size = 4
        tq = 5
        tv = 6
        dim = 7
        scores = np.ones((batch_size, tq, tv))
        value = np.ones((batch_size, tv, dim))
        layer = BaseDenseAttention()
        layer.build(None)  # The input shape is not used by this layer
        _, _ = layer._apply_scores(scores=scores, value=value, training=True)
        # Make sure the rng is not built and no tf.random.Generator created.
        self.assertFalse(layer._random_generator._built)
        self.assertIsNone(getattr(layer._random_generator, "_generator", None))


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class LowerTriangularMaskTest(tf.test.TestCase, parameterized.TestCase):
    def test_square_shape(self):
        actual = _lower_triangular_mask([3, 3])
        expected = np.array(
            [[True, False, False], [True, True, False], [True, True, True]],
            dtype=np.bool_,
        )
        self.assertAllEqual(expected, actual)

    def test_orthogonal_shape(self):
        actual = _lower_triangular_mask([3, 2])
        expected = np.array(
            [[True, False], [True, True], [True, True]], dtype=np.bool_
        )
        self.assertAllEqual(expected, actual)

    def test_three_dim(self):
        actual = _lower_triangular_mask([1, 3, 3])
        expected = np.array(
            [[[True, False, False], [True, True, False], [True, True, True]]],
            dtype=np.bool_,
        )
        self.assertAllEqual(expected, actual)


if __name__ == "__main__":
    tf.test.main()
