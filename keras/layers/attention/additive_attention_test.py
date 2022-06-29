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
"""Tests AdditiveAttention layer."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.mixed_precision import policy
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class AdditiveAttentionTest(tf.test.TestCase, parameterized.TestCase):
    def test_calculate_scores_one_dim(self):
        # Query tensor of shape [1, 1, 1]
        q = np.array([[[1.1]]], dtype=np.float32)
        # Key tensor of shape [1, 1, 1]
        k = np.array([[[1.6]]], dtype=np.float32)
        attention_layer = keras.layers.AdditiveAttention()
        attention_layer.build(input_shape=([1, 1, 1], [1, 1, 1]))
        # Scale tensor of shape [1]
        attention_layer.scale = np.array([[[0.5]]], dtype=np.float32)
        actual = attention_layer._calculate_scores(query=q, key=k)

        # Expected tensor of shape [1, 1, 1].
        # expected000 = 0.5 * tanh(1.1 + 1.6) = 0.49550372683
        expected = np.array([[[0.49550372683]]], dtype=np.float32)
        self.assertAllClose(expected, actual)

    def test_calculate_scores_multi_dim(self):
        # Query tensor of shape [1, 2, 4]
        q = np.array(
            [[[1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3]]], dtype=np.float32
        )
        # Key tensor of shape [1, 3, 4]
        k = np.array(
            [
                [
                    [1.5, 1.6, 1.7, 1.8],
                    [2.5, 2.6, 2.7, 2.8],
                    [3.5, 3.6, 3.7, 3.8],
                ]
            ],
            dtype=np.float32,
        )
        attention_layer = keras.layers.AdditiveAttention()
        attention_layer.build(input_shape=([1, 2, 4], [1, 3, 4]))
        # Scale tensor of shape [4]
        attention_layer.scale = np.array(
            [[[0.5, 0.6, 0.7, 0.8]]], dtype=np.float32
        )
        actual = attention_layer._calculate_scores(query=q, key=k)

        # expected000 = 0.5*tanh(1.+1.5) + 0.6*tanh(1.1+1.6) + \
        #     0.7*tanh(1.2+1.7) + 0.8*tanh(1.3+1.8) = 2.58044532581
        # expected001 = 0.5*tanh(1.+2.5) + 0.6*tanh(1.1+2.6) + \
        #     0.7*tanh(1.2+2.7) + 0.8*tanh(1.3+2.8) = 2.59734317449
        # expected002 = 0.5*tanh(1.+3.5) + 0.6*tanh(1.1+3.6) + \
        #     0.7*tanh(1.2+3.7) + 0.8*tanh(1.3+3.8) = 2.59964024652
        # expected010 = 0.5*tanh(2.+1.5) + 0.6*tanh(2.1+1.6) + \
        #     0.7*tanh(2.2+1.7) + 0.8*tanh(2.3+1.8) = 2.59734317449
        # expected011 = 0.5*tanh(2.+2.5) + 0.6*tanh(2.1+2.6) + \
        #     0.7*tanh(2.2+2.7) + 0.8*tanh(2.3+2.8) = 2.59964024652
        # expected012 = 0.5*tanh(2.+3.5) + 0.6*tanh(2.1+3.6) + \
        #     0.7*tanh(2.2+3.7) + 0.8*tanh(2.3+3.8) = 2.59995130916
        expected = np.array(
            [
                [
                    [2.58044532581, 2.59734317449, 2.59964024652],
                    [2.59734317449, 2.59964024652, 2.59995130916],
                ]
            ],
            dtype=np.float32,
        )
        self.assertAllClose(expected, actual)

    def test_calculate_scores_one_dim_batch_size_two(self):
        # Query tensor of shape [2, 1, 1]
        q = np.array([[[1.1]], [[2.1]]], dtype=np.float32)
        # Key tensor of shape [2, 1, 1]
        k = np.array([[[1.6]], [[2.6]]], dtype=np.float32)
        attention_layer = keras.layers.AdditiveAttention()
        attention_layer.build(input_shape=([2, 1, 1], [2, 1, 1]))
        # Scale tensor of shape [1]
        attention_layer.scale = np.array([[[0.5]]], dtype=np.float32)
        actual = attention_layer._calculate_scores(query=q, key=k)

        # Expected tensor of shape [2, 1, 1].
        # expected000 = 0.5 * tanh(1.1 + 1.6) = 0.49550372683
        # expected100 = 0.5 * tanh(2.1 + 2.6) = 0.49991728277
        expected = np.array(
            [[[0.49550372683]], [[0.49991728277]]], dtype=np.float32
        )
        self.assertAllClose(expected, actual)

    def test_shape(self):
        # Query tensor of shape [1, 2, 4]
        q = np.array(
            [[[1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3]]], dtype=np.float32
        )
        # Value tensor of shape [1, 3, 4]
        v = np.array(
            [
                [
                    [1.5, 1.6, 1.7, 1.8],
                    [2.5, 2.6, 2.7, 2.8],
                    [3.5, 3.6, 3.7, 3.8],
                ]
            ],
            dtype=np.float32,
        )
        # Value mask tensor of shape [1, 3]
        v_mask = np.array([[True, True, False]], dtype=np.bool_)
        attention_layer = keras.layers.AdditiveAttention()
        actual = attention_layer([q, v], mask=[None, v_mask])

        expected_shape = [1, 2, 4]
        self.assertAllEqual(expected_shape, tf.shape(actual))

    def test_shape_no_scale(self):
        # Query tensor of shape [1, 2, 4]
        q = np.array(
            [[[1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3]]], dtype=np.float32
        )
        # Value tensor of shape [1, 3, 4]
        v = np.array(
            [
                [
                    [1.5, 1.6, 1.7, 1.8],
                    [2.5, 2.6, 2.7, 2.8],
                    [3.5, 3.6, 3.7, 3.8],
                ]
            ],
            dtype=np.float32,
        )
        # Value mask tensor of shape [1, 3]
        v_mask = np.array([[True, True, False]], dtype=np.bool_)
        attention_layer = keras.layers.AdditiveAttention(use_scale=False)
        actual = attention_layer([q, v], mask=[None, v_mask])

        expected_shape = [1, 2, 4]
        self.assertAllEqual(expected_shape, tf.shape(actual))

    def test_shape_with_key(self):
        # Query tensor of shape [1, 2, 4]
        q = np.array(
            [[[1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3]]], dtype=np.float32
        )
        # Value tensor of shape [1, 3, 4]
        v = np.array(
            [
                [
                    [1.5, 1.6, 1.7, 1.8],
                    [2.5, 2.6, 2.7, 2.8],
                    [3.5, 3.6, 3.7, 3.8],
                ]
            ],
            dtype=np.float32,
        )
        # Key tensor of shape [1, 3, 4]
        k = np.array(
            [
                [
                    [1.5, 1.6, 1.7, 1.8],
                    [2.5, 2.6, 2.7, 2.8],
                    [3.5, 3.6, 3.7, 3.8],
                ]
            ],
            dtype=np.float32,
        )
        # Value mask tensor of shape [1, 3]
        v_mask = np.array([[True, True, False]], dtype=np.bool_)
        attention_layer = keras.layers.AdditiveAttention()
        actual = attention_layer([q, v, k], mask=[None, v_mask])

        expected_shape = [1, 2, 4]
        self.assertAllEqual(expected_shape, tf.shape(actual))

    def test_multi_dim(self):
        # Query tensor of shape [1, 1, 1]
        q = np.array([[[1.1]]], dtype=np.float32)
        # Value tensor of shape [1, 3, 1]
        v = np.array([[[1.6], [0.7], [-0.8]]], dtype=np.float32)
        # Value mask tensor of shape [1, 3]
        v_mask = np.array([[True, True, False]], dtype=np.bool_)
        attention_layer = keras.layers.AdditiveAttention()
        attention_layer.build(input_shape=([1, 1, 1], [1, 3, 1]))
        # Scale tensor of shape [1]
        attention_layer.scale = np.array([[[0.5]]], dtype=np.float32)
        actual = attention_layer([q, v], mask=[None, v_mask])

        # Expected scores of shape [1, 1, 3]
        # scores = [[[0.5 * tanh(1.1 + 1.6),
        #             0.5 * tanh(1.1 + 0.7),
        #             0.5 * tanh(1.1 - 0.8)]]]
        #        = [[[0.49550372683, 0.47340300642, 0.14565630622]]]
        # Expected attention distribution = softmax(scores) with zeros in
        # positions where v_mask == False.
        # => attention_distribution000
        #      = exp(0.49550372683)/(exp(0.49550372683) + exp(0.47340300642))
        #      = 0.50552495521
        #    attention_distribution001
        #      = exp(0.47340300642)/(exp(0.49550372683) + exp(0.47340300642))
        #      = 0.49447504478
        #    attention_distribution002 = 0
        #
        # Expected tensor of shape [1, 1, 1].
        # expected000 = 0.50552495521 * 1.6 + 0.49447504478 * 0.7 - 0 * 0.8
        #             = 1.15497245968
        expected = np.array([[[1.15497245968]]], dtype=np.float32)
        self.assertAllClose(expected, actual)

    def test_multi_dim_with_key(self):
        # Query tensor of shape [1, 1, 1]
        q = np.array([[[1.1]]], dtype=np.float32)
        # Value tensor of shape [1, 3, 1]
        v = np.array([[[0.5], [0.8], [-0.3]]], dtype=np.float32)
        # Key tensor of shape [1, 3, 1]
        k = np.array([[[1.6], [0.7], [-0.8]]], dtype=np.float32)
        # Value mask tensor of shape [1, 3]
        v_mask = np.array([[True, True, False]], dtype=np.bool_)
        attention_layer = keras.layers.AdditiveAttention()
        attention_layer.build(input_shape=([1, 1, 1], [1, 3, 1]))
        # Scale tensor of shape [1]
        attention_layer.scale = np.array([[[0.5]]], dtype=np.float32)
        actual = attention_layer([q, v, k], mask=[None, v_mask])

        # Expected scores of shape [1, 1, 3]
        # scores = [[[0.5 * tanh(1.1 + 1.6),
        #             0.5 * tanh(1.1 + 0.7),
        #             0.5 * tanh(1.1 - 0.8)]]]
        #        = [[[0.49550372683, 0.47340300642, 0.14565630622]]]
        # Expected attention distribution = softmax(scores) with zeros in
        # positions where v_mask == False.
        # => attention_distribution000
        #        = exp(0.49550372683)/(exp(0.49550372683) + exp(0.47340300642))
        #        = 0.50552495521
        #    attention_distribution001
        #        = exp(0.47340300642)/(exp(0.49550372683) + exp(0.47340300642))
        #        = 0.49447504478
        #    attention_distribution002 = 0
        #
        # Expected tensor of shape [1, 1, 1].
        # expected000 = 0.50552495521 * 0.5 + 0.49447504478 * 0.8 - 0 * 0.3
        #             = 0.64834251342
        expected = np.array([[[0.64834251342]]], dtype=np.float32)
        self.assertAllClose(expected, actual)

    def test_multi_dim_with_query_mask(self):
        # Query tensor of shape [1, 2, 1]
        q = np.array([[[1.1], [-0.5]]], dtype=np.float32)
        # Value tensor of shape [1, 3, 1]
        v = np.array([[[1.6], [0.7], [-0.8]]], dtype=np.float32)
        # Query mask tensor of shape [1, 2]
        q_mask = np.array([[True, False]], dtype=np.bool_)
        # Value mask tensor of shape [1, 3]
        v_mask = np.array([[True, True, False]], dtype=np.bool_)
        attention_layer = keras.layers.AdditiveAttention()
        attention_layer.build(input_shape=([1, 1, 1], [1, 3, 1]))
        # Scale tensor of shape [1]
        attention_layer.scale = np.array([[[0.5]]], dtype=np.float32)
        actual = attention_layer([q, v], mask=[q_mask, v_mask])

        # Expected scores of shape [1, 2, 3]
        # scores = [[[0.5 * tanh(1.1 + 1.6),
        #             0.5 * tanh(1.1 + 0.7),
        #             0.5 * tanh(1.1 - 0.8)],
        #            [0.5 * tanh(-0.5 + 1.6),
        #             0.5 * tanh(-0.5 + 0.7),
        #             0.5 * tanh(-0.5 - 0.8)]]]
        #        = [[[0.49550372683, 0.47340300642, 0.14565630622],
        #            [0.40024951088, 0.09868766011, -0.43086157965]]]
        # Expected attention distribution = softmax(scores) with zeros in
        # positions where v_mask == False.
        # => attention_distribution000
        #        = exp(0.49550372683)/(exp(0.49550372683) + exp(0.47340300642))
        #        = 0.50552495521
        #    attention_distribution001
        #        = exp(0.47340300642)/(exp(0.49550372683) + exp(0.47340300642))
        #        = 0.49447504478
        #    attention_distribution002 = 0
        # => attention_distribution010
        #        = exp(0.40024951088)/(exp(0.40024951088) + exp(0.09868766011))
        #        = 0.57482427975
        #    attention_distribution011
        #        = exp(0.09868766011)/(exp(0.40024951088) + exp(0.09868766011))
        #        = 0.42517572025
        #    attention_distribution012 = 0
        #
        # Expected tensor of shape [1, 2, 1] with zeros where  q_mask == False.
        # expected000 = 0.50552495521 * 1.6 + 0.49447504478 * 0.7 - 0 * 0.8
        #             = 1.15497245968
        # expected000 = 0
        expected = np.array([[[1.15497245968], [0.0]]], dtype=np.float32)
        self.assertAllClose(expected, actual)

    def test_serialization(self):
        # Test serialization with use_scale
        layer = keras.layers.AdditiveAttention(use_scale=True)

        config = keras.layers.serialize(layer)
        new_layer = keras.layers.deserialize(config)
        self.assertEqual(new_layer.use_scale, True)

        config = layer.get_config()
        new_layer = keras.layers.AdditiveAttention.from_config(config)
        self.assertEqual(new_layer.use_scale, True)

    @test_utils.enable_v2_dtype_behavior
    def test_mixed_float16_policy(self):
        # Test case for GitHub issue:
        # https://github.com/tensorflow/tensorflow/issues/46064
        with policy.policy_scope("mixed_float16"):
            q = tf.cast(tf.random.uniform((2, 3, 4), seed=1), "float16")
            v = tf.cast(tf.random.uniform((2, 3, 4), seed=2), "float16")
            k = tf.cast(tf.random.uniform((2, 3, 4), seed=3), "float16")
            layer = keras.layers.AdditiveAttention()
            _ = layer([q, v, k], use_causal_mask=True)


if __name__ == "__main__":
    tf.test.main()
