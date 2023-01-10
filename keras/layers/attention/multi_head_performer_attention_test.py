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
"""Tests for the MultiHeadPerformerAttention layer."""

import math
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils

from keras.layers.attention import multi_head_performer_attention as mhpa

# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@test_combinations.run_all_keras_modes
class MultiHeadPerformerAttentionTest(test_combinations.TestCase):

  def test_non_causal_performer_relu_self_attention(self):
    """Test bi-directional Performer-ReLU."""
    test_layer = mhpa.MultiHeadPerformerAttention(
        num_heads=12, key_dim=64, performer_type="relu"
    )
    # Create a 3-dimensional input (the first dimension is implicit).
    query = keras.Input(shape=(40, 80))
    output, _ = test_layer(query, query, training=True)
    self.assertEqual(output.shape.as_list(), [None, 40, 80])

  def test_non_causal_performer_softmax_self_attention(self):
    """Test bi-directional Performer-approx-softmax."""
    test_layer = mhpa.MultiHeadPerformerAttention(
        num_heads=12, key_dim=64, performer_type="softmax"
    )
    # Create a 3-dimensional input (the first dimension is implicit).
    query = keras.Input(shape=(40, 80))
    output, _ = test_layer(query, query, training=True, num_rand_features=128)
    self.assertEqual(output.shape.as_list(), [None, 40, 80])

  def test_causal_performer_relu_self_attention_training_mode(self):
    """Test uni-directional Performer-ReLU: training mode."""
    test_layer = mhpa.MultiHeadPerformerAttention(
        num_heads=12, key_dim=64, performer_type="relu"
    )
    # Create a 3-dimensional input (the first dimension is implicit).
    query = keras.Input(shape=(40, 80))
    output, _ = test_layer(query, query, training=True, use_causal_mask=True)
    self.assertEqual(output.shape.as_list(), [None, 40, 80])

  def test_causal_performer_softmax_self_attention_training_mode(self):
    """Test uni-directional Performer-approx-softmax: training mode."""
    test_layer = mhpa.MultiHeadPerformerAttention(
        num_heads=12, key_dim=64, performer_type="softmax"
    )
    # Create a 3-dimensional input (the first dimension is implicit).
    query = keras.Input(shape=(40, 80))
    output, _ = test_layer(
        query, query, training=True, use_causal_mask=True, num_rand_features=128
    )
    self.assertEqual(output.shape.as_list(), [None, 40, 80])

  def test_causal_performer_relu_self_attention_inference_mode(self):
    """Test uni-directional Performer-ReLU: inference mode."""
    batch_dim = 1
    num_heads = 1
    key_dim = 8
    dim = 8
    length = 3
    ################ PERFORMERS' SUPERSTATE INITIALIZATION #####################
    cache_num = tf.zeros((batch_dim, num_heads, key_dim, key_dim))
    cache_den = tf.zeros((batch_dim, num_heads, key_dim))
    cache = {"num": cache_num, "den": cache_den}
    ############################################################################
    test_layer = mhpa.MultiHeadPerformerAttention(
        num_heads=num_heads, key_dim=key_dim, performer_type="relu"
    )
    # Create a 3-dimensional input.
    inputs_q = tf.random.normal((batch_dim, length, dim))
    inputs_v = tf.random.normal((batch_dim, length, dim))
    inputs_k = tf.random.normal((batch_dim, length, dim))

    first_query = tf.expand_dims(inputs_q[:, 0, :], axis=-2)
    second_query = tf.expand_dims(inputs_q[:, 1, :], axis=-2)
    third_query = tf.expand_dims(inputs_q[:, 2, :], axis=-2)

    first_value = tf.expand_dims(inputs_v[:, 0, :], axis=-2)
    second_value = tf.expand_dims(inputs_v[:, 1, :], axis=-2)
    third_value = tf.expand_dims(inputs_v[:, 2, :], axis=-2)

    first_key = tf.expand_dims(inputs_k[:, 0, :], axis=-2)
    second_key = tf.expand_dims(inputs_k[:, 1, :], axis=-2)
    third_key = tf.expand_dims(inputs_k[:, 2, :], axis=-2)

    # RUNNING CAUSAL PERFORMER IN THE TRAINING MODE
    groundtruth_output = test_layer(
        inputs_q, inputs_v, key=inputs_k, training=True, use_causal_mask=True
    )
    # RUNNING CAUSAL PERFORMER IN THE INFERENCE MODE
    test_output_first, cache = test_layer(
        first_query,
        first_value,
        key=first_key,
        training=False,
        use_causal_mask=True,
        cache=cache,
    )
    test_output_second, cache = test_layer(
        second_query,
        second_value,
        key=second_key,
        training=False,
        use_causal_mask=True,
        cache=cache,
    )
    test_output_third, cache = test_layer(
        third_query,
        third_value,
        key=third_key,
        training=False,
        use_causal_mask=True,
        cache=cache,
    )

    print("Groundtruth output:")
    print(groundtruth_output)
    print("Inference output:")
    print([test_output_first, test_output_second, test_output_third])

  def test_softmax_noncausal_attention_block_output(self):
    """Test FAVOR-approx-softmax attention block."""
    batch_size = 1
    length = 2
    num_heads = 1
    dim = 8
    num_random_features = 1000
    query = tf.random.normal([batch_size, length, num_heads, dim])
    key = tf.random.normal([batch_size, length, num_heads, dim])
    value = tf.random.normal([batch_size, length, num_heads, dim])
    kernel_transformation = mhpa.softmax_kernel_transformation
    projection_matrix = mhpa.create_projection_matrix(num_random_features, dim)
    query = tf.cast(query, tf.float64)
    key = tf.cast(key, tf.float64)
    value = tf.cast(value, tf.float64)
    projection_matrix = tf.cast(projection_matrix, tf.float64)
    attention_block_output = mhpa.favor_attention(
        query, key, value, None, kernel_transformation, False, projection_matrix
    )

    query = tf.multiply(query, 1.0 / math.sqrt(float(dim)))
    attention_scores = tf.einsum("BXHD,BYHD->BXYH", query, key)
    attention_scores = tf.nn.softmax(attention_scores, axis=2)
    exact_attention_block_output = tf.einsum(
        "BXYH,BYHD->BXHD", attention_scores, value
    )
    max_error = 0.5

    favor_output, groundtruth_output = [
        exact_attention_block_output,
        attention_block_output,
    ]
    error = np.max(
        np.abs((groundtruth_output - favor_output) / groundtruth_output)
    )
    self.assertLess(error, max_error)

  def test_cossim_noncausal_attention_block_output(self):
    """Test cosine-similarity attention block."""
    batch_size = 1
    length = 2
    num_heads = 1
    dim = 8
    num_random_features = 1000
    query = tf.random.normal([batch_size, length, num_heads, dim])
    key = tf.random.normal([batch_size, length, num_heads, dim])
    value = tf.random.normal([batch_size, length, num_heads, dim])
    projection_matrix = mhpa.create_projection_matrix(num_random_features, dim)
    query = tf.cast(query, tf.float64)
    key = tf.cast(key, tf.float64)
    value = tf.cast(value, tf.float64)

    key_prime = mhpa.cossim_kernel_transformation(
        key, False, projection_matrix, 0.0, num_random_features
    )
    query_prime = mhpa.cossim_kernel_transformation(
        query, True, projection_matrix, 0.0, num_random_features
    )
    attention_scores = tf.einsum("BXHD,BYHD->BXYH", query_prime, key_prime)
    attention_scores = tf.nn.softmax(attention_scores, axis=2)
    attention_block_output = tf.einsum(
        "BXYH,BYHD->BXHD", attention_scores, value
    )

    query = tf.math.l2_normalize(query, axis=[-1])
    key = tf.math.l2_normalize(key, axis=[-1])
    query = tf.multiply(query, math.sqrt(float(dim)))
    attention_scores = tf.einsum("BXHD,BYHD->BXYH", query, key)
    attention_scores = tf.nn.softmax(attention_scores, axis=2)
    exact_attention_block_output = tf.einsum(
        "BXYH,BYHD->BXHD", attention_scores, value
    )
    max_error = 0.5
    favor_output, groundtruth_output = [
        exact_attention_block_output,
        attention_block_output,
    ]
    error = np.max(
        np.abs((groundtruth_output - favor_output) / groundtruth_output)
    )
    self.assertLess(error, max_error)

  def test_chunked_causal_attention(self):
    """Test chunked-causal FAVOR attention block."""
    batch_size = 1
    length = 128
    num_heads = 1
    dim = 16
    query = tf.random.normal([batch_size, length, num_heads, dim])
    key = tf.random.normal([batch_size, length, num_heads, dim])
    value = tf.random.normal([batch_size, length, num_heads, dim])
    query = tf.cast(query, tf.float64)
    key = tf.cast(key, tf.float64)
    value = tf.cast(value, tf.float64)
    attention_block_output = mhpa.favor_attention(
        query, key, value, None, mhpa.relu_kernel_transformation, True
    )

    chunked_attention_block_output = mhpa.favor_attention(
        query,
        key,
        value,
        None,
        mhpa.relu_kernel_transformation,
        True,
        use_chunked_causal=True,
    )

    max_error = 0.0001

    chunked_output, groundtruth_output = [
        chunked_attention_block_output,
        attention_block_output,
    ]
    error = np.max(np.abs(groundtruth_output - chunked_output))
    self.assertLess(error, max_error)


if __name__ == "__main__":
  tf.test.main()
