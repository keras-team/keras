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
"""Attention layer that can be used in sequence DNN/CNN models.

This file follows the terminology of https://arxiv.org/abs/1706.03762 Figure 2.
Attention is formed by three tensors: Query, Key and Value.
"""


import tensorflow.compat.v2 as tf

from keras.layers.attention.base_dense_attention import BaseDenseAttention

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.Attention")
class Attention(BaseDenseAttention):
    """Dot-product attention layer, a.k.a. Luong-style attention.

    Inputs are `query` tensor of shape `[batch_size, Tq, dim]`, `value` tensor
    of shape `[batch_size, Tv, dim]` and `key` tensor of shape
    `[batch_size, Tv, dim]`. The calculation follows the steps:

    1. Calculate scores with shape `[batch_size, Tq, Tv]` as a `query`-`key` dot
       product: `scores = tf.matmul(query, key, transpose_b=True)`.
    2. Use scores to calculate a distribution with shape
       `[batch_size, Tq, Tv]`: `distribution = tf.nn.softmax(scores)`.
    3. Use `distribution` to create a linear combination of `value` with
       shape `[batch_size, Tq, dim]`:
       `return tf.matmul(distribution, value)`.

    Args:
      use_scale: If `True`, will create a scalar variable to scale the attention
        scores.
      dropout: Float between 0 and 1. Fraction of the units to drop for the
        attention scores. Defaults to 0.0.
      score_mode: Function to use to compute attention scores, one of
        `{"dot", "concat"}`. `"dot"` refers to the dot product between the query
        and key vectors. `"concat"` refers to the hyperbolic tangent of the
        concatenation of the query and key vectors.

    Call Args:

      inputs: List of the following tensors:
        * query: Query `Tensor` of shape `[batch_size, Tq, dim]`.
        * value: Value `Tensor` of shape `[batch_size, Tv, dim]`.
        * key: Optional key `Tensor` of shape `[batch_size, Tv, dim]`. If not
          given, will use `value` for both `key` and `value`, which is the
          most common case.
      mask: List of the following tensors:
        * query_mask: A boolean mask `Tensor` of shape `[batch_size, Tq]`.
          If given, the output will be zero at the positions where
          `mask==False`.
        * value_mask: A boolean mask `Tensor` of shape `[batch_size, Tv]`.
          If given, will apply the mask such that values at positions where
          `mask==False` do not contribute to the result.
      return_attention_scores: bool, it `True`, returns the attention scores
        (after masking and softmax) as an additional output argument.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (no dropout).
      use_causal_mask: Boolean. Set to `True` for decoder self-attention. Adds a
        mask such that position `i` cannot attend to positions `j > i`. This
        prevents the flow of information from the future towards the past.
        Defaults to `False`.

    Output:

      Attention outputs of shape `[batch_size, Tq, dim]`.
      [Optional] Attention scores after masking and softmax with shape
        `[batch_size, Tq, Tv]`.

    The meaning of `query`, `value` and `key` depend on the application. In the
    case of text similarity, for example, `query` is the sequence embeddings of
    the first piece of text and `value` is the sequence embeddings of the second
    piece of text. `key` is usually the same tensor as `value`.

    Here is a code example for using `Attention` in a CNN+Attention network:

    ```python
    # Variable-length int sequences.
    query_input = tf.keras.Input(shape=(None,), dtype='int32')
    value_input = tf.keras.Input(shape=(None,), dtype='int32')

    # Embedding lookup.
    token_embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=64)
    # Query embeddings of shape [batch_size, Tq, dimension].
    query_embeddings = token_embedding(query_input)
    # Value embeddings of shape [batch_size, Tv, dimension].
    value_embeddings = token_embedding(value_input)

    # CNN layer.
    cnn_layer = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        # Use 'same' padding so outputs have the same shape as inputs.
        padding='same')
    # Query encoding of shape [batch_size, Tq, filters].
    query_seq_encoding = cnn_layer(query_embeddings)
    # Value encoding of shape [batch_size, Tv, filters].
    value_seq_encoding = cnn_layer(value_embeddings)

    # Query-value attention of shape [batch_size, Tq, filters].
    query_value_attention_seq = tf.keras.layers.Attention()(
        [query_seq_encoding, value_seq_encoding])

    # Reduce over the sequence axis to produce encodings of shape
    # [batch_size, filters].
    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
        query_seq_encoding)
    query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
        query_value_attention_seq)

    # Concatenate query and document encodings to produce a DNN input layer.
    input_layer = tf.keras.layers.Concatenate()(
        [query_encoding, query_value_attention])

    # Add DNN layers, and create Model.
    # ...
    ```
    """

    def __init__(self, use_scale=False, score_mode="dot", **kwargs):
        super().__init__(**kwargs)
        self.use_scale = use_scale
        self.score_mode = score_mode
        if self.score_mode not in ["dot", "concat"]:
            raise ValueError(
                f"Received: score_mode={score_mode}. Acceptable values "
                'are: ["dot", "concat"]'
            )

    def build(self, input_shape):
        """Creates variable when `use_scale` is True or `score_mode` is
        `concat`."""
        if self.use_scale:
            self.scale = self.add_weight(
                name="scale",
                shape=(),
                initializer="ones",
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.scale = None
        if self.score_mode == "concat":
            self.concat_score_weight = self.add_weight(
                name="concat_score_weight",
                shape=(),
                initializer="ones",
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.concat_score_weight = None
        super().build(input_shape)

    def _calculate_scores(self, query, key):
        """Calculates attention scores as a query-key dot product.

        Args:
          query: Query tensor of shape `[batch_size, Tq, dim]`.
          key: Key tensor of shape `[batch_size, Tv, dim]`.
        Returns:
          Tensor of shape `[batch_size, Tq, Tv]`.
        """
        if self.score_mode == "dot":
            scores = tf.matmul(query, key, transpose_b=True)
            if self.scale is not None:
                scores *= self.scale
        elif self.score_mode == "concat":
            # Reshape tensors to enable broadcasting.
            # Reshape into [batch_size, Tq, 1, dim].
            q_reshaped = tf.expand_dims(query, axis=-2)
            # Reshape into [batch_size, 1, Tv, dim].
            k_reshaped = tf.expand_dims(key, axis=-3)
            if self.scale is not None:
                scores = self.concat_score_weight * tf.reduce_sum(
                    tf.tanh(self.scale * (q_reshaped + k_reshaped)), axis=-1
                )
            else:
                scores = self.concat_score_weight * tf.reduce_sum(
                    tf.tanh(q_reshaped + k_reshaped), axis=-1
                )

        return scores

    def get_config(self):
        config = {"use_scale": self.use_scale, "score_mode": self.score_mode}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
