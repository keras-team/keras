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
"""Additive attention layer that can be used in sequence DNN/CNN models.

This file follows the terminology of https://arxiv.org/abs/1706.03762 Figure 2.
Attention is formed by three tensors: Query, Key and Value.
"""


import tensorflow.compat.v2 as tf

from keras.layers.attention.base_dense_attention import BaseDenseAttention

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.AdditiveAttention")
class AdditiveAttention(BaseDenseAttention):
    """Additive attention layer, a.k.a. Bahdanau-style attention.

    Inputs are `query` tensor of shape `[batch_size, Tq, dim]`, `value` tensor
    of shape `[batch_size, Tv, dim]` and `key` tensor of shape
    `[batch_size, Tv, dim]`. The calculation follows the steps:

    1. Reshape `query` and `key` into shapes `[batch_size, Tq, 1, dim]`
       and `[batch_size, 1, Tv, dim]` respectively.
    2. Calculate scores with shape `[batch_size, Tq, Tv]` as a non-linear
       sum: `scores = tf.reduce_sum(tf.tanh(query + key), axis=-1)`
    3. Use scores to calculate a distribution with shape
       `[batch_size, Tq, Tv]`: `distribution = tf.nn.softmax(scores)`.
    4. Use `distribution` to create a linear combination of `value` with
       shape `[batch_size, Tq, dim]`:
       `return tf.matmul(distribution, value)`.

    Args:
      use_scale: If `True`, will create a variable to scale the attention
        scores.
      dropout: Float between 0 and 1. Fraction of the units to drop for the
        attention scores. Defaults to 0.0.

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
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (no dropout).
      return_attention_scores: bool, it `True`, returns the attention scores
        (after masking and softmax) as an additional output argument.
      use_causal_mask: Boolean. Set to `True` for decoder self-attention. Adds a
        mask such that position `i` cannot attend to positions `j > i`. This
        prevents the flow of information from the future towards the past.
        Defaults to `False`.`

    Output:

      Attention outputs of shape `[batch_size, Tq, dim]`.
      [Optional] Attention scores after masking and softmax with shape
        `[batch_size, Tq, Tv]`.

    The meaning of `query`, `value` and `key` depend on the application. In the
    case of text similarity, for example, `query` is the sequence embeddings of
    the first piece of text and `value` is the sequence embeddings of the second
    piece of text. `key` is usually the same tensor as `value`.

    Here is a code example for using `AdditiveAttention` in a CNN+Attention
    network:

    ```python
    # Variable-length int sequences.
    query_input = tf.keras.Input(shape=(None,), dtype='int32')
    value_input = tf.keras.Input(shape=(None,), dtype='int32')

    # Embedding lookup.
    token_embedding = tf.keras.layers.Embedding(max_tokens, dimension)
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
    query_value_attention_seq = tf.keras.layers.AdditiveAttention()(
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

    def __init__(self, use_scale=True, **kwargs):
        super().__init__(**kwargs)
        self.use_scale = use_scale

    def build(self, input_shape):
        v_shape = tf.TensorShape(input_shape[1])
        dim = v_shape[-1]
        dim = tf.compat.dimension_value(dim)
        if self.use_scale:
            self.scale = self.add_weight(
                name="scale",
                shape=[dim],
                initializer="glorot_uniform",
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.scale = None
        super().build(input_shape)

    def _calculate_scores(self, query, key):
        """Calculates attention scores as a nonlinear sum of query and key.

        Args:
          query: Query tensor of shape `[batch_size, Tq, dim]`.
          key: Key tensor of shape `[batch_size, Tv, dim]`.
        Returns:
          Tensor of shape `[batch_size, Tq, Tv]`.
        """
        # Reshape tensors to enable broadcasting.
        # Reshape into [batch_size, Tq, 1, dim].
        q_reshaped = tf.expand_dims(query, axis=-2)
        # Reshape into [batch_size, 1, Tv, dim].
        k_reshaped = tf.expand_dims(key, axis=-3)
        if self.use_scale:
            scale = self.scale
        else:
            scale = 1.0
        return tf.reduce_sum(scale * tf.tanh(q_reshaped + k_reshaped), axis=-1)

    def get_config(self):
        config = {"use_scale": self.use_scale}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
