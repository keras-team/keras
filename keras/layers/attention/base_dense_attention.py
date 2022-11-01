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
"""Base class for attention layers that can be used in sequence DNN/CNN models.

This file follows the terminology of https://arxiv.org/abs/1706.03762 Figure 2.
Attention is formed by three tensors: Query, Key and Value.
"""

import tensorflow.compat.v2 as tf
from absl import logging

from keras import backend
from keras.engine import base_layer
from keras.utils import control_flow_util


class BaseDenseAttention(base_layer.BaseRandomLayer):
    """Base Attention class for Dense networks.

    This class is suitable for Dense or CNN networks, and not for RNN networks.

    Implementations of attention mechanisms should inherit from this class, and
    reuse the `apply_attention_scores()` method.

    Args:
      dropout: Float between 0 and 1. Fraction of the units to drop for the
        attention scores.

    Call Args:
      inputs: List of the following tensors:
        * query: Query `Tensor` of shape `[batch_size, Tq, dim]`.
        * value: Value `Tensor` of shape `[batch_size, Tv, dim]`.
        * key: Optional key `Tensor` of shape `[batch_size, Tv, dim]`. If not
          given, will use `value` for both `key` and `value`, which is the most
          common case.
      mask: List of the following tensors:
        * query_mask: A boolean mask `Tensor` of shape `[batch_size, Tq]`. If
          given, the output will be zero at the positions where `mask==False`.
        * value_mask: A boolean mask `Tensor` of shape `[batch_size, Tv]`. If
          given, will apply the mask such that values at positions where
          `mask==False` do not contribute to the result.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (no dropout).
      return_attention_scores: bool, if `True`, returns the attention scores
        (after masking and softmax) as an additional output argument.

    Output:

      Attention outputs of shape `[batch_size, Tq, dim]`.
      [Optional] Attention scores after masking and softmax with shape
        `[batch_size, Tq, Tv]`.
    """

    def __init__(self, dropout=0.0, **kwargs):
        # Deprecated field `causal` determines whether to using causal masking.
        # Use `use_causal_mask` in call() method instead.
        if "causal" in kwargs:
            logging.warning(
                "`causal` argument is deprecated. Please use `use_causal_mask` "
                "in call() method to specify causal masking."
            )
        self.causal = kwargs.pop("causal", False)
        super().__init__(**kwargs)
        self.dropout = dropout
        self.supports_masking = True

    def build(self, input_shape):
        # Skip RNG initialization if dropout rate is 0. This will let the layer
        # be purely stateless, with no reference to any variable.
        if self.dropout > 0:
            super().build(input_shape)
        self.built = True

    def _calculate_scores(self, query, key):
        """Calculates attention scores.

        Args:
          query: Query tensor of shape `[batch_size, Tq, dim]`.
          key: Key tensor of shape `[batch_size, Tv, dim]`.

        Returns:
          Tensor of shape `[batch_size, Tq, Tv]`.
        """
        return NotImplementedError

    def _apply_scores(self, scores, value, scores_mask=None, training=None):
        """Applies attention scores to the given value tensor.

        To use this method in your attention layer, follow the steps:

        * Use `query` tensor of shape `[batch_size, Tq]` and `key` tensor of
          shape `[batch_size, Tv]` to calculate the attention `scores`.
        * Pass `scores` and `value` tensors to this method. The method applies
          `scores_mask`, calculates `attention_distribution = softmax(scores)`,
          then returns `matmul(attention_distribution, value).
        * Apply `query_mask` and return the result.

        Args:
          scores: Scores float tensor of shape `[batch_size, Tq, Tv]`.
          value: Value tensor of shape `[batch_size, Tv, dim]`.
          scores_mask: A boolean mask `Tensor` of shape `[batch_size, 1, Tv]` or
            `[batch_size, Tq, Tv]`. If given, scores at positions where
            `scores_mask==False` do not contribute to the result. It must
            contain at least one `True` value in each line along the last
            dimension.
          training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (no dropout).

        Returns:
          Tensor of shape `[batch_size, Tq, dim]`.
          Attention scores after masking and softmax with shape
            `[batch_size, Tq, Tv]`.
        """
        if scores_mask is not None:
            padding_mask = tf.logical_not(scores_mask)
            # Bias so padding positions do not contribute to attention
            # distribution.  Note 65504. is the max float16 value.
            if scores.dtype is tf.float16:
                scores -= 65504.0 * tf.cast(padding_mask, dtype=scores.dtype)
            else:
                scores -= 1.0e9 * tf.cast(padding_mask, dtype=scores.dtype)
        if training is None:
            training = backend.learning_phase()
        weights = tf.nn.softmax(scores)

        if self.dropout > 0:

            def dropped_weights():
                return self._random_generator.dropout(
                    weights, rate=self.dropout
                )

            weights = control_flow_util.smart_cond(
                training, dropped_weights, lambda: tf.identity(weights)
            )
        return tf.matmul(weights, value), weights

    # TODO(b/125916026): Consider exposing a __call__ method with named args.
    def call(
        self,
        inputs,
        mask=None,
        training=None,
        return_attention_scores=False,
        use_causal_mask=False,
    ):
        self._validate_call_args(inputs=inputs, mask=mask)
        q = inputs[0]
        v = inputs[1]
        k = inputs[2] if len(inputs) > 2 else v
        q_mask = mask[0] if mask else None
        v_mask = mask[1] if mask else None
        scores = self._calculate_scores(query=q, key=k)
        if v_mask is not None:
            # Mask of shape [batch_size, 1, Tv].
            v_mask = tf.expand_dims(v_mask, axis=-2)
        if self.causal or use_causal_mask:
            # Creates a lower triangular mask, so position i cannot attend to
            # positions j>i. This prevents the flow of information from the
            # future into the past.
            scores_shape = tf.shape(scores)
            # causal_mask_shape = [1, Tq, Tv].
            causal_mask_shape = tf.concat(
                [tf.ones_like(scores_shape[:-2]), scores_shape[-2:]], axis=0
            )
            causal_mask = _lower_triangular_mask(causal_mask_shape)
        else:
            causal_mask = None
        scores_mask = _merge_masks(v_mask, causal_mask)
        result, attention_scores = self._apply_scores(
            scores=scores, value=v, scores_mask=scores_mask, training=training
        )
        if q_mask is not None:
            # Mask of shape [batch_size, Tq, 1].
            q_mask = tf.expand_dims(q_mask, axis=-1)
            result *= tf.cast(q_mask, dtype=result.dtype)
        if return_attention_scores:
            return result, attention_scores
        return result

    def compute_mask(self, inputs, mask=None):
        self._validate_call_args(inputs=inputs, mask=mask)
        if mask:
            q_mask = mask[0]
            if q_mask is None:
                return None
            return tf.convert_to_tensor(q_mask)
        return None

    def compute_output_shape(self, input_shape):
        # return_attention_scores argument of BaseDenseAttention.call method
        # is ignored. Output shape of attention_scores cannot be returned.
        return tf.TensorShape(input_shape[0])

    def _validate_call_args(self, inputs, mask):
        """Validates arguments of the call method."""
        class_name = self.__class__.__name__
        if not isinstance(inputs, list):
            raise ValueError(
                f"{class_name} layer must be called on a list of inputs, "
                "namely [query, value] or [query, value, key]. "
                f"Received: {inputs}."
            )
        if len(inputs) < 2 or len(inputs) > 3:
            raise ValueError(
                f"{class_name} layer accepts inputs list of length 2 or 3, "
                "namely [query, value] or [query, value, key]. "
                f"Received length: {len(inputs)}."
            )
        if mask:
            if not isinstance(mask, list):
                raise ValueError(
                    f"{class_name} layer mask must be a list, "
                    f"namely [query_mask, value_mask]. Received: {mask}."
                )
            if len(mask) < 2 or len(mask) > len(inputs):
                raise ValueError(
                    f"{class_name} layer mask must be a list of length 2, "
                    "namely [query_mask, value_mask]. "
                    f"Received length: {len(mask)}."
                )

    def get_config(self):
        config = {
            "dropout": self.dropout,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _lower_triangular_mask(shape):
    """Creates a lower-triangular boolean mask over the last 2 dimensions."""
    row_index = tf.cumsum(tf.ones(shape=shape, dtype=tf.int32), axis=-2)
    col_index = tf.cumsum(tf.ones(shape=shape, dtype=tf.int32), axis=-1)
    return tf.greater_equal(row_index, col_index)


def _merge_masks(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return tf.logical_and(x, y)
