# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Unit Normalization layer."""
# pylint: disable=g-bad-import-order
# pylint: disable=g-direct-tensorflow-import
# pylint: disable=g-classes-have-attributes

import tensorflow.compat.v2 as tf

from keras.engine import base_layer
from keras.utils import tf_utils

from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.UnitNormalization', v1=[])
class UnitNormalization(base_layer.Layer):
  """Unit normalization layer.

  Normalize a batch of inputs so that each input in the batch has a L2 norm
  equal to 1 (across the axes specified in `axis`).

  Example:

  >>> data = tf.constant(np.arange(6).reshape(2, 3), dtype=tf.float32)
  >>> normalized_data = tf.keras.layers.UnitNormalization()(data)
  >>> print(tf.reduce_sum(normalized_data[0, :] ** 2).numpy())
  1.0

  Args:
    axis: Integer or list/tuple. The axis or axes to normalize across. Typically
      this is the features axis or axes. The left-out axes are typically the
      batch axis or axes. Defaults to `-1`, the last dimension in
      the input.
  """

  def __init__(self,
               axis=-1,
               **kwargs):
    super().__init__(**kwargs)
    if isinstance(axis, (list, tuple)):
      self.axis = list(axis)
    elif isinstance(axis, int):
      self.axis = axis
    else:
      raise TypeError(
          'Invalid value for `axis` argument: '
          'expected an int or a list/tuple of ints. '
          f'Received: axis={axis}')
    self.supports_masking = True

  def build(self, input_shape):
    self.axis = tf_utils.validate_axis(self.axis, input_shape)

  def call(self, inputs):
    inputs = tf.cast(inputs, self.compute_dtype)
    return tf.linalg.l2_normalize(inputs, axis=self.axis)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = super(UnitNormalization, self).get_config()
    config.update({'axis': self.axis})
    return config
