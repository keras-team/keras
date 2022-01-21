# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Global average pooling 1D layer."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

from keras import backend
from keras.layers.pooling.base_global_pooling1d import GlobalPooling1D
import tensorflow.compat.v2 as tf

from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.GlobalAveragePooling1D',
              'keras.layers.GlobalAvgPool1D')
class GlobalAveragePooling1D(GlobalPooling1D):
  """Global average pooling operation for temporal data.

  Examples:

  >>> input_shape = (2, 3, 4)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.GlobalAveragePooling1D()(x)
  >>> print(y.shape)
  (2, 4)

  Args:
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, steps, features)` while `channels_first`
      corresponds to inputs with shape
      `(batch, features, steps)`.
    keepdims: A boolean, whether to keep the temporal dimension or not.
      If `keepdims` is `False` (default), the rank of the tensor is reduced
      for spatial dimensions.
      If `keepdims` is `True`, the temporal dimension are retained with
      length 1.
      The behavior is the same as for `tf.reduce_mean` or `np.mean`.

  Call arguments:
    inputs: A 3D tensor.
    mask: Binary tensor of shape `(batch_size, steps)` indicating whether
      a given step should be masked (excluded from the average).

  Input shape:
    - If `data_format='channels_last'`:
      3D tensor with shape:
      `(batch_size, steps, features)`
    - If `data_format='channels_first'`:
      3D tensor with shape:
      `(batch_size, features, steps)`

  Output shape:
    - If `keepdims`=False:
      2D tensor with shape `(batch_size, features)`.
    - If `keepdims`=True:
      - If `data_format='channels_last'`:
        3D tensor with shape `(batch_size, 1, features)`
      - If `data_format='channels_first'`:
        3D tensor with shape `(batch_size, features, 1)`
  """

  def __init__(self, data_format='channels_last', **kwargs):
    super(GlobalAveragePooling1D, self).__init__(data_format=data_format,
                                                 **kwargs)
    self.supports_masking = True

  def call(self, inputs, mask=None):
    steps_axis = 1 if self.data_format == 'channels_last' else 2
    if mask is not None:
      mask = tf.cast(mask, inputs[0].dtype)
      mask = tf.expand_dims(
          mask, 2 if self.data_format == 'channels_last' else 1)
      inputs *= mask
      return backend.sum(
          inputs, axis=steps_axis,
          keepdims=self.keepdims) / tf.reduce_sum(
              mask, axis=steps_axis, keepdims=self.keepdims)
    else:
      return backend.mean(inputs, axis=steps_axis, keepdims=self.keepdims)

  def compute_mask(self, inputs, mask=None):
    return None


# Alias

GlobalAvgPool1D = GlobalAveragePooling1D
