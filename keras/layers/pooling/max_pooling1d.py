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
"""Max pooling 1D layer."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

import functools

from keras import backend
from keras.layers.pooling.base_pooling1d import Pooling1D

from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.MaxPool1D', 'keras.layers.MaxPooling1D')
class MaxPooling1D(Pooling1D):
  """Max pooling operation for 1D temporal data.

  Downsamples the input representation by taking the maximum value over a
  spatial window of size `pool_size`. The window is shifted by `strides`.  The
  resulting output, when using the `"valid"` padding option, has a shape of:
  `output_shape = (input_shape - pool_size + 1) / strides)`

  The resulting output shape when using the `"same"` padding option is:
  `output_shape = input_shape / strides`

  For example, for `strides=1` and `padding="valid"`:

  >>> x = tf.constant([1., 2., 3., 4., 5.])
  >>> x = tf.reshape(x, [1, 5, 1])
  >>> max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=2,
  ...    strides=1, padding='valid')
  >>> max_pool_1d(x)
  <tf.Tensor: shape=(1, 4, 1), dtype=float32, numpy=
  array([[[2.],
          [3.],
          [4.],
          [5.]]], dtype=float32)>

  For example, for `strides=2` and `padding="valid"`:

  >>> x = tf.constant([1., 2., 3., 4., 5.])
  >>> x = tf.reshape(x, [1, 5, 1])
  >>> max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=2,
  ...    strides=2, padding='valid')
  >>> max_pool_1d(x)
  <tf.Tensor: shape=(1, 2, 1), dtype=float32, numpy=
  array([[[2.],
          [4.]]], dtype=float32)>

  For example, for `strides=1` and `padding="same"`:

  >>> x = tf.constant([1., 2., 3., 4., 5.])
  >>> x = tf.reshape(x, [1, 5, 1])
  >>> max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=2,
  ...    strides=1, padding='same')
  >>> max_pool_1d(x)
  <tf.Tensor: shape=(1, 5, 1), dtype=float32, numpy=
  array([[[2.],
          [3.],
          [4.],
          [5.],
          [5.]]], dtype=float32)>

  Args:
    pool_size: Integer, size of the max pooling window.
    strides: Integer, or None. Specifies how much the pooling window moves
      for each pooling step.
      If None, it will default to `pool_size`.
    padding: One of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, steps, features)` while `channels_first`
      corresponds to inputs with shape
      `(batch, features, steps)`.

  Input shape:
    - If `data_format='channels_last'`:
      3D tensor with shape `(batch_size, steps, features)`.
    - If `data_format='channels_first'`:
      3D tensor with shape `(batch_size, features, steps)`.

  Output shape:
    - If `data_format='channels_last'`:
      3D tensor with shape `(batch_size, downsampled_steps, features)`.
    - If `data_format='channels_first'`:
      3D tensor with shape `(batch_size, features, downsampled_steps)`.
  """

  def __init__(self, pool_size=2, strides=None,
               padding='valid', data_format='channels_last', **kwargs):

    super(MaxPooling1D, self).__init__(
        functools.partial(backend.pool2d, pool_mode='max'),
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        **kwargs)


# Alias

MaxPool1D = MaxPooling1D
