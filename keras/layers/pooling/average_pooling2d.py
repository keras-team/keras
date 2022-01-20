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
"""Average pooling 2D layer."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

from keras.layers.pooling.base_pooling2d import Pooling2D
import tensorflow.compat.v2 as tf

from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.AveragePooling2D', 'keras.layers.AvgPool2D')
class AveragePooling2D(Pooling2D):
  """Average pooling operation for spatial data.

  Downsamples the input along its spatial dimensions (height and width)
  by taking the average value over an input window
  (of size defined by `pool_size`) for each channel of the input.
  The window is shifted by `strides` along each dimension.

  The resulting output when using `"valid"` padding option has a shape
  (number of rows or columns) of:
  `output_shape = math.floor((input_shape - pool_size) / strides) + 1`
  (when `input_shape >= pool_size`)

  The resulting output shape when using the `"same"` padding option is:
  `output_shape = math.floor((input_shape - 1) / strides) + 1`

  For example, for `strides=(1, 1)` and `padding="valid"`:

  >>> x = tf.constant([[1., 2., 3.],
  ...                  [4., 5., 6.],
  ...                  [7., 8., 9.]])
  >>> x = tf.reshape(x, [1, 3, 3, 1])
  >>> avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
  ...    strides=(1, 1), padding='valid')
  >>> avg_pool_2d(x)
  <tf.Tensor: shape=(1, 2, 2, 1), dtype=float32, numpy=
    array([[[[3.],
             [4.]],
            [[6.],
             [7.]]]], dtype=float32)>

  For example, for `stride=(2, 2)` and `padding="valid"`:

  >>> x = tf.constant([[1., 2., 3., 4.],
  ...                  [5., 6., 7., 8.],
  ...                  [9., 10., 11., 12.]])
  >>> x = tf.reshape(x, [1, 3, 4, 1])
  >>> avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
  ...    strides=(2, 2), padding='valid')
  >>> avg_pool_2d(x)
  <tf.Tensor: shape=(1, 1, 2, 1), dtype=float32, numpy=
    array([[[[3.5],
             [5.5]]]], dtype=float32)>

  For example, for `strides=(1, 1)` and `padding="same"`:

  >>> x = tf.constant([[1., 2., 3.],
  ...                  [4., 5., 6.],
  ...                  [7., 8., 9.]])
  >>> x = tf.reshape(x, [1, 3, 3, 1])
  >>> avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
  ...    strides=(1, 1), padding='same')
  >>> avg_pool_2d(x)
  <tf.Tensor: shape=(1, 3, 3, 1), dtype=float32, numpy=
    array([[[[3.],
             [4.],
             [4.5]],
            [[6.],
             [7.],
             [7.5]],
            [[7.5],
             [8.5],
             [9.]]]], dtype=float32)>

  Args:
    pool_size: integer or tuple of 2 integers,
      factors by which to downscale (vertical, horizontal).
      `(2, 2)` will halve the input in both spatial dimension.
      If only one integer is specified, the same window length
      will be used for both dimensions.
    strides: Integer, tuple of 2 integers, or None.
      Strides values.
      If None, it will default to `pool_size`.
    padding: One of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".

  Input shape:
    - If `data_format='channels_last'`:
      4D tensor with shape `(batch_size, rows, cols, channels)`.
    - If `data_format='channels_first'`:
      4D tensor with shape `(batch_size, channels, rows, cols)`.

  Output shape:
    - If `data_format='channels_last'`:
      4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
    - If `data_format='channels_first'`:
      4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.
  """

  def __init__(self,
               pool_size=(2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               **kwargs):
    super(AveragePooling2D, self).__init__(
        tf.nn.avg_pool,
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, **kwargs)


# Alias

AvgPool2D = AveragePooling2D
