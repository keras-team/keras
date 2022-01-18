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
"""Keras zero-padding layer for 2D input."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

from keras import backend
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.utils import conv_utils
import tensorflow.compat.v2 as tf

from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.ZeroPadding2D')
class ZeroPadding2D(Layer):
  """Zero-padding layer for 2D input (e.g. picture).

  This layer can add rows and columns of zeros
  at the top, bottom, left and right side of an image tensor.

  Examples:

  >>> input_shape = (1, 1, 2, 2)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> print(x)
  [[[[0 1]
     [2 3]]]]
  >>> y = tf.keras.layers.ZeroPadding2D(padding=1)(x)
  >>> print(y)
  tf.Tensor(
    [[[[0 0]
       [0 0]
       [0 0]
       [0 0]]
      [[0 0]
       [0 1]
       [2 3]
       [0 0]]
      [[0 0]
       [0 0]
       [0 0]
       [0 0]]]], shape=(1, 3, 4, 2), dtype=int64)

  Args:
    padding: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
      - If int: the same symmetric padding
        is applied to height and width.
      - If tuple of 2 ints:
        interpreted as two different
        symmetric padding values for height and width:
        `(symmetric_height_pad, symmetric_width_pad)`.
      - If tuple of 2 tuples of 2 ints:
        interpreted as
        `((top_pad, bottom_pad), (left_pad, right_pad))`
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".

  Input shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, rows, cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, rows, cols)`

  Output shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, padded_rows, padded_cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, padded_rows, padded_cols)`
  """

  def __init__(self, padding=(1, 1), data_format=None, **kwargs):
    super(ZeroPadding2D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    if isinstance(padding, int):
      self.padding = ((padding, padding), (padding, padding))
    elif hasattr(padding, '__len__'):
      if len(padding) != 2:
        raise ValueError('`padding` should have two elements. '
                         f'Received: {padding}.')
      height_padding = conv_utils.normalize_tuple(
          padding[0], 2, '1st entry of padding', allow_zero=True)
      width_padding = conv_utils.normalize_tuple(
          padding[1], 2, '2nd entry of padding', allow_zero=True)
      self.padding = (height_padding, width_padding)
    else:
      raise ValueError('`padding` should be either an int, '
                       'a tuple of 2 ints '
                       '(symmetric_height_pad, symmetric_width_pad), '
                       'or a tuple of 2 tuples of 2 ints '
                       '((top_pad, bottom_pad), (left_pad, right_pad)). '
                       f'Received: {padding}.')
    self.input_spec = InputSpec(ndim=4)

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      if input_shape[2] is not None:
        rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
      else:
        rows = None
      if input_shape[3] is not None:
        cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
      else:
        cols = None
      return tf.TensorShape(
          [input_shape[0], input_shape[1], rows, cols])
    elif self.data_format == 'channels_last':
      if input_shape[1] is not None:
        rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
      else:
        rows = None
      if input_shape[2] is not None:
        cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
      else:
        cols = None
      return tf.TensorShape(
          [input_shape[0], rows, cols, input_shape[3]])

  def call(self, inputs):
    return backend.spatial_2d_padding(
        inputs, padding=self.padding, data_format=self.data_format)

  def get_config(self):
    config = {'padding': self.padding, 'data_format': self.data_format}
    base_config = super(ZeroPadding2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
