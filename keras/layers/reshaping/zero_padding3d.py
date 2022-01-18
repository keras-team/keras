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
"""Keras zero-padding layer for 3D input."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

from keras import backend
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.utils import conv_utils
import tensorflow.compat.v2 as tf

from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.ZeroPadding3D')
class ZeroPadding3D(Layer):
  """Zero-padding layer for 3D data (spatial or spatio-temporal).

  Examples:

  >>> input_shape = (1, 1, 2, 2, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> y = tf.keras.layers.ZeroPadding3D(padding=2)(x)
  >>> print(y.shape)
  (1, 5, 6, 6, 3)

  Args:
    padding: Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
      - If int: the same symmetric padding
        is applied to height and width.
      - If tuple of 3 ints:
        interpreted as two different
        symmetric padding values for height and width:
        `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
      - If tuple of 3 tuples of 2 ints:
        interpreted as
        `((left_dim1_pad, right_dim1_pad), (left_dim2_pad,
          right_dim2_pad), (left_dim3_pad, right_dim3_pad))`
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      while `channels_first` corresponds to inputs with shape
      `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".

  Input shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad,
          depth)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, depth, first_axis_to_pad, second_axis_to_pad,
          third_axis_to_pad)`

  Output shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, first_padded_axis, second_padded_axis, third_axis_to_pad,
          depth)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, depth, first_padded_axis, second_padded_axis,
          third_axis_to_pad)`
  """

  def __init__(self, padding=(1, 1, 1), data_format=None, **kwargs):
    super(ZeroPadding3D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    if isinstance(padding, int):
      self.padding = ((padding, padding), (padding, padding), (padding,
                                                               padding))
    elif hasattr(padding, '__len__'):
      if len(padding) != 3:
        raise ValueError('`padding` should have 3 elements. '
                         f'Received: {padding}.')
      dim1_padding = conv_utils.normalize_tuple(
          padding[0], 2, '1st entry of padding', allow_zero=True)
      dim2_padding = conv_utils.normalize_tuple(
          padding[1], 2, '2nd entry of padding', allow_zero=True)
      dim3_padding = conv_utils.normalize_tuple(
          padding[2], 2, '3rd entry of padding', allow_zero=True)
      self.padding = (dim1_padding, dim2_padding, dim3_padding)
    else:
      raise ValueError(
          '`padding` should be either an int, '
          'a tuple of 3 ints '
          '(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad), '
          'or a tuple of 3 tuples of 2 ints '
          '((left_dim1_pad, right_dim1_pad),'
          ' (left_dim2_pad, right_dim2_pad),'
          ' (left_dim3_pad, right_dim2_pad)). '
          f'Received: {padding}.')
    self.input_spec = InputSpec(ndim=5)

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      if input_shape[2] is not None:
        dim1 = input_shape[2] + self.padding[0][0] + self.padding[0][1]
      else:
        dim1 = None
      if input_shape[3] is not None:
        dim2 = input_shape[3] + self.padding[1][0] + self.padding[1][1]
      else:
        dim2 = None
      if input_shape[4] is not None:
        dim3 = input_shape[4] + self.padding[2][0] + self.padding[2][1]
      else:
        dim3 = None
      return tf.TensorShape(
          [input_shape[0], input_shape[1], dim1, dim2, dim3])
    elif self.data_format == 'channels_last':
      if input_shape[1] is not None:
        dim1 = input_shape[1] + self.padding[0][0] + self.padding[0][1]
      else:
        dim1 = None
      if input_shape[2] is not None:
        dim2 = input_shape[2] + self.padding[1][0] + self.padding[1][1]
      else:
        dim2 = None
      if input_shape[3] is not None:
        dim3 = input_shape[3] + self.padding[2][0] + self.padding[2][1]
      else:
        dim3 = None
      return tf.TensorShape(
          [input_shape[0], dim1, dim2, dim3, input_shape[4]])

  def call(self, inputs):
    return backend.spatial_3d_padding(
        inputs, padding=self.padding, data_format=self.data_format)

  def get_config(self):
    config = {'padding': self.padding, 'data_format': self.data_format}
    base_config = super(ZeroPadding3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
