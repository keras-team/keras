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
"""Private base class for pooling 3D layers."""
# pylint: disable=g-classes-have-attributes

from keras import backend
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.utils import conv_utils
import tensorflow.compat.v2 as tf


class Pooling3D(Layer):
  """Pooling layer for arbitrary pooling functions, for 3D inputs.

  This class only exists for code reuse. It will never be an exposed API.

  Args:
    pool_function: The pooling function to apply, e.g. `tf.nn.max_pool2d`.
    pool_size: An integer or tuple/list of 3 integers:
      (pool_depth, pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, depth, height, width, channels)`
      while `channels_first` corresponds to
      inputs with shape `(batch, channels, depth, height, width)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_function, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, **kwargs):
    super(Pooling3D, self).__init__(name=name, **kwargs)
    if data_format is None:
      data_format = backend.image_data_format()
    if strides is None:
      strides = pool_size
    self.pool_function = pool_function
    self.pool_size = conv_utils.normalize_tuple(pool_size, 3, 'pool_size')
    self.strides = conv_utils.normalize_tuple(
        strides, 3, 'strides', allow_zero=True)
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=5)

  def call(self, inputs):
    pool_shape = (1,) + self.pool_size + (1,)
    strides = (1,) + self.strides + (1,)

    if self.data_format == 'channels_first':
      # TF does not support `channels_first` with 3D pooling operations,
      # so we must handle this case manually.
      # TODO(fchollet): remove this when TF pooling is feature-complete.
      inputs = tf.transpose(inputs, (0, 2, 3, 4, 1))

    outputs = self.pool_function(
        inputs,
        ksize=pool_shape,
        strides=strides,
        padding=self.padding.upper())

    if self.data_format == 'channels_first':
      outputs = tf.transpose(outputs, (0, 4, 1, 2, 3))
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      len_dim1 = input_shape[2]
      len_dim2 = input_shape[3]
      len_dim3 = input_shape[4]
    else:
      len_dim1 = input_shape[1]
      len_dim2 = input_shape[2]
      len_dim3 = input_shape[3]
    len_dim1 = conv_utils.conv_output_length(len_dim1, self.pool_size[0],
                                             self.padding, self.strides[0])
    len_dim2 = conv_utils.conv_output_length(len_dim2, self.pool_size[1],
                                             self.padding, self.strides[1])
    len_dim3 = conv_utils.conv_output_length(len_dim3, self.pool_size[2],
                                             self.padding, self.strides[2])
    if self.data_format == 'channels_first':
      return tf.TensorShape(
          [input_shape[0], input_shape[1], len_dim1, len_dim2, len_dim3])
    else:
      return tf.TensorShape(
          [input_shape[0], len_dim1, len_dim2, len_dim3, input_shape[4]])

  def get_config(self):
    config = {
        'pool_size': self.pool_size,
        'padding': self.padding,
        'strides': self.strides,
        'data_format': self.data_format
    }
    base_config = super(Pooling3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
