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
"""Keras upsampling layer for 3D inputs."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

from keras import backend
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.utils import conv_utils
import tensorflow.compat.v2 as tf

from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.UpSampling3D')
class UpSampling3D(Layer):
  """Upsampling layer for 3D inputs.

  Repeats the 1st, 2nd and 3rd dimensions
  of the data by `size[0]`, `size[1]` and `size[2]` respectively.

  Examples:

  >>> input_shape = (2, 1, 2, 1, 3)
  >>> x = tf.constant(1, shape=input_shape)
  >>> y = tf.keras.layers.UpSampling3D(size=2)(x)
  >>> print(y.shape)
  (2, 2, 4, 2, 3)

  Args:
    size: Int, or tuple of 3 integers.
      The upsampling factors for dim1, dim2 and dim3.
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
        `(batch_size, dim1, dim2, dim3, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, dim1, dim2, dim3)`

  Output shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)`
  """

  def __init__(self, size=(2, 2, 2), data_format=None, **kwargs):
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.size = conv_utils.normalize_tuple(size, 3, 'size')
    self.input_spec = InputSpec(ndim=5)
    super(UpSampling3D, self).__init__(**kwargs)

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      dim1 = self.size[0] * input_shape[
          2] if input_shape[2] is not None else None
      dim2 = self.size[1] * input_shape[
          3] if input_shape[3] is not None else None
      dim3 = self.size[2] * input_shape[
          4] if input_shape[4] is not None else None
      return tf.TensorShape(
          [input_shape[0], input_shape[1], dim1, dim2, dim3])
    else:
      dim1 = self.size[0] * input_shape[
          1] if input_shape[1] is not None else None
      dim2 = self.size[1] * input_shape[
          2] if input_shape[2] is not None else None
      dim3 = self.size[2] * input_shape[
          3] if input_shape[3] is not None else None
      return tf.TensorShape(
          [input_shape[0], dim1, dim2, dim3, input_shape[4]])

  def call(self, inputs):
    return backend.resize_volumes(
        inputs, self.size[0], self.size[1], self.size[2], self.data_format)

  def get_config(self):
    config = {'size': self.size, 'data_format': self.data_format}
    base_config = super(UpSampling3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
