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
"""Keras upsampling layer for 1D inputs."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

from keras import backend
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
import tensorflow.compat.v2 as tf

from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.UpSampling1D')
class UpSampling1D(Layer):
  """Upsampling layer for 1D inputs.

  Repeats each temporal step `size` times along the time axis.

  Examples:

  >>> input_shape = (2, 2, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> print(x)
  [[[ 0  1  2]
    [ 3  4  5]]
   [[ 6  7  8]
    [ 9 10 11]]]
  >>> y = tf.keras.layers.UpSampling1D(size=2)(x)
  >>> print(y)
  tf.Tensor(
    [[[ 0  1  2]
      [ 0  1  2]
      [ 3  4  5]
      [ 3  4  5]]
     [[ 6  7  8]
      [ 6  7  8]
      [ 9 10 11]
      [ 9 10 11]]], shape=(2, 4, 3), dtype=int64)

  Args:
    size: Integer. Upsampling factor.

  Input shape:
    3D tensor with shape: `(batch_size, steps, features)`.

  Output shape:
    3D tensor with shape: `(batch_size, upsampled_steps, features)`.
  """

  def __init__(self, size=2, **kwargs):
    super(UpSampling1D, self).__init__(**kwargs)
    self.size = int(size)
    self.input_spec = InputSpec(ndim=3)

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    size = self.size * input_shape[1] if input_shape[1] is not None else None
    return tf.TensorShape([input_shape[0], size, input_shape[2]])

  def call(self, inputs):
    output = backend.repeat_elements(inputs, self.size, axis=1)
    return output

  def get_config(self):
    config = {'size': self.size}
    base_config = super(UpSampling1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
