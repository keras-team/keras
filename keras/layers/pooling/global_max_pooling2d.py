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
"""Global max pooling 2D layer."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

from keras import backend
from keras.layers.pooling.base_global_pooling2d import GlobalPooling2D

from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.GlobalMaxPool2D', 'keras.layers.GlobalMaxPooling2D')
class GlobalMaxPooling2D(GlobalPooling2D):
  """Global max pooling operation for spatial data.

  Examples:

  >>> input_shape = (2, 4, 5, 3)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.GlobalMaxPool2D()(x)
  >>> print(y.shape)
  (2, 3)

  Args:
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
    keepdims: A boolean, whether to keep the spatial dimensions or not.
      If `keepdims` is `False` (default), the rank of the tensor is reduced
      for spatial dimensions.
      If `keepdims` is `True`, the spatial dimensions are retained with
      length 1.
      The behavior is the same as for `tf.reduce_max` or `np.max`.

  Input shape:
    - If `data_format='channels_last'`:
      4D tensor with shape `(batch_size, rows, cols, channels)`.
    - If `data_format='channels_first'`:
      4D tensor with shape `(batch_size, channels, rows, cols)`.

  Output shape:
    - If `keepdims`=False:
      2D tensor with shape `(batch_size, channels)`.
    - If `keepdims`=True:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, 1, 1, channels)`
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, 1, 1)`
  """

  def call(self, inputs):
    if self.data_format == 'channels_last':
      return backend.max(inputs, axis=[1, 2], keepdims=self.keepdims)
    else:
      return backend.max(inputs, axis=[2, 3], keepdims=self.keepdims)


# Alias

GlobalMaxPool2D = GlobalMaxPooling2D
