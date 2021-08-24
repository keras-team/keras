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
"""Contains the Masking layer."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

from keras.engine.base_layer import Layer
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.Masking')
class Masking(Layer):
  """Masks a sequence by using a mask value to skip timesteps.

  For each timestep in the input tensor (dimension #1 in the tensor),
  if all values in the input tensor at that timestep
  are equal to `mask_value`, then the timestep will be masked (skipped)
  in all downstream layers (as long as they support masking).

  If any downstream layer does not support masking yet receives such
  an input mask, an exception will be raised.

  Example:

  Consider a Numpy data array `x` of shape `(samples, timesteps, features)`,
  to be fed to an LSTM layer. You want to mask timestep #3 and #5 because you
  lack data for these timesteps. You can:

  - Set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
  - Insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

  ```python
  samples, timesteps, features = 32, 10, 8
  inputs = np.random.random([samples, timesteps, features]).astype(np.float32)
  inputs[:, 3, :] = 0.
  inputs[:, 5, :] = 0.

  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Masking(mask_value=0.,
                                    input_shape=(timesteps, features)))
  model.add(tf.keras.layers.LSTM(32))

  output = model(inputs)
  # The time step 3 and 5 will be skipped from LSTM calculation.
  ```

  See [the masking and padding guide](
    https://www.tensorflow.org/guide/keras/masking_and_padding)
  for more details.
  """

  def __init__(self, mask_value=0., **kwargs):
    super(Masking, self).__init__(**kwargs)
    self.supports_masking = True
    self.mask_value = mask_value
    self._compute_output_and_mask_jointly = True

  def compute_mask(self, inputs, mask=None):
    return tf.reduce_any(tf.not_equal(inputs, self.mask_value), axis=-1)

  def call(self, inputs):
    boolean_mask = tf.reduce_any(
        tf.not_equal(inputs, self.mask_value), axis=-1, keepdims=True)
    outputs = inputs * tf.cast(boolean_mask, inputs.dtype)
    # Compute the mask and outputs simultaneously.
    outputs._keras_mask = tf.squeeze(boolean_mask, axis=-1)  # pylint: disable=protected-access
    return outputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {'mask_value': self.mask_value}
    base_config = super(Masking, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
