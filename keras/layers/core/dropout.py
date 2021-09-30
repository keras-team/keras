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
"""Contains the dropout layer."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

from keras import backend
from keras.engine.base_layer import Layer
from keras.utils import control_flow_util
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.Dropout')
class Dropout(Layer):
  """Applies Dropout to the input.

  The Dropout layer randomly sets input units to 0 with a frequency of `rate`
  at each step during training time, which helps prevent overfitting.
  Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over
  all inputs is unchanged.

  Note that the Dropout layer only applies when `training` is set to True
  such that no values are dropped during inference. When using `model.fit`,
  `training` will be appropriately set to True automatically, and in other
  contexts, you can set the kwarg explicitly to True when calling the layer.

  (This is in contrast to setting `trainable=False` for a Dropout layer.
  `trainable` does not affect the layer's behavior, as Dropout does
  not have any variables/weights that can be frozen during training.)

  >>> tf.random.set_seed(0)
  >>> layer = tf.keras.layers.Dropout(.2, input_shape=(2,))
  >>> data = np.arange(10).reshape(5, 2).astype(np.float32)
  >>> print(data)
  [[0. 1.]
   [2. 3.]
   [4. 5.]
   [6. 7.]
   [8. 9.]]
  >>> outputs = layer(data, training=True)
  >>> print(outputs)
  tf.Tensor(
  [[ 0.    1.25]
   [ 2.5   3.75]
   [ 5.    6.25]
   [ 7.5   8.75]
   [10.    0.  ]], shape=(5, 2), dtype=float32)

  Args:
    rate: Float between 0 and 1. Fraction of the input units to drop.
    noise_shape: 1D integer tensor representing the shape of the
      binary dropout mask that will be multiplied with the input.
      For instance, if your inputs have shape
      `(batch_size, timesteps, features)` and
      you want the dropout mask to be the same for all timesteps,
      you can use `noise_shape=(batch_size, 1, features)`.
    seed: A Python integer to use as random seed.

  Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).
  """

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    # Note that the constructor is annotated with
    # @no_automatic_dependency_tracking. This is to skip the auto
    # tracking of self._random_generator instance, which is an AutoTrackable.
    # The backend.RandomGenerator could contain a tf.random.Generator instance
    # which will have tf.Variable as the internal state. We want to avoid saving
    # that state into model.weights and checkpoints for backward compatibility
    # reason. In the meantime, we still need to make them visible to SavedModel
    # when it is tracing the tf.function for the `call()`.
    # See _list_extra_dependencies_for_serialization below for more details.
    super(Dropout, self).__init__(**kwargs)
    if isinstance(rate, (int, float)) and not 0 <= rate <= 1:
      raise ValueError(f'Invalid value {rate} received for '
                       f'`rate`, expected a value between 0 and 1.')
    self.rate = rate
    self.noise_shape = noise_shape
    self.seed = seed
    self.supports_masking = True
    self._random_generator = backend.RandomGenerator(seed)

  def build(self, input_shape):
    self._random_generator._maybe_init()  # pylint: disable=protected-access

  def _get_noise_shape(self, inputs):
    # Subclasses of `Dropout` may implement `_get_noise_shape(self, inputs)`,
    # which will override `self.noise_shape`, and allows for custom noise
    # shapes with dynamically sized inputs.
    if self.noise_shape is None:
      return None

    concrete_inputs_shape = tf.shape(inputs)
    noise_shape = []
    for i, value in enumerate(self.noise_shape):
      noise_shape.append(concrete_inputs_shape[i] if value is None else value)
    return tf.convert_to_tensor(noise_shape)

  def call(self, inputs, training=None):
    if training is None:
      training = backend.learning_phase()

    def dropped_inputs():
      return self._random_generator.dropout(
          inputs, self.rate, noise_shape=self._get_noise_shape(inputs))

    output = control_flow_util.smart_cond(training, dropped_inputs,
                                          lambda: tf.identity(inputs))
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'rate': self.rate,
        'noise_shape': self.noise_shape,
        'seed': self.seed
    }
    base_config = super(Dropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _list_extra_dependencies_for_serialization(self, serialization_cache):
    # This method exposes the self._random_generator to SavedModel only
    # (not layer.weights and checkpoint).
    deps = super()._list_extra_dependencies_for_serialization(
        serialization_cache)
    deps['_random_generator'] = self._random_generator
    return deps
