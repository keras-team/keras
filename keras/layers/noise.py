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
"""Layers that operate regularization via the addition of noise."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

from keras import backend
from keras.engine import base_layer
from keras.utils import tf_utils

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow.python.util.tf_export import keras_export


class BaseRandomLayer(base_layer.Layer):
  """A layer handle the random nubmer creation and savemodel behavior."""

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def __init__(self, seed=None, **kwargs):
    # Note that the constructor is annotated with
    # @no_automatic_dependency_tracking. This is to skip the auto
    # tracking of self._random_generator instance, which is an AutoTrackable.
    # The backend.RandomGenerator could contain a tf.random.Generator instance
    # which will have tf.Variable as the internal state. We want to avoid saving
    # that state into model.weights and checkpoints for backward compatibility
    # reason. In the meantime, we still need to make them visible to SavedModel
    # when it is tracing the tf.function for the `call()`.
    # See _list_extra_dependencies_for_serialization below for more details.
    super().__init__(**kwargs)
    self._random_generator = backend.RandomGenerator(seed)

  def _list_extra_dependencies_for_serialization(self, serialization_cache):
    # This method exposes the self._random_generator to SavedModel
    # only (not layer.weights and checkpoint).
    deps = super()._list_extra_dependencies_for_serialization(
        serialization_cache)
    deps['_random_generator'] = self._random_generator
    return deps


@keras_export('keras.layers.GaussianNoise')
class GaussianNoise(BaseRandomLayer):
  """Apply additive zero-centered Gaussian noise.

  This is useful to mitigate overfitting
  (you could see it as a form of random data augmentation).
  Gaussian Noise (GS) is a natural choice as corruption process
  for real valued inputs.

  As it is a regularization layer, it is only active at training time.

  Args:
    stddev: Float, standard deviation of the noise distribution.
    seed: Integer, optional random seed to enable deterministic behavior.

  Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode (adding noise) or in inference mode (doing nothing).

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as input.
  """

  def __init__(self, stddev, seed=None, **kwargs):
    super(GaussianNoise, self).__init__(seed=seed, **kwargs)
    self.supports_masking = True
    self.stddev = stddev
    self.seed = seed

  def call(self, inputs, training=None):

    def noised():
      return inputs + self._random_generator.random_normal(
          shape=tf.shape(inputs),
          mean=0.,
          stddev=self.stddev,
          dtype=inputs.dtype)

    return backend.in_train_phase(noised, inputs, training=training)

  def get_config(self):
    config = {'stddev': self.stddev, 'seed': self.seed}
    base_config = super(GaussianNoise, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape


@keras_export('keras.layers.GaussianDropout')
class GaussianDropout(BaseRandomLayer):
  """Apply multiplicative 1-centered Gaussian noise.

  As it is a regularization layer, it is only active at training time.

  Args:
    rate: Float, drop probability (as with `Dropout`).
      The multiplicative noise will have
      standard deviation `sqrt(rate / (1 - rate))`.
    seed: Integer, optional random seed to enable deterministic behavior.

  Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as input.
  """

  def __init__(self, rate, seed=None, **kwargs):
    super(GaussianDropout, self).__init__(seed=seed, **kwargs)
    self.supports_masking = True
    self.rate = rate
    self.seed = seed

  def call(self, inputs, training=None):
    if 0 < self.rate < 1:

      def noised():
        stddev = np.sqrt(self.rate / (1.0 - self.rate))
        return inputs * self._random_generator.random_normal(
            shape=tf.shape(inputs),
            mean=1.0,
            stddev=stddev,
            dtype=inputs.dtype)

      return backend.in_train_phase(noised, inputs, training=training)
    return inputs

  def get_config(self):
    config = {'rate': self.rate, 'seed': self.seed}
    base_config = super(GaussianDropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape


@keras_export('keras.layers.AlphaDropout')
class AlphaDropout(BaseRandomLayer):
  """Applies Alpha Dropout to the input.

  Alpha Dropout is a `Dropout` that keeps mean and variance of inputs
  to their original values, in order to ensure the self-normalizing property
  even after this dropout.
  Alpha Dropout fits well to Scaled Exponential Linear Units
  by randomly setting activations to the negative saturation value.

  Args:
    rate: float, drop probability (as with `Dropout`).
      The multiplicative noise will have
      standard deviation `sqrt(rate / (1 - rate))`.
    seed: Integer, optional random seed to enable deterministic behavior.

  Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as input.
  """

  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    super(AlphaDropout, self).__init__(seed=seed, **kwargs)
    self.rate = rate
    self.noise_shape = noise_shape
    self.seed = seed
    self.supports_masking = True

  def _get_noise_shape(self, inputs):
    return self.noise_shape if self.noise_shape else tf.shape(inputs)

  def call(self, inputs, training=None):
    if 0. < self.rate < 1.:
      noise_shape = self._get_noise_shape(inputs)

      def dropped_inputs(inputs=inputs, rate=self.rate):  # pylint: disable=missing-docstring
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        alpha_p = -alpha * scale

        kept_idx = tf.greater_equal(
            self._random_generator.random_uniform(noise_shape), rate)
        kept_idx = tf.cast(kept_idx, inputs.dtype)

        # Get affine transformation params
        a = ((1 - rate) * (1 + rate * alpha_p**2))**-0.5
        b = -a * alpha_p * rate

        # Apply mask
        x = inputs * kept_idx + alpha_p * (1 - kept_idx)

        # Do affine transformation
        return a * x + b

      return backend.in_train_phase(dropped_inputs, inputs, training=training)
    return inputs

  def get_config(self):
    config = {'rate': self.rate, 'seed': self.seed}
    base_config = super(AlphaDropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape
