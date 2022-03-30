# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Adamax optimizer implementation."""

from keras.optimizers.optimizer_experimental import optimizer
from keras.utils import generic_utils
import tensorflow.compat.v2 as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util.tf_export import keras_export


# pylint: disable=g-classes-have-attributes
@generic_utils.register_keras_serializable()
@keras_export('keras.optimizers.experimental.Adamax', v1=[])
class Adamax(optimizer.Optimizer):
  """Optimizer that implements the Adamax algorithm.

  Adamax, a variant of Adam based on the infinity norm, is a first-order
  gradient-based optimization method. Due to its capability of adjusting the
  learning rate based on data characteristics, it is suited to learn
  time-variant process, e.g., speech data with dynamically changed noise
  conditions. Default parameters follow those provided in the paper (see
  references below).

  Initialization:

  ```python
  m = 0  # Initialize initial 1st moment vector
  u = 0  # Initialize the exponentially weighted infinity norm
  t = 0  # Initialize timestep
  ```

  The update rule for parameter `w` with gradient `g` is
  described at the end of section 7.1 of the paper (see the referenece section):

  ```python
  t += 1
  m = beta1 * m + (1 - beta) * g
  u = max(beta2 * u, abs(g))
  current_lr = learning_rate / (1 - beta1 ** t)
  w = w - current_lr * m / (u + epsilon)
  ```

  Args:
    learning_rate: A `tf.Tensor`, floating point value, a schedule that is a
      `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
      that takes no arguments and returns the actual value to use. The
      learning rate. Defaults to 0.001.
    beta_1: A float value or a constant float tensor. The exponential decay
      rate for the 1st moment estimates.
    beta_2: A float value or a constant float tensor. The exponential decay
      rate for the exponentially weighted infinity norm.
    epsilon: A small constant for numerical stability.
    {{base_optimizer_keyword_args}}

  Reference:
    - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
  """

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               clipnorm=None,
               clipvalue=None,
               global_clipnorm=None,
               use_ema=False,
               ema_momentum=0.99,
               ema_overwrite_frequency=None,
               jit_compile=True,
               name='Adamax',
               **kwargs):
    super(Adamax, self).__init__(
        name=name,
        clipnorm=clipnorm,
        clipvalue=clipvalue,
        global_clipnorm=global_clipnorm,
        use_ema=use_ema,
        ema_momentum=ema_momentum,
        ema_overwrite_frequency=ema_overwrite_frequency,
        jit_compile=jit_compile,
        **kwargs)
    self._learning_rate = self._build_learning_rate(learning_rate)
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon

  def build(self, var_list):
    """Initialize optimizer variables.

    Adamax optimizer has 2 types of variables: momentums (denoted as m),
    exponentially weighted infinity norm (denoted as u).

    Args:
      var_list: list of model variables to build Adamax variables on.
    """
    super().build(var_list)
    if hasattr(self, '_built') and self._built:
      return
    self._built = True
    self._m = []
    self._u = []
    for var in var_list:
      self._m.append(
          self.add_variable_from_reference(
              model_variable=var, variable_name='m'))
      self._u.append(
          self.add_variable_from_reference(
              model_variable=var, variable_name='u'))

  def update_step(self, gradient, variable):
    """Update step given gradient and the associated model variable."""
    lr = tf.cast(self.learning_rate, variable.dtype)
    local_step = tf.cast(self.iterations + 1, variable.dtype)
    beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)

    var_key = self._var_key(variable)
    m = self._m[self._index_dict[var_key]]
    u = self._u[self._index_dict[var_key]]

    if isinstance(gradient, tf.IndexedSlices):
      # Sparse gradients.
      indices = gradient.indices
      m.assign_add(-m * (1 - self.beta_1))
      m.scatter_add(
          tf.IndexedSlices(gradient.values * (1 - self.beta_1), indices))
      u.assign(u * self.beta_2)
      u_slice = tf.gather(u, indices)
      u_slice_incremental = tf.maximum(
          u_slice,
          tf.abs(gradient.values)) - u_slice
      u.scatter_add(tf.IndexedSlices(u_slice_incremental, indices))
      variable.assign_sub((lr * m) / ((1 - beta_1_power) * (u + self.epsilon)))
    else:
      # Dense gradients.
      m.assign_add((gradient - m) * (1 - self.beta_1))
      u.assign(tf.maximum(self.beta_2 * u, tf.abs(gradient)))
      variable.assign_sub((lr * m) / ((1 - beta_1_power) * (u + self.epsilon)))

  def get_config(self):
    config = super(Adamax, self).get_config()

    config.update({
        'learning_rate': self._serialize_hyperparameter(self._learning_rate),
        'beta_1': self.beta_1,
        'beta_2': self.beta_2,
        'epsilon': self.epsilon,
    })
    return config


Adamax.__doc__ = Adamax.__doc__.replace(
    '{{base_optimizer_keyword_args}}', optimizer.base_optimizer_keyword_args)
