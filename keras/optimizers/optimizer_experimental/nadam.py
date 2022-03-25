# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Nadam optimizer implementation."""

from keras.optimizers.optimizer_experimental import optimizer
from keras.utils import generic_utils
import tensorflow.compat.v2 as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util.tf_export import keras_export


# pylint: disable=g-classes-have-attributes
@generic_utils.register_keras_serializable()
@keras_export('keras.optimizers.experimental.Nadam', v1=[])
class Nadam(optimizer.Optimizer):
  r"""Optimizer that implements the Nadam algorithm.

  Much like Adam is essentially RMSprop with momentum, Nadam is Adam with
  Nesterov momentum.

  Args:
    learning_rate: A `tf.Tensor`, floating point value, a schedule that is a
      `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
      that takes no arguments and returns the actual value to use. The
      learning rate. Defaults to 0.001.
    beta_1: A float value or a constant float tensor, or a callable
      that takes no arguments and returns the actual value to use. The
      exponential decay rate for the 1st moment estimates. Defaults to 0.9.
    beta_2: A float value or a constant float tensor, or a callable
      that takes no arguments and returns the actual value to use. The
      exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
    epsilon: A small constant for numerical stability. This epsilon is
      "epsilon hat" in the Kingma and Ba paper (in the formula just before
      Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
      1e-7.
    {{base_optimizer_keyword_args}}

  Reference:
    - [Dozat, 2015](http://cs229.stanford.edu/proj2015/054_report.pdf).

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
               name='Nadam',
               **kwargs):
    super().__init__(
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

    Nadam optimizer has 2 types of variables: momentums and velocities.

    Args:
      var_list: list of model variables to build Nadam variables on.
    """
    super().build(var_list)
    if getattr(self, '_built', False):
      return
    self._built = True
    self._momentums = []
    self._velocities = []
    self._u_product = tf.Variable(1.0, dtype=var_list[0].dtype)
    # Keep a counter on how many times of _u_product has been computed to
    # avoid duplicated computations.
    self._u_product_counter = 1

    for var in var_list:
      self._momentums.append(
          self.add_variable_from_reference(
              model_variable=var, variable_name='m'))
      self._velocities.append(
          self.add_variable_from_reference(
              model_variable=var, variable_name='v'))

  def update_step(self, gradient, variable):
    """Update step given gradient and the associated model variable."""
    var_dtype = variable.dtype
    lr = tf.cast(self.learning_rate, var_dtype)
    local_step = tf.cast(self.iterations + 1, var_dtype)
    next_step = tf.cast(self.iterations + 2, var_dtype)
    decay = tf.cast(0.96, var_dtype)
    beta_1 = tf.cast(self.beta_1, var_dtype)
    beta_2 = tf.cast(self.beta_2, var_dtype)
    u_t = beta_1 * (1. - 0.5 * (tf.pow(decay, local_step)))
    u_t_1 = beta_1 * (1. - 0.5 * (tf.pow(decay, next_step)))
    def get_cached_u_product():
      return self._u_product

    def compute_new_u_product():
      u_product_t = self._u_product * u_t
      self._u_product.assign(u_product_t)
      self._u_product_counter += 1
      return u_product_t

    u_product_t = tf.cond(
        self._u_product_counter == (self.iterations + 2),
        true_fn=get_cached_u_product,
        false_fn=compute_new_u_product)
    u_product_t_1 = u_product_t * u_t_1
    beta_2_power = tf.pow(beta_2, local_step)

    var_key = self._var_key(variable)
    m = self._momentums[self._index_dict[var_key]]
    v = self._velocities[self._index_dict[var_key]]

    if isinstance(gradient, tf.IndexedSlices):
      # Sparse gradients.
      m.assign_add(-m * (1 - beta_1))
      m.scatter_add(
          tf.IndexedSlices(gradient.values * (1 - beta_1),
                           gradient.indices))
      v.assign_add(-v * (1 - beta_2))
      v.scatter_add(
          tf.IndexedSlices(
              tf.square(gradient.values) * (1 - beta_2), gradient.indices))
      m_hat = (
          u_t_1 * m / (1 - u_product_t_1) + (1 - u_t) * gradient /
          (1 - u_product_t))
      v_hat = v / (1 - beta_2_power)

      variable.assign_sub((m_hat * lr) / (tf.sqrt(v_hat) + self.epsilon))
    else:
      # Dense gradients.
      m.assign_add((gradient - m) * (1 - beta_1))
      v.assign_add((tf.square(gradient) - v) * (1 - beta_2))
      m_hat = (
          u_t_1 * m / (1 - u_product_t_1) + (1 - u_t) * gradient /
          (1 - u_product_t))
      v_hat = v / (1 - beta_2_power)

      variable.assign_sub((m_hat * lr) / (tf.sqrt(v_hat) + self.epsilon))

  def get_config(self):
    config = super().get_config()

    config.update({
        'learning_rate': self._serialize_hyperparameter(self._learning_rate),
        'beta_1': self.beta_1,
        'beta_2': self.beta_2,
        'epsilon': self.epsilon,
    })
    return config

Nadam.__doc__ = Nadam.__doc__.replace(
    '{{base_optimizer_keyword_args}}', optimizer.base_optimizer_keyword_args)
