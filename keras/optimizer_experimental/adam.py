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
"""Adam optimizer implementation."""

from keras.optimizer_experimental import optimizer
from keras.utils import generic_utils
import tensorflow.compat.v2 as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util.tf_export import keras_export


@generic_utils.register_keras_serializable()
@keras_export('keras.optimizers.experimental.Adam', v1=[])
class Adam(optimizer.Optimizer):
  r"""Optimizer that implements the Adam algorithm.

  Adam optimization is a stochastic gradient descent method that is based on
  adaptive estimation of first-order and second-order moments.

  According to
  [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
  the method is "*computationally
  efficient, has little memory requirement, invariant to diagonal rescaling of
  gradients, and is well suited for problems that are large in terms of
  data/parameters*".

  Attributes:
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
    amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
      the paper "On the Convergence of Adam and beyond". Defaults to `False`.
    clipnorm: see the `clipnorm` argument of `optimizer_experimental.Optimizer`.
    clipvalue: see the `clipvalue` argument of
      `optimizer_experimental.Optimizer`.
    global_clipnorm: see the `global_clipnorm` argument of
      `optimizer_experimental.Optimizer`.
    use_ema: see the `use_ema` argument of `optimizer_experimental.Optimizer`.
    ema_momentum: see the `ema_momentum` argument of
      `optimizer_experimental.Optimizer`.
    ema_overwrite_frequency: see the `ema_overwrite_frequency` argument of
      `optimizer_experimental.Optimizer`.
    jit_compile: see the `jit_compile` argument of
      `optimizer_experimental.Optimizer`.
    name: Optional name prefix for the operations created when applying
      gradients. Defaults to `"Adam"`.
    **kwargs: see the `**kwargs` argument of `optimizer_experimental.Optimizer`.

  Reference:
    - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
    - [Reddi et al., 2018](
        https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.

  Notes:

  The default value of 1e-7 for epsilon might not be a good default in
  general. For example, when training an Inception network on ImageNet a
  current good choice is 1.0 or 0.1. Note that since Adam uses the
  formulation just before Section 2.1 of the Kingma and Ba paper rather than
  the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
  hat" in the paper.

  The sparse implementation of this algorithm (used when the gradient is an
  IndexedSlices object, typically because of `tf.gather` or an embedding
  lookup in the forward pass) does apply momentum to variable slices even if
  they were not used in the forward pass (meaning they have a gradient equal
  to zero). Momentum decay (beta1) is also applied to the entire momentum
  accumulator. This means that the sparse behavior is equivalent to the dense
  behavior (in contrast to some momentum implementations which ignore momentum
  unless a variable slice was actually used).
  """

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               amsgrad=False,
               clipnorm=None,
               clipvalue=None,
               global_clipnorm=None,
               use_ema=False,
               ema_momentum=0.99,
               ema_overwrite_frequency=None,
               jit_compile=False,
               name='Adam',
               **kwargs):
    super(Adam, self).__init__(
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
    self.amsgrad = amsgrad

  def build(self, var_list):
    """Initialize optimizer variables.

    Adam optimizer has 3 types of variables: momentums, velocities and
    velocity_hat (only set when amsgrad is applied),

    Args:
      var_list: list of model variables to build Adam variables on.
    """
    super().build(var_list)
    if hasattr(self, '_built') and self._built:
      return
    self._built = True
    self._momentums = []
    self._velocities = []
    for var in var_list:
      self._momentums.append(
          self.add_variable_from_reference(
              model_variable=var, variable_name='m'))
      self._velocities.append(
          self.add_variable_from_reference(
              model_variable=var, variable_name='v'))
    if self.amsgrad:
      self._velocity_hats = []
      for var in var_list:
        self._velocity_hats.append(
            self.add_variable_from_reference(
                model_variable=var, variable_name='vhat'))

  def update_step(self, gradient, variable):
    """Update step given gradient and the associated model variable."""
    if self._var_key(variable) not in self._index_dict:
      raise KeyError(f'Optimizer cannot recognize variable {variable.name}, '
                     f'this usually means you are calling an optimizer '
                     f'previously used on a different model. Please try '
                     f'creating a new optimizer instance.')
    beta_1_power = None
    beta_2_power = None
    lr = tf.cast(self.learning_rate, variable.dtype)
    local_step = tf.cast(self.iterations + 1, variable.dtype)
    beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
    beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)

    var_key = self._var_key(variable)
    m = self._momentums[self._index_dict[var_key]]
    v = self._velocities[self._index_dict[var_key]]

    alpha = (lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power))

    if isinstance(gradient, tf.IndexedSlices):
      # Sparse gradients.
      m.assign_add(-m * (1 - self.beta_1))
      m.scatter_add(
          tf.IndexedSlices(gradient.values * (1 - self.beta_1),
                           gradient.indices))
      v.assign_add(-v * (1 - self.beta_2))
      v.scatter_add(
          tf.IndexedSlices(
              tf.square(gradient.values) * (1 - self.beta_2), gradient.indices))
      if self.amsgrad:
        v_hat = self._velocity_hats[self._index_dict[var_key]]
        v_hat.assign(tf.maximum(v_hat, v))
        v = v_hat
      variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
    else:
      # Dense gradients.
      m.assign_add((gradient - m) * (1 - self.beta_1))
      v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))
      if self.amsgrad:
        v_hat = self._velocity_hats[self._index_dict[var_key]]
        v_hat.assign(tf.maximum(v_hat, v))
        v = v_hat
      variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))

  def get_config(self):
    config = super(Adam, self).get_config()

    config.update({
        'learning_rate': self._serialize_hyperparameter(self._learning_rate),
        'beta_1': self.beta_1,
        'beta_2': self.beta_2,
        'epsilon': self.epsilon,
        'amsgrad': self.amsgrad,
    })
    return config
