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
"""Adagrad optimizer implementation."""

from keras import initializers
from keras.optimizer_experimental import optimizer
from keras.utils import generic_utils
import tensorflow.compat.v2 as tf


@generic_utils.register_keras_serializable()
class Adagrad(optimizer.Optimizer):
  r"""Optimizer that implements the Adagrad algorithm.

  Adagrad is an optimizer with parameter-specific learning rates,
  which are adapted relative to how frequently a parameter gets
  updated during training. The more updates a parameter receives,
  the smaller the updates.

  Attributes:
    learning_rate: Initial value for the learning rate:
      either a floating point value,
      or a `tf.keras.optimizers.schedules.LearningRateSchedule` instance.
      Defaults to 0.001.
      Note that `Adagrad` tends to benefit from higher initial learning rate
      values compared to other optimizers.
      To match the exact form in the original paper, use 1.0.
    initial_accumulator_value: Floating point value.
      Starting value for the accumulators (per-parameter momentum values).
      Must be non-negative.
    epsilon: Small floating point value used to maintain numerical stability.
    gradients_clip_option: an instance of
      `optimizer_experimental.GradientsClipOption`, for attributes related to
      gradients clipping, such as clipnorm and clipvalue. Default to None
      (not applying gradients clipping).
    ema_option: an instance of `optimizer_experimental.EMAOption`, for
      attributes related to exponenatial moving average, such as `use_ema` (a
      boolean field indicates if EMA is used) and EMA momentum. Default to None
      (not applying EMA).
    jit_compile: Bool, default to False. If True, the optimizer will use XLA
        acceleration. `jit_compile` can only be False when using Parameter
        Server Strategy.
    name: Optional name prefix for the operations created when applying
      gradients.  Defaults to `"Adagrad"`.
    **kwargs: keyword arguments only used for backward compatibility with
      `optimizer_v2.OptimizerV2`. Any new code using
      `optimizer_experimental.Optimizer` should leave this parameter empty.

  Reference:
    - [Duchi et al., 2011](
      http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).
  """

  def __init__(self,
               learning_rate=0.001,
               initial_accumulator_value=0.1,
               epsilon=1e-7,
               gradients_clip_option=None,
               ema_option=None,
               jit_compile=False,
               name='Adagrad',
               **kwargs):
    super(Adagrad, self).__init__(
        gradients_clip_option=gradients_clip_option,
        ema_option=ema_option,
        jit_compile=jit_compile,
        name=name,
        **kwargs)
    self._learning_rate = self._build_learning_rate(learning_rate)
    self.initial_accumulator_value = initial_accumulator_value
    self.epsilon = epsilon

  def build(self, var_list):
    super().build(var_list)
    if hasattr(self, '_built') and self._built:
      return
    self._built = True
    self._accumulators = []
    initializer = initializers.Constant(self.initial_accumulator_value)
    for var in var_list:
      self._accumulators.append(
          self.add_variable_from_reference(
              var, 'accumulator', initializer(shape=var.shape,
                                              dtype=var.dtype)))

  def update_step(self, grad, variable, params=None):
    """Update step given gradient and the associated model variable."""
    if self._var_key(variable) not in self._index_dict:
      raise KeyError(f'Optimizer cannot recognize variable {variable.name}, '
                     f'this usually means you are calling an optimizer '
                     f'previously used on a different model. Please try '
                     f'creating a new optimizer instance.')
    lr = tf.cast(self.learning_rate, variable.dtype)

    var_key = self._var_key(variable)
    accumulator = self._accumulators[self._index_dict[var_key]]

    if isinstance(grad, tf.IndexedSlices):
      # Sparse gradients.
      accumulator.scatter_add(
          tf.IndexedSlices(grad.values * grad.values, grad.indices))
    else:
      # Dense gradients.
      accumulator.assign_add(grad * grad)
    variable.assign_sub(lr * grad / tf.sqrt(accumulator + self.epsilon))

  def get_config(self):
    config = super(Adagrad, self).get_config()

    config.update({
        'learning_rate': self._serialize_hyperparameter(self._learning_rate),
        'initial_accumulator_value': self.initial_accumulator_value,
        'epsilon': self.epsilon,
    })
    return config
