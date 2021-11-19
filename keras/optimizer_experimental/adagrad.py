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
    clipnorm: float. If set, the gradient of each weight is individually
      clipped so that its norm is no higher than this value.
    clipvalue: float. If set, the gradient of each weight is clipped to be
      no higher than this value.
    global_clipnorm: float. If set, the gradient of all weights is clipped
      so that their global norm is no higher than this value.
    use_ema: boolean, default to False. If True, exponential moving average
      (EMA) is applied. EMA consists of computing an exponential moving
      average of the weights of the model (as the weight values change after
      each training batch), and periodically overwriting the weights with
      their moving average.
    ema_momentum: float, default to 0.99. Only used if `use_ema=True`. This is
      the momentum to use when computing the EMA of the model's weights:
      `new_average = ema_momentum * old_average +
       (1 - ema_momentum) * current_variable_value`.
    ema_overwrite_frequency: int or None, default to 100. Only used if
      `use_ema=True`. Every ema_overwrite_frequency steps of iterations, we
      overwrite the model variable by its stored moving average. If None, we
      do not overwrite model variables in the middle of training, and users
      need to overwrite the model variable by calling
      `finalize_variable_update()`.
    name: Optional name prefix for the operations created when applying
      gradients.  Defaults to `"Adagrad"`.

  Reference:
    - [Duchi et al., 2011](
      http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).
  """

  def __init__(self,
               learning_rate=0.001,
               initial_accumulator_value=0.1,
               epsilon=1e-7,
               clipnorm=None,
               clipvalue=None,
               global_clipnorm=None,
               use_ema=False,
               ema_momentum=0.99,
               ema_overwrite_frequency=100,
               name='Adagrad'):
    super(Adagrad, self).__init__(
        clipnorm=clipnorm,
        clipvalue=clipvalue,
        global_clipnorm=global_clipnorm,
        use_ema=use_ema,
        ema_momentum=ema_momentum,
        ema_overwrite_frequency=ema_overwrite_frequency,
        name=name)
    self._learning_rate = self._build_learning_rate(learning_rate)
    self.initial_accumulator_value = initial_accumulator_value
    self.epsilon = 1e-7

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
