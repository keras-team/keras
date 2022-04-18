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
"""DTensor specific Keras optimizers."""

from keras.dtensor import dtensor_api as dtensor
from keras.optimizers.optimizer_experimental import adadelta
from keras.optimizers.optimizer_experimental import adagrad
from keras.optimizers.optimizer_experimental import adam
from keras.optimizers.optimizer_experimental import optimizer as optimizer_lib
from keras.optimizers.optimizer_experimental import rmsprop
from keras.optimizers.optimizer_experimental import sgd
from keras.optimizers.schedules import learning_rate_schedule

import tensorflow.compat.v2 as tf

from tensorflow.python.util.tf_export import keras_export  # pylint: disable=g-direct-tensorflow-import
from tensorflow.tools.docs import doc_controls


# pylint: disable=protected-access,missing-class-docstring
class Optimizer(optimizer_lib._BaseOptimizer):
  """DTensor specific optimizers.

  The major changes for this class is that all the variable init logic will be
  mesh/layout aware.

  """
  # Note that we didn't subclass optimizer_lib.Optimizer since it contains the
  # extra logic of handling distribution strategy, which we don't need for
  # DTensor

  def __init__(self, name, mesh=None):
    """Create a new Optimizer.

    Args:
      name: String. The name of the optimizer, which will appear in all the
        state variables created by this optimizer.
      mesh: dtensor.Mesh. The optional Mesh which will be used to create
        the states. Note that usually the state variable will use the layout
        from the corresponding model variables. This mesh only used for global
        variables like globle steps, learning rate, etc.
    """
    # TODO(scottzhu): Skip the gradients_clip_option and ema_option for now, and
    # will cover them in future if really needed.
    # TODO(scottzhu): We might want to make mesh to be required in future.
    self._mesh = mesh
    super().__init__(name=name)

  def _create_iteration_variable(self):
    init_val = tf.constant(0, dtype=tf.int64)
    if self._mesh:
      init_val = dtensor.copy_to_mesh(
          init_val, dtensor.Layout.replicated(self._mesh, rank=0))
    with tf.init_scope():
      # Lift the variable creation to init scope to avoid environment issue.
      self._iterations = dtensor.DVariable(init_val, name='iteration')

  ################## Override methods from keras.Optimizer ################
  def add_variable_from_reference(self,
                                  model_variable,
                                  variable_name,
                                  initial_value=None):
    """Create an optimizer variable from model variable.

    Create an optimizer variable based on the information of model variable.
    For example, in SGD optimizer momemtum, for each model variable, a
    corresponding momemtum variable is created of the same shape and dtype.

    Args:
      model_variable: The corresponding model variable to the optimizer variable
        to be created.
      variable_name: The name prefix of the optimizer variable to be created.
        The create variables name will follow the pattern
        `{variable_name}/{model_variable.name}`, e.g., `momemtum/dense_1`.
      initial_value: The initial value of the optimizer variable, if None, the
        value will be default to 0.

    Returns:
      An optimizer variable.
    """
    if initial_value is None:
      # Use tf.zeros_like which will propagate the layout information from the
      # model weights if any.
      initial_value = tf.zeros_like(model_variable)
    elif isinstance(initial_value, tf.Tensor):
      initial_value = dtensor.copy_to_mesh(
          initial_value,
          dtensor.Layout.replicated(self._mesh, rank=initial_value.shape.rank))
    return dtensor.DVariable(
        initial_value=initial_value,
        name=f'{variable_name}/{model_variable._shared_name}',
        dtype=model_variable.dtype,
        trainable=False)

  @doc_controls.do_not_generate_docs
  def aggregate_gradients(self, grads_and_vars):
    # Hide the aggregate_gradients from Optimizer.aggregate_gradients
    raise NotImplementedError(
        'Dtensor doesn\'t need to manually aggregate gradients')

  def _var_key(self, variable):
    """Get a unique identifier of the given variable."""
    return optimizer_lib._BaseOptimizer._var_key(self, variable)

  def apply_gradients(self, grads_and_vars):
    """Apply gradients to variables.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.

    Returns:
      None

    Raises:
      TypeError: If `grads_and_vars` is malformed.
    """
    # Explicitly call the _BaseOptimizer to avoid any chance of using
    # Optimizers.apply_gradients which contains distribution strategy logic.
    optimizer_lib._BaseOptimizer.apply_gradients(self, grads_and_vars)

  def _internal_apply_gradients(self, grads_and_vars):
    """Helper function of apply gradients.

    This is required for separating out distributed training logic.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.
    """
    # Explicitly call the _BaseOptimizer to avoid any chance of using
    # Optimizers.apply_gradients which contains distribution strategy logic.
    optimizer_lib._BaseOptimizer._internal_apply_gradients(self, grads_and_vars)

  def _overwrite_model_variables_with_average_value_helper(self, var_list):
    """Helper function to _overwrite_model_variables_with_average_value."""
    (optimizer_lib._BaseOptimizer.
     _overwrite_model_variables_with_average_value_helper(self, var_list))

  def _build_learning_rate(self, learning_rate):
    if isinstance(learning_rate, learning_rate_schedule.LearningRateSchedule):
      # Create a variable to hold the current learning rate.
      # Note that the init value `learning_rate(self.iterations)` should have
      # the correct layout information from self.iterations.
      self._current_learning_rate = dtensor.DVariable(
          learning_rate(self.iterations),
          name='learning_rate',
          dtype=tf.float32)
      return learning_rate
    init_val = tf.constant(learning_rate, dtype=tf.float32)
    if self._mesh:
      init_val = dtensor.copy_to_mesh(
          init_val, dtensor.Layout.replicated(self._mesh, rank=0))
    return dtensor.DVariable(init_val, name='learning_rate')


@keras_export('keras.dtensor.experimental.optimizers.Adadelta', v1=[])
class Adadelta(Optimizer, adadelta.Adadelta):

  def __init__(self,
               learning_rate=0.001,
               rho=0.95,
               epsilon=1e-7,
               gradients_clip_option=None,
               ema_option=None,
               name='Adadelta',
               mesh=None):
    # Skip the adam.Adadelta.__init__ and only call the Optimizer.__init__
    # this is to skip the keras.Optimizer.__init__, which contains the logic
    # of distribution strategy. Same for all the optimizers subclasses.
    Optimizer.__init__(self, name=name, mesh=mesh)
    self._learning_rate = self._build_learning_rate(learning_rate)
    self.rho = rho
    self.epsilon = epsilon


@keras_export('keras.dtensor.experimental.optimizers.Adagrad', v1=[])
class Adagrad(Optimizer, adagrad.Adagrad):

  def __init__(self,
               learning_rate=0.001,
               initial_accumulator_value=0.1,
               epsilon=1e-7,
               gradients_clip_option=None,
               ema_option=None,
               name='Adagrad',
               mesh=None):
    Optimizer.__init__(self, name=name, mesh=mesh)
    self._learning_rate = self._build_learning_rate(learning_rate)
    self.initial_accumulator_value = initial_accumulator_value
    self.epsilon = epsilon


@keras_export('keras.dtensor.experimental.optimizers.Adam', v1=[])
class Adam(Optimizer, adam.Adam):

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               amsgrad=False,
               gradients_clip_option=None,
               ema_option=None,
               name='Adam',
               mesh=None):
    Optimizer.__init__(self, name=name, mesh=mesh)
    self._learning_rate = self._build_learning_rate(learning_rate)
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.amsgrad = amsgrad


@keras_export('keras.dtensor.experimental.optimizers.RMSprop', v1=[])
class RMSprop(Optimizer, rmsprop.RMSprop):

  def __init__(self,
               learning_rate=0.001,
               rho=0.9,
               momentum=0.0,
               epsilon=1e-7,
               centered=False,
               gradients_clip_option=None,
               ema_option=None,
               jit_compile=False,
               name='RMSprop',
               mesh=None):
    Optimizer.__init__(self, name=name, mesh=mesh)
    self._learning_rate = self._build_learning_rate(learning_rate)
    self.rho = rho
    self.momentum = momentum
    self.epsilon = epsilon
    self.centered = centered


@keras_export('keras.dtensor.experimental.optimizers.SGD', v1=[])
class SGD(Optimizer, sgd.SGD):

  def __init__(self,
               learning_rate=0.01,
               momentum=0.0,
               nesterov=False,
               amsgrad=False,
               gradients_clip_option=None,
               ema_option=None,
               jit_compile=False,
               name='SGD',
               mesh=None):
    Optimizer.__init__(self, name=name, mesh=mesh)
    self._learning_rate = self._build_learning_rate(learning_rate)
    self.momentum = momentum
    self.nesterov = nesterov
    if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
      raise ValueError('`momentum` must be between [0, 1].')


Adadelta.__doc__ = Optimizer.__doc__ + adadelta.Adadelta.__doc__
Adagrad.__doc__ = Optimizer.__doc__ + adagrad.Adagrad.__doc__
Adam.__doc__ = Optimizer.__doc__ + adam.Adam.__doc__
RMSprop.__doc__ = Optimizer.__doc__ + rmsprop.RMSprop.__doc__
SGD.__doc__ = Optimizer.__doc__ + sgd.SGD.__doc__
