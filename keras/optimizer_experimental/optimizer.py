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
"""Base class of optimizer.

This is under development, and subject to interface/implementation changes.
"""

import abc
from absl import logging

from keras import backend
from keras import initializers
from keras.optimizer_v2 import learning_rate_schedule
from keras.optimizer_v2 import utils as optimizer_utils
import tensorflow.compat.v2 as tf


class _BaseOptimizer(tf.Module):
  """Optimizer base class, which only supports non-distribute use case."""

  def __init__(self,
               name,
               clipnorm=None,
               clipvalue=None,
               global_clipnorm=None,
               use_ema=False,
               ema_momentum=0.99,
               ema_overwrite_frequency=100,
               jit_compile=False,
               **kwargs):
    self._name = name
    self._clipnorm = clipnorm
    self._global_clipnorm = global_clipnorm
    self._clipvalue = clipvalue
    self._use_ema = use_ema
    self._jit_compile = jit_compile
    if use_ema:
      # Verify the arguments related to EMA.
      if ema_momentum > 1 or ema_momentum < 0:
        raise ValueError("`ema_momentum` must be in the range [0, 1]. "
                         f"Received: ema_momentum={ema_momentum}")
      if ema_overwrite_frequency and (not isinstance(
          ema_overwrite_frequency, int) or ema_overwrite_frequency < 1):
        raise ValueError(
            "`ema_overwrite_frequency` must be an integer > 1 or None. "
            f"Received: ema_overwrite_frequency={ema_overwrite_frequency}")
    self._ema_momentum = ema_momentum
    self._ema_overwrite_frequency = ema_overwrite_frequency

    if self._clipnorm is not None and self._global_clipnorm is not None:
      raise ValueError(f"At most one of `clipnorm` and `global_clipnorm` can "
                       f"be set. Received: clipnorm={self._clipnorm}, "
                       f"global_clipnorm={self._global_clipnorm}.")

    self._create_iteration_variable()
    self._process_kwargs(kwargs)

  def _create_iteration_variable(self):
    """Create the iterations counter variable."""
    with tf.init_scope():
      # Lift the variable creation to init scope to avoid environment issue.
      self._iterations = tf.Variable(0, name="iteration", dtype=tf.int64)

  def _process_kwargs(self, kwargs):
    legacy_kwargs = {
        "lr", "decay", "gradient_transformers", "gradient_aggregator"
    }
    for k in kwargs:
      if k in legacy_kwargs:
        logging.warning(
            "%s is deprecated in `optimizer_experimental.Optimizer`"
            ", please check the docstring for valid arguments.", k)
      else:
        raise TypeError(f"{k} is not a valid argument, kwargs should be empty "
                        " for `optimizer_experimental.Optimizer`.")

  def _var_key(self, variable):
    """Get a unique identifier of the given variable."""
    # Get the distributed variable if it exists.
    # TODO(b/199214315): replace _unique_id with ref() after fixing ref() issues
    # on AggregatingVariable.
    return variable._unique_id  # pylint: disable=protected-access

  @abc.abstractmethod
  def update_step(self, gradient, variable):
    """Function to update variable value based on given gradients.

    This method must be implemented in customized optimizers.

    Args:
      gradient: backpropagated gradient of the given variable.
      variable: variable whose value needs to be updated.

    Returns:
      An `Operation` that applies the specified gradients.

    """
    raise NotImplementedError

  def compute_gradients(self, loss, var_list, tape=None):
    """Compute gradients of loss on trainable variables.

    Args:
      loss: `Tensor` or callable. If a callable, `loss` should take no arguments
        and return the value to minimize.
      var_list: list or tuple of `Variable` objects to update to minimize
        `loss`.
      tape: (Optional) `tf.GradientTape`.

    Returns:
      A list of (gradient, variable) pairs. Variable is always present, but
      gradient can be `None`.
    """
    if tape is None:
      tape = tf.GradientTape()
    if callable(loss):
      with tape:
        tape.watch(var_list)
        loss = loss()
    grads = tape.gradient(loss, var_list)
    return list(zip(grads, var_list))

  def _clip_gradients(self, grads):
    clipped_grads = []
    if self._clipnorm and self._clipnorm > 0:
      for g in grads:
        if g is None:
          clipped_grads.append(g)
        else:
          clipped_grads.append(tf.clip_by_norm(g, self._clipnorm))
      return clipped_grads

    if self._global_clipnorm and self._global_clipnorm > 0:
      return tf.clip_by_global_norm(grads, self._global_clipnorm)[0]

    if self._clipvalue and self._clipvalue > 0:
      for g in grads:
        if g is None:
          clipped_grads.append(g)
        else:
          clipped_grads.append(
              tf.clip_by_value(
                  g,
                  clip_value_min=-self._clipvalue,  # pylint: disable=invalid-unary-operand-type
                  clip_value_max=self._clipvalue))
      return clipped_grads

    return grads

  @property
  def iterations(self):
    """The number of training steps this `optimizer` has run.

    By default, iterations would be incremented by one every time
    `apply_gradients()` is called.
    """
    return self._iterations

  @property
  def learning_rate(self):
    if not hasattr(self, "_learning_rate") or self._learning_rate is None:
      raise ValueError("Missing learning rate, please set self.learning_rate at"
                       " optimizer creation time.")
    lr = self._learning_rate
    if isinstance(lr, learning_rate_schedule.LearningRateSchedule):
      # If the optimizer takes in LearningRateSchedule, then each call to
      # learning_rate would return `self._current_learning_rate`, which is
      # updated at each call to `apply_gradients`.
      return self._current_learning_rate
    return lr

  @learning_rate.setter
  def learning_rate(self, learning_rate):
    if isinstance(self._learning_rate,
                  learning_rate_schedule.LearningRateSchedule):
      raise TypeError("This optimizer was created with a `LearningRateSchedule`"
                      " object as its `learning_rate` constructor argument, "
                      "hence its learning rate is not settable. If you need the"
                      " learning rate to be settable, you should instantiate "
                      "the optimizer with a float `learning_rate` argument.")
    self._learning_rate.assign(learning_rate)

  def _build_learning_rate(self, learning_rate):
    if isinstance(learning_rate, learning_rate_schedule.LearningRateSchedule):
      # Create a variable to hold the current learning rate.
      self._current_learning_rate = tf.Variable(
          learning_rate(self.iterations),
          name="learning_rate",
          dtype=tf.float32)
      return learning_rate
    return tf.Variable(
        learning_rate, name="learning_rate", dtype=backend.floatx())

  @abc.abstractmethod
  def build(self, var_list):
    """Initialize the optimizer's variables, such as momemtum variables.

    This function has to be implemented by subclass optimizers, and subclass
    optimizers need to call `super().build(var_list)`.

    Args:
      var_list: List of model variables to build optimizers on. For example, SGD
        optimizer with momentum will store one momentum variable corresponding
        to each model variable.
    """
    if getattr(self, "_built", False):
      return
    self._build_index_dict(var_list)
    if self._use_ema:
      self._model_variables_moving_average = []
      for var in var_list:
        # Make a copy of the model variables, we will use the copy to store the
        # moving average of model variables.
        self._model_variables_moving_average.append(
            self.add_variable_from_reference(var, "average", initial_value=var))

  def _build_index_dict(self, var_list):
    """Build variable to index dictionary.

    Build a dictionary that maps variable to the index of it in the given
    var_list.

    Args:
      var_list: List of variables to build index dict on.

    Returns:
      None
    """
    self._index_dict = {}
    for i, var in enumerate(var_list):
      var_key = self._var_key(var)
      self._index_dict[var_key] = i

  def add_variable(self, shape, dtype=None, initializer="zeros", name=None):
    """Create an optimizer variable.

    Args:
      shape: A list of integers, a tuple of integers, or a 1-D Tensor of type
        int32. Defaults to scalar if unspecified.
      dtype: The DType of the optimizer variable to be created. Defaults to
        `tf.keras.backend.floatx` if unspecified.
      initializer: string or callable. Initializer instance.
      name: The name of the optimizer variable to be created.

    Returns:
      An optimizer variable, in the format of tf.Variable.

    """
    if isinstance(initializer, str):
      initializer = initializers.get(initializer)
    if dtype is None:
      dtype = backend.floatx()
    if shape is None:
      shape = []
    return tf.Variable(
        initial_value=initializer(shape, dtype), name=name, trainable=False)

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
      initial_value = tf.zeros(
          shape=model_variable.shape, dtype=model_variable.dtype)
    return tf.Variable(
        initial_value=initial_value,
        name=f"{variable_name}/{model_variable._shared_name}",  # pylint: disable=protected-access
        dtype=model_variable.dtype,
        trainable=False)

  def minimize(self, loss, var_list, tape=None):
    """Minimize `loss` by updating `var_list`.

    This method simply computes gradient using `tf.GradientTape` and calls
    `apply_gradients()`. If you want to process the gradient before applying
    then call `tf.GradientTape` and `apply_gradients()` explicitly instead
    of using this function.

    Args:
      loss: `Tensor` or callable. If a callable, `loss` should take no arguments
        and return the value to minimize.
      var_list: list or tuple of `Variable` objects to update to minimize
        `loss`.
      tape: (Optional) `tf.GradientTape`.

    Returns:
      None
    """
    grads_and_vars = self.compute_gradients(loss, var_list, tape)
    self.apply_gradients(grads_and_vars)

  def apply_gradients(self, grads_and_vars):
    """Apply gradients to variables.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.

    Returns:
      None

    Raises:
      TypeError: If `grads_and_vars` is malformed.
    """
    if isinstance(self._learning_rate,
                  learning_rate_schedule.LearningRateSchedule):
      # Compute the current learning rate at the beginning of variable update.
      self._current_learning_rate.assign(self._learning_rate(self.iterations))
    grads_and_vars = optimizer_utils.filter_empty_gradients(grads_and_vars)
    grads, trainable_variables = zip(*grads_and_vars)
    scope_name = self._name or "optimizer"
    with tf.name_scope(scope_name):
      with tf.init_scope():
        # Lift variable creation to init scope to avoid environment issues.
        self.build(trainable_variables)
    grads = self._clip_gradients(grads)
    grads_and_vars = list(zip(grads, trainable_variables))
    self._internal_apply_gradients(grads_and_vars)

  def _internal_apply_gradients(self, grads_and_vars):
    """Helper function of apply gradients.

    This is required for separating out distributed training logic.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.
    """
    update_step = self.update_step
    if self._jit_compile:
      update_step = tf.function(update_step, jit_compile=True)
    for grad, var in grads_and_vars:
      update_step(grad, var)
    self.iterations.assign_add(1)

  def _update_model_variables_moving_average(self, var_list):
    """Update the stored moving average using the latest value."""
    if self._use_ema:
      for (var, average) in zip(var_list, self._model_variables_moving_average):
        average.assign(self._ema_momentum * average +
                       (1 - self._ema_momentum) * var)

  def _overwrite_model_variables_with_average_value(self, var_list):
    """Overwrite model variables with its moving average."""
    if len(var_list) != len(self._model_variables_moving_average):
      raise ValueError(f"The length of model variables ({len(var_list)}) to "
                       f"override does not match the length of model variables "
                       f"stored in the optimizer "
                       f"({len(self._model_variables_moving_average)}). Please "
                       f"check if the optimizer was called on your model.")
    self._overwrite_model_variables_with_average_value_helper(var_list)

  def _overwrite_model_variables_with_average_value_helper(self, var_list):
    """Helper function that overwrites model variables."""
    for var, average_var in zip(var_list, self._model_variables_moving_average):
      var.assign(average_var)

  def finalize_variable_values(self, var_list):
    """Set the final value of model's trainable variables.

    Sometimes there are some extra steps before ending the variable updates,
    such as overriding the model variables with its average value.

    Args:
      var_list: list of model variables.
    """
    self._overwrite_model_variables_with_average_value(var_list)

  def _serialize_hyperparameter(self, hyperparameter):
    """Serialize a hyperparameter that can be a numeric or callable."""
    if isinstance(hyperparameter, learning_rate_schedule.LearningRateSchedule):
      return learning_rate_schedule.serialize(hyperparameter)
    if isinstance(hyperparameter, tf.Variable):
      return hyperparameter.numpy()
    if callable(hyperparameter):
      return hyperparameter()
    return hyperparameter

  def get_config(self):
    """Returns the config of the optimizer.

    An optimizer config is a Python dictionary (serializable)
    containing the configuration of an optimizer.
    The same optimizer can be reinstantiated later
    (without any saved state) from this configuration.

    Subclass optimizer should override this method to include other
    hyperparameters.

    Returns:
        Python dictionary.
    """
    config = {
        "clipnorm": self._clipnorm,
        "global_clipnorm": self._global_clipnorm,
        "clipvalue": self._clipvalue,
        "use_ema": self._use_ema,
        "ema_momentum": self._ema_momentum,
        "ema_overwrite_frequency": self._ema_overwrite_frequency,
        "jit_compile": self._jit_compile,
    }
    return config

  @classmethod
  def from_config(cls, config):
    """Creates an optimizer from its config.

    This method is the reverse of `get_config`, capable of instantiating the
    same optimizer from the config dictionary.

    Args:
        config: A Python dictionary, typically the output of get_config.

    Returns:
        An optimizer instance.
    """
    if "learning_rate" in config:
      if isinstance(config["learning_rate"], dict):
        config["learning_rate"] = learning_rate_schedule.deserialize(
            config["learning_rate"])
    return cls(**config)


class Optimizer(_BaseOptimizer):
  """Abstract optimizer base class.

  This class supports distributed training. If you want to implement your own
  optimizer, please subclass this class instead of _BaseOptimizer.

  Attributes:
    name: string. The name to use for momentum accumulator weights created by
      the optimizer.
    clipnorm: float. If set, the gradient of each weight is individually
      clipped so that its norm is no higher than this value.
    clipvalue: float. If set, the gradient of each weight is clipped to be no
      higher than this value.
    global_clipnorm: float. If set, the gradient of all weights is clipped so
      that their global norm is no higher than this value.
    use_ema: boolean, default to False. If True, exponential moving average
      (EMA) is applied. EMA consists of computing an exponential moving
      average of the weights of the model (as the weight values change after
      each training batch), and periodically overwriting the weights with
      their moving average.
    ema_momentum: float, default to 0.99. Only used if `use_ema=True`. This is
      the momentum to use when computing the EMA of the model's weights:
        `new_average = ema_momentum * old_average + (1 - ema_momentum) *
        current_variable_value`.
    ema_overwrite_frequency: int or None, default to 100. Only used if
      `use_ema=True`. Every ema_overwrite_frequency steps of iterations, we
      overwrite the model variable by its stored moving average. If None, we
      do not overwrite model variables in the middle of training, and users
      need to explicitly overwrite the model variable by calling
      `finalize_variable_update()`.
    jit_compile: bool, default to False. If True, the optimizer will use XLA
      acceleration. `jit_compile` can only be False when using Parameter
      Server Strategy.
    **kwargs: keyword arguments only used for backward compatibility with
      `optimizer_v2.OptimizerV2`. Any new code using
      `optimizer_experimental.Optimizer` should leave this parameter empty.
  """

  def __init__(self,
               name,
               clipnorm=None,
               clipvalue=None,
               global_clipnorm=None,
               use_ema=False,
               ema_momentum=0.99,
               ema_overwrite_frequency=100,
               jit_compile=False,
               **kwargs):
    """Create a new Optimizer."""

    super().__init__(name, clipnorm, clipvalue, global_clipnorm, use_ema,
                     ema_momentum, ema_overwrite_frequency, jit_compile,
                     **kwargs)
    self._distribution_strategy = tf.distribute.get_strategy()

  def add_variable_from_reference(self,
                                  model_variable,
                                  variable_name,
                                  initial_value=None):
    """Create an optimizer variable.

    Create an optimizer variable based on the information of model variable.
    The created optimizer variable will have the same shape and dtype as the
    model variable, and placed at the same device.

    Args:
      model_variable: The corresponding model variable to the optimizer variable
        to be created.
      variable_name: The name prefix of the optimizer variable to be created.
      initial_value: The initial value of the optimizer variable, if None, the
        value will be default to 0.

    Returns:
      An optimizer variable.
    """
    strategy = tf.distribute.get_strategy()
    with strategy.extended.colocate_vars_with(model_variable):
      return super(Optimizer,
                   self).add_variable_from_reference(model_variable,
                                                     variable_name,
                                                     initial_value)

  def _var_key(self, variable):
    """Get a unique identifier of the given variable."""
    # pylint: disable=protected-access
    # Get the distributed variable if it exists.
    # TODO(b/197554203): replace _distributed_container() with a public api.
    if hasattr(variable, "_distributed_container"):
      variable = variable._distributed_container()
    return super(Optimizer, self)._var_key(variable)

  def aggregate_gradients(self, grads_and_vars):
    """Aggregate gradients on all devices.

    By default we will perform reduce_sum of gradients across devices. Users can
    implement their own aggregation logic by overriding this method.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.

    Returns:
      List of (gradient, variable) pairs.
    """
    return optimizer_utils.all_reduce_sum_gradients(grads_and_vars)

  def apply_gradients(self, grads_and_vars, skip_gradients_aggregation=False):
    """Apply gradients to variables.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.
      skip_gradients_aggregation: If true, gradients aggregation will not be
        performed inside optimizer. Usually this arg is set to True when you
        write custom code aggregating gradients outside the optimizer.

    Returns:
      None

    Raises:
      TypeError: If `grads_and_vars` is malformed.
      RuntimeError: If called in a cross-replica context.
    """
    if not skip_gradients_aggregation:
      grads_and_vars = self.aggregate_gradients(grads_and_vars)
    super().apply_gradients(grads_and_vars)

  def _internal_apply_gradients(self, grads_and_vars):
    tf.__internal__.distribute.interim.maybe_merge_call(
        self._distributed_apply_gradients_fn, self._distribution_strategy,
        grads_and_vars)

  def _overwrite_model_variables_with_average_value_helper(self, var_list):
    """Helper function to _overwrite_model_variables_with_average_value.

    This function overwrites variables on each device.
    Args:
      var_list: list of model variables.
    """
    strategy = self._distribution_strategy
    # Override model variable by the stored average value on all devices.
    for var, average_var in zip(var_list, self._model_variables_moving_average):
      strategy.extended.update(
          var, lambda a, b: a.assign(b), args=(average_var,))

  def _update_model_variables_moving_average(self, var_list):
    """Update the stored moving average using the latest value."""
    if self._use_ema:
      def update_average(average, var):
        average.assign(self._ema_momentum * average +
                       (1 - self._ema_momentum) * var)

      for (var, average) in zip(var_list, self._model_variables_moving_average):
        self._distribution_strategy.extended.update(
            average, update_average, args=(var,), group=False)

  def _distributed_apply_gradients_fn(self, distribution, grads_and_vars,
                                      **kwargs):
    """`apply_gradients` using a `DistributionStrategy`."""

    def apply_grad_to_update_var(var, grad):
      update_step = self.update_step
      if self._jit_compile:
        update_step = tf.function(update_step, jit_compile=True)
      return update_step(grad, var)

    for grad, var in grads_and_vars:
      distribution.extended.update(
          var, apply_grad_to_update_var, args=(grad,), group=False)
    self.iterations.assign_add(1)

    if self._use_ema:
      _, var_list = zip(*grads_and_vars)
      self._update_model_variables_moving_average(var_list)
      if self._ema_overwrite_frequency:
        # Only when self._ema_overwrite_frequency is not None, we overwrite the
        # model variables.
        should_overwrite_model_vars = (
            self.iterations % self._ema_overwrite_frequency == 0)
        tf.cond(
            should_overwrite_model_vars,
            true_fn=lambda: self._overwrite_model_variables_with_average_value(  # pylint: disable=g-long-lambda
                var_list),
            false_fn=lambda: None)


class RestoredOptimizer(Optimizer):

  def __init__(self):
    super(RestoredOptimizer, self).__init__("RestoredOptimizer")

  def get_config(self):
    raise NotImplementedError(
        "Restoring functional Optimizers from SavedModels is not currently "
        "supported. Please file a feature request if this limitation bothers "
        "you.")


# Register the optimizer for loading from saved_model purpose.
tf.__internal__.saved_model.load.register_revived_type(
    "experimentalOptimizer",
    lambda obj: isinstance(obj, Optimizer),
    versions=[
        tf.__internal__.saved_model.load.VersionedTypeRegistration(
            object_factory=lambda proto: RestoredOptimizer(),
            version=2,
            min_producer_version=1,
            min_consumer_version=1)
    ])
