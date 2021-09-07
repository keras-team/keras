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
"""Base class of optimizer."""

import tensorflow as tf


class BaseOptimizer(tf.__internal__.tracking.AutoTrackable):
  """Abstract optimizer base class, which does not support distributed training."""

  def __init__(self, **kwargs):
    raise NotImplementedError

  def _populate_kwargs(self, kwargs):
    raise NotImplementedError

  def _var_key(self, variable):
    """Get a unique identifier of the given variable."""
    raise NotImplementedError

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

  def _compute_gradients(self, loss, var_list, tape=None):
    raise NotImplementedError

  def _clip_gradients(self, grads):
    raise NotImplementedError

  @property
  def iterations(self):
    """Variable. The number of training steps this Optimizer has run."""
    raise NotImplementedError

  @property
  def learning_rate(self):
    raise NotImplementedError

  @learning_rate.setter
  def learning_rate(self, learning_rate):
    raise NotImplementedError

  def build(self, var_list):
    """Initialize the optimizer's variables, such as momemtum and velocity."""
    raise NotImplementedError

  def add_variable(self, model_variable, variable_name, initial_value=None):
    """Helper function to create optimizer variable."""
    raise NotImplementedError

  def minimize(self, loss, var_list, tape=None):
    """Minimize `loss` by updating `var_list`."""
    raise NotImplementedError

  def apply_gradients(self, grads_and_vars):
    """Apply gradients to variables.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.

    Returns:
      None

    Raises:
      TypeError: If `grads_and_vars` is malformed.
      ValueError: If none of the variables have gradients.
      RuntimeError: If called in a cross-replica context.
    """
    raise NotImplementedError

  def _internal_apply_gradients(self, grads_and_vars):
    """Helper function of apply gradients."""
    raise NotImplementedError

  def _serialize_hyperparameter(self, hyperparameter):
    """Serialize a hyperparameter that can be a float, callable, or Tensor."""
    raise NotImplementedError

  def get_config(self):
    """Returns the config of the optimizer.

    An optimizer config is a Python dictionary (serializable)
    containing the configuration of an optimizer.
    The same optimizer can be reinstantiated later
    (without any saved state) from this configuration.

    Returns:
        Python dictionary.
    """
    raise NotImplementedError

  @classmethod
  def from_config(cls, config):
    """Creates an optimizer from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same optimizer from the config
    dictionary.

    Args:
        config: A Python dictionary, typically the output of get_config.

    Returns:
        An optimizer instance.
    """
    raise NotImplementedError


class Optimizer(BaseOptimizer):
  """Abstract optimizer base class.

  This class supports distributed training.
  """

  def __init__(self, **kwargs):
    super(Optimizer, self).__init__(**kwargs)
    raise NotImplementedError

  def add_variable(self, model_variable, variable_name, initial_value=None):
    raise NotImplementedError

  def _aggregate_gradient(self, grads_and_vars):
    raise NotImplementedError

  def apply_gradients(self, grads_and_vars):
    """Override to support distributed training."""
    raise NotImplementedError

  def _internal_apply_gradients(self, grads_and_vars):
    raise NotImplementedError

  def _distributed_apply(self, distribution, grads_and_vars, params=None):
    """`apply_gradients` using a `DistributionStrategy`."""
    raise NotImplementedError


class RestoredOptimizer(Optimizer):

  def __init__(self):
    super(RestoredOptimizer, self).__init__()
    raise NotImplementedError

  def get_config(self):
    raise NotImplementedError


# Register the optimizer for loading from saved_model purpose.
tf.__internal__.saved_model.load.register_revived_type(
    "optimizerV3",
    lambda obj: isinstance(obj, Optimizer),
    versions=[
        tf.__internal__.saved_model.load.VersionedTypeRegistration(
            object_factory=lambda proto: RestoredOptimizer(),
            version=2,
            min_producer_version=1,
            min_consumer_version=1)
    ])
