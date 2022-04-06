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
"""Sharpness Aware Minimization implementation."""

import copy

from keras.engine import data_adapter
from keras.layers import deserialize as deserialize_layer
from keras.models import Model
from keras.utils import generic_utils
import tensorflow.compat.v2 as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util.tf_export import keras_export

# pylint: disable=g-classes-have-attributes


@generic_utils.register_keras_serializable()
@keras_export("keras.models.experimental.SharpnessAwareMinimization", v1=[])
class SharpnessAwareMinimization(Model):
  """Sharpness aware minimization (SAM) training flow.

  Sharpness-aware minimization (SAM) is a technique that improves the model
  generalization and provides robustness to label noise.

  Args:
    model: `tf.keras.Model` instance. The inner model that does the
      forward-backward pass.
    rho: float, defaults to 0.05. The gradients scaling factor.
    name: string, defaults to None. The name of the SAM model.

  Reference:
    [Pierre Foret et al., 2020](https://arxiv.org/abs/2010.01412)
  """

  def __init__(self, model, rho=0.05, name=None):
    super().__init__(name=name)
    self.model = model
    self.rho = rho

  def train_step(self, data):
    """The logic of one SAM training step.

    Args:
      data: A nested structure of `Tensor`s. It should be of structure
        (x, y, sample_weight) or (x, y).

    Returns:
      A dict mapping metric names to running average values.
    """
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
    with tf.GradientTape() as tape:
      pred = self.model(x)
      loss = self.compiled_loss(y, pred)
    trainable_variables = self.model.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)

    epsilon_w_cache = []
    gradients_order2_norm = self._gradients_order2_norm(gradients)
    scale = self.rho / (gradients_order2_norm + 1e-12)

    for (gradient, variable) in zip(gradients, trainable_variables):
      epsilon_w = gradient * scale
      variable.assign_add(epsilon_w)
      epsilon_w_cache.append(epsilon_w)

    with tf.GradientTape() as tape:
      pred = self(x)
      loss = self.compiled_loss(y, pred)

    gradients = tape.gradient(loss, trainable_variables)
    for (variable, epsilon_w) in zip(trainable_variables, epsilon_w_cache):
      # Restore the variable to its original value before `apply_gradients()`.
      variable.assign_sub(epsilon_w)

    self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    self.compiled_metrics.update_state(y, pred, sample_weight)
    return {m.name: m.result() for m in self.metrics}

  def call(self, inputs):
    """Forward pass of SAM.

    SAM delegates the forward pass call to the wrapped model.

    Args:
      inputs: Tensor. The model inputs.

    Returns:
      A Tensor, the outputs of the wrapped model for given `inputs`.
    """
    return self.model(inputs)

  def get_config(self):
    config = super().get_config()
    config.update({
        "model": generic_utils.serialize_keras_object(self.model),
        "rho": self.rho,
    })
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    # Avoid mutating the input dict.
    config = copy.deepcopy(config)
    model = deserialize_layer(
        config.pop("model"), custom_objects=custom_objects)
    return cls(model, **config)

  def _gradients_order2_norm(self, gradients):
    norm = tf.norm(
        tf.stack([tf.norm(grad) for grad in gradients if grad is not None]))
    return norm
