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
"""Implementation for GSAM"""

import copy

from keras.engine import data_adapter
from keras.layers import deserialize as deserialize_layer
from keras.models import Model
from keras.utils import generic_utils
import tensorflow.compat.v2 as tf
from keras.models.sharpness_aware_minimization import SharpnessAwareMinimization

# pylint: disable=g-classes-have-attributes


@generic_utils.register_keras_serializable()
class SharpnessAwareMinimization(SharpnessAwareMinimization):
  """GSAM training flow.
  Surrogate Gap Guided Sharpness Aware Minimization (GSAM) is an improvement over Sharpness-Aware-Minimization (SAM)
  Args:
    model: `tf.keras.Model` instance. The inner model that does the
      forward-backward pass.
    rho_max: float, defaults to 0.05. The max gradients scaling factor.
    rho_min: float, defaults to 0.01. The min gradients scaling factor.
    lr_max: float, defaults to 0.001. The max learning rate.
    lr_min: float, defaults to 0.0, the min learning rate.
    eps: float, defaults to 1e-12, avoid division by 0.
    name: string, defaults to None. The name of the SAM model.
  Reference:
    [Juntang Zhuang et al., 2022](https://openreview.net/pdf?id=edONMAnhLu-)
  """

  def __init__(self, model, rho_max=0.05, rho_min=0.01, lr_max=0.001, lr_min=0.0, eps=1e-12, name=None):
    super().__init__(name=name)
    self.model = model
    self.rho_max = rho_max
    self.rho_min = rho_min
    self.lr_max = lr_max
    self.lr_min = lr_min
    self.eps = eps

  def train_step(self, data):
    """The logic of one GSAM training step.
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

    # update rho according to schedule
    lr = self.optimizer.lr
    rho = self.rho_min + (self.rho_max - self.rho_min) * (lr - self.lr_min) / (self.lr_max - self.lr_min)
    
    # calculate perturbation amplitude
    epsilon_w_cache = []
    gradients_order2_norm = self._gradients_order2_norm(gradients)
    scale = rho / (gradients_order2_norm + self.eps)

    # perturb weights, get "perturbed" gradient
    for (gradient, variable) in zip(gradients, trainable_variables):
      epsilon_w = gradient * scale
      variable.assign_add(epsilon_w)
      epsilon_w_cache.append(epsilon_w)

    with tf.GradientTape() as tape:
      pred = self(x)
      loss = self.compiled_loss(y, pred)

    gradients_perturbed = tape.gradient(loss, trainable_variables)
    
    # 
    for (variable, epsilon_w) in zip(trainable_variables, epsilon_w_cache):
      # Restore the variable to its original value before `apply_gradients()`.
      variable.assign_sub(epsilon_w)

    self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    self.compiled_metrics.update_state(y, pred, sample_weight)
    return {m.name: m.result() for m in self.metrics}

  def get_config(self):
    config = super().get_config()
    config.update({
        "model": generic_utils.serialize_keras_object(self.model),
        "rho_max": self.rho_max,
        "rho_min": self.rho_min,
        "lr_max": self.lr_max,
        "lr_min": self.lr_min,
        "eps": self.eps,
    })
    return config
  
