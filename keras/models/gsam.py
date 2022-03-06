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

# TF utils.
def _inner_product(vectors1: List[tf.Tensor],
                   vectors2: List[tf.Tensor]) -> float:
  """Compute inner product between vector1 and vector2."""
  return tf.math.reduce_sum(
    tf.stack([ 
      tf.math.reduce_sum(vector1*vector2) for (vector1, vector2) in zip(vectors1, vectors2) 
      if (vector1 is not None) and (vector2 is not None)
    ])
  )

def _vector_norm(vectors: List[tf.Tensor]) -> float:
  """Compute the L2 norm of vector"""
  return tf.math.sqrt( _inner_product(vectors, vectors) )

def _decompose_parallel_vertical(
    vectors1: List[tf.Tensor],
    vectors2: List[tf.Tensor],
    eps: float = 1e-12) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
  """Decompose vector2 onto parallel and vertical to vectors1."""
  inner_product = _inner_product(vectors1, vectors2)
  norm1 = _vector_norm(vectors1)
  norm2 = _vector_norm(vectors2)
  scale1 = 1.0 / (norm1 + eps)
  scale2 = 1.0 / (norm2 + eps)
  cosine = inner_product * scale1 * scale2
  parallel, vectical = [], []
  for vector1, vector2 in zip(vectors1, vectors2):
    parallel_item = vector1 * scale1 * norm2 * cosine
    parallel.append(parallel_item)
    vectical.append(vector2 - parallel_item)
  return parallel, vectical


@generic_utils.register_keras_serializable()
class GSAM(SharpnessAwareMinimization):
  """GSAM training flow.
  Surrogate Gap Guided Sharpness Aware Minimization (GSAM) is an improvement over Sharpness-Aware-Minimization (SAM)
  Args:
    model: `tf.keras.Model` instance. The inner model that does the
      forward-backward pass.
    rho_max: float, defaults to 0.05. The max gradients scaling factor.
    rho_min: float, defaults to 0.01. The min gradients scaling factor.
    lr_max: float, defaults to 0.001. The max learning rate.
    lr_min: float, defaults to 0.0. The min learning rate.
    eps: float, defaults to 1e-12.
    alpha: float, defaults to 0.05. The $\alpha$ term in GSAM
    name: string, defaults to None. The name of the GSAM model.
  Reference:
    [Juntang Zhuang et al., 2022](https://openreview.net/pdf?id=edONMAnhLu-)
  """

  def __init__(self, model, rho_max=0.05, rho_min=0.01, lr_max=0.001, lr_min=0.0, eps=1e-12, alpha=0.05, name=None):
    super().__init__(name=name)
    self.model = model
    self.rho_max = rho_max
    self.rho_min = rho_min
    self.lr_max = lr_max
    self.lr_min = lr_min
    self.eps = eps
    self.alpha = alpha

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

    # update rho according to schedule.
    lr = self.optimizer.lr
    rho = self.rho_min + (self.rho_max - self.rho_min) * (lr - self.lr_min) / (self.lr_max - self.lr_min)
    
    # calculate perturbation amplitude.
    epsilon_w_cache = []
    gradients_order2_norm = self._gradients_order2_norm(gradients)
    scale = rho / (gradients_order2_norm + self.eps)

    # perturb weights, get "perturbed" gradient.
    for (gradient, variable) in zip(gradients, trainable_variables):
      epsilon_w = gradient * scale
      variable.assign_add(epsilon_w)
      epsilon_w_cache.append(epsilon_w)

    with tf.GradientTape() as tape:
      pred = self(x)
      loss = self.compiled_loss(y, pred)

    gradients_perturbed = tape.gradient(loss, trainable_variables)
    
    # decompose gradients onto parallel and vertical to gradients_perturbed.
    parallels, verticals = _decompose_parallel_vertical(gradients_perturbed, gradients)
    
    # get GSAM update direction
    for (gradient_perturbed, vertical) in zip(gradients_perturbed, verticals):
      gradient_perturbed.assign_sub( self.alpha * vertical)
    
    # Restore the variable to its original value before `apply_gradients()`.
    for (variable, epsilon_w) in zip(trainable_variables, epsilon_w_cache):
      variable.assign_sub(epsilon_w)

    # update with `gradients_perturbed`
    self.optimizer.apply_gradients(zip(gradients_perturbed, trainable_variables))

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
        "alpha": self.alpha,
    })
    return config
  
