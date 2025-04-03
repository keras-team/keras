# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for methods in `hessian.py`."""

import functools

from absl.testing import absltest
import chex
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from optax.second_order import _hessian


NUM_CLASSES = 2
NUM_SAMPLES = 3
NUM_FEATURES = 4


class HessianTest(chex.TestCase):

  def setUp(self):
    super().setUp()

    self.data = np.random.rand(NUM_SAMPLES, NUM_FEATURES)
    self.labels = np.random.randint(NUM_CLASSES, size=NUM_SAMPLES)

    class MLP(nn.Module):
      """A simple multilayer perceptron model for image classification."""

      @nn.compact
      def __call__(self, x):
        # Flattens images in the batch.
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=5)(x)
        x = nn.relu(x)
        x = nn.Dense(features=NUM_CLASSES)(x)
        return x

    net = MLP()
    self.parameters = net.init({'params': jax.random.PRNGKey(0)}, self.data)[
        'params'
    ]

    def loss(params, inputs, targets):
      log_probs = net.apply({'params': params}, inputs)
      return -jnp.mean(jax.nn.one_hot(targets, NUM_CLASSES) * log_probs)

    self.loss_fn = loss

    def jax_hessian_diag(loss_fun, params, inputs, targets):
      """This is the 'ground-truth' obtained via the JAX library."""
      flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)

      def flattened_loss(flat_params):
        return loss_fun(unravel_fn(flat_params), inputs, targets)

      flat_hessian = jax.hessian(flattened_loss)(flat_params)
      return jnp.diag(flat_hessian)

    self.hessian_diag = jax_hessian_diag(
        self.loss_fn, self.parameters, self.data, self.labels
    )

  @chex.all_variants
  def test_hessian_diag(self):
    hessian_diag_fn = self.variant(
        functools.partial(_hessian.hessian_diag, self.loss_fn)
    )
    actual = hessian_diag_fn(self.parameters, self.data, self.labels)
    np.testing.assert_array_almost_equal(self.hessian_diag, actual, 5)


if __name__ == '__main__':
  absltest.main()
