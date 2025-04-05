# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for methods in `complex_valued.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import transform
from optax._src import update
from optax.contrib import _complex_valued


def _loss_fun_complex_to_real(z):
  return (z.conj() * z).real.sum()


def _loss_fun_real_to_real(params):
  x, y = params
  return _loss_fun_complex_to_real(x + y * 1j)


class ComplexValuedTest(chex.TestCase):

  @chex.all_variants
  @parameterized.named_parameters([
      ('adam', transform.scale_by_adam),
      ('param_block_norm', transform.scale_by_param_block_norm),
  ])
  def test_split_real_and_imaginary(self, scaler_constr):

    def do_update(loss_fun, optimizer, params, opt_state):
      loss, grads = jax.value_and_grad(loss_fun)(params)
      # Complex gradients need to be conjugated before being added to parameters
      grads = jax.tree.map(lambda x: x.conj(), grads)
      updates, opt_state = self.variant(optimizer.update)(
          grads, opt_state, params
      )
      params = update.apply_updates(params, updates)
      return loss, grads, params, opt_state

    x = jnp.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    y = jnp.array([[0.5, -0.5, 0], [0.1, 0.3, -0.2]])
    z = x + y * 1j

    optimizer = scaler_constr()
    optimizer_complex = _complex_valued.split_real_and_imaginary(optimizer)
    opt_state = self.variant(optimizer.init)((x, y))
    opt_state_complex = self.variant(optimizer_complex.init)(z)

    # Check that the loss, the gradients, and the parameters are the same for
    # real-to-real and complex-to-real loss functions in each step
    for _ in range(3):
      loss, (gx, gy), (x, y), opt_state = do_update(
          _loss_fun_real_to_real, optimizer, (x, y), opt_state
      )
      loss_complex, gz, z, opt_state_complex = do_update(
          _loss_fun_complex_to_real, optimizer_complex, z, opt_state_complex
      )
      np.testing.assert_allclose(loss, loss_complex)
      np.testing.assert_allclose(gx + gy * 1j, gz)
      np.testing.assert_allclose(x + y * 1j, z)


if __name__ == '__main__':
  absltest.main()
