# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for the SAM optimizer in `sam.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from optax._src import alias
from optax._src import combine
from optax._src import numerics
from optax._src import update
from optax.contrib import _sam
from optax.tree_utils import _state_utils

_BASE_OPTIMIZERS_UNDER_TEST = [
    dict(base_opt_name='sgd', base_opt_kwargs=dict(learning_rate=1e-3)),
]
_ADVERSARIAL_OPTIMIZERS_UNDER_TEST = [
    dict(adv_opt_name='sgd', adv_opt_kwargs=dict(learning_rate=1e-5)),
    dict(adv_opt_name='adam', adv_opt_kwargs=dict(learning_rate=1e-4)),
]


def _setup_parabola(dtype):
  """Quadratic function as an optimization target."""
  initial_params = jnp.array([-1.0, 10.0, 1.0], dtype=dtype)
  final_params = jnp.array([1.0, -1.0, 1.0], dtype=dtype)

  @jax.grad
  def get_updates(params):
    return jnp.sum(numerics.abs_sq(params - final_params))

  return initial_params, final_params, get_updates


class SAMTest(chex.TestCase):

  @parameterized.product(
      _BASE_OPTIMIZERS_UNDER_TEST,
      _ADVERSARIAL_OPTIMIZERS_UNDER_TEST,
      sync_period=(2,),
      target=(_setup_parabola,),
      dtype=('float32',),
      opaque_mode=(False, True),
  )
  def test_optimization(
      self,
      base_opt_name,
      base_opt_kwargs,
      adv_opt_name,
      adv_opt_kwargs,
      sync_period,
      target,
      dtype,
      opaque_mode,
  ):
    dtype = jnp.dtype(dtype)
    base_opt = getattr(alias, base_opt_name)(**base_opt_kwargs)
    adv_opt = combine.chain(
        _sam.normalize(), getattr(alias, adv_opt_name)(**adv_opt_kwargs)
    )
    opt = _sam.sam(
        base_opt, adv_opt, sync_period=sync_period, opaque_mode=opaque_mode
    )
    initial_params, final_params, get_updates = target(dtype)

    if opaque_mode:
      update_kwargs = dict(grad_fn=lambda p, _: get_updates(p))
    else:
      update_kwargs = {}

    @jax.jit
    def step(params, state):
      updates = get_updates(params)
      updates, state = opt.update(updates, state, params, **update_kwargs)
      params = update.apply_updates(params, updates)
      return params, state

    params = initial_params
    state = opt.init(params)
    # A no-op change, to verify that tree map works.
    state = _state_utils.tree_map_params(opt, lambda v: v, state)

    for _ in range(25000 * sync_period):
      params, state = step(params, state)

    chex.assert_trees_all_close(params, final_params, rtol=3e-2, atol=3e-2)


if __name__ == '__main__':
  absltest.main()
