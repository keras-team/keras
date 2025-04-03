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
"""Tests for methods in `optax.transforms._layouts.py`."""

from absl.testing import absltest
import chex
import jax.numpy as jnp
from optax._src import alias
from optax._src import update
from optax.transforms import _layouts


class LayoutsTest(absltest.TestCase):

  def test_flatten(self):
    def init_params():
      return (jnp.array(2.0), jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))

    per_step_updates = (
        jnp.array(1.0),
        jnp.array([500.0, 5.0]),
        jnp.array([300.0, 3.0]),
    )

    # First calculate new params without flattening
    optax_sgd_params = init_params()
    sgd = alias.sgd(1e-2, 0.0)
    state_sgd = sgd.init(optax_sgd_params)
    updates_sgd, _ = sgd.update(per_step_updates, state_sgd)
    sgd_params_no_flatten = update.apply_updates(optax_sgd_params, updates_sgd)

    # And now calculate new params with flattening
    optax_sgd_params = init_params()
    sgd = _layouts.flatten(sgd)

    state_sgd = sgd.init(optax_sgd_params)
    updates_sgd, _ = sgd.update(per_step_updates, state_sgd)
    sgd_params_flatten = update.apply_updates(optax_sgd_params, updates_sgd)

    # Test that both give the same result
    chex.assert_trees_all_close(
        sgd_params_no_flatten, sgd_params_flatten, atol=1e-7, rtol=1e-7
    )


if __name__ == "__main__":
  absltest.main()
