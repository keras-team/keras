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
"""Tests for methods in `optax.transforms._constraining.py`."""

from absl.testing import absltest
import chex
import jax.numpy as jnp
from optax._src import combine
from optax._src import transform
from optax._src import update
from optax.transforms import _accumulation
from optax.transforms import _constraining


STEPS = 50
LR = 1e-2


class ConstraintsTest(chex.TestCase):

  def test_keep_params_nonnegative(self):
    grads = (
        jnp.array([500.0, -500.0, 0.0]),
        jnp.array([500.0, -500.0, 0.0]),
        jnp.array([500.0, -500.0, 0.0]),
    )

    params = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
        jnp.array([0.0, 0.0, 0.0]),
    )

    # vanilla sgd
    opt = combine.chain(
        _accumulation.trace(decay=0, nesterov=False), transform.scale(-LR)
    )
    opt_state = opt.init(params)

    updates, _ = opt.update(grads, opt_state, params)
    new_params = update.apply_updates(params, updates)

    chex.assert_trees_all_close(
        new_params,
        (
            jnp.array([-6.0, 4.0, -1.0]),
            jnp.array([-4.0, 6.0, 1.0]),
            jnp.array([-5.0, 5.0, 0.0]),
        ),
    )

    # sgd with keeping parameters non-negative
    opt = combine.chain(
        _accumulation.trace(decay=0, nesterov=False),
        transform.scale(-LR),
        _constraining.keep_params_nonnegative(),
    )
    opt_state = opt.init(params)

    updates, _ = opt.update(grads, opt_state, params)
    new_params = update.apply_updates(params, updates)

    chex.assert_trees_all_close(
        new_params,
        (
            jnp.array([0.0, 4.0, 0.0]),
            jnp.array([0.0, 6.0, 1.0]),
            jnp.array([0.0, 5.0, 0.0]),
        ),
    )

  @chex.all_variants
  def test_zero_nans(self):
    params = (jnp.zeros([3]), jnp.zeros([3]), jnp.zeros([3]))

    opt = _constraining.zero_nans()
    opt_state = self.variant(opt.init)(params)
    update_fn = self.variant(opt.update)

    chex.assert_trees_all_close(
        opt_state, _constraining.ZeroNansState((jnp.array(False),) * 3)
    )

    # Check an upate with nans
    grads_with_nans = (
        jnp.ones([3]),
        jnp.array([1.0, float('nan'), float('nan')]),
        jnp.array([float('nan'), 1.0, 1.0]),
    )
    updates, opt_state = update_fn(grads_with_nans, opt_state)
    chex.assert_trees_all_close(
        opt_state,
        _constraining.ZeroNansState(
            (jnp.array(False), jnp.array(True), jnp.array(True))
        ),
    )
    chex.assert_trees_all_close(
        updates,
        (jnp.ones([3]), jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 1.0])),
    )

    # Check an upate with nans and infs
    grads_with_nans_infs = (
        jnp.ones([3]),
        jnp.array([1.0, float('nan'), float('nan')]),
        jnp.array([float('inf'), 1.0, 1.0]),
    )
    updates, opt_state = update_fn(grads_with_nans_infs, opt_state)
    chex.assert_trees_all_close(
        opt_state,
        _constraining.ZeroNansState(
            (jnp.array(False), jnp.array(True), jnp.array(False))
        ),
    )
    chex.assert_trees_all_close(
        updates,
        (
            jnp.ones([3]),
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([float('inf'), 1.0, 1.0]),
        ),
    )

    # Check an update with only good values
    grads = (jnp.ones([3]), jnp.ones([3]), jnp.ones([3]))
    updates, opt_state = update_fn(grads, opt_state)
    chex.assert_trees_all_close(
        opt_state,
        _constraining.ZeroNansState(
            (jnp.array(False), jnp.array(False), jnp.array(False))
        ),
    )
    chex.assert_trees_all_close(updates, grads)

  def test_none_arguments(self):
    tf = _constraining.keep_params_nonnegative()
    state = tf.init(jnp.array([1.0, 2.0, 3.0]))
    with self.assertRaises(ValueError):
      tf.update(jnp.array([1.0, 2.0, 3.0]), state, None)


if __name__ == '__main__':
  absltest.main()
