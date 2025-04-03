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
"""Tests for methods in `optax.transforms._accumulation.py`."""

from absl.testing import absltest
import chex
import flax
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import alias
from optax._src import combine
from optax._src import transform
from optax._src import update
from optax.transforms import _accumulation
from optax.transforms import _constraining


class Loss(flax.linen.Module):

  @flax.linen.compact
  def __call__(self, x, dtype=jnp.float32, param_dtype=jnp.float32):
    return jnp.sum(
        flax.linen.Dense(10, dtype=dtype, param_dtype=param_dtype)(x) ** 2
    )


class AccumulationTest(chex.TestCase):

  @chex.all_variants
  def test_ema(self):
    values = jnp.array([5.0, 7.0])
    decay = 0.9
    d = decay

    ema = _accumulation.ema(decay=decay, debias=False)
    state = ema.init(values[0])  # init to zeroes

    transform_fn = self.variant(ema.update)
    mean, state = transform_fn(values[0], state)
    np.testing.assert_allclose(mean, (1 - d) * values[0], atol=1e-4)

    mean, _ = transform_fn(values[1], state)
    np.testing.assert_allclose(
        mean, (1 - d) * (values[1] + d * values[0]), atol=1e-2
    )

  @chex.all_variants
  def test_ema_debias(self):
    values = jnp.array([5.0, 7.0])
    decay = 0.9
    d = decay

    ema = _accumulation.ema(decay=decay)
    state = ema.init(values[0])

    transform_fn = self.variant(ema.update)
    mean, state = transform_fn(values[0], state)
    np.testing.assert_allclose(mean, values[0], atol=1e-4)

    mean, state = transform_fn(values[1], state)
    np.testing.assert_allclose(
        mean,
        ((1 - d) * values[1] + d * (1 - d) * values[0]) / (1 - d**2),
        atol=1e-2,
    )
    # The state must not be debiased.
    np.testing.assert_allclose(
        state.ema, (1 - d) * values[1] + d * (1 - d) * values[0], atol=1e-2
    )

  def test_skip_not_finite(self):
    step = jnp.zeros([], dtype=jnp.int32)

    with self.subTest('test_pos_inf'):
      should_skip, skip_state = _accumulation.skip_not_finite(
          [jnp.array(float('inf')), jnp.zeros([])], step, None
      )
      self.assertTrue(bool(should_skip))
      self.assertTrue(bool(skip_state['should_skip']))
      self.assertEqual(int(skip_state['num_not_finite']), 1)

    with self.subTest('test_neg_inf'):
      should_skip, skip_state = _accumulation.skip_not_finite(
          [jnp.array(-float('inf')), jnp.zeros([])], step, None
      )
      self.assertTrue(bool(should_skip))
      self.assertTrue(bool(skip_state['should_skip']))
      self.assertEqual(int(skip_state['num_not_finite']), 1)

    with self.subTest('test_nan'):
      should_skip, skip_state = _accumulation.skip_not_finite(
          [jnp.array(float('nan')), jnp.zeros([])], step, None
      )
      self.assertTrue(bool(should_skip))
      self.assertTrue(bool(skip_state['should_skip']))
      self.assertEqual(int(skip_state['num_not_finite']), 1)

    with self.subTest('test_finite'):
      should_skip, skip_state = _accumulation.skip_not_finite(
          [jnp.array(11.0), jnp.zeros([])], step, None
      )
      self.assertFalse(bool(should_skip))
      self.assertFalse(bool(skip_state['should_skip']))
      self.assertEqual(int(skip_state['num_not_finite']), 0)

  def test_skip_large_updates(self):
    step = jnp.zeros([], dtype=jnp.int32)

    with self.subTest('test_inf'):
      should_skip, skip_state = _accumulation.skip_large_updates(
          [jnp.array(float('inf')), jnp.zeros([])], step, None, 100.0
      )
      self.assertTrue(bool(should_skip))
      self.assertTrue(bool(skip_state['should_skip']))
      self.assertEqual(float(skip_state['norm_squared']), float('inf'))

    with self.subTest('test_nan'):
      should_skip, skip_state = _accumulation.skip_large_updates(
          [jnp.array(float('nan')), jnp.zeros([])], step, None, 100.0
      )
      self.assertTrue(bool(should_skip))
      self.assertTrue(bool(skip_state['should_skip']))
      # Recall that NaN != NaN.
      norm_squared = float(skip_state['norm_squared'])
      self.assertNotEqual(norm_squared, norm_squared)

    with self.subTest('test_large'):
      should_skip, skip_state = _accumulation.skip_large_updates(
          [jnp.array(11.0), jnp.zeros([])], step, None, 100.0
      )
      self.assertTrue(bool(should_skip))
      self.assertTrue(bool(skip_state['should_skip']))
      self.assertEqual(float(skip_state['norm_squared']), 121.0)

    with self.subTest('test_small'):
      should_skip, skip_state = _accumulation.skip_large_updates(
          [jnp.zeros([]), jnp.zeros([])], step, None, 100.0
      )
      self.assertFalse(bool(should_skip))
      self.assertFalse(bool(skip_state['should_skip']))
      self.assertEqual(float(skip_state['norm_squared']), 0.0)

  @chex.variants(with_jit=True, without_jit=True, with_pmap=True)
  def test_multi_steps(self):
    batch_size = 32
    x_size = 7
    # Parameters should be updated only every `k_steps` optimization steps.
    k_steps = 4
    data = jnp.ones([batch_size, x_size])
    loss = Loss()

    params = loss.init({'params': jax.random.PRNGKey(0)}, data)['params']

    def loss_apply(params, data):
      return loss.apply({'params': params}, data)

    ms_opt = _accumulation.MultiSteps(
        # Use a non-trivial inner optimizer:
        # * it has a state,
        # * it requires the params for the update.
        combine.chain(
            transform.scale_by_adam(),
            transform.add_decayed_weights(1e-2),
            transform.scale(-1e-4),
        ),
        k_steps,
    )

    opt_init, opt_update = ms_opt.gradient_transformation()

    # Put the training in one function, to check that the update is indeed
    # jittable.
    def train_step(data, opt_state, params):
      grad = jax.grad(loss_apply)(params, data)
      updates, opt_state = opt_update(grad, opt_state, params)
      return updates, opt_state

    opt_state = opt_init(params)

    prev_loss = loss_apply(params, data)
    for idx in range(5 * k_steps):
      updates, opt_state = self.variant(train_step)(data, opt_state, params)
      new_params = update.apply_updates(params, updates)
      new_loss = loss_apply(new_params, data)
      if idx % k_steps < k_steps - 1:
        # The parameters should not have changed and the loss should be
        # constant.
        jax.tree.map(np.testing.assert_array_equal, new_params, params)
        np.testing.assert_equal(new_loss, prev_loss)
        self.assertFalse(ms_opt.has_updated(opt_state))
      else:
        # This is a step where parameters should actually have been updated, and
        # the loss should accordingly go down.
        np.testing.assert_array_less(new_loss, prev_loss)
        prev_loss = new_loss
        self.assertTrue(ms_opt.has_updated(opt_state))
      params = new_params

  def test_multi_steps_every_k_schedule(self):
    # Test a non-trivial schedule which varies over time.
    ms_opt = _accumulation.MultiSteps(
        alias.sgd(1e-4), lambda grad_step: jnp.where(grad_step < 2, 1, 3)
    )
    opt_init, opt_update = ms_opt.gradient_transformation()
    params = dict(a=jnp.zeros([]))
    opt_state = opt_init(params)
    grad = dict(a=jnp.zeros([]))
    self.assertFalse(ms_opt.has_updated(opt_state))
    # First two steps have 1 mini-step per update.
    for _ in range(2):
      _, opt_state = opt_update(grad, opt_state, params)
      self.assertTrue(ms_opt.has_updated(opt_state))
    # Subsequently, mini-steps should have 3 mini-steps per update.
    for _ in range(5):
      for _ in range(2):
        _, opt_state = opt_update(grad, opt_state, params)
        self.assertFalse(ms_opt.has_updated(opt_state))
      _, opt_state = opt_update(grad, opt_state, params)
      self.assertTrue(ms_opt.has_updated(opt_state))

  def test_multi_steps_zero_nans(self):
    # Test that MultiStep is compatible with zero_nans
    # https://github.com/google-deepmind/optax/issues/828
    ms_opt = _accumulation.MultiSteps(
        combine.chain(_constraining.zero_nans(), alias.sgd(1e-4)),
        every_k_schedule=2,
    )
    opt_init, opt_update = ms_opt.gradient_transformation()
    params = dict(a=jnp.zeros([]))
    opt_state = opt_init(params)
    grad = dict(a=jnp.zeros([]))
    opt_update(grad, opt_state, params)

  def test_multi_steps_computes_mean(self):
    k_steps = 4
    ms_opt = _accumulation.MultiSteps(
        transform.scale(1.0), k_steps, use_grad_mean=True
    )
    opt_init, opt_update = ms_opt.gradient_transformation()
    params = dict(a=jnp.zeros([]))
    opt_state = opt_init(params)
    grads = [dict(a=jnp.ones([]) * i) for i in [1, 2, 3, 4]]
    self.assertFalse(ms_opt.has_updated(opt_state))

    # First 3 steps don't update.
    for grad in grads[:-1]:
      _, opt_state = opt_update(grad, opt_state, params)
      self.assertFalse(ms_opt.has_updated(opt_state))

    # Actual update.
    new_params, opt_state = opt_update(grads[-1], opt_state, params)
    self.assertTrue(ms_opt.has_updated(opt_state))
    np.testing.assert_array_equal(new_params['a'], 2.5)

  def test_multi_steps_skip_not_finite(self):
    k_steps = 2
    ms_opt = _accumulation.MultiSteps(
        alias.sgd(1.0),
        k_steps,
        should_skip_update_fn=_accumulation.skip_not_finite,
    )
    opt_init, opt_update = ms_opt.gradient_transformation()
    opt_init = jax.jit(opt_init)
    opt_update = jax.jit(opt_update)
    params = dict(a=jnp.zeros([]))
    opt_state = opt_init(params)

    with self.subTest('test_good_updates'):
      updates, opt_state = opt_update(dict(a=jnp.ones([])), opt_state, params)
      self.assertEqual(int(opt_state.mini_step), 1)
      params = update.apply_updates(params, updates)
      updates, opt_state = opt_update(dict(a=jnp.ones([])), opt_state, params)
      self.assertEqual(int(opt_state.mini_step), 0)
      params = update.apply_updates(params, updates)
      np.testing.assert_array_equal(params['a'], jnp.negative(jnp.ones([])))

    with self.subTest('test_inf_updates'):
      updates, opt_state = opt_update(
          dict(a=jnp.array(float('inf'))), opt_state, params
      )
      self.assertEqual(int(opt_state.mini_step), 0)  # No increase in mini_step
      params = update.apply_updates(params, updates)
      np.testing.assert_array_equal(params['a'], jnp.negative(jnp.ones([])))

    with self.subTest('test_nan_updates'):
      updates, opt_state = opt_update(
          dict(a=jnp.full([], float('nan'))), opt_state, params
      )
      self.assertEqual(int(opt_state.mini_step), 0)  # No increase in mini_step
      params = update.apply_updates(params, updates)
      np.testing.assert_array_equal(params['a'], jnp.negative(jnp.ones([])))

    with self.subTest('test_final_good_updates'):
      updates, opt_state = opt_update(dict(a=jnp.ones([])), opt_state, params)
      self.assertEqual(int(opt_state.mini_step), 1)
      params = update.apply_updates(params, updates)
      updates, opt_state = opt_update(dict(a=jnp.ones([])), opt_state, params)
      self.assertEqual(int(opt_state.mini_step), 0)
      params = update.apply_updates(params, updates)
      np.testing.assert_array_equal(
          params['a'], jnp.negative(jnp.full([], 2.0))
      )

  def test_multi_steps_mixed_precision(self):
    batch_size = 32
    x_size = 7
    k_steps = 4
    data = jnp.ones([batch_size, x_size])
    loss = Loss()

    def loss_apply(params, data):
      return loss.apply({'params': params}, data)

    # Compare optimizer dtypes with and without MultiSteps
    def create_optimizer(k_steps):
      base_opt = combine.chain(
          transform.scale_by_adam(),
          transform.add_decayed_weights(1e-2),
          transform.scale(-1e-4),
      )
      ms_opt = _accumulation.MultiSteps(base_opt, k_steps)
      return base_opt, ms_opt

    opt, ms_opt = create_optimizer(k_steps)
    ms_opt_init, ms_opt_update = ms_opt.gradient_transformation()

    # General train step
    def train_step(data, opt_state, params, opt_update_fn, upd_dtype):
      grad = jax.grad(loss_apply)(params, data)
      grad = jax.tree.map(
          lambda x: x.astype(upd_dtype), grad
      )  # mimic varying dtype
      updates, opt_state = opt_update_fn(grad, opt_state, params)
      return updates, opt_state

    # Utility for dtype comparison
    def compare_dtypes(tree1, tree2):
      return jax.tree_util.tree_all(
          jax.tree_util.tree_map(lambda x, y: x.dtype == y.dtype, tree1, tree2)
      )

    # Iterate over parameter and update dtypes
    dtypes = [jnp.float32, jnp.float16, jnp.bfloat16]
    for upd_dtype in dtypes:
      for param_dtype in dtypes:
        with self.subTest(
            f'upd_dtype={upd_dtype.__name__}-param_dtype={param_dtype.__name__}'
        ):
          # Initialize parameters with current combination of dtypes
          params = loss.init(
              {'params': jax.random.PRNGKey(0)}, data, param_dtype=param_dtype
          )['params']
          opt_state = opt.init(params)
          ms_opt_state = ms_opt_init(params)

          for _ in range(k_steps + 1):
            updates, opt_state = train_step(
                data, opt_state, params, opt.update, upd_dtype
            )
            ms_updates, ms_opt_state = train_step(
                data, ms_opt_state, params, ms_opt_update, upd_dtype
            )
            self.assertTrue(compare_dtypes(updates, ms_updates))
            new_params = update.apply_updates(params, ms_updates)
            self.assertTrue(compare_dtypes(params, new_params))
            params = new_params


if __name__ == '__main__':
  absltest.main()
