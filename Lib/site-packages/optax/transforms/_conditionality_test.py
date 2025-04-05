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
"""Tests for methods in `optax.transforms._conditionality.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import transform
from optax._src import update
from optax.transforms import _conditionality
from optax.transforms import _constraining


def _build_sgd():
  return alias.sgd(1.0)


def _build_stateful_sgd():
  # This SGD behaves like _build_sgd but also tests the optimizer state. The
  # momentum is set to zero rather than None so that the momentum terms are
  # calculated, but do not change the results.
  return alias.sgd(1.0, momentum=0.0)


def _build_sgd_extra_args():

  def init_fn(params):
    del params
    return {'foo': 1}

  def update_fn(grads, state, params=None, *, foo=None, **extra_args):
    del extra_args, foo, params
    return grads, state

  t = base.GradientTransformationExtraArgs(init_fn, update_fn)
  return combine.chain(_build_sgd(), t)


class ConditionalityTest(parameterized.TestCase):

  @chex.variants(with_jit=True, without_jit=True, with_pmap=True)
  @parameterized.named_parameters(
      ('sgd', _build_sgd),
      ('stateful_sgd', _build_stateful_sgd),
      ('sgd_extra_args', _build_sgd_extra_args),
  )
  def test_apply_if_finite(self, opt_builder):
    one = jnp.array(1.0)
    nan = jnp.array(jnp.nan)

    def fn(p, x):
      return p * x

    params = jnp.array(0.0)
    opt = _conditionality.apply_if_finite(opt_builder(), 2)
    state = opt.init(params)
    grads_fn = jax.grad(self.variant(fn))
    # Do one successful param update
    grads = grads_fn(params, one)
    updates, state = opt.update(grads, state, params)
    params = update.apply_updates(params, updates)
    # We know exactly what should be the value of params since we are
    # effectively using sgd in all cases.
    self.assertEqual(-1.0, float(jax.tree.flatten(params)[0][0]))
    self.assertTrue(bool(getattr(state, 'last_finite')))
    # Check 2 rejected param updates
    for step in range(2):
      grads = grads_fn(params, nan)
      updates, state = opt.update(grads, state, params)
      params = update.apply_updates(params, updates)
      self.assertEqual(-1.0, float(jax.tree.flatten(params)[0][0]))
      self.assertFalse(bool(getattr(state, 'last_finite')))
      self.assertEqual(step + 1, int(getattr(state, 'notfinite_count')))
    # Next successful param update
    grads = grads_fn(params, one)
    updates, state = opt.update(grads, state, params)
    params = update.apply_updates(params, updates)
    self.assertEqual(-2.0, float(jax.tree.flatten(params)[0][0]))
    self.assertTrue(bool(getattr(state, 'last_finite')))
    # Again 2 rejected param updates
    for step in range(2):
      grads = grads_fn(params, nan)
      updates, state = opt.update(grads, state, params)
      params = update.apply_updates(params, updates)
      self.assertEqual(-2.0, float(jax.tree.flatten(params)[0][0]))
      self.assertFalse(bool(getattr(state, 'last_finite')))
      self.assertEqual(step + 1, int(getattr(state, 'notfinite_count')))
    # Next param update with NaN is accepted since we reached maximum
    grads = grads_fn(params, nan)
    updates, state = opt.update(grads, state, params)
    params = update.apply_updates(params, updates)
    self.assertTrue(bool(jnp.isnan(jax.tree.flatten(params)[0][0])))
    self.assertEqual(5, int(getattr(state, 'total_notfinite')))

  def test_apply_if_finite_pmap(self):
    # Unlike in `test_apply_if_finite`:
    # * pmap is applied to the gradient computation and the optimization;
    # * the NaNs are caused inside the function and do not come from the inputs.
    half = jnp.ones([1]) / 2.0
    two = jnp.ones([1]) * 2.0  # Causes a NaN in arctanh

    def fn(p, x):
      return jnp.arctanh(x) * p

    opt = _conditionality.apply_if_finite(alias.sgd(1.0), 2)

    def fn_update(params, opt_state, x):
      grads = jax.grad(fn)(params, x)
      grads = jax.lax.psum(grads, axis_name='i')
      updates, new_opt_state = opt.update(grads, opt_state, params)
      new_params = update.apply_updates(params, updates)
      return new_params, new_opt_state

    fn_update = jax.pmap(fn_update, axis_name='i')

    params = jnp.array(0.0)
    opt_state = opt.init(params)
    params = jax.tree.map(lambda x: x[None], params)
    opt_state = jax.tree.map(lambda x: x[None], opt_state)
    # Do one successful param update
    params, opt_state = fn_update(params, opt_state, half)
    self.assertTrue(bool(opt_state.last_finite))
    # Check 2 rejected param updates
    for step in range(2):
      params, opt_state = fn_update(params, opt_state, two)
      self.assertFalse(bool(opt_state.last_finite))
      self.assertEqual(step + 1, opt_state.notfinite_count.item())
    # Next successful param update
    params, opt_state = fn_update(params, opt_state, half)
    self.assertTrue(bool(opt_state.last_finite))
    # Again 2 rejected param updates
    for step in range(2):
      params, opt_state = fn_update(params, opt_state, two)
      self.assertFalse(bool(opt_state.last_finite))
      self.assertEqual(step + 1, opt_state.notfinite_count.item())
    # Next param update with NaN is accepted since we reached maximum
    _, opt_state = fn_update(params, opt_state, two)
    self.assertEqual(5, opt_state.total_notfinite.item())


class ConditionallyTransformTest(chex.TestCase):
  """Tests for the conditionally_transform wrapper."""

  NUM_STEPS = 3

  @chex.all_variants
  def test_stateless_inner(self):
    params = jnp.zeros([])
    grads = jnp.ones([])

    def should_update(step):
      return step < ConditionallyTransformTest.NUM_STEPS

    opt = _conditionality.conditionally_transform(
        transform.scale(2.0), should_update
    )
    state = opt.init(params)
    update_fn = self.variant(opt.update)
    for _ in range(ConditionallyTransformTest.NUM_STEPS):
      updates, state = update_fn(grads, state)
      self.assertEqual(updates, 2.0)
    # Further updates stop calling the inner optimizer.
    for _ in range(5):
      updates, state = update_fn(grads, state)
      self.assertEqual(updates, 1.0)

  @chex.all_variants
  def test_statefull_inner(self):
    params = jnp.zeros([])
    grads_with_nan = jnp.array(float('nan'))
    grads = jnp.ones([])

    def should_update(step):
      return step < ConditionallyTransformTest.NUM_STEPS

    opt = _conditionality.conditionally_transform(
        _constraining.zero_nans(), should_update
    )
    state = opt.init(params)
    update_fn = self.variant(opt.update)
    for _ in range(ConditionallyTransformTest.NUM_STEPS - 1):
      updates, state = update_fn(grads_with_nan, state)
      self.assertEqual(updates, 0.0)
      self.assertEqual(state.inner_state.found_nan, True)
    updates, state = update_fn(grads, state)
    self.assertEqual(updates, 1.0)
    self.assertEqual(state.inner_state.found_nan, False)
    # Further updates stop calling the inner optimizer.
    for _ in range(5):
      updates, state = update_fn(grads_with_nan, state)
      # Warning: do not use assertEqual with a NaN as NaN == NaN returns False.
      self.assertTrue(jnp.isnan(updates))
      # Inner state is not be updated.
      self.assertEqual(state.inner_state.found_nan, False)


class ConditionallyMaskTest(chex.TestCase):
  """Tests for the conditionally_mask wrapper."""

  NUM_STEPS = 3
  MIN_LOSS = 0.1

  @chex.all_variants
  def test_stateless_inner(self):
    params = jnp.zeros([])
    grads = jnp.ones([])

    def should_update(step):
      return step < ConditionallyMaskTest.NUM_STEPS

    opt = _conditionality.conditionally_mask(
        transform.scale(2.0), should_update
    )
    state = opt.init(params)
    update_fn = self.variant(opt.update)
    for _ in range(ConditionallyMaskTest.NUM_STEPS):
      updates, state = update_fn(grads, state)
      self.assertEqual(updates, 2.0)
    # Further updates stop calling the inner optimizer.
    for _ in range(5):
      updates, state = update_fn(grads, state)
      self.assertEqual(updates, 0.0)

  @chex.all_variants
  def test_statefull_inner(self):
    params = jnp.zeros([])
    grads_with_nan = jnp.array(float('nan'))
    grads = jnp.ones([])

    def should_update(step):
      return step < ConditionallyMaskTest.NUM_STEPS

    opt = _conditionality.conditionally_mask(
        _constraining.zero_nans(), should_update
    )
    state = opt.init(params)
    update_fn = self.variant(opt.update)
    for _ in range(ConditionallyMaskTest.NUM_STEPS - 1):
      updates, state = update_fn(grads_with_nan, state)
      self.assertEqual(updates, 0.0)
      self.assertEqual(state.inner_state.found_nan, True)
    updates, state = update_fn(grads, state)
    self.assertEqual(updates, 1.0)
    self.assertEqual(state.inner_state.found_nan, False)
    # Further updates stop calling the inner optimizer.
    for _ in range(5):
      updates, state = update_fn(grads_with_nan, state)
      self.assertEqual(updates, 0.0)
      # Inner state is not be updated.
      self.assertEqual(state.inner_state.found_nan, False)

  @chex.all_variants
  def test_stateless_inner_with_extra_args(self):
    params = jnp.zeros([])
    grads = jnp.ones([])

    def should_update(step, loss, **extra_args):
      del step, extra_args
      return loss > ConditionallyMaskTest.MIN_LOSS

    opt = _conditionality.conditionally_mask(
        transform.scale(2.0), should_update, forward_extra_args=True
    )
    state = opt.init(params)
    update_fn = self.variant(opt.update)
    for _ in range(ConditionallyMaskTest.NUM_STEPS):
      updates, state = update_fn(grads, state, loss=0.2)
      self.assertEqual(updates, 2.0)
    # Further updates stop calling the inner optimizer.
    for _ in range(5):
      updates, state = update_fn(grads, state, loss=0.0)
      self.assertEqual(updates, 0.0)


if __name__ == '__main__':
  absltest.main()
