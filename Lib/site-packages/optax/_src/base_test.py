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
"""Tests for base functions in `base.py`."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import base

# pylint:disable=no-value-for-parameter


class BaseTest(chex.TestCase):

  def test_typing(self):
    """Ensure that the type annotations work for the update function."""

    def f(updates, opt_state, params=None):
      del params
      return updates, opt_state

    def g(f: base.TransformUpdateFn):
      updates = np.zeros([])
      params = np.zeros([])
      opt_state = np.zeros([])

      f(updates, opt_state)
      f(updates, opt_state, params)
      f(updates, opt_state, params=params)

    g(f)

  @chex.all_variants
  def test_set_to_zero_returns_tree_of_correct_zero_arrays(self):
    """Tests that zero transform returns a tree of zeros of correct shape."""
    grads = ({'a': np.ones((3, 4)), 'b': 1.0}, np.ones((1, 2, 3)))
    updates, _ = self.variant(base.set_to_zero().update)(
        grads, base.EmptyState()
    )
    correct_zeros = ({'a': np.zeros((3, 4)), 'b': 0.0}, np.zeros((1, 2, 3)))
    chex.assert_trees_all_close(updates, correct_zeros, rtol=0)

  @chex.all_variants(with_pmap=False)
  def test_set_to_zero_is_stateless(self):
    """Tests that the zero transform returns an empty state."""
    self.assertEqual(
        self.variant(base.set_to_zero().init)(params=None), base.EmptyState()
    )


class ExtraArgsTest(chex.TestCase):

  def test_isinstance(self):
    """Locks in behavior for comparing transformations."""

    def init_fn(params):
      del params
      return {}

    def update_fn(updates, state, params=None):
      del params
      return updates, state

    t1 = base.GradientTransformation(init_fn, update_fn)
    self.assertIsInstance(t1, base.GradientTransformation)
    self.assertNotIsInstance(t1, base.GradientTransformationExtraArgs)

    t2 = base.with_extra_args_support(t1)
    self.assertIsInstance(t2, base.GradientTransformation)
    self.assertIsInstance(t2, base.GradientTransformationExtraArgs)

    with self.subTest('args_correctly_ignored'):
      state = t2.init({})
      t2.update({}, state, ignored_arg='hi')

    t3 = base.with_extra_args_support(t2)
    self.assertIsInstance(t3, base.GradientTransformation)
    self.assertIsInstance(t3, base.GradientTransformationExtraArgs)

  def test_extra_args_with_callback(self):
    """An example of using extra args to log the learning rate."""

    def init_fn(params):
      del params
      return {}

    def update_fn(updates, state, *, metrics_logger=None, **extra_args):
      del extra_args

      if metrics_logger:
        metrics_logger('learning_rate', 0.3)
      return updates, state

    t = base.GradientTransformationExtraArgs(init_fn, update_fn)

    @jax.jit
    def f(params):
      state = t.init(params)

      metrics = {}

      def metrics_logger(name, value):
        metrics[name] = value

      t.update(params, state, metrics_logger=metrics_logger)
      return metrics

    metrics = f({'a': 1})
    self.assertEqual(metrics['learning_rate'], 0.3)


class StatelessTest(chex.TestCase):
  """Tests for the stateless transformation."""

  @chex.all_variants
  def test_stateless(self):
    params = {'a': jnp.zeros((1, 2)), 'b': jnp.ones((1,))}
    updates = {'a': jnp.ones((1, 2)), 'b': jnp.full((1,), 2.0)}

    @base.stateless
    def opt(g, p):
      return jax.tree.map(lambda g_, p_: g_ + 0.1 * p_, g, p)

    state = opt.init(params)
    update_fn = self.variant(opt.update)
    new_updates, _ = update_fn(updates, state, params)
    expected_updates = {'a': jnp.ones((1, 2)), 'b': jnp.array([2.1])}
    chex.assert_trees_all_close(new_updates, expected_updates)

  @chex.all_variants
  def test_stateless_no_params(self):
    updates = {'linear': jnp.full((5, 3), 3.0)}

    @base.stateless
    def opt(g, _):
      return jax.tree.map(lambda g_: g_ * 2, g)

    state = opt.init(None)  # pytype: disable=wrong-arg-types  # numpy-scalars
    update_fn = self.variant(opt.update)
    new_updates, _ = update_fn(updates, state)
    expected_updates = {'linear': jnp.full((5, 3), 6.0)}
    chex.assert_trees_all_close(new_updates, expected_updates)

  def test_init_returns_emptystate(self):
    def weight_decay(g, p):
      return jax.tree.map(lambda g_, p_: g_ + 0.1 * p_, g, p)

    opt = base.stateless(weight_decay)
    state = opt.init(None)  # pytype: disable=wrong-arg-types  # numpy-scalars
    self.assertIsInstance(state, base.EmptyState)


class StatelessWithTreeMapTest(chex.TestCase):
  """Tests for the stateless_with_tree_map transformation."""

  @chex.all_variants
  def test_stateless_with_tree_map(self):
    params = {'a': jnp.zeros((1, 2)), 'b': jnp.ones((1,))}
    updates = {'a': jnp.ones((1, 2)), 'b': jnp.full((1,), 2.0)}

    opt = base.stateless_with_tree_map(lambda g, p: g + 0.1 * p)
    state = opt.init(params)
    update_fn = self.variant(opt.update)
    new_updates, _ = update_fn(updates, state, params)
    expected_updates = {'a': jnp.ones((1, 2)), 'b': jnp.array([2.1])}
    chex.assert_trees_all_close(new_updates, expected_updates)

  @chex.all_variants
  def test_stateless_with_tree_map_no_params(self):
    updates = {'linear': jnp.full((5, 3), 3.0)}

    opt = base.stateless_with_tree_map(lambda g, _: g * 2.0)
    state = opt.init(None)  # pytype: disable=wrong-arg-types  # numpy-scalars
    update_fn = self.variant(opt.update)
    new_updates, _ = update_fn(updates, state)
    expected_updates = {'linear': jnp.full((5, 3), 6.0)}
    chex.assert_trees_all_close(new_updates, expected_updates)

  def test_init_returns_emptystate(self):
    opt = base.stateless_with_tree_map(lambda g, p: g + 0.1 * p)
    state = opt.init(None)  # pytype: disable=wrong-arg-types  # numpy-scalars
    self.assertIsInstance(state, base.EmptyState)


if __name__ == '__main__':
  absltest.main()
