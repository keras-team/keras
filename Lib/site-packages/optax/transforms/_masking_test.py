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
"""Tests for methods in `optax.transforms._masking.py`."""

import copy
import dataclasses
from typing import cast

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import transform
from optax._src import update
from optax.transforms import _masking
from optax.tree_utils import _state_utils


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


class MaskedTest(chex.TestCase):
  """Tests for the masked wrapper."""

  def test_tree_map_params(self):
    params = {
        'a': {
            'b': (jnp.zeros((1, 2)), jnp.zeros((2, 2))),
        },
        'c': {
            'd': jnp.zeros((1, 2)),
            'e': (jnp.zeros((1, 2)), jnp.zeros((1, 2))),
        },
    }

    sharding_axes = {
        'a': {
            'b': (1, 2),
        },
        'c': {
            'd': 1,
            'e': (1, 2),
        },
    }

    mask = {
        'a': {
            'b': (True, False),
        },
        'c': {
            'd': True,
            'e': (False, True),
        },
    }

    expected = {
        'a': {
            'b': (jnp.ones((1, 2)), jnp.zeros((2, 2))),
        },
        'c': {
            'd': jnp.ones((1, 2)),
            'e': (jnp.ones((1, 2)), jnp.ones((1, 2))),
        },
    }

    def init_fn(params):
      return {'count': 1, 'params': params, 'params_copy': params}

    def update_fn(updates, state, params=None):
      del params
      return updates, state

    inner = base.GradientTransformation(init_fn, update_fn)
    masked = _masking.masked(inner, mask)

    def increment_dim_1(v):
      return v + 1 if v.shape[0] == 1 else v

    # For this optimizer, tree_map_params should have the same effect on a
    # masked optimizer state as it does on an unmasked optimizer state.
    with self.subTest('inner'):
      state = inner.init(params)
      result = _state_utils.tree_map_params(inner, increment_dim_1, state)
      chex.assert_trees_all_equal(result, inner.init(expected))

    with self.subTest('masked'):
      state = masked.init(params)
      result = _state_utils.tree_map_params(masked, increment_dim_1, state)
      chex.assert_trees_all_equal(result, masked.init(expected))

    with self.subTest('masked_with_extra_args'):
      # Users wishing to pass additional arguments with the same tree structure
      # as the original params pytree will need to add the additional `is_leaf`
      # callable. This makes it possible to ignore the masked parts of the
      # pytree.

      # Replace all non-masked parameters in the opt-state tree with the
      # sharding axis values given in the tree above. Everything else is set to
      # None.
      new_state = _state_utils.tree_map_params(
          masked,
          lambda p, axis: None if isinstance(p, _masking.MaskedNode) else axis,
          state,
          sharding_axes,
          is_leaf=lambda v: isinstance(v, _masking.MaskedNode),
          transform_non_params=lambda v: None,
      )

      sharded_params = {
          'a': {
              'b': (1, None),
          },
          'c': {
              'd': 1,
              'e': (None, 2),
          },
      }

      # Required to make pytype happy
      new_state = cast(_masking.MaskedState, new_state)

      chex.assert_equal(None, new_state.inner_state['count'])
      chex.assert_equal(sharded_params, new_state.inner_state['params'])
      chex.assert_equal(sharded_params, new_state.inner_state['params_copy'])

  @chex.all_variants
  @parameterized.named_parameters(
      ('sgd', _build_sgd, False),
      ('stateful_sgd', _build_stateful_sgd, False),
      ('sgd_w_mask_fn', _build_sgd, True),
      ('stateful_sgd_w_mask_fn', _build_stateful_sgd, True),
  )
  def test_masked(self, opt_builder, use_fn):
    mask = {'a': True, 'b': [False, True], 'c': {'d': True, 'e': (False, True)}}
    mask_arg = lambda _: mask if use_fn else mask
    params = {'a': 1.0, 'b': [2.0, 3.0], 'c': {'d': 4.0, 'e': (5.0, 6.0)}}
    params = jax.tree.map(jnp.asarray, params)
    input_updates = jax.tree.map(lambda x: x / 10.0, params)

    # Negate the updates wherever the mask is True
    def masked_negate(updates):
      return jax.tree.map(lambda upd, m: -upd if m else upd, updates, mask)

    correct_updates = masked_negate(input_updates)

    init_fn, update_fn = _masking.masked(opt_builder(), mask_arg)
    update_fn = self.variant(update_fn)
    state = self.variant(init_fn)(params)

    with self.subTest('tree_map_params'):
      result = _state_utils.tree_map_params(init_fn, lambda v: v, state)
      chex.assert_trees_all_equal_structs(result, state)

    updates, state = update_fn(input_updates, state, params)
    chex.assert_trees_all_close(updates, correct_updates)

    # Check repeated application, this time with no params.
    correct_updates = masked_negate(correct_updates)
    updates, _ = update_fn(updates, state)
    chex.assert_trees_all_close(updates, correct_updates)

  @chex.all_variants
  @parameterized.named_parameters(
      ('sgd', _build_sgd),
      ('stateful_sgd', _build_stateful_sgd),
  )
  def test_prefix_mask(self, opt_builder):
    """Test when the mask is a prefix of the updates PyTree."""
    mask = {'a': True, 'b': False, 'c': {'d': False, 'e': True}}
    params = {
        'a': 1.0,
        'b': {'f': 2.0},
        'c': {'d': 3.0, 'e': ([4.0, 5.0], 6.0)},
    }
    params = jax.tree.map(jnp.asarray, params)
    input_updates = jax.tree.map(lambda x: x / 10.0, params)

    # Negate the updates wherever the mask (or mask parent) is True
    def _masked_sgd_on_updates(m, upd):
      return jax.tree.map(lambda x: -x, upd) if m else upd

    correct_updates = jax.tree.map(_masked_sgd_on_updates, mask, input_updates)

    init_fn, update_fn = _masking.masked(opt_builder(), mask)
    update_fn = self.variant(update_fn)
    state = self.variant(init_fn)(params)
    updates, state = update_fn(input_updates, state, params)
    chex.assert_trees_all_close(updates, correct_updates)

    # Check repeated application, this time with no params.
    correct_updates = jax.tree.map(
        _masked_sgd_on_updates, mask, correct_updates
    )
    updates, _ = update_fn(updates, state)
    chex.assert_trees_all_close(updates, correct_updates)

  @chex.all_variants
  def test_update_requires_params(self):
    weight_decay = 0.1
    mask = {'a': True, 'b': [False, True], 'c': {'d': True, 'e': (False, True)}}
    params = {'a': 1.0, 'b': [2.0, 3.0], 'c': {'d': 4.0, 'e': (5.0, 6.0)}}
    params = jax.tree.map(jnp.asarray, params)
    input_updates = jax.tree.map(lambda x: x / 10.0, params)

    correct_updates = jax.tree.map(
        lambda m, u, p: u + weight_decay * p if m else u,
        mask,
        input_updates,
        params,
    )

    init_fn, update_fn = _masking.masked(
        transform.add_decayed_weights(weight_decay), mask
    )
    update_fn = self.variant(update_fn)

    state = self.variant(init_fn)(params)
    updates, state = update_fn(input_updates, state, params)
    chex.assert_trees_all_close(updates, correct_updates)

    params = update.apply_updates(params, updates)

    # Test repeated application
    new_correct_updates = jax.tree.map(
        lambda m, u, p: u + weight_decay * p if m else u,
        mask,
        correct_updates,
        params,
    )
    updates, _ = update_fn(correct_updates, state, params)
    chex.assert_trees_all_close(updates, new_correct_updates)

  @parameterized.parameters(list, tuple, dict)
  def test_empty(self, container):
    init_fn, update_fn = _masking.masked(_build_sgd(), container())
    update_fn(container(), init_fn(container()))

  @parameterized.parameters(
      (False, False), (False, True), (True, False), (True, True)
  )
  def test_tree_mismatch_fails(self, extra_key_in_mask, use_fn):
    mask = {'a': True, 'b': [False, True], 'c': {'d': True, 'e': (False, True)}}
    mask_arg = lambda _: mask if use_fn else mask
    params = {'a': 1.0, 'b': [2.0, 3.0], 'c': {'d': 4.0, 'e': (5.0, 6.0)}}
    params = jax.tree.map(jnp.asarray, params)

    if extra_key_in_mask:
      mask['c']['extra'] = True
    else:
      params['c']['extra'] = 7

    init_fn = _masking.masked(_build_sgd(), mask_arg)[0]
    with self.assertRaises(ValueError):
      init_fn(params)

  @chex.all_variants
  def test_mask_fn(self):
    params = {'a': jnp.ones((1, 2)), 'b': (jnp.ones((1,)), np.ones((1, 2, 3)))}
    mask_fn = lambda p: jax.tree.map(lambda x: x.ndim > 1, p)
    init_fn, update_fn = _masking.masked(
        transform.add_decayed_weights(0.1), mask_fn
    )
    update_fn = self.variant(update_fn)

    state = self.variant(init_fn)(params)
    grads = jax.tree.map(lambda x: x * 2, params)
    updates, _ = update_fn(grads, state, params)
    np.testing.assert_allclose(updates['a'], grads['a'] + 0.1 * params['a'])
    np.testing.assert_allclose(updates['b'][0], grads['b'][0])
    np.testing.assert_allclose(
        updates['b'][1], grads['b'][1] + 0.1 * params['b'][1]
    )

  @chex.all_variants
  @parameterized.named_parameters(
      ('sgd', _build_sgd),
      ('stateful_sgd', _build_stateful_sgd),
  )
  def test_nested_mask(self, opt_builder):
    # https://github.com/deepmind/optax/issues/271
    params = {
        'linear_1': {'w': jnp.zeros((1, 1)), 'b': jnp.zeros(1)},
        'linear_2': {'w': jnp.zeros((1, 2)), 'b': jnp.zeros(2)},
        'linear_3': {'w': jnp.zeros((2, 3)), 'b': jnp.zeros(3)},
    }

    outer_mask = lambda p: jax.tree.map(lambda x: x.ndim > 1, p)
    inner_mask = jax.tree.map(lambda _: True, params)
    inner_mask['linear_2'] = False

    inner = _masking.masked(opt_builder(), inner_mask)
    init_fn, update_fn = _masking.masked(inner, outer_mask)

    input_updates = jax.tree.map(jnp.ones_like, params)
    correct_updates = copy.deepcopy(input_updates)
    correct_updates['linear_1']['w'] *= -1.0
    correct_updates['linear_3']['w'] *= -1.0

    state = self.variant(init_fn)(params)
    updates, _ = self.variant(update_fn)(input_updates, state, params)
    chex.assert_trees_all_close(updates, correct_updates)

  @chex.all_variants
  def test_masked_state_structure(self):
    # https://github.com/deepmind/optax/issues/271
    params = {
        'a': [jnp.ones(1), (jnp.ones(2), jnp.ones(3))],
        'b': {'c': jnp.ones(4), 'd': jnp.ones(5)},
    }
    mask = {'a': [True, (True, False)], 'b': False}
    tx = _masking.masked(_build_stateful_sgd(), mask)
    trace = self.variant(tx.init)(params).inner_state[0].trace
    expected_trace = {
        'a': [jnp.zeros(1), (jnp.zeros(2), _masking.MaskedNode())],
        'b': _masking.MaskedNode(),
    }
    chex.assert_trees_all_equal_structs(trace, expected_trace)

  def test_mask_compatible_callable_pytree(self):
    """Test if mask is compatible with callable pytrees a la equinox."""
    # https://github.com/google-deepmind/optax/issues/913

    # Define a dataclass with a callable
    @dataclasses.dataclass
    class _MaskCompatibleModule:
      w: jax.Array

      def __call__(self, x):
        return self.w.dot(x)

    # Make it a pytree by registering it as a pytree node
    jtu.register_pytree_node(
        _MaskCompatibleModule,
        lambda m: ((m.w,), None),
        lambda _, c: _MaskCompatibleModule(*c),
    )
    module = _MaskCompatibleModule(jnp.array([1.0, 2.0, 3.0]))

    with self.subTest('test _mask_callable utility'):
      # The module is then callable (like the Module class in equinox)
      self.assertTrue(callable(module))
      # But it won't be treated as a callable by the masking logic
      self.assertFalse(_masking._mask_callable(module))

    with self.subTest('test masked transform on callable pytree'):
      mask = _MaskCompatibleModule(False)
      opt = _masking.masked(_build_sgd(), mask)
      state = opt.init(module)
      updates = module
      new_updates, _ = opt.update(updates, state)
      chex.assert_trees_all_equal(updates, new_updates)


if __name__ == '__main__':
  absltest.main()
