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
"""Tests for methods in `optax.tree_utils._state_utils.py`."""

import dataclasses
from typing import Optional, TypedDict, cast

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import transform
from optax.schedules import _inject
from optax.schedules import _schedule
from optax.tree_utils import _state_utils


@dataclasses.dataclass
class FakeShardSpec:
  sharding_axis: Optional[int]


class ScaleByAdamStateDict(TypedDict):
  """An opt state that uses dictionaries instead of classes."""

  count: chex.Array
  params: TypedDict('Params', {'mu': chex.ArrayTree, 'nu': chex.ArrayTree})


def _scale_by_adam_with_dicts():
  """An implementation of adam using dictionary-based opt states."""

  t = transform.scale_by_adam()

  def init(params):
    state = t.init(params)
    state = cast(transform.ScaleByAdamState, state)

    return ScaleByAdamStateDict(
        count=state.count,
        params={'mu': state.mu, 'nu': state.nu},
    )

  def update(updates, state, params=None):
    state = transform.ScaleByAdamState(
        count=state['count'],
        mu=state['params']['mu'],
        nu=state['params']['nu'],
    )

    _, state = t.update(updates, state, params)
    state = cast(transform.ScaleByAdamState, state)
    return ScaleByAdamStateDict(
        count=state.count,
        params={'mu': state.mu, 'nu': state.nu},
    )

  return base.GradientTransformation(init, update)


class StateUtilsTest(absltest.TestCase):

  def test_dict_based_optimizers(self):
    """Test we can map over params also for optimizer states using dicts."""
    opt = combine.chain(
        _scale_by_adam_with_dicts(),
        transform.add_decayed_weights(1e-3),
    )

    params = _fake_params()
    params_sharding_spec = _fake_param_sharding()
    opt_state = opt.init(params)

    opt_state_sharding_spec = _state_utils.tree_map_params(
        opt,
        lambda _, spec: spec,
        opt_state,
        params_sharding_spec,
        transform_non_params=lambda _: FakeShardSpec(None),
    )

    expected = (
        {
            'count': FakeShardSpec(sharding_axis=None),
            'params': {
                'mu': {
                    'my/fake/module': {
                        'b': FakeShardSpec(sharding_axis=1),
                        'w': FakeShardSpec(sharding_axis=0),
                    },
                    'my/other/fake/module': {
                        'b': FakeShardSpec(sharding_axis=3),
                        'w': FakeShardSpec(sharding_axis=2),
                    },
                },
                'nu': {
                    'my/fake/module': {
                        'b': FakeShardSpec(sharding_axis=1),
                        'w': FakeShardSpec(sharding_axis=0),
                    },
                    'my/other/fake/module': {
                        'b': FakeShardSpec(sharding_axis=3),
                        'w': FakeShardSpec(sharding_axis=2),
                    },
                },
            },
        },
        base.EmptyState(),
    )

    self.assertEqual(expected, opt_state_sharding_spec)

  def test_state_chex_dataclass(self):
    @chex.dataclass
    class Foo:
      count: int
      v: chex.ArrayTree

    def init(params):
      return Foo(count=0, v=params)

    params = {
        'w': 0,
    }

    state = init(params)
    state = _state_utils.tree_map_params(init, lambda v: v + 1, state)
    state = cast(Foo, state)

    self.assertEqual(int(state.count), 0)
    self.assertEqual(state.v, {'w': jnp.array(1)})

  def test_adam(self):
    params = _fake_params()
    params_sharding_spec = _fake_param_sharding()

    opt = alias.adam(1e-4)
    opt_state = opt.init(params)

    opt_state_sharding_spec = _state_utils.tree_map_params(
        opt,
        lambda _, spec: spec,
        opt_state,
        params_sharding_spec,
        transform_non_params=lambda _: FakeShardSpec(None),
    )

    expected = (
        transform.ScaleByAdamState(  # pytype:disable=wrong-arg-types
            count=FakeShardSpec(sharding_axis=None),
            mu={
                'my/fake/module': {
                    'w': FakeShardSpec(sharding_axis=0),
                    'b': FakeShardSpec(sharding_axis=1),
                },
                'my/other/fake/module': {
                    'w': FakeShardSpec(sharding_axis=2),
                    'b': FakeShardSpec(sharding_axis=3),
                },
            },
            nu={
                'my/fake/module': {
                    'w': FakeShardSpec(sharding_axis=0),
                    'b': FakeShardSpec(sharding_axis=1),
                },
                'my/other/fake/module': {
                    'w': FakeShardSpec(sharding_axis=2),
                    'b': FakeShardSpec(sharding_axis=3),
                },
            },
        ),
        base.EmptyState(),
    )

    self.assertEqual(expected, opt_state_sharding_spec)

  def test_inject_hparams(self):
    opt = _inject.inject_hyperparams(alias.adamw)(learning_rate=1e-3)

    params = _fake_params()
    state = opt.init(params)
    state = _state_utils.tree_map_params(opt, lambda v: v + 1, state)
    state = cast(_inject.InjectHyperparamsState, state)

    self.assertEqual(1e-3, state.hyperparams['learning_rate'])
    params_plus_one = jax.tree.map(lambda v: v + 1, params)
    mu = getattr(state.inner_state[0], 'mu')
    chex.assert_trees_all_close(mu, params_plus_one)

  def test_map_params_to_none(self):
    opt = alias.adagrad(1e-4)

    params = {'a': jnp.zeros((1, 2))}
    state = opt.init(params)
    state = _state_utils.tree_map_params(opt, lambda _: None, state)
    self.assertEqual(
        state,
        (
            transform.ScaleByRssState(sum_of_squares={'a': None}),
            base.EmptyState(),
        ),
    )

  def test_map_non_params_to_none(self):
    """Test for dangerous edge-cases in tree when returning None values."""

    opt = alias.adam(_schedule.linear_schedule(1e-2, 1e-4, 10))

    params = {'a': jnp.zeros((1, 2))}
    state = opt.init(params)

    state = _state_utils.tree_map_params(
        opt, lambda v: 1, state, transform_non_params=lambda _: None
    )

    expected = (
        transform.ScaleByAdamState(  # pytype:disable=wrong-arg-types
            count=None,
            mu={'a': 1},
            nu={'a': 1},
        ),
        transform.ScaleByScheduleState(  # pytype:disable=wrong-arg-types
            count=None
        ),
    )
    self.assertEqual(state, expected)

  def test_tree_get_all_with_path(self):
    params = jnp.array([1.0, 2.0, 3.0])

    with self.subTest('Test with flat tree'):
      tree = ()
      found_values = _state_utils.tree_get_all_with_path(tree, 'foo')
      self.assertEmpty(found_values)
      tree = jnp.array([1.0, 2.0, 3.0])
      found_values = _state_utils.tree_get_all_with_path(tree, 'foo')
      self.assertEmpty(found_values)

    with self.subTest('Test with single value in state'):
      key = 'count'
      opt = transform.scale_by_adam()
      state = opt.init(params)
      found_values = _state_utils.tree_get_all_with_path(state, key)
      expected_result = [(
          (_state_utils.NamedTupleKey('ScaleByAdamState', 'count'),),
          jnp.array(0.0),
      )]
      self.assertEqual(found_values, expected_result)

    with self.subTest('Test with no value in state'):
      key = 'apple'
      opt = alias.adam(learning_rate=1.0)
      state = opt.init(params)
      found_values = _state_utils.tree_get_all_with_path(state, key)
      self.assertEmpty(found_values)

    with self.subTest('Test with multiple values in state'):
      key = 'learning_rate'
      opt = combine.chain(
          _inject.inject_hyperparams(alias.sgd)(learning_rate=1.0),
          combine.chain(
              alias.adam(learning_rate=1.0),
              _inject.inject_hyperparams(alias.adam)(learning_rate=1e-4),
          ),
      )
      state = opt.init(params)
      found_values = _state_utils.tree_get_all_with_path(state, key)
      expected_result = [
          (
              (
                  jtu.SequenceKey(idx=0),
                  _state_utils.NamedTupleKey(
                      'InjectStatefulHyperparamsState', 'hyperparams'
                  ),
                  jtu.DictKey(key='learning_rate'),
              ),
              jnp.array(1.0),
          ),
          (
              (
                  jtu.SequenceKey(idx=1),
                  jtu.SequenceKey(idx=1),
                  _state_utils.NamedTupleKey(
                      'InjectStatefulHyperparamsState', 'hyperparams'
                  ),
                  jtu.DictKey(key='learning_rate'),
              ),
              jnp.array(1e-4),
          ),
      ]
      self.assertEqual(found_values, expected_result)

    with self.subTest('Test with optional filtering'):
      state = dict(hparams=dict(learning_rate=1.0), learning_rate='foo')

      # Without filtering two values are found
      found_values = _state_utils.tree_get_all_with_path(state, 'learning_rate')
      self.assertLen(found_values, 2)

      # With filtering only the float entry is returned
      filtering = lambda _, value: isinstance(value, float)
      found_values = _state_utils.tree_get_all_with_path(
          state, 'learning_rate', filtering=filtering
      )
      self.assertLen(found_values, 1)
      expected_result = [(
          (jtu.DictKey(key='hparams'), jtu.DictKey(key='learning_rate')),
          1.0,
      )]
      self.assertEqual(found_values, expected_result)

    with self.subTest('Test to get a subtree (here hyperparams_states)'):
      opt = _inject.inject_hyperparams(alias.sgd)(learning_rate=lambda x: x)
      filtering = lambda _, value: isinstance(value, tuple)
      state = opt.init(params)
      found_values = _state_utils.tree_get_all_with_path(
          state, 'learning_rate', filtering=filtering
      )
      expected_result = [(
          (
              _state_utils.NamedTupleKey(
                  'InjectStatefulHyperparamsState', 'hyperparams_states'
              ),
              jtu.DictKey(key='learning_rate'),
          ),
          _inject.WrappedScheduleState(
              count=jnp.array(0, dtype=jnp.dtype('int32'))
          ),
      )]
      self.assertEqual(found_values, expected_result)

    with self.subTest('Test with nested tree containing a key'):
      tree = dict(a=dict(a=1.0))
      found_values = _state_utils.tree_get_all_with_path(tree, 'a')
      expected_result = [
          ((jtu.DictKey(key='a'),), {'a': 1.0}),
          ((jtu.DictKey(key='a'), jtu.DictKey(key='a')), 1.0),
      ]
      self.assertEqual(found_values, expected_result)

  def test_tree_get(self):
    params = jnp.array([1.0, 2.0, 3.0])

    with self.subTest('Test with unique value matching the key'):
      solver = _inject.inject_hyperparams(alias.sgd)(learning_rate=42.0)
      state = solver.init(params)
      lr = _state_utils.tree_get(state, 'learning_rate')
      self.assertEqual(lr, 42.0)

    with self.subTest('Test with no value matching the key'):
      solver = _inject.inject_hyperparams(alias.sgd)(learning_rate=42.0)
      state = solver.init(params)
      ema = _state_utils.tree_get(state, 'ema')
      self.assertIsNone(ema)
      ema = _state_utils.tree_get(state, 'ema', default=7.0)
      self.assertEqual(ema, 7.0)

    with self.subTest('Test with multiple values matching the key'):
      solver = combine.chain(
          _inject.inject_hyperparams(alias.sgd)(learning_rate=42.0),
          _inject.inject_hyperparams(alias.sgd)(learning_rate=42.0),
      )
      state = solver.init(params)
      self.assertRaises(KeyError, _state_utils.tree_get, state, 'learning_rate')

    with self.subTest('Test jitted tree_get'):
      opt = _inject.inject_hyperparams(alias.sgd)(
          learning_rate=lambda x: 1 / (x + 1)
      )
      state = opt.init(params)
      filtering = lambda _, value: isinstance(value, jnp.ndarray)

      @jax.jit
      def get_learning_rate(state):
        return _state_utils.tree_get(
            state, 'learning_rate', filtering=filtering
        )

      for i in range(4):
        # we simply update state, we don't care about updates.
        _, state = opt.update(params, state)
        lr = get_learning_rate(state)
        self.assertEqual(lr, 1 / (i + 1))

    with self.subTest('Test with optional filtering'):
      state = dict(hparams=dict(learning_rate=1.0), learning_rate='foo')

      # Without filtering raises an error
      self.assertRaises(KeyError, _state_utils.tree_get, state, 'learning_rate')

      # With filtering, fetches the float entry
      filtering = lambda path, value: isinstance(value, float)
      lr = _state_utils.tree_get(state, 'learning_rate', filtering=filtering)
      self.assertEqual(lr, 1.0)

    with self.subTest('Test filtering for specific state'):
      opt = combine.chain(
          transform.add_noise(1.0, 0.9, 0), transform.scale_by_adam()
      )
      state = opt.init(params)

      filtering = (
          lambda path, _: isinstance(path[-1], _state_utils.NamedTupleKey)
          and path[-1].tuple_name == 'ScaleByAdamState'
      )

      count = _state_utils.tree_get(state, 'count', filtering=filtering)
      self.assertEqual(count, jnp.asarray(0, dtype=jnp.dtype('int32')))

    with self.subTest('Test extracting a state'):
      opt = combine.chain(
          transform.add_noise(1.0, 0.9, 0), transform.scale_by_adam()
      )
      state = opt.init(params)
      noise_state = _state_utils.tree_get(state, 'AddNoiseState')
      expected_result = transform.AddNoiseState(
          count=jnp.asarray(0),
          rng_key=jnp.array([0, 0], dtype=jnp.dtype('uint32')),
      )
      chex.assert_trees_all_equal(noise_state, expected_result)

  def test_tree_set(self):
    params = jnp.array([1.0, 2.0, 3.0])

    with self.subTest('Test with flat tree'):
      tree = ()
      self.assertRaises(KeyError, _state_utils.tree_set, tree, foo=1.0)
      tree = jnp.array([1.0, 2.0, 3.0])
      self.assertRaises(KeyError, _state_utils.tree_set, tree, foo=1.0)

    with self.subTest('Test modifying an injected hyperparam'):
      opt = _inject.inject_hyperparams(alias.adam)(learning_rate=1.0)
      state = opt.init(params)
      new_state = _state_utils.tree_set(state, learning_rate=2.0, b1=3.0)
      lr = _state_utils.tree_get(new_state, 'learning_rate')
      self.assertEqual(lr, 2.0)

    with self.subTest('Test modifying an attribute of the state'):
      opt = _inject.inject_hyperparams(alias.adam)(learning_rate=1.0)
      state = opt.init(params)
      new_state = _state_utils.tree_set(state, learning_rate=2.0, b1=3.0)
      b1 = _state_utils.tree_get(new_state, 'b1')
      self.assertEqual(b1, 3.0)

    with self.subTest('Test modifying a value not present in the state'):
      opt = _inject.inject_hyperparams(alias.adam)(learning_rate=1.0)
      state = opt.init(params)
      self.assertRaises(KeyError, _state_utils.tree_set, state, ema=2.0)

    with self.subTest('Test jitted tree_set'):

      @jax.jit
      def set_learning_rate(state, lr):
        return _state_utils.tree_set(state, learning_rate=lr)

      modified_state = state
      lr = 1.0
      for i in range(4):
        modified_state = set_learning_rate(modified_state, lr / (i + 1))
        # we simply update state, we don't care about updates.
        _, modified_state = opt.update(params, modified_state)
        modified_lr = _state_utils.tree_get(modified_state, 'learning_rate')
        self.assertEqual(modified_lr, lr / (i + 1))

    with self.subTest('Test modifying several values at once'):
      opt = combine.chain(
          alias.adam(learning_rate=1.0), alias.adam(learning_rate=1.0)
      )
      state = opt.init(params)
      new_state = _state_utils.tree_set(state, count=2.0)
      found_values = _state_utils.tree_get_all_with_path(new_state, 'count')
      self.assertLen(found_values, 2)
      for _, value in found_values:
        self.assertEqual(value, 2.0)

    with self.subTest('Test with optional filtering'):
      state = dict(hparams=dict(learning_rate=1.0), learning_rate='foo')
      filtering = lambda _, value: isinstance(value, float)
      new_state = _state_utils.tree_set(state, filtering, learning_rate=0.5)
      found_values = _state_utils.tree_get_all_with_path(
          new_state, 'learning_rate'
      )
      expected_result = [
          ((jtu.DictKey(key='learning_rate'),), 'foo'),
          ((jtu.DictKey(key='hparams'), jtu.DictKey(key='learning_rate')), 0.5),
      ]
      self.assertEqual(found_values, expected_result)

    with self.subTest('Test with nested trees and filtering'):
      tree = dict(a=dict(a=1.0), b=dict(a=1))
      filtering = lambda _, value: isinstance(value, float)
      new_tree = _state_utils.tree_set(tree, filtering, a=2.0)
      expected_result = dict(a=dict(a=2.0), b=dict(a=1))
      self.assertEqual(new_tree, expected_result)

    with self.subTest('Test setting a subtree'):
      tree = dict(a=dict(a=1.0), b=dict(a=1))
      filtering = lambda _, value: isinstance(value, dict)
      new_tree = _state_utils.tree_set(tree, filtering, a=dict(c=0.0))
      expected_result = dict(a=dict(c=0.0), b=dict(a=1))
      self.assertEqual(new_tree, expected_result)

    with self.subTest('Test setting a specific state'):
      opt = combine.chain(
          transform.add_noise(1.0, 0.9, 0), transform.scale_by_adam()
      )
      state = opt.init(params)

      filtering = (
          lambda path, _: isinstance(path[-1], _state_utils.NamedTupleKey)
          and path[-1].tuple_name == 'ScaleByAdamState'
      )

      new_state = _state_utils.tree_set(state, filtering, count=jnp.array(42))
      expected_result = (
          transform.AddNoiseState(
              count=jnp.array(0),
              rng_key=jnp.array([0, 0], dtype=jnp.dtype('uint32')),
          ),
          transform.ScaleByAdamState(
              count=jnp.array(42),
              mu=jnp.array([0.0, 0.0, 0.0]),
              nu=jnp.array([0.0, 0.0, 0.0]),
          ),
      )
      chex.assert_trees_all_equal(new_state, expected_result)

    with self.subTest('Test setting a state'):
      opt = combine.chain(
          transform.add_noise(1.0, 0.9, 0), transform.scale_by_adam()
      )
      state = opt.init(params)
      new_noise_state = transform.AddNoiseState(
          count=jnp.array(42),
          rng_key=jnp.array([4, 8], dtype=jnp.dtype('uint32')),
      )
      new_state = _state_utils.tree_set(state, AddNoiseState=new_noise_state)
      expected_result = (
          transform.AddNoiseState(
              count=jnp.array(42),
              rng_key=jnp.array([4, 8], dtype=jnp.dtype('uint32')),
          ),
          transform.ScaleByAdamState(
              count=jnp.array(0),
              mu=jnp.array([0.0, 0.0, 0.0]),
              nu=jnp.array([0.0, 0.0, 0.0]),
          ),
      )
      chex.assert_trees_all_equal(new_state, expected_result)


def _fake_params():
  return {
      'my/fake/module': {
          'w': jnp.zeros((1, 2)),
          'b': jnp.zeros((3, 4)),
      },
      'my/other/fake/module': {
          'w': jnp.zeros((1, 2)),
          'b': jnp.zeros((3, 4)),
      },
  }


def _fake_param_sharding():
  return {
      'my/fake/module': {
          'w': FakeShardSpec(0),
          'b': FakeShardSpec(1),
      },
      'my/other/fake/module': {
          'w': FakeShardSpec(2),
          'b': FakeShardSpec(3),
      },
  }


if __name__ == '__main__':
  absltest.main()
