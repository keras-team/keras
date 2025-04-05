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
"""Tests for methods in `optax.transforms._combining.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from optax._src import alias
from optax._src import base
from optax._src import transform
from optax._src import update
from optax.transforms import _accumulation
from optax.transforms import _combining

STEPS = 50
LR = 1e-2


class CombiningTest(chex.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    self.per_step_updates = (jnp.array([500.0, 5.0]), jnp.array([300.0, 3.0]))

  @chex.all_variants
  def test_chain(self):
    transformations = [
        transform.scale_by_adam(),
        _accumulation.trace(decay=0, nesterov=False),
        transform.scale(-LR),
    ]

    # Apply updates with chain.
    chain_params = self.init_params
    chained_transforms = _combining.chain(*transformations)
    state = chained_transforms.init(chain_params)
    self.assertIsInstance(state, tuple)

    @self.variant
    def update_fn(updates, state):
      return chained_transforms.update(updates, state)

    for _ in range(STEPS):
      updates, state = update_fn(self.per_step_updates, state)
      self.assertIsInstance(state, tuple)
      chain_params = update.apply_updates(chain_params, updates)

    # Manually apply sequence of transformations.
    manual_params = self.init_params
    states = [t.init(manual_params) for t in transformations]
    for _ in range(STEPS):
      updates = self.per_step_updates
      new_states = []
      for t, s in zip(transformations, states):
        updates, state = t.update(updates, s)
        new_states.append(state)
      manual_params = update.apply_updates(manual_params, updates)
      states = new_states

    # Check equivalence.
    chex.assert_trees_all_close(manual_params, chain_params, rtol=1e-4)


def _map_keys_fn(fn):
  def map_fn(nested_dict):
    return {
        k: map_fn(v) if isinstance(v, dict) else fn(k, v)
        for k, v in nested_dict.items()
    }

  return map_fn


class ExtraArgsTest(chex.TestCase):

  def test_extra_args(self):
    def init_fn(params):
      del params
      return tuple()

    # Arguments required by a transformation should be keyword-only.
    # For example, the loss argument in this transformation.
    def update_fn(updates, state, params=None, *, loss, **extra_args):
      # Extra args should always be accepted.
      del extra_args, params
      assert loss == 1
      return updates, state

    t = base.GradientTransformationExtraArgs(init_fn, update_fn)
    result = _combining.chain(alias.adam(1e-3), t)
    self.assertIsInstance(result, base.GradientTransformationExtraArgs)

    params = {'a': 1, 'b': 2}
    state = result.init(params)
    result.update(params, state, loss=1, ignored_kwarg='hi')

  def test_extra_args_chaining(self):
    def init_fn(params):
      del params
      return {}

    def update_fn(updates, state, params=None):
      del params
      return updates, state

    # Possible gotcha: Chaining regular gradient transformations results in
    # a transformation that supports extra args.
    t1 = base.GradientTransformation(init_fn, update_fn)
    t2 = _combining.chain(t1, t1)
    self.assertIsInstance(t2, base.GradientTransformation)
    self.assertIsInstance(t2, base.GradientTransformationExtraArgs)

    t3 = base.with_extra_args_support(t2)
    self.assertIsInstance(t3, base.GradientTransformationExtraArgs)

  def test_extra_args_positional_params(self):
    def init_fn(params):
      del params
      return tuple()

    def update_fn(updates, state, params=None):
      assert params is not None
      return updates, state

    def update_fn_kwargs(updates, state, params=None, **extra_args):
      del extra_args
      assert params is not None
      return updates, state

    t1 = base.GradientTransformation(init_fn, update_fn)
    t2 = base.GradientTransformationExtraArgs(init_fn, update_fn_kwargs)
    opt = _combining.chain(t1, t2)
    params = {'a': 1, 'b': 2}
    state = opt.init(params)
    opt.update(params, state, params, ignored_kwarg='hi')
    opt.update(params, state, params=params, ignored_kwarg='hi')


class PartitionTest(chex.TestCase):
  """Tests for the partition wrapper."""

  @chex.all_variants
  @parameterized.parameters(True, False)
  def test_partition(self, use_fn):
    params = {'a1': 1.0, 'b1': 2.0, 'z1': {'a2': 3.0, 'z2': {'c1': 4.0}}}
    params = jax.tree.map(jnp.asarray, params)
    input_updates = jax.tree.map(lambda x: x / 10.0, params)
    tx_dict = {
        'a': transform.scale(-1.0),
        'b': transform.ema(0.0),  # stateful
        'c': transform.scale(2.0),
    }
    param_labels = _map_keys_fn(lambda k, _: k[0])
    if not use_fn:
      param_labels = param_labels(params)
    tx = _combining.partition(tx_dict, param_labels)
    update_fn = self.variant(tx.update)
    state = self.variant(tx.init)(params)

    correct_update_fn = _map_keys_fn(
        lambda k, v: {'a': -v, 'b': v, 'c': 2.0 * v}[k[0]]
    )

    updates, state = update_fn(input_updates, state, params)
    correct_updates = correct_update_fn(input_updates)
    chex.assert_trees_all_close(updates, correct_updates)

    # Check repeated application, this time with no params.
    correct_updates = correct_update_fn(correct_updates)
    updates, _ = update_fn(updates, state)
    chex.assert_trees_all_close(updates, correct_updates)

  def test_extra_args(self):

    class ArgNotEqual1Error(ValueError):
      """Raised when argument not set as expected."""

    def init(params):
      return {'mu': params}

    def update_with_arg(updates, state, params=None, *, arg, **extra_args):
      del params, extra_args
      if arg != 1:
        raise ArgNotEqual1Error()
      return updates, state

    def update_without_arg(updates, state, params=None):
      del params
      return updates, state

    opt_no_arg = base.GradientTransformation(init, update_without_arg)
    opt_extra_arg = base.GradientTransformationExtraArgs(init, update_with_arg)

    opt = _combining.partition(
        {
            'a': opt_no_arg,
            'b': opt_extra_arg,
        },
        ('a', 'b'),
    )

    fake_params = ({'u': jnp.array([1])}, {'v': jnp.array([1])})
    state = opt.init(fake_params)

    with self.assertRaises(TypeError):
      opt.update(fake_params, state)
    with self.assertRaises(ArgNotEqual1Error):
      opt.update(fake_params, state, arg=2, ignored_kwarg='hi')
    opt.update(fake_params, state, arg=1, ignored_kwarg='hi')

  @parameterized.parameters(list, tuple, dict)
  def test_empty(self, container):
    init_fn, update_fn = _combining.partition({0: alias.sgd(1.0)}, lambda _: 0)
    updates, _ = update_fn(container(), init_fn(container()))
    self.assertEqual(updates, container())

  @chex.all_variants
  @parameterized.parameters(
      (False, False), (False, True), (True, False), (True, True)
  )
  def test_labels_mismatch(self, use_extra_label, use_fn):
    # The labels from label_fn must be a subet of the keys for the tx.
    params = {'a': 1.0, 'b': [2.0, 3.0], 'c': {'d': 4.0, 'e': (5.0, 6.0)}}
    params = jax.tree.map(jnp.asarray, params)
    label_tree = {'a': 0, 'b': [1, 0], 'c': 1}  # prefix of params

    if use_extra_label:
      label_tree['a'] = 3

    transforms = {
        0: alias.sgd(1.0),
        1: alias.adam(1.0, b1=0.0, b2=0.0),
        2: _accumulation.trace(1.0),
    }
    init_fn, update_fn = _combining.partition(
        transforms, (lambda _: label_tree) if use_fn else label_tree
    )

    if use_extra_label:
      with self.assertRaises(ValueError):
        self.variant(init_fn)(params)
    else:
      state = self.variant(init_fn)(params)
      updates = jax.tree.map(lambda x: x / 10.0, params)
      self.variant(update_fn)(updates, state)


def scale_by_loss():
  """Scale the gradient by the absolute value of the loss."""

  def update_fn(updates, state, params, *, loss, **extra_args):
    del params, extra_args
    updates = jax.tree.map(lambda u: u / loss, updates)
    return updates, state

  return base.GradientTransformationExtraArgs(base.init_empty_state, update_fn)


class NamedChainTest(absltest.TestCase):

  def test_named_chain(self):
    tx = _combining.named_chain(
        ('scale', transform.scale(0.1)),
        ('scale_loss', scale_by_loss()),
    )

    params = {'a': jnp.ones((4,))}
    grads = params

    opt_state = tx.init(params)
    updates, _ = tx.update(grads, opt_state, params, loss=0.1)

    chex.assert_trees_all_close(updates, {'a': jnp.ones((4,))})


if __name__ == '__main__':
  absltest.main()
