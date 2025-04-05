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
"""Common tests for contributed optimizers.

Additional specific tests are implemented in additional files
"""

import functools
import inspect

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from optax import contrib
from optax._src import alias
from optax._src import combine
from optax._src import numerics
from optax._src import update
from optax.schedules import _inject
from optax.transforms import _accumulation
from optax.tree_utils import _state_utils

# Testing contributions coded as GradientTransformations
_MAIN_OPTIMIZERS_UNDER_TEST = [
    dict(opt_name='acprop', opt_kwargs=dict(learning_rate=1e-3)),
    dict(opt_name='cocob', opt_kwargs={}),
    dict(opt_name='cocob', opt_kwargs=dict(weight_decay=1e-2)),
    dict(opt_name='dadapt_adamw', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='dog', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='dowg', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='momo', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='momo_adam', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='prodigy', opt_kwargs=dict(learning_rate=1e-1)),
    dict(
        opt_name='schedule_free_sgd',
        opt_kwargs=dict(learning_rate=1e-2, warmup_steps=5000),
    ),
    dict(
        opt_name='schedule_free_adamw',
        opt_kwargs=dict(learning_rate=1e-2, warmup_steps=5000),
    ),
]
for optimizer in _MAIN_OPTIMIZERS_UNDER_TEST:
  optimizer['wrapper_name'] = None
  optimizer['wrapper_kwargs'] = None

# Testing contributions coded as wrappers
# (just with sgd as we just want the behavior of the wrapper)
_MAIN_OPTIMIZERS_UNDER_TEST += [
    dict(
        opt_name='sgd',
        opt_kwargs=dict(learning_rate=1e-1),
        wrapper_name='mechanize',
        wrapper_kwargs=dict(weight_decay=0.0),
    ),
    dict(
        opt_name='sgd',
        opt_kwargs=dict(learning_rate=1e-2),
        wrapper_name='schedule_free',
        wrapper_kwargs=dict(learning_rate=1e-2),
    ),
    dict(
        opt_name='sgd',
        opt_kwargs=dict(learning_rate=1e-3),
        wrapper_name='reduce_on_plateau',
        wrapper_kwargs={},
    ),
]

# Adding here instantiations of wrappers with any base optimizer
_BASE_OPTIMIZERS = [
    dict(opt_name='sgd', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='sgd', opt_kwargs=dict(learning_rate=1.0, momentum=0.9)),
    dict(opt_name='adam', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='adamw', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='adamax', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='adamaxw', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='adan', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='amsgrad', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='lamb', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='lion', opt_kwargs=dict(learning_rate=1.0, b1=0.99)),
    dict(opt_name='noisy_sgd', opt_kwargs=dict(learning_rate=1.0, eta=1e-4)),
    dict(opt_name='novograd', opt_kwargs=dict(learning_rate=1.0)),
    dict(
        opt_name='optimistic_gradient_descent',
        opt_kwargs=dict(learning_rate=1.0, alpha=0.7, beta=0.1),
    ),
    dict(opt_name='rmsprop', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='rmsprop', opt_kwargs=dict(learning_rate=1.0, momentum=0.9)),
    dict(opt_name='adabelief', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='radam', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='sm3', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='yogi', opt_kwargs=dict(learning_rate=1.0, b1=0.99)),
]
# TODO(harshm): make LARS and Fromage work with mechanic.
_OTHER_OPTIMIZERS_UNDER_TEST = [
    dict(
        opt_name=base_opt['opt_name'],
        opt_kwargs=base_opt['opt_kwargs'],
        wrapper_name='mechanize',
        wrapper_kwargs=dict(weight_decay=0.0),
    )
    for base_opt in _BASE_OPTIMIZERS
]

_ALL_OPTIMIZERS_UNDER_TEST = tuple(
    _MAIN_OPTIMIZERS_UNDER_TEST + _OTHER_OPTIMIZERS_UNDER_TEST
)
_MAIN_OPTIMIZERS_UNDER_TEST = tuple(_MAIN_OPTIMIZERS_UNDER_TEST)


def _get_opt_factory(opt_name):
  """Get optimizer factory."""
  if hasattr(contrib, opt_name):
    return getattr(contrib, opt_name)
  if hasattr(alias, opt_name):
    return getattr(alias, opt_name)
  raise ValueError(f'Unknown optimizer: {opt_name}')


def _wrap_opt(opt, wrapper_name, wrapper_kwargs):
  if wrapper_name == 'reduce_on_plateau':
    return combine.chain(opt, contrib.reduce_on_plateau(**wrapper_kwargs))
  else:
    return getattr(contrib, wrapper_name)(opt, **wrapper_kwargs)


def _setup_parabola(dtype):
  """Quadratic function as an optimization target."""
  initial_params = jnp.array([-1.0, 10.0, 1.0], dtype=dtype)
  final_params = jnp.array([1.0, -1.0, 1.0], dtype=dtype)

  @jax.value_and_grad
  def get_updates(params):
    return jnp.sum(numerics.abs_sq(params - final_params))

  return initial_params, final_params, get_updates


def _setup_rosenbrock(dtype):
  """Rosenbrock function as an optimization target."""
  a = 1.0
  b = 100.0

  initial_params = jnp.array([0.0, 0.0], dtype=dtype)
  final_params = jnp.array([a, a**2], dtype=dtype)

  @jax.value_and_grad
  def get_updates(params):
    return numerics.abs_sq(a - params[0]) + b * numerics.abs_sq(
        params[1] - params[0] ** 2
    )

  return initial_params, final_params, get_updates


class ContribTest(chex.TestCase):

  @parameterized.product(
      _ALL_OPTIMIZERS_UNDER_TEST,
      target=(_setup_parabola, _setup_rosenbrock),
      dtype=('float32',),
  )
  def test_optimizers(
      self,
      opt_name,
      opt_kwargs,
      wrapper_name,
      wrapper_kwargs,
      target,
      dtype,
  ):
    dtype = jnp.dtype(dtype)
    opt = _get_opt_factory(opt_name)(**opt_kwargs)
    if wrapper_name is not None:
      opt = _wrap_opt(opt, wrapper_name, wrapper_kwargs)
    initial_params, final_params, get_updates = target(dtype)

    @jax.jit
    def step(params, state):
      value, updates = get_updates(params)
      if (
          opt_name in ['momo', 'momo_adam']
          or wrapper_name == 'reduce_on_plateau'
      ):
        update_kwargs = {'value': value}
      else:
        update_kwargs = {}
      updates, state = opt.update(updates, state, params, **update_kwargs)
      params = update.apply_updates(params, updates)
      return params, state

    params = initial_params
    state = opt.init(params)
    with self.subTest('Test that tree_map_params works'):
      # A no-op change, to verify that tree map works.
      state = _state_utils.tree_map_params(opt, lambda v: v, state)

    with self.subTest('Test that optimization works'):

      def f(params_state, _):
        return step(*params_state), None

      (params, state), _ = jax.lax.scan(f, (params, state), length=30_000)

      if (
          opt_name in ['schedule_free_sgd', 'schedule_free_adamw']
          or wrapper_name == 'schedule_free'
      ):
        params = contrib.schedule_free_eval_params(state, params)
      chex.assert_trees_all_close(params, final_params, rtol=3e-2, atol=3e-2)

  @chex.all_variants
  @parameterized.product(_MAIN_OPTIMIZERS_UNDER_TEST)
  def test_optimizers_can_be_wrapped_in_inject_hyperparams(
      self, opt_name, opt_kwargs, wrapper_name=None, wrapper_kwargs=None
  ):
    """Checks that optimizers can be wrapped in inject_hyperparams."""
    # See also https://github.com/deepmind/optax/issues/412.
    # When debugging this, make sure that options like weight decay or not
    # are checked by asserting wehter such a value is None or not (see e.g. the
    # logic in schedule_free_adamw). Some hyperparameters may not be supported
    # by inject_hyperparams (e.g. warmup_steps). In that case (if you're sure
    # you can ignore such hyperparameter), add the exception below.
    if wrapper_name == 'reduce_on_plateau':
      # TODO(vroulet): discuss adding support for reduce_on_plateau
      # so removing all assertions in its definition
      self.skipTest('reduce_on_plateau is not supported by inject_hyperparams.')
    if wrapper_name is None:
      factory = _get_opt_factory(opt_name)
      hparams = opt_kwargs
    else:
      base_opt = _get_opt_factory(opt_name)(**opt_kwargs)
      factory = getattr(contrib, wrapper_name)
      factory = functools.partial(factory, base_opt)
      hparams = wrapper_kwargs
    opt = factory(**hparams)

    # Add here the hyperparameters that cannot be injected with
    # inject_hyperparams.
    static_args = []
    for uninjectable_hparam in ['warmup_steps', 'num_betas']:
      if uninjectable_hparam in inspect.signature(factory).parameters.keys():
        static_args.append(uninjectable_hparam)
    static_args = tuple(static_args)
    opt_inject = _inject.inject_hyperparams(factory, static_args)(**hparams)

    params = [jnp.negative(jnp.ones((2, 3))), jnp.ones((2, 5, 2))]
    grads = [jnp.ones((2, 3)), jnp.negative(jnp.ones((2, 5, 2)))]

    if opt_name in ['momo', 'momo_adam'] or wrapper_name == 'reduce_on_plateau':
      update_kwargs = {'value': jnp.array(1.0)}
    else:
      update_kwargs = {}

    state = self.variant(opt.init)(params)
    updates, new_state = self.variant(opt.update)(
        grads, state, params, **update_kwargs
    )

    state_inject = self.variant(opt_inject.init)(params)
    updates_inject, new_state_inject = self.variant(opt_inject.update)(
        grads, state_inject, params, **update_kwargs
    )

    with self.subTest('Equality of updates.'):
      chex.assert_trees_all_close(updates_inject, updates, rtol=1e-5)
    with self.subTest('Equality of new optimizer states.'):
      chex.assert_trees_all_close(
          new_state_inject.inner_state, new_state, rtol=1e-5, atol=1e-5
      )

  # Not testing with `without_device=True` because without_device set the
  # variables to the host which appears to convert then the dtype, so we
  # lose control of the dtype and the test fails.
  @chex.variants(
      with_jit=True, without_jit=True, with_device=True, with_pmap=True
  )
  @parameterized.product(
      _MAIN_OPTIMIZERS_UNDER_TEST, dtype=('bfloat16', 'float32')
  )
  def test_preserve_dtype(
      self, opt_name, opt_kwargs, dtype, wrapper_name=None, wrapper_kwargs=None
  ):
    """Test that the optimizers return updates of same dtype as params."""
    # When debugging this test, note that operations like
    # x = 0.5**jnp.asarray(1, dtype=jnp.int32)
    # (appearing in e.g. optax.tree_utils.tree_bias_correction)
    # are promoted (strictly) to float32 when jitted
    # see https://github.com/google/jax/issues/23337
    # This may end up letting updates have a dtype different from params.
    # The solution is to fix the dtype of the result to the desired dtype
    # (just as done in optax.tree_utils.tree_bias_correction).
    # Otherwise, just make sure that all variables defined in the optimizer have
    # the same dtype as the parameters.
    dtype = jnp.dtype(dtype)
    opt = _get_opt_factory(opt_name)(**opt_kwargs)
    if wrapper_name is not None:
      opt = _wrap_opt(opt, wrapper_name, wrapper_kwargs)
    fun = lambda x: jnp.sum(x**2)

    params = jnp.array([1.0, 2.0], dtype=dtype)
    value, grads = jax.value_and_grad(fun)(params)
    state = self.variant(opt.init)(params)
    if opt_name in ['momo', 'momo_adam'] or wrapper_name == 'reduce_on_plateau':
      update_kwargs = {'value': value}
    else:
      update_kwargs = {}
    updates, _ = self.variant(opt.update)(grads, state, params, **update_kwargs)
    self.assertEqual(updates.dtype, params.dtype)

  @chex.variants(
      with_jit=True, without_jit=True, with_device=True, with_pmap=True
  )
  @parameterized.product(
      _MAIN_OPTIMIZERS_UNDER_TEST, dtype=('bfloat16', 'float32')
  )
  def test_gradient_accumulation(
      self, opt_name, opt_kwargs, dtype, wrapper_name=None, wrapper_kwargs=None
  ):
    """Test that the optimizers can safely be used with optax.MultiSteps."""
    # Checks for issues like https://github.com/google-deepmind/optax/issues/377
    # Should pass as long as test_preserve_dtype passes.
    dtype = jnp.dtype(dtype)
    opt = _get_opt_factory(opt_name)(**opt_kwargs)
    if wrapper_name is not None:
      opt = _wrap_opt(opt, wrapper_name, wrapper_kwargs)
    opt = _accumulation.MultiSteps(opt, every_k_schedule=4)

    fun = lambda x: jnp.sum(x**2)

    params = jnp.array([1.0, 2.0], dtype=dtype)
    value, grads = jax.value_and_grad(fun)(params)
    state = self.variant(opt.init)(params)
    if opt_name in ['momo', 'momo_adam'] or wrapper_name == 'reduce_on_plateau':
      update_kwargs = {'value': value}
    else:
      update_kwargs = {}
    updates, _ = self.variant(opt.update)(grads, state, params, **update_kwargs)
    chex.assert_trees_all_equal(updates, jnp.zeros_like(grads))


if __name__ == '__main__':
  absltest.main()
