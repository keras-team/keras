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
"""Tests for methods defined in `alias.py`."""

from collections.abc import Callable
from typing import Any, Union

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax import flatten_util
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
from optax._src import alias
from optax._src import base
from optax._src import numerics
from optax._src import transform
from optax._src import update
from optax.losses import _classification
from optax.schedules import _inject
from optax.transforms import _accumulation
import optax.tree_utils as otu
import scipy.optimize as scipy_optimize
from sklearn import datasets
from sklearn import linear_model


##############
# COMMON TESTS
##############


_OPTIMIZERS_UNDER_TEST = (
    dict(opt_name='sgd', opt_kwargs=dict(learning_rate=1e-3, momentum=0.9)),
    dict(opt_name='adadelta', opt_kwargs=dict(learning_rate=0.1)),
    dict(opt_name='adafactor', opt_kwargs=dict(learning_rate=5e-3)),
    dict(opt_name='adagrad', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='adam', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='adamw', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='adamax', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='adamaxw', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='adan', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='amsgrad', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='lars', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='lamb', opt_kwargs=dict(learning_rate=1e-3)),
    dict(
        opt_name='lion',
        opt_kwargs=dict(learning_rate=1e-2, weight_decay=1e-4),
    ),
    dict(opt_name='nadam', opt_kwargs=dict(learning_rate=1e-2)),
    dict(opt_name='nadamw', opt_kwargs=dict(learning_rate=1e-2)),
    dict(opt_name='noisy_sgd', opt_kwargs=dict(learning_rate=1e-3, eta=1e-4)),
    dict(opt_name='novograd', opt_kwargs=dict(learning_rate=1e-3)),
    dict(
        opt_name='optimistic_gradient_descent',
        opt_kwargs=dict(learning_rate=2e-3, alpha=0.7, beta=0.1),
    ),
    dict(
        opt_name='optimistic_adam',
        opt_kwargs=dict(learning_rate=2e-3),
    ),
    dict(opt_name='rmsprop', opt_kwargs=dict(learning_rate=5e-3)),
    dict(opt_name='rmsprop', opt_kwargs=dict(learning_rate=5e-3, momentum=0.9)),
    dict(opt_name='sign_sgd', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='fromage', opt_kwargs=dict(learning_rate=5e-3)),
    dict(opt_name='adabelief', opt_kwargs=dict(learning_rate=1e-2)),
    dict(opt_name='radam', opt_kwargs=dict(learning_rate=5e-3)),
    dict(opt_name='rprop', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='sm3', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='yogi', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='polyak_sgd', opt_kwargs=dict(max_learning_rate=1.0)),
)


def _setup_parabola(dtype):
  """Quadratic function as an optimization target."""
  initial_params = jnp.array([-1.0, 10.0, 1.0], dtype=dtype)
  final_params = jnp.array([1.0, -1.0, 1.0], dtype=dtype)

  if jnp.iscomplexobj(dtype):
    final_params *= 1 + 1j

  def objective(params):
    return jnp.sum(numerics.abs_sq(params - final_params))

  return initial_params, final_params, objective


def _setup_rosenbrock(dtype):
  """Rosenbrock function as an optimization target."""
  a = 1.0
  b = 100.0

  if jnp.iscomplexobj(dtype):
    a *= 1 + 1j

  initial_params = jnp.array([0.0, 0.0], dtype=dtype)
  final_params = jnp.array([a, a**2], dtype=dtype)

  def objective(params):
    return numerics.abs_sq(a - params[0]) + b * numerics.abs_sq(
        params[1] - params[0] ** 2
    )

  return initial_params, final_params, objective


class AliasTest(chex.TestCase):

  @parameterized.product(
      _OPTIMIZERS_UNDER_TEST,
      target=(_setup_parabola, _setup_rosenbrock),
      dtype=(jnp.float32, jnp.complex64),
  )
  def test_optimization(self, opt_name, opt_kwargs, target, dtype):
    if opt_name in (
        'fromage',
        'noisy_sgd',
        'sm3',
        'optimistic_gradient_descent',
        'optimistic_adam',
        'lion',
        'rprop',
        'adadelta',
        'adan',
        'polyak_sgd',
        'sign_sgd',
    ) and jnp.iscomplexobj(dtype):
      raise absltest.SkipTest(
          f'{opt_name} does not support complex parameters.'
      )

    if opt_name in ('sign_sgd',) and target is _setup_rosenbrock:
      raise absltest.SkipTest(
          f'{opt_name} requires learning rate scheduling to solve the'
          ' Rosenbrockfunction'
      )

    opt = getattr(alias, opt_name)(**opt_kwargs)
    initial_params, final_params, objective = target(dtype)

    @jax.jit
    def step(params, state):
      value, updates = jax.value_and_grad(objective)(params)
      # Complex gradients need to be conjugated before being added to parameters
      # https://gist.github.com/wdphy16/118aef6fb5f82c49790d7678cf87da29
      updates = jax.tree.map(lambda x: x.conj(), updates)
      if opt_name == 'polyak_sgd':
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
      state = otu.tree_map_params(opt, lambda v: v, state)

    with self.subTest('Test that optimization works'):
      for _ in range(10000):
        params, state = step(params, state)

      chex.assert_trees_all_close(params, final_params, rtol=3e-2, atol=3e-2)

  @chex.all_variants
  @parameterized.product(_OPTIMIZERS_UNDER_TEST)
  def test_optimizers_can_be_wrapped_in_inject_hyperparams(
      self, opt_name, opt_kwargs
  ):
    """Checks that optimizers can be wrapped in inject_hyperparams."""
    # See also https://github.com/google-deepmind/optax/issues/412.
    opt_factory = getattr(alias, opt_name)
    opt = opt_factory(**opt_kwargs)
    if opt_name == 'adafactor':
      # Adafactor wrapped in inject_hyperparams currently needs a static
      # argument to be specified in order to be jittable. See issue
      # https://github.com/google-deepmind/optax/issues/412.
      opt_inject = _inject.inject_hyperparams(
          opt_factory, static_args=('min_dim_size_to_factor',)
      )(**opt_kwargs)
    else:
      opt_inject = _inject.inject_hyperparams(opt_factory)(**opt_kwargs)

    params = [jnp.negative(jnp.ones((2, 3))), jnp.ones((2, 5, 2))]
    grads = [jnp.ones((2, 3)), jnp.negative(jnp.ones((2, 5, 2)))]

    state = self.variant(opt.init)(params)
    if opt_name == 'polyak_sgd':
      update_kwargs = {'value': jnp.array(0.0)}
    else:
      update_kwargs = {}
    updates, new_state = self.variant(opt.update)(
        grads, state, params, **update_kwargs
    )

    state_inject = self.variant(opt_inject.init)(params)
    updates_inject, new_state_inject = self.variant(opt_inject.update)(
        grads, state_inject, params, **update_kwargs
    )

    with self.subTest('Equality of updates.'):
      chex.assert_trees_all_close(updates_inject, updates, rtol=1e-4)
    with self.subTest('Equality of new optimizer states.'):
      chex.assert_trees_all_close(
          new_state_inject.inner_state, new_state, rtol=1e-4
      )

  @parameterized.product(
      params_dtype=('bfloat16', 'float32', 'complex64', None),
      state_dtype=('bfloat16', 'float32', 'complex64', None),
      opt_name=('sgd_mom', 'adam', 'adamw'),
  )
  def test_explicit_dtype(self, params_dtype, state_dtype, opt_name):
    if opt_name == 'sgd_mom':
      opt = alias.sgd(0.1, momentum=0.9, accumulator_dtype=state_dtype)
      attribute_name = 'trace'
    elif opt_name in ['adam', 'adamw']:
      opt = getattr(alias, opt_name)(0.1, mu_dtype=state_dtype)
      attribute_name = 'mu'
    else:
      raise ValueError(f'Unsupported optimizer: {opt_name}')

    params_dtype = jax.dtypes.canonicalize_dtype(params_dtype)
    params = jnp.array([0.0, 0.0], dtype=params_dtype)
    state = opt.init(params)

    with self.subTest('Test that attribute dtype is correct'):
      if state_dtype is None:
        expected_dtype = params_dtype
      else:
        expected_dtype = jax.dtypes.canonicalize_dtype(state_dtype)
      attribute = otu.tree_get(state, attribute_name)
      self.assertEqual(expected_dtype, attribute.dtype)

  # Not testing with `without_device=True` because without_device set the
  # variables to the host which appears to convert then the dtype, so we
  # lose control of the dtype and the test fails.
  @chex.variants(
      with_jit=True, without_jit=True, with_device=True, with_pmap=True
  )
  @parameterized.product(_OPTIMIZERS_UNDER_TEST, dtype=('bfloat16', 'float32'))
  def test_preserve_dtype(self, opt_name, opt_kwargs, dtype):
    """Test that the optimizers return updates of same dtype as gradients."""
    # When debugging this test, note that operations like
    # x = 0.5**jnp.asarray(1, dtype=jnp.int32)
    # (appearing in e.g. optax.tree_utils.tree_bias_correction)
    # are promoted (strictly) to float32 when jitted
    # see https://github.com/google/jax/issues/23337
    # This may end up letting updates have a dtype different from params.
    # The solution is to fix the dtype of the result to the desired dtype
    # (just as done in optax.tree_utils.tree_bias_correction).
    dtype = jnp.dtype(dtype)
    opt_factory = getattr(alias, opt_name)
    opt = opt_factory(**opt_kwargs)
    fun = lambda x: jnp.sum(x**2)

    params = jnp.array([1.0, 2.0], dtype=dtype)
    grads = jax.grad(fun)(params)
    state = self.variant(opt.init)(params)
    if opt_name == 'polyak_sgd':
      update_kwargs = {'value': fun(params)}
    else:
      update_kwargs = {}
    updates, _ = self.variant(opt.update)(grads, state, params, **update_kwargs)
    self.assertEqual(updates.dtype, grads.dtype)

  @chex.variants(
      with_jit=True, without_jit=True, with_device=True, with_pmap=True
  )
  @parameterized.product(_OPTIMIZERS_UNDER_TEST, dtype=('bfloat16', 'float32'))
  def test_gradient_accumulation(self, opt_name, opt_kwargs, dtype):
    """Test that the optimizers can safely be used with optax.MultiSteps."""
    # Checks for issues like https://github.com/google-deepmind/optax/issues/377
    dtype = jnp.dtype(dtype)
    opt_factory = getattr(alias, opt_name)
    base_opt = opt_factory(**opt_kwargs)
    opt = _accumulation.MultiSteps(base_opt, every_k_schedule=4)

    fun = lambda x: jnp.sum(x**2)

    params = jnp.array([1.0, 2.0], dtype=dtype)
    grads = jax.grad(fun)(params)
    state = self.variant(opt.init)(params)
    if opt_name == 'polyak_sgd':
      update_kwargs = {'value': fun(params)}
    else:
      update_kwargs = {}
    updates, _ = self.variant(opt.update)(grads, state, params, **update_kwargs)
    chex.assert_trees_all_equal(updates, jnp.zeros_like(grads))


##########################
# ALGORITHM SPECIFIC TESTS
##########################


#######
# LBFGS


def _run_opt(
    opt: base.GradientTransformationExtraArgs,
    fun: Callable[[chex.ArrayTree], jnp.ndarray],
    init_params: chex.ArrayTree,
    maxiter: int = 500,
    tol: float = 1e-3,
) -> tuple[chex.ArrayTree, base.OptState]:
  """Run LBFGS solver by iterative calls to grad transform and apply_updates."""
  value_and_grad_fun = jax.value_and_grad(fun)

  def stopping_criterion(carry):
    _, _, count, grad = carry
    return (otu.tree_l2_norm(grad) >= tol) & (count < maxiter)

  def step(carry):
    params, state, count, _ = carry
    value, grad = value_and_grad_fun(params)
    updates, state = opt.update(
        grad, state, params, value=value, grad=grad, value_fn=fun
    )
    params = update.apply_updates(params, updates)
    return params, state, count + 1, grad

  init_state = opt.init(init_params)
  init_grad = jax.grad(fun)(init_params)
  final_params, final_state, *_ = jax.lax.while_loop(
      stopping_criterion, step, (init_params, init_state, 0, init_grad)
  )

  return final_params, final_state


def _materialize_approx_inv_hessian(
    diff_params_memory: jnp.ndarray,
    diff_updates_memory: jnp.ndarray,
    weights_memory: jnp.ndarray,
    memory_idx: int,
) -> jnp.ndarray:
  """Computes approximate inverse hessian in lbfgs as product of matrices."""
  # Equation (7.19) in "Numerical Optimization" by Nocedal and Wright, 1999
  # Notations differ from reference above with the following correspondences
  # dws -> s, dus -> y, rhos -> rhos, V -> V, P -> H

  # Shorten names for better readability in terms of math, see
  # :func:`optax.scale_by_lbfgs` for mathematical formulas.
  dws, dus, rhos = diff_params_memory, diff_updates_memory, weights_memory
  k = memory_idx
  # m below is the memory size
  m, d = diff_params_memory.shape

  dws = jnp.roll(dws, -k, axis=0)
  dus = jnp.roll(dus, -k, axis=0)
  rhos = jnp.roll(rhos, -k, axis=0)

  id_mat = jnp.eye(d, d)
  # pylint: disable=invalid-name
  P = id_mat
  safe_dot = lambda x, y: jnp.dot(x, y, precision=jax.lax.Precision.HIGHEST)
  for j in range(m):
    V = id_mat - rhos[j] * jnp.outer(dus[j], dws[j])
    P = safe_dot(V.T, safe_dot(P, V)) + rhos[j] * jnp.outer(dws[j], dws[j])
  # pylint: enable=invalid-name
  precond_mat = P
  return precond_mat


def _plain_preconditioning(
    diff_params_memory: Union[list[jnp.ndarray], jnp.ndarray],
    diff_updates_memory: Union[list[jnp.ndarray], jnp.ndarray],
    updates: jnp.ndarray,
    identity_scale: float = 1.0,
) -> jnp.ndarray:
  """Plain implementation of lbfgs preconditioning."""
  # Algorithm 7.4 in "Numerical Optimization" by Nocedal and Wright, 1999
  # Notations differ from reference above with the following correspondences
  # dws -> s, dus -> y, rhos -> rhos, precond_factor -> V, precond_mat -> H,
  # identity_scale -> gamma

  # 1. Operates on list of vectors rather than stacked trees.
  # 2. Computes weights (rhos) of the rank one matrices directly rather than
  # accessing these weights from past memory.
  # 3. Uses plain for loops rather than scan.

  # Shorten names for better readability in terms of math, see
  # :func:`optax.scale_by_lbfgs` for mathematical formulas.
  dws, dus = diff_params_memory, diff_updates_memory
  # m below is the memory size
  m = len(dws)

  if m == 0:
    return identity_scale * updates

  dws = jnp.array(dws)
  dus = jnp.array(dus)

  rhos = jnp.zeros(m)
  alphas = jnp.zeros(m)

  # Compute right product.
  def right_product(j, tup):
    rhos, alphas, u = tup
    i = m - j - 1
    # rhos[i] = 1. / jnp.sum(dws[i] * dus[i])
    rhos = rhos.at[i].set(1.0 / jnp.sum(dws[i] * dus[i]))
    # alphas[i] = rhos[i] * jnp.sum(dws[i] * r)
    alphas = alphas.at[i].set(rhos[i] * jnp.sum(dws[i] * u))
    u = u - alphas[i] * dus[i]
    return rhos, alphas, u

  # for i in reversed(range(m)):
  rhos, alphas, pu = jax.lax.fori_loop(
      0, m, right_product, (rhos, alphas, updates)
  )

  pu = pu * identity_scale

  # Compute left product.
  def left_product(i, u):
    beta = rhos[i] * jnp.sum(dus[i] * u)
    return u + dws[i] * (alphas[i] - beta)

  # for i in range(m):
  pu = jax.lax.fori_loop(0, m, left_product, pu)

  return pu


def _plain_lbfgs(
    fun: Callable[[jnp.ndarray], jnp.ndarray],
    init_params: jnp.ndarray,
    stepsize: float = 1e-3,
    maxiter: int = 500,
    tol: float = 1e-3,
    memory_size: int = 10,
    scale_init_precond: bool = True,
) -> jnp.ndarray:
  """Plain implementation of LBFGS."""
  # Algorithm 7.5 in "Numerical Optimization" by Nocedal and Wright, 1999
  # Notations differ from reference above with the following correspondences
  # dws -> s, dus -> y, identity_scale -> gamma
  value_and_grad_fun = jax.value_and_grad(fun)

  w = init_params
  _, g = value_and_grad_fun(init_params)
  dws = []
  dus = []

  for it in range(maxiter):
    if scale_init_precond:
      if it == 0:
        identity_scale = jnp.minimum(1.0, 1.0 / jnp.sqrt(jnp.sum(g**2)))
      else:
        identity_scale = jnp.vdot(dus[-1], dws[-1])
        identity_scale /= jnp.sum(dus[-1] ** 2)
    else:
      identity_scale = 1.0

    direction = -_plain_preconditioning(dws, dus, g, identity_scale)
    w_old, g_old = w, g
    w = w + stepsize * direction
    _, g = value_and_grad_fun(w)

    dws.append(w - w_old)
    dus.append(g - g_old)

    if len(dws) > memory_size:
      dws = dws[1:]  # Pop left.
      dus = dus[1:]

    grad_norm = jnp.sqrt(jnp.sum(g**2))

    if grad_norm <= tol:
      break

  return w


def _get_problem(
    name: str,
) -> dict[str, Any]:
  """Get test function in given numpy (xnp) framework."""

  def rosenbrock(x, xnp):
    return xnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

  def himmelblau(p):
    x, y = p
    return (x**2 + y - 11.0) ** 2 + (x + y**2 - 7.0) ** 2

  def matyas(p):
    x, y = p
    return 0.26 * (x**2 + y**2) - 0.48 * x * y

  def eggholder(p, xnp):
    x, y = p
    return -(y + 47) * xnp.sin(
        xnp.sqrt(xnp.abs(x / 2.0 + y + 47.0))
    ) - x * xnp.sin(xnp.sqrt(xnp.abs(x - (y + 47.0))))

  def zakharov(x, xnp):
    ii = xnp.arange(1, len(x) + 1, step=1, dtype=x.dtype)
    sum1 = (x**2).sum()
    sum2 = (0.5 * ii * x).sum()
    answer = sum1 + sum2**2 + sum2**4
    return answer

  problems = dict(
      rosenbrock=dict(
          fun=lambda x: rosenbrock(x, jnp),
          numpy_fun=lambda x: rosenbrock(x, np),
          init=np.zeros(2),
          minimum=0.0,
          minimizer=np.ones(2),
      ),
      himmelblau=dict(
          fun=himmelblau,
          numpy_fun=himmelblau,
          init=np.ones(2),
          minimum=0.0,
          # himmelblau has actually multiple minimizers, we simply consider one.
          minimizer=np.array([3.0, 2.0]),
      ),
      matyas=dict(
          fun=matyas,
          numpy_fun=matyas,
          init=np.ones(2) * 6.0,
          minimum=0.0,
          minimizer=np.zeros(2),
      ),
      eggholder=dict(
          fun=lambda x: eggholder(x, jnp),
          numpy_fun=lambda x: eggholder(x, np),
          init=np.ones(2) * 6.0,
          minimum=-959.6407,
          minimizer=np.array([512.0, 404.22319]),
      ),
      zakharov=dict(
          fun=lambda x: zakharov(x, jnp),
          numpy_fun=lambda x: zakharov(x, np),
          init=np.array([600.0, 700.0, 200.0, 100.0, 90.0, 1e3]),
          minimum=0.0,
          minimizer=np.zeros(6),
      ),
  )
  return problems[name]


class LBFGSTest(chex.TestCase):

  def test_plain_preconditioning(self):
    key = jrd.PRNGKey(0)
    key_ws, key_us, key_vec = jrd.split(key, 3)
    m = 4
    d = 3
    dws = jrd.normal(key_ws, (m, d))
    dus = jrd.normal(key_us, (m, d))
    rhos = 1.0 / jnp.sum(dws * dus, axis=1)
    vec = jrd.normal(key_vec, (d,))
    plain_precond_vec = _plain_preconditioning(dws, dus, vec)
    precond_mat = _materialize_approx_inv_hessian(dws, dus, rhos, memory_idx=0)
    expected_precond_vec = precond_mat.dot(
        vec, precision=jax.lax.Precision.HIGHEST
    )
    chex.assert_trees_all_close(
        plain_precond_vec, expected_precond_vec, rtol=1e-5
    )

  @parameterized.product(idx=[0, 1, 2, 3])
  def test_preconditioning_by_lbfgs_on_vectors(self, idx: int):
    key = jrd.PRNGKey(0)
    key_ws, key_us, key_vec = jrd.split(key, 3)
    m = 4
    d = 3
    dws = jrd.normal(key_ws, (m, d))
    dus = jrd.normal(key_us, (m, d))
    rhos = 1.0 / jnp.sum(dws * dus, axis=1)
    vec = jrd.normal(key_vec, (d,))

    # Test for all possible indexes
    precond_mat = _materialize_approx_inv_hessian(
        dws, dus, rhos, memory_idx=idx
    )
    expected_precond_vec = precond_mat.dot(
        vec, precision=jax.lax.Precision.HIGHEST
    )

    lbfgs_precond_vec = transform._precondition_by_lbfgs(
        vec, dws, dus, rhos, identity_scale=1.0, memory_idx=idx
    )

    chex.assert_trees_all_close(
        lbfgs_precond_vec, expected_precond_vec, atol=1e-5, rtol=1e-5
    )

  @parameterized.product(idx=[0, 1, 2, 3])
  def test_preconditioning_by_lbfgs_on_trees(self, idx: int):
    key = jrd.PRNGKey(0)
    key_ws, key_us, key_vec = jrd.split(key, 3)
    m = 4
    shapes = ((3, 2), (5,))

    dws = tuple(
        jrd.normal(k, (m, *s))
        for k, s in zip(jrd.split(key_ws, len(shapes)), shapes)
    )
    dus = tuple(
        jrd.normal(k, (m, *s))
        for k, s in zip(jrd.split(key_us, len(shapes)), shapes)
    )
    vec = tuple(
        jrd.normal(k, s)
        for k, s in zip(jrd.split(key_vec, len(shapes)), shapes)
    )

    flat_dws = [
        flatten_util.ravel_pytree(jax.tree.map(lambda dw: dw[i], dws))[0]  # pylint: disable=cell-var-from-loop
        for i in range(m)
    ]
    flat_dus = [
        flatten_util.ravel_pytree(jax.tree.map(lambda du: du[i], dus))[0]  # pylint: disable=cell-var-from-loop
        for i in range(m)
    ]
    flat_dws, flat_dus = jnp.stack(flat_dws), jnp.stack(flat_dus)
    flat_vec = flatten_util.ravel_pytree(vec)[0]

    inv_rhos = [jnp.dot(flat_dws[i], flat_dus[i]) for i in range(m)]
    rhos = 1.0 / jnp.array(inv_rhos)

    lbfgs_precond_vec = transform._precondition_by_lbfgs(
        vec,
        dws,
        dus,
        rhos,
        identity_scale=1.0,
        memory_idx=idx,
    )
    flat_lbfgs_precond_vec = flatten_util.ravel_pytree(lbfgs_precond_vec)[0]

    flat_precond_mat = _materialize_approx_inv_hessian(
        flat_dws, flat_dus, rhos, idx
    )
    expected_flat_precond_vec = jnp.dot(
        flat_precond_mat, flat_vec, precision=jax.lax.Precision.HIGHEST
    )

    chex.assert_trees_all_close(
        flat_lbfgs_precond_vec, expected_flat_precond_vec, atol=1e-3, rtol=1e-3
    )

  @parameterized.product(
      problem_name=[
          'rosenbrock',
          'himmelblau',
          'matyas',
          'eggholder',
          'zakharov',
      ],
      scale_init_precond=[True, False],
  )
  def test_against_plain_implementation(
      self, problem_name: str, scale_init_precond: bool
  ):
    problem = _get_problem(problem_name)
    fun, init_params = problem['fun'], problem['init']
    learning_rate = 1e-3
    memory_size = 5
    maxiter = 15
    tol = 1e-3
    opt = alias.lbfgs(
        learning_rate=learning_rate,
        memory_size=memory_size,
        scale_init_precond=scale_init_precond,
        linesearch=None,
    )
    lbfgs_sol, _ = _run_opt(
        opt, fun, init_params, maxiter=maxiter, tol=tol
    )
    expected_lbfgs_sol = _plain_lbfgs(
        fun,
        init_params,
        stepsize=learning_rate,
        maxiter=maxiter,
        tol=tol,
        memory_size=memory_size,
        scale_init_precond=scale_init_precond,
    )
    chex.assert_trees_all_close(
        lbfgs_sol, expected_lbfgs_sol, atol=1e-5, rtol=1e-5
    )

  def test_handling_pytrees(self):
    def fun_(x):
      return jnp.sum(
          100.0 * (x[..., 1:] - x[..., :-1] ** 2.0) ** 2.0
          + (1 - x[..., :-1]) ** 2.0
      )

    def fun(x):
      return otu.tree_sum(jax.tree.map(fun_, x))

    key = jrd.PRNGKey(0)
    init_array = jrd.normal(key, (2, 4))
    init_tree = (init_array[0], init_array[1])

    opt = alias.lbfgs()
    sol_arr, _ = _run_opt(opt, fun, init_array, maxiter=3)
    sol_tree, _ = _run_opt(opt, fun, init_tree, maxiter=3)
    sol_tree = jnp.stack((sol_tree[0], sol_tree[1]))
    chex.assert_trees_all_close(sol_arr, sol_tree, rtol=5 * 1e-5, atol=5 * 1e-5)

  @parameterized.product(scale_init_precond=[True, False])
  def test_multiclass_logreg(self, scale_init_precond):
    data = datasets.make_classification(
        n_samples=10, n_features=5, n_classes=3, n_informative=3, random_state=0
    )

    def fun(params):
      inputs, labels = data
      weights, bias = params
      logits = jnp.dot(inputs, weights) + bias
      losses = _classification.softmax_cross_entropy_with_integer_labels(
          logits, labels
      )
      return jnp.mean(losses)

    weights_init = jnp.zeros((data[0].shape[1], 3))
    biases_init = jnp.zeros(3)
    init_params = (weights_init, biases_init)

    opt = alias.lbfgs(scale_init_precond=scale_init_precond)
    sol, _ = _run_opt(opt, fun, init_params, tol=1e-3)

    # Check optimality conditions.
    self.assertLessEqual(otu.tree_l2_norm(jax.grad(fun)(sol)), 1e-2)

  @parameterized.product(scale_init_precond=[True, False])
  def test_binary_logreg(self, scale_init_precond):
    inputs, labels = datasets.make_classification(
        n_samples=10, n_features=5, n_classes=2, n_informative=3, random_state=0
    )
    data = (inputs, labels)

    def fun(weights):
      inputs, labels = data
      logits = jnp.dot(inputs, weights)
      losses = jax.tree.map(
          lambda z, y: jax.nn.softplus(jnp.where(y, -z, z)), logits, labels
      )
      return jnp.mean(losses)

    init_params = jnp.zeros(inputs.shape[1])
    opt = alias.lbfgs(scale_init_precond=scale_init_precond)
    sol, _ = _run_opt(opt, fun, init_params, tol=1e-6)

    # Check optimality conditions.
    self.assertLessEqual(otu.tree_l2_norm(jax.grad(fun)(sol)), 1e-2)

    # Compare against sklearn.
    logreg = linear_model.LogisticRegression(
        fit_intercept=False,
        C=1.0 / (1e-6 * inputs.shape[0]),
        tol=1e-5,
        solver='liblinear',
        penalty='l2',
        random_state=0,
    )
    logreg = logreg.fit(inputs, labels)
    sol_skl = (
        logreg.coef_.ravel() if logreg.coef_.shape[0] == 1 else logreg.coef_.T
    )
    chex.assert_trees_all_close(sol, sol_skl, atol=5e-2)

  @parameterized.product(
      problem_name=[
          'rosenbrock',
          'himmelblau',
          'matyas',
          'eggholder',
          'zakharov',
      ],
  )
  def test_against_scipy(self, problem_name: str):
    # Taken from previous jaxopt tests

    tol = 1e-5
    problem = _get_problem(problem_name)
    init_params = problem['init']
    jnp_fun, np_fun = problem['fun'], problem['numpy_fun']

    opt = alias.lbfgs()
    optax_sol, _ = _run_opt(
        opt, jnp_fun, init_params, maxiter=500, tol=tol
    )
    scipy_sol = scipy_optimize.minimize(np_fun, init_params, method='BFGS').x

    # 1. Check minimizer obtained against known minimizer or scipy minimizer
    with self.subTest('Check minimizer'):
      if problem_name in ['matyas', 'zakharov']:
        chex.assert_trees_all_close(
            optax_sol, problem['minimizer'], atol=tol, rtol=tol
        )
      else:
        chex.assert_trees_all_close(optax_sol, scipy_sol, atol=tol, rtol=tol)

    with self.subTest('Check minimum'):
      # 2. Check if minimum is reached or equal to scipy's found value
      if problem_name == 'eggholder':
        chex.assert_trees_all_close(
            jnp_fun(optax_sol), np_fun(scipy_sol), atol=tol, rtol=tol
        )
      else:
        chex.assert_trees_all_close(
            jnp_fun(optax_sol), problem['minimum'], atol=tol, rtol=tol
        )

  def test_minimize_bad_initialization(self):
    # This test runs deliberately "bad" initial values to test that handling
    # of failed line search, etc. is the same across implementations
    tol = 1e-5
    problem = _get_problem('himmelblau')
    init_params = np.array([92, 0.001])
    jnp_fun, np_fun = problem['fun'], problem['numpy_fun']
    minimum = problem['minimum']
    opt = alias.lbfgs()
    optax_sol, _ = _run_opt(opt, jnp_fun, init_params, tol=tol)
    scipy_sol = scipy_optimize.minimize(
        fun=np_fun,
        jac=jax.grad(np_fun),
        method='BFGS',
        x0=init_params,
    ).x
    chex.assert_trees_all_close(
        np_fun(scipy_sol), jnp_fun(optax_sol), atol=tol, rtol=tol
    )
    chex.assert_trees_all_close(jnp_fun(optax_sol), minimum, atol=tol, rtol=tol)

  def test_steep_objective(self):
    # See jax related issue https://github.com/google/jax/issues/4594
    tol = 1e-5
    n = 2
    mat = jnp.eye(n) * 1e4

    def fun(x):
      return jnp.mean((mat @ x) ** 2)

    opt = alias.lbfgs()
    sol, _ = _run_opt(opt, fun, init_params=jnp.ones(n), tol=tol)
    chex.assert_trees_all_close(sol, jnp.zeros(n), atol=tol, rtol=tol)


if __name__ == '__main__':
  absltest.main()
