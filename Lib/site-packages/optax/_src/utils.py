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
"""Utility functions for testing."""

from collections.abc import Callable
import functools
import inspect
import operator
from typing import Any, Optional, Sequence, Union

import chex
from etils import epy
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import linear_algebra
from optax._src import numerics


with epy.lazy_imports():
  import jax.scipy.stats.norm as multivariate_normal  # pylint: disable=g-import-not-at-top,ungrouped-imports


def tile_second_to_last_dim(a: chex.Array) -> chex.Array:
  ones = jnp.ones_like(a)
  a = jnp.expand_dims(a, axis=-1)
  return jnp.expand_dims(ones, axis=-2) * a


def canonicalize_dtype(
    dtype: Optional[chex.ArrayDType],
) -> Optional[chex.ArrayDType]:
  """Canonicalise a dtype, skip if None."""
  if dtype is not None:
    return jax.dtypes.canonicalize_dtype(dtype)
  return dtype


@functools.partial(
    chex.warn_deprecated_function, replacement='optax.tree_utils.tree_cast'
)
def cast_tree(
    tree: chex.ArrayTree, dtype: Optional[chex.ArrayDType]
) -> chex.ArrayTree:
  return otu.tree_cast(tree, dtype)


def set_diags(a: jax.Array, new_diags: chex.Array) -> chex.Array:
  """Set the diagonals of every DxD matrix in an input of shape NxDxD.

  Args:
    a: rank 3, tensor NxDxD.
    new_diags: NxD matrix, the new diagonals of each DxD matrix.

  Returns:
    NxDxD tensor, with the same contents as `a` but with the diagonal
      changed to `new_diags`.
  """
  a_dim, new_diags_dim = len(a.shape), len(new_diags.shape)
  if a_dim != 3:
    raise ValueError(f'Expected `a` to be a 3D tensor, got {a_dim}D instead')
  if new_diags_dim != 2:
    raise ValueError(
        f'Expected `new_diags` to be a 2D array, got {new_diags_dim}D instead'
    )
  n, d, d1 = a.shape
  n_diags, d_diags = new_diags.shape
  if d != d1:
    raise ValueError(
        f'Shape mismatch: expected `a.shape` to be {(n, d, d)}, '
        f'got {(n, d, d1)} instead'
    )
  if d_diags != d or n_diags != n:
    raise ValueError(
        f'Shape mismatch: expected `new_diags.shape` to be {(n, d)}, '
        f'got {(n_diags, d_diags)} instead'
    )

  indices1 = jnp.repeat(jnp.arange(n), d)
  indices2 = jnp.tile(jnp.arange(d), n)
  indices3 = indices2

  # Use numpy array setting
  a = a.at[indices1, indices2, indices3].set(new_diags.flatten())
  return a


class MultiNormalDiagFromLogScale:
  """MultiNormalDiag which directly exposes its input parameters."""

  def __init__(self, loc: chex.Array, log_scale: chex.Array):
    self._log_scale = log_scale
    self._scale = jnp.exp(log_scale)
    self._mean = loc
    self._param_shape = jax.lax.broadcast_shapes(
        self._mean.shape, self._scale.shape
    )

  def sample(self, shape: Sequence[int], seed: chex.PRNGKey) -> chex.Array:
    sample_shape = tuple(shape) + self._param_shape
    return (
        jax.random.normal(seed, shape=sample_shape) * self._scale + self._mean
    )

  def log_prob(self, x: chex.Array) -> chex.Array:
    log_prob = multivariate_normal.logpdf(x, loc=self._mean, scale=self._scale)
    # Sum over parameter axes.
    sum_axis = [-(i + 1) for i in range(len(self._param_shape))]
    return jnp.sum(log_prob, axis=sum_axis)

  @property
  def log_scale(self) -> chex.Array:
    return self._log_scale

  @property
  def params(self) -> Sequence[chex.Array]:
    return [self._mean, self._log_scale]


def multi_normal(
    loc: chex.Array, log_scale: chex.Array
) -> MultiNormalDiagFromLogScale:
  return MultiNormalDiagFromLogScale(loc=loc, log_scale=log_scale)


@jax.custom_vjp
def _scale_gradient(inputs: chex.ArrayTree, scale: float) -> chex.ArrayTree:
  """Internal gradient scaling implementation."""
  del scale  # Only used for the backward pass defined in _scale_gradient_bwd.
  return inputs


def _scale_gradient_fwd(
    inputs: chex.ArrayTree, scale: float
) -> tuple[chex.ArrayTree, float]:
  return _scale_gradient(inputs, scale), scale


def _scale_gradient_bwd(
    scale: float, g: chex.ArrayTree
) -> tuple[chex.ArrayTree, None]:
  return (jax.tree.map(lambda g_: g_ * scale, g), None)


_scale_gradient.defvjp(_scale_gradient_fwd, _scale_gradient_bwd)


def scale_gradient(inputs: chex.ArrayTree, scale: float) -> chex.ArrayTree:
  """Scales gradients for the backwards pass.

  Args:
    inputs: A nested array.
    scale: The scale factor for the gradient on the backwards pass.

  Returns:
    An array of the same structure as `inputs`, with scaled backward gradient.
  """
  # Special case scales of 1. and 0. for more efficiency.
  if scale == 1.0:
    return inputs
  elif scale == 0.0:
    return jax.lax.stop_gradient(inputs)
  else:
    return _scale_gradient(inputs, scale)


def _extract_fns_kwargs(
    fns: tuple[Callable[..., Any], ...],
    kwargs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  """Split ``kwargs`` into sub_kwargs to be fed to each function in ``fns``.

  Given a dictionary of arguments ``kwargs`` and a list of functions
  ``fns = (fn_1, ..., fn_n)``, this utility splits the ``kwargs`` in several
  dictionaries ``(fn_1_kwargs, ..., fn_n_kwargs), remaining_kwargs``. Each
  dictionary ``fn_i_kwargs`` correspond to a subset of ``{key: values}`` pairs
  from ``kwargs`` such that ``key`` is one possible argument of the function
  ``fn_i``. The ``remaining_kwargs`` argument consist in all pairs
  ``{key: values}`` from ``kwargs`` whose ``key`` does not match any argument
  of any of the functions ``fns``.

  Args:
    fns: tuple of functions to feed kwargs to.
    kwargs: dictionary of keyword variables to be fed to funs.

  Returns:
    (fn_1_kwargs, ..., fn_n_kwargs)
      Keyword arguments for each function taken from kwargs.
    remaining_kwargs
      Keyword arguments present in kwargs but not in any of the input functions.

  Examples:
    >>> import optax
    >>> def fn1(a, b): return a+b
    >>> def fn2(c, d): return c+d
    >>> kwargs = {'b':1., 'd':2., 'e':3.}
    >>> fns_kwargs, remaining_kwargs = _extract_fns_kwargs((fn1, fn2), kwargs)
    >>> print(fns_kwargs)
    [{'b': 1.0}, {'d': 2.0}]
    >>> print(remaining_kwargs)
    {'e': 3.0}
    >>> # Possible usage
    >>> def super_fn(a, c, **kwargs):
    ...  (fn1_kwargs, fn2_kwargs), _ = _extract_fns_kwargs((fn1, fn2), kwargs)
    ...  return fn1(a, **fn1_kwargs) + fn2(c, **fn2_kwargs)
    >>> print(super_fn(1., 2., b=3., d=4.))
    10.0
  """
  fns_arg_names = [list(inspect.signature(fn).parameters.keys()) for fn in fns]
  fns_kwargs = [
      {k: v for k, v in kwargs.items() if k in fn_arg_names}
      for fn_arg_names in fns_arg_names
  ]
  all_possible_arg_names = functools.reduce(operator.add, fns_arg_names)
  remaining_keys = [k for k in kwargs.keys() if k not in all_possible_arg_names]
  remaining_kwargs = {k: v for k, v in kwargs.items() if k in remaining_keys}
  return fns_kwargs, remaining_kwargs


def value_and_grad_from_state(
    value_fn: Callable[..., Union[jax.Array, float]],
) -> Callable[..., tuple[Union[float, jax.Array], base.Updates]]:
  r"""Alternative to ``jax.value_and_grad`` that fetches value, grad from state.

  Line-search methods such as :func:`optax.scale_by_backtracking_linesearch`
  require to compute the gradient and objective function at the candidate
  iterate. This objective value and gradient can be re-used in the next
  iteration to save some computations using this utility function.

  Args:
    value_fn: function returning a scalar (float or array of dimension 1),
      amenable to differentiation in jax using :func:`jax.value_and_grad`.

  Returns:
    A callable akin to :func:`jax.value_and_grad` that fetches value
    and grad from the state if present. If no value or grad are found or if
    multiple value and grads are found this function raises an error. If a value
    is found but is infinite or nan, the value and grad are computed using
    :func:`jax.value_and_grad`. If the gradient found in the state is None,
    raises an Error.


  Examples:
    >>> import optax
    >>> import jax.numpy as jnp
    >>> def fn(x): return jnp.sum(x ** 2)
    >>> solver = optax.chain(
    ...     optax.sgd(learning_rate=1.),
    ...     optax.scale_by_backtracking_linesearch(
    ...         max_backtracking_steps=15, store_grad=True
    ...     )
    ... )
    >>> value_and_grad = optax.value_and_grad_from_state(fn)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: {:.2E}'.format(fn(params)))
    Objective function: 1.40E+01
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...   value, grad = value_and_grad(params, state=opt_state)
    ...   updates, opt_state = solver.update(
    ...       grad, opt_state, params, value=value, grad=grad, value_fn=fn
    ...   )
    ...   params = optax.apply_updates(params, updates)
    ...   print('Objective function: {:.2E}'.format(fn(params)))
    Objective function: 5.04E+00
    Objective function: 1.81E+00
    Objective function: 6.53E-01
    Objective function: 2.35E-01
    Objective function: 8.47E-02
  """

  def _value_and_grad(
      params: base.Params,
      *fn_args: Any,
      state: base.OptState,
      **fn_kwargs: dict[str, Any],
  ):
    value = otu.tree_get(state, 'value')
    grad = otu.tree_get(state, 'grad')
    if (value is None) or (grad is None):
      raise ValueError(
          'Value or gradient not found in the state. '
          'Make sure that these values are stored in the state by the '
          'optimizer.'
      )
    value, grad = jax.lax.cond(
        (~jnp.isinf(value)) & (~jnp.isnan(value)),
        lambda *_: (value, grad),
        lambda p, a, kwa: jax.value_and_grad(value_fn)(p, *a, **kwa),
        params,
        fn_args,
        fn_kwargs,
    )
    return value, grad

  return _value_and_grad


# TODO(b/183800387): remove legacy aliases.
safe_norm = numerics.safe_norm
safe_int32_increment = numerics.safe_int32_increment
global_norm = linear_algebra.global_norm
