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
"""Linear algebra utilities used in optimization."""

from collections.abc import Callable
import functools
from typing import Optional, Union

import chex
import jax
from jax import lax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics


def _normalize_tree(x):
  # divide by the L2 norm of the tree weights.
  return otu.tree_scalar_mul(1.0 / otu.tree_l2_norm(x), x)


def global_norm(updates: base.PyTree) -> chex.Array:
  """Compute the global norm across a nested structure of tensors."""
  return jnp.sqrt(
      sum(jnp.sum(numerics.abs_sq(x)) for x in jax.tree.leaves(updates))
  )


def _power_iteration_cond_fun(error_tolerance, num_iters, loop_vars):
  normalized_eigvec, unnormalized_eigvec, eig, iter_num = loop_vars
  residual = otu.tree_sub(
      unnormalized_eigvec, otu.tree_scalar_mul(eig, normalized_eigvec)
  )
  residual_norm = otu.tree_l2_norm(residual)
  converged = jnp.abs(residual_norm / eig) < error_tolerance
  return ~converged & (iter_num < num_iters)


def power_iteration(
    matrix: Union[chex.Array, Callable[[chex.ArrayTree], chex.ArrayTree]],
    *,
    v0: Optional[chex.ArrayTree] = None,
    num_iters: int = 100,
    error_tolerance: float = 1e-6,
    precision: lax.Precision = lax.Precision.HIGHEST,
    key: Optional[chex.PRNGKey] = None,
) -> tuple[chex.Numeric, chex.ArrayTree]:
  r"""Power iteration algorithm.

  This algorithm computes the dominant eigenvalue and its associated eigenvector
  of a diagonalizable matrix. This matrix can be given as an array or as a
  callable implementing a matrix-vector product.

  Args:
    matrix: a square matrix, either as an array or a callable implementing a
      matrix-vector product.
    v0: initial vector approximating the dominiant eigenvector. If ``matrix`` is
      an array of size (n, n), v0 must be a vector of size (n,). If instead
      ``matrix`` is a callable, then v0 must be a tree with the same structure
      as the input of this callable. If this argument is None and ``matrix`` is
      an array, then a random vector sampled from a uniform distribution in [-1,
      1] is used as initial vector.
    num_iters: Number of power iterations.
    error_tolerance: Iterative exit condition. The procedure stops when the
      relative error of the estimate of the dominant eigenvalue is below this
      threshold.
    precision: precision XLA related flag, the available options are: a)
      lax.Precision.DEFAULT (better step time, but not precise); b)
      lax.Precision.HIGH (increased precision, slower); c) lax.Precision.HIGHEST
      (best possible precision, slowest).
    key: random key for the initialization of ``v0`` when not given explicitly.
      When this argument is None, `jax.random.PRNGKey(0)` is used.

  Returns:
    A pair (eigenvalue, eigenvector), where eigenvalue is the dominant
    eigenvalue of ``matrix`` and eigenvector is its associated eigenvector.

  References:
    Wikipedia contributors. `Power iteration
    <https://en.wikipedia.org/w/index.php?tit0le=Power_iteration>`_.

  .. versionchanged:: 0.2.2
    ``matrix`` can be a callable. Reversed the order of the return parameters,
    from (eigenvector, eigenvalue) to (eigenvalue, eigenvector).
  """
  if callable(matrix):
    mvp = matrix
    if v0 is None:
      # v0 must be given as we don't know the underlying pytree structure.
      raise ValueError('v0 must be provided when `matrix` is a callable.')
  else:
    mvp = lambda v: jnp.matmul(matrix, v, precision=precision)
    if v0 is None:
      if key is None:
        key = jax.random.PRNGKey(0)
      # v0 is uniformly distributed in [-1, 1]
      v0 = jax.random.uniform(
          key,
          shape=matrix.shape[-1:],
          dtype=matrix.dtype,
          minval=-1.0,
          maxval=1.0,
      )

  v0 = _normalize_tree(v0)

  cond_fun = functools.partial(
      _power_iteration_cond_fun,
      error_tolerance,
      num_iters,
  )

  def _body_fun(loop_vars):
    _, z, _, iter_num = loop_vars
    eigvec = _normalize_tree(z)
    z = mvp(eigvec)
    eig = otu.tree_vdot(eigvec, z)
    return eigvec, z, eig, iter_num + 1

  init_vars = (v0, mvp(v0), jnp.asarray(0.0), jnp.asarray(0))
  _, unormalized_eigenvector, dominant_eigenvalue, _ = jax.lax.while_loop(
      cond_fun, _body_fun, init_vars
  )
  normalized_eigenvector = _normalize_tree(unormalized_eigenvector)
  return dominant_eigenvalue, normalized_eigenvector


def matrix_inverse_pth_root(
    matrix: chex.Array,
    p: int,
    num_iters: int = 100,
    ridge_epsilon: float = 1e-6,
    error_tolerance: float = 1e-6,
    precision: lax.Precision = lax.Precision.HIGHEST,
):
  """Computes `matrix^(-1/p)`, where `p` is a positive integer.

  This function uses the Coupled newton iterations algorithm for
  the computation of a matrix's inverse pth root.

  Args:
    matrix: the symmetric PSD matrix whose power it to be computed
    p: exponent, for p a positive integer.
    num_iters: Maximum number of iterations.
    ridge_epsilon: Ridge epsilon added to make the matrix positive definite.
    error_tolerance: Error indicator, useful for early termination.
    precision: precision XLA related flag, the available options are: a)
      lax.Precision.DEFAULT (better step time, but not precise); b)
      lax.Precision.HIGH (increased precision, slower); c) lax.Precision.HIGHEST
      (best possible precision, slowest).

  Returns:
    matrix^(-1/p)

  References:
    [Functions of Matrices, Theory and Computation,
     Nicholas J Higham, Pg 184, Eq 7.18](
     https://epubs.siam.org/doi/book/10.1137/1.9780898717778)
  """

  # We use float32 for the matrix inverse pth root.
  # Switch to f64 if you have hardware that supports it.
  matrix_size = matrix.shape[0]
  alpha = jnp.asarray(-1.0 / p, jnp.float32)
  identity = jnp.eye(matrix_size, dtype=jnp.float32)
  max_ev, _ = power_iteration(
      matrix=matrix, num_iters=100, error_tolerance=1e-6, precision=precision
  )
  ridge_epsilon = ridge_epsilon * jnp.maximum(max_ev, 1e-16)

  def _unrolled_mat_pow_1(mat_m):
    """Computes mat_m^1."""
    return mat_m

  def _unrolled_mat_pow_2(mat_m):
    """Computes mat_m^2."""
    return jnp.matmul(mat_m, mat_m, precision=precision)

  def _unrolled_mat_pow_4(mat_m):
    """Computes mat_m^4."""
    mat_pow_2 = _unrolled_mat_pow_2(mat_m)
    return jnp.matmul(mat_pow_2, mat_pow_2, precision=precision)

  def _unrolled_mat_pow_8(mat_m):
    """Computes mat_m^4."""
    mat_pow_4 = _unrolled_mat_pow_4(mat_m)
    return jnp.matmul(mat_pow_4, mat_pow_4, precision=precision)

  def mat_power(mat_m, p):
    """Computes mat_m^p, for p == 1, 2, 4 or 8.

    Args:
      mat_m: a square matrix
      p: a positive integer

    Returns:
      mat_m^p
    """
    # We unrolled the loop for performance reasons.
    exponent = jnp.round(jnp.log2(p))
    return lax.switch(
        jnp.asarray(exponent, jnp.int32),
        [
            _unrolled_mat_pow_1,
            _unrolled_mat_pow_2,
            _unrolled_mat_pow_4,
            _unrolled_mat_pow_8,
        ],
        (mat_m),
    )

  def _iter_condition(state):
    (i, unused_mat_m, unused_mat_h, unused_old_mat_h, error, run_step) = state
    error_above_threshold = jnp.logical_and(error > error_tolerance, run_step)
    return jnp.logical_and(i < num_iters, error_above_threshold)

  def _iter_body(state):
    (i, mat_m, mat_h, unused_old_mat_h, error, unused_run_step) = state
    mat_m_i = (1 - alpha) * identity + alpha * mat_m
    new_mat_m = jnp.matmul(mat_power(mat_m_i, p), mat_m, precision=precision)
    new_mat_h = jnp.matmul(mat_h, mat_m_i, precision=precision)
    new_error = jnp.max(jnp.abs(new_mat_m - identity))
    # sometimes error increases after an iteration before decreasing and
    # converging. 1.2 factor is used to bound the maximal allowed increase.
    return (
        i + 1,
        new_mat_m,
        new_mat_h,
        mat_h,
        new_error,
        new_error < error * 1.2,
    )

  if matrix_size == 1:
    resultant_mat_h = (matrix + ridge_epsilon) ** alpha
    error = 0
  else:
    damped_matrix = matrix + ridge_epsilon * identity

    z = (1 + p) / (2 * jnp.linalg.norm(damped_matrix))
    new_mat_m_0 = damped_matrix * z
    new_error = jnp.max(jnp.abs(new_mat_m_0 - identity))
    new_mat_h_0 = identity * jnp.power(z, 1.0 / p)
    init_state = tuple(
        [0, new_mat_m_0, new_mat_h_0, new_mat_h_0, new_error, True]
    )
    _, mat_m, mat_h, old_mat_h, _, convergence = lax.while_loop(
        _iter_condition, _iter_body, init_state
    )
    error = jnp.max(jnp.abs(mat_m - identity))
    is_converged = jnp.asarray(convergence, old_mat_h.dtype)
    resultant_mat_h = is_converged * mat_h + (1 - is_converged) * old_mat_h
    resultant_mat_h = jnp.asarray(resultant_mat_h, matrix.dtype)
  return resultant_mat_h, error
