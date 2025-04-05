# Copyright 2021 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""A JIT-compatible library for QDWH-based polar decomposition.

QDWH is short for QR-based dynamically weighted Halley iteration. The Halley
iteration implemented through QR decmopositions does not require matrix
inversion. This is desirable for multicore and heterogeneous computing systems.

Reference: Nakatsukasa, Yuji, Zhaojun Bai, and François Gygi.
"Optimizing Halley's iteration for computing the matrix polar decomposition."
SIAM Journal on Matrix Analysis and Applications 31, no. 5 (2010): 2700-2720.
https://epubs.siam.org/doi/abs/10.1137/090774999
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import core
from jax._src.lax import linalg as lax_linalg


# Helpers for working with padded shapes
def _mask(x, dims, alternative=0):
  """Masks `x` up to the dynamic shape `dims`.

  Replaces values outside those dimensions with `alternative`. `alternative` is
  broadcast with `x`.
  """
  assert jnp.ndim(x) == len(dims)
  mask = None
  for i, d in enumerate(dims):
    if d is not None:
      mask_dim_i = lax.broadcasted_iota(jnp.int32, x.shape, i) < d
      mask = mask_dim_i if mask is None else (mask & mask_dim_i)
  return x if mask is None else jnp.where(mask, x, alternative)

def _pad_in_dim(x, low=0, high=0, interior=0, fill_value=0, axis=0):
  pads = [(0, 0, 0)] * x.ndim
  pads[axis] = (low, high, interior)
  return lax.pad(x, jnp.array(fill_value, x.dtype), pads)

def _dynamic_concat(a, b, m, axis=0):
  "Concatenates padded arrays `a` and `b` where the true size of `a` is `m`."
  if m is None:
    return jnp.concatenate([a, b], axis=axis)
  return lax.dynamic_update_slice_in_dim(
      _pad_in_dim(a, high=b.shape[axis], axis=axis), b, m, axis)


def _use_qr(u, m, n, params):
  """QDWH iteration using QR decomposition.

  Args:
  u: a matrix, with static (padded) shape M x N.
  m, n: the dynamic shape of the matrix, where m <= M and n <= N.
  params: the QDWH parameters.
  """
  a_minus_e_by_sqrt_c, sqrt_c, e = params
  M, N = u.shape

  y = _dynamic_concat(sqrt_c * u, jnp.eye(N, dtype=jnp.dtype(u)), m)
  q, _ = lax_linalg.qr(y, full_matrices=False)
  # q1 = q[:m, :]
  q1 = _mask(lax.slice(q, (0, 0), (M, N)), (m, n))
  # q2 = (q[m:, :]).T.conj()
  q2 = lax.dynamic_slice_in_dim(q, m, N, axis=0)
  q2 = _mask(q2, (n, n)).T.conj()
  return e * u + a_minus_e_by_sqrt_c * (q1 @ q2)


def _use_cholesky(u, m, n, params):
  """QDWH iteration using Cholesky decomposition.

  Args:
  u: a matrix, with static (padded) shape M x N
  m, n: the dynamic shape of the matrix, where m <= M and n <= N.
  params: the QDWH parameters.
  """
  a_minus_e, c, e = params
  _, N = u.shape
  x = c * (u.T.conj() @ u) + jnp.eye(N, dtype=jnp.dtype(u))
  # Pads the lower-right corner with the identity matrix to prevent the Cholesky
  # decomposition from failing due to the matrix not being PSD if padded with
  # zeros.
  x = _mask(x, (n, n), jnp.eye(N, dtype=x.dtype))

  # `y` is lower triangular.
  y = lax_linalg.cholesky(x, symmetrize_input=False)

  z = lax_linalg.triangular_solve(
      y, u.T, left_side=True, lower=True, conjugate_a=True).conj()

  z = lax_linalg.triangular_solve(y, z, left_side=True, lower=True,
                                  transpose_a=True, conjugate_a=True).T.conj()

  return e * u + a_minus_e * z


def _qdwh(x, m, n, max_iterations, eps):
  """QR-based dynamically weighted Halley iteration for polar decomposition."""

  # Estimates `alpha` and `beta = alpha * l`, where `alpha` is an estimate of
  # norm(x, 2) such that `alpha >= norm(x, 2)` and `beta` is a lower bound for
  # the smallest singular value of x.
  if eps is None:
    eps = float(jnp.finfo(x.dtype).eps)
  one_norm = jnp.linalg.norm(x, ord=1)
  inf_norm = jnp.linalg.norm(x, ord=jnp.inf)
  alpha_inverse = lax.rsqrt(one_norm) * lax.rsqrt(inf_norm)
  alpha_inverse = jnp.where(one_norm == 0, 1, alpha_inverse)
  u = x * alpha_inverse.astype(x.dtype)

  l = eps

  # Iteration tolerances.
  tol_l = 10.0 * eps / 2.0
  tol_norm = jnp.cbrt(tol_l)

  def get_qr_params(a, b, c):
    e = b / c
    a_minus_e = a - e
    sqrt_c = c ** (1 / 2)
    return (a_minus_e / sqrt_c, sqrt_c, e)

  def get_chol_params(a, b, c):
    e = b / c
    a_minus_e = a - e
    return (a_minus_e, c, e)

  CHOLESKY_CUTOFF = 100

  qr_coefs = []
  chol_coefs = []
  k = 0
  while l + tol_l < 1 and k < max_iterations:
    k += 1
    l2 = l * l
    dd = (4 * (1 / l2 - 1) / l2) ** (1 / 3)
    sqd = (1.0 + dd) ** (1 / 2)
    a = sqd + (2 - dd + 2 * (2 - l2) / (l2 * sqd)) ** (1 / 2)
    b = (a - 1) ** 2 / 4
    c = a + b - 1
    l = l * (a + b * l2) / (1 + c * l2)
    if c > CHOLESKY_CUTOFF:
      qr_coefs.append(get_qr_params(a, b, c))
    else:
      chol_coefs.append(get_chol_params(a, b, c))

  def iteration(k, state, update_fn, coefs, test_convergence):
    u, _ = state

    if coefs is None:
      # As l → 1, the coefficients a, b, c → 3, 1, 3, which is Halley's method.
      params = get_chol_params(3, 1, 3)
    else:
      params = lax.dynamic_index_in_dim(coefs, k, keepdims=False)

    u_prev = u
    u = update_fn(u, m, n, params)

    is_not_converged = True
    if test_convergence:
      is_not_converged = jnp.linalg.norm(u - u_prev) > tol_norm
    return u, is_not_converged

  def iterate(u, coefs, **kwargs):
    if not coefs:
      return u, True
    coefs = jnp.array(coefs).astype(x.dtype)
    body = functools.partial(iteration, coefs=coefs, **kwargs)
    return lax.fori_loop(0, len(coefs), body, (u, True))

  u, _ = iterate(
      u, coefs=qr_coefs, update_fn=_use_qr, test_convergence=False
  )
  u, is_not_converged = iterate(
      u, coefs=chol_coefs, update_fn=_use_cholesky, test_convergence=True
  )

  # If l has converged but u still has not, continue with Halley's method
  # (coef = None) until convergence.
  def cond_fun(state):
    k, _, is_not_converged = state
    return jnp.logical_and(is_not_converged, k < max_iterations)

  def body_fun(state):
    k, u, is_not_converged = state
    u, is_not_converged = iteration(
        k,
        (u, is_not_converged),
        coefs=None,
        update_fn=_use_cholesky,
        test_convergence=True,
    )
    return k + 1, u, is_not_converged

  k = len(qr_coefs) + len(chol_coefs)
  num_iters, u, is_not_converged = lax.while_loop(
      cond_fun, body_fun, (k, u, is_not_converged)
  )

  # Applies Newton-Schulz refinement for better accuracy.
  u = 1.5 * u - 0.5 * u @ (u.T.conj() @ u)

  h = u.T.conj() @ x
  h = (h + h.T.conj()) / 2

  # Converged within the maximum number of iterations.
  is_converged = jnp.logical_not(is_not_converged)

  return u, h, num_iters, is_converged


# TODO: Add pivoting.
@functools.partial(
    jax.jit, static_argnames=('is_hermitian', 'max_iterations', 'eps')
)
def qdwh(
    x,
    *,
    is_hermitian: bool = False,
    max_iterations: int | None = None,
    eps: float | None = None,
    dynamic_shape: tuple[int, int] | None = None,
):
  """QR-based dynamically weighted Halley iteration for polar decomposition.

  Args:
    x: A full-rank matrix, with shape `M x N`. The matrix may be padded up to
      that size from a smaller true shape (``dynamic_shape``).
    is_hermitian: True if `x` is Hermitian. Default to `False`. This parameter
      is currently unused, but exists for backward compatibility.
    eps: The final result will satisfy ``|x_k - x_k-1| < |x_k| *
      (4*eps)**(1/3)`` where `x_k` is the iterate.
    max_iterations: Iterations will terminate after this many steps even if the
      above is unsatisfied.
    dynamic_shape: the unpadded shape as an ``(m, n)`` tuple; optional.

  Returns:
    A four-tuple of (u, h, num_iters, is_converged) containing the
    polar decomposition of `x = u * h`, the number of iterations to compute `u`,
    and `is_converged`, whose value is `True` when the convergence is achieved
    within the maximum number of iterations.
  """
  # TODO: Possibly take advantage of Hermitian inputs to speed up the QDWH step.
  is_hermitian = core.concrete_or_error(
      bool, is_hermitian, 'The `is_hermitian` argument must be statically '
      'specified to use `qdwh` within JAX transformations.')

  if max_iterations is None:
    max_iterations = 10
  else:
    max_iterations = core.concrete_or_error(
        int, max_iterations, 'The `max_iterations` argument must be statically '
        'specified to use `qdwh` within JAX transformations.')

  M, N = x.shape
  if M < N:
    raise ValueError('The input matrix of shape M x N must have M >= N.')
  if dynamic_shape is not None:
    m, n = dynamic_shape
    x = _mask(x, (m, n))
  else:
    m, n = M, N

  with jax.default_matmul_precision('float32'):
    u, h, num_iters, is_converged = _qdwh(x, m, n, max_iterations, eps)

  return u, h, num_iters, is_converged
