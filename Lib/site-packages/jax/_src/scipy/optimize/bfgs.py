# Copyright 2020 The JAX Authors.
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
# limitations under the License.
"""The Broyden-Fletcher-Goldfarb-Shanno minimization algorithm."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.scipy.optimize.line_search import line_search


class _BFGSResults(NamedTuple):
  """Results from BFGS optimization.

  Parameters:
    converged: True if minimization converged.
    failed: True if line search failed.
    k: integer the number of iterations of the BFGS update.
    nfev: integer total number of objective evaluations performed.
    ngev: integer total number of jacobian evaluations
    nhev: integer total number of hessian evaluations
    x_k: array containing the last argument value found during the search. If
      the search converged, then this value is the argmin of the objective
      function.
    f_k: array containing the value of the objective function at `x_k`. If the
      search converged, then this is the (local) minimum of the objective
      function.
    g_k: array containing the gradient of the objective function at `x_k`. If
      the search converged the l2-norm of this tensor should be below the
      tolerance.
    H_k: array containing the inverse of the estimated Hessian.
    status: int describing end state.
    line_search_status: int describing line search end state (only means
      something if line search fails).
  """
  converged: bool | jax.Array
  failed: bool | jax.Array
  k: int | jax.Array
  nfev: int | jax.Array
  ngev: int | jax.Array
  nhev: int | jax.Array
  x_k: jax.Array
  f_k: jax.Array
  g_k: jax.Array
  H_k: jax.Array
  old_old_fval: jax.Array
  status: int | jax.Array
  line_search_status: int | jax.Array


_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)


def minimize_bfgs(
    fun: Callable,
    x0: jax.Array,
    maxiter: int | None = None,
    norm=jnp.inf,
    gtol: float = 1e-5,
    line_search_maxiter: int = 10,
) -> _BFGSResults:
  """Minimize a function using BFGS.

  Implements the BFGS algorithm from
    Algorithm 6.1 from Wright and Nocedal, 'Numerical Optimization', 1999, pg.
    136-143.

  Args:
    fun: function of the form f(x) where x is a flat ndarray and returns a real
      scalar. The function should be composed of operations with vjp defined.
    x0: initial guess.
    maxiter: maximum number of iterations.
    norm: order of norm for convergence check. Default inf.
    gtol: terminates minimization when |grad|_norm < g_tol.
    line_search_maxiter: maximum number of linesearch iterations.

  Returns:
    Optimization result.
  """

  if maxiter is None:
    maxiter = jnp.size(x0) * 200

  d = x0.shape[0]

  initial_H = jnp.eye(d, dtype=x0.dtype)
  f_0, g_0 = jax.value_and_grad(fun)(x0)
  state = _BFGSResults(
      converged=jnp.linalg.norm(g_0, ord=norm) < gtol,
      failed=False,
      k=0,
      nfev=1,
      ngev=1,
      nhev=0,
      x_k=x0,
      f_k=f_0,
      g_k=g_0,
      H_k=initial_H,
      old_old_fval=f_0 + jnp.linalg.norm(g_0) / 2,
      status=0,
      line_search_status=0,
  )

  def cond_fun(state):
    return (jnp.logical_not(state.converged)
            & jnp.logical_not(state.failed)
            & (state.k < maxiter))

  def body_fun(state):
    p_k = -_dot(state.H_k, state.g_k)
    line_search_results = line_search(
        fun,
        state.x_k,
        p_k,
        old_fval=state.f_k,
        old_old_fval=state.old_old_fval,
        gfk=state.g_k,
        maxiter=line_search_maxiter,
    )
    state = state._replace(
        nfev=state.nfev + line_search_results.nfev,
        ngev=state.ngev + line_search_results.ngev,
        failed=line_search_results.failed,
        line_search_status=line_search_results.status,
    )
    s_k = line_search_results.a_k * p_k
    x_kp1 = state.x_k + s_k
    f_kp1 = line_search_results.f_k
    g_kp1 = line_search_results.g_k
    y_k = g_kp1 - state.g_k
    rho_k = jnp.reciprocal(_dot(y_k, s_k))

    sy_k = s_k[:, jnp.newaxis] * y_k[jnp.newaxis, :]
    w = jnp.eye(d, dtype=rho_k.dtype) - rho_k * sy_k
    H_kp1 = (_einsum('ij,jk,lk', w, state.H_k, w)
             + rho_k * s_k[:, jnp.newaxis] * s_k[jnp.newaxis, :])
    H_kp1 = jnp.where(jnp.isfinite(rho_k), H_kp1, state.H_k)
    converged = jnp.linalg.norm(g_kp1, ord=norm) < gtol

    state = state._replace(
        converged=converged,
        k=state.k + 1,
        x_k=x_kp1,
        f_k=f_kp1,
        g_k=g_kp1,
        H_k=H_kp1,
        old_old_fval=state.f_k,
    )
    return state

  state = lax.while_loop(cond_fun, body_fun, state)
  status = jnp.where(
      state.converged,
      0,  # converged
      jnp.where(
          state.k == maxiter,
          1,  # max iters reached
          jnp.where(
              state.failed,
              2 + state.line_search_status, # ls failed (+ reason)
              -1,  # undefined
          )
      )
  )
  state = state._replace(status=status)
  return state
