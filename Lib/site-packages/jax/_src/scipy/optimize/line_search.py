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

from __future__ import annotations

from typing import NamedTuple
from functools import partial

from jax._src.numpy.util import promote_dtypes_inexact
import jax.numpy as jnp
import jax
from jax import lax

_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)


def _cubicmin(a, fa, fpa, b, fb, c, fc):
  dtype = jnp.result_type(a, fa, fpa, b, fb, c, fc)
  C = fpa
  db = b - a
  dc = c - a
  denom = (db * dc) ** 2 * (db - dc)
  d1 = jnp.array([[dc ** 2, -db ** 2],
                  [-dc ** 3, db ** 3]], dtype=dtype)
  d2 = jnp.array([fb - fa - C * db, fc - fa - C * dc], dtype=dtype)
  A, B = _dot(d1, d2) / denom

  radical = B * B - 3. * A * C
  xmin = a + (-B + jnp.sqrt(radical)) / (3. * A)

  return xmin


def _quadmin(a, fa, fpa, b, fb):
  D = fa
  C = fpa
  db = b - a
  B = (fb - D - C * db) / (db ** 2)
  xmin = a - C / (2. * B)
  return xmin


def _binary_replace(replace_bit, original_dict, new_dict, keys=None):
  if keys is None:
    keys = new_dict.keys()
  return {key: jnp.where(replace_bit, new_dict[key], original_dict[key])
          for key in keys}


class _ZoomState(NamedTuple):
  done: bool | jax.Array
  failed: bool | jax.Array
  j: int | jax.Array
  a_lo: float | jax.Array
  phi_lo: float | jax.Array
  dphi_lo: float | jax.Array
  a_hi: float | jax.Array
  phi_hi: float | jax.Array
  dphi_hi: float | jax.Array
  a_rec: float | jax.Array
  phi_rec: float | jax.Array
  a_star: float | jax.Array
  phi_star: float | jax.Array
  dphi_star: float | jax.Array
  g_star: float | jax.Array
  nfev: int | jax.Array
  ngev: int | jax.Array


def _zoom(restricted_func_and_grad, wolfe_one, wolfe_two, a_lo, phi_lo,
          dphi_lo, a_hi, phi_hi, dphi_hi, g_0, pass_through):
  """
  Implementation of zoom. Algorithm 3.6 from Wright and Nocedal, 'Numerical
  Optimization', 1999, pg. 59-61. Tries cubic, quadratic, and bisection methods
  of zooming.
  """
  state = _ZoomState(
      done=False,
      failed=False,
      j=0,
      a_lo=a_lo,
      phi_lo=phi_lo,
      dphi_lo=dphi_lo,
      a_hi=a_hi,
      phi_hi=phi_hi,
      dphi_hi=dphi_hi,
      a_rec=(a_lo + a_hi) / 2.,
      phi_rec=(phi_lo + phi_hi) / 2.,
      a_star=1.,
      phi_star=phi_lo,
      dphi_star=dphi_lo,
      g_star=g_0,
      nfev=0,
      ngev=0,
  )
  delta1 = 0.2
  delta2 = 0.1

  def body(state):
    # Body of zoom algorithm. We use boolean arithmetic to avoid using jax.cond
    # so that it works on GPU/TPU.
    dalpha = (state.a_hi - state.a_lo)
    a = jnp.minimum(state.a_hi, state.a_lo)
    b = jnp.maximum(state.a_hi, state.a_lo)
    cchk = delta1 * dalpha
    qchk = delta2 * dalpha

    # This will cause the line search to stop, and since the Wolfe conditions
    # are not satisfied the minimization should stop too.
    threshold = jnp.where((jnp.finfo(dalpha.dtype).bits < 64), 1e-5, 1e-10)
    state = state._replace(failed=state.failed | (dalpha <= threshold))

    # Cubmin is sometimes nan, though in this case the bounds check will fail.
    a_j_cubic = _cubicmin(state.a_lo, state.phi_lo, state.dphi_lo, state.a_hi,
                          state.phi_hi, state.a_rec, state.phi_rec)
    use_cubic = (state.j > 0) & (a_j_cubic > a + cchk) & (a_j_cubic < b - cchk)
    a_j_quad = _quadmin(state.a_lo, state.phi_lo, state.dphi_lo, state.a_hi, state.phi_hi)
    use_quad = (~use_cubic) & (a_j_quad > a + qchk) & (a_j_quad < b - qchk)
    a_j_bisection = (state.a_lo + state.a_hi) / 2.
    use_bisection = (~use_cubic) & (~use_quad)

    a_j = jnp.where(use_cubic, a_j_cubic, state.a_rec)
    a_j = jnp.where(use_quad, a_j_quad, a_j)
    a_j = jnp.where(use_bisection, a_j_bisection, a_j)

    # TODO(jakevdp): should we use some sort of fixed-point approach here instead?
    phi_j, dphi_j, g_j = restricted_func_and_grad(a_j)
    phi_j = phi_j.astype(state.phi_lo.dtype)
    dphi_j = dphi_j.astype(state.dphi_lo.dtype)
    g_j = g_j.astype(state.g_star.dtype)
    state = state._replace(nfev=state.nfev + 1,
                           ngev=state.ngev + 1)

    hi_to_j = wolfe_one(a_j, phi_j) | (phi_j >= state.phi_lo)
    star_to_j = wolfe_two(dphi_j) & (~hi_to_j)
    hi_to_lo = (dphi_j * (state.a_hi - state.a_lo) >= 0.) & (~hi_to_j) & (~star_to_j)
    lo_to_j = (~hi_to_j) & (~star_to_j)

    state = state._replace(
        **_binary_replace(
            hi_to_j,
            state._asdict(),
            dict(
                a_hi=a_j,
                phi_hi=phi_j,
                dphi_hi=dphi_j,
                a_rec=state.a_hi,
                phi_rec=state.phi_hi,
            ),
        ),
    )

    # for termination
    state = state._replace(
        done=star_to_j | state.done,
        **_binary_replace(
            star_to_j,
            state._asdict(),
            dict(
                a_star=a_j,
                phi_star=phi_j,
                dphi_star=dphi_j,
                g_star=g_j,
            )
        ),
    )
    state = state._replace(
        **_binary_replace(
            hi_to_lo,
            state._asdict(),
            dict(
                a_hi=state.a_lo,
                phi_hi=state.phi_lo,
                dphi_hi=state.dphi_lo,
                a_rec=state.a_hi,
                phi_rec=state.phi_hi,
            ),
        ),
    )
    state = state._replace(
        **_binary_replace(
            lo_to_j,
            state._asdict(),
            dict(
                a_lo=a_j,
                phi_lo=phi_j,
                dphi_lo=dphi_j,
                a_rec=state.a_lo,
                phi_rec=state.phi_lo,
            ),
        ),
    )
    state = state._replace(j=state.j + 1)
    # Choose higher cutoff for maxiter than Scipy as Jax takes longer to find
    # the same value - possibly floating point issues?
    state = state._replace(failed= state.failed | (state.j >= 30))
    return state

  state = lax.while_loop(lambda state: (~state.done) & (~pass_through) & (~state.failed),
                         body,
                         state)

  return state


class _LineSearchState(NamedTuple):
  done: bool | jax.Array
  failed: bool | jax.Array
  i: int | jax.Array
  a_i1: float | jax.Array
  phi_i1: float | jax.Array
  dphi_i1: float | jax.Array
  nfev: int | jax.Array
  ngev: int | jax.Array
  a_star: float | jax.Array
  phi_star: float | jax.Array
  dphi_star: float | jax.Array
  g_star: jax.Array


class _LineSearchResults(NamedTuple):
  """Results of line search.

  Parameters:
    failed: True if the strong Wolfe criteria were satisfied
    nit: integer number of iterations
    nfev: integer number of functions evaluations
    ngev: integer number of gradients evaluations
    k: integer number of iterations
    a_k: integer step size
    f_k: final function value
    g_k: final gradient value
    status: integer end status
  """
  failed: bool | jax.Array
  nit: int | jax.Array
  nfev: int | jax.Array
  ngev: int | jax.Array
  k: int | jax.Array
  a_k: int | jax.Array
  f_k: jax.Array
  g_k: jax.Array
  status: bool | jax.Array


def line_search(f, xk, pk, old_fval=None, old_old_fval=None, gfk=None, c1=1e-4,
                c2=0.9, maxiter=20):
  """Inexact line search that satisfies strong Wolfe conditions.

  Algorithm 3.5 from Wright and Nocedal, 'Numerical Optimization', 1999, pg. 59-61

  Args:
    fun: function of the form f(x) where x is a flat ndarray and returns a real
      scalar. The function should be composed of operations with vjp defined.
    x0: initial guess.
    pk: direction to search in. Assumes the direction is a descent direction.
    old_fval, gfk: initial value of value_and_gradient as position.
    old_old_fval: unused argument, only for scipy API compliance.
    maxiter: maximum number of iterations to search
    c1, c2: Wolfe criteria constant, see ref.

  Returns: LineSearchResults
  """
  xk, pk = promote_dtypes_inexact(xk, pk)
  def restricted_func_and_grad(t):
    t = jnp.array(t, dtype=pk.dtype)
    phi, g = jax.value_and_grad(f)(xk + t * pk)
    dphi = jnp.real(_dot(g, pk))
    return phi, dphi, g

  if old_fval is None or gfk is None:
    phi_0, dphi_0, gfk = restricted_func_and_grad(0)
  else:
    phi_0 = old_fval
    dphi_0 = jnp.real(_dot(gfk, pk))
  if old_old_fval is not None:
    candidate_start_value = 1.01 * 2 * (phi_0 - old_old_fval) / dphi_0
    start_value = jnp.where(candidate_start_value > 1, 1.0, candidate_start_value)
  else:
    start_value = 1

  def wolfe_one(a_i, phi_i):
    # actually negation of W1
    return phi_i > phi_0 + c1 * a_i * dphi_0

  def wolfe_two(dphi_i):
    return jnp.abs(dphi_i) <= -c2 * dphi_0

  state = _LineSearchState(
      done=False,
      failed=False,
      # algorithm begins at 1 as per Wright and Nocedal, however Scipy has a
      # bug and starts at 0. See https://github.com/scipy/scipy/issues/12157
      i=1,
      a_i1=0.,
      phi_i1=phi_0,
      dphi_i1=dphi_0,
      nfev=1 if (old_fval is None or gfk is None) else 0,
      ngev=1 if (old_fval is None or gfk is None) else 0,
      a_star=0.,
      phi_star=phi_0,
      dphi_star=dphi_0,
      g_star=gfk,
  )

  def body(state):
    # no amax in this version, we just double as in scipy.
    # unlike original algorithm we do our next choice at the start of this loop
    a_i = jnp.where(state.i == 1, start_value, state.a_i1 * 2.)

    phi_i, dphi_i, g_i = restricted_func_and_grad(a_i)
    state = state._replace(nfev=state.nfev + 1,
                           ngev=state.ngev + 1)

    star_to_zoom1 = wolfe_one(a_i, phi_i) | ((phi_i >= state.phi_i1) & (state.i > 1))
    star_to_i = wolfe_two(dphi_i) & (~star_to_zoom1)
    star_to_zoom2 = (dphi_i >= 0.) & (~star_to_zoom1) & (~star_to_i)

    zoom1 = _zoom(restricted_func_and_grad,
                  wolfe_one,
                  wolfe_two,
                  state.a_i1,
                  state.phi_i1,
                  state.dphi_i1,
                  a_i,
                  phi_i,
                  dphi_i,
                  gfk,
                  ~star_to_zoom1)

    state = state._replace(nfev=state.nfev + zoom1.nfev,
                           ngev=state.ngev + zoom1.ngev)

    zoom2 = _zoom(restricted_func_and_grad,
                  wolfe_one,
                  wolfe_two,
                  a_i,
                  phi_i,
                  dphi_i,
                  state.a_i1,
                  state.phi_i1,
                  state.dphi_i1,
                  gfk,
                  ~star_to_zoom2)

    state = state._replace(nfev=state.nfev + zoom2.nfev,
                           ngev=state.ngev + zoom2.ngev)

    state = state._replace(
        done=star_to_zoom1 | state.done,
        failed=(star_to_zoom1 & zoom1.failed) | state.failed,
        **_binary_replace(
            star_to_zoom1,
            state._asdict(),
            zoom1._asdict(),
            keys=['a_star', 'phi_star', 'dphi_star', 'g_star'],
        ),
    )
    state = state._replace(
        done=star_to_i | state.done,
        **_binary_replace(
            star_to_i,
            state._asdict(),
            dict(
                a_star=a_i,
                phi_star=phi_i,
                dphi_star=dphi_i,
                g_star=g_i,
            ),
        ),
    )
    state = state._replace(
        done=star_to_zoom2 | state.done,
        failed=(star_to_zoom2 & zoom2.failed) | state.failed,
        **_binary_replace(
            star_to_zoom2,
            state._asdict(),
            zoom2._asdict(),
            keys=['a_star', 'phi_star', 'dphi_star', 'g_star'],
        ),
    )
    state = state._replace(i=state.i + 1, a_i1=a_i, phi_i1=phi_i, dphi_i1=dphi_i)
    return state

  state = lax.while_loop(lambda state: (~state.done) & (state.i <= maxiter) & (~state.failed),
                         body,
                         state)

  status = jnp.where(
      state.failed,
      jnp.array(1),  # zoom failed
          jnp.where(
              state.i > maxiter,
              jnp.array(3),  # maxiter reached
              jnp.array(0),  # passed (should be)
          ),
  )
  # Step sizes which are too small causes the optimizer to get stuck with a
  # direction of zero in <64 bit mode - avoid with a floor on minimum step size.
  alpha_k = jnp.asarray(state.a_star)
  alpha_k = jnp.where((jnp.finfo(alpha_k.dtype).bits != 64)
                    & (jnp.abs(alpha_k) < 1e-8),
                      jnp.sign(alpha_k) * 1e-8,
                      alpha_k)
  results = _LineSearchResults(
      failed=state.failed | (~state.done),
      nit=state.i - 1,  # because iterations started at 1
      nfev=state.nfev,
      ngev=state.ngev,
      k=state.i,
      a_k=alpha_k,
      f_k=state.phi_star,
      g_k=state.g_star,
      status=status,
  )
  return results
