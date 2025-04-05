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

from jax import lax
import jax.numpy as jnp
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import promote_args_inexact
from jax._src.scipy.special import betaln
from jax._src.typing import Array, ArrayLike


def logpmf(k: ArrayLike, n: ArrayLike, a: ArrayLike, b: ArrayLike,
           loc: ArrayLike = 0) -> Array:
  r"""Beta-binomial log probability mass function.

  JAX implementation of :obj:`scipy.stats.betabinom` ``logpmf``

  The beta-binomial distribution's probability mass function is defined as

  .. math::

     f(k, n, a, b) = {n \choose k}\frac{B(k+a,n-k-b)}{B(a,b)}

  where :math:`B(a, b)` is the :func:`~jax.scipy.special.beta` function. It is
  defined for :math:`n\ge 0`, :math:`a>0`, :math:`b>0`, and non-negative integers `k`.

  Args:
    k: arraylike, value at which to evaluate the PMF
    n: arraylike, distribution shape parameter
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of logpmf values

  See Also:
    :func:`jax.scipy.stats.betabinom.pmf`
  """
  k, n, a, b, loc = promote_args_inexact("betabinom.logpmf", k, n, a, b, loc)
  y = lax.sub(lax.floor(k), loc)
  one = _lax_const(y, 1)
  zero = _lax_const(y, 0)
  combiln = lax.neg(lax.add(lax.log1p(n), betaln(lax.add(lax.sub(n,y), one), lax.add(y,one))))
  beta_lns = lax.sub(betaln(lax.add(y,a), lax.add(lax.sub(n,y),b)), betaln(a,b))
  log_probs = lax.add(combiln, beta_lns)
  log_probs = jnp.where(jnp.logical_and(lax.eq(y, zero), lax.eq(n, zero)), 0., log_probs)
  y_cond = jnp.logical_or(jnp.logical_or(lax.lt(y, lax.neg(loc)), lax.gt(y, n)),
                          lax.le(lax.add(y, a), zero))
  log_probs = jnp.where(y_cond, -jnp.inf, log_probs)
  n_a_b_cond = jnp.logical_or(jnp.logical_or(lax.lt(n, zero), lax.le(a, zero)), lax.le(b, zero))
  return jnp.where(n_a_b_cond, jnp.nan, log_probs)


def pmf(k: ArrayLike, n: ArrayLike, a: ArrayLike, b: ArrayLike,
        loc: ArrayLike = 0) -> Array:
  r"""Beta-binomial probability mass function.

  JAX implementation of :obj:`scipy.stats.betabinom` ``pmf``.

  The beta-binomial distribution's probability mass function is defined as

  .. math::

     f(k, n, a, b) = {n \choose k}\frac{B(k+a,n-k-b)}{B(a,b)}

  where :math:`B(a, b)` is the :func:`~jax.scipy.special.beta` function. It is
  defined for :math:`n\ge 0`, :math:`a>0`, :math:`b>0`, and non-negative integers `k`.

  Args:
    k: arraylike, value at which to evaluate the PMF
    n: arraylike, distribution shape parameter
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of pmf values

  See Also:
    :func:`jax.scipy.stats.betabinom.logpmf`
  """
  return lax.exp(logpmf(k, n, a, b, loc))
