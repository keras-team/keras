# Copyright 2023 The JAX Authors.
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
from jax._src.numpy.util import promote_args_inexact
from jax._src.lax.lax import _const as _lax_const
from jax._src.scipy.special import gammaln, xlogy, xlog1py
from jax._src.typing import Array, ArrayLike


def logpmf(k: ArrayLike, n: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Binomial log probability mass function.

  JAX implementation of :obj:`scipy.stats.binom` ``logpmf``.

  The binomial probability mass function is defined as

  .. math::

     f(k, n, p) = {n \choose k}p^k(1-p)^{n-k}

  for :math:`0\le p\le 1` and non-negative integers :math:`k`.

  Args:
    k: arraylike, value at which to evaluate the PMF
    n: arraylike, distribution shape parameter
    p: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of logpmf values.

  See Also:
    :func:`jax.scipy.stats.binom.pmf`
  """
  k, n, p, loc = promote_args_inexact("binom.logpmf", k, n, p, loc)
  y = lax.sub(k, loc)
  zero = _lax_const(y, 0)
  comb_term = lax.sub(
      gammaln(n + 1),
      lax.add(gammaln(y + 1), gammaln(n - y + 1))
  )
  log_linear_term = lax.add(xlogy(y, p), xlog1py(lax.sub(n, y), lax.neg(p)))
  log_probs = lax.add(comb_term, log_linear_term)
  y_n_cond = jnp.logical_or(jnp.logical_and(lax.eq(y, zero), lax.eq(n, zero)),
                            lax.eq(log_linear_term, zero))
  log_probs = jnp.where(y_n_cond, 0., log_probs)
  return jnp.where(lax.ge(k, loc) & lax.lt(k, loc + n + 1), log_probs, -jnp.inf)


def pmf(k: ArrayLike, n: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Binomial probability mass function.

  JAX implementation of :obj:`scipy.stats.binom` ``pmf``.

  The binomial probability mass function is defined as

  .. math::

     f(k, n, p) = {n \choose k}p^k(1-p)^{n-k}

  for :math:`0\le p\le 1` and non-negative integers :math:`k`.

  Args:
    k: arraylike, value at which to evaluate the PMF
    n: arraylike, distribution shape parameter
    p: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
      array of pmf values.

  See Also:
    :func:`jax.scipy.stats.binom.logpmf`
  """
  return lax.exp(logpmf(k, n, p, loc))
