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
from jax._src.scipy.special import gammaln, xlogy
from jax._src.typing import Array, ArrayLike


def logpmf(k: ArrayLike, n: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Negative-binomial log probability mass function.

  JAX implementation of :obj:`scipy.stats.nbinom` ``logpmf``.

  The negative-binomial probability mass function is given by

  .. math::

     f(k) = {{k+n-1} \choose {n-1}}p^n(1-p)^k

  for :math:`k \ge 0` and :math:`0 \le p \le 1`.

  Args:
    k: arraylike, value at which to evaluate the PMF
    n: arraylike, distribution shape parameter
    p: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of logpdf values.

  See Also:
    :func:`jax.scipy.stats.nbinom.pmf`
  """
  k, n, p, loc = promote_args_inexact("nbinom.logpmf", k, n, p, loc)
  one = _lax_const(k, 1)
  y = lax.sub(k, loc)
  comb_term = lax.sub(
    lax.sub(gammaln(lax.add(y, n)), gammaln(n)), gammaln(lax.add(y, one))
  )
  log_linear_term = lax.add(xlogy(n, p), xlogy(y, lax.sub(one, p)))
  log_probs = lax.add(comb_term, log_linear_term)
  return jnp.where(lax.lt(k, loc), -jnp.inf, log_probs)


def pmf(k: ArrayLike, n: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Negative-binomial probability mass function.

  JAX implementation of :obj:`scipy.stats.nbinom` ``pmf``.

  The negative-binomial probability mass function is given by

  .. math::

     f(k) = {{k+n-1} \choose {n-1}}p^n(1-p)^k

  for :math:`k \ge 0` and :math:`0 \le p \le 1`.

  Args:
    k: arraylike, value at which to evaluate the PMF
    n: arraylike, distribution shape parameter
    p: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of pmf values.

  See Also:
    :func:`jax.scipy.stats.nbinom.logpmf`
  """
  return lax.exp(logpmf(k, n, p, loc))
