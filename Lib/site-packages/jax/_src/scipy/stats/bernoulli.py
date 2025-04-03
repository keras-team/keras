# Copyright 2018 The JAX Authors.
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

from jax import lax
import jax.numpy as jnp
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import promote_args_inexact
from jax._src.typing import Array, ArrayLike
from jax.scipy.special import xlogy, xlog1py


def logpmf(k: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Bernoulli log probability mass function.

  JAX implementation of :obj:`scipy.stats.bernoulli` ``logpmf``

  The Bernoulli probability mass function is defined as

  .. math::

     f(k) = \begin{cases}
       1 - p, & k = 0 \\
       p, & k = 1 \\
       0, & \mathrm{otherwise}
     \end{cases}

  Args:
    k: arraylike, value at which to evaluate the PMF
    p: arraylike, distribution shape parameter
    loc: arraylike, distribution offset

  Returns:
    array of logpmf values

  See Also:
    - :func:`jax.scipy.stats.bernoulli.cdf`
    - :func:`jax.scipy.stats.bernoulli.pmf`
    - :func:`jax.scipy.stats.bernoulli.ppf`
  """
  k, p, loc = promote_args_inexact("bernoulli.logpmf", k, p, loc)
  zero = _lax_const(k, 0)
  one = _lax_const(k, 1)
  x = lax.sub(k, loc)
  log_probs = xlogy(x, p) + xlog1py(lax.sub(one, x), -p)
  return jnp.where(jnp.logical_or(lax.lt(x, zero), lax.gt(x, one)),
                  -jnp.inf, log_probs)


def pmf(k: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Bernoulli probability mass function.

  JAX implementation of :obj:`scipy.stats.bernoulli` ``pmf``

  The Bernoulli probability mass function is defined as

  .. math::

     f(k) = \begin{cases}
       1 - p, & k = 0 \\
       p, & k = 1 \\
       0, & \mathrm{otherwise}
     \end{cases}

  Args:
    k: arraylike, value at which to evaluate the PMF
    p: arraylike, distribution shape parameter
    loc: arraylike, distribution offset

  Returns:
    array of pmf values

  See Also:
    - :func:`jax.scipy.stats.bernoulli.cdf`
    - :func:`jax.scipy.stats.bernoulli.logpmf`
    - :func:`jax.scipy.stats.bernoulli.ppf`
  """
  return jnp.exp(logpmf(k, p, loc))


def cdf(k: ArrayLike, p: ArrayLike) -> Array:
  r"""Bernoulli cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.bernoulli` ``cdf``

  The Bernoulli cumulative distribution function is defined as:

  .. math::

     f_{cdf}(k, p) = \sum_{i=0}^k f_{pmf}(k, p)

  where :math:`f_{pmf}(k, p)` is the Bernoulli probability mass function
  :func:`jax.scipy.stats.bernoulli.pmf`.

  Args:
    k: arraylike, value at which to evaluate the CDF
    p: arraylike, distribution shape parameter
    loc: arraylike, distribution offset

  Returns:
    array of cdf values

  See Also:
    - :func:`jax.scipy.stats.bernoulli.logpmf`
    - :func:`jax.scipy.stats.bernoulli.pmf`
    - :func:`jax.scipy.stats.bernoulli.ppf`
  """
  k, p = promote_args_inexact('bernoulli.cdf', k, p)
  zero, one = _lax_const(k, 0), _lax_const(k, 1)
  conds = [
    jnp.isnan(k) | jnp.isnan(p) | (p < zero) | (p > one),
    lax.lt(k, zero),
    jnp.logical_and(lax.ge(k, zero), lax.lt(k, one)),
    lax.ge(k, one)
    ]
  vals = [jnp.nan, zero, one - p, one]
  return jnp.select(conds, vals)


def ppf(q: ArrayLike, p: ArrayLike) -> Array:
  """Bernoulli percent point function.

  JAX implementation of :obj:`scipy.stats.bernoulli` ``ppf``

  The percent point function is the inverse of the cumulative
  distribution function, :func:`jax.scipy.stats.bernoulli.cdf`.

  Args:
    k: arraylike, value at which to evaluate the PPF
    p: arraylike, distribution shape parameter
    loc: arraylike, distribution offset

  Returns:
    array of ppf values

  See Also:
    - :func:`jax.scipy.stats.bernoulli.cdf`
    - :func:`jax.scipy.stats.bernoulli.logpmf`
    - :func:`jax.scipy.stats.bernoulli.pmf`
  """
  q, p = promote_args_inexact('bernoulli.ppf', q, p)
  zero, one = _lax_const(q, 0), _lax_const(q, 1)
  return jnp.where(
    jnp.isnan(q) | jnp.isnan(p) | (p < zero) | (p > one) | (q < zero) | (q > one),
    jnp.nan,
    jnp.where(lax.le(q, one - p), zero, one)
  )
