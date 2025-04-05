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
from jax.scipy.special import xlogy, gammaln, gammaincc


def logpmf(k: ArrayLike, mu: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Poisson log probability mass function.

  JAX implementation of :obj:`scipy.stats.poisson` ``logpmf``.

  The Poisson probability mass function is given by

  .. math::

     f(k) = e^{-\mu}\frac{\mu^k}{k!}

  and is defined for :math:`k \ge 0` and :math:`\mu \ge 0`.

  Args:
    k: arraylike, value at which to evaluate the PMF
    mu: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of logpmf values.

  See Also:
    - :func:`jax.scipy.stats.poisson.cdf`
    - :func:`jax.scipy.stats.poisson.pmf`
  """
  k, mu, loc = promote_args_inexact("poisson.logpmf", k, mu, loc)
  zero = _lax_const(k, 0)
  x = lax.sub(k, loc)
  log_probs = xlogy(x, mu) - gammaln(x + 1) - mu
  return jnp.where(jnp.logical_or(lax.lt(x, zero),
                                  lax.ne(jnp.round(k), k)), -jnp.inf, log_probs)


def pmf(k: ArrayLike, mu: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Poisson probability mass function.

  JAX implementation of :obj:`scipy.stats.poisson` ``pmf``.

  The Poisson probability mass function is given by

  .. math::

     f(k) = e^{-\mu}\frac{\mu^k}{k!}

  and is defined for :math:`k \ge 0` and :math:`\mu \ge 0`.

  Args:
    k: arraylike, value at which to evaluate the PMF
    mu: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of pmf values.

  See Also:
    - :func:`jax.scipy.stats.poisson.cdf`
    - :func:`jax.scipy.stats.poisson.logpmf`
  """
  return jnp.exp(logpmf(k, mu, loc))


def cdf(k: ArrayLike, mu: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Poisson cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.poisson` ``cdf``.

  The cumulative distribution function is defined as:

  .. math::

     f_{cdf}(k, p) = \sum_{i=0}^k f_{pmf}(k, p)

  where :math:`f_{pmf}(k, p)` is the probability mass function
  :func:`jax.scipy.stats.poisson.pmf`.

  Args:
    k: arraylike, value at which to evaluate the CDF
    mu: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of cdf values.

  See Also:
    - :func:`jax.scipy.stats.poisson.pmf`
    - :func:`jax.scipy.stats.poisson.logpmf`
  """
  k, mu, loc = promote_args_inexact("poisson.logpmf", k, mu, loc)
  zero = _lax_const(k, 0)
  x = lax.sub(k, loc)
  p = gammaincc(jnp.floor(1 + x), mu)
  return jnp.where(lax.lt(x, zero), zero, p)
