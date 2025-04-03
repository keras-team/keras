# Copyright 2022 The JAX Authors.
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
from jax._src.numpy.util import promote_args_inexact, promote_args_numeric
from jax._src.scipy.special import gammaln, xlogy
from jax._src.typing import Array, ArrayLike


def logpmf(x: ArrayLike, n: ArrayLike, p: ArrayLike) -> Array:
  r"""Multinomial log probability mass function.

  JAX implementation of :obj:`scipy.stats.multinomial` ``logpdf``.

  The multinomial probability distribution is given by

  .. math::

     f(x, n, p) = n! \prod_{i=1}^k \frac{p_i^{x_i}}{x_i!}

  with :math:`n = \sum_i x_i`.

  Args:
    x: arraylike, value at which to evaluate the PMF
    n: arraylike, distribution shape parameter
    p: arraylike, distribution shape parameter

  Returns:
    array of logpmf values.

  See Also:
    :func:`jax.scipy.stats.multinomial.pmf`
  """
  p, = promote_args_inexact("multinomial.logpmf", p)
  x, n = promote_args_numeric("multinomial.logpmf", x, n)
  if not jnp.issubdtype(x.dtype, jnp.integer):
    raise ValueError(f"x and n must be of integer type; got x.dtype={x.dtype}, n.dtype={n.dtype}")
  x = x.astype(p.dtype)
  n = n.astype(p.dtype)
  logprobs = gammaln(n + 1) + jnp.sum(xlogy(x, p) - gammaln(x + 1), axis=-1)
  return jnp.where(jnp.equal(jnp.sum(x), n), logprobs, -jnp.inf)


def pmf(x: ArrayLike, n: ArrayLike, p: ArrayLike) -> Array:
  r"""Multinomial probability mass function.

  JAX implementation of :obj:`scipy.stats.multinomial` ``pmf``.

  The multinomial probability distribution is given by

  .. math::

     f(x, n, p) = n! \prod_{i=1}^k \frac{p_i^{x_i}}{x_i!}

  with :math:`n = \sum_i x_i`.

  Args:
    x: arraylike, value at which to evaluate the PMF
    n: arraylike, distribution shape parameter
    p: arraylike, distribution shape parameter

  Returns:
    array of pmf values

  See Also:
    :func:`jax.scipy.stats.multinomial.logpmf`
  """
  return lax.exp(logpmf(x, n, p))
