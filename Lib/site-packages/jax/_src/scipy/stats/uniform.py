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
from jax import numpy as jnp
from jax.numpy import where, inf, logical_or
from jax._src.typing import Array, ArrayLike
from jax._src.numpy.util import promote_args_inexact


def logpdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Uniform log probability distribution function.

  JAX implementation of :obj:`scipy.stats.uniform` ``logpdf``.

  The uniform distribution pdf is given by

  .. math::

     f(x) = \begin{cases}
       1 & 0 \le x \le 1 \\
       0 & \mathrm{otherwise}
     \end{cases}

  Args:
    x: arraylike, value at which to evaluate the PDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logpdf values

  See Also:
    - :func:`jax.scipy.stats.uniform.cdf`
    - :func:`jax.scipy.stats.uniform.pdf`
    - :func:`jax.scipy.stats.uniform.ppf`
  """
  x, loc, scale = promote_args_inexact("uniform.logpdf", x, loc, scale)
  log_probs = lax.neg(lax.log(scale))
  return where(logical_or(lax.gt(x, lax.add(loc, scale)),
                          lax.lt(x, loc)),
               -inf, log_probs)


def pdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Uniform probability distribution function.

  JAX implementation of :obj:`scipy.stats.uniform` ``pdf``.

  The uniform distribution pdf is given by

  .. math::

     f(x) = \begin{cases}
       1 & 0 \le x \le 1 \\
       0 & \mathrm{otherwise}
     \end{cases}

  Args:
    x: arraylike, value at which to evaluate the PDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values.

  See Also:
    - :func:`jax.scipy.stats.uniform.cdf`
    - :func:`jax.scipy.stats.uniform.logpdf`
    - :func:`jax.scipy.stats.uniform.ppf`
  """
  return lax.exp(logpdf(x, loc, scale))


def cdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Uniform cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.uniform` ``cdf``.

  The cdf is defined as

  .. math::

     f_{cdf} = \int_{-\infty}^x f_{pdf}(y) \mathrm{d}y

  where here :math:`f_{pdf}` is the probability distribution function,
  :func:`jax.scipy.stats.uniform.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of cdf values.

  See Also:
    - :func:`jax.scipy.stats.uniform.pdf`
    - :func:`jax.scipy.stats.uniform.logpdf`
    - :func:`jax.scipy.stats.uniform.ppf`
  """
  x, loc, scale = promote_args_inexact("uniform.cdf", x, loc, scale)
  zero, one = jnp.array(0, x.dtype), jnp.array(1, x.dtype)
  conds = [lax.lt(x, loc), lax.gt(x, lax.add(loc, scale)), lax.ge(x, loc) & lax.le(x, lax.add(loc, scale))]
  vals = [zero, one, lax.div(lax.sub(x, loc), scale)]

  return jnp.select(conds, vals)


def ppf(q: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  """Uniform distribution percent point function.

  JAX implementation of :obj:`scipy.stats.uniform` ``ppf``.

  The percent point function is defined as the inverse of the
  cumulative distribution function, :func:`jax.scipy.stats.uniform.cdf`.

  Args:
    q: arraylike, value at which to evaluate the PPF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of ppf values.

  See Also:
    - :func:`jax.scipy.stats.uniform.cdf`
    - :func:`jax.scipy.stats.uniform.pdf`
    - :func:`jax.scipy.stats.uniform.logpdf`
  """
  q, loc, scale = promote_args_inexact("uniform.ppf", q, loc, scale)
  return where(
    jnp.isnan(q) | (q < 0) | (q > 1),
    jnp.nan,
    lax.add(loc, lax.mul(scale, q))
  )
