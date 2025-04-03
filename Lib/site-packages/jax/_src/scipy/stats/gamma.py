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
from jax.scipy.special import gammaln, xlogy, gammainc, gammaincc


def logpdf(x: ArrayLike, a: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Gamma log probability distribution function.

  JAX implementation of :obj:`scipy.stats.gamma` ``logpdf``.

  The Gamma probability distribution is given by

  .. math::

     f(x, a) = \frac{1}{\Gamma(a)}x^{a-1}e^{-x}

  Where :math:`\Gamma(a)` is the :func:`~jax.scipy.special.gamma` function.
  It is defined for :math:`x \ge 0` and :math:`a > 0`.

  Args:
    x: arraylike, value at which to evaluate the PDF
    a: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logpdf values.

  See Also:
    - :func:`jax.scipy.stats.gamma.cdf`
    - :func:`jax.scipy.stats.gamma.pdf`
    - :func:`jax.scipy.stats.gamma.sf`
    - :func:`jax.scipy.stats.gamma.logcdf`
    - :func:`jax.scipy.stats.gamma.logsf`
  """
  x, a, loc, scale = promote_args_inexact("gamma.logpdf", x, a, loc, scale)
  ok = lax.ge(x, loc)
  one = _lax_const(x, 1)
  y = jnp.where(ok, lax.div(lax.sub(x, loc), scale), one)
  log_linear_term = lax.sub(xlogy(lax.sub(a, one), y), y)
  shape_terms = lax.add(gammaln(a), lax.log(scale))
  log_probs = lax.sub(log_linear_term, shape_terms)
  return jnp.where(ok, log_probs, -jnp.inf)


def pdf(x: ArrayLike, a: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Gamma probability distribution function.

  JAX implementation of :obj:`scipy.stats.gamma` ``pdf``.

  The Gamma probability distribution is given by

  .. math::

     f(x, a) = \frac{1}{\Gamma(a)}x^{a-1}e^{-x}

  Where :math:`\Gamma(a)` is the :func:`~jax.scipy.special.gamma` function.
  It is defined for :math:`x \ge 0` and :math:`a > 0`.

  Args:
    x: arraylike, value at which to evaluate the PDF
    a: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values.

  See Also:
    - :func:`jax.scipy.stats.gamma.cdf`
    - :func:`jax.scipy.stats.gamma.sf`
    - :func:`jax.scipy.stats.gamma.logcdf`
    - :func:`jax.scipy.stats.gamma.logpdf`
    - :func:`jax.scipy.stats.gamma.logsf`
  """
  return lax.exp(logpdf(x, a, loc, scale))


def cdf(x: ArrayLike, a: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Gamma cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.gamma` ``cdf``.

  The cdf is defined as

  .. math::

     f_{cdf}(x, a) = \int_{-\infty}^x f_{pdf}(y, a)\mathrm{d}y

  where :math:`f_{pdf}` is the probability density function,
  :func:`jax.scipy.stats.gamma.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    a: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of cdf values.

  See Also:
    - :func:`jax.scipy.stats.gamma.pdf`
    - :func:`jax.scipy.stats.gamma.sf`
    - :func:`jax.scipy.stats.gamma.logcdf`
    - :func:`jax.scipy.stats.gamma.logpdf`
    - :func:`jax.scipy.stats.gamma.logsf`
  """
  x, a, loc, scale = promote_args_inexact("gamma.cdf", x, a, loc, scale)
  return gammainc(
    a,
    lax.clamp(
      _lax_const(x, 0),
      lax.div(lax.sub(x, loc), scale),
      _lax_const(x, jnp.inf),
    )
  )


def logcdf(x: ArrayLike, a: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Gamma log cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.gamma` ``logcdf``.

  The cdf is defined as

  .. math::

     f_{cdf}(x, a) = \int_{-\infty}^x f_{pdf}(y, a)\mathrm{d}y

  where :math:`f_{pdf}` is the probability density function,
  :func:`jax.scipy.stats.gamma.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    a: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logcdf values.

  See Also:
    - :func:`jax.scipy.stats.gamma.cdf`
    - :func:`jax.scipy.stats.gamma.pdf`
    - :func:`jax.scipy.stats.gamma.sf`
    - :func:`jax.scipy.stats.gamma.logpdf`
    - :func:`jax.scipy.stats.gamma.logsf`
  """
  return lax.log(cdf(x, a, loc, scale))


def sf(x: ArrayLike, a: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Gamma survival function.

  JAX implementation of :obj:`scipy.stats.gamma` ``sf``.

  The survival function is defined as

  .. math::

     f_{sf}(x, k) = 1 - f_{cdf}(x, k)

  where :math:`f_{cdf}(x, k)` is the cumulative distribution function,
  :func:`jax.scipy.stats.gamma.cdf`.

  Args:
    x: arraylike, value at which to evaluate the SF
    a: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of sf values.

  See Also:
    - :func:`jax.scipy.stats.gamma.cdf`
    - :func:`jax.scipy.stats.gamma.pdf`
    - :func:`jax.scipy.stats.gamma.logcdf`
    - :func:`jax.scipy.stats.gamma.logpdf`
    - :func:`jax.scipy.stats.gamma.logsf`
  """
  x, a, loc, scale = promote_args_inexact("gamma.sf", x, a, loc, scale)
  return gammaincc(a, lax.div(lax.sub(x, loc), scale))


def logsf(x: ArrayLike, a: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Gamma log survival function.

  JAX implementation of :obj:`scipy.stats.gamma` ``logsf``.

  The survival function is defined as

  .. math::

     f_{sf}(x, k) = 1 - f_{cdf}(x, k)

  where :math:`f_{cdf}(x, k)` is the cumulative distribution function,
  :func:`jax.scipy.stats.gamma.cdf`.

  Args:
    x: arraylike, value at which to evaluate the SF
    a: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logsf values.

  See Also:
    - :func:`jax.scipy.stats.gamma.cdf`
    - :func:`jax.scipy.stats.gamma.pdf`
    - :func:`jax.scipy.stats.gamma.sf`
    - :func:`jax.scipy.stats.gamma.logcdf`
    - :func:`jax.scipy.stats.gamma.logpdf`
  """
  return lax.log(sf(x, a, loc, scale))
