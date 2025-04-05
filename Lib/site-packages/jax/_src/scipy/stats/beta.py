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
from jax.scipy.special import betaln, betainc, xlogy, xlog1py


def logpdf(x: ArrayLike, a: ArrayLike, b: ArrayLike,
           loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Beta log probability distribution function.

  JAX implementation of :obj:`scipy.stats.beta` ``logpdf``.

  The pdf of the beta function is:

  .. math::

    f(x, a, b) = \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} x^{a-1}(1-x)^{b-1}

  where :math:`\Gamma` is the :func:`~jax.scipy.special.gamma` function,
  It is defined for :math:`0\le x\le 1` and :math:`b>0`.

  Args:
    x: arraylike, value at which to evaluate the PDF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logpdf values

  See Also:
    - :func:`jax.scipy.stats.beta.cdf`
    - :func:`jax.scipy.stats.beta.pdf`
    - :func:`jax.scipy.stats.beta.sf`
    - :func:`jax.scipy.stats.beta.logcdf`
    - :func:`jax.scipy.stats.beta.logsf`
  """
  x, a, b, loc, scale = promote_args_inexact("beta.logpdf", x, a, b, loc, scale)
  one = _lax_const(x, 1)
  zero = _lax_const(a, 0)
  shape_term = lax.neg(betaln(a, b))
  y = lax.div(lax.sub(x, loc), scale)
  log_linear_term = lax.add(xlogy(lax.sub(a, one), y),
                            xlog1py(lax.sub(b, one), lax.neg(y)))
  log_probs = lax.sub(lax.add(shape_term, log_linear_term), lax.log(scale))
  result = jnp.where(jnp.logical_or(lax.gt(x, lax.add(loc, scale)),
                                    lax.lt(x, loc)), -jnp.inf, log_probs)
  result_positive_constants = jnp.where(jnp.logical_or(jnp.logical_or(lax.le(a, zero), lax.le(b, zero)),
                                                       lax.le(scale, zero)), jnp.nan, result)
  return result_positive_constants


def pdf(x: ArrayLike, a: ArrayLike, b: ArrayLike,
        loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Beta probability distribution function.

  JAX implementation of :obj:`scipy.stats.beta` ``pdf``.

  The pdf of the beta function is:

  .. math::

    f(x, a, b) = \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} x^{a-1}(1-x)^{b-1}

  where :math:`\Gamma` is the :func:`~jax.scipy.special.gamma` function.
  It is defined for :math:`0\le x\le 1` and :math:`b>0`.

  Args:
    x: arraylike, value at which to evaluate the PDF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values

  See Also:
    - :func:`jax.scipy.stats.beta.cdf`
    - :func:`jax.scipy.stats.beta.sf`
    - :func:`jax.scipy.stats.beta.logcdf`
    - :func:`jax.scipy.stats.beta.logpdf`
    - :func:`jax.scipy.stats.beta.logsf`
  """
  return lax.exp(logpdf(x, a, b, loc, scale))


def cdf(x: ArrayLike, a: ArrayLike, b: ArrayLike,
        loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Beta cumulative distribution function

  JAX implementation of :obj:`scipy.stats.beta` ``cdf``.

  The cdf is defined as

  .. math::

     f_{cdf}(x, a, b) = \int_{-\infty}^x f_{pdf}(y, a, b)\mathrm{d}y

  where :math:`f_{pdf}` is the beta distribution probability density function,
  :func:`jax.scipy.stats.beta.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of cdf values

  See Also:
    - :func:`jax.scipy.stats.beta.pdf`
    - :func:`jax.scipy.stats.beta.sf`
    - :func:`jax.scipy.stats.beta.logcdf`
    - :func:`jax.scipy.stats.beta.logpdf`
    - :func:`jax.scipy.stats.beta.logsf`
  """
  x, a, b, loc, scale = promote_args_inexact("beta.cdf", x, a, b, loc, scale)
  return betainc(
    a,
    b,
    lax.clamp(
      _lax_const(x, 0),
      lax.div(lax.sub(x, loc), scale),
      _lax_const(x, 1),
    )
  )


def logcdf(x: ArrayLike, a: ArrayLike, b: ArrayLike,
           loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Beta log cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.beta` ``logcdf``.

  The cdf is defined as

  .. math::

     f_{cdf}(x, a, b) = \int_{-\infty}^x f_{pdf}(y, a, b)\mathrm{d}y

  where :math:`f_{pdf}` is the beta distribution probability density function,
  :func:`jax.scipy.stats.beta.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logcdf values

  See Also:
    - :func:`jax.scipy.stats.beta.cdf`
    - :func:`jax.scipy.stats.beta.pdf`
    - :func:`jax.scipy.stats.beta.sf`
    - :func:`jax.scipy.stats.beta.logpdf`
    - :func:`jax.scipy.stats.beta.logsf`
  """
  return lax.log(cdf(x, a, b, loc, scale))


def sf(x: ArrayLike, a: ArrayLike, b: ArrayLike,
       loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Beta distribution survival function.

  JAX implementation of :obj:`scipy.stats.beta` ``sf``.

  The survival function is defined as

  .. math::

     f_{sf}(x, a, b) = 1 - f_{cdf}(x, a, b)

  where :math:`f_{cdf}(x, a, b)` is the beta cumulative distribution function,
  :func:`jax.scipy.stats.beta.cdf`.

  Args:
    x: arraylike, value at which to evaluate the SF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of sf values.

  See Also:
    - :func:`jax.scipy.stats.beta.cdf`
    - :func:`jax.scipy.stats.beta.pdf`
    - :func:`jax.scipy.stats.beta.logcdf`
    - :func:`jax.scipy.stats.beta.logpdf`
    - :func:`jax.scipy.stats.beta.logsf`
  """
  x, a, b, loc, scale = promote_args_inexact("beta.sf", x, a, b, loc, scale)
  return betainc(
    b,
    a,
    1 - lax.clamp(
      _lax_const(x, 0),
      lax.div(lax.sub(x, loc), scale),
      _lax_const(x, 1),
    )
  )


def logsf(x: ArrayLike, a: ArrayLike, b: ArrayLike,
          loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Beta distribution log survival function.

  JAX implementation of :obj:`scipy.stats.beta` ``logsf``.

  The survival function is defined as

  .. math::

     f_{sf}(x, a, b) = 1 - f_{cdf}(x, a, b)

  where :math:`f_{cdf}(x, a, b)` is the beta cumulative distribution function,
  :func:`jax.scipy.stats.beta.cdf`.

  Args:
    x: arraylike, value at which to evaluate the SF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logsf values.

  See Also:
    - :func:`jax.scipy.stats.beta.cdf`
    - :func:`jax.scipy.stats.beta.pdf`
    - :func:`jax.scipy.stats.beta.sf`
    - :func:`jax.scipy.stats.beta.logcdf`
    - :func:`jax.scipy.stats.beta.logpdf`
  """
  return lax.log(sf(x, a, b, loc, scale))
