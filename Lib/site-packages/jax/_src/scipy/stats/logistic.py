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

from jax.scipy.special import expit, logit

from jax import lax
import jax.numpy as jnp
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import promote_args_inexact
from jax._src.typing import Array, ArrayLike


def logpdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Logistic log probability distribution function.

  JAX implementation of :obj:`scipy.stats.logistic` ``logpdf``.

  The logistic probability distribution function is given by

  .. math::

     f(x) = \frac{e^{-x}}{(1 + e^{-x})^2}

  Args:
    x: arraylike, value at which to evaluate the PDF
    a: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logpdf values.

  See Also:
    - :func:`jax.scipy.stats.logistic.cdf`
    - :func:`jax.scipy.stats.logistic.pdf`
    - :func:`jax.scipy.stats.logistic.sf`
    - :func:`jax.scipy.stats.logistic.isf`
    - :func:`jax.scipy.stats.logistic.ppf`
  """
  x, loc, scale = promote_args_inexact("logistic.logpdf", x, loc, scale)
  x = lax.div(lax.sub(x, loc), scale)
  two = _lax_const(x, 2)
  half_x = lax.div(x, two)
  return lax.sub(lax.mul(lax.neg(two), jnp.logaddexp(half_x, lax.neg(half_x))), lax.log(scale))


def pdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Logistic probability distribution function.

  JAX implementation of :obj:`scipy.stats.logistic` ``pdf``.

  The logistic probability distribution function is given by

  .. math::

     f(x) = \frac{e^{-x}}{(1 + e^{-x})^2}

  Args:
    x: arraylike, value at which to evaluate the PDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values.

  See Also:
    - :func:`jax.scipy.stats.logistic.cdf`
    - :func:`jax.scipy.stats.logistic.sf`
    - :func:`jax.scipy.stats.logistic.isf`
    - :func:`jax.scipy.stats.logistic.logpdf`
    - :func:`jax.scipy.stats.logistic.ppf`
  """
  return lax.exp(logpdf(x, loc, scale))


def ppf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  """Logistic distribution percent point function.

  JAX implementation of :obj:`scipy.stats.logistic` ``ppf``.

  The percent point function is defined as the inverse of the
  cumulative distribution function, :func:`jax.scipy.stats.logistic.cdf`.

  Args:
    x: arraylike, value at which to evaluate the PPF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of ppf values.

  See Also:
    - :func:`jax.scipy.stats.logistic.cdf`
    - :func:`jax.scipy.stats.logistic.pdf`
    - :func:`jax.scipy.stats.logistic.sf`
    - :func:`jax.scipy.stats.logistic.isf`
    - :func:`jax.scipy.stats.logistic.logpdf`
  """
  x, loc, scale = promote_args_inexact("logistic.ppf", x, loc, scale)
  return lax.add(lax.mul(logit(x), scale), loc)


def sf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  """Logistic distribution survival function.

  JAX implementation of :obj:`scipy.stats.logistic` ``sf``

  The survival function is defined as

  .. math::

     f_{sf}(x, k) = 1 - f_{cdf}(x, k)

  where :math:`f_{cdf}(x, k)` is the cumulative distribution function,
  :func:`jax.scipy.stats.logistic.cdf`.

  Args:
    x: arraylike, value at which to evaluate the SF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of sf values.

  See Also:
    - :func:`jax.scipy.stats.logistic.cdf`
    - :func:`jax.scipy.stats.logistic.pdf`
    - :func:`jax.scipy.stats.logistic.isf`
    - :func:`jax.scipy.stats.logistic.logpdf`
    - :func:`jax.scipy.stats.logistic.ppf`
  """
  x, loc, scale = promote_args_inexact("logistic.sf", x, loc, scale)
  return expit(lax.neg(lax.div(lax.sub(x, loc), scale)))


def isf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  """Logistic distribution inverse survival function.

  JAX implementation of :obj:`scipy.stats.logistic` ``isf``.

  Returns the inverse of the survival function,
  :func:`jax.scipy.stats.logistic.sf`.

  Args:
    x: arraylike, value at which to evaluate the ISF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of isf values.

  See Also:
    - :func:`jax.scipy.stats.logistic.cdf`
    - :func:`jax.scipy.stats.logistic.pdf`
    - :func:`jax.scipy.stats.logistic.sf`
    - :func:`jax.scipy.stats.logistic.logpdf`
    - :func:`jax.scipy.stats.logistic.ppf`
  """
  x, loc, scale = promote_args_inexact("logistic.isf", x, loc, scale)
  return lax.add(lax.mul(lax.neg(logit(x)), scale), loc)


def cdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Logistic cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.logistic` ``cdf``.

  The cdf is defined as

  .. math::

     f_{cdf}(x, k) = \int_{-\infty}^x f_{pdf}(y, k)\mathrm{d}y

  where :math:`f_{pdf}` is the probability density function,
  :func:`jax.scipy.stats.logistic.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of cdf values.

  See Also:
    - :func:`jax.scipy.stats.logistic.pdf`
    - :func:`jax.scipy.stats.logistic.sf`
    - :func:`jax.scipy.stats.logistic.isf`
    - :func:`jax.scipy.stats.logistic.logpdf`
    - :func:`jax.scipy.stats.logistic.ppf`
  """
  x, loc, scale = promote_args_inexact("logistic.cdf", x, loc, scale)
  return expit(lax.div(lax.sub(x, loc), scale))
