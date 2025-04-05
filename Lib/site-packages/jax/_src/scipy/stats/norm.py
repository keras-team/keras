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

import numpy as np

from jax import lax
import jax.numpy as jnp
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import promote_args_inexact
from jax._src.typing import Array, ArrayLike
from jax.scipy import special


def logpdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Normal log probability distribution function.

  JAX implementation of :obj:`scipy.stats.norm` ``logpdf``.

  The normal distribution pdf is given by

  .. math::

     f(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}

  Args:
    x: arraylike, value at which to evaluate the PDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logpdf values.

  See Also:
    - :func:`jax.scipy.stats.norm.cdf`
    - :func:`jax.scipy.stats.norm.pdf`
    - :func:`jax.scipy.stats.norm.sf`
    - :func:`jax.scipy.stats.norm.logcdf`
    - :func:`jax.scipy.stats.norm.logsf`
    - :func:`jax.scipy.stats.norm.isf`
    - :func:`jax.scipy.stats.norm.ppf`
  """
  x, loc, scale = promote_args_inexact("norm.logpdf", x, loc, scale)
  scale_sqrd = lax.square(scale)
  log_normalizer = lax.log(lax.mul(_lax_const(x, 2 * np.pi), scale_sqrd))
  quadratic = lax.div(lax.square(lax.sub(x, loc)), scale_sqrd)
  return lax.div(lax.add(log_normalizer, quadratic), _lax_const(x, -2))


def pdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Normal probability distribution function.

  JAX implementation of :obj:`scipy.stats.norm` ``pdf``.

  The normal distribution pdf is given by

  .. math::

     f(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}

  Args:
    x: arraylike, value at which to evaluate the PDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values.

  See Also:
    - :func:`jax.scipy.stats.norm.cdf`
    - :func:`jax.scipy.stats.norm.sf`
    - :func:`jax.scipy.stats.norm.logcdf`
    - :func:`jax.scipy.stats.norm.logpdf`
    - :func:`jax.scipy.stats.norm.logsf`
    - :func:`jax.scipy.stats.norm.isf`
    - :func:`jax.scipy.stats.norm.ppf`
  """
  return lax.exp(logpdf(x, loc, scale))


def cdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Normal cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.norm` ``cdf``.

  The cdf is defined as

  .. math::

     f_{cdf}(x, k) = \int_{-\infty}^x f_{pdf}(y, k)\mathrm{d}y

  where :math:`f_{pdf}` is the probability density function,
  :func:`jax.scipy.stats.norm.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of cdf values.

  See Also:
    - :func:`jax.scipy.stats.norm.pdf`
    - :func:`jax.scipy.stats.norm.sf`
    - :func:`jax.scipy.stats.norm.logcdf`
    - :func:`jax.scipy.stats.norm.logpdf`
    - :func:`jax.scipy.stats.norm.logsf`
    - :func:`jax.scipy.stats.norm.isf`
    - :func:`jax.scipy.stats.norm.ppf`
  """
  x, loc, scale = promote_args_inexact("norm.cdf", x, loc, scale)
  return special.ndtr(lax.div(lax.sub(x, loc), scale))


def logcdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Normal log cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.norm` ``logcdf``.

  The cdf is defined as

  .. math::

     f_{cdf}(x, k) = \int_{-\infty}^x f_{pdf}(y, k)\mathrm{d}y

  where :math:`f_{pdf}` is the probability density function,
  :func:`jax.scipy.stats.norm.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logcdf values.

  See Also:
    - :func:`jax.scipy.stats.norm.cdf`
    - :func:`jax.scipy.stats.norm.pdf`
    - :func:`jax.scipy.stats.norm.sf`
    - :func:`jax.scipy.stats.norm.logpdf`
    - :func:`jax.scipy.stats.norm.logsf`
    - :func:`jax.scipy.stats.norm.isf`
    - :func:`jax.scipy.stats.norm.ppf`
  """
  x, loc, scale = promote_args_inexact("norm.logcdf", x, loc, scale)
  return special.log_ndtr(lax.div(lax.sub(x, loc), scale))


def ppf(q: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  """Normal distribution percent point function.

  JAX implementation of :obj:`scipy.stats.norm` ``ppf``.

  The percent point function is defined as the inverse of the
  cumulative distribution function, :func:`jax.scipy.stats.norm.cdf`.

  Args:
    q: arraylike, value at which to evaluate the PPF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of ppf values.

  See Also:
    - :func:`jax.scipy.stats.norm.cdf`
    - :func:`jax.scipy.stats.norm.pdf`
    - :func:`jax.scipy.stats.norm.sf`
    - :func:`jax.scipy.stats.norm.logcdf`
    - :func:`jax.scipy.stats.norm.logpdf`
    - :func:`jax.scipy.stats.norm.logsf`
    - :func:`jax.scipy.stats.norm.isf`
  """
  return jnp.asarray(special.ndtri(q) * scale + loc, float)


def logsf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  """Normal distribution log survival function.

  JAX implementation of :obj:`scipy.stats.norm` ``logsf``.

  The survival function is defined as

  .. math::

     f_{sf}(x) = 1 - f_{cdf}(x)

  where :math:`f_{cdf}(x)` is the cumulative distribution function,
  :func:`jax.scipy.stats.norm.cdf`.

  Args:
    x: arraylike, value at which to evaluate the SF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logsf values.

  See Also:
    - :func:`jax.scipy.stats.norm.cdf`
    - :func:`jax.scipy.stats.norm.pdf`
    - :func:`jax.scipy.stats.norm.sf`
    - :func:`jax.scipy.stats.norm.logcdf`
    - :func:`jax.scipy.stats.norm.logpdf`
    - :func:`jax.scipy.stats.norm.isf`
    - :func:`jax.scipy.stats.norm.ppf`
  """
  x, loc, scale = promote_args_inexact("norm.logsf", x, loc, scale)
  return logcdf(-x, -loc, scale)


def sf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  """Normal distribution survival function.

  JAX implementation of :obj:`scipy.stats.norm` ``sf``.

  The survival function is defined as

  .. math::

     f_{sf}(x) = 1 - f_{cdf}(x)

  where :math:`f_{cdf}(x)` is the cumulative distribution function,
  :func:`jax.scipy.stats.norm.cdf`.

  Args:
    x: arraylike, value at which to evaluate the SF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of sf values.

  See Also:
    - :func:`jax.scipy.stats.norm.cdf`
    - :func:`jax.scipy.stats.norm.pdf`
    - :func:`jax.scipy.stats.norm.logcdf`
    - :func:`jax.scipy.stats.norm.logpdf`
    - :func:`jax.scipy.stats.norm.logsf`
    - :func:`jax.scipy.stats.norm.isf`
    - :func:`jax.scipy.stats.norm.ppf`
  """
  x, loc, scale = promote_args_inexact("norm.sf", x, loc, scale)
  return cdf(-x, -loc, scale)


def isf(q: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  """Normal distribution inverse survival function.

  JAX implementation of :obj:`scipy.stats.norm` ``isf``.

  Returns the inverse of the survival function,
  :func:`jax.scipy.stats.norm.sf`.

  Args:
    x: arraylike, value at which to evaluate the ISF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of isf values.

  See Also:
    - :func:`jax.scipy.stats.norm.cdf`
    - :func:`jax.scipy.stats.norm.pdf`
    - :func:`jax.scipy.stats.norm.sf`
    - :func:`jax.scipy.stats.norm.logcdf`
    - :func:`jax.scipy.stats.norm.logpdf`
    - :func:`jax.scipy.stats.norm.logsf`
    - :func:`jax.scipy.stats.norm.ppf`
  """
  return ppf(lax.sub(_lax_const(q, 1), q), loc, scale)
