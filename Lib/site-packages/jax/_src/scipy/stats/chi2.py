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
from jax._src.typing import Array, ArrayLike
from jax.scipy.special import gammainc, gammaincc


def logpdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Chi-square log probability distribution function.

  JAX implementation of :obj:`scipy.stats.chi2` ``logpdf``.

  The chi-square probability distribution function is given by:

  .. math::

     f(x, k) = \begin{cases}
       \frac{x^{k/2-1}e^{-x/2}}{2^{k/2}\Gamma(k/2)} & x \ge 0 \\
       0 & \mathrm{otherwise}
     \end{cases}

  for :math:`k` degrees of freedom, and where :math:`\Gamma` is the
  :func:`~jax.scipy.special.gamma` function. JAX follows the scipy
  convention of using ``df`` to denote degrees of freedom.

  Args:
    x: arraylike, value at which to evaluate the PDF
    df: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logpdf values.

  See Also:
    - :func:`jax.scipy.stats.chi2.cdf`
    - :func:`jax.scipy.stats.chi2.pdf`
    - :func:`jax.scipy.stats.chi2.sf`
    - :func:`jax.scipy.stats.chi2.logcdf`
    - :func:`jax.scipy.stats.chi2.logsf`
  """
  x, df, loc, scale = promote_args_inexact("chi2.logpdf", x, df, loc, scale)
  one = _lax_const(x, 1)
  two = _lax_const(x, 2)
  y = lax.div(lax.sub(x, loc), scale)
  df_on_two = lax.div(df, two)

  kernel = lax.sub(lax.mul(lax.sub(df_on_two, one), lax.log(y)), lax.div(y,two))

  nrml_cnst = lax.neg(lax.add(lax.lgamma(df_on_two),lax.div(lax.mul(lax.log(two), df),two)))

  log_probs = lax.add(lax.sub(nrml_cnst, lax.log(scale)), kernel)
  return jnp.where(lax.lt(x, loc), -jnp.inf, log_probs)


def pdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Chi-square probability distribution function.

  JAX implementation of :obj:`scipy.stats.chi2` ``pdf``.

  The chi-square probability distribution function is given by:

  .. math::

     f(x, k) = \begin{cases}
       \frac{x^{k/2-1}e^{-x/2}}{2^{k/2}\Gamma(k/2)} & x \ge 0 \\
       0 & \mathrm{otherwise}
     \end{cases}

  for :math:`k` degrees of freedom, and where :math:`\Gamma` is the
  :func:`~jax.scipy.special.gamma` function. JAX follows the scipy
  convention of using ``df`` to denote degrees of freedom.

  Args:
    x: arraylike, value at which to evaluate the PDF
    df: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values.

  See Also:
    - :func:`jax.scipy.stats.chi2.cdf`
    - :func:`jax.scipy.stats.chi2.sf`
    - :func:`jax.scipy.stats.chi2.logcdf`
    - :func:`jax.scipy.stats.chi2.logpdf`
    - :func:`jax.scipy.stats.chi2.logsf`
  """
  return lax.exp(logpdf(x, df, loc, scale))


def cdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Chi-square cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.chi2` ``cdf``.

  The cdf is defined as

  .. math::

     f_{cdf}(x, k) = \int_{-\infty}^x f_{pdf}(y, k)\mathrm{d}y

  where :math:`f_{pdf}` is the probability density function,
  :func:`jax.scipy.stats.chi2.pdf`. JAX follows the scipy
  convention of using ``df`` to denote degrees of freedom.

  Args:
    x: arraylike, value at which to evaluate the CDF
    df: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of cdf values.

  See Also:
    - :func:`jax.scipy.stats.chi2.pdf`
    - :func:`jax.scipy.stats.chi2.sf`
    - :func:`jax.scipy.stats.chi2.logcdf`
    - :func:`jax.scipy.stats.chi2.logpdf`
    - :func:`jax.scipy.stats.chi2.logsf`
  """
  x, df, loc, scale = promote_args_inexact("chi2.cdf", x, df, loc, scale)
  two = _lax_const(scale, 2)
  return gammainc(
    lax.div(df, two),
    lax.clamp(
      _lax_const(x, 0),
      lax.div(
        lax.sub(x, loc),
        lax.mul(scale, two),
      ),
      _lax_const(x, jnp.inf),
    ),
  )


def logcdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Chi-square log cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.chi2` ``logcdf``.

  The cdf is defined as

  .. math::

     f_{cdf}(x, k) = \int_{-\infty}^x f_{pdf}(y, k)\mathrm{d}y

  where :math:`f_{pdf}` is the probability density function,
  :func:`jax.scipy.stats.chi2.pdf`. JAX follows the scipy
  convention of using ``df`` to denote degrees of freedom.

  Args:
    x: arraylike, value at which to evaluate the CDF
    df: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logcdf values

  See Also:
    - :func:`jax.scipy.stats.chi2.cdf`
    - :func:`jax.scipy.stats.chi2.pdf`
    - :func:`jax.scipy.stats.chi2.sf`
    - :func:`jax.scipy.stats.chi2.logpdf`
    - :func:`jax.scipy.stats.chi2.logsf`
  """
  return lax.log(cdf(x, df, loc, scale))


def sf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Chi-square survival function.

  JAX implementation of :obj:`scipy.stats.chi2` ``sf``.

  The survival function is defined as

  .. math::

     f_{sf}(x, k) = 1 - f_{cdf}(x, k)

  where :math:`f_{cdf}(x, k)` is the cumulative distribution function,
  :func:`jax.scipy.stats.chi2.cdf`. JAX follows the scipy
  convention of using ``df`` to denote degrees of freedom.

  Args:
    x: arraylike, value at which to evaluate the SF
    df: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of sf values.

  See Also:
    - :func:`jax.scipy.stats.chi2.cdf`
    - :func:`jax.scipy.stats.chi2.pdf`
    - :func:`jax.scipy.stats.chi2.logcdf`
    - :func:`jax.scipy.stats.chi2.logpdf`
    - :func:`jax.scipy.stats.chi2.logsf`
  """
  x, df, loc, scale = promote_args_inexact("chi2.sf", x, df, loc, scale)
  two = _lax_const(scale, 2)
  return gammaincc(
    lax.div(df, two),
    lax.clamp(
      _lax_const(x, 0),
      lax.div(
        lax.sub(x, loc),
        lax.mul(scale, two),
      ),
      _lax_const(x, jnp.inf),
    ),
  )


def logsf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Chi-square log survival function.

  JAX implementation of :obj:`scipy.stats.chi2` ``logsf``.

  The survival function is defined as

  .. math::

     f_{sf}(x, k) = 1 - f_{cdf}(x, k)

  where :math:`f_{cdf}(x, k)` is the cumulative distribution function,
  :func:`jax.scipy.stats.chi2.cdf`. JAX follows the scipy
  convention of using ``df`` to denote degrees of freedom.

  Args:
    x: arraylike, value at which to evaluate the SF
    df: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logsf values.

  See Also:
    - :func:`jax.scipy.stats.chi2.cdf`
    - :func:`jax.scipy.stats.chi2.pdf`
    - :func:`jax.scipy.stats.chi2.sf`
    - :func:`jax.scipy.stats.chi2.logcdf`
    - :func:`jax.scipy.stats.chi2.logpdf`
  """
  return lax.log(sf(x, df, loc, scale))
