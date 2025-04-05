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
from jax._src.numpy.util import promote_args_inexact
from jax._src.scipy.stats import norm
from jax._src.scipy.special import logsumexp, log_ndtr, ndtr


def _log_diff(x, y):
  return logsumexp(
    jnp.array([x, y]),
    b=jnp.array([jnp.ones_like(x), -jnp.ones_like(y)]),
    axis=0
  )


def _log_gauss_mass(a, b):
  """Log of Gaussian probability mass within an interval"""
  a, b = jnp.array(a), jnp.array(b)
  a, b = jnp.broadcast_arrays(a, b)

  # Note: Docstring carried over from scipy
  # Calculations in right tail are inaccurate, so we'll exploit the
  # symmetry and work only in the left tail
  case_left = b <= 0
  case_right = a > 0
  case_central = ~(case_left | case_right)

  def mass_case_left(a, b):
    return _log_diff(log_ndtr(b), log_ndtr(a))

  def mass_case_right(a, b):
    return mass_case_left(-b, -a)

  def mass_case_central(a, b):
    # Note: Docstring carried over from scipy
    # Previously, this was implemented as:
    # left_mass = mass_case_left(a, 0)
    # right_mass = mass_case_right(0, b)
    # return _log_sum(left_mass, right_mass)
    # Catastrophic cancellation occurs as np.exp(log_mass) approaches 1.
    # Correct for this with an alternative formulation.
    # We're not concerned with underflow here: if only one term
    # underflows, it was insignificant; if both terms underflow,
    # the result can't accurately be represented in logspace anyway
    # because sc.log1p(x) ~ x for small x.
    return jnp.log1p(-ndtr(a) - ndtr(-b))

  out = jnp.select(
    [case_left, case_right, case_central],
    [mass_case_left(a, b), mass_case_right(a, b), mass_case_central(a, b)]
  )
  return out


def logpdf(x, a, b, loc=0, scale=1):
  r"""Truncated normal log probability distribution function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``logpdf``.

  The truncated normal probability distribution is given by

  .. math::

     f(x, a, b) = \begin{cases}
       \frac{1}{\sqrt{2\pi}}e^{-x^2/2} & a \le x \le b \\
       0 & \mathrm{otherwise}
     \end{cases}

  where :math:`a` and :math:`b` are effectively specified in number of
  standard deviations from zero. JAX uses the scipy nomenclature
  of ``loc`` for the centroid and ``scale`` for the standard deviation.

  Args:
    x: arraylike, value at which to evaluate the PDF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logpdf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.cdf`
    - :func:`jax.scipy.stats.truncnorm.pdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logcdf`
    - :func:`jax.scipy.stats.truncnorm.logsf`
  """
  x, a, b, loc, scale = promote_args_inexact("truncnorm.logpdf", x, a, b, loc, scale)
  val = lax.sub(norm.logpdf(x, loc, scale), _log_gauss_mass(a, b))

  x_scaled = lax.div(lax.sub(x, loc), scale)
  val = jnp.where((x_scaled < a) | (x_scaled > b), -jnp.inf, val)
  val = jnp.where(a >= b, jnp.nan, val)
  return val


def pdf(x, a, b, loc=0, scale=1):
  r"""Truncated normal probability distribution function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``pdf``.

  The truncated normal probability distribution is given by

  .. math::

     f(x, a, b) = \begin{cases}
       \frac{1}{\sqrt{2\pi}}e^{-x^2/2} & a \le x \le b \\
       0 & \mathrm{otherwise}
     \end{cases}

  where :math:`a` and :math:`b` are effectively specified in number of
  standard deviations from the centroid. JAX uses the scipy nomenclature
  of ``loc`` for the centroid and ``scale`` for the standard deviation.

  Args:
    x: arraylike, value at which to evaluate the PDF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.cdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logcdf`
    - :func:`jax.scipy.stats.truncnorm.logpdf`
    - :func:`jax.scipy.stats.truncnorm.logsf`
  """
  return lax.exp(logpdf(x, a, b, loc, scale))


def logsf(x, a, b, loc=0, scale=1):
  """Truncated normal distribution log survival function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``logsf``

  The survival function is defined as

  .. math::

     f_{sf}(x) = 1 - f_{cdf}(x)

  where :math:`f_{cdf}(x)` is the cumulative distribution function,
  :func:`jax.scipy.stats.truncnorm.cdf`.

  Args:
    x: arraylike, value at which to evaluate the SF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logsf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.cdf`
    - :func:`jax.scipy.stats.truncnorm.pdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logcdf`
    - :func:`jax.scipy.stats.truncnorm.logpdf`
  """
  x, a, b, loc, scale = promote_args_inexact("truncnorm.logsf", x, a, b, loc, scale)
  return logcdf(-x, -b, -a, -loc, scale)


def sf(x, a, b, loc=0, scale=1):
  """Truncated normal distribution log survival function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``logsf``

  The survival function is defined as

  .. math::

     f_{sf}(x) = 1 - f_{cdf}(x)

  where :math:`f_{cdf}(x)` is the cumulative distribution function,
  :func:`jax.scipy.stats.truncnorm.cdf`.

  Args:
    x: arraylike, value at which to evaluate the SF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of sf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.cdf`
    - :func:`jax.scipy.stats.truncnorm.pdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logcdf`
    - :func:`jax.scipy.stats.truncnorm.logpdf`
  """
  return lax.exp(logsf(x, a, b, loc, scale))


def logcdf(x, a, b, loc=0, scale=1):
  r"""Truncated normal log cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``logcdf``.

  The cdf is defined as

  .. math::

     f_{cdf} = \int_{-\infty}^x f_{pdf}(y) \mathrm{d}y

  where here :math:`f_{pdf}` is the probability distribution function,
  :func:`jax.scipy.stats.truncnorm.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logcdf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.cdf`
    - :func:`jax.scipy.stats.truncnorm.pdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logpdf`
    - :func:`jax.scipy.stats.truncnorm.logsf`
  """
  x, a, b, loc, scale = promote_args_inexact("truncnorm.logcdf", x, a, b, loc, scale)
  x, a, b = jnp.broadcast_arrays(x, a, b)
  x = lax.div(lax.sub(x, loc), scale)
  logcdf = _log_gauss_mass(a, x) - _log_gauss_mass(a, b)
  logsf = _log_gauss_mass(x, b) - _log_gauss_mass(a, b)

  logcdf = jnp.select(
    # third condition: avoid catastrophic cancellation (from scipy)
    [x >= b, x <= a, logcdf > -0.1, x > a],
    [0, -jnp.inf, jnp.log1p(-jnp.exp(logsf)), logcdf]
  )
  logcdf = jnp.where(a >= b, jnp.nan, logcdf)
  return logcdf


def cdf(x, a, b, loc=0, scale=1):
  r"""Truncated normal cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``cdf``.

  The cdf is defined as

  .. math::

     f_{cdf} = \int_{-\infty}^x f_{pdf}(y) \mathrm{d}y

  where here :math:`f_{pdf}` is the probability distribution function,
  :func:`jax.scipy.stats.truncnorm.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of cdf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.pdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logcdf`
    - :func:`jax.scipy.stats.truncnorm.logpdf`
    - :func:`jax.scipy.stats.truncnorm.logsf`
  """
  return lax.exp(logcdf(x, a, b, loc, scale))
