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
from jax._src.numpy.util import promote_args_inexact
from jax._src.typing import Array, ArrayLike


def logpdf(x: ArrayLike, beta: ArrayLike) -> Array:
  r"""Generalized normal log probability distribution function.

  JAX implementation of :obj:`scipy.stats.gennorm` ``logpdf``.

  The generalized normal probability distribution function is defined as

  .. math::

     f(x, \beta) = \frac{\beta}{2\Gamma(1/\beta)}\exp(-|x|^\beta)

  where :math:`\Gamma` is the :func:`~jax.scipy.special.gamma` function, and
  :math:`\beta > 0`.

  Args:
    x: arraylike, value at which to evaluate the PDF
    beta: arraylike, distribution shape parameter

  Returns:
    array of logpdf values.

  See Also:
    - :func:`jax.scipy.stats.gennorm.cdf`
    - :func:`jax.scipy.stats.gennorm.pdf`
  """
  x, beta = promote_args_inexact("gennorm.logpdf", x, beta)
  return lax.log(.5 * beta) - lax.lgamma(1/beta) - lax.abs(x)**beta


def cdf(x: ArrayLike, beta: ArrayLike) -> Array:
  r"""Generalized normal cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.gennorm` ``cdf``.

  The cdf is defined as

  .. math::

     f_{cdf}(x, k) = \int_{-\infty}^x f_{pdf}(y, k)\mathrm{d}y

  where :math:`f_{pdf}` is the probability density function,
  :func:`jax.scipy.stats.gennorm.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    beta: arraylike, distribution shape parameter

  Returns:
    array of cdf values.

  See Also:
    - :func:`jax.scipy.stats.gennorm.pdf`
    - :func:`jax.scipy.stats.gennorm.logpdf`
  """
  x, beta = promote_args_inexact("gennorm.cdf", x, beta)
  return .5 * (1 + lax.sign(x) * lax.igamma(1/beta, lax.abs(x)**beta))


def pdf(x: ArrayLike, beta: ArrayLike) -> Array:
  r"""Generalized normal probability distribution function.

  JAX implementation of :obj:`scipy.stats.gennorm` ``pdf``.

  The generalized normal probability distribution function is defined as

  .. math::

     f(x, \beta) = \frac{\beta}{2\Gamma(1/\beta)}\exp(-|x|^\beta)

  where :math:`\Gamma` is the :func:`~jax.scipy.special.gamma` function, and
  :math:`\beta > 0`.

  Args:
    x: arraylike, value at which to evaluate the PDF
    beta: arraylike, distribution shape parameter

  Returns:
    array of pdf values.

  See Also:
    - :func:`jax.scipy.stats.gennorm.cdf`
    - :func:`jax.scipy.stats.gennorm.logpdf`
  """
  return lax.exp(logpdf(x, beta))
