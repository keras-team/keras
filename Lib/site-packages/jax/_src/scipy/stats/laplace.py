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
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import promote_args_inexact
from jax._src.typing import Array, ArrayLike


def logpdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Laplace log probability distribution function.

  JAX implementation of :obj:`scipy.stats.laplace` ``logpdf``.

  The Laplace probability distribution function is given by

  .. math::

     f(x) = \frac{1}{2} e^{-|x|}

  Args:
    x: arraylike, value at which to evaluate the PDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logpdf values.

  See Also:
    - :func:`jax.scipy.stats.laplace.cdf`
    - :func:`jax.scipy.stats.laplace.pdf`
  """
  x, loc, scale = promote_args_inexact("laplace.logpdf", x, loc, scale)
  two = _lax_const(x, 2)
  linear_term = lax.div(lax.abs(lax.sub(x, loc)), scale)
  return lax.neg(lax.add(linear_term, lax.log(lax.mul(two, scale))))


def pdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Laplace probability distribution function.

  JAX implementation of :obj:`scipy.stats.laplace` ``pdf``.

  The Laplace probability distribution function is given by

  .. math::

     f(x) = \frac{1}{2} e^{-|x|}

  Args:
    x: arraylike, value at which to evaluate the PDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values.

  See Also:
    - :func:`jax.scipy.stats.laplace.cdf`
    - :func:`jax.scipy.stats.laplace.logpdf`
  """
  return lax.exp(logpdf(x, loc, scale))


def cdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Laplace cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.laplace` ``cdf``.

  The cdf is defined as

  .. math::

     f_{cdf}(x, k) = \int_{-\infty}^x f_{pdf}(y, k)\mathrm{d}y

  where :math:`f_{pdf}` is the probability density function,
  :func:`jax.scipy.stats.laplace.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of cdf values.

  See Also:
    - :func:`jax.scipy.stats.laplace.pdf`
    - :func:`jax.scipy.stats.laplace.logpdf`
  """
  x, loc, scale = promote_args_inexact("laplace.cdf", x, loc, scale)
  half = _lax_const(x, 0.5)
  one = _lax_const(x, 1)
  zero = _lax_const(x, 0)
  diff = lax.div(lax.sub(x, loc), scale)
  return lax.select(lax.le(diff, zero),
                    lax.mul(half, lax.exp(diff)),
                    lax.sub(one, lax.mul(half, lax.exp(lax.neg(diff)))))
