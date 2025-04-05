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
from jax._src.numpy.util import promote_args_inexact
from jax._src.typing import Array, ArrayLike


def logpdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Exponential log probability distribution function.

  JAX implementation of :obj:`scipy.stats.expon` ``logpdf``.

  The Exponential probability distribution function is

  .. math::

     f(x) = \begin{cases}
       e^{-x} & x \ge 0 \\
       0 & \mathrm{otherwise}
     \end{cases}

  Args:
    x: arraylike, value at which to evaluate the PDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logpdf values.

  See Also:
    :func:`jax.scipy.stats.expon.pdf`
  """
  x, loc, scale = promote_args_inexact("expon.logpdf", x, loc, scale)
  log_scale = lax.log(scale)
  linear_term = lax.div(lax.sub(x, loc), scale)
  log_probs = lax.neg(lax.add(linear_term, log_scale))
  return jnp.where(lax.lt(x, loc), -jnp.inf, log_probs)


def pdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Exponential probability distribution function.

  JAX implementation of :obj:`scipy.stats.expon` ``pdf``.

  The Exponential probability distribution function is

  .. math::

     f(x) = \begin{cases}
       e^{-x} & x \ge 0 \\
       0 & \mathrm{otherwise}
     \end{cases}

  Args:
    x: arraylike, value at which to evaluate the PDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values.

  See Also:
    :func:`jax.scipy.stats.expon.logpdf`
  """
  return lax.exp(logpdf(x, loc, scale))
