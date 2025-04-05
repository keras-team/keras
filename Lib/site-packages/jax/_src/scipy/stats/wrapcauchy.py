# Copyright 2023 The JAX Authors.
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


def logpdf(x: ArrayLike, c: ArrayLike) -> Array:
  r"""Wrapped Cauchy log probability distribution function.

  JAX implementation of :obj:`scipy.stats.wrapcauchy` ``logpdf``.

  The wrapped Cauchy probability distribution function is given by

  .. math::

     f(x, c) = \frac{1-c^2}{2\pi(1+c^2-2c\cos x)}

  for :math:`0<c<1`, and where normalization is on the domain :math:`0\le x\le 2\pi`.

  Args:
    x: arraylike, value at which to evaluate the PDF
    c: arraylike, distribution shape parameter

  Returns:
    array of logpdf values.

  See Also:
    :func:`jax.scipy.stats.wrapcauchy.pdf`
  """
  x, c = promote_args_inexact('wrapcauchy.logpdf', x, c)
  return jnp.where(
    lax.gt(c, _lax_const(c, 0)) & lax.lt(c, _lax_const(c, 1)),
    jnp.where(
      lax.ge(x, _lax_const(x, 0)) & lax.le(x, _lax_const(x, jnp.pi * 2)),
      jnp.log(1 - c * c) - jnp.log(2 * jnp.pi) - jnp.log(1 + c * c - 2 * c * jnp.cos(x)),
      -jnp.inf,
    ),
    jnp.nan,
  )


def pdf(x: ArrayLike, c: ArrayLike) -> Array:
  r"""Wrapped Cauchy probability distribution function.

  JAX implementation of :obj:`scipy.stats.wrapcauchy` ``pdf``.

  The wrapped Cauchy probability distribution function is given by

  .. math::

     f(x, c) = \frac{1-c^2}{2\pi(1+c^2-2c\cos x)}

  for :math:`0<c<1`, and where normalization is on the domain :math:`0\le x\le 2\pi`.

  Args:
    x: arraylike, value at which to evaluate the PDF
    c: arraylike, distribution shape parameter

  Returns:
    array of pdf values.

  See Also:
    :func:`jax.scipy.stats.wrapcauchy.logpdf`
  """
  return lax.exp(logpdf(x, c))
