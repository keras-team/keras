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
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import promote_args_inexact
from jax._src.typing import Array, ArrayLike


def logpdf(x: ArrayLike, kappa: ArrayLike) -> Array:
  r"""von Mises log probability distribution function.

  JAX implementation of :obj:`scipy.stats.vonmises` ``logpdf``.

  The von Mises probability distribution function is given by

  .. math::

     f(x, \kappa) = \frac{1}{2\pi I_0(\kappa)}e^{\kappa\cos x}

  Where :math:`I_0` is the modified Bessel function :func:`~jax.scipy.special.i0`
  and :math:`\kappa\ge 0`, and the distribution is normalized in the interval
  :math:`-\pi \le x \le \pi`.

  Args:
    x: arraylike, value at which to evaluate the PDF
    kappa: arraylike, distribution shape parameter

  Returns:
    array of logpdf values.

  See Also:
    :func:`jax.scipy.stats.vonmises.pdf`
  """
  x, kappa = promote_args_inexact('vonmises.logpdf', x, kappa)
  zero = _lax_const(kappa, 0)
  return jnp.where(lax.gt(kappa, zero), kappa * (jnp.cos(x) - 1) - jnp.log(2 * jnp.pi * lax.bessel_i0e(kappa)), jnp.nan)


def pdf(x: ArrayLike, kappa: ArrayLike) -> Array:
  r"""von Mises probability distribution function.

  JAX implementation of :obj:`scipy.stats.vonmises` ``pdf``.

  The von Mises probability distribution function is given by

  .. math::

     f(x, \kappa) = \frac{1}{2\pi I_0(\kappa)}e^{\kappa\cos x}

  Where :math:`I_0` is the modified Bessel function :func:`~jax.scipy.special.i0`
  and :math:`\kappa\ge 0`, and the distribution is normalized in the interval
  :math:`-\pi \le x \le \pi`.

  Args:
    x: arraylike, value at which to evaluate the PDF
    kappa: arraylike, distribution shape parameter

  Returns:
    array of pdf values.

  See Also:
    :func:`jax.scipy.stats.vonmises.logpdf`
  """
  return lax.exp(logpdf(x, kappa))
