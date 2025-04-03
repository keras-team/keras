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

from jax import lax
import jax.numpy as jnp
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import promote_args_inexact
from jax.scipy.special import xlog1py
from jax._src.typing import Array, ArrayLike


def logpmf(k: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Geometric log probability mass function.

  JAX implementation of :obj:`scipy.stats.geom` ``logpmf``.

  The Geometric probability mass function is given by

  .. math::

     f(k) = (1 - p)^{k-1}p

  for :math:`k\ge 1` and :math:`0 \le p \le 1`.

  Args:
    k: arraylike, value at which to evaluate the PMF
    p: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of logpmf values.

  See Also:
    :func:`jax.scipy.stats.geom.pmf`
  """
  k, p, loc = promote_args_inexact("geom.logpmf", k, p, loc)
  zero = _lax_const(k, 0)
  one = _lax_const(k, 1)
  x = lax.sub(k, loc)
  log_probs = xlog1py(lax.sub(x, one), -p) + lax.log(p)
  return jnp.where(lax.le(x, zero), -jnp.inf, log_probs)


def pmf(k: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Geometric probability mass function.

  JAX implementation of :obj:`scipy.stats.geom` ``pmf``.

  The Geometric probability mass function is given by

  .. math::

     f(k) = (1 - p)^{k-1}p

  for :math:`k\ge 1` and :math:`0 \le p \le 1`.

  Args:
    k: arraylike, value at which to evaluate the PMF
    p: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of pmf values.

  See Also:
    :func:`jax.scipy.stats.geom.logpmf`
  """
  return jnp.exp(logpmf(k, p, loc))
