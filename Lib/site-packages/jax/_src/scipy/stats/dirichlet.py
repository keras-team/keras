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
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import promote_dtypes_inexact
from jax.scipy.special import gammaln, xlogy
from jax._src.typing import Array, ArrayLike


def _is_simplex(x: Array) -> Array:
  x_sum = jnp.sum(x, axis=0)
  return jnp.all(x > 0, axis=0) & (abs(x_sum - 1) < 1E-6)


def logpdf(x: ArrayLike, alpha: ArrayLike) -> Array:
  r"""Dirichlet log probability distribution function.

  JAX implementation of :obj:`scipy.stats.dirichlet` ``logpdf``.

  The Dirichlet probability density function is

  .. math::

     f(\mathbf{x}) = \frac{1}{B(\mathbf{\alpha})} \prod_{i=1}^K x_i^{\alpha_i - 1}

  where :math:`B(\mathbf{\alpha})` is the :func:`~jax.scipy.special.beta` function
  in a :math:`K`-dimensional vector space.

  Args:
    x: arraylike, value at which to evaluate the PDF
    alpha: arraylike, distribution shape parameter

  Returns:
    array of logpdf values.

  See Also:
    :func:`jax.scipy.stats.dirichlet.pdf`
  """
  return _logpdf(*promote_dtypes_inexact(x, alpha))

def _logpdf(x: Array, alpha: Array) -> Array:
  if alpha.ndim != 1:
    raise ValueError(
      f"`alpha` must be one-dimensional; got alpha.shape={alpha.shape}"
    )
  if x.shape[0] not in (alpha.shape[0], alpha.shape[0] - 1):
    raise ValueError(
      "`x` must have either the same number of entries as `alpha` "
      f"or one entry fewer; got x.shape={x.shape}, alpha.shape={alpha.shape}"
    )
  one = _lax_const(x, 1)
  if x.shape[0] != alpha.shape[0]:
    x = jnp.concatenate([x, lax.sub(one, x.sum(0, keepdims=True))], axis=0)
  normalize_term = jnp.sum(gammaln(alpha)) - gammaln(jnp.sum(alpha))
  if x.ndim > 1:
    alpha = lax.broadcast_in_dim(alpha, alpha.shape + (1,) * (x.ndim - 1), (0,))
  log_probs = lax.sub(jnp.sum(xlogy(lax.sub(alpha, one), x), axis=0), normalize_term)
  return jnp.where(_is_simplex(x), log_probs, -jnp.inf)


def pdf(x: ArrayLike, alpha: ArrayLike) -> Array:
  r"""Dirichlet probability distribution function.

  JAX implementation of :obj:`scipy.stats.dirichlet` ``pdf``.

  The Dirichlet probability density function is

  .. math::

     f(\mathbf{x}) = \frac{1}{B(\mathbf{\alpha})} \prod_{i=1}^K x_i^{\alpha_i - 1}

  where :math:`B(\mathbf{\alpha})` is the :func:`~jax.scipy.special.beta` function
  in a :math:`K`-dimensional vector space.

  Args:
    x: arraylike, value at which to evaluate the PDF
    alpha: arraylike, distribution shape parameter

  Returns:
    array of pdf values.

  See Also:
    :func:`jax.scipy.stats.dirichlet.logpdf`
  """
  return lax.exp(logpdf(x, alpha))
