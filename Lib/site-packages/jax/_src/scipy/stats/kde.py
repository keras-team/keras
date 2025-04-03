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

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

import jax.numpy as jnp
from jax import jit, lax, random, vmap
from jax._src.numpy.util import check_arraylike, promote_dtypes_inexact
from jax._src.tree_util import register_pytree_node_class
from jax.scipy import linalg, special


@register_pytree_node_class
@dataclass(frozen=True, init=False)
class gaussian_kde:
  """Gaussian Kernel Density Estimator

  JAX implementation of :class:`scipy.stats.gaussian_kde`.

  Parameters:
    dataset: arraylike, real-valued. Data from which to estimate the distribution.
      If 1D, shape is (n_data,). If 2D, shape is (n_dimensions, n_data).
    bw_method: string, scalar, or callable. Either "scott", "silverman", a scalar
      value, or a callable function which takes ``self`` as a parameter.
    weights: arraylike, optional. Weights of the same shape as the dataset.
  """
  neff: Any
  dataset: Any
  weights: Any
  covariance: Any
  inv_cov: Any

  def __init__(self, dataset, bw_method=None, weights=None):
    check_arraylike("gaussian_kde", dataset)
    dataset = jnp.atleast_2d(dataset)
    if jnp.issubdtype(lax.dtype(dataset), jnp.complexfloating):
      raise NotImplementedError("gaussian_kde does not support complex data")
    if not dataset.size > 1:
      raise ValueError("`dataset` input should have multiple elements.")

    d, n = dataset.shape
    if weights is not None:
      check_arraylike("gaussian_kde", weights)
      dataset, weights = promote_dtypes_inexact(dataset, weights)
      weights = jnp.atleast_1d(weights)
      weights /= jnp.sum(weights)
      if weights.ndim != 1:
        raise ValueError("`weights` input should be one-dimensional.")
      if len(weights) != n:
        raise ValueError("`weights` input should be of length n")
    else:
      dataset, = promote_dtypes_inexact(dataset)
      weights = jnp.full(n, 1.0 / n, dtype=dataset.dtype)

    self._setattr("dataset", dataset)
    self._setattr("weights", weights)
    neff = self._setattr("neff", 1 / jnp.sum(weights**2))

    bw_method = "scott" if bw_method is None else bw_method
    if bw_method == "scott":
      factor = jnp.power(neff, -1. / (d + 4))
    elif bw_method == "silverman":
      factor = jnp.power(neff * (d + 2) / 4.0, -1. / (d + 4))
    elif jnp.isscalar(bw_method) and not isinstance(bw_method, str):
      factor = bw_method
    elif callable(bw_method):
      factor = bw_method(self)
    else:
      raise ValueError(
          "`bw_method` should be 'scott', 'silverman', a scalar, or a callable."
      )

    data_covariance = jnp.atleast_2d(
        jnp.cov(dataset, rowvar=1, bias=False, aweights=weights))
    data_inv_cov = jnp.linalg.inv(data_covariance)
    covariance = data_covariance * factor**2
    inv_cov = data_inv_cov / factor**2
    self._setattr("covariance", covariance)
    self._setattr("inv_cov", inv_cov)

  def _setattr(self, name, value):
    # Frozen dataclasses don't support setting attributes so we have to
    # overload that operation here as they do in the dataclass implementation
    object.__setattr__(self, name, value)
    return value

  def tree_flatten(self):
    return ((self.neff, self.dataset, self.weights, self.covariance,
             self.inv_cov), None)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    kde = cls.__new__(cls)
    kde._setattr("neff", children[0])
    kde._setattr("dataset", children[1])
    kde._setattr("weights", children[2])
    kde._setattr("covariance", children[3])
    kde._setattr("inv_cov", children[4])
    return kde

  @property
  def d(self):
    return self.dataset.shape[0]

  @property
  def n(self):
    return self.dataset.shape[1]

  def evaluate(self, points):
    """Evaluate the Gaussian KDE on the given points."""
    check_arraylike("evaluate", points)
    points = self._reshape_points(points)
    result = _gaussian_kernel_eval(False, self.dataset.T, self.weights[:, None],
                                   points.T, self.inv_cov)
    return result[:, 0]

  def __call__(self, points):
    return self.evaluate(points)

  def integrate_gaussian(self, mean, cov):
    """Integrate the distribution weighted by a Gaussian."""
    mean = jnp.atleast_1d(jnp.squeeze(mean))
    cov = jnp.atleast_2d(cov)

    if mean.shape != (self.d,):
      raise ValueError(f"mean does not have dimension {self.d}")
    if cov.shape != (self.d, self.d):
      raise ValueError(f"covariance does not have dimension {self.d}")

    chol = linalg.cho_factor(self.covariance + cov)
    norm = jnp.sqrt(2 * np.pi)**self.d * jnp.prod(jnp.diag(chol[0]))
    norm = 1.0 / norm
    return _gaussian_kernel_convolve(chol, norm, self.dataset, self.weights,
                                     mean)

  def integrate_box_1d(self, low, high):
    """Integrate the distribution over the given limits."""
    if self.d != 1:
      raise ValueError("integrate_box_1d() only handles 1D pdfs")
    if jnp.ndim(low) != 0 or jnp.ndim(high) != 0:
      raise ValueError(
          "the limits of integration in integrate_box_1d must be scalars")
    sigma = jnp.squeeze(jnp.sqrt(self.covariance))
    low = jnp.squeeze((low - self.dataset) / sigma)
    high = jnp.squeeze((high - self.dataset) / sigma)
    return jnp.sum(self.weights * (special.ndtr(high) - special.ndtr(low)))

  def integrate_kde(self, other):
    """Integrate the product of two Gaussian KDE distributions."""
    if other.d != self.d:
      raise ValueError("KDEs are not the same dimensionality")

    chol = linalg.cho_factor(self.covariance + other.covariance)
    norm = jnp.sqrt(2 * np.pi)**self.d * jnp.prod(jnp.diag(chol[0]))
    norm = 1.0 / norm

    sm, lg = (self, other) if self.n < other.n else (other, self)
    result = vmap(partial(_gaussian_kernel_convolve, chol, norm, lg.dataset,
                          lg.weights),
                  in_axes=1)(sm.dataset)
    return jnp.sum(result * sm.weights)

  def resample(self, key, shape=()):
    r"""Randomly sample a dataset from the estimated pdf

    Args:
      key: a PRNG key used as the random key.
      shape: optional, a tuple of nonnegative integers specifying the result
        batch shape; that is, the prefix of the result shape excluding the last
        axis.

    Returns:
      The resampled dataset as an array with shape `(d,) + shape`.
    """
    ind_key, eps_key = random.split(key)
    ind = random.choice(ind_key, self.n, shape=shape, p=self.weights)
    eps = random.multivariate_normal(eps_key,
                                     jnp.zeros(self.d, self.covariance.dtype),
                                     self.covariance,
                                     shape=shape,
                                     dtype=self.dataset.dtype).T
    return self.dataset[:, ind] + eps

  def pdf(self, x):
    """Probability density function"""
    return self.evaluate(x)

  def logpdf(self, x):
    """Log probability density function"""
    check_arraylike("logpdf", x)
    x = self._reshape_points(x)
    result = _gaussian_kernel_eval(True, self.dataset.T, self.weights[:, None],
                                   x.T, self.inv_cov)
    return result[:, 0]

  def integrate_box(self, low_bounds, high_bounds, maxpts=None):
    """This method is not implemented in the JAX interface."""
    del low_bounds, high_bounds, maxpts
    raise NotImplementedError(
        "only 1D box integrations are supported; use `integrate_box_1d`")

  def set_bandwidth(self, bw_method=None):
    """This method is not implemented in the JAX interface."""
    del bw_method
    raise NotImplementedError(
        "dynamically changing the bandwidth method is not supported")

  def _reshape_points(self, points):
    if jnp.issubdtype(lax.dtype(points), jnp.complexfloating):
      raise NotImplementedError(
          "gaussian_kde does not support complex coordinates")
    points = jnp.atleast_2d(points)
    d, m = points.shape
    if d != self.d:
      if d == 1 and m == self.d:
        points = jnp.reshape(points, (self.d, 1))
      else:
        raise ValueError(
            "points have dimension {}, dataset has dimension {}".format(
                d, self.d))
    return points


def _gaussian_kernel_convolve(chol, norm, target, weights, mean):
  diff = target - mean[:, None]
  alpha = linalg.cho_solve(chol, diff)
  arg = 0.5 * jnp.sum(diff * alpha, axis=0)
  return norm * jnp.sum(jnp.exp(-arg) * weights)


@partial(jit, static_argnums=0)
def _gaussian_kernel_eval(in_log, points, values, xi, precision):
  points, values, xi, precision = promote_dtypes_inexact(
      points, values, xi, precision)
  d = points.shape[1]

  if xi.shape[1] != d:
    raise ValueError("points and xi must have same trailing dim")
  if precision.shape != (d, d):
    raise ValueError("precision matrix must match data dims")

  whitening = linalg.cholesky(precision, lower=True)
  points = jnp.dot(points, whitening)
  xi = jnp.dot(xi, whitening)
  log_norm = jnp.sum(jnp.log(
      jnp.diag(whitening))) - 0.5 * d * jnp.log(2 * np.pi)

  def kernel(x_test, x_train, y_train):
    arg = log_norm - 0.5 * jnp.sum(jnp.square(x_train - x_test))
    if in_log:
      return jnp.log(y_train) + arg
    else:
      return y_train * jnp.exp(arg)

  reduce = special.logsumexp if in_log else jnp.sum
  reduced_kernel = lambda x: reduce(vmap(kernel, in_axes=(None, 0, 0))
                                    (x, points, values),
                                    axis=0)
  mapped_kernel = vmap(reduced_kernel)

  return mapped_kernel(xi)
