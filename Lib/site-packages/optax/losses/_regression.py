# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Regression losses."""

import functools
from typing import Optional, Union

import chex
import jax.numpy as jnp


def squared_error(
    predictions: chex.Array,
    targets: Optional[chex.Array] = None,
) -> chex.Array:
  """Calculates the squared error for a set of predictions.

  Mean Squared Error can be computed as squared_error(a, b).mean().

  Args:
    predictions: a vector of arbitrary shape `[...]`.
    targets: a vector with shape broadcastable to that of `predictions`; if not
      provided then it is assumed to be a vector of zeros.

  Returns:
    elementwise squared differences, with same shape as `predictions`.

  .. note::
    l2_loss = 0.5 * squared_error, where the 0.5 term is standard in
    "Pattern Recognition and Machine Learning" by Bishop, but not
    "The Elements of Statistical Learning" by Tibshirani.
  """
  chex.assert_type([predictions], float)
  if targets is not None:
    # Avoid broadcasting logic for "-" operator.
    chex.assert_equal_shape((predictions, targets))
  errors = predictions - targets if targets is not None else predictions
  return errors**2


def l2_loss(
    predictions: chex.Array,
    targets: Optional[chex.Array] = None,
) -> chex.Array:
  """Calculates the L2 loss for a set of predictions.

  Args:
    predictions: a vector of arbitrary shape `[...]`.
    targets: a vector with shape broadcastable to that of `predictions`; if not
      provided then it is assumed to be a vector of zeros.

  Returns:
    elementwise squared differences, with same shape as `predictions`.

  .. note::
    the 0.5 term is standard in "Pattern Recognition and Machine Learning"
    by Bishop, but not "The Elements of Statistical Learning" by Tibshirani.
  """
  return 0.5 * squared_error(predictions, targets)


@functools.partial(chex.warn_only_n_pos_args_in_future, n=2)
def huber_loss(
    predictions: chex.Array,
    targets: Optional[chex.Array] = None,
    delta: float = 1.0,
) -> chex.Array:
  """Huber loss, similar to L2 loss close to zero, L1 loss away from zero.

  If gradient descent is applied to the `huber loss`, it is equivalent to
  clipping gradients of an `l2_loss` to `[-delta, delta]` in the backward pass.

  Args:
    predictions: a vector of arbitrary shape `[...]`.
    targets: a vector with shape broadcastable to that of `predictions`; if not
      provided then it is assumed to be a vector of zeros.
    delta: the bounds for the huber loss transformation, defaults at 1.

  Returns:
    elementwise huber losses, with the same shape of `predictions`.

  References:
    `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`_, Wikipedia.
  """
  chex.assert_type([predictions], float)
  errors = (predictions - targets) if (targets is not None) else predictions
  # 0.5 * err^2                  if |err| <= d
  # 0.5 * d^2 + d * (|err| - d)  if |err| > d
  abs_errors = jnp.abs(errors)
  quadratic = jnp.minimum(abs_errors, delta)
  # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
  linear = abs_errors - quadratic
  return 0.5 * quadratic**2 + delta * linear


def log_cosh(
    predictions: chex.Array,
    targets: Optional[chex.Array] = None,
) -> chex.Array:
  """Calculates the log-cosh loss for a set of predictions.

  log(cosh(x)) is approximately `(x**2) / 2` for small x and `abs(x) - log(2)`
  for large x.  It is a twice differentiable alternative to the Huber loss.

  Args:
    predictions: a vector of arbitrary shape `[...]`.
    targets: a vector with shape broadcastable to that of `predictions`; if not
      provided then it is assumed to be a vector of zeros.

  Returns:
    the log-cosh loss, with same shape as `predictions`.

  References:
    Chen et al, `Log Hyperbolic Cosine Loss Improves Variational Auto-Encoder
    <https://openreview.net/pdf?id=rkglvsC9Ym>`, 2019
  """
  chex.assert_type([predictions], float)
  errors = (predictions - targets) if (targets is not None) else predictions
  # log(cosh(x)) = log((exp(x) + exp(-x))/2) = log(exp(x) + exp(-x)) - log(2)
  return jnp.logaddexp(errors, -errors) - jnp.log(2.0).astype(errors.dtype)


@functools.partial(chex.warn_only_n_pos_args_in_future, n=2)
def cosine_similarity(
    predictions: chex.Array,
    targets: chex.Array,
    epsilon: float = 0.0,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[chex.Array, None] = None,
) -> chex.Array:
  r"""Computes the cosine similarity between targets and predictions.

  The cosine **similarity** is a measure of similarity between vectors defined
  as the cosine of the angle between them, which is also the inner product of
  those vectors normalized to have unit norm.

  Args:
    predictions: The predicted vectors, with shape `[..., dim]`.
    targets: Ground truth target vectors, with shape `[..., dim]`.
    epsilon: minimum norm for terms in the denominator of the cosine similarity.
    axis: Axis or axes along which to compute.
    where: Elements to include in the computation.

  Returns:
    cosine similarity measures, with shape `[...]`.

  References:
    `Cosine similarity <https://en.wikipedia.org/wiki/Cosine_similarity>`_,
    Wikipedia.

  .. versionchanged:: 0.2.4
    Added ``axis`` and ``where`` arguments.
  """
  chex.assert_type([predictions, targets], float)
  a = predictions
  b = targets

  # dot = (a * b).sum(axis=axis, where=where)
  # a_norm2 = jnp.square(a).sum(axis=axis, where=where)
  # b_norm2 = jnp.square(b).sum(axis=axis, where=where)
  # return dot / jnp.sqrt((a_norm2 * b_norm2))

  a_norm2 = jnp.square(a).sum(axis=axis, where=where, keepdims=True)
  b_norm2 = jnp.square(b).sum(axis=axis, where=where, keepdims=True)
  a_norm = jnp.sqrt(a_norm2.clip(epsilon))
  b_norm = jnp.sqrt(b_norm2.clip(epsilon))
  a_unit = a / a_norm
  b_unit = b / b_norm
  return (a_unit * b_unit).sum(axis=axis, where=where)


@functools.partial(chex.warn_only_n_pos_args_in_future, n=2)
def cosine_distance(
    predictions: chex.Array,
    targets: chex.Array,
    epsilon: float = 0.0,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[chex.Array, None] = None,
) -> chex.Array:
  r"""Computes the cosine distance between targets and predictions.

  The cosine **distance**, implemented here, measures the **dissimilarity**
  of two vectors as the opposite of cosine **similarity**: `1 - cos(\theta)`.

  Args:
    predictions: The predicted vectors, with shape `[..., dim]`.
    targets: Ground truth target vectors, with shape `[..., dim]`.
    epsilon: minimum norm for terms in the denominator of the cosine similarity.
    axis: Axis or axes along which to compute.
    where: Elements to include in the computation.

  Returns:
    cosine distances, with shape `[...]`.

  References:
    `Cosine distance
    <https://en.wikipedia.org/wiki/Cosine_similarity#Cosine_distance>`_,
    Wikipedia.

  .. versionchanged:: 0.2.4
    Added ``axis`` and ``where`` arguments.
  """
  chex.assert_type([predictions, targets], float)
  # cosine distance = 1 - cosine similarity.
  return 1.0 - cosine_similarity(
      predictions, targets, epsilon=epsilon, axis=axis, where=where
  )
