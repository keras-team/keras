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
"""Ranking losses.

A ranking loss is a differentiable function that expresses the cost of a ranking
induced by item scores compared to a ranking induced from relevance labels.

Ranking losses are designed to operate on the last dimension of its inputs. The
leading dimensions are considered batch dimensions.

Standalone usage:

>>> scores = jnp.array([2., 1., 3.])
>>> labels = jnp.array([1., 0., 0.])
>>> loss = optax.losses.ranking_softmax_loss(scores, labels)
>>> print(f"{loss:.3f}")
1.408

Usage with a batch of data and a mask to indicate valid items.

>>> scores = jnp.array([[2., 1., 0.], [1., 0.5, 1.5]])
>>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
>>> where = jnp.array([[True, True, False], [True, True, True]])
>>> loss = optax.losses.ranking_softmax_loss(scores, labels, where=where)
>>> print(f"{loss:.3f}")
0.497

To compute gradients of each loss function, please use standard JAX
transformations such as :func:`jax.grad` or :func:`jax.value_and_grad`:

>>> scores = jnp.asarray([2., 1., 3.])
>>> labels = jnp.asarray([1., 0., 0.])
>>> grads = jax.grad(optax.losses.ranking_softmax_loss)(scores, labels)
>>> print([float(f"{grad:.3f}") for grad in grads])
[-0.755, 0.09, 0.665]
"""

from collections.abc import Callable
from typing import Optional

import chex
import jax
import jax.numpy as jnp


def _safe_reduce(
    a: chex.Array,
    where: Optional[chex.Array] = None,
    reduce_fn: Optional[Callable[..., chex.Array]] = None,
) -> chex.Array:
  """Reduces the values of given array while preventing NaN in the output.

  For :func:`jax.numpy.mean` reduction, this additionally prevents ``NaN`` in
  the output if all elements are masked. This can happen with pairwise losses
  where there are no valid pairs because all labels are the same. In this
  situation, 0 is returned instead.

  When there is no ``reduce_fn``, this will set elements of ``a`` to 0 according
  to the ``where`` mask.

  Args:
    a: The :class:`jax.Array` to reduce.
    where: An optional :class:`jax.Array` indicating which elements to include
      in the reduction.
    reduce_fn: The function used to reduce. If None, no reduction is performed.

  Returns:
    The result of reducing the values of ``a`` using given ``reduce_fn``.
  """
  # Reduce values if there is a reduce_fn, otherwise keep the values as-is.
  output = reduce_fn(a, where=where) if reduce_fn is not None else a

  if reduce_fn is jnp.mean:
    # For mean reduction, we have to check whether the input contains any NaN
    # values, to ensure that masked mean reduction does not hide them (see
    # below).
    is_input_valid = jnp.logical_not(jnp.any(jnp.isnan(a)))

    # The standard jnp.mean implementation returns NaN if `where` is False
    # everywhere. This can happen in our case, e.g. pairwise losses with no
    # valid pairs. Instead, we prefer that the loss returns 0 in these cases.
    # Note that this only hides those NaN values if the input did not contain
    # any NaN values. Otherwise it just returns the output as-is.
    output = jnp.where(jnp.isnan(output) & is_input_valid, 0.0, output)

  if reduce_fn is None and where is not None:
    # When there is no reduce_fn (i.e. we are returning an unreduced
    # loss/metric), set the values of `a` to 0 for invalid (masked) items.
    # This makes sure that manual sum reduction on an unreduced loss works as
    # expected:
    # `jnp.sum(loss_fn(reduce_fn=None)) == loss_fn(reduce_fn=jnp.sum)`
    output = jnp.where(where, output, 0.0)

  return output


def ranking_softmax_loss(
    logits: chex.Array,
    labels: chex.Array,
    *,
    where: Optional[chex.Array] = None,
    weights: Optional[chex.Array] = None,
    reduce_fn: Optional[Callable[..., chex.Array]] = jnp.mean
) -> chex.Array:
  r"""Ranking softmax loss.

  Definition:

  .. math::
      \ell(s, y) = -\sum_i y_i \log \frac{\exp(s_i)}{\sum_j \exp(s_j)}

  Args:
    logits: A ``[..., list_size]``-:class:`~jax.Array`, indicating the score of
      each item.
    labels: A ``[..., list_size]``-:class:`~jax.Array`, indicating the relevance
      label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.Array`, indicating
      which items are valid for computing the loss. Items for which this is
      False will be ignored when computing the loss.
    weights: An optional ``[..., list_size]``-:class:`~jax.Array`, indicating
      the weight for each item.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The ranking softmax loss.
  """
  chex.assert_type([logits], float)
  labels = labels.astype(logits.dtype)

  # Applies mask so that masked elements do not count towards the loss.
  if where is not None:
    labels = jnp.where(where, labels, jnp.zeros_like(labels))
    logits = jnp.where(where, logits, -jnp.ones_like(logits) * jnp.inf)

  # Apply weights to labels.
  if weights is not None:
    labels *= weights

  # Scales labels and logits to match the cross entropy loss.
  logits_log_softmax = jax.nn.log_softmax(logits, axis=-1)

  # Computes per-element cross entropy.
  softmax_cross_entropy = labels * logits_log_softmax

  # Reduces softmax cross-entropy loss.
  loss = -jnp.sum(softmax_cross_entropy, axis=-1, where=where)

  # Setup mask to ignore lists with only invalid items in reduce_fn.
  if where is not None:
    where = jnp.any(where, axis=-1)

  return _safe_reduce(loss, where=where, reduce_fn=reduce_fn)
