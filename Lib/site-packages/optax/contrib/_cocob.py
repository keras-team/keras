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
"""Backpropagating variant of the COntinuous COin Betting stochastic algorithm.

COCOB is a contributed optimizer implemented from Algorithm 2 of "Training Deep
Networks without Learning Rates Through Coin Betting" by Francesco Orabona and
Tatiana Tommasi.
"""
from collections.abc import Callable
from typing import Any, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import combine
from optax._src import transform
import optax.tree_utils as otu


class COCOBState(NamedTuple):
  """State for COntinuous COin Betting."""

  init_particles: base.Updates
  cumulative_gradients: base.Updates
  scale: base.Updates
  subgradients: base.Updates
  reward: base.Updates


def scale_by_cocob(
    alpha: float = 100, eps: float = 1e-8
) -> base.GradientTransformation:
  """Rescale updates according to the COntinuous COin Betting algorithm.

  See :func:`optax.contrib.cocob` for more details.

  Args:
    alpha: fraction to bet parameter of the COCOB optimizer
    eps: jitter term to avoid dividing by 0

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    init_adapt = otu.tree_zeros_like(params)
    init_scale = otu.tree_ones_like(params)
    init_scale = otu.tree_scalar_mul(eps, init_scale)
    return COCOBState(
        init_particles=params,
        cumulative_gradients=init_adapt,
        scale=init_scale,
        subgradients=init_adapt,
        reward=init_adapt,
    )

  def update_fn(updates, state, params):
    init_particles, cumulative_grads, scale, subgradients, reward = state

    scale = jax.tree.map(
        lambda L, c: jnp.maximum(L, jnp.abs(c)), scale, updates
    )
    subgradients = jax.tree.map(
        lambda G, c: G + jnp.abs(c), subgradients, updates
    )
    reward = jax.tree.map(
        lambda R, c, p, p0: jnp.maximum(R - c * (p - p0), 0),
        reward,
        updates,
        params,
        init_particles,
    )
    cumulative_grads = jax.tree.map(
        lambda C, c: C - c, cumulative_grads, updates
    )

    new_updates = jax.tree.map(
        lambda p, p0, C, L, G, R: (
            -p + (p0 + C / (L * jnp.maximum(G + L, alpha * L)) * (L + R))
        ),
        params,
        init_particles,
        cumulative_grads,
        scale,
        subgradients,
        reward,
    )

    new_state = COCOBState(
        init_particles=init_particles,
        cumulative_gradients=cumulative_grads,
        scale=scale,
        subgradients=subgradients,
        reward=reward,
    )
    return new_updates, new_state

  return base.GradientTransformation(init_fn, update_fn)


def cocob(
    learning_rate: base.ScalarOrSchedule = 1.0,
    alpha: float = 100,
    eps: float = 1e-8,
    weight_decay: float = 0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  """Rescale updates according to the COntinuous COin Betting algorithm.

  Algorithm for stochastic subgradient descent. Uses a gambling algorithm to
  find the minimizer of a non-smooth objective function by accessing its
  subgradients. All we need is a good gambling strategy. See Algorithm 2 of:

  Args:
    learning_rate: optional learning rate to e.g. inject some scheduler
    alpha: fraction to bet parameter of the COCOB optimizer
    eps: jitter term to avoid dividing by 0
    weight_decay: L2 penalty
    mask: mask for weight decay

  Returns:
    A `GradientTransformation` object.

  References:
    Orabana et al, `Training Deep Networks without Learning Rates Through Coin
    Betting <https://arxiv.org/pdf/1705.07795.pdf>`_, 2017
  """
  return combine.chain(
      transform.add_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate, flip_sign=False),
      scale_by_cocob(alpha, eps),
  )
