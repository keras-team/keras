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
"""Additive components in gradient transformations."""

from collections.abc import Callable
from typing import Any, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics
from optax._src import wrappers


def add_decayed_weights(
    weight_decay: Union[float, jax.Array] = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  """Add parameter scaled by `weight_decay`.

  Args:
    weight_decay: A scalar weight decay rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree, or a
      Callable that returns such a pytree given the params/updates. The leaves
      should be booleans, `True` for leaves/subtrees you want to apply the
      transformation to, and `False` for those you want to skip.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    updates = jax.tree.map(
        lambda g, p: None if g is None else g + weight_decay * p,
        updates,
        params,
        is_leaf=lambda x: x is None,
    )
    return updates, state

  # If mask is not `None`, apply mask to the gradient transformation.
  # E.g. it is common to skip weight decay on bias units and batch stats.
  if mask is not None:
    return wrappers.masked(
        base.GradientTransformation(base.init_empty_state, update_fn), mask
    )
  return base.GradientTransformation(base.init_empty_state, update_fn)


class AddNoiseState(NamedTuple):
  """State for adding gradient noise. Contains a count for annealing."""

  count: chex.Array
  rng_key: chex.PRNGKey


def add_noise(
    eta: float, gamma: float, seed: int
) -> base.GradientTransformation:
  """Add gradient noise.

  Args:
    eta: Base variance of the gaussian noise added to the gradient.
    gamma: Decay exponent for annealing of the variance.
    seed: Seed for random number generation.

  Returns:
    A :class:`optax.GradientTransformation` object.

  References:
    Neelakantan et al, `Adding Gradient Noise Improves Learning for Very Deep
    Networks <https://arxiv.org/abs/1511.06807>`_, 2015
  """

  def init_fn(params):
    del params
    return AddNoiseState(
        count=jnp.zeros([], jnp.int32), rng_key=jax.random.PRNGKey(seed)
    )

  def update_fn(updates, state, params=None):  # pylint: disable=missing-docstring
    del params
    count_inc = numerics.safe_increment(state.count)
    standard_deviation = jnp.sqrt(eta / count_inc**gamma)

    rng_key, sample_key = jax.random.split(state.rng_key)
    noise = otu.tree_random_like(
        sample_key, target_tree=updates, sampler=jax.random.normal
    )
    updates = otu.tree_add_scalar_mul(
        tree_x=updates, scalar=standard_deviation, tree_y=noise
    )
    return updates, AddNoiseState(count=count_inc, rng_key=rng_key)

  return base.GradientTransformation(init_fn, update_fn)
