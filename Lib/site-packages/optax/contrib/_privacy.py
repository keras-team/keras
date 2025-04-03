# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Differential Privacy utilities."""

from typing import Any, NamedTuple, Optional

import jax
from optax._src import base
from optax._src import clipping
from optax._src import combine
from optax._src import transform


class DifferentiallyPrivateAggregateState(NamedTuple):
  """State containing PRNGKey for `differentially_private_aggregate`."""

  # TODO(optax-dev): rng_key used to be annotated as `jnp.array` but that is
  # not a valid annotation (it's a function and secretely resolved to `Any`).
  # We should add back typing.
  rng_key: Any


def differentially_private_aggregate(
    l2_norm_clip: float, noise_multiplier: float, seed: int
) -> base.GradientTransformation:
  """Aggregates gradients based on the DPSGD algorithm.

  Args:
    l2_norm_clip: maximum L2 norm of the per-example gradients.
    noise_multiplier: ratio of standard deviation to the clipping norm.
    seed: initial seed used for the jax.random.PRNGKey

  Returns:
    A :class:`optax.GradientTransformation`.

  References:
    Abadi et al, 2016 `Deep Learning with Differential Privacy
    <https://arxiv.org/abs/1607.00133>`_, 2016

  .. warning::
    Unlike other transforms, `differentially_private_aggregate` expects
    the input updates to have a batch dimension in the 0th axis. That is, this
    function expects per-example gradients as input (which are easy to obtain in
    JAX using `jax.vmap`). It can still be composed with other transformations
    as long as it is the first in the chain.
  """
  noise_std = l2_norm_clip * noise_multiplier

  def init_fn(params):
    del params
    return DifferentiallyPrivateAggregateState(rng_key=jax.random.PRNGKey(seed))

  def update_fn(updates, state, params=None):
    del params
    grads_flat, grads_treedef = jax.tree.flatten(updates)
    bsize = grads_flat[0].shape[0]
    clipped, _ = clipping.per_example_global_norm_clip(grads_flat, l2_norm_clip)

    new_key, *rngs = jax.random.split(state.rng_key, len(grads_flat) + 1)
    noised = [
        (g + noise_std * jax.random.normal(r, g.shape, g.dtype)) / bsize
        for g, r in zip(clipped, rngs)
    ]
    return (
        jax.tree.unflatten(grads_treedef, noised),
        DifferentiallyPrivateAggregateState(rng_key=new_key),
    )

  return base.GradientTransformation(init_fn, update_fn)


def dpsgd(
    learning_rate: base.ScalarOrSchedule,
    l2_norm_clip: float,
    noise_multiplier: float,
    seed: int,
    momentum: Optional[float] = None,
    nesterov: bool = False,
) -> base.GradientTransformation:
  """The DPSGD optimizer.

  Differential privacy is a standard for privacy guarantees of algorithms
  learning from aggregate databases including potentially sensitive information.
  DPSGD offers protection against a strong adversary with full knowledge of the
  training mechanism and access to the model's parameters.

  Args:
    learning_rate: A fixed global scaling factor.
    l2_norm_clip: Maximum L2 norm of the per-example gradients.
    noise_multiplier: Ratio of standard deviation to the clipping norm.
    seed: Initial seed used for the jax.random.PRNGKey
    momentum: Decay rate used by the momentum term, when it is set to `None`,
      then momentum is not used at all.
    nesterov: Whether Nesterov momentum is used.

  Returns:
    A :class:`optax.GradientTransformation`.

  References:
    Abadi et al, 2016 `Deep Learning with Differential Privacy
    <https://arxiv.org/abs/1607.00133>`_, 2016

  .. warning::
    This :class:`optax.GradientTransformation` expects input updates to have a
    batch dimension on the 0th axis. That is, this function expects per-example
    gradients as input (which are easy to obtain in JAX using `jax.vmap`).
  """
  return combine.chain(
      differentially_private_aggregate(
          l2_norm_clip=l2_norm_clip,
          noise_multiplier=noise_multiplier,
          seed=seed,
      ),
      (
          transform.trace(decay=momentum, nesterov=nesterov)
          if momentum is not None
          else base.identity()
      ),
      transform.scale_by_learning_rate(learning_rate),
  )
