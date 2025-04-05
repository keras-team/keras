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
"""Apply transformed gradient updates to parameters."""

import chex
import jax
import jax.numpy as jnp
from optax._src import base


def apply_updates(params: base.Params, updates: base.Updates) -> base.Params:
  """Applies an update to the corresponding parameters.

  This is a utility functions that applies an update to a set of parameters, and
  then returns the updated parameters to the caller. As an example, the update
  may be a gradient transformed by a sequence of`GradientTransformations`. This
  function is exposed for convenience, but it just adds updates and parameters;
  you may also apply updates to parameters manually, using `jax.tree.map`
  (e.g. if you want to manipulate updates in custom ways before applying them).

  Args:
    params: a tree of parameters.
    updates: a tree of updates, the tree structure and the shape of the leaf
      nodes must match that of `params`.

  Returns:
    Updated parameters, with same structure, shape and type as `params`.
  """
  return jax.tree.map(
      lambda p, u: (
          None if p is None else jnp.asarray(p + u).astype(jnp.asarray(p).dtype)
      ),
      params,
      updates,
      is_leaf=lambda x: x is None,
  )


def incremental_update(
    new_tensors: base.Params, old_tensors: base.Params, step_size: chex.Numeric
) -> base.Params:
  """Incrementally update parameters via polyak averaging.

  Polyak averaging tracks an (exponential moving) average of the past
  parameters of a model, for use at test/evaluation time.

  Args:
    new_tensors: the latest value of the tensors.
    old_tensors: a moving average of the values of the tensors.
    step_size: the step_size used to update the polyak average on each step.

  Returns:
    an updated moving average `step_size*new+(1-step_size)*old` of the params.

  References:
    [Polyak et al, 1991](https://epubs.siam.org/doi/10.1137/0330046)
  """
  return jax.tree.map(
      lambda new, old: (
          None if new is None else step_size * new + (1.0 - step_size) * old
      ),
      new_tensors,
      old_tensors,
      is_leaf=lambda x: x is None,
  )


def periodic_update(
    new_tensors: base.Params,
    old_tensors: base.Params,
    steps: chex.Array,
    update_period: int,
) -> base.Params:
  """Periodically update all parameters with new values.

  A slow copy of a model's parameters, updated every K actual updates, can be
  used to implement forms of self-supervision (in supervised learning), or to
  stabilize temporal difference learning updates (in reinforcement learning).

  Args:
    new_tensors: the latest value of the tensors.
    old_tensors: a slow copy of the model's parameters.
    steps: number of update steps on the "online" network.
    update_period: every how many steps to update the "target" network.

  Returns:
    a slow copy of the model's parameters, updated every `update_period` steps.

  References:
    [Grill et al., 2020](https://arxiv.org/abs/2006.07733)
    [Mnih et al., 2015](https://arxiv.org/abs/1312.5602)
  """
  return jax.lax.cond(
      jnp.mod(steps, update_period) == 0,
      lambda _: new_tensors,
      lambda _: old_tensors,
      None,
  )
