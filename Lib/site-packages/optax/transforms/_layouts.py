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
"""Wrappers changing the layouts of the tensors that transforms operate on."""

import jax
import jax.numpy as jnp
import numpy as np
from optax._src import base


def flatten(
    inner: base.GradientTransformation,
) -> base.GradientTransformationExtraArgs:
  """Flattens parameters and gradients for init and update of inner transform.

  This can reduce the overhead of performing many calculations on lots of small
  variables, at the cost of slightly increased memory usage.

  Args:
    inner: Inner transformation to flatten inputs for.

  Returns:
    New :class:`optax.GradientTransformationExtraArgs`
  """

  inner = base.with_extra_args_support(inner)

  def _flatten(params):
    """Flattens and concatenates all tensors in params to a single vector."""
    params, _ = jax.tree.flatten(params)
    return jnp.concatenate([jnp.reshape(param, [-1]) for param in params])

  def _unflatten(updates, flat):
    """Extracts tensors from flat, using the structure and shapes of params."""
    updates_flat, treedef = jax.tree.flatten(updates)
    offsets = []
    for update in updates_flat:
      size = np.size(update)
      if offsets:
        offsets.append(size + offsets[-1])
      else:
        offsets.append(size)
    del offsets[-1]
    flat_split = jnp.split(flat, offsets)
    reshaped = [
        jnp.reshape(flat_update, update.shape)
        for flat_update, update in zip(flat_split, updates_flat)
    ]
    return jax.tree.unflatten(treedef, reshaped)

  def init_fn(params):
    flat = _flatten(params)
    return inner.init(flat)

  def update_fn(updates, state, params=None, **extra_args):
    if params is not None:
      params = _flatten(params)
    updates_flat, state = inner.update(
        _flatten(updates), state, params, **extra_args
    )
    updates = _unflatten(updates, updates_flat)
    return updates, state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)
