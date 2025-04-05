# Copyright 2024 The Flax Authors.
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

"""Utils for Orbax Checkpointing, available even after Flax Checkpointing is deprecated."""

import warnings
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp
from jax.sharding import Mesh

PyTree = Any


def is_multi_device_array(value: Any) -> bool:
  """Instruct Orbax to save this array with Tensorstore instead of msgpack."""
  if isinstance(value, jax.Array):
    return not value.is_fully_replicated
  return False


def save_args_from_target(target: Any) -> Any:
  return jax.tree_util.tree_map(lambda _: ocp.SaveArgs(), target)


def maybe_construct_transformations(
  target: Any, transforms: Any | None
) -> Any:
  if transforms is not None:
    return transforms
  flat_transforms = {}
  flat_target = ocp.utils.to_flat_dict(target, sep='/', keep_empty_nodes=True)
  for k, v in flat_target.items():
    if v is None:
      flat_transforms[k] = ocp.Transform(use_fallback=True)
  return flat_transforms


def restore_args_from_target(target: Any, mesh: Mesh | None = None) -> Any:
  """Creates Orbax `restore_args` given a target Pytree.

  Args:
    target: The Pytree that has the same structure as the checkpoint. The arrays
      restored from checkpoint will have the same `sharding` as the target
      Pytree's corresponding arrays.
    mesh: DEPRECATED ARG. Please simply use your mesh to create the arrays
      in your `target`, no need to pass it here.

  Returns:
    A Pytree of Orbax `RestoreArgs` or `ArrayRestoreArgs`
  """

  def find_sharding(x):
    if hasattr(x, 'sharding'):
      return x.sharding
    return None

  # Simpler case: no JAX arrays
  if not any(
      jax.tree_util.tree_flatten(jax.tree_util.tree_map(find_sharding, target))[
          0
      ]
  ):
    return jax.tree_util.tree_map(
        lambda x: ocp.RestoreArgs(restore_type=np.ndarray), target
    )

  # JAX arrays: find sharding from the given target and create RestoreArgs
  sharding_tree = jax.tree_util.tree_map(find_sharding, target)
  if mesh is not None:
    warnings.warn(
        (
            'restore_args_from_target(): `mesh` arg is deprecated. Simply'
            ' calling the function with target pytree should suffice.'
        ),
        DeprecationWarning,
    )
    def substitute_embedding(s):
      return jax.sharding.NamedSharding(mesh, s.spec)
    sharding_tree = jax.tree_util.tree_map(substitute_embedding, sharding_tree)
  restore_args = ocp.checkpoint_utils.construct_restore_args(
      target, sharding_tree, set_global_shape=False
  )
  return restore_args
