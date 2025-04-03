# Copyright 2024 The Orbax Authors.
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

"""Utility functions for Orbax.

NOTE: Functions in this file are deprecated in favor of corresponding functions
in sub-modules. Please use those functions instead, and do not add new
functions here.

TODO(b/266449081) Increase unit test coverage.
"""

from typing import Any

from etils import epath
import jax
import numpy as np
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import async_utils
from orbax.checkpoint._src.path import step as step_lib
from orbax.checkpoint._src.tree import utils as tree_utils


TMP_DIR_SUFFIX = step_lib.TMP_DIR_SUFFIX
# TODO(b/260759189): Deprecate this prefix when no longer in use by JAX MG.
_AGGREGATED_PREFIX = 'AGGREGATED://'
# Used in a msgpack checkpoint file to denote a leaf value that has been written
# individually. Typically, this may indicate an array that was written using
# Tensorstore rather than its value being directly stored in the msgpack file.
# To avoid duplication, we replace the value with a placeholder prefix and other
# relevant information (see functions below).
_PLACEHOLDER_PREFIX = 'PLACEHOLDER://'
PyTree = Any

sync_global_processes = multihost.sync_global_processes
sync_global_devices = multihost.sync_global_processes
broadcast_one_to_all = multihost.broadcast_one_to_all
reached_preemption = multihost.reached_preemption
is_primary_host = multihost.is_primary_host


async_makedirs = async_utils.async_makedirs
async_write_bytes = async_utils.async_write_bytes
async_exists = async_utils.async_exists

is_gcs_path = step_lib.is_gcs_path
checkpoint_steps = step_lib.checkpoint_steps
any_checkpoint_step = step_lib.any_checkpoint_step
is_checkpoint_finalized = step_lib.is_checkpoint_finalized
is_tmp_checkpoint = step_lib.is_tmp_checkpoint
tmp_checkpoints = step_lib.tmp_checkpoints
cleanup_tmp_directories = step_lib.cleanup_tmp_directories
get_save_directory = step_lib.get_save_directory
record_saved_duration = step_lib.record_saved_duration
step_from_checkpoint_name = step_lib.step_from_checkpoint_name
checkpoint_steps_paths = step_lib.checkpoint_steps_paths

deserialize_tree = tree_utils.deserialize_tree
from_flat_dict = tree_utils.from_flat_dict
# TODO: b/365169723 - Remove public access to this function.
from_flattened_with_keypath = tree_utils.from_flattened_with_keypath
serialize_tree = tree_utils.serialize_tree
to_flat_dict = tree_utils.to_flat_dict
is_sequence_key = tree_utils.is_sequence_key
is_dict_key = tree_utils.is_dict_key
tuple_path_from_keypath = tree_utils.tuple_path_from_keypath
get_key_name = tree_utils.get_key_name
is_empty_node = tree_utils.is_empty_node
is_empty_or_leaf = tree_utils.is_empty_or_leaf
to_shape_dtype_struct = tree_utils.to_shape_dtype_struct


def leaf_is_placeholder(leaf: Any) -> bool:
  """Determines if `leaf` represents a placeholder for a non-aggregated value."""
  return isinstance(leaf, str) and (
      leaf.startswith(_PLACEHOLDER_PREFIX)
      or leaf.startswith(_AGGREGATED_PREFIX)
  )


def leaf_placeholder(name: str) -> str:
  """Constructs value to act as placeholder for non-aggregated value."""
  return _PLACEHOLDER_PREFIX + name


def name_from_leaf_placeholder(placeholder: str) -> str:
  """Gets the param name from a placeholder with the correct prefix."""
  if not leaf_is_placeholder(placeholder):
    msg = (
        'Requested name from placeholder, but value did not contain required'
        ' prefix.'
    )
    raise ValueError(msg)
  if placeholder.startswith(_AGGREGATED_PREFIX):
    return placeholder.replace(_AGGREGATED_PREFIX, '', 1)
  elif placeholder.startswith(_PLACEHOLDER_PREFIX):
    return placeholder.replace(_PLACEHOLDER_PREFIX, '', 1)
  else:
    raise ValueError('Found placeholder beginning with unexpected prefix.')


def all_leaves_are_placeholders(tree: PyTree) -> bool:
  """Determines if all leaves in `tree` are placeholders."""
  return all(leaf_is_placeholder(leaf) for leaf in jax.tree.leaves(tree))


def pytree_structure(directory: epath.PathLike) -> PyTree:
  """Reconstruct state dict from saved model format in `directory`."""
  directory = epath.Path(directory)
  jax.monitoring.record_event('/jax/orbax/deprecation/inferred_structure')

  def add_nested_key(subtree, nested_key, key_name):
    if not nested_key:
      return subtree

    current = nested_key[0]

    if len(nested_key) == 1:
      assert current not in subtree
      subtree[current] = leaf_placeholder(key_name)
      return subtree

    subkeys = nested_key[1:]
    if current not in subtree:
      subtree[current] = {}
    subtree[current] = add_nested_key(subtree[current], subkeys, key_name)
    return subtree

  keys = directory.iterdir()
  tree = {}
  for k in keys:
    # Sharding file stores sharding data that is only used by orbax. Therefore,
    # it shouldn't be included here. See b/279969796 for more details.
    if k.name == '_sharding':
      continue
    if k.name == '_METADATA':
      continue
    # array_metadatas is not a checkpoint param. Only used when ocdbt is used.
    # ocdbt is still disabled in some projects like paxml.
    if k.name == 'array_metadatas':
      continue
    tree = add_nested_key(tree, k.name.split('.'), k.name)
  return tree


def is_scalar(x):
  return isinstance(x, (int, float, np.number))


def fully_replicated_host_local_array_to_global_array(
    arr: jax.Array,
) -> jax.Array:
  """Converts a host local array from to global jax.Array.

  In most cases, the local array is expected to have been produced by pmap.

  Args:
    arr: Host local array

  Returns:
    A global array.
  """
  # input `arr` is fully replicated, so it's shape is the global shape.
  global_shape = arr.addressable_data(0).shape

  # Create a 1D mesh to create fully replicated global jax.Array.
  mesh = jax.sharding.Mesh(np.array(jax.devices()), axis_names=('x',))
  partition_spec = (
      jax.sharding.PartitionSpec(None)
      if global_shape
      else jax.sharding.PartitionSpec()
  )
  # pmap-produced Array has a "scrambled" device order.
  dbs = sorted(
      [shard.data for shard in arr.addressable_shards],
      key=lambda x: list(x.devices())[0].id,
  )
  if jax.config.jax_pmap_no_rank_reduction and global_shape != arr.shape:
    dbs = [s[0] for s in dbs]
    global_shape = global_shape[1:]
  result = jax.make_array_from_single_device_arrays(
      global_shape, jax.sharding.NamedSharding(mesh, partition_spec), dbs
  )
  return result
