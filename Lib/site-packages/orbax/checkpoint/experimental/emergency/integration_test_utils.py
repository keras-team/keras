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

"""Test base classes for emergency.CheckpointManager."""

# pylint: disable=protected-access

import math
from typing import Any

from absl import logging
from etils import epath
import jax
from jax.experimental import mesh_utils
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint.experimental.emergency import checkpoint_manager


PyTree = Any

CheckpointManager = checkpoint_manager.CheckpointManager
CheckpointManagerOptions = checkpoint_manager.CheckpointManagerOptions
LocalCheckpointOptions = checkpoint_manager.LocalCheckpointOptions
PersistentCheckpointOptions = checkpoint_manager.PersistentCheckpointOptions


def swap_slices_in_mesh(
    mesh: jax.sharding.Mesh, *, replica_axis_index: int = 0
) -> jax.sharding.Mesh:
  """Reverses the ordering of devices such that slices swap IDs."""
  devices = []
  for slice_id in range(
      multislice.slice_count(mesh, replica_axis_index=replica_axis_index)
  ):
    devices.append(
        multislice.slice_devices(
            mesh, replica_id=slice_id, replica_axis_index=replica_axis_index
        )
    )
  devices.reverse()
  devices = np.stack(devices, axis=replica_axis_index)
  return jax.sharding.Mesh(devices, mesh.axis_names)


def _replace_abstract_array_sharding_with_mesh(
    sds: jax.ShapeDtypeStruct, mesh: jax.sharding.Mesh
) -> jax.ShapeDtypeStruct:
  assert isinstance(sds.sharding, jax.sharding.NamedSharding)
  sds.sharding = jax.sharding.NamedSharding(mesh, sds.sharding.spec)
  return sds


def fill_unspecified_mesh_axes(
    parallelism_vals, target_product, parallelism_type
):
  """Evaluates unspecified DCN/ICI parallelism values."""
  if -1 in parallelism_vals:
    assert parallelism_vals.count(-1) == 1, (
        f'Found unspecified values (-1) for more than one {parallelism_type}   '
        '   parallelism axis. At most one axis can be unspecified.'
    )

    determined_val = target_product / np.prod(parallelism_vals) * -1

    assert determined_val >= 1 and determined_val.is_integer, (
        'Unspecified value unable to be determined with the given     '
        f' {parallelism_type} parallelism values'
    )

    parallelism_vals[parallelism_vals.index(-1)] = int(determined_val)

  target_type = 'slices' if parallelism_type == 'DCN' else 'devices per slice'
  assert np.prod(parallelism_vals) == target_product, (
      f'Number of {target_type} {target_product} does not match the product'
      f' of the {parallelism_type} parallelism {np.prod(parallelism_vals)}'
  )

  return parallelism_vals


def create_global_mesh(
    num_slices: int,
    *,
    replica_axis_index: int = 0,
    tensor_parallelism: int | None = None,
    fsdp_parallelism: int | None = None,
) -> jax.sharding.Mesh:
  """Creates a global mesh with the given number of slices."""
  devices = jax.devices()
  num_devices = len(devices)
  num_devices_per_slice = num_devices // num_slices
  tensor_parallelism = tensor_parallelism or 2
  fsdp_parallelism = fsdp_parallelism or -1

  if num_slices <= 1:
    raise ValueError(f'Must run with at least 2 slices. Found: {num_slices}.')

  mesh_axes = [
      'data',
      'stage',
      'fsdp',
      'fsdp_transpose',
      'sequence',
      'tensor',
      'expert',
      'autoregressive',
  ]
  if replica_axis_index != 0:
    if replica_axis_index == 2 or replica_axis_index == 5:
      raise ValueError(
          '`replica_axis` cannot take the place of `fsdp` or `tensor`.'
      )
    swap_axis_name = mesh_axes[replica_axis_index]
    mesh_axes[replica_axis_index] = 'data'
    mesh_axes[0] = swap_axis_name

  # data_parallelism,
  # pipeline_parallelism,
  # fsdp_parallelism,
  # fsdp_transpose_parallelism,
  # sequence_parallelism,
  # tensor_parallelism,
  # expert_parallelism,
  # autoregressive_parallelism,
  dcn_parallelism = [
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
  ]
  dcn_parallelism[replica_axis_index] = num_slices
  ici_parallelism = [
      1,
      1,
      fsdp_parallelism,
      1,
      1,
      tensor_parallelism,
      1,
      1,
  ]

  # Find possible unspecified parallelisms
  ici_parallelism = fill_unspecified_mesh_axes(
      ici_parallelism, num_devices_per_slice, 'ICI'
  )
  if math.prod(ici_parallelism) * num_slices != num_devices:
    raise ValueError(
        'Mesh could not be constructed with given parallelism dimensions.'
        f' Found: tensor_parallelism={tensor_parallelism},'
        f' fsdp_parallelism={fsdp_parallelism}, num_slices={num_slices},'
        f' num_devices={num_devices}.'
    )
  devices_array = mesh_utils.create_hybrid_device_mesh(
      ici_parallelism,
      dcn_parallelism,
      devices,
      allow_split_physical_axes=False,
  )
  logging.info(
      'Creating mesh with axes: %s',
      {axis: dim for axis, dim in zip(mesh_axes, devices_array.shape)},
  )
  return jax.sharding.Mesh(devices_array, mesh_axes)


def _log_array_info(name: str, arr: jax.Array):
  logging.info(
      '%s: shape=%s, dtype=%s, sharding=%s',
      name,
      arr.shape,
      arr.dtype,
      arr.sharding,
  )


def create_train_state(global_mesh: jax.sharding.Mesh) -> PyTree:
  """Create a fake train state for testing."""
  dim_size = 32
  train_state = {
      'a_1d': test_utils.create_sharded_array(
          np.arange(dim_size), global_mesh, jax.sharding.PartitionSpec(None)
      ),
      'b_1d': test_utils.create_sharded_array(
          np.arange(dim_size),
          global_mesh,
          jax.sharding.PartitionSpec('tensor'),
      ),
      'c_2d': test_utils.create_sharded_array(
          np.arange(dim_size**2).reshape((dim_size, dim_size)),
          global_mesh,
          jax.sharding.PartitionSpec(None, 'tensor'),
      ),
      'd_2d': test_utils.create_sharded_array(
          np.arange(dim_size**2).reshape((dim_size, dim_size)),
          global_mesh,
          jax.sharding.PartitionSpec('tensor', None),
      ),
      'e_2d': test_utils.create_sharded_array(
          np.arange(dim_size**2).reshape((dim_size, dim_size)),
          global_mesh,
          jax.sharding.PartitionSpec('tensor', 'fsdp'),
      ),
      'f_2d': test_utils.create_sharded_array(
          np.arange(dim_size**2).reshape((dim_size, dim_size)),
          global_mesh,
          jax.sharding.PartitionSpec('fsdp', 'tensor'),
      ),
      'g_2d': test_utils.create_sharded_array(
          np.arange(dim_size**2).reshape((dim_size, dim_size)),
          global_mesh,
          jax.sharding.PartitionSpec(None, None),
      ),
      'h_3d': test_utils.create_sharded_array(
          np.arange(dim_size**3).reshape((dim_size, dim_size, dim_size)),
          global_mesh,
          jax.sharding.PartitionSpec('tensor', None, 'fsdp'),
      ),
      'i_3d': test_utils.create_sharded_array(
          np.arange(dim_size**3).reshape((dim_size, dim_size, dim_size)),
          global_mesh,
          jax.sharding.PartitionSpec(None, None, 'tensor'),
      ),
      'j_3d': test_utils.create_sharded_array(
          np.arange(dim_size**3).reshape((dim_size, dim_size, dim_size)),
          global_mesh,
          jax.sharding.PartitionSpec(None, None, 'fsdp'),
      ),
      'k_3d': test_utils.create_sharded_array(
          np.arange(dim_size**3).reshape((dim_size, dim_size, dim_size)),
          global_mesh,
          jax.sharding.PartitionSpec(None, None, None),
      ),
      'scalar': test_utils.create_sharded_array(
          123, global_mesh, jax.sharding.PartitionSpec()
      ),
      # Taken from failing MaxText run.
      'custom_array': test_utils.create_sharded_array(
          np.arange(8192 * 64).reshape((8192, 64)),
          global_mesh,
          jax.sharding.PartitionSpec('tensor', None),
      ),
  }
  for k, v in train_state.items():
    _log_array_info(k, v)
  return train_state


def create_checkpoint_manager(
    global_mesh: jax.sharding.Mesh,
    abstract_state: PyTree,
    local_directory: epath.Path,
    persistent_directory: epath.Path,
    *,
    use_async: bool = True,
    replica_axis_index: int = 0,
) -> CheckpointManager:
  """Create CheckpointManager for testing."""
  options = CheckpointManagerOptions(
      local=LocalCheckpointOptions(save_interval_steps=2, max_to_keep=2),
      persistent=PersistentCheckpointOptions(
          save_interval_steps=5, max_to_keep=3
      ),
      enable_async_checkpointing=use_async,
      replica_axis_index=replica_axis_index,
  )
  return CheckpointManager(
      local_directory=local_directory,
      persistent_directory=persistent_directory,
      global_mesh=global_mesh,
      abstract_state=abstract_state,
      options=options,
  )
