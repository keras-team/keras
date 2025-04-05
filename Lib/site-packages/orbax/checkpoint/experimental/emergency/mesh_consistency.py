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

"""Utils for ensuring mesh consistency in emergency checkpoint restoration."""

import json
import time
from typing import Any, List
from absl import logging
from etils import epath
import jax
from orbax.checkpoint import options as options_lib
from orbax.checkpoint.experimental.emergency import multihost as emergency_multihost
from orbax.checkpoint.path import atomicity_defaults
from orbax.checkpoint.path import step as step_lib



_PROCESS_METADATA_FOLDER = 'process_metadata'
_PROCESS_METADATA_FILE_NAME = 'process_metadata.json'
_GLOBAL_PROCESS_METADATA_FILE_NAME = 'global_process_metadata.json'
_MESH_METADATA_FILE_NAME = 'mesh_metadata.json'

_PRIMARY_REPLICA_ID = 0
_SECONDARY_REPLICA_ID = 1

PyTree = Any


def process_metadata_folder(
    directory: epath.Path
) -> epath.Path:
  return directory


def read_process_metadata(directory: epath.Path):
  """Read process metadata from the given path."""
  metadata_folder = process_metadata_folder(directory)
  if not metadata_folder.exists():
    raise FileNotFoundError(
        f'Process metadata folder does not exist at {metadata_folder}. The'
        ' local checkpoint cannot be restored.'
    )
  if step_lib.is_tmp_checkpoint(metadata_folder):
    raise ValueError(
        f'Process metadata folder was not finalized at {metadata_folder}.'
        ' The local checkpoint cannot be restored.'
    )
  logging.info('Loading process index metadata from %s', metadata_folder)

  distributed_to_device_ids = json.loads(
      (metadata_folder / _GLOBAL_PROCESS_METADATA_FILE_NAME).read_text()
  )
  device_ids = json.loads(
      (metadata_folder / _MESH_METADATA_FILE_NAME).read_text()
  )
  return distributed_to_device_ids, device_ids


async def save_process_metadata(
    directory: epath.Path,
    global_mesh: jax.sharding.Mesh,
    distributed_to_device_ids: List[List[int]],
):
  """Saves process metadata to local storage. Runs on every process."""
  metadata_folder = process_metadata_folder(directory)
  logging.info('Saving process index metadata at %s', metadata_folder)
  if metadata_folder.exists():
    logging.warning(
        'Process metadata folder already exists at %s. Overwriting.',
        metadata_folder,
    )
    metadata_folder.rmtree()

  multiprocessing_options = options_lib.MultiprocessingOptions(
      primary_host=None
  )
  tmp_path = atomicity_defaults.get_default_temporary_path_class(
      metadata_folder
  ).from_final(metadata_folder, multiprocessing_options=multiprocessing_options)
  await tmp_path.create()

  (tmp_path.get() / _GLOBAL_PROCESS_METADATA_FILE_NAME).write_text(
      json.dumps(distributed_to_device_ids)
  )
  (tmp_path.get() / _MESH_METADATA_FILE_NAME).write_text(
      json.dumps([int(id) for id in global_mesh.device_ids.flatten()])
  )
  tmp_path.finalize(
  )


def consistent_restore_mesh_from_metadata(
    global_mesh: jax.sharding.Mesh,
    current_distributed_to_device_ids: List[List[int]],
    previous_distributed_to_device_ids: List[List[int]],
    previous_device_ids: List[int],
) -> jax.sharding.Mesh:
  """Create a mesh consistent with the saved metadata.

  Each device has its own specific local shard.
  Naive example:
    Before restart: device X (with id 0) saves shard 0.
    After restart:  device X (with new id 1) reads shard 0.
  To ensure the same device X (with different software ids) reads the
  same locally available shard and represent it in the correct index
  within the global jax.Array, we use the `restore_mesh`.

  More context on `restore_mesh`:
  1. We can think of mesh as being backed by a list of devices.
  2. Default mesh follows the default device id order [0, ..., n-1]. Or
      the user may permute it according to their needs.
  3. After restart, the user will construct the same software mesh as (2).
  4. But a given hardware device may change its id because of scheduler
      or runtime quirks.
  5. Goal: construct the mesh with the same hardware device order as
      before restart, that may not follow the current software ids.
  5. Thus, we shuffle the device order within the mesh by checking how
      each device's software ids changed across restarts.

  Args:
    global_mesh: The global mesh, provided by the user.
    current_distributed_to_device_ids: The distributed id to range of device ids
      mapping of the current incarnation.
    previous_distributed_to_device_ids: The distributed id to range of device
      ids mapping of the previous incarnation.
    previous_device_ids: The device ids of the previous incarnation.

  Returns:
    A mesh that is the same as the mesh used to save the local checkpoint.
  """
  assert isinstance(previous_device_ids, list)
  logging.info(
      'From process metadata, distributed_to_device_ids=%s',
      previous_distributed_to_device_ids,
  )
  logging.info('From process metadata, device_ids=%s', previous_device_ids)
  consistent_mesh = emergency_multihost.consistent_restore_mesh(
      jax.devices(),
      global_mesh,
      previous_device_ids,
      previous_distributed_to_device_ids=previous_distributed_to_device_ids,
      current_distributed_to_device_ids=current_distributed_to_device_ids,
  )
  logging.info(
      'Created consistent mesh with device_ids=%s',
      consistent_mesh.device_ids.flatten(),
  )
  return consistent_mesh


def consistent_restore_mesh_to_global_mesh(
    state: PyTree,
    shardings: PyTree,
) -> PyTree:
  """Transfers from consistent restore mesh to global mesh."""
  logging.info('Transferring from consistent restore mesh to global mesh')

  start_transfer = time.time()
  resharded_state = jax.device_put(state, shardings, donate=True)
  transfer_elapsed_s = time.time() - start_transfer
  logging.info(
      'Finished transferring from consistent restore mesh to global mesh'
      ' in %.2fs',
      transfer_elapsed_s,
  )
  jax.monitoring.record_event_duration_secs(
      '/orbax/emergency/checkpoint/read/transfer_global_shard_duration_secs',
      transfer_elapsed_s,
  )

  return resharded_state
