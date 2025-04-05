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

"""Utils for debugging local checkpoints with missing chunks."""

import asyncio
from typing import Any
from absl import logging
from etils import epath
import jax
from orbax.checkpoint._src.arrays import abstract_arrays
from orbax.checkpoint._src.arrays import types
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.serialization import types as serialization_types
from orbax.checkpoint._src.tree import utils as tree_utils
import tensorstore as ts


Index = types.Index
Shape = types.Shape
ChunkId = tuple[int, ...]
PyTree = Any


def index_to_chunk_id(
    idx: Index, global_shape: Shape, shard_shape: Shape
) -> ChunkId:
  """Converts index to chunk id."""
  assert len(idx) == len(global_shape) == len(shard_shape)
  chunks = tuple([gs / ss for gs, ss in zip(global_shape, shard_shape)])
  chunk_id = [-1] * len(chunks)
  for d, sl in enumerate(idx):
    start = sl.start
    stop = sl.stop
    if start is None:
      start = 0
    if stop is None:
      stop = global_shape[d]
    assert sl.step == 1 or sl.step is None
    assert start % shard_shape[d] == 0 and stop % shard_shape[d] == 0
    assert stop - start == shard_shape[d]
    chunk_id[d] = start // shard_shape[d]
  return tuple(chunk_id)


def chunk_id_to_index(chunk_id: ChunkId, shard_shape: Shape) -> Index:
  """Converts chunk id to index."""
  assert len(chunk_id) == len(shard_shape)
  idx = []
  for d, dim_id in enumerate(chunk_id):
    start = dim_id * shard_shape[d]
    stop = start + shard_shape[d]
    idx.append(slice(start, stop))
  assert len(idx) == len(shard_shape)
  return tuple(idx)


async def get_chunk_ids_from_tensorstore(
    t: ts.TensorStore, *, use_zarr3: bool, shape: Shape | None = None
) -> list[ChunkId]:
  """List chunks available in the TensorStore KVStore."""
  separator = '/' if use_zarr3 else '.'
  kvstore = await t.kvstore.list()
  kvstore = [v.decode('utf-8') for v in kvstore]
  result = [
      tuple([int(x) for x in v.split(separator) if x != 'c'])
      for v in kvstore
      if 'zarr' not in v
  ]
  if shape is not None:
    for chunk_id in result:
      assert len(chunk_id) == len(shape)
  return result


async def open_tensorstore(
    directory: epath.Path,
    param_name: str,
    *,
    use_ocdbt: bool,
    use_zarr3: bool,
) -> ts.TensorStore:
  """Opens TensorStore for the given parameter for reading."""
  info = serialization_types.ParamInfo(
      name=param_name,
      path=directory / param_name,
      parent_dir=directory,
      is_ocdbt_checkpoint=use_ocdbt,
      use_zarr3=use_zarr3,
      ts_context=type_handlers.get_ts_context(use_ocdbt=use_ocdbt),
  )
  tspec = type_handlers.get_json_tspec_read(info, use_ocdbt=use_ocdbt)
  return await ts.open(
      ts.Spec(tspec),
      read=True,
      write=False,
      create=False,
      open=True,
      context=info.ts_context,
  )


async def get_present_and_missing_chunks(
    directory: epath.Path,
    param_name: str,
    array: jax.Array | jax.ShapeDtypeStruct,
    *,
    use_ocdbt: bool,
    use_zarr3: bool,
) -> tuple[list[ChunkId], list[ChunkId]]:
  """From TensorStore, get present and missing chunk ids."""
  t = await open_tensorstore(
      directory, param_name, use_ocdbt=use_ocdbt, use_zarr3=use_zarr3
  )
  present_chunk_ids = await get_chunk_ids_from_tensorstore(
      t, use_zarr3=use_zarr3, shape=array.shape
  )
  chunk_ids_set = set(present_chunk_ids)
  missing_chunk_ids = []
  for d, idx in array.sharding.devices_indices_map(array.shape).items():
    if d not in jax.local_devices():  # non-local indices not expected
      continue
    chunk_id = index_to_chunk_id(
        idx, array.shape, array.sharding.shard_shape(array.shape)
    )
    if chunk_id not in chunk_ids_set:
      missing_chunk_ids.append(chunk_id)
  return present_chunk_ids, missing_chunk_ids


async def print_chunk_debug_info(
    directory: epath.Path,
    restore_args: PyTree | None = None,
):
  """Prints debug info before restoring the parameter."""
  arrays = jax.tree.map(abstract_arrays.to_shape_dtype_struct, restore_args)
  param_names = tree_utils.get_param_names(arrays, include_empty_nodes=False)
  flat_arrays, _ = jax.tree.flatten(arrays)
  flat_param_names, _ = jax.tree.flatten(param_names)
  logging.vlog(1, 'Found %d arrays.', len(flat_arrays))
  logging.vlog(1, 'Param names to check: %s', flat_param_names)
  assert len(flat_arrays) == len(flat_param_names)

  chunk_infos = await asyncio.gather(*[
      get_present_and_missing_chunks(
          directory, n, a, use_ocdbt=True, use_zarr3=True
      )
      for n, a in zip(flat_param_names, flat_arrays)
  ])

  for arr, param_name, chunk_info in zip(
      flat_arrays, flat_param_names, chunk_infos
  ):
    present_chunk_ids, missing_chunk_ids = chunk_info
    shard_shape = arr.sharding.shard_shape(arr.shape)

    logging.vlog(1, 'Logging present and missing chunks for %s.', param_name)
    if present_chunk_ids:
      logging.info('Present chunks:')
      for chunk_id in present_chunk_ids:
        logging.vlog(
            1,
            '  [present] index: %s -- chunk ID: %s',
            chunk_id_to_index(chunk_id, shard_shape),
            chunk_id,
        )
    if missing_chunk_ids:
      logging.vlog(1, 'Missing chunks:')
      for chunk_id in missing_chunk_ids:
        logging.vlog(
            1,
            '  [missing] index: %s -- chunk ID: %s',
            chunk_id_to_index(chunk_id, shard_shape),
            chunk_id,
        )


def print_devices_indices_debug_info(
    restore_args: PyTree,
):
  """Prints debug info before restoring the parameter."""
  arrays = jax.tree.map(abstract_arrays.to_shape_dtype_struct, restore_args)
  param_names = tree_utils.get_param_names(arrays, include_empty_nodes=False)
  flat_arrays, _ = jax.tree.flatten(arrays)
  flat_param_names, _ = jax.tree.flatten(param_names)
  assert len(flat_arrays) == len(flat_param_names)

  for arr, param_name in zip(flat_arrays, flat_param_names):
    devices_indices_map = arr.sharding.devices_indices_map(arr.shape)
    logging.vlog(1, 'Device -> index map for %s.', param_name)
    for d, idx in devices_indices_map.items():
      logging.vlog(1, '  %s -> %s', d, idx)
