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

"""Storage for `array_metadata.ArrayMetadata` (not value.ArrayMetadata)."""

import asyncio
import json
import threading
import time
from typing import Any, Iterator, List, Sequence, Tuple
from absl import logging
from etils import epath
import jax
from orbax.checkpoint._src.metadata import array_metadata as array_metadata_lib
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import types


class PathResolver:
  """Resolves paths for the ArrayMetadata store read and write."""

  _metadata_subdir = 'array_metadatas'

  def _file_name(self, process_index: int | str) -> str:
    return f'process_{process_index}'

  def get_process_index(self, file_path: epath.Path) -> int:
    """Returns the process index from the file path."""
    process_index = file_path.name.removeprefix('process_')
    if process_index.isdigit():
      return int(process_index)
    raise ValueError(
        f'Invalid ArrayMetadata file path: {file_path}; expected file name'
        ' to start with "process_"'
    )

  def get_write_file_path(
      self, checkpoint_dir: epath.Path, process_index: int
  ) -> epath.Path:
    """Returns the file path to write."""
    return (
        checkpoint_dir / self._metadata_subdir / self._file_name(process_index)
    )

  def get_read_file_paths(
      self, checkpoint_dir: epath.Path, process_index: int | None = None
  ) -> Iterator[epath.Path] | epath.Path | None:
    """Returns the file paths to read.

    Args:
      checkpoint_dir: The base path containing metadata for each process.
      process_index: The process index to read. If None, then read all processes
        under `checkpoint_dir`.

    Returns:
      Iterator of file paths to read if `process_index` is None. A file path to
      read if `process_index` is not None. None if `process_index` is not None
      but metadata file does not exist.
    """
    if process_index is None:
      file_name_pattern = self._file_name('*')
      return checkpoint_dir.glob(f'{self._metadata_subdir}/{file_name_pattern}')
    file_path = (
        checkpoint_dir / self._metadata_subdir / self._file_name(process_index)
    )
    if file_path.exists():
      return file_path
    return None


class SerDeserializer:
  """Serializes and deserializes `array_metadata.ArrayMetadata`."""

  def _to_dict(
      self, array_metadata: array_metadata_lib.ArrayMetadata
  ) -> dict[str, Any]:
    """Converts `array_metadata` to a dictionary."""
    return {
        'array_metadata': {
            'param_name': array_metadata.param_name,
            'write_shape': array_metadata.write_shape,
            'chunk_shape': array_metadata.chunk_shape,
            'ext_metadata': array_metadata.ext_metadata,
        }
    }

  def _from_dict(self, obj: dict[str, Any]) -> Any:
    """Converts a json object to `SerializedArrayMetadata` or `obj`."""
    if 'array_metadata' in obj:
      array_metadata = obj['array_metadata']
      return array_metadata_lib.SerializedArrayMetadata(
          param_name=array_metadata['param_name'],
          write_shape=tuple(array_metadata['write_shape']),
          chunk_shape=tuple(array_metadata['chunk_shape']),
          ext_metadata=array_metadata.get('ext_metadata'),
      )
    return obj

  def serialize(
      self, array_metadatas: Sequence[array_metadata_lib.ArrayMetadata]
  ) -> str:
    """Serializes `array_metadatas` to string."""
    obj = {
        'array_metadatas': [
            self._to_dict(array_metadata) for array_metadata in array_metadatas
        ]
    }
    return json.dumps(obj)

  def deserialize(
      self, serialized: str
  ) -> List[array_metadata_lib.SerializedArrayMetadata]:
    """Deserializes `serialized` to `array_metadata.ArrayMetadata`."""
    obj = json.loads(serialized, object_hook=self._from_dict)
    return obj['array_metadatas']


class Store:
  """Storage for `array_metadata.ArrayMetadata` (not value.ArrayMetadata)."""

  def __init__(
      self,
      path_resolver: PathResolver = PathResolver(),
      ser_deser: SerDeserializer = SerDeserializer(),
      primary_host: int | None = 0,  # None means all hosts are primary hosts.
      write_timeout_secs: int = 600,  # 10 minutes.
  ):
    self._path_resolver = path_resolver
    self._ser_deser = ser_deser
    self._primary_host = primary_host
    self._write_timeout_secs = write_timeout_secs

  def set_primary_host(self, primary_host: int | None) -> None:
    """Sets the primary host."""
    self._primary_host = primary_host

  async def _maybe_create_base_dir(self, base_dir: epath.Path) -> None:
    """Primary host creates the base directory, rest of the hosts wait."""
    if multihost.is_primary_host(self._primary_host):
      # primary host creates, rest of the hosts wait.
      return await asyncio.to_thread(
          base_dir.mkdir, parents=True, exist_ok=True
      )

    # non-primary host waits for primary host to create the base dir/folder.
    async def wait_for_base_dir_creation():
      while not await asyncio.to_thread(base_dir.exists):
        await asyncio.sleep(0.25)

    try:
      await asyncio.wait_for(
          wait_for_base_dir_creation(), timeout=self._write_timeout_secs
      )
    except asyncio.TimeoutError as e:
      primary_process = (
          'LOCAL' if self._primary_host is None else self._primary_host
      )
      raise ValueError(
          f'[process_index={multihost.process_index()}] Timed out waiting for'
          f' array_metadatas base directory creation: {base_dir}.'
          f' timeout={self._write_timeout_secs} seconds.'
          f' primary_process={primary_process}'
      ) from e

  async def write(
      self,
      checkpoint_dir: epath.Path,
      array_metadatas: Sequence[array_metadata_lib.ArrayMetadata],
      process_index: int,
  ) -> None:
    """Writes `array_metadatas` to a file under `checkpoint_dir`.

    See `PathResolver.get_write_file_path()` for the file path resolution.

    Args:
      checkpoint_dir: The base path containing metadata for each process.
      array_metadatas: The sequence of metadata to write.
      process_index: The Jax process index used to resolve the file path.
    """
    file_path = self._path_resolver.get_write_file_path(
        checkpoint_dir, process_index
    )
    await self._maybe_create_base_dir(file_path.parent)
    await asyncio.to_thread(
        file_path.write_text, self._ser_deser.serialize(array_metadatas)
    )
    logging.info(
        '[process=%s][thread=%s] Wrote %d array_metadata.ArrayMetadata to %s',
        multihost.process_index(),
        threading.current_thread().name,
        len(array_metadatas),
        file_path,
    )

  async def _get_array_metadatas(
      self,
      array_metadatas_file_path: epath.Path,
  ) -> Tuple[epath.Path, list[array_metadata_lib.SerializedArrayMetadata]]:
    serialized = await asyncio.to_thread(array_metadatas_file_path.read_text)
    return array_metadatas_file_path, self._ser_deser.deserialize(serialized)

  async def read(
      self,
      checkpoint_dir: epath.Path,
      process_index: int | None = None,
  ) -> (
      dict[int, List[array_metadata_lib.SerializedArrayMetadata]]
      | List[array_metadata_lib.SerializedArrayMetadata]
      | None
  ):
    """Reads `SerializedArrayMetadata` from storage under `checkpoint_dir`.

    Args:
      checkpoint_dir: The base path containing metadata for each process.
      process_index: The process index to read. If None, then read all processes
        under `checkpoint_dir`.

    Returns:
      A dictionary of process index to list of metadata if `process_index`
      is None. A list of metadata if `process_index` is not None. None if
      metadata does not exist.
    """
    if not await asyncio.to_thread(checkpoint_dir.exists):
      raise ValueError(
          f'Checkpoint directory does not exist: {checkpoint_dir}.'
      )
    start_time = time.time()
    file_paths = self._path_resolver.get_read_file_paths(
        checkpoint_dir, process_index
    )
    if file_paths is None:
      logging.warning(
          '[process=%s][thread=%s] No metadata found for process_index=%s,'
          ' checkpoint_dir=%s. If the checkpoint does not contain jax.Array'
          ' then it is expected. If checkpoint contains jax.Array then it'
          ' should lead to an error eventually; if no error is raised then it'
          ' is a bug.',
          multihost.process_index(),
          threading.current_thread().name,
          process_index,
          checkpoint_dir,
      )
      return None

    if isinstance(file_paths, epath.Path):
      _, result = await self._get_array_metadatas(file_paths)
      logging.vlog(
          1,
          '[process=%s][thread=%s] Read %s metadata from metadata path=%s'
          ' in %s seconds.',
          multihost.process_index(),
          threading.current_thread().name,
          len(result),
          file_paths,
          time.time() - start_time,
      )
      return result

    getter_coros = []
    for file_path in file_paths:
      getter_coros.append(self._get_array_metadatas(file_path))
    path_metadatas_pairs = await asyncio.gather(*getter_coros)
    result = {
        self._path_resolver.get_process_index(file_path): metadatas
        for file_path, metadatas in path_metadatas_pairs
    }
    if not result:
      logging.warning(
          '[process=%s][thread=%s] No metadata found for any process_index,'
          ' checkpoint_dir=%s. time elapsed=%s seconds. If the checkpoint does'
          ' not contain jax.Array then it is expected. If checkpoint contains'
          ' jax.Array then it should lead to an error eventually; if no error'
          ' is raised then it is a bug.',
          multihost.process_index(),
          threading.current_thread().name,
          checkpoint_dir,
          time.time() - start_time,
      )
      return None

    logging.vlog(
        1,
        '[process=%s][thread=%s] Read all metadata from checkpoint_dir=%s in %s'
        ' seconds.',
        multihost.process_index(),
        threading.current_thread().name,
        checkpoint_dir,
        time.time() - start_time,
    )
    return result


def resolve_array_metadata_store(
    type_handler_registry: types.TypeHandlerRegistry,
) -> Store | None:
  """Returns the ArrayMetadata Store or None.

  Gets ArrayMetadata Store from TypeHandler for jax.Array. ArrayMetadata
  persistence is only supported for jax.Array.

  Args:
    type_handler_registry: TypeHandlerRegistry to search TypeHandler for
      jax.Array.
  """
  try:
    array_handler = type_handler_registry.get(jax.Array)
  except ValueError:
    logging.warning(
        'No jax.Array in TypeHandlerRegistry: ArrayMetadata persistence is'
        ' disabled.'
    )
    return None
  else:
    if hasattr(array_handler, '_array_metadata_store'):
      return array_handler._array_metadata_store  # pylint: disable=protected-access
    else:
      logging.warning(
          'No ArrayMetadata Store defined in TypeHandler for jax.Array: %s!'
          ' Subchunking is disabled.',
          array_handler.__class__.__qualname__,
      )
      return None


class Validator:
  """Validates ArrayMetadata."""

  def validate_all_array_metadatas(
      self,
      array_metadatas: dict[
          int, List[array_metadata_lib.SerializedArrayMetadata]
      ],
  ) -> None:
    """Validates that all processes have the same array metadatas.

    Args:
      array_metadatas: A dictionary of process index to list of metadata.
    """
    start_time = time.time()
    ref_process_index, ref_process_array_metadatas = next(
        iter(array_metadatas.items())
    )
    if not ref_process_array_metadatas:
      raise ValueError(
          'ArrayMetadata Store contains no metadata for process_index='
          f'{ref_process_index}.'
      )
    if len(array_metadatas) == 1:
      # check if the number of processes are indeed just one.
      self._validate_process_count(ref_process_index=ref_process_index)
      logging.warning(
          '[process=%s][thread=%s] Skipped cross-host ArrayMetadata validation'
          ' because only one process is found: process_index=%s.',
          multihost.process_index(),
          threading.current_thread().name,
          ref_process_index,
      )
      return

    ref_process_cache = {
        array_metadata.param_name: array_metadata
        for array_metadata in ref_process_array_metadatas
    }
    for process_index, process_array_metadatas in array_metadatas.items():
      if process_index == ref_process_index:
        continue
      process_cache = {
          array_metadata.param_name: array_metadata
          for array_metadata in process_array_metadatas
      }
      # Check if the number of params is the same.
      self._validate_param_count(
          ref_process_index=ref_process_index,
          ref_process_array_metadatas=ref_process_array_metadatas,
          ref_process_cache=ref_process_cache,
          process_index=process_index,
          process_array_metadatas=process_array_metadatas,
          process_cache=process_cache,
      )
      # Check if the params are the same.
      self._validate_params(
          ref_process_index=ref_process_index,
          ref_process_cache=ref_process_cache,
          process_index=process_index,
          process_cache=process_cache,
      )
      # Check if the chunk_shape, write_shape and ext_metadata are the same for
      # each param.
      self._validate_metadata(
          ref_process_index=ref_process_index,
          ref_process_cache=ref_process_cache,
          process_index=process_index,
          process_cache=process_cache,
      )
    logging.info(
        '[process=%s][thread=%s] Validated ArrayMetadata from all %s hosts in'
        ' %s seconds.',
        multihost.process_index(),
        threading.current_thread().name,
        len(array_metadatas),
        time.time() - start_time,
    )

  def _validate_process_count(
      self,
      *,
      ref_process_index: int,
  ) -> None:
    """Validates that the number of processes is just one."""
    process_count = multihost.process_count()
    if process_count != 1:
      raise ValueError(
          'ArrayMetadata Store contains metadata from just one process'
          f' (process_index={ref_process_index}), but found'
          f' {process_count} processes.'
      )

  def _validate_param_count(
      self,
      *,
      ref_process_index: int,
      ref_process_array_metadatas: List[
          array_metadata_lib.SerializedArrayMetadata
      ],
      ref_process_cache: dict[str, array_metadata_lib.SerializedArrayMetadata],
      process_index: int,
      process_array_metadatas: List[array_metadata_lib.SerializedArrayMetadata],
      process_cache: dict[str, array_metadata_lib.SerializedArrayMetadata],
  ) -> None:
    """Validates that the number of params is the same."""
    if len(ref_process_array_metadatas) != len(process_array_metadatas):
      missing_in_process = ref_process_cache.keys() - process_cache.keys()
      missing_in_ref_process = process_cache.keys() - ref_process_cache.keys()
      diff_msg = 'Diff:'
      if missing_in_process:
        diff_msg += (
            f' process_index={process_index} is missing'
            f' {len(missing_in_process)} params: {missing_in_process}.'
        )
      if missing_in_ref_process:
        diff_msg += (
            f' process_index={ref_process_index} is missing'
            f' {len(missing_in_ref_process)} params:'
            f' {missing_in_ref_process}.'
        )
      raise ValueError(
          'ArrayMetadata Store contains different number of params:'
          f' process_index={ref_process_index} has'
          f' {len(ref_process_array_metadatas)}, but'
          f' process_index={process_index} has'
          f' {len(process_array_metadatas)} params. {diff_msg}'
      )

  def _validate_params(
      self,
      *,
      ref_process_index: int,
      ref_process_cache: dict[str, array_metadata_lib.SerializedArrayMetadata],
      process_index: int,
      process_cache: dict[str, array_metadata_lib.SerializedArrayMetadata],
  ) -> None:
    """Validates that the params are the same."""
    missing_in_process = ref_process_cache.keys() - process_cache.keys()
    if missing_in_process:
      raise ValueError(
          'ArrayMetadata Store contains different params: comparing with'
          f' process_index={ref_process_index},'
          f' process_index={process_index} is missing'
          f' {len(missing_in_process)} params: {missing_in_process}.'
      )
    missing_in_ref_process = process_cache.keys() - ref_process_cache.keys()
    if missing_in_ref_process:
      raise ValueError(
          'ArrayMetadata Store contains different params: comparing with'
          f' process_index={process_index},'
          f' process_index={ref_process_index} is missing'
          f' {len(missing_in_ref_process)} params: {missing_in_ref_process}.'
      )

  def _validate_metadata(
      self,
      *,
      ref_process_index: int,
      ref_process_cache: dict[str, array_metadata_lib.SerializedArrayMetadata],
      process_index: int,
      process_cache: dict[str, array_metadata_lib.SerializedArrayMetadata],
  ) -> None:
    """Validates that chunk_shape and write_shape are the same for a param."""
    for param_name, array_metadata in process_cache.items():
      ref_array_metadata = ref_process_cache[param_name]
      if array_metadata.chunk_shape != ref_array_metadata.chunk_shape:
        raise ValueError(
            'ArrayMetadata Store contains different chunk_shape for param:'
            f' {param_name}. process_index={process_index} has'
            f' {array_metadata.chunk_shape}, but'
            f' process_index={ref_process_index} has'
            f' {ref_array_metadata.chunk_shape}.'
        )
      if array_metadata.write_shape != ref_array_metadata.write_shape:
        raise ValueError(
            'ArrayMetadata Store contains different write_shape for param:'
            f' {param_name}. process_index={process_index} has'
            f' {array_metadata.write_shape}, but'
            f' process_index={ref_process_index} has'
            f' {ref_array_metadata.write_shape}.'
        )

      # Check if the RANDOM_KEY_IMPL is the same.
      if not isinstance(
          array_metadata.ext_metadata, type(ref_array_metadata.ext_metadata)
      ):
        raise ValueError(
            'Different ext_metadata types param:'
            f' {param_name}. process_index={process_index} has'
            f' {array_metadata.ext_metadata}, but'
            f' process_index={ref_process_index} has'
            f' {ref_array_metadata.ext_metadata}.'
        )

      if (
          isinstance(ref_array_metadata.ext_metadata, dict)
          and isinstance(array_metadata.ext_metadata, dict)
      ) and (
          (
              ref_key := ref_array_metadata.ext_metadata.get(
                  array_metadata_lib.RANDOM_KEY_IMPL
              )
          )
          != (
              arr_key := array_metadata.ext_metadata.get(
                  array_metadata_lib.RANDOM_KEY_IMPL
              )
          )
      ):
        raise ValueError(
            'ArrayMetadata Store contains different'
            f' ext_metadata.{array_metadata_lib.RANDOM_KEY_IMPL} for param:'
            f' {param_name}. process_index={process_index} has {ref_key}, but'
            f' process_index={ref_process_index} has {arr_key}.'
        )


async def validate_all_array_metadatas(
    validator: Validator,
    array_metadata_store: Store,
    directory: epath.Path,
) -> None:
  """Validates that all processes have the same array metadatas.

  Args:
    validator: The `Validator` instance for validation.
    array_metadata_store: The `Store` instance for reading metadata from
      `directory`.
    directory: The checkpoint directory with array_metadatas subdir.
  """
  array_metadatas = await array_metadata_store.read(directory)
  if array_metadatas is not None:
    assert isinstance(array_metadatas, dict)  # read all processes.
    validator.validate_all_array_metadatas(array_metadatas)
