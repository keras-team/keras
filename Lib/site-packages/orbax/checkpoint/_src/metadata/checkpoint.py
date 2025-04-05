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

"""Manages metadata of checkpoints at root and step level (not item level)."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import threading
from typing import Any, Protocol, TypeVar

from absl import logging
from etils import epath
from orbax.checkpoint._src.logging import step_statistics
from orbax.checkpoint._src.metadata import metadata_serialization_utils as utils

_STEP_METADATA_FILENAME = '_CHECKPOINT_METADATA'
_ROOT_METADATA_FILENAME = '_ROOT_METADATA'
_LEGACY_ROOT_METADATA_FILENAME = 'metadata'

CompositeCheckpointHandlerTypeStrs = utils.CompositeCheckpointHandlerTypeStrs
CheckpointHandlerTypeStr = utils.CheckpointHandlerTypeStr
CompositeItemMetadata = utils.CompositeItemMetadata
SingleItemMetadata = utils.SingleItemMetadata
StepStatistics = step_statistics.SaveStepStatistics
SerializedMetadata = TypeVar('SerializedMetadata', bound=dict[str, Any])


def _sanitize_metadata_path(path: epath.PathLike) -> epath.Path:
  """Sanitizes the path and returns it as an `epath.Path`."""
  path = epath.Path(path)
  if not path.exists():
    raise FileNotFoundError(f'Path does not exist: {path}')
  if not path.is_dir():
    raise NotADirectoryError(f'Path is not a directory: {path}')
  return path


def step_metadata_file_path(path: epath.PathLike) -> epath.Path:
  """The path to step metadata file for a given checkpoint directory."""
  return _sanitize_metadata_path(path) / _STEP_METADATA_FILENAME


def root_metadata_file_path(
    path: epath.PathLike, *, legacy: bool = False
) -> epath.Path:
  """The path to root metadata file for a given checkpoint directory."""
  filename = (
      _LEGACY_ROOT_METADATA_FILENAME if legacy else _ROOT_METADATA_FILENAME
  )
  return _sanitize_metadata_path(path) / filename


@dataclasses.dataclass
class StepMetadata:
  """Metadata of a checkpoint at step level (not per item).

  Attributes:
    item_handlers: Map of item name to its checkpoint handler. Or a single
      checkpoint handler for non composite checkpoints.
    item_metadata: Map of item name to its metadata. Or a single metadata for
      non composite checkpoints.
    metrics: User-provided metrics (accuracy, loss, etc.)
    performance_metrics: Performance metrics (time, memory, etc.)
    init_timestamp_nsecs: timestamp when uncommitted checkpoint was initialized.
      Specified as nano seconds since epoch. default=None.
    commit_timestamp_nsecs: commit timestamp of a checkpoint, specified as nano
      seconds since epoch. default=None.
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.
  """

  item_handlers: (
      dict[str, CheckpointHandlerTypeStr] | CheckpointHandlerTypeStr | None
  ) = dataclasses.field(
      default=None,
      metadata={'processor': utils.validate_and_process_item_handlers},
  )
  item_metadata: CompositeItemMetadata | SingleItemMetadata | None = (
      dataclasses.field(
          default=None,
          metadata={'processor': utils.validate_and_process_item_metadata},
      )
  )
  metrics: dict[str, Any] = dataclasses.field(
      default_factory=dict,
      metadata={'processor': utils.validate_and_process_metrics},
  )
  performance_metrics: StepStatistics = dataclasses.field(
      default_factory=StepStatistics,
      metadata={'processor': utils.validate_and_process_performance_metrics},
  )
  init_timestamp_nsecs: int | None = dataclasses.field(
      default=None,
      metadata={'processor': utils.validate_and_process_init_timestamp_nsecs},
  )
  commit_timestamp_nsecs: int | None = dataclasses.field(
      default=None,
      metadata={'processor': utils.validate_and_process_commit_timestamp_nsecs},
  )
  custom_metadata: dict[str, Any] = dataclasses.field(
      default_factory=dict,
      metadata={'processor': utils.validate_and_process_custom_metadata},
  )


@dataclasses.dataclass
class RootMetadata:
  """Metadata of a checkpoint at root level (contains all steps).

  Attributes:
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.
  """

  custom_metadata: dict[str, Any] | None = dataclasses.field(
      default_factory=dict,
      metadata={'processor': utils.validate_and_process_custom_metadata},
  )


class MetadataStore(Protocol):
  """Manages storage of `SerializedMetadata`."""

  def is_blocking_writer(self) -> bool:
    """Returns True if the store performs blocking writes, otherwise False."""
    ...

  def write(
      self,
      file_path: epath.PathLike,
      metadata: SerializedMetadata,
  ) -> None:
    """[Over]Writes `metadata` to `file`."""
    ...

  def read(
      self, file_path: epath.PathLike
  ) -> SerializedMetadata | None:
    """Reads `file` and returns `SerializedMetadata`."""
    ...

  def update(
      self,
      file_path: epath.PathLike,
      **kwargs,
  ) -> None:
    """Safely updates `SerializedMetadata` at `file`.

    If no updatable `SerializedMetadata` is found at `file`, then it creates a
    new one with `kwargs` attributes.

    Args:
      file_path: path to metadata file in checkpoint dir (for
        `RootMetadata`) or step dir (for `StepMetadata`).
      **kwargs: Attributes of `SerializedMetadata` in kwargs format.
    """
    ...

  def wait_until_finished(self) -> None:
    """Waits for completion of non blocking writes if applicable."""
    ...

  def close(self) -> None:
    """Closes the store after cleaning up resources if any."""
    ...


class _MetadataStoreImpl(MetadataStore):
  """Basic internal reusable impl of `SerializedMetadata` storage.

  It is neither thread safe, nor does it check for read/write capabilities.
  """

  def is_blocking_writer(self) -> bool:
    return True

  def write(
      self,
      file_path: epath.PathLike,
      metadata: SerializedMetadata,
  ) -> None:
    metadata_file = epath.Path(file_path)
    if not metadata_file.parent.name or not metadata_file.parent.exists():
      raise ValueError(f'Metadata path does not exist: {metadata_file.parent}')
    json_data = json.dumps(metadata)
    bytes_written = metadata_file.write_text(json_data)
    if bytes_written == 0:
      raise ValueError(
          f'Failed to write Metadata={metadata},'
          f' json={json_data} to {metadata_file}'
      )
    logging.log_every_n(
        logging.INFO,
        'Wrote Metadata=%s, json=%s to %s',
        100,
        metadata,
        json_data,
        metadata_file,
    )

  def read(
      self, file_path: epath.PathLike
  ) -> SerializedMetadata | None:
    metadata_file = epath.Path(file_path)
    if not metadata_file.exists():
      logging.warning(
          'Metadata file does not exist: %s', metadata_file
      )
      return None
    try:
      raw_data = metadata_file.read_text()
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error(
          'Failed to read Metadata file: %s, error: %s',
          metadata_file,
          e,
      )
      return None
    try:
      json_data = json.loads(raw_data)
    except json.decoder.JSONDecodeError as e:
      # TODO(b/340287956): Found empty metadata files, how is it possible.
      logging.error(
          'Failed to json parse Metadata file: %s, '
          'file content: %s, '
          'error: %s',
          metadata_file,
          raw_data,
          e,
      )
      return None
    logging.log_every_n(
        logging.INFO,
        'Read Metadata=%s from %s',
        500,
        json_data,
        metadata_file,
    )
    return json_data

  def update(
      self,
      file_path: epath.PathLike,
      **kwargs,
  ) -> None:
    metadata_file = epath.Path(file_path)
    metadata = dict(self.read(metadata_file) or {})
    for k, v in kwargs.items():
      metadata[k] = v
    self.write(metadata_file, metadata)
    logging.log_every_n(
        logging.INFO,
        'Updated Metadata=%s to %s',
        100,
        metadata,
        metadata_file,
    )


@dataclasses.dataclass
class _BlockingMetadataStore(MetadataStore):
  """Manages storage of `SerializedMetadata` with blocking writes.

  Write operations are thread safe: within a process multiple threads write
  without corrupting data.

  NOTE: Write operations are not guaranteed to be safe across processes. But it
  should be okay as writes are expected to be called from just one jax process.

  Read operations are inherently thread safe and *process safe* too.

  Attributes:
    enable_write: if True then write operations are allowed, otherwise write
      operations are **no op**. Read operations are always allowed.
  """

  enable_write: bool
  # TODO(niketkb): Support locking per path.
  _write_lock: threading.RLock = dataclasses.field(init=False)
  _store_impl: _MetadataStoreImpl = dataclasses.field(init=False)

  def __post_init__(self):
    self._store_impl = _MetadataStoreImpl()
    if self.enable_write:
      self._write_lock = threading.RLock()

  def is_blocking_writer(self) -> bool:
    return True

  def write(
      self,
      file_path: epath.PathLike,
      metadata: SerializedMetadata,
  ) -> None:
    if not self.enable_write:
      return
    with self._write_lock:
      self._store_impl.write(file_path, metadata)

  def read(
      self, file_path: epath.PathLike
  ) -> SerializedMetadata | None:
    return self._store_impl.read(file_path)

  def update(
      self,
      file_path: epath.PathLike,
      **kwargs,
  ) -> None:
    if not self.enable_write:
      return
    with self._write_lock:
      self._store_impl.update(file_path, **kwargs)


@dataclasses.dataclass
class _NonBlockingMetadataStore(MetadataStore):
  """Manages storage of `SerializedMetadata` with non blocking writes.

  By default it behaves like a read only `MetadataStore`. But the
  same instance is reused if user requests for a write-enabled instance in the
  same process.

  The writes are non blocking. Read responses don't reflect in progress writes.
  """

  enable_write: bool
  _write_lock: threading.RLock = dataclasses.field(init=False)
  _store_impl: _MetadataStoreImpl = dataclasses.field(init=False)
  # We need to make sure that only one thread writes/updates to a given path.
  # A single threaded executor is a simple solution. We can improve it by
  # introducing a multi threaded executor but setting up tasks such that all
  # tasks for a path is handled by the same thread.
  _single_thread_executor: concurrent.futures.ThreadPoolExecutor = (
      dataclasses.field(init=False)
  )
  # List of futures associated with write/update calls. It gets reset after each
  # `wait_until_finished()` call.
  _write_futures: list[concurrent.futures.Future[None]] = dataclasses.field(
      init=False
  )

  def __post_init__(self):
    self._store_impl = _MetadataStoreImpl()
    if self.enable_write:
      self._write_lock = threading.RLock()
      self._single_thread_executor = concurrent.futures.ThreadPoolExecutor(
          max_workers=1, thread_name_prefix='metadata_store'
      )
      self._write_futures = []

  def is_blocking_writer(self) -> bool:
    return False

  def _add_to_write_futures(
      self, future: concurrent.futures.Future[None]
  ) -> None:
    """Adds `future` to the `_write_futures` list while removing `done` futures."""
    if not self.enable_write:
      return
    with self._write_lock:
      self._write_futures.append(future)

  def _write_and_log(
      self,
      file_path: epath.PathLike,
      metadata: SerializedMetadata,
  ) -> None:
    """Writes `metadata` and logs error if any."""
    try:
      self._store_impl.write(file_path, metadata)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception(
          'Failed to write metadata=%s path=%s: %s',
          metadata,
          file_path,
          e,
      )
      raise

  def write(
      self,
      file_path: epath.PathLike,
      metadata: SerializedMetadata,
  ) -> None:
    """[Over]Writes `metadata` in non blocking manner."""
    if not self.enable_write:
      logging.warning(
          'Write requested but enable_write=false, metadata=%s'
          ' path=%s',
          metadata,
          file_path,
      )
      return
    with self._write_lock:
      future = self._single_thread_executor.submit(
          self._write_and_log, file_path, metadata
      )
      self._add_to_write_futures(future)

  def read(
      self, file_path: epath.PathLike
  ) -> SerializedMetadata | None:
    """Reads metadata from `file_path` and returns a `SerializedMetadata`."""
    return self._store_impl.read(file_path)

  def _update_and_log(self, file: epath.PathLike, **kwargs) -> None:
    """Updates metadata attributes and logs error if any."""
    try:
      self._store_impl.update(file, **kwargs)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception(
          'Failed to update metadata=%s path=%s: %s',
          kwargs,
          file,
          e,
      )
      raise

  def update(
      self,
      file_path: epath.PathLike,
      **kwargs,
  ) -> None:
    """Updates `SerializedMetadata` in non blocking manner.

    If no updatable `SerializedMetadata` is found at `file_path`, then it
    creates a new one with `kwargs` attributes.

    Args:
      file_path: path to the metadata file in checkpoint dir (for
        `RootMetadata`) or step dir (for `StepMetadata`).
      **kwargs: Attributes of `SerializedMetadata` is kwargs format.
    """
    if not self.enable_write:
      logging.warning(
          'Update requested but enable_write=false, kwargs=%s'
          ' file=%s',
          kwargs,
          file_path,
      )
      return
    with self._write_lock:
      future = self._single_thread_executor.submit(
          self._update_and_log, file_path, **kwargs
      )
      self._add_to_write_futures(future)

  def wait_until_finished(self) -> None:
    """Waits for completion of writes."""
    if not self.enable_write:
      return
    with self._write_lock:
      while self._write_futures:
        future = self._write_futures.pop(0)
        future.result()

  def close(self) -> None:
    """Closes the store after cleaning up resources if any."""
    if self.enable_write:
      with self._write_lock:
        self._single_thread_executor.shutdown()
        logging.info('Closing %s', self)


_METADATA_STORE_FOR_WRITES = _BlockingMetadataStore(enable_write=True)
_METADATA_STORE_FOR_READS = _BlockingMetadataStore(enable_write=False)
_METADATA_STORE_NON_BLOCKING_FOR_READS = (
    _NonBlockingMetadataStore(enable_write=False)
)


def metadata_store(
    *,
    enable_write: bool,
    blocking_write: bool = False,
) -> MetadataStore:
  """Returns `MetadataStore` instance based on `enable_write` value.

  Write operations are thread safe: within a process multiple threads write
  without corrupting data.

  NOTE: Write operations are not guaranteed to be safe across processes. But it
  should be okay as writes are expected to be called from just one jax process.

  Read operations are inherently thread safe and *process safe* too.

  NOTE: `MetadataStore` instance created with `enable_write=True`
  and `blocking_write=False` must be closed with `.close()` to release thread
  resources. Prefer to reuse an instance created for this scenario.

  Args:
    enable_write: if True then write operations are allowed, otherwise write
      operations are **no op**. Read operations are always allowed.
    blocking_write: if True then write operations are blocking, otherwise non
      blocking. Read responses don't reflect in progress writes.
  """
  if not blocking_write:
    if enable_write:
      return _NonBlockingMetadataStore(enable_write=True)
    return _METADATA_STORE_NON_BLOCKING_FOR_READS

  if enable_write:
    return _METADATA_STORE_FOR_WRITES
  return _METADATA_STORE_FOR_READS
