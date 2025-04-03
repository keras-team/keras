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

"""Global configuration options."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Protocol

import numpy as np
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types



@dataclasses.dataclass(frozen=True, kw_only=True)
class AsyncOptions:
  """Options used to configure async behavior.

  timeout_secs: The timeout in seconds for the async save operation.
  post_finalization_callback: A function that is called after the async save
    operation is complete.
  create_directories_asynchronously: If true, create directories asynchronously
    in the background.
  """

  timeout_secs: int = 600  # 10 minutes.
  post_finalization_callback: Callable[[], None] | None = None
  create_directories_asynchronously: bool = False


@dataclasses.dataclass(frozen=True, kw_only=True)
class MultiprocessingOptions:
  """Options used to configure multiprocessing behavior.

  primary_host: the host id of the primary host.  Default to 0.  If it's set
    to None, then all hosts will be considered as primary.  It's useful in
    the case that all hosts are only working with local storage.
  active_processes: A set of process indices (corresponding to
    `multihost.process_index()`) over which `CheckpointManager` is expected to
    be called. This makes it possible to have a `CheckpointManager` instance
    that runs over a subset of processes, rather than all processes as it is
    normally expected to do. If specified, `primary_host` must belong to
    `active_processes`.
  barrier_sync_fn: A function used to customize process synchronization
    behavior.
  barrier_sync_key_prefix: A string to be prepended to the barrier sync key
    used to synchronize processes. This is useful to avoid collisions with
    other barrier syncs if another CheckpointManager is being used
    concurrently.
  """

  primary_host: int | None = 0
  active_processes: set[int] | None = None
  barrier_sync_fn: multihost.BarrierSyncFn | None = None
  barrier_sync_key_prefix: str | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class FileOptions:
  """Options used to configure checkpoint directories and files.

  Attributes:
    path_permission_mode: Path permission mode for step directories, user
      metadata files. e.g. 0o750. Please check
      https://github.com/google/etils/blob/main/etils/epath/backend.py if your
  """

  path_permission_mode: int | None = None



@dataclasses.dataclass(frozen=True, kw_only=True)
class PyTreeOptions:
  """Options used to configure PyTree saving and loading.

  Attributes:
    saving: Options for saving PyTrees.
    loading: Options for loading PyTrees.
  """

  @dataclasses.dataclass(frozen=True, kw_only=True)
  class Saving:
    """Options for saving PyTrees.

    create_array_storage_options_fn: A function that is called in order to
      create `ArrayOptions.Saving.StorageOptions` for each leaf in a PyTree,
      when it is being saved. It is called similar to:
      `jax.tree.map_with_path(create_array_storage_options_fn, pytree_to_save)`.
      If provided, it overrides any default settings in
      `ArrayOptions.Saving.StorageOptions`.
    pytree_metadata_options: Options for managing PyTree metadata.
    partial_update: NOT IMPLEMENTED.
    """

    class CreateArrayStorageOptionsFn(Protocol):

      def __call__(
          self, key: tree_types.PyTreeKeyPath, value: Any
      ) -> ArrayOptions.Saving.StorageOptions:
        ...

    create_array_storage_options_fn: CreateArrayStorageOptionsFn | None = None
    pytree_metadata_options: tree_metadata.PyTreeMetadataOptions = (
        dataclasses.field(default_factory=tree_metadata.PyTreeMetadataOptions)
    )
    partial_update: bool = False

  @dataclasses.dataclass(frozen=True, kw_only=True)
  class Loading:
    """Options for loading PyTrees.

    partial_load: NOT IMPLEMENTED.
    """

    partial_load: bool = False

  saving: Saving = dataclasses.field(default_factory=Saving)
  loading: Loading = dataclasses.field(default_factory=Loading)
  # TODO(dnlng): Add `LeafHandlerRegistry`.


@dataclasses.dataclass(frozen=True, kw_only=True)
class ArrayOptions:
  """Options used to configure array saving and loading.

  Attributes:
    saving: Options for saving arrays.
    loading: Options for loading arrays.
  """

  @dataclasses.dataclass(frozen=True, kw_only=True)
  class Saving:
    """Options for saving arrays.

    Attributes:
      concurrent_bytes: Max concurrent bytes that are allowed for writing. Can
        help to reduce the possibility of OOM's when large checkpoints are
        saved.
      storage_options: Options used to customize array storage behavior for
        individual leaves. See below.
      use_ocdbt: Enables OCDBT format.
      use_zarr3: If True, use Zarr3 format.
      ocdbt_target_data_file_size: Specifies the target size (in bytes) of each
        OCDBT data file.  It only applies when OCDBT is enabled and Zarr3 must
        be turned on.  If left unspecified, default size is 2GB.  A value of 0
        indicates no maximum file size limit.  For best results, ensure
        chunk_byte_size is smaller than this value.  For more details, refer to
        https://google.github.io/tensorstore/kvstore/ocdbt/index.html#json-kvstore/ocdbt.target_data_file_size
      enable_pinned_host_transfer: If False, disables transfer to pinned host
        when copying from device to host, regardless of the presence of pinned
        host memory.
      enable_post_merge_validation: If True, enables validation of the
        parameters after the finalize step.
    """

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class StorageOptions:
      """Options used to customize array storage behavior for individual leaves.

      dtype:
        If provided, casts the parameter to the given dtype before saving.
        Note that the parameter must be compatible with the given type (e.g.
        jnp.bfloat16 is not compatible with np.ndarray).
      chunk_byte_size:
        This is an experimental feature that automatically chooses the largest
        chunk
        shape possible, while keeping the chunk byte size less than or equal
        to the
        specified chunk_byte_size. Both the write_chunk_shape and
        read_chunk_shape
        are automatically set to the chosen shape. This uses a greedy
        algorithm that
        prioritizes splitting the largest dimensions first.
      shard_axes: An optional list of axes that should be prioritized when
        sharding array for storage. If empty, storage sharding implementation
        will prioritize axes which are already sharded.
      """

      dtype: np.typing.DTypeLike | None = None
      chunk_byte_size: int | None = None
      shard_axes: tuple[int, ...] = tuple()

    concurrent_bytes: int | None = None
    storage_options: StorageOptions = dataclasses.field(
        default_factory=StorageOptions
    )
    use_ocdbt: bool = True
    use_zarr3: bool = True
    ocdbt_target_data_file_size: int | None = None
    enable_pinned_host_transfer: bool = False
    enable_post_merge_validation: bool = True

  @dataclasses.dataclass(frozen=True, kw_only=True)
  class Loading:
    """Options for loading arrays.

    concurrent_bytes: Max concurrent bytes that are allowed for reading. Can
      help to reduce the possibility of OOM's when large checkpoints are
      restored.
    enable_padding_and_truncation: If True, restoration allows silent
      truncating/padding of arrays if the stored array shape does not match the
      target shape. Otherwise, raises an error.
    """
    concurrent_bytes: int | None = None
    enable_padding_and_truncation: bool = False

  saving: Saving = dataclasses.field(default_factory=Saving)
  loading: Loading = dataclasses.field(default_factory=Loading)
