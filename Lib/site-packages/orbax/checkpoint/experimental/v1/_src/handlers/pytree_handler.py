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

"""Implementation of `CheckpointableHandler` for PyTrees."""

import dataclasses
from typing import Awaitable, Sequence

from etils import epath
import jax
import numpy as np
from orbax.checkpoint import checkpoint_utils
from orbax.checkpoint import options as v0_options_lib
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.handlers import base_pytree_checkpoint_handler
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types

PathLike = path_types.PathLike
AsyncResponse = async_types.AsyncResponse
CheckpointableHandler = handler_types.CheckpointableHandler
PyTree = tree_types.PyTree


def _get_v0_save_args(
    checkpointable: PyTree,
    create_array_storage_options_fn: (
        options_lib.PyTreeOptions.Saving.CreateArrayStorageOptionsFn | None
    ),
) -> PyTree | None:
  """Returns save args that are compatible with the V0 API."""
  if create_array_storage_options_fn is None:
    return None

  def _leaf_get_v0_save_args(k, v):
    array_storage_options = create_array_storage_options_fn(k, v)
    save_dtype = (
        np.dtype(array_storage_options.dtype)
        if array_storage_options.dtype
        else None
    )
    return type_handlers.SaveArgs(
        dtype=save_dtype,
        chunk_byte_size=array_storage_options.chunk_byte_size,
        shard_axes=array_storage_options.shard_axes,
    )

  return jax.tree.map_with_path(_leaf_get_v0_save_args, checkpointable)


def create_v0_handler(
    context: context_lib.Context,
    *,
    type_handler_registry: type_handlers.TypeHandlerRegistry = type_handlers.GLOBAL_TYPE_HANDLER_REGISTRY,
    array_metadata_validator: array_metadata_store_lib.Validator = array_metadata_store_lib.Validator(),
) -> base_pytree_checkpoint_handler.BasePyTreeCheckpointHandler:
  """Creates a V0 handler from a V1 context."""
  return base_pytree_checkpoint_handler.BasePyTreeCheckpointHandler(
      save_concurrent_bytes=context.array_options.saving.concurrent_bytes,
      restore_concurrent_bytes=context.array_options.loading.concurrent_bytes,
      use_ocdbt=context.array_options.saving.use_ocdbt,
      use_zarr3=context.array_options.saving.use_zarr3,
      multiprocessing_options=v0_options_lib.MultiprocessingOptions(
          primary_host=context.multiprocessing_options.primary_host,
          active_processes=context.multiprocessing_options.active_processes,
          barrier_sync_key_prefix=context.multiprocessing_options.barrier_sync_key_prefix,
      ),
      type_handler_registry=type_handler_registry,
      enable_post_merge_validation=context.array_options.saving.enable_post_merge_validation,
      pytree_metadata_options=context.pytree_options.saving.pytree_metadata_options,
      array_metadata_validator=array_metadata_validator,
  )


def create_v0_save_args(
    context: context_lib.Context,
    checkpointable: PyTree,
) -> base_pytree_checkpoint_handler.BasePyTreeSaveArgs:
  """Creates v0 CheckpointArgs for saving."""
  return base_pytree_checkpoint_handler.BasePyTreeSaveArgs(
      item=checkpointable,
      save_args=_get_v0_save_args(
          checkpointable,
          context.pytree_options.saving.create_array_storage_options_fn,
      ),
      ocdbt_target_data_file_size=context.array_options.saving.ocdbt_target_data_file_size,
      enable_pinned_host_transfer=context.array_options.saving.enable_pinned_host_transfer,
  )


def create_v0_restore_args(
    context: context_lib.Context,
    abstract_checkpointable: PyTree | None,
) -> base_pytree_checkpoint_handler.BasePyTreeRestoreArgs:
  """Creates v0 CheckpointArgs for restoration."""

  def _set_enable_padding_and_truncation(a):
    if not isinstance(a, type_handlers.ArrayRestoreArgs):
      return a
    return dataclasses.replace(
        a,
        strict=not context.array_options.loading.enable_padding_and_truncation,
    )

  restore_args = checkpoint_utils.construct_restore_args(
      abstract_checkpointable
  )
  restore_args = jax.tree.map(_set_enable_padding_and_truncation, restore_args)
  return base_pytree_checkpoint_handler.BasePyTreeRestoreArgs(
      item=abstract_checkpointable,
      restore_args=restore_args,
  )


class PyTreeHandler(
    CheckpointableHandler[
        PyTree,
        PyTree,
        PyTree,
    ]
):
  """An implementation of `CheckpointableHandler` for PyTrees."""

  def __init__(
      self,
      *,
      context: context_lib.Context | None = None,
      type_handler_registry: type_handlers.TypeHandlerRegistry = type_handlers.GLOBAL_TYPE_HANDLER_REGISTRY,
      array_metadata_validator: array_metadata_store_lib.Validator = (
          array_metadata_store_lib.Validator()
      ),
  ):
    context = context_lib.get_context(context)
    self._context = context
    self._multiprocessing_options = context.multiprocessing_options
    self._handler_impl = create_v0_handler(
        context,
        type_handler_registry=type_handler_registry,
        array_metadata_validator=array_metadata_validator,
    )

  def _finalize(self, directory: path_types.Path):
    if multihost.is_primary_host(self._multiprocessing_options.primary_host):
      self._handler_impl.finalize(directory)

  async def _background_save(
      self,
      directory: path_types.Path,
      *,
      commit_futures: Sequence[future.Future],
      operation_id: str,
  ):
    for f in commit_futures:
      f.result()
    # Global sync to ensure all participating processes have completed their
    # save operations before proceeding to finalize.
    barrier_name = f'save_and_finalize_{operation_id}_commit_complete'
    multihost.sync_global_processes(
        barrier_name, processes=self._multiprocessing_options.active_processes
    )
    # Finalize.
    self._finalize(directory)
    # Global sync to ensure all hosts are aware that the finalize operation
    # has completed before returning to the user.
    barrier_name = f'save_and_finalize_{operation_id}_finalize_complete'
    multihost.sync_global_processes(
        barrier_name, processes=self._multiprocessing_options.active_processes
    )

  async def save(
      self, directory: path_types.PathLike, checkpointable: PyTree
  ) -> Awaitable[None]:
    directory = epath.Path(directory)
    commit_futures = await self._handler_impl.async_save(
        directory,
        args=create_v0_save_args(self._context, checkpointable),
    )
    assert commit_futures

    # TODO(b/398310070): Move operation ID generation to `Context`.
    operation_id = (
        synchronization.HandlerAwaitableSignalOperationIdGenerator.get_current_operation_id()
    )
    return self._background_save(
        directory, commit_futures=commit_futures, operation_id=operation_id
    )

  async def _background_load(
      self,
      directory: path_types.Path,
      abstract_checkpointable: PyTree | None = None,
  ) -> PyTree:
    return self._handler_impl.restore(
        directory,
        args=create_v0_restore_args(self._context, abstract_checkpointable),
    )

  async def load(
      self,
      directory: path_types.PathLike,
      abstract_checkpointable: PyTree | None = None,
  ) -> Awaitable[PyTree]:
    directory = epath.Path(directory)
    return self._background_load(directory, abstract_checkpointable)

  async def metadata(self, directory: path_types.PathLike) -> PyTree:
    directory = epath.Path(directory)
    return self._handler_impl.metadata(directory)
