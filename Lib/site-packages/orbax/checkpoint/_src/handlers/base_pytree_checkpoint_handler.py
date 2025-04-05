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

"""BasePyTreeCheckpointHandler class.

Implementation of `CheckpointHandler` interface dealing with JAX PyTrees. Much
of the underlying reading/writing logic for individual leaf types can be
customized, and is delegated to the `TypeHandler` class.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import json
import sys
import time
from typing import Any, List, Optional, Sequence, Tuple, Union
import uuid

from absl import logging
from etils import epath
import humanize
import jax
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import options as options_lib
from orbax.checkpoint import utils
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.handlers import async_checkpoint_handler
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.metadata import empty_values
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import format_utils
from orbax.checkpoint._src.serialization import serialization
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.serialization import types
from orbax.checkpoint._src.tree import types as tree_types
from orbax.checkpoint._src.tree import utils as tree_utils
import tensorstore as ts



PyTree = Any
TupleKey = Tuple[str, ...]
RestoreArgs = type_handlers.RestoreArgs
ArrayRestoreArgs = type_handlers.ArrayRestoreArgs
SaveArgs = type_handlers.SaveArgs
ParamInfo = type_handlers.ParamInfo
TypeHandler = type_handlers.TypeHandler
TypeHandlerRegistry = type_handlers.TypeHandlerRegistry

# TODO(b/298487158) Clean up protected access.
LimitInFlightBytes = type_handlers.LimitInFlightBytes
CheckpointArgs = checkpoint_args.CheckpointArgs
register_with_handler = checkpoint_args.register_with_handler
get_param_names = tree_utils.get_param_names
PYTREE_METADATA_FILE = format_utils.PYTREE_METADATA_FILE

DEFAULT_CONCURRENT_GB = 96



def _default_sizeof_values(values: Sequence[Any]) -> Sequence[int]:
  return [sys.getsizeof(v) for v in values]


def _get_batch_memory_size(
    handler: TypeHandler, values: Sequence[Any]
) -> Tuple[int, int]:
  """Gets memory size for a batch of leaf values."""
  try:
    write_sizes, read_sizes = zip(*handler.memory_size(values))
  except NotImplementedError:
    logging.warning(
        '`memory_size` is not implemented for `TypeHandler` of type: %s. Using'
        ' the a default implementation to measure value memory consumption that'
        ' may result in inaccurate estimation.',
        type(handler),
    )
    write_sizes = read_sizes = _default_sizeof_values(values)
  assert len(write_sizes) == len(values)
  assert len(read_sizes) == len(values)
  return sum(write_sizes), sum(read_sizes)


def _log_io_metrics(
    size: int,
    start_time: float,
    bytes_per_sec_metric: str,
    bytes_metric: Optional[str] = None,
):
  """Logs the bytes per second metric."""
  time_elapsed = time.time() - start_time
  bytes_per_sec = (
      float('nan') if time_elapsed == 0 else float(size) / time_elapsed
  )
  logging.info(
      '[process=%d] %s: %s/s (total bytes: %s) (time elapsed: %s) (per-host)',
      multihost.process_index(),
      bytes_per_sec_metric,
      humanize.naturalsize(bytes_per_sec, binary=True),
      humanize.naturalsize(size, binary=True),
      humanize.naturaldelta(time_elapsed, minimum_unit='microseconds'),
  )
  jax.monitoring.record_event_duration_secs(bytes_per_sec_metric, bytes_per_sec)
  if bytes_metric is not None:
    jax.monitoring.record_event(bytes_metric, bytes=size)


@dataclasses.dataclass
class _BatchRequest:
  """Represents a a request for batched serialization or deserialization.

  Attributes:
    handler: Used to serialize or deserialize the parameters.
    keys: Used to identify the original tree keys so that the PyTree can be
      reconstructed.
    values: Values to serialize.
    infos: ParamInfos.
    args: List of SaveArgs or RestoreArgs.
  """

  handler: TypeHandler
  keys: List[str]
  values: List[Any]
  infos: List[ParamInfo]
  args: List[Union[SaveArgs, RestoreArgs]]

  def __post_init__(self):
    length = len(self.values)
    if not all((
        length == len(self.infos),
        length == len(self.args),
        length == len(self.keys),
    )):
      raise AssertionError('Found `_BatchRequest` with mismatched parameters.')


def batched_serialization_requests(
    tree: PyTree,
    param_infos: PyTree,
    args: PyTree,
    registry: TypeHandlerRegistry,
) -> List[_BatchRequest]:
  """Gets a list of batched serialization or deserialization requests."""
  grouped = {}

  def _group_value(
      keypath: Tuple[Any, ...],
      info: ParamInfo,
      value: Union[Any, tree_metadata.ValueMetadataEntry],
      arg: Union[SaveArgs, RestoreArgs],
  ):
    nonlocal grouped
    tuple_key = tree_utils.tuple_path_from_keypath(keypath)
    if info.skip_deserialize:
      return

    if isinstance(arg, RestoreArgs):
      assert isinstance(value, tree_metadata.ValueMetadataEntry), type(value)
      metadata_restore_type = value.value_type
      requested_restore_type = arg.restore_type or metadata_restore_type
      # TODO(cpgaffney): Add a warning message if the requested_restore_type
      # is not the same as the metadata_restore_type.
      if empty_values.is_empty_typestr(requested_restore_type):
        # Skip deserialization of empty node using TypeHandler.
        return
      type_for_registry_lookup = requested_restore_type
    elif isinstance(arg, SaveArgs):
      # Skip serialization of empty node using TypeHandler.
      if tree_utils.is_empty_node(value):
        return
      type_for_registry_lookup = type(value)
    else:
      raise AssertionError(
          f'Expected `RestoreArgs` or `SaveArgs`. Got {type(arg)}.'
      )

    try:
      handler = registry.get(type_for_registry_lookup)
    except ValueError as e:
      raise ValueError(
          f'TypeHandler lookup failed for: type={type_for_registry_lookup},'
          f' keypath={keypath}, ParamInfo={info}, RestoreArgs={arg},'
          f' value={value}'
      ) from e

    if handler not in grouped:
      grouped[handler] = _BatchRequest(handler, [], [], [], [])
    request = grouped[handler]
    grouped[handler] = dataclasses.replace(
        request,
        keys=request.keys + [tuple_key],
        values=request.values + [value],
        infos=request.infos + [info],
        args=request.args + [arg],
    )

  jax.tree_util.tree_map_with_path(
      _group_value,
      param_infos,
      tree,
      args,
  )
  return list(grouped.values())


def _fill_missing_save_or_restore_args(
    item: PyTree, args: Optional[PyTree], *, mode: str
) -> PyTree:
  """Fills in missing values in the tree of SaveArgs or RestoreArgs.

  Values may be "missing" because of empty nodes in `item`. After returning, all
  keys in `item`, with empty nodes or not, will have a corresponding value
  in the result.

  Args:
    item: tree to save or target to restore.
    args: tree of SaveArgs or RestoreArgs. May be None, if the user did not
      provide it.
    mode: 'save' or 'restore'.

  Returns:
    A tree of SaveArgs or RestoreArgs with missing values filled in.
  """

  # Because of empty states, the user-provided args may not contain
  # all necessary arguments. These should be filled in with default args.
  def _maybe_set_default_save_args(_, leaf_args):
    if isinstance(leaf_args, (SaveArgs, RestoreArgs)):
      return leaf_args
    elif mode == 'save':
      return SaveArgs()
    elif mode == 'restore':
      return RestoreArgs()
    else:
      raise ValueError(f'Unknown mode: {mode}.')

  return jax.tree_util.tree_map(
      _maybe_set_default_save_args,
      item,
      item if args is None else args,
      is_leaf=utils.is_empty_or_leaf,
  )




class BasePyTreeCheckpointHandler(
    async_checkpoint_handler.AsyncCheckpointHandler
):
  """A CheckpointHandler implementation for any PyTree structure.

  Largely serves as the implementation for `PyTreeCheckpointHandler`. Users are
  advised not to use this class directly.
  """

  def __init__(
      self,
      *,
      save_concurrent_bytes: Optional[int] = None,
      restore_concurrent_bytes: Optional[int] = None,
      use_ocdbt: bool = True,
      use_zarr3: bool = False,
      multiprocessing_options: options_lib.MultiprocessingOptions = options_lib.MultiprocessingOptions(),
      type_handler_registry: TypeHandlerRegistry = type_handlers.GLOBAL_TYPE_HANDLER_REGISTRY,
      enable_post_merge_validation: bool = True,
      pytree_metadata_options: tree_metadata.PyTreeMetadataOptions = (
          tree_metadata.PYTREE_METADATA_OPTIONS
      ),
      array_metadata_validator: array_metadata_store_lib.Validator = (
          array_metadata_store_lib.Validator()
      ),
  ):
    """Creates BasePyTreeCheckpointHandler.

    Args:
      save_concurrent_bytes: max concurrent bytes that are allowed to be
        written. Can help to reduce the possibility of OOM's when large
        checkpoints are saved.
      restore_concurrent_bytes: max concurrent bytes that are allowed to be
        restored. Can help to reduce the possibility of OOM's when large
        checkpoints are restored.
      use_ocdbt: Whether to use OCDBT format for saving.
      use_zarr3: If True, use Zarr ver3 otherwise Zarr ver2.
      multiprocessing_options: See orbax.checkpoint.options.
      type_handler_registry: a type_handlers.TypeHandlerRegistry. If not
        specified, the global type handler registry will be used.
      enable_post_merge_validation: If True, enables validation of the
        parameters after the finalize step.
      pytree_metadata_options: `PyTreeMetadataOptions` to manage metadata.
      array_metadata_validator: Validator for ArrayMetadata.
    """
    self._save_concurrent_bytes = save_concurrent_bytes
    self._restore_concurrent_bytes = restore_concurrent_bytes
    self._use_ocdbt = use_ocdbt
    self._use_zarr3 = use_zarr3
    self._primary_host = multiprocessing_options.primary_host
    self._type_handler_registry = type_handler_registry
    self._enable_post_merge_validation = enable_post_merge_validation
    self._pytree_metadata_options = pytree_metadata_options
    # Get ArrayMetadata Store from TypeHandler for jax.Array.
    # ArrayMetadata persistence is only supported for jax.Array.
    self._array_metadata_store = (
        array_metadata_store_lib.resolve_array_metadata_store(
            type_handler_registry
        )
    )
    if self._array_metadata_store:
      self._array_metadata_store.set_primary_host(self._primary_host)
    self._array_metadata_validator = array_metadata_validator


    jax.monitoring.record_event(
        '/jax/orbax/pytree_checkpoint_handler/init/ocdbt'
    )
    logging.info(
        'Created BasePyTreeCheckpointHandler: use_ocdbt=%s, use_zarr3=%s,'
        ' pytree_metadata_options=%s, array_metadata_store=%s',
        self._use_ocdbt,
        self._use_zarr3,
        self._pytree_metadata_options,
        self._array_metadata_store,
    )

  def get_param_names(self, item: PyTree) -> PyTree:
    """Gets parameter names for PyTree elements."""
    return get_param_names(item)

  def _get_param_infos(
      self,
      item: PyTree,
      directory: epath.Path,
      *,
      use_ocdbt: bool = True,
      use_zarr3: Optional[bool] = None,
      ocdbt_target_data_file_size: Optional[int] = None,
      enable_pinned_host_transfer: bool = False,
      byte_limiter: Optional[serialization.ByteLimiter] = None,
      raise_array_data_missing_error: bool = True,
  ) -> PyTree:
    """Returns parameter information for elements in `item`.

    At minimum, this method should extract the names of each parameter for
    saving/restoring.

    Args:
      item: a PyTree to extract information from.
      directory: a directory where checkpoint files are located.
      use_ocdbt: Whether to use OCDBT for writing or reading.
      use_zarr3: Whether to use zarr3.
      ocdbt_target_data_file_size: Specifies the target size (in bytes) of each
        OCDBT data file.
      enable_pinned_host_transfer: See ParamInfo docs.
      byte_limiter: ByteLimiter object.
      raise_array_data_missing_error: See documentation in ParamInfo.

    Returns:
      A PyTree matching `item` of ParamInfo.
    """
    if use_zarr3 is None:
      use_zarr3 = self._use_zarr3
    names = self.get_param_names(item)
    ts_context = ts_utils.get_ts_context(use_ocdbt=use_ocdbt)

    def _param_info(name, value):
      skip_deserialize = False
      if isinstance(value, tree_metadata.ValueMetadataEntry):
        skip_deserialize = value.skip_deserialize
      return ParamInfo(
          name=name,
          path=(directory / name),
          parent_dir=directory,
          skip_deserialize=skip_deserialize,
          is_ocdbt_checkpoint=use_ocdbt,
          use_zarr3=use_zarr3,
          enable_pinned_host_transfer=enable_pinned_host_transfer,
          ocdbt_target_data_file_size=ocdbt_target_data_file_size,
          byte_limiter=byte_limiter,
          ts_context=ts_context,
          value_typestr=types.get_param_typestr(
              value, self._type_handler_registry, self._pytree_metadata_options
          ),
          raise_array_data_missing_error=raise_array_data_missing_error,
      )

    return jax.tree.map(
        _param_info, names, item, is_leaf=utils.is_empty_or_leaf
    )

  async def async_save(
      self,
      directory: epath.Path,
      args: BasePyTreeSaveArgs,
  ) -> Optional[List[future.Future]]:
    """Saves a PyTree to a given directory.

    This operation is compatible with a multi-host, multi-device setting. Tree
    leaf values must be supported by the type_handler_registry given in the
    constructor. Standard supported types include Python scalars, `np.ndarray`,
    `jax.Array`, and strings.

    After saving, all files will be located in "directory/".
    A JSON metadata file will be present to store the tree structure.

    Example usage::

      ckptr = Checkpointer(BasePyTreeCheckpointHandler())
      item = {
          'layer0': {
              'w': np.ndarray(...),
              'b': np.ndarray(...),
          },
          'layer1': {
              'w': np.ndarray(...),
              'b': np.ndarray(...),
          },
      }
      # Note: save_args may be None if no customization is desired for saved
      # parameters.
      # Eventually calls through to `async_save`.
      ckptr.save(path, item, save_args)

    Args:
      directory: save location directory.
      args: `BasePyTreeSaveArgs` (see below).

    Returns:
      A Future that will commit the data to `directory` when awaited. Copying
      the data from its source will be awaited in this function.
    """
    start_time = time.time()
    item = args.item
    if not item:
      raise ValueError('Found empty item.')
    save_args = args.save_args
    ocdbt_target_data_file_size = args.ocdbt_target_data_file_size
    custom_metadata = args.custom_metadata

    save_args = _fill_missing_save_or_restore_args(item, save_args, mode='save')
    byte_limiter = serialization.get_byte_limiter(self._save_concurrent_bytes)
    param_infos = self._get_param_infos(
        item,
        directory,
        use_ocdbt=self._use_ocdbt,
        ocdbt_target_data_file_size=ocdbt_target_data_file_size,
        enable_pinned_host_transfer=args.enable_pinned_host_transfer,
        byte_limiter=byte_limiter,
    )
    assert all(
        leaf.parent_dir == directory for leaf in jax.tree.leaves(param_infos)
    )

    serialize_ops = []  # List of (coros -> List of futures)
    batch_requests = batched_serialization_requests(
        item,
        param_infos,
        save_args,
        self._type_handler_registry,
    )
    tree_memory_size = 0
    for request in batch_requests:
      serialize_ops += [
          request.handler.serialize(request.values, request.infos, request.args)
      ]
      write_size, _ = _get_batch_memory_size(request.handler, request.values)
      tree_memory_size += write_size
    # Await copy futures. Returns List[List[future.Future]].
    commit_futures = await asyncio.gather(*serialize_ops)
    # Flatten to List[future.Future].
    commit_futures, _ = jax.tree.flatten(commit_futures)

    if logging.vlog_is_on(1):
      logging.vlog(1, 'param_info: %s', param_infos)
      logging.vlog(1, 'save_args: %s', save_args)

    save_futures = []
    if multihost.is_primary_host(self._primary_host):
      operation_id = (
          synchronization.HandlerAwaitableSignalOperationIdGenerator.get_current_operation_id()
      )
      save_futures.append(
          future.CommitFuture(
              self._write_metadata_after_commits(
                  commit_futures,
                  checkpoint_dir=directory,
                  param_infos=param_infos,
                  save_args=save_args,
                  custom_metadata=custom_metadata,
                  use_zarr3=self._use_zarr3,
                  operation_id=operation_id,
              ),
              name='write_metadata_after_commits',
          )
      )
    else:
      save_futures += commit_futures

    _log_io_metrics(
        tree_memory_size,
        start_time,
        '/jax/checkpoint/write/blocking_bytes_per_sec',
    )
    return [
        future.ChainedFuture(
            save_futures,
            functools.partial(
                _log_io_metrics,
                tree_memory_size,
                start_time,
                '/jax/checkpoint/write/bytes_per_sec',
                '/jax/checkpoint/write/bytes',
            ),
        )
    ]

  def save(self, directory: epath.Path, *args, **kwargs):
    """Saves the provided item.

    Blocks until both copy and commit complete.

    See async_save.

    Args:
      directory: the directory to save to.
      *args: additional arguments for save.
      **kwargs: additional arguments for save.
    """

    async def async_save(*args, **kwargs):
      commit_futures = await self.async_save(*args, **kwargs)  # pytype: disable=bad-return-type
      # Futures are already running, so sequential waiting is equivalent to
      # concurrent waiting.
      if commit_futures:  # May be None.
        for f in commit_futures:
          f.result()  # Block on result.

    asyncio_utils.run_sync(async_save(directory, *args, **kwargs))

  async def _maybe_deserialize(
      self,
      item: PyTree,
      metadata: PyTree,
      param_infos: PyTree,
      restore_args: PyTree,
  ) -> Tuple[int, PyTree]:
    """Deserializes values or skips."""
    flat_metadata = tree_utils.to_flat_dict(metadata)
    byte_limiter = serialization.get_byte_limiter(
        self._restore_concurrent_bytes
    )
    param_infos = jax.tree.map(
        lambda info: dataclasses.replace(info, byte_limiter=byte_limiter),
        param_infos,
    )
    batch_requests = batched_serialization_requests(
        metadata,
        param_infos,
        restore_args,
        self._type_handler_registry,
    )
    deserialized_batches = []
    deserialized_batches_ops = []
    for request in batch_requests:
      deserialized_batches_ops.append(
          request.handler.deserialize(request.infos, request.args)
      )
    deserialized_batches += await asyncio.gather(*deserialized_batches_ops)

    tree_memory_size = 0
    flat_restored = {}
    for request, deserialized in zip(batch_requests, deserialized_batches):
      _, read_size = _get_batch_memory_size(request.handler, deserialized)
      tree_memory_size += read_size
      for key, value in zip(request.keys, deserialized):
        flat_restored[key] = value
    # Add in empty nodes from the metadata tree.
    for key in flat_metadata.keys():
      if key not in flat_restored:
        flat_restored[key] = empty_values.get_empty_value_from_typestr(
            flat_metadata[key].value_type, self._pytree_metadata_options
        )
    # Restore using `item` as the target structure. If there are any custom
    # nodes (e.g. optax.EmptyState), these will replace None values in
    # flat_restored.
    return tree_memory_size, tree_utils.from_flat_dict(
        flat_restored, target=item
    )

  def restore(
      self,
      directory: epath.Path,
      args: Optional[BasePyTreeRestoreArgs] = None,
  ) -> PyTree:
    """Restores a PyTree from the checkpoint directory at the given path.

    In the most basic case, only `directory` is required. The tree will be
    restored exactly as saved, and all leaves will be restored as the correct
    types (assuming the tree metadata is present).

    However, `restore_args` is often required as well. This PyTree gives a
    `RestoreArgs` object (or subclass) for every leaf in the tree. Many types,
    such as string or `np.ndarray` do not require any special options for
    restoration. When restoring an individual leaf as `jax.Array`, however,
    some properties may be required.

    One example is `sharding`, which defines how a `jax.Array` in the restored
    tree should be partitioned. `mesh` and `mesh_axes` can also be used to
    specify `sharding`, but `sharding` is the preferred way of specifying this
    partition since `mesh` and `mesh_axes` only constructs
    `jax.sharding.NamedSharding`. For more information, see `ArrayTypeHandler`
    documentation and JAX sharding documentation.

    Example::

      ckptr = Checkpointer(BasePyTreeCheckpointHandler())
      restore_args = {
          'layer0': {
              'w': RestoreArgs(),
              'b': RestoreArgs(),
          },
          'layer1': {
              'w': ArrayRestoreArgs(
                  # Restores as jax.Array, regardless of how it was saved.
                  restore_type=jax.Array,
                  sharding=jax.sharding.Sharding(...),
                  # Warning: may truncate or pad!
                  global_shape=(x, y),
                ),
              'b': ArrayRestoreArgs(
                  restore_type=jax.Array,
                  sharding=jax.sharding.Sharding(...),
                  global_shape=(x, y),
                ),
          },
      }
      ckptr.restore(path, restore_args=restore_args)

    Providing `item` is typically only necessary when restoring a custom PyTree
    class (or when using transformations). In this case, the restored object
    will take on the same structure as `item`.

    Example::

      @flax.struct.dataclass
      class TrainState:
        layer0: dict[str, jax.Array]
        layer1: dict[str, jax.Array]

      ckptr = Checkpointer(BasePyTreeCheckpointHandler())
      train_state = TrainState(
          layer0={
              'w': jax.Array(...),  # zeros
              'b': jax.Array(...),  # zeros
          },
          layer1={
              'w': jax.Array(...),  # zeros
              'b': jax.Array(...),  # zeros
          },
      )
      restore_args = jax.tree.map(_make_restore_args, train_state)
      ckptr.restore(path, item=train_state, restore_args=restore_args)
      # restored tree is of type `TrainState`.

    Args:
      directory: saved checkpoint location directory.
      args: `BasePyTreeRestoreArgs` (see below).

    Returns:
      A PyTree matching the structure of `item`.

    Raises:
      FileNotFoundError: `directory` does not exist or is missing required files
      ValueError: `transforms` is provided without `item`.
      ValueError: `transforms` contains elements with `multi_value_fn`.
    """
    start_time = time.time()
    args = args or BasePyTreeRestoreArgs()
    item = args.item
    restore_args = args.restore_args

    logging.vlog(1, 'directory=%s, restore_args=%s', directory, restore_args)
    if not directory.exists():
      raise FileNotFoundError(
          f'Requested directory for restore does not exist at {directory}'
      )
    # Get value metadata tree and use_zarr3 from serialized pytree metadata.
    internal_tree_metadata = self._read_metadata_file(directory)
    value_metadata_tree = internal_tree_metadata.as_nested_tree()
    if not value_metadata_tree:
      raise ValueError(
          f'Found empty checkpoint PyTree metadata in directory={directory}.'
      )
    use_zarr3 = (
        internal_tree_metadata.use_zarr3
        if internal_tree_metadata.use_zarr3 is not None
        else self._use_zarr3
    )
    raise_array_data_missing_error = (
        internal_tree_metadata.store_array_data_equal_to_fill_value
    )
    del internal_tree_metadata
    # Prep for restore.
    if item is None:
      item = value_metadata_tree
    restore_args = _fill_missing_save_or_restore_args(
        item, restore_args, mode='restore'
    )
    restore_args = tree_metadata.serialize_tree(
        restore_args, self._pytree_metadata_options
    )
    param_infos = self._get_param_infos(
        item=value_metadata_tree,
        directory=directory,
        use_ocdbt=type_handlers.is_ocdbt_checkpoint(directory),
        use_zarr3=use_zarr3,
        raise_array_data_missing_error=raise_array_data_missing_error,
    )
    # Begin restore.
    tree_memory_size, restored_item = asyncio_utils.run_sync(
        self._maybe_deserialize(
            item, value_metadata_tree, param_infos, restore_args
        )
    )

    if logging.vlog_is_on(1):
      logging.vlog(1, 'param_infos: %s', param_infos)
      logging.vlog(1, 'checkpoint_restore_args: %s', restore_args)
      logging.vlog(1, 'restored_item: %s', jax.tree.structure(restored_item))
      logging.vlog(
          1,
          'ts_metrics: %s',
          json.dumps(ts.experimental_collect_matching_metrics('/tensorstore/')),
      )

    _log_io_metrics(
        tree_memory_size,
        start_time,
        '/jax/checkpoint/read/bytes_per_sec',
        '/jax/checkpoint/read/bytes',
    )
    return restored_item

  async def _get_param_infos_with_write_shape(
      self,
      param_infos: PyTree,
      checkpoint_dir: epath.Path,
      array_metadata_store: array_metadata_store_lib.Store,
  ) -> PyTree:
    """Returns `param_infos` updated with `write_shape`.

    Args:
      param_infos: A PyTree of ParamInfo to be updated.
      checkpoint_dir: The checkpoint directory where write_shape metadata is
        saved in ArrayMetadata store.
      array_metadata_store: The ArrayMetadata store to read write_shape metadata
        from.
    """
    if not utils.is_primary_host(self._primary_host):
      return param_infos
    # Extract write_shape from ArrayMetadata for current process_index.
    process_index = multihost.process_index()
    array_metadatas = await array_metadata_store.read(
        checkpoint_dir, process_index=process_index
    )
    if array_metadatas is None:
      jax_array_param_info = type_handlers.any_jax_array_param_info(param_infos)
      if jax_array_param_info is not None:
        raise ValueError(
            f'No ArrayMetadata found for process_index={process_index} in the'
            f' checkpoint directory: {checkpoint_dir}. But input PyTree'
            ' contains at least one jax.Array param_info:'
            f' {jax_array_param_info}.'
        )
      return param_infos

    assert isinstance(array_metadatas, list)
    array_metadatas_cache = {
        array_metadata.param_name: array_metadata
        for array_metadata in array_metadatas
    }

    def update_param_info(param_info: types.ParamInfo) -> types.ParamInfo:
      if not type_handlers.represents_jax_array(param_info):
        return param_info
      if param_info.name not in array_metadatas_cache:
        raise ValueError(
            f'No ArrayMetadata found for param_info: {param_info}, checkpoint'
            f' directory: {checkpoint_dir}, process_index={process_index}.'
        )
      return dataclasses.replace(
          param_info,
          write_shape=array_metadatas_cache[param_info.name].write_shape,
      )

    return jax.tree.map(update_param_info, param_infos)

  def _write_metadata_file(
      self,
      directory: epath.Path,
      *,
      param_infos: PyTree,
      save_args: PyTree,
      custom_metadata: tree_types.JsonType | None = None,
      use_zarr3: bool = False,
  ) -> future.Future:
    async def _save_fn(param_infos):
      if utils.is_primary_host(self._primary_host):
        metadata_write_start_time = time.time()
        path = directory / PYTREE_METADATA_FILE
        metadata_content = tree_metadata.InternalTreeMetadata.build(
            param_infos,
            save_args=save_args,
            use_zarr3=use_zarr3,
            custom_metadata=custom_metadata,
            pytree_metadata_options=self._pytree_metadata_options,
        )
        logging.vlog(
            1,
            'Writing pytree metadata file: %s with pytree_metadata_options: %s',
            path,
            self._pytree_metadata_options,
        )
        path.write_text(json.dumps(metadata_content.to_json()))
        jax.monitoring.record_event_duration_secs(
            '/jax/checkpoint/write/async/metadata_write_duration_secs',
            time.time() - metadata_write_start_time,
        )

    return future.CommitFuture(
        _save_fn(param_infos),
    )

  async def _write_metadata_after_commits(
      self,
      commit_futures: List[future.Future],
      checkpoint_dir: epath.Path,
      *,
      param_infos: PyTree,
      save_args: PyTree,
      custom_metadata: tree_types.JsonType | None = None,
      use_zarr3: bool,
      operation_id: str | None = None,
  ) -> None:
    if not utils.is_primary_host(self._primary_host):
      return
    for commit_future in commit_futures:
      await asyncio.to_thread(commit_future.result)
    # `write_shape` is extracted from ArrayMetadata store saved during
    # materialization of commit_futures. Then it is written to the pytree
    # metadata.
    # TODO(b/390465017): Simplify all metadata related code in this module after
    # removing overriding of self._write_metadata_file() in subclasses. All
    # metadata related code can be moved to a separate class and
    # BasePyTreeCheckpointHandler should delegate all metadata related code to
    # that class.
    if self._array_metadata_store is not None:
      param_infos = await self._get_param_infos_with_write_shape(
          param_infos, checkpoint_dir, self._array_metadata_store
      )

    async def metadata_coro():
      self._write_metadata_file(
          checkpoint_dir,
          param_infos=param_infos,
          save_args=save_args,
          custom_metadata=custom_metadata,
          use_zarr3=use_zarr3,
      ).result()

    metadata_write_future = future.CommitFutureAwaitingContractedSignals(
        metadata_coro(),
        name='write_metadata_file',
        operation_id=operation_id,
    )
    await asyncio.to_thread(metadata_write_future.result)

  def _read_metadata_file(
      self, directory: epath.Path
  ) -> tree_metadata.InternalTreeMetadata:
    """Reads metadata file and returns a tree of restore types.

    Args:
      directory: directory

    Returns:
      orbax.checkpoint.metadata.InternalTreeMetadata

    Raises:
      FileNotFoundError: if the metadata file is not found.
    """
    path = directory / PYTREE_METADATA_FILE
    if not path.exists():
      raise FileNotFoundError(
          f'Metadata file (named {PYTREE_METADATA_FILE}) does not exist at'
          f' {directory}.'
      )
    logging.vlog(
        1,
        'Reading pytree metadata file: %s with pytree_metadata_options: %s',
        path,
        self._pytree_metadata_options,
    )
    return tree_metadata.InternalTreeMetadata.from_json(
        json.loads(path.read_text()),
        pytree_metadata_options=self._pytree_metadata_options,
    )


  def metadata(self, directory: epath.Path) -> tree_metadata.TreeMetadata:
    """Returns tree metadata.

    The result will be a PyTree matching the structure of the saved checkpoint.
    Note that if the item saved was a custom class, the restored metadata will
    be returned as a nested dictionary representation.

    Example::

      {
        'layer0': {
            'w': ArrayMetadata(dtype=jnp.float32, shape=(8, 8), shards=(1, 2)),
            'b': ArrayMetadata(dtype=jnp.float32, shape=(8,), shards=(1,)),
        },
        'step': ScalarMetadata(dtype=jnp.int64),
      }

    If the required metadata file is not present, this method will raise an
    error.

    Args:
      directory: checkpoint location.

    Returns:
      tree containing metadata.
    """
    is_ocdbt_checkpoint = type_handlers.is_ocdbt_checkpoint(directory)
    internal_tree_metadata = self._read_metadata_file(directory)
    return tree_metadata.build_default_tree_metadata(
        internal_tree_metadata.as_custom_metadata(
            directory,
            self._type_handler_registry,
            use_ocdbt=is_ocdbt_checkpoint,
        ),
        custom_metadata=internal_tree_metadata.custom_metadata,
    )

  def finalize(self, directory: epath.Path) -> None:
    """Finalization step.

    Called automatically by the Checkpointer/AsyncCheckpointer just before the
    checkpoint is considered "finalized" in the sense of ensuring atomicity. See
    documentation for `type_handlers.merge_ocdbt_per_process_files`.

    Args:
      directory: Path where the checkpoint is located.
    """
    finalize_coros = []
    if self._array_metadata_store is not None:
      if self._primary_host is None:
        logging.log_first_n(
            logging.WARNING,
            '[process=%s] Skipped cross-host ArrayMetadata validation'
            ' because all hosts are primary (e.g. local storage).',
            1,  # log only once
            multihost.process_index(),
        )
      elif utils.is_primary_host(self._primary_host):
        finalize_coros.append(
            array_metadata_store_lib.validate_all_array_metadatas(
                self._array_metadata_validator,
                self._array_metadata_store,
                directory,
            )
        )

    async def merge_ocdbt_per_process_files():
      merge_start_time = time.time()
      ts_context = ts_utils.get_ts_context(use_ocdbt=True)
      await type_handlers.merge_ocdbt_per_process_files(
          directory,
          ts_context=ts_context,
          use_zarr3=self._use_zarr3,
          enable_validation=self._enable_post_merge_validation,
      )
      jax.monitoring.record_event_duration_secs(
          '/jax/checkpoint/write/async/ocdbt_merge_duration_secs',
          time.time() - merge_start_time,
      )

    finalize_coros.append(merge_ocdbt_per_process_files())

    async def _fn():
      await asyncio.gather(*finalize_coros)

    asyncio_utils.run_sync(_fn())


@register_with_handler(BasePyTreeCheckpointHandler, for_save=True)
@dataclasses.dataclass
class BasePyTreeSaveArgs(CheckpointArgs):
  """Parameters for saving a PyTree.

  Attributes:
    item (required): a PyTree to be saved.
    save_args: a PyTree with the same structure of `item`, which consists of
      `ocp.SaveArgs` objects as values. `None` can be used for values where no
      `SaveArgs` are specified.
    ocdbt_target_data_file_size: Specifies the target size (in bytes) of each
      OCDBT data file.  It only applies when OCDBT is enabled and Zarr3 must be
      turned on.  If left unspecified, default size is 2GB.  A value of 0
      indicates no maximum file size limit.  For best results, ensure
      chunk_byte_size is smaller than this value.  For more details, refer to
      https://google.github.io/tensorstore/kvstore/ocdbt/index.html#json-kvstore/ocdbt.target_data_file_size
    enable_pinned_host_transfer: If False, disables transfer to pinned host when
      copying from device to host, regardless of the presence of pinned host
      memory.
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.
  """

  item: PyTree
  save_args: Optional[PyTree] = None
  ocdbt_target_data_file_size: Optional[int] = None
  enable_pinned_host_transfer: bool = False
  custom_metadata: tree_types.JsonType | None = None


@register_with_handler(BasePyTreeCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class BasePyTreeRestoreArgs(CheckpointArgs):
  """Parameters for restoring a PyTree.

  Attributes (all optional):
    item: provides the tree structure for the restored item. If not provided,
      will infer the structure from the saved checkpoint. Transformations will
      not be run in this case. Necessary particularly in the case where the
      caller needs to restore the tree as a custom object.
    restore_args: optional object containing additional arguments for
      restoration. It should be a PyTree matching the structure of `item`, or
      if `item` is not provided, then it should match the structure of the
      checkpoint. Each value in the tree should be a `RestoreArgs` object (OR
      a subclass of `RestoreArgs`). Importantly, note that when restoring a
      leaf as a certain type, a specific subclass of `RestoreArgs` may be
      required. `RestoreArgs` also provides the option to customize the
      restore type of an individual leaf.
  """

  item: Optional[PyTree] = None
  restore_args: Optional[PyTree] = None
