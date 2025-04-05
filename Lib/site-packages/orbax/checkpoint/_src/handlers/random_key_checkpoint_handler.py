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

"""RandomKeyCheckpointHandlers for saving and restoring individual Jax and Numpy random keys."""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, List, Mapping, Optional, Tuple, Union

from etils import epath
import jax
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.handlers import array_checkpoint_handler
from orbax.checkpoint._src.handlers import async_checkpoint_handler
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
from orbax.checkpoint._src.handlers import json_checkpoint_handler
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.serialization import type_handlers

NumpyRandomKeyType = Union[tuple, dict]

ArrayRestoreArgs = array_checkpoint_handler.ArrayRestoreArgs
ArraySaveArgs = array_checkpoint_handler.ArraySaveArgs
CheckpointArgs = checkpoint_args.CheckpointArgs
CompositeArgs = composite_checkpoint_handler.CompositeArgs
CompositeCheckpointHandler = (
    composite_checkpoint_handler.CompositeCheckpointHandler
)
JsonRestoreArgs = json_checkpoint_handler.JsonRestoreArgs
JsonSaveArgs = json_checkpoint_handler.JsonSaveArgs
PyTreeRestoreArgs = pytree_checkpoint_handler.PyTreeRestoreArgs
PyTreeSaveArgs = pytree_checkpoint_handler.PyTreeSaveArgs
register_with_handler = checkpoint_args.register_with_handler


class BaseRandomKeyCheckpointHandler(
    async_checkpoint_handler.AsyncCheckpointHandler
):
  """Base handle saving and restoring individual Jax random key in both typed and untyped format."""

  def __init__(self, key_name: str):
    """Initializes the handler.

    Args:
      key_name: Provides a name for the directory under which Tensorstore files
        will be saved. Defaults to 'random_key'.
    """
    self._key_name = key_name
    self._key_metadata = f'{self._key_name}_metadata'
    self._handler = CompositeCheckpointHandler()

  @abc.abstractmethod
  def checkpoint_save_args(
      self, args: CheckpointArgs
  ) -> Tuple[CheckpointArgs, JsonSaveArgs]:
    """Return the `args` for saving item and metadata.

    Args:
      args: An ocp.checkpoint_args.CheckpointArgs.

    Returns:
      Return a tuple of CheckpointArgs, and JsonSaveArgs.
      The CheckpointArgs is to save the actual random key and the JsonSaveArgs
      is to save any extra metadata
    """
    pass

  @abc.abstractmethod
  def checkpoint_restore_args(
      self,
      args: CheckpointArgs,
  ) -> CheckpointArgs:
    """Return the `args` for retoring the item.

    Args:
      args: An ocp.checkpoint_args.CheckpointArgs.

    Returns:
      Return a CheckpointArgs to restore the item
    """
    pass

  @abc.abstractmethod
  def post_restore(self, item: Any, metadata: Mapping[Any, Any]) -> Any:
    """Final operations needed to be done on the returning result."""
    pass

  async def async_save(
      self,
      directory: epath.Path,
      args: CheckpointArgs,
  ) -> Optional[List[future.Future]]:
    """Saves a random key asynchronously.

    Args:
      directory: Folder in which to save.
      args: An ocp.checkpoint_args.CheckpointArgs.

    Returns:
      A list of commit futures which can be run to complete the save.
    """
    item_arg, metadata_arg = self.checkpoint_save_args(args)
    return await self._handler.async_save(
        directory,
        CompositeArgs(**{
            self._key_name: item_arg,
            self._key_metadata: metadata_arg,
        }),
    )

  def save(self, directory: epath.Path, *args, **kwargs):
    """Saves a random key synchronously."""

    async def async_save():
      commit_futures = await self.async_save(directory, *args, **kwargs)  # pytype: disable=bad-return-type
      # Futures are already running, so sequential waiting is equivalent to
      # concurrent waiting.
      if commit_futures:  # May be None.
        for f in commit_futures:
          f.result()  # Block on result.

    asyncio_utils.run_sync(async_save())

  def restore(
      self,
      directory: epath.Path,
      args: CheckpointArgs,
  ) -> Any:
    """Restores a random key.

    Args:
      directory: folder from which to read.
      args: An ocp.checkpoint_args.CheckpointArgs.

    Returns:
      The restored object.
    """
    item_arg = self.checkpoint_restore_args(args)
    result = self._handler.restore(
        directory,
        args=CompositeArgs(**{
            self._key_name: item_arg,
            self._key_metadata: JsonRestoreArgs(),
        }),
    )

    return self.post_restore(result[self._key_name], result[self._key_metadata])

  def finalize(self, directory: epath.Path):
    self._handler.finalize(directory)

  def close(self):
    self._handler.close()


class JaxRandomKeyCheckpointHandler(BaseRandomKeyCheckpointHandler):
  """Handles saving and restoring individual Jax random key in both typed and untyped format."""

  def __init__(self, key_name: Optional[str] = None):
    """Initializes the handler.

    Args:
      key_name: Provides a name for the directory under which Tensorstore files
        will be saved. Defaults to 'jax_random_key'.
    """
    super().__init__(key_name or 'jax_random_key')

  def checkpoint_save_args(
      self, args: JaxRandomKeySaveArgs
  ) -> Tuple[CheckpointArgs, JsonSaveArgs]:
    item = args.item
    save_args = args.save_args

    if isinstance(item, jax.Array):
      if is_typed := jax.dtypes.issubdtype(item.dtype, jax.dtypes.prng_key):
        item = jax.random.key_data(item)
      metadata = {'typed': is_typed}
    else:
      raise TypeError(f'Unsupported type: {type(item)}.')

    return (
        ArraySaveArgs(jax.random.key_data(item), save_args),
        JsonSaveArgs(metadata),
    )

  def checkpoint_restore_args(
      self, args: JaxRandomKeyRestoreArgs
  ) -> CheckpointArgs:
    return ArrayRestoreArgs(restore_args=args.restore_args)

  def post_restore(self, item: Any, metadata: Mapping[Any, Any]) -> Any:
    if metadata['typed']:
      return jax.random.wrap_key_data(item)
    else:
      return item


@register_with_handler(JaxRandomKeyCheckpointHandler, for_save=True)
@dataclasses.dataclass
class JaxRandomKeySaveArgs(CheckpointArgs):
  """Parameters for saving a JAX random key.

  Attributes:
    item (required): a JAX random key.
    save_args: a `ocp.SaveArgs` object specifying save options.
  """

  item: jax.Array
  save_args: Optional[type_handlers.SaveArgs] = None


@register_with_handler(JaxRandomKeyCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class JaxRandomKeyRestoreArgs(CheckpointArgs):
  """Jax random key restore args.

  Attributes:
    restore_args: a `ocp.RestoreArgs` object specifying restore options for
      JaxArray.
  """

  restore_args: Optional[type_handlers.RestoreArgs] = None


class NumpyRandomKeyCheckpointHandler(BaseRandomKeyCheckpointHandler):
  """Saves Nnumpy random key in legacy or non-lagacy format."""

  def __init__(self, key_name: Optional[str] = None):
    """Initializes NumpyRandomKeyCheckpointHandler.

    Args:
      key_name: Provides a name for the directory under which Tensorstore files
        will be saved. Defaults to 'np_random_key'.
    """
    super().__init__(key_name or 'np_random_key')

  def checkpoint_save_args(
      self, args: NumpyRandomKeySaveArgs
  ) -> Tuple[CheckpointArgs, JsonSaveArgs]:
    item = args.item

    if isinstance(item, tuple):
      metadata = {'legacy': True}
    elif isinstance(item, dict):
      metadata = {'legacy': False}
    else:
      raise TypeError(f'Unsupported type: {type(item)}.')

    return (PyTreeSaveArgs(item), JsonSaveArgs(metadata))

  def checkpoint_restore_args(
      self, args: NumpyRandomKeyRestoreArgs
  ) -> CheckpointArgs:
    return PyTreeRestoreArgs()

  def post_restore(self, item: Any, metadata: Mapping[Any, Any]) -> Any:
    if metadata['legacy']:
      return tuple(item)
    else:
      return item


@register_with_handler(NumpyRandomKeyCheckpointHandler, for_save=True)
@dataclasses.dataclass
class NumpyRandomKeySaveArgs(CheckpointArgs):
  """Parameters for saving a Numpy random key.

  Attributes:
    item (required): a Numpy random key in legacy or nonlegacy format
  """

  item: NumpyRandomKeyType


@register_with_handler(NumpyRandomKeyCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class NumpyRandomKeyRestoreArgs(CheckpointArgs):
  """Numpy random key restore args."""

  pass
