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

"""Provides definitions for AggregateHandler and implementations."""

import abc
from concurrent import futures
import functools
from typing import Any, Optional

from etils import epath
from orbax.checkpoint import future as orbax_future
from orbax.checkpoint import msgpack_utils
from orbax.checkpoint import utils
from orbax.checkpoint._src.metadata import tree as tree_metadata

PyTree = Any


class AggregateHandler(abc.ABC):
  """Interface for reading and writing a PyTree using a specific format."""

  @abc.abstractmethod
  async def serialize(
      self, path: epath.Path, item: PyTree
  ) -> orbax_future.Future:
    """Serializes and writes `item` to a given `path`.

    The function is compatible with a multihost setting, but does not include
    extra logic to ensure atomicity.

    Args:
      path: the folder to which the item should be written.
      item: a PyTree.
    """
    pass

  @abc.abstractmethod
  def deserialize(self, path: epath.Path) -> PyTree:
    """Reads and deserializes a PyTree from the given directory."""
    pass

  @abc.abstractmethod
  def close(self):
    """Closes the handler."""
    pass


class MsgpackHandler(AggregateHandler):
  """An implementation of AggregateHandler that uses msgpack to store the tree."""

  def __init__(
      self,
      primary_host: Optional[int] = 0,
      pytree_metadata_options: tree_metadata.PyTreeMetadataOptions = (
          tree_metadata.PYTREE_METADATA_OPTIONS
      ),
  ):
    self._executor = futures.ThreadPoolExecutor(max_workers=1)
    self._primary_host = primary_host
    self._pytree_metadata_options = pytree_metadata_options

  async def serialize(
      self, path: epath.Path, item: PyTree
  ) -> orbax_future.Future:
    """See superclass documentation."""

    def _serialize_fn(x):
      if utils.is_primary_host(self._primary_host):
        if self._pytree_metadata_options.support_rich_types:
          raise NotImplementedError(
              'Orbax does not support rich typed metadata in legacy msgpack'
              ' checkpoint format. Please set'
              ' PyTreeMetadataOptions.support_rich_types to False.'
          )
        serializable_dict = tree_metadata.serialize_tree(
            x, self._pytree_metadata_options
        )
        msgpack = msgpack_utils.msgpack_serialize(serializable_dict)
        # Explicit "copy" phase is not needed because msgpack only contains
        # basic types and numpy arrays.
        return path.write_bytes(msgpack)
      return 0

    return self._executor.submit(functools.partial(_serialize_fn, item))

  def deserialize(self, path: epath.Path) -> PyTree:
    """See superclass documentation."""
    if path.exists():
      msgpack = path.read_bytes()
      return msgpack_utils.msgpack_restore(msgpack)
    else:
      raise FileNotFoundError(f'Checkpoint does not exist at {path}.')

  def close(self):
    """See superclass documentation."""
    self._executor.shutdown()
