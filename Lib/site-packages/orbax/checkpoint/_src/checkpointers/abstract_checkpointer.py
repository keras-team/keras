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

"""AbstractCheckpointer."""

import abc
from typing import Any, Optional
from absl import logging
from etils import epath
from orbax.checkpoint import version


class AbstractCheckpointer(abc.ABC):
  """An interface allowing atomic save and restore for a single object.

  Typically, an implementation of this class should rely on a CheckpointHandler
  object, which type-specific logic can be delegated to. In this way, the
  Checkpointer can be used for many different types, while itself only handling
  common logic related to atomicity, synchronization, or asynchronous thread
  management.
  """

  def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
    logging.log_first_n(logging.INFO, version.get_details(), n=1)
    return super().__new__(cls)

  @abc.abstractmethod
  def save(self, directory: epath.PathLike, *args, **kwargs):
    """Saves the given item to the provided directory.

    Args:
      directory: a path to which to save.
      *args: additional args to provide to the CheckpointHandler's save method.
      **kwargs: additional keyword args to provide to the CheckpointHandler's
        save method.
    """
    pass

  @abc.abstractmethod
  def restore(self, directory: epath.PathLike, *args, **kwargs) -> Any:
    """Restores from the provided directory.

    Delegates to underlying handler.

    Args:
      directory: a path to restore from.
      *args: additional args to provide to the CheckpointHandler's restore
        method.
      **kwargs: additional keyword args to provide to the CheckpointHandler's
        restore method.

    Returns:
      a restored object
    """
    pass

  def structure(self, directory: epath.PathLike) -> Optional[Any]:
    """DEPRECATED.

    The structure of the saved object at `directory`.

    Delegates to underlying handler.

    Args:
      directory: a path to a saved checkpoint.

    Returns:
      the object structure or None, if the underlying handler does not implement
      `structure`.
    """
    pass

  def metadata(self, directory: epath.PathLike) -> Optional[Any]:
    """Returns metadata about the saved item.

    Ideally, this is a cheap way to collect information about the checkpoint
    without requiring a full restoration.

    Args:
      directory: the directory where the checkpoint is located.

    Returns:
      item metadata
    """
    pass

  def close(self):
    """Closes the Checkpointer."""
    pass
