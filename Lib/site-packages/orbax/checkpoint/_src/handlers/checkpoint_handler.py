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

"""CheckpointHandler interface."""

import abc
from typing import Any, Optional
from etils import epath


class CheckpointHandler(abc.ABC):
  """An interface providing save/restore methods used on a savable item.

  Item may be a PyTree, Dataset, or any other supported object.

  NOTE: Users should avoid using CheckpointHandler independently. Use
  Checkpointer or CheckpointManager.
  """

  @abc.abstractmethod
  def save(self, directory: epath.Path, *args, **kwargs):
    """Saves the provided item synchronously.

    Args:
      directory: the directory to save to.
      *args: additional arguments for save.
      **kwargs: additional arguments for save.
    """
    pass

  @abc.abstractmethod
  def restore(self, directory: epath.Path, *args, **kwargs) -> Any:
    """Restores the provided item synchronously.

    Args:
      directory: the directory to restore from.
      *args: additional arguments for restore.
      **kwargs: additional arguments for restore.

    Returns:
      The restored item.
    """
    pass

  def metadata(self, directory: epath.Path) -> Optional[Any]:
    """Returns metadata about the saved item.

    Ideally, this is a cheap way to collect information about the checkpoint
    without requiring a full restoration.

    Args:
      directory: the directory where the checkpoint is located.

    Returns:
      item metadata
    """
    pass

  def finalize(self, directory: epath.Path) -> None:
    """Optional, custom checkpoint finalization callback.

    Will be called on a single worker after all workers have finished writing.

    Args:
      directory: the directory where the checkpoint is located.
    """
    pass

  def close(self):
    """Closes the CheckpointHandler."""
    pass

  @classmethod
  def typestr(cls) -> str:
    """A unique identifier for the CheckpointHandler type."""
    return f"{cls.__module__}.{cls.__qualname__}"
