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

"""Defines the types and Protocols for creating and finalizing temporary paths.

Chiefly defines the TemporaryPath Protocol for creating and finalizing temporary
paths, which allow other implementations.
"""

from __future__ import annotations

from typing import Optional, Protocol

from etils import epath
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.metadata import checkpoint as checkpoint_metadata



# TODO(b/326119183) Support configuration of temporary path detection
# (currently handled by `tmp_checkpoints` util methods).
class TemporaryPath(Protocol):
  """Class that represents a temporary path.

  Importantly, the temporary path always has a corresponding finalized path, and
  is primarily constructed from this path. The class contains logic to create
  the temporary path, and to finalize it into the final path.

  NOTE: All methods are intended to be called across all active processes,
  except for `finalize`, which is only called on the primary host.
  """

  @classmethod
  def from_final(
      cls,
      final_path: epath.Path,
      *,
      checkpoint_metadata_store: Optional[
          checkpoint_metadata.MetadataStore
      ] = None,
      file_options: Optional[options_lib.FileOptions] = None,
      multiprocessing_options: Optional[
          options_lib.MultiprocessingOptions
      ] = None,
  ) -> TemporaryPath:
    """Creates a TemporaryPath from a final path."""
    ...

  @classmethod
  def match(cls, temporary_path: epath.Path, final_path: epath.Path) -> bool:
    """Determines if `temporary_path` could correspond to `final_path`."""
    ...

  def get(self) -> epath.Path:
    """Constructs the temporary path without actually creating it."""
    ...

  def get_final(self) -> epath.Path:
    """Returns the final path without creating it."""
    ...

  async def create(
      self,
      *,
      file_options: options_lib.FileOptions = options_lib.FileOptions(),
  ) -> epath.Path:
    """Creates the temporary path on disk."""
    ...

  def finalize(
      self,
  ):
    """Finalizes the temporary path into the final path.

    NOTE: This method is only called on the primary host. This is in contrast
    with all other methods in this class, which are called across all active
    processes.

    This function is called from a background thread.

    """
    ...
