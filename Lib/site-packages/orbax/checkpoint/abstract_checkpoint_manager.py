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

"""Abstract class to manage checkpoints: AbstractCheckpointManager."""

import abc
from typing import Any, Mapping, Optional, Protocol, Sequence, Union

from etils import epath
from orbax.checkpoint import args as args_lib
from orbax.checkpoint._src.metadata import checkpoint

PyTree = Any
SaveParams = Mapping[str, Any]
RestoreParams = SaveParams


class AbstractCheckpointManager(Protocol):
  """Interface to manage checkpoints.

  Allows a user to save and restore objects for which a Checkpointer
  implementation exists (e.g. PyTreeCheckpointer for PyTrees). The class
  keeps track of multiple checkpointable objects in the following structure::

    path/to/directory/    (top-level directory)
      0/    (step)
        params/    (first saveable)
          ...
        metadata/    (second saveable)
          ...
      1/    (step)
        ...
      2/    (step)
        ...
      ...
  """

  @property
  @abc.abstractmethod
  def directory(self) -> epath.Path:
    """Returns the top-level directory containing checkpoints for all items."""

  @abc.abstractmethod
  def all_steps(self, read: bool = False) -> Sequence[int]:
    """Returns all steps tracked by the manager.

    Args:
      read: If True, forces a read directly from the storage location.
        Otherwise, a cached result can be returned.

    Returns:
      A sequence of steps (integers)
    """

  @abc.abstractmethod
  def latest_step(self) -> Optional[int]:
    """Returns the latest step saved.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """

  @abc.abstractmethod
  def best_step(self) -> Optional[int]:
    """Returns the best step saved, as defined by `options.best_fn`.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """

  @abc.abstractmethod
  def reload(self):
    """Reloads internal properties.

    Resets internal cache of checkpoint steps, in case the directory managed
    by this object has been updated externally.
    """

  @abc.abstractmethod
  def reached_preemption(self, step: int) -> bool:
    """Returns True if a preemption sync point has been reached."""

  @abc.abstractmethod
  def should_save(self, step: int) -> bool:
    """Returns True if a checkpoint should be saved for the current step.

    This depends the previous step and save interval.

    Args:
      step: int

    Returns:
      True if the checkpoint should be saved.
    """

  @abc.abstractmethod
  def delete(self, step: int):
    """Deletes a step checkpoint."""

  @abc.abstractmethod
  def save(
      self,
      step: int,
      *args,
      **kwargs,
  ) -> bool:
    """Saves the given step."""

  @abc.abstractmethod
  def restore(
      self,
      step: Optional[int],
      *args,
      **kwargs,
  ) -> Union[Any, Mapping[str, Any], args_lib.Composite]:
    """Restores the given step."""

  @abc.abstractmethod
  def item_metadata(
      self, step: int
  ) -> Union[Any, Mapping[str, Any], args_lib.Composite]:
    """Returns metadata for all known items."""

  @abc.abstractmethod
  def metadata(
      self, step: int | None = None,
  ) -> checkpoint.StepMetadata | checkpoint.RootMetadata:
    """Returns `StepMetadata` for the specified step, or `RootMetadata` all.

    If step is specified, only return `StepMetadata` for that step.
    Otherwise, return `RootMetadata`.

    Args:
      step: Step for which to retrieve `StepMetadata`. If None, returns
        `RootMetadata`.

    Returns:
      Metadata for the specified step (`StepMetadata`), or all steps
      (`RootMetadata`).
    """

  @abc.abstractmethod
  def metrics(self, step: int) -> Optional[PyTree]:
    """Returns metrics for step, if present."""

  @abc.abstractmethod
  def wait_until_finished(self):
    """Blocks until any incomplete save operations are completed.

    Note that this method will typically be a no-op if all checkpointers are
    synchronous, since old checkpoints are already cleaned up immediately after
    completing `save`, and there is no background thread to wait for.

    If some checkpointers are of type AsyncCheckpointer, however, this method
    will wait until each of these checkpointers is finished.
    """

  @abc.abstractmethod
  def check_for_errors(self):
    """Checks for any outstanding errors in completed asynchronous save operations.

    Delegates to underlying Checkpointer.
    """

  @abc.abstractmethod
  def close(self):
    """Waits for outstanding operations to finish and closes Checkpointers."""
