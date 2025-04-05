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

"""Snapshot represents operations on how to create, delete snapshots of a checkpoint."""

import abc
import dataclasses

from absl import logging
from etils import epath
from orbax.checkpoint._src.path import utils


SNAPSHOTTING_TIME = "snapshotting_time"


@dataclasses.dataclass
class Snapshot(abc.ABC):
  """Represents a snapshot of a checkpoint."""

  @classmethod
  def create_snapshot(cls, src: str, dst: str):
    """Creates a snapshot of the checkpoint."""
    pass

  @classmethod
  def release_snapshot(cls, path: str):
    """Deletes a snapshot of the checkpoint."""
    pass




@dataclasses.dataclass
class DefaultSnapshot(Snapshot):
  """Creates a copy of the checkpoint in the snapshot folder."""

  @classmethod
  def create_snapshot(cls, src: str, dst: str):
    """Creates a deep copy of the checkpoint."""
    if not epath.Path(dst).is_absolute():
      raise ValueError(
          f"Snapshot destination must be absolute, but was '{dst}'."
      )
    if not epath.Path(src).exists():
      raise ValueError(f"Snapshot source does not exist: {src}'.")
    t = utils.Timer()
    utils.recursively_copy_files(src, dst)
    logging.debug(
        "Snapshot copy: %fs",
        t.get_duration(),
    )
    return dst

  @classmethod
  def release_snapshot(cls, path: str) -> bool:
    """Deletes a snapshot of the checkpoint."""
    if not epath.Path(path).exists():
      logging.error("Snapshot does not exist: %s", path)
      return False
    try:
      epath.Path(path).rmtree()
    except OSError as e:
      logging.error(e)
      return False
    else:
      return True


def create_instance(dst: str, *args, **kwargs):
  return DefaultSnapshot(*args, **kwargs)
