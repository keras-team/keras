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

"""Utilities for checkpoint format detection."""

from typing import List
from etils import epath
from orbax.checkpoint._src.metadata import checkpoint as checkpoint_metadata

_OCDBT_MANIFEST_FILE = 'manifest.ocdbt'
PYTREE_METADATA_FILE = '_METADATA'
_CHECKPOINT_FILE = 'checkpoint'  # Old msgpack file.


def is_ocdbt_checkpoint(path: epath.PathLike) -> bool:
  """Determines whether a checkpoint uses OCDBT format."""
  path = epath.Path(path)
  return (path / _OCDBT_MANIFEST_FILE).exists()


def _has_msgpack_metadata_file(path: epath.Path) -> bool:
  return (path / _CHECKPOINT_FILE).exists()


def _has_pytree_metadata_file(path: epath.Path) -> bool:
  return (path / PYTREE_METADATA_FILE).exists() or _has_msgpack_metadata_file(
      path
  )


def _has_zarray_files(path: epath.Path) -> List[bool]:
  return [(p / '.zarray').exists() for p in path.iterdir() if p.is_dir()]


def _has_tensorstore_data_files(path: epath.Path) -> bool:
  return is_ocdbt_checkpoint(path) or any(_has_zarray_files(path))


def _is_pytree_checkpoint(path: epath.Path) -> bool:
  return (
      _has_pytree_metadata_file(path) and _has_tensorstore_data_files(path)
  ) or (
      _has_msgpack_metadata_file(path)
      and all([not x for x in _has_zarray_files(path)])
  )


def is_orbax_checkpoint(path: epath.PathLike) -> bool:
  """Returns whether the path is LIKELY an Orbax checkpoint.

  It is always possible to assemble files within a directory to look like an
  Orbax checkpoint, even if loading would later reject it as invalid. This
  function provides an easy heuristic as to what constitutes an Orbax
  checkpoint.

  Furthermore, Orbax provides a high degree of customizability, so many
  customized edge cases are possible that would not be detected by this
  function. However, most standard use cases should be covered.

  IMPORTANT: The function is unlikely to return a false positive (True when the
  checkpoint is NOT an Orbax checkpoint), but is more likely to return a
  false negative (False when the checkpoint IS an Orbax checkpoint).

  The function is expected to return True when the directory points to a step
  directory (saved by `CheckpointManager`) or an item directory (saved by
  `PyTreeCheckpointHandler`). It will return False when pointing to a root
  directory (containing multiple step directories).


  Args:
    path: A directory corresponding to an Orbax checkpoint.

  Returns:
    True if the path is LIKELY an Orbax checkpoint.

  Raises:
    FileNotFoundError: If the path does not exist.
    NotADirectoryError: If the path is not a directory.
  """
  path = epath.Path(path)
  if not path.exists():
    raise FileNotFoundError(f'Checkpoint path {path} does not exist.')
  if not path.is_dir():
    raise NotADirectoryError(f'Checkpoint path {path} is not a directory.')
  metadata_store = checkpoint_metadata.metadata_store(enable_write=False)
  # Path points to a single step checkpoint with valid metadata.
  if metadata_store.read(path) is not None:
    return True

  # Path points to valid PyTree checkpoint without newer metadata.
  if _is_pytree_checkpoint(path):
    return True

  # Path points to a root directory containing a valid PyTree checkpoint.
  if any([_is_pytree_checkpoint(p) for p in path.iterdir()]):
    return True

  return False
