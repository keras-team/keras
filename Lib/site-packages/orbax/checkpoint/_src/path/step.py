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

"""Orbax step storage entities."""

import abc
import concurrent
import dataclasses
import datetime
import functools
import os
import re
import time
from typing import Callable, Generic, Iterator, List, Optional, Protocol, Sequence, Set, TypeVar

from absl import logging
from etils import epath
import jax
import numpy as np
from orbax.checkpoint._src.metadata import checkpoint
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.multihost import multihost


_GCS_PATH_PREFIX = ('gs://',)
_COMMIT_SUCCESS_FILE = 'commit_success.txt'
TMP_DIR_SUFFIX = '.orbax-checkpoint-tmp-'
# prefix_1000.orbax-checkpoint-tmp-1010101
# OR
# 1000.orbax-checkpoint-tmp-1010101
TMP_DIR_STEP_PATTERN = r'.*?_*?(\d+)\.orbax-checkpoint-tmp-\d+'
_LAST_CHECKPOINT_WRITE_TIME = time.time()

# This file mode gives full permissions to OWNER, GROUP and OTHER.
WORLD_READABLE_MODE = 0o777

MetadataT = TypeVar('MetadataT', bound='Metadata')


@dataclasses.dataclass(frozen=True)
class Metadata:
  """Metadata of a step.

  Attributes:
    step: step number of the checkpoint.
    path: path to the checkpoint.
  """

  step: int
  path: epath.Path

  @functools.cached_property
  def _checkpoint_metadata(self) -> Optional[checkpoint.StepMetadata]:
    """Returns checkpoint metadata of this step."""
    metadata_dict = checkpoint.metadata_store(enable_write=False).read(
        file_path=checkpoint.step_metadata_file_path(self.path)
    )
    if metadata_dict is None:
      return None
    return step_metadata_serialization.deserialize(metadata_dict)

  @property
  def init_timestamp_nsecs(self) -> Optional[int]:
    """Returns init timestamp of uncommitted checkpoint of this step.

    It is specified as nano seconds since epoch.
    """
    metadata = self._checkpoint_metadata
    if metadata is None:
      return None
    return metadata.init_timestamp_nsecs

  @property
  def commit_timestamp_nsecs(self) -> Optional[int]:
    """Returns commit timestamp of the checkpoint of this step.

    It is specified as nano seconds since epoch.
    """
    metadata = self._checkpoint_metadata
    if metadata is None:
      return None
    return metadata.commit_timestamp_nsecs

  @property
  def commit_timestamp(self) -> datetime.datetime:
    """Returns commit timestamp of the checkpoint of this step.

    It is specified as datetime in UTC timezone.
    """
    commit_timestamp_nsecs = self.commit_timestamp_nsecs
    if commit_timestamp_nsecs is not None:
      timestamp_sec = commit_timestamp_nsecs / 1e9
    else:
      timestamp_sec = self.path.stat().mtime
    return datetime.datetime.fromtimestamp(
        timestamp_sec, tz=datetime.timezone.utc
    )


class NameFormat(Protocol, Generic[MetadataT]):
  """Protocol responsible for naming and querying steps."""

  def build_name(self, step: int) -> str:
    """Returns `step` name.

    *Implementation hint:* Implement it to build a name for the given step using
    the class's custom formatting attributes. Since it is mainly meant for
    building names to save checkpoints, it can raise error if this NameFormat is
    just meant for finding already existing step paths.

    Args:
      step: Step number.
    """
    ...

  @abc.abstractmethod
  def find_all(self, base_path: epath.PathLike) -> Iterator[MetadataT]:
    """Returns metadata of all steps.

    NOTE: Ignores uncommitted checkpoints.

    *Implementation hint:* Implement it to find all step folders under
    `base_path` performing IO operations if needed. Use
    `build_step_metadatas(...)` helper function to build all the `MetadataT`
    using the found step paths.

    Args:
      base_path: *root* Path under which Step folders are placed.
    """
    ...

  @abc.abstractmethod
  def find_step(self, base_path: epath.PathLike, step: int) -> MetadataT:
    """Returns the metadata for `step` or raises ValueError.

    NOTE: Ignores uncommitted checkpoints.

    *Implementation hint:* Implement it to find the step folder under
    `base_path` performing IO operations if needed.

    Args:
      base_path: *root* Path under which Step folders are placed.
      step: Step number.

    Raises:
      ValueError if no committed paths for the requested step is found.
    """
    ...


def build_step_path(
    base_path: epath.PathLike, name_format: NameFormat[Metadata], step: int
) -> epath.Path:
  """Returns `step` path under `base_path` for step `name_format`."""
  return epath.Path(base_path) / name_format.build_name(step)


def build_step_metadatas(
    step_paths: Iterator[epath.Path],
    build_metadata: Callable[[epath.Path], Optional[MetadataT]],
) -> Iterator[MetadataT]:
  """Yields filtered metadata mapped with `step_paths`.

  Args:
    step_paths: Iterator of step paths.
    build_metadata: Callable to match and build step metadata from `step_paths`
      elements. If a `step_paths` element doesn't match then it returns None.

  Yields:
    Step metadata.
  """
  with concurrent.futures.ThreadPoolExecutor() as executor:
    metadata_futures = [
        executor.submit(build_metadata, step_path) for step_path in step_paths
    ]
    for future in concurrent.futures.as_completed(metadata_futures):
      metadata = future.result()
      if metadata is not None:
        yield metadata


def step_prefix_with_underscore(step_prefix: Optional[str]) -> str:
  """Returns `step_prefix` appended with `underscore` or <empty> if None."""
  return '' if step_prefix is None else f'{step_prefix}_'


def latest_step_metadata(
    root_path: epath.PathLike, name_format: NameFormat[MetadataT]
) -> Optional[MetadataT]:
  """Returns step.MetadataT of the latest step in `root_path`."""
  return max(
      sorted(
          name_format.find_all(root_path),
          key=lambda metadata: metadata.path.name,
          reverse=True,
      ),
      default=None,
      key=lambda metadata: metadata.step,
  )


def step_metadata_of_checkpoint_path(
    checkpoint_path: epath.PathLike, name_format: NameFormat[MetadataT]
) -> MetadataT:
  """Returns step.MetadataT of given `checkpoint_path`."""
  checkpoint_path = epath.Path(checkpoint_path)
  all_step_metadata = list(name_format.find_all(checkpoint_path.parent))
  for step_metadata in all_step_metadata:
    if step_metadata.path.name == checkpoint_path.name:
      return step_metadata
  raise ValueError(
      'Failed to resolve step metadata of checkpoint path with'
      f' NameFormat={name_format}, checkpoint path={checkpoint_path}, path'
      f' name({checkpoint_path.name}) did not match with available step names:'
      f' {[step_metadata.path.name for step_metadata in all_step_metadata]}.'
      ' Please check if the given path is really a checkpoint path.'
  )


# TODO(b/337858698): Works with CompositeNameFormat.write_name_format only. Also
# support read_name_formats.
def find_step_path(
    base_path: epath.PathLike,
    name_format: NameFormat[Metadata],
    *,
    step: int,
    include_uncommitted: bool = False,
) -> epath.Path:
  """Returns `step` path under `base_path` for step `name_format`.

  NOTE: Experimental function, subject to change.

  Args:
    base_path: directory path containing step subdirs.
    name_format: NameFormat of the target `step`.
    step: target step number.
    include_uncommitted: if True then uncommitted steps are considered in search
      too, otherwise only committed steps are looked up.

  Raises:
    ValueError if the target step path does not exist.
  """
  base_path = epath.Path(base_path)
  if not include_uncommitted:
    return name_format.find_step(base_path, step).path

  # First try finding uncommitted step.
  if is_gcs_path(base_path):
    uncommitted_step_path = build_step_path(base_path, name_format, step)
  else:
    step_name = name_format.build_name(step)
    uncommitted_step_path = None
    for uncommitted_name in tmp_checkpoints(base_path):
      if uncommitted_name.startswith(f'{step_name}{TMP_DIR_SUFFIX}'):
        uncommitted_step_path = base_path / uncommitted_name
        break
  if uncommitted_step_path and uncommitted_step_path.exists():
    return uncommitted_step_path
  # Uncommitted step not found, return committed one or raise error.
  return name_format.find_step(base_path, step).path


def maybe_find_step_metadata(
    base_path: epath.PathLike,
    name_format: NameFormat[Metadata],
    *,
    step: int,
) -> Metadata | None:
  """Returns `Metadata` for `step` with `name_format` or None."""
  try:
    return name_format.find_step(base_path, step)
  except ValueError:
    return None


@dataclasses.dataclass(frozen=True)
class _StandardNameFormat(NameFormat[Metadata]):
  """NameFormat for 'standard' steps for common Orbax use cases.

  NOTE: Ignores uncommitted checkpoints.

  Naming examples:
   * step_prefix=None    step_format_fixed_length=None  ->  23
   * step_prefix=None    step_format_fixed_length=4     ->  0023
   * step_prefix=step    step_format_fixed_length=None  ->  step_23
   * step_prefix=step    step_format_fixed_length=4     ->  step_0023

  Attributes:
    step_prefix: Optional fixed string prefixed to step. Note an *underscore* is
      appended before applying it.
    step_format_fixed_length: Optional length of the zero padded step. e.g. 6
      for 000123.
  """

  step_prefix: Optional[str] = None
  step_format_fixed_length: Optional[int] = None

  def build_name(self, step: int) -> str:
    """Returns `(prefix_)?(zero padding)?step` name."""
    if self.step_format_fixed_length is not None:
      step_str = f'{step:0{self.step_format_fixed_length}d}'
    else:
      step_str = f'{step}'

    # [prefix]step
    return f'{step_prefix_with_underscore(self.step_prefix)}{step_str}'

  def _build_metadata(
      self, step_path: epath.Path, step: Optional[int] = None
  ) -> Optional[Metadata]:
    """Returns metadata for given `step_path` if it is valid or None."""
    if not _is_checkpoint_finalized_ignore_error(step_path):
      return None

    if step is not None:
      # step already known, just check exists.
      if step_path.exists():
        return Metadata(step=step, path=step_path)

    # Regex: [prefix]*(step)
    if self.step_format_fixed_length and self.step_format_fixed_length > 0:
      zero_present = rf'0\d{{{self.step_format_fixed_length-1}}}'
      zero_not_present = rf'[1-9]\d{{{self.step_format_fixed_length-1}}}\d*'
      zero_padded_step_group = rf'({zero_present}|{zero_not_present})'
    else:
      zero_padded_step_group = r'(0|[1-9]\d*)'
    name_regex = f'^{step_prefix_with_underscore(self.step_prefix)}{zero_padded_step_group}$'

    match = re.search(name_regex, step_path.name)
    if match is None:
      return None
    (step_,) = match.groups()
    step_ = int(step_)

    return Metadata(step=step_, path=step_path)

  def find_all(self, base_path: epath.PathLike) -> Iterator[Metadata]:
    """Returns metadata of all steps matching with name_format attributes."""
    # <step_prefix>_?<0 padding>?*
    step_paths = epath.Path(base_path).glob(
        f'{step_prefix_with_underscore(self.step_prefix)}*'
    )
    return build_step_metadatas(step_paths, self._build_metadata)

  def find_step(self, base_path: epath.PathLike, step: int) -> Metadata:
    """Returns the metadata for `step` or raises ValueError."""
    step_path = build_step_path(base_path, self, step)
    metadata = self._build_metadata(step_path, step=step)
    if metadata is not None:
      return metadata

    # Raise detailed error message.
    raise ValueError(
        f'No step path found with name={self.build_name(step)},'
        f' NameFormat={self} for step={step} under {base_path}.'
    )


def standard_name_format(
    step_prefix: Optional[str] = None,
    step_format_fixed_length: Optional[int] = None,
) -> NameFormat[Metadata]:
  """Returns NameFormat for 'standard' steps for common Orbax use cases.

  NOTE: Ignores uncommitted checkpoints.

  Naming examples:
   * step_prefix=None    step_format_fixed_length=None  ->  23
   * step_prefix=None    step_format_fixed_length=4     ->  0023
   * step_prefix=step    step_format_fixed_length=None  ->  step_23
   * step_prefix=step    step_format_fixed_length=4     ->  step_0023

  Args:
    step_prefix: Optional fixed string prefixed to step. Note an *underscore* is
      appended before applying it.
    step_format_fixed_length: Optional length of the zero padded step. e.g. 6
      for 000123.
  """
  return _StandardNameFormat(
      step_prefix=step_prefix,
      step_format_fixed_length=step_format_fixed_length,
  )


@dataclasses.dataclass(frozen=True)
class _CompositeNameFormat(NameFormat[Metadata]):
  """Supports reading multiple step namings, but just one format to write.

  Attributes:
    write_name_format: NameFormat used to build step names meant for writing
      checkpoints. Must be present in `read_name_formats` at a preferred
      priority position.
    read_name_formats: Sequence (ordered) of NameFormats used to find steps for
      reading checkpoints. It acts like an *or*, where the first one to match is
      returned.
  """

  write_name_format: NameFormat[Metadata]
  read_name_formats: Sequence[NameFormat[Metadata]]

  def __post_init__(self):
    if self.write_name_format not in self.read_name_formats:
      raise ValueError(
          f'write_name_format: {self.write_name_format} must be present in'
          f' read_name_formats: {self.read_name_formats}.'
      )

  def build_name(self, step: int) -> str:
    """Returns `step` name using `write_name_format`."""
    return self.write_name_format.build_name(step)

  def find_all(self, base_path: epath.PathLike) -> Iterator[Metadata]:
    """Returns metadata of all steps."""
    found_paths = set()
    for read_name_format in self.read_name_formats:
      for step_metadata in read_name_format.find_all(base_path):
        if step_metadata.path not in found_paths:
          found_paths.add(step_metadata.path)
          yield step_metadata

  def find_step(self, base_path: epath.PathLike, step: int) -> Metadata:
    """Returns the metadata for `step` or raises ValueError."""
    errors = []  # Used to raise the final collated error if needed.
    for read_name_format in self.read_name_formats:
      try:
        return read_name_format.find_step(base_path, step)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.info(
            'Failed to find step=%s with NameFormat=%s under %s. Error: %s',
            step,
            read_name_format,
            base_path,
            e,
        )
        errors.append(e)

    # Raise the concatenated errors.
    messages = [f'{e}' for e in errors]
    raise ValueError('\n'.join(messages))


def composite_name_format(
    write_name_format: NameFormat[Metadata],
    read_name_formats: Sequence[NameFormat[Metadata]],
) -> NameFormat[Metadata]:
  """Returns *composite* NameFormat supporting multiple read/single write formats.

  Args:
    write_name_format: NameFormat used to build step names meant for writing
      checkpoints. Must be present in `read_name_formats` at a preferred
      priority position.
    read_name_formats: Sequence (ordered) of NameFormats used to find steps for
      reading checkpoints. Please note that to resolve conflicts (and avoid
      raising errors) in case of multiple NameFormats matching a given step, the
      sequence should be provided in highest to lowest priority order:
      NameFormat appearing earlier in the sequence is preferred.
  """
  return _CompositeNameFormat(write_name_format, read_name_formats)


# TODO(b/337137764) Can't move it to path/utils due to cyclic dependency.
# Explore other options.
def is_gcs_path(path: epath.Path) -> bool:
  return path.as_posix().startswith(_GCS_PATH_PREFIX)


# TODO(b/337137764) Can't move it to path/utils due to cyclic dependency.
# Explore other options.
def get_save_directory(
    step: int,
    directory: epath.PathLike,
    name: Optional[str] = None,
    step_prefix: Optional[str] = None,
    override_directory: Optional[epath.PathLike] = None,
    step_format_fixed_length: Optional[int] = None,
    step_name_format: Optional[NameFormat[Metadata]] = None,
) -> epath.Path:
  """Returns the standardized path to a save directory for a single item.

  Args:
    step: Step number.
    directory: Top level checkpoint directory.
    name: Item name ('params', 'state', 'dataset', etc.).
    step_prefix: Prefix applied to `step` (e.g. 'checkpoint').
    override_directory: If provided, step, directory, and step_prefix are
      ignored.
    step_format_fixed_length: Uses a fixed number of digits with leading zeros
      to represent the step number. If None, there are no leading zeros.
    step_name_format: NameFormat used to define step name for step and under
      given root directory. If provided, `step_prefix` and
      `step_format_fixed_length` are ignored.

  Returns:
    A directory.
  """
  if directory is None:
    raise ValueError('Directory cannot be None.')
  directory = epath.Path(directory)
  if override_directory is not None:
    result = epath.Path(override_directory)
  else:
    step_name_format = step_name_format or standard_name_format(
        step_prefix=step_prefix,
        step_format_fixed_length=step_format_fixed_length,
    )
    result = build_step_path(directory, step_name_format, step)
  if name is not None:
    result /= name
  return result


def is_tmp_checkpoint(path: epath.PathLike) -> bool:
  """Determines whether a directory is a tmp checkpoint path."""
  path = epath.Path(path)
  if os.path.islink(path):
    logging.warning(
        'Path %s is a symbolic link, not treating it as a tmp checkpoint.', path
    )
    return False
  if not path.exists():
    raise ValueError(f'Path {path} does not exist.')
  if not path.is_dir():
    return False
  if is_gcs_path(path) and not (path / _COMMIT_SUCCESS_FILE).exists():
    return True
  if TMP_DIR_SUFFIX in path.name:
    return True
  return False


def is_checkpoint_finalized(path: epath.PathLike) -> bool:
  """Determines if the given path represents a finalized checkpoint.

  Path takes the form::

    path/to/my/dir/<name>.orbax-checkpoint-tmp-<timestamp>/  # not finalized
    path/to/my/dir/<name>/  # finalized

  Alternatively::

    gs://path/to/my/dir/<name>/  # finalized
      commit_success.txt
      ...
    gs://<path/to/my/dir/<name>/  # not finalized
      ...

  Args:
    path: Directory.

  Returns:
    True if the checkpoint is finalized.

  Raises:
    ValueError if the provided path is not a directory. Valid checkpoint paths
    must be a directory.
  """
  path = epath.Path(path)
  if not path.exists():
    raise ValueError(f'Path {path} does not exist.')
  if not path.is_dir():
    raise ValueError(f'Path {path} is not a directory. Not a valid checkpoint')
  if is_gcs_path(path) and not (path / _COMMIT_SUCCESS_FILE).exists():
    logging.warning(
        'This GCS path %s does not contain the %s file used to indicate a'
        ' successfully written GCS checkpoint. If the checkpoint was'
        ' originally saved with GCS, the checkpoint was not successfully'
        ' written. If the checkpoint was saved differently and copied, you'
        ' need to add %s to the checkpoint directory.',
        path,
        _COMMIT_SUCCESS_FILE,
        _COMMIT_SUCCESS_FILE,
    )

    return False
  if TMP_DIR_SUFFIX in path.name:
    return False
  return True


def _is_checkpoint_finalized_ignore_error(path: epath.PathLike) -> bool:
  try:
    return is_checkpoint_finalized(path)
  except ValueError:
    return False


def tmp_checkpoints(checkpoint_dir: epath.PathLike) -> List[str]:
  """Returns a list of tmp checkpoint dir names in `checkpoint_dir`."""
  checkpoint_dir = epath.Path(checkpoint_dir)
  return [s.name for s in checkpoint_dir.iterdir() if is_tmp_checkpoint(s)]


def cleanup_tmp_directories(
    directory: epath.PathLike,
    primary_host: Optional[int] = 0,
    active_processes: Optional[Set[int]] = None,
    barrier_sync_key_prefix: Optional[str] = None,
):
  """Cleanup steps in `directory` with tmp files, as these are not finalized."""
  directory = epath.Path(directory)
  if multihost.is_primary_host(primary_host):
    logging.info('Cleaning up existing temporary directories at %s.', directory)
    tmp_files = tmp_checkpoints(directory)
    for tmp_file in tmp_files:
      (directory / tmp_file).rmtree()
  multihost.sync_global_processes(
      multihost.unique_barrier_key(
          'cleanup_tmp_dirs',
          prefix=barrier_sync_key_prefix,
      ),
      timeout=multihost.DIRECTORY_DELETION_TIMEOUT,
      processes=active_processes,
  )


# TODO(b/337137764) Close to checkpoint finalization, but explore other options.
def record_saved_duration(checkpoint_start_time: float):
  """Record program duration that is accounted for by this checkpoint.

  For the very first checkpoint, this is the interval between program init and
  current checkpoint start time.

  Note that we use the checkpoint start time instead of end time. The saved
  duration should not include parallel training duration while the async
  checkpoint is being written in the background.

  Args:
    checkpoint_start_time: Start time of current checkpoint.
  """
  global _LAST_CHECKPOINT_WRITE_TIME
  # Note: for the very first checkpoint, this is the interval between program
  # init and the current checkpoint start time.
  duration_since_last_checkpoint = (
      checkpoint_start_time - _LAST_CHECKPOINT_WRITE_TIME
  )
  # TODO(hanyangtay): Remove version guard.
  if jax.version.__version_info__ > (0, 3, 25):
    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/write/duration_since_last_checkpoint_secs',
        duration_since_last_checkpoint,
    )
  _LAST_CHECKPOINT_WRITE_TIME = checkpoint_start_time


def _is_legacy_step_checkpoint(path: epath.Path) -> bool:
  """Determines if the path resembles an Orbax step directory.

  Note that this is not foolproof, and users should not add extra files to the
  checkpoint directory beyond what is done by CheckpointManager.

  Args:
    path: path to check.

  Returns:
    bool indicating whether the path resembles an Orbax step directory.
  """
  name = path.name
  # Path must be a directory and either a digit, or end in '_' + digit.
  return path.is_dir() and (name.isdigit() or name.split('_')[-1].isdigit())


def _is_legacy_finalized_step_checkpoint(path: epath.Path) -> bool:
  return _is_legacy_step_checkpoint(
      path
  ) and _is_checkpoint_finalized_ignore_error(path)


def step_from_checkpoint_name(name: str) -> int:
  """Returns the step from a checkpoint name. Also works for tmp checkpoints."""
  if name.isdigit():
    return int(name)
  elif name.split('_')[-1].isdigit():
    split = name.split('_')
    if len(split) == 2 and split[0]:
      return int(split[-1])
  elif tmp_match := re.match(TMP_DIR_STEP_PATTERN, name):
    return int(tmp_match.group(1))
  raise ValueError(f'Unrecognized name format: {name}.')


def checkpoint_steps_paths(
    checkpoint_dir: epath.PathLike,
) -> List[epath.Path]:
  """Returns a list of finalized checkpoint paths in the directory."""
  checkpoint_dir = epath.Path(checkpoint_dir)
  if not checkpoint_dir.exists():
    raise ValueError(f'Path {checkpoint_dir} does not exist.')

  with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {
        step_dir: executor.submit(
            _is_legacy_finalized_step_checkpoint, step_dir
        )
        for step_dir in checkpoint_dir.iterdir()
    }
    return [step_dir for step_dir, future in futures.items() if future.result()]


def checkpoint_steps(
    checkpoint_dir: epath.PathLike, single_host_load_and_broadcast: bool = False
) -> List[int]:
  """Returns a list of finalized checkpoint steps in the directory."""

  def _checkpoint_steps(path: epath.Path) -> List[int]:
    return [
        step_from_checkpoint_name(s.name) for s in checkpoint_steps_paths(path)
    ]

  checkpoint_dir = epath.Path(checkpoint_dir)
  if single_host_load_and_broadcast:
    max_steps = len(list(checkpoint_dir.iterdir()))
    # Read the step list only from host 0, and then broadcast the list.
    # This minimizes queries on non-leader processes.
    padded_step_list = np.array([-1] * max_steps)
    if multihost.process_index() == 0:
      steps = np.array(_checkpoint_steps(checkpoint_dir))
      assert len(steps) <= max_steps
      padded_step_list[0 : len(steps)] = steps
    padded_step_list = multihost.broadcast_one_to_all(padded_step_list)
    return [step for step in padded_step_list if step >= 0]
  return _checkpoint_steps(checkpoint_dir)


def any_checkpoint_step(checkpoint_dir: epath.PathLike) -> Optional[int]:
  """Returns any finalized checkpoint step in the directory or None.

  This avoids iterating over the entire directory.

  Args:
    checkpoint_dir: Checkpoint directory.

  Returns:
    Any finalized checkpoint step in the directory or None.
  """
  checkpoint_dir = epath.Path(checkpoint_dir)
  for s in checkpoint_dir.iterdir():
    if _is_legacy_finalized_step_checkpoint(s):
      return step_from_checkpoint_name(s.name)
  return None
