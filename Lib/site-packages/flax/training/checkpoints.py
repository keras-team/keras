# Copyright 2024 The Flax Authors.
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

"""Checkpointing helper functions.

Handles saving and restoring optimizer checkpoints based on step-number or
other numerical metric in filename.  Cleans up older / worse-performing
checkpoint files.
"""

import functools
import os
import pathlib
import re
import time
import warnings
from concurrent.futures import thread
from typing import (
  Any,
)
from collections.abc import Callable, Iterable

import jax
import orbax.checkpoint as ocp
from absl import logging
from jax import monitoring, process_index
from jax import tree_util as jtu
from jax.experimental.array_serialization.serialization import (
  GlobalAsyncCheckpointManager,
  get_tensorstore_spec,
)
from jax.experimental.multihost_utils import sync_global_devices

from flax import config, core, errors, io, serialization, traverse_util
from flax.training import orbax_utils


_READ_CHECKPOINT_EVENT: str = '/jax/checkpoint/read/durations_sec'
_WRITE_CHECKPOINT_EVENT: str = '/jax/checkpoint/write/durations_sec'


# Single-group reg-exps for int or float numerical substrings.
# captures sign:
SIGNED_FLOAT_RE = re.compile(r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
# does not capture sign:
UNSIGNED_FLOAT_RE = re.compile(
  r'[-+]?((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)'
)
# Module name followed by number.
MODULE_NUM_RE = re.compile(r'(.*)_\d+$')
# Alternative schemes handled by `gfile`, e.g. on Google Cloud Storage (GCS).
SCHEME_RE = re.compile('^(?P<scheme>[a-z][a-z0-9.+-]+://)?(?P<path>.*)', re.I)

# Multiprocess arrays (GlobalDeviceArray, or JAX array with multiprocess
# sharding) is across processes and will be stored in directories with this
# postfix, separated from the non-distributed data (e.g. the larger pytree)
MP_ARRAY_POSTFIX = '_gda'
# Occurrences of multiprocess arrays in the target pytree will be
# replaced by this string placeholder.
MP_ARRAY_PH = '//GDAPlaceholder:'

# Add a copy-success file to a distributed array directory to indicate the
# array save is complete.
# We need this for GCS because GCS's directory move is not atomic.
COMMIT_SUCCESS_FILE = 'commit_success.txt'

# Orbax main checkpoint file name.
ORBAX_CKPT_FILENAME = 'checkpoint'
ORBAX_MANIFEST_OCDBT = 'manifest.ocdbt'
ORBAX_METADATA_FILENAME = '_METADATA'

PyTree = Any

# TODO(flax-dev): Remove this once flax is using the latest jax release
# containing jax.Array attribute.
MultiprocessArrayType = Any


def _is_multiprocess_array(value: Any) -> bool:
  """Use GlobalAsyncCheckpointManager to save the array if it's only partially available on this host."""
  if isinstance(value, jax.Array):
    return not value.is_fully_addressable
  return False


def _checkpoint_path(
  ckpt_dir: str, step: int | float | str, prefix: str = 'checkpoint_'
) -> str:
  return os.path.join(ckpt_dir, f'{prefix}{step}')


def _checkpoint_path_step(path: str) -> float | None:
  """Returns the step number of a checkpoint path."""
  for s in SIGNED_FLOAT_RE.split(path)[::-1]:
    if SIGNED_FLOAT_RE.match(s):
      return float(s)
  return None


def _allowempty_listdir(path: str):
  try:
    return io.listdir(path)
  except io.NotFoundError:
    return []


def _safe_remove(path: str):
  """Identify whether a path is a dir or list and choose the correct remove method."""
  if io.isdir(path):
    io.rmtree(path)
  else:
    io.remove(path)


def _is_orbax_checkpoint(path: str) -> bool:
  return (
      io.exists(os.path.join(path, ORBAX_CKPT_FILENAME))
      or io.exists(os.path.join(path, ORBAX_METADATA_FILENAME))
      or io.exists(os.path.join(path, ORBAX_MANIFEST_OCDBT))
  )


class AsyncManager:
  """A simple object to track async checkpointing.

  How to use: create an instance and pass to save_checkpoint() calls:
    am = AsyncManager()
    save_checkpoint(..., async_manager=am)
  """

  def __init__(self, max_workers: int = 1):
    self.executor = thread.ThreadPoolExecutor(max_workers=max_workers)
    self.save_future = None

  def wait_previous_save(self):
    """Block until the previous save finishes, to keep files' consistency."""
    if self.save_future and not self.save_future.done():
      logging.warning(
        'The previous async save_checkpoint has not finished yet. Waiting '
        'for it to complete before the next save.'
      )
      self.save_future.result()

  def save_async(self, task: Callable[[], Any]):
    """Run a task async.

    The future will be tracked as self.save_future.

    Args:
      task: The callable to be executed asynchronously.
    """
    self.wait_previous_save()
    self.save_future = self.executor.submit(task)  # type: ignore


def _split_mp_arrays(
  target: dict[str, Any]
) -> tuple[dict[str, Any], list[tuple[MultiprocessArrayType, str]]]:
  """Split out the multiprocess arrays from the target pytree to save."""
  # When target is a single leaf instead of a pytree dict.
  if not isinstance(target, (core.FrozenDict, dict)):
    if _is_multiprocess_array(target):
      return MP_ARRAY_PH, [(target, '')]
    return target, []
  # Traverse the target and handle distributed arrays.
  flattened = traverse_util.flatten_dict(target, keep_empty_nodes=True)
  mpa_targets = []
  for key, value in flattened.items():
    if _is_multiprocess_array(value):
      subpath = '/'.join(key)
      mpa_targets.append((value, subpath))
      flattened[key] = MP_ARRAY_PH + subpath
  target = traverse_util.unflatten_dict(flattened)
  return target, mpa_targets


def _make_mpa_dirs(
  mpa_targets: list[tuple[MultiprocessArrayType, str]], tmp_path: str
):
  # Temporary array path is not used in GCS.
  if tmp_path.startswith('gs://'):
    return
  mpa_tmp_path = tmp_path + MP_ARRAY_POSTFIX
  # Clean up the previous MPA dir, in case some leftover from last preemption
  # lingers.
  if io.exists(mpa_tmp_path):
    logging.info('Removing outdated MPA temporary files at %s', mpa_tmp_path)
    io.rmtree(mpa_tmp_path)
  _, mpa_subpaths = zip(*mpa_targets)
  for subpath in mpa_subpaths:
    io.makedirs(os.path.join(mpa_tmp_path, subpath))


def _save_mpas(
  gda_manager,
  mpa_targets: list[tuple[MultiprocessArrayType, str]],
  tmp_path: str,
  final_path: str,
  base_path: str,
  keep: int,
  overwrite: bool,
  keep_every_n_steps: int | None,
  ckpt_start_time: float,
  async_manager: AsyncManager | None = None,
):
  """Save the multiprocess arrays given the paths."""
  mpa_list, mpa_subpaths = zip(*mpa_targets)
  mpa_tmp_path, mpa_final_path = (
    tmp_path + MP_ARRAY_POSTFIX,
    final_path + MP_ARRAY_POSTFIX,
  )
  write_commit_success = False
  # If the checkpoint directory is a GCS directory, then keep the final
  # checkpoint directory as the temporary checkpoint directory. This is because
  # renames are not atomic on GCS. When restoring check for the existence of a
  # success file.
  # TODO: figure out a way to unit-test the behavior.
  if tmp_path.startswith('gs://'):
    mpa_tmp_path = mpa_final_path
    write_commit_success = True
  mpa_paths = [os.path.join(mpa_tmp_path, x) for x in mpa_subpaths]
  ts_specs = [get_tensorstore_spec(x) for x in mpa_paths]
  gda_manager.serialize(
    list(mpa_list),
    ts_specs,
    on_commit_callback=functools.partial(
      _save_commit,
      tmp_path,
      final_path,
      base_path,
      keep,
      overwrite,
      keep_every_n_steps,
      ckpt_start_time,
      has_mpa=True,
      write_commit_success=write_commit_success,
      async_manager=async_manager,
    ),
  )


def _restore_mpas(
  state_dict,
  target: Any | None,
  ckpt_path: str,
  step: int | float | None,
  gda_manager: GlobalAsyncCheckpointManager | None,
  allow_partial: bool = False,
):
  """Restore the multiprocess arrays given the target structure and type."""

  def _check_mpa_errors():
    if not gda_manager:
      raise errors.MPACheckpointingRequiredError(ckpt_path, step)
    if not target and not allow_partial:
      raise errors.MPARestoreTargetRequiredError(ckpt_path, step)

  def _safe_deserialize(
    target_mpas: list[tuple[tuple[Any, ...], MultiprocessArrayType, str]],
    gda_manager: Any,
  ) -> list[MultiprocessArrayType]:
    gda_manager.wait_until_finished()

    # Check if reading from GCS and the array dir is potentially corrupted.
    if ckpt_path.startswith('gs://') and not io.exists(
      os.path.join(ckpt_path + MP_ARRAY_POSTFIX, COMMIT_SUCCESS_FILE)
    ):
      raise errors.MPARestoreDataCorruptedError(step, ckpt_path)

    # Check if the given target array types are valid.
    shardings = []
    for _, arr, path in target_mpas:
      if isinstance(arr, jax.Array):
        shardings.append(arr.sharding)

    # Restore the arrays.
    ts_specs = [get_tensorstore_spec(path) for _, _, path in target_mpas]
    return gda_manager.deserialize(shardings, ts_specs)

  # When target is a single leaf instead of a pytree dict.
  if not isinstance(state_dict, (core.FrozenDict, dict)):
    if (
      _is_multiprocess_array(target)
      and isinstance(state_dict, str)
      and state_dict.startswith(MP_ARRAY_PH)
    ):
      _check_mpa_errors()
      return _safe_deserialize(
        [((), target, ckpt_path + MP_ARRAY_POSTFIX)], gda_manager
      )[0]
    return state_dict

  # Go through the restored checkpoint pytree for all MPAs
  flattened = traverse_util.flatten_dict(state_dict, keep_empty_nodes=True)
  target_flattened = {}
  if target:
    target_flattened = traverse_util.flatten_dict(
      serialization.to_state_dict(target), keep_empty_nodes=True
    )
  # A list of (state_dict_key, target_array, array_file_path) for every array
  # to be restored
  target_mpas = []
  for key, value in flattened.items():
    if isinstance(value, str) and value.startswith(MP_ARRAY_PH):
      _check_mpa_errors()
      if (
        not target
        or (key not in target_flattened)
        or (not _is_multiprocess_array(target_flattened[key]))
      ):
        if allow_partial:
          logging.warning(
            'Multiprocess array %s could not be restored because a valid'
            ' array is not found in target at the corresponding location.'
            ' Proceed to restore other arrays because'
            ' allow_partial_restoration=True',
            key,
          )
        else:
          raise errors.MPARestoreTargetRequiredError(ckpt_path, step, key)
      else:
        mpa_path = os.path.join(
          ckpt_path + MP_ARRAY_POSTFIX, value[len(MP_ARRAY_PH) :]
        )
        target_mpas.append((key, target_flattened[key], mpa_path))

  # If any MPA needs to be restored, call deserialize
  if target_mpas:
    mpa_list = _safe_deserialize(target_mpas, gda_manager)
    for mpa, (key, _, _) in zip(mpa_list, target_mpas):
      flattened[key] = mpa
    state_dict = traverse_util.unflatten_dict(flattened)
  return state_dict


def natural_sort(file_list: Iterable[str], signed: bool = True) -> list[str]:
  """Natural sort for filenames with numerical substrings.

  Args:
    file_list: list of paths to sort containing numerical substrings.
    signed: bool: if leading '-' (or '+') signs should be included in numerical
      substrings as a sign or treated as a separator.

  Returns:
    List of filenames sorted 'naturally', not lexicographically: any
    integer substrings are used to subsort numerically. e.g.
    file_1, file_10, file_2  -->  file_1, file_2, file_10
    file_0.1, file_-0.2, file_2.0  -->  file_-0.2, file_0.1, file_2.0
  """
  float_re = SIGNED_FLOAT_RE if signed else UNSIGNED_FLOAT_RE

  def maybe_num(s):
    if float_re.match(s):
      return float(s)
    else:
      return s

  def split_keys(s):
    return [maybe_num(c) for c in float_re.split(s)]

  return sorted(file_list, key=split_keys)


def safe_normpath(path: str) -> str:
  """Normalizes path safely to get around `io.glob()` limitations."""
  match = SCHEME_RE.match(path)
  assert match is not None
  d = match.groupdict()
  return (d['scheme'] or '') + os.path.normpath(d['path'])


def _remove_invalid_ckpts(
  ckpt_path: str,
  base_path: str,
  keep: int,
  overwrite: bool,
  keep_every_n_steps: int | None,
  has_mpa: bool,
) -> None:
  """Clean up the checkpoint space according to `overwrite`, `keep`, and `keep_every_n_steps` parameters."""
  dir_path, prefix = os.path.split(base_path)
  checkpoint_files: list[Any] = [
    pathlib.PurePath(c) for c in _allowempty_listdir(dir_path)
  ]
  checkpoint_files = [
    os.path.join(dir_path, c)
    for c in checkpoint_files
    if c.match(f'{prefix}*') and not c.match(f'*{MP_ARRAY_POSTFIX}')
  ]
  checkpoint_files = natural_sort(checkpoint_files)

  # Remove newer checkpoints
  if overwrite and ckpt_path in checkpoint_files:
    ind = checkpoint_files.index(ckpt_path) + 1
    newer_ckpts = checkpoint_files[ind:]
    checkpoint_files = checkpoint_files[:ind]
    for path in newer_ckpts:
      logging.info('Removing checkpoint at %s', path)
      if has_mpa:
        # MPA might be removed already but the main ckpt is still there. This
        # can happen if the job is previously preempted after deleting the MPA
        # checkpoint folder and before deleting the main checkpoint.
        if io.exists(path + MP_ARRAY_POSTFIX):
          io.rmtree(path + MP_ARRAY_POSTFIX)
      _safe_remove(path)

  # Remove old checkpoint files.
  last_kept = -float('inf')
  if len(checkpoint_files) > keep:
    old_ckpts = checkpoint_files[:-keep]
    # Note: old_ckpts is sorted from oldest to newest.
    for path in old_ckpts:
      if keep_every_n_steps:
        step_number = _checkpoint_path_step(path)
        if step_number and (step_number - last_kept) >= keep_every_n_steps:
          logging.debug(
            'Not deleting %s, because last_kept=%f and keeping '
            'every %d steps.',
            path,
            last_kept,
            keep_every_n_steps,
          )
          last_kept = step_number
          continue
      logging.info('Removing checkpoint at %s', path)
      if has_mpa:
        # MPA might be removed already but the main ckpt is still there.
        if io.exists(path + MP_ARRAY_POSTFIX):
          io.rmtree(path + MP_ARRAY_POSTFIX)
      _safe_remove(path)


def _save_commit(
  ckpt_tmp_path: str,
  ckpt_path: str,
  base_path: str,
  keep: int,
  overwrite: bool,
  keep_every_n_steps: int | None,
  ckpt_start_time: float,
  has_mpa: bool,
  write_commit_success: bool,
  async_manager: AsyncManager | None = None,
) -> None:
  """Commit changes after saving checkpoints to disk.

  This function does the following, sequentially:
    1. Make sure all ckpt writing finishes, and rename them from temp path to
    the final path.
    2. Remove newer checkpoints (files that ordered larger than this save) if
    `overwrite=True`.
    3. Remove old checkpoint files based on `keep` and `keep_every_n_steps`.
    4. Record program duration saved by this checkpoint.
  """
  mpa_ckpt_tmp_path, mpa_ckpt_path = (
    ckpt_tmp_path + MP_ARRAY_POSTFIX,
    ckpt_path + MP_ARRAY_POSTFIX,
  )
  # Rename the multiprocess array path once serialization and writing finished.
  if has_mpa:
    if write_commit_success:
      commit_success_path = os.path.join(mpa_ckpt_path, COMMIT_SUCCESS_FILE)
      with io.GFile(commit_success_path, 'w') as f:
        f.write(f'Checkpoint commit was successful to {mpa_ckpt_path}')
    else:
      # Commits are a two stage process (renaming the array folder and renaming
      # the main ckpt file in sequential order). We always try to overwrite
      # here because the array ckpt might be already renamed in a previously
      # interrupted commit. NOTE: io.rename does not support overwriting
      # directories via `rename` so we manually overwrite it.
      if io.exists(mpa_ckpt_path):
        logging.info('Removing outdated checkpoint at %s', mpa_ckpt_path)
        io.rmtree(mpa_ckpt_path)
      io.rename(mpa_ckpt_tmp_path, mpa_ckpt_path)
  # Commit the main checkpoint file after arrays (if any) are committed
  if async_manager:
    async_manager.wait_previous_save()
  io.rename(ckpt_tmp_path, ckpt_path, overwrite=overwrite)
  logging.info('Saved checkpoint at %s', ckpt_path)

  # Remove newer and older invalid checkpoints.
  _remove_invalid_ckpts(
    ckpt_path, base_path, keep, overwrite, keep_every_n_steps, has_mpa
  )
  # Record checkpoint-related metrics.
  ocp.utils.record_saved_duration(ckpt_start_time)
  if async_manager:
    jax.monitoring.record_event_duration_secs(
      '/jax/checkpoint/write/async/total_duration_secs',
      time.time() - ckpt_start_time,
    )


def _check_overwrite_error(
  ckpt_tmp_path: str, ckpt_path: str, base_path: str, step: int
):
  """Throw error if a ckpt file of this step or higher already exists."""
  dir_path, prefix = os.path.split(base_path)
  checkpoint_files: list[Any] = [
    pathlib.PurePath(c) for c in _allowempty_listdir(dir_path)
  ]
  checkpoint_files = [
    os.path.join(dir_path, c)
    for c in checkpoint_files
    if c.match(f'{prefix}*') and not c.match(f'*{MP_ARRAY_POSTFIX}')
  ]
  if ckpt_path in checkpoint_files:
    raise errors.InvalidCheckpointError(ckpt_path, step)
  checkpoint_files.append(ckpt_path)

  checkpoint_files = natural_sort(checkpoint_files)
  # Handle the case if the job was preempted after the temporary checkpoint
  # was written, but before it was renamed to the final checkpoint name
  if checkpoint_files[-1] == ckpt_tmp_path:
    checkpoint_files.pop()
  if ckpt_path != checkpoint_files[-1]:
    raise errors.InvalidCheckpointError(ckpt_path, step)


def _save_main_ckpt_file(
  target: bytes,
  has_mpa: bool,
  paths: tuple[str, str],
  base_path: str,
  step: int,
  keep: int,
  overwrite: bool,
  keep_every_n_steps: int | None,
  ckpt_start_time: float,
):
  """Save the main checkpoint file via file system."""
  ckpt_tmp_path, ckpt_path = paths
  io.makedirs(os.path.dirname(ckpt_path))

  with io.GFile(ckpt_tmp_path, 'wb') as fp:
    fp.write(target)

  # Postpone the commitment of checkpoint to after MPA writes are done.
  if not has_mpa:
    _save_commit(
      ckpt_tmp_path,
      ckpt_path,
      base_path,
      keep,
      overwrite,
      keep_every_n_steps,
      ckpt_start_time,
      has_mpa=False,
      write_commit_success=False,
    )


def _get_checkpoint_paths(
  ckpt_dir: str | os.PathLike,
  step: int | float,
  prefix: str = 'checkpoint_',
) -> tuple[str, str, str]:
  """Generate the checkpoint paths used in this save operation."""
  ckpt_dir = os.fspath(ckpt_dir)  # Pathlib -> str
  logging.info('Saving checkpoint at step: %s', step)
  # normalize path because io.glob() can modify path './', '//' ...
  ckpt_dir = safe_normpath(ckpt_dir)
  ckpt_tmp_path = _checkpoint_path(ckpt_dir, 'tmp', prefix)
  ckpt_path = _checkpoint_path(ckpt_dir, step, prefix)
  base_path = os.path.join(ckpt_dir, prefix)
  return ckpt_path, ckpt_tmp_path, base_path


def save_checkpoint(
  ckpt_dir: str | os.PathLike,
  target: PyTree,
  step: int | float,
  prefix: str = 'checkpoint_',
  keep: int = 1,
  overwrite: bool = False,
  keep_every_n_steps: int | None = None,
  async_manager: AsyncManager | None = None,
  orbax_checkpointer: ocp.Checkpointer | None = None,
) -> str:
  """Save a checkpoint of the model. Suitable for single-host.

  In this method, every JAX process saves the checkpoint on its own. Do not
  use it if you have multiple processes and you intend for them to save data
  to a common directory (e.g., a GCloud bucket). To save multi-process
  checkpoints to a shared storage or to save ``GlobalDeviceArray``s, use
  ``save_checkpoint_multiprocess()`` instead.

  Pre-emption safe by writing to temporary before a final rename and cleanup
  of past files. However, if async_manager is used, the final
  commit will happen inside an async callback, which can be explicitly waited
  by calling ``async_manager.wait_previous_save()``.

  Example usage::

    >>> from flax.training import checkpoints
    >>> import jax.numpy as jnp
    >>> import tempfile

    >>> with tempfile.TemporaryDirectory() as dir_path:
    ...   test_object = {
    ...     'a': jnp.array([1, 2, 3], jnp.int32),
    ...     'b': jnp.array([1, 1, 1], jnp.int32),
    ...   }
    ...   file_path = checkpoints.save_checkpoint(
    ...     dir_path, target=test_object, step=0, prefix='test_', keep=1
    ...   )
    ...   restored_object = checkpoints.restore_checkpoint(
    ...     file_path, target=None
    ...   )
    >>> restored_object
    {'a': Array([1, 2, 3], dtype=int32), 'b': Array([1, 1, 1], dtype=int32)}

  Args:
    ckpt_dir: str or pathlib-like path to store checkpoint files in.
    target: serializable flax object, usually a flax optimizer.
    step: int or float: training step number or other metric number.
    prefix: str: checkpoint file name prefix.
    keep: number of past checkpoint files to keep.
    overwrite: overwrite existing checkpoint files if a checkpoint at the
      current or a later step already exits (default: False).
    keep_every_n_steps: if defined, keep every checkpoints every n steps (in
      addition to keeping the last 'keep' checkpoints).
    async_manager: if defined, the save will run without blocking the main
      thread. Only works for single host. Note that an ongoing save will still
      block subsequent saves, to make sure overwrite/keep logic works correctly.
    orbax_checkpointer: if defined, the save will be done by ocp. In the future,
      all Flax checkpointing features will be migrated to Orbax, and starting to
      use an ``orbax_checkpointer`` is recommended. Please check out the
      checkpointing guide
      (https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html#save-checkpoints)
      for how to use Orbax checkpointers.

  Returns:
    Filename of saved checkpoint.
  """
  jax.monitoring.record_event('/jax/flax/checkpoint/save')
  start_time = time.time()
  # Make sure all saves are finished before the logic of checking and removing
  # outdated checkpoints happens.
  if async_manager:
    async_manager.wait_previous_save()

  ckpt_path, ckpt_tmp_path, base_path = _get_checkpoint_paths(
    ckpt_dir, step, prefix
  )

  if config.flax_use_orbax_checkpointing or orbax_checkpointer:
    logging.info(
      'Using Orbax as backend to save Flax checkpoints. For potential'
      ' troubleshooting see:'
      ' https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html#orbax-as-backend-troubleshooting'
    )
    if jax.process_count() > 1:
      logging.warning(
        'Multiple JAX processes detected when calling single-process'
        ' `save_checkpoint`. Your devices will HANG if this function is only'
        ' called on process 0! Troubleshoot at:'
        ' https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html#if-your-devices-hang-when-writing-checkpoints'
      )

    # Make sure any previous work is done before making file changes.
    if orbax_checkpointer and isinstance(
      orbax_checkpointer, ocp.AsyncCheckpointer
    ):
      orbax_checkpointer.wait_until_finished()
    # If no checkpointer provided, save synchronously with default setting.
    if not orbax_checkpointer:
      orbax_checkpointer = ocp.Checkpointer(
        ocp.PyTreeCheckpointHandler()
      )
    # Check singular target.
    if jtu.treedef_is_leaf(jtu.tree_structure(target)) and not isinstance(
      orbax_checkpointer._handler,
      ocp.ArrayCheckpointHandler,  # pylint: disable=protected-access
    ):
      raise ValueError(
        'Orbax backend only accept pytree as save target. To save singular'
        ' objects like numbers or Numpy arrays, checkout'
        ' https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html#if-you-don-t-save-pytrees'
      )

    orbax_checkpointer.save(
      ckpt_path, target, force=overwrite
    )
    # Do a process check here in case people call this for multihost.
    if process_index() == 0:
      _remove_invalid_ckpts(
        ckpt_path, base_path, keep, overwrite, keep_every_n_steps, True
      )
    end_time = time.time()
    monitoring.record_event_duration_secs(
      _WRITE_CHECKPOINT_EVENT, end_time - start_time
    )
    return ckpt_path

  warnings.warn(
    (
      'Flax Checkpointing will soon be deprecated in favor of Orbax'
      ' (https://github.com/google/orbax). Please refer to the Checkpoint'
      ' Upgrade Guide'
      ' (https://flax.readthedocs.io/en/latest/guides/converting_and_upgrading/orbax_upgrade_guide.html)'
      ' to self-migrate your code to ocp.'
    ),
    DeprecationWarning,
  )
  if not overwrite:
    _check_overwrite_error(ckpt_tmp_path, ckpt_path, base_path, step)  # type: ignore

  target = serialization.to_bytes(target)

  # Save the files via I/O sync or async.
  def save_main_ckpt_task():
    jax.monitoring.record_event('/jax/flax/checkpoint/save_main_ckpt_task')
    return _save_main_ckpt_file(
      target,
      False,
      (ckpt_tmp_path, ckpt_path),
      base_path,
      step,
      keep,
      overwrite,
      keep_every_n_steps,
      start_time,
    )

  if async_manager:
    async_manager.save_async(save_main_ckpt_task)
  else:
    save_main_ckpt_task()
  end_time = time.time()
  monitoring.record_event_duration_secs(
    _WRITE_CHECKPOINT_EVENT, end_time - start_time
  )
  return ckpt_path


def save_checkpoint_multiprocess(
  ckpt_dir: str | os.PathLike,
  target: PyTree,
  step: int | float,
  prefix: str = 'checkpoint_',
  keep: int = 1,
  overwrite: bool = False,
  keep_every_n_steps: int | None = None,
  async_manager: AsyncManager | None = None,
  gda_manager: GlobalAsyncCheckpointManager | None = None,
  orbax_checkpointer: ocp.Checkpointer | None = None,
) -> str:
  """Save a checkpoint of the model in multi-process environment.

  Use this method to save ``GlobalDeviceArray``s, or to save data to a
  common directory. Only process 0 will save the main checkpoint file and
  remove old checkpoint files.

  Pre-emption safe by writing to temporary before a final rename and cleanup
  of past files. However, if async_manager or gda_manager is used, the final
  commit will happen inside an async callback, which can be explicitly waited
  by calling ``async_manager.wait_previous_save()`` or
  ``gda_manager.wait_until_finished()``.

  Args:
    ckpt_dir: str or pathlib-like path to store checkpoint files in.
    target: serializable flax object, usually a flax optimizer.
    step: int or float: training step number or other metric number.
    prefix: str: checkpoint file name prefix.
    keep: number of past checkpoint files to keep.
    overwrite: overwrite existing checkpoint files if a checkpoint at the
      current or a later step already exits (default: False).
    keep_every_n_steps: if defined, keep every checkpoints every n steps (in
      addition to keeping the last 'keep' checkpoints).
    async_manager: if defined, the save will run without blocking the main
      thread. Only works for single host. Note that an ongoing save will still
      block subsequent saves, to make sure overwrite/keep logic works correctly.
    gda_manager: required if target contains a JAX GlobalDeviceArray. Will save
      the GDAs to a separate subdirectory with postfix "_gda" asynchronously.
      Same as async_manager, this will block subsequent saves.
    orbax_checkpointer: if defined, the save will be done by Orbax In the
      future, all Flax checkpointing features will be migrated to Orbax, and
      starting to use an ``orbax_checkpointer`` is recommended. Please check out
      the checkpointing guide
      (https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html#save-checkpoints)
      for how to use Orbax checkpointers.

  Returns:
    Filename of saved checkpoint.
  """
  jax.monitoring.record_event('/jax/flax/checkpoint/save')
  start_time = time.time()
  # Make sure all saves are finished before the logic of checking and removing
  # outdated checkpoints happens.
  sync_global_devices('Flax:Checkpoint:StartSave')
  if async_manager:
    async_manager.wait_previous_save()
  if gda_manager:
    gda_manager.wait_until_finished()
    sync_global_devices('Flax:Checkpoint:WaitLastSaveDone')

  ckpt_path, ckpt_tmp_path, base_path = _get_checkpoint_paths(
    ckpt_dir, step, prefix
  )

  if config.flax_use_orbax_checkpointing or orbax_checkpointer:
    logging.info(
      'Using Orbax as backend to save Flax checkpoints. For potential'
      ' troubleshooting see:'
      ' https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html#orbax-as-backend-troubleshooting'
    )
    # Make sure any previous work is done before making file changes.
    if orbax_checkpointer and isinstance(
      orbax_checkpointer, ocp.AsyncCheckpointer
    ):
      orbax_checkpointer.wait_until_finished()

    # If no checkpointer provided, save synchronously with default setting.
    if not orbax_checkpointer:
      orbax_checkpointer = ocp.Checkpointer(
        ocp.PyTreeCheckpointHandler()
      )
    # Check singular target.
    if jtu.treedef_is_leaf(jtu.tree_structure(target)) and not isinstance(
      orbax_checkpointer._handler,
      ocp.ArrayCheckpointHandler,  # pylint: disable=protected-access
    ):
      raise ValueError(
        'Orbax backend only accept pytree as save target. To save singular'
        ' objects like numbers or Numpy arrays, checkout'
        ' https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html#if-you-don-t-save-pytrees'
      )

    if process_index() == 0:
      _remove_invalid_ckpts(
        ckpt_path, base_path, keep, overwrite, keep_every_n_steps, True
      )
    orbax_checkpointer.save(
      ckpt_path, target, force=overwrite
    )
    end_time = time.time()
    monitoring.record_event_duration_secs(
      _WRITE_CHECKPOINT_EVENT, end_time - start_time
    )
    return ckpt_path

  warnings.warn(
    (
      'Flax Checkpointing will soon be deprecated in favor of Orbax'
      ' (https://github.com/google/orbax). Please refer to the Checkpoint'
      ' Upgrade Guide'
      ' (https://flax.readthedocs.io/en/latest/guides/converting_and_upgrading/orbax_upgrade_guide.html)'
      ' to self-migrate your code to ocp.'
    ),
    DeprecationWarning,
  )

  target = serialization.to_state_dict(target)
  target, mpa_targets = _split_mp_arrays(target)
  target = serialization.msgpack_serialize(target)
  has_mpa = bool(mpa_targets)

  if not overwrite:
    _check_overwrite_error(ckpt_tmp_path, ckpt_path, base_path, step)  # type: ignore
    sync_global_devices('Flax:Checkpoint:CheckOverwriteBeforeSave')

  # Save the files via I/O sync or async.
  def save_main_ckpt_task():
    jax.monitoring.record_event('/jax/flax/checkpoint/save_main_ckpt_task')
    return _save_main_ckpt_file(
      target,
      has_mpa,
      (ckpt_tmp_path, ckpt_path),
      base_path,
      step,
      keep,
      overwrite,
      keep_every_n_steps,
      start_time,
    )

  # Write the main checkpoint file only via process 0, to avoid race condition.
  if process_index() == 0:
    if async_manager:
      async_manager.save_async(save_main_ckpt_task)
    else:
      save_main_ckpt_task()

  if has_mpa:
    if not gda_manager:
      raise errors.MPACheckpointingRequiredError(ckpt_path, step)
    # Creating the directory containing GDAs explicitly. This should happen only
    # on process 0 and before any worker starts to write GDA data.
    if process_index() == 0:
      _make_mpa_dirs(mpa_targets, ckpt_tmp_path)
    sync_global_devices('Flax:Checkpoint:AfterCreateMPADir')
    _save_mpas(
      gda_manager,
      mpa_targets,
      ckpt_tmp_path,
      ckpt_path,
      base_path,
      keep,
      overwrite,
      keep_every_n_steps,
      start_time,
      async_manager,
    )

  end_time = time.time()
  monitoring.record_event_duration_secs(
    _WRITE_CHECKPOINT_EVENT, end_time - start_time
  )
  return ckpt_path


def _all_checkpoints(
  ckpt_dir: str | os.PathLike, prefix: str = 'checkpoint_'
) -> list[str]:
  """Retrieve all checkpoint paths in directory.

  Args:
    ckpt_dir: str: directory of checkpoints to restore from.
    prefix: str: name prefix of checkpoint files.

  Returns:
    Sorted list of checkpoint paths or empty list if no checkpoints were found.
  """
  ckpt_dir = os.fspath(ckpt_dir)  # Pathlib -> str
  checkpoint_files: list[Any] = [
    pathlib.PurePath(c) for c in _allowempty_listdir(ckpt_dir)
  ]
  checkpoint_files = [
    os.path.join(ckpt_dir, c)
    for c in checkpoint_files
    if c.match(f'{prefix}*')
    and not c.match(f'{prefix}tmp')
    and not c.match(f'*{MP_ARRAY_POSTFIX}')
    and not c.match(f'*{ocp.utils.TMP_DIR_SUFFIX}*')
  ]
  checkpoint_files = natural_sort(checkpoint_files)
  if checkpoint_files:
    return checkpoint_files
  else:
    return []


def latest_checkpoint(
  ckpt_dir: str | os.PathLike, prefix: str = 'checkpoint_'
) -> str | None:
  """Retrieve the path of the latest checkpoint in a directory.

  Args:
    ckpt_dir: str: directory of checkpoints to restore from.
    prefix: str: name prefix of checkpoint files.

  Returns:
    The latest checkpoint path or None if no checkpoints were found.
  """
  checkpoint_files = _all_checkpoints(ckpt_dir, prefix)
  if checkpoint_files:
    return checkpoint_files[-1]
  else:
    return None


def available_steps(
  ckpt_dir: str | os.PathLike,
  prefix: str = 'checkpoint_',
  step_type: type = int,
) -> list[int | float]:
  """Return step numbers of available checkpoints in a directory.


  Args:
    ckpt_dir: str: directory of checkpoints to restore from.
    prefix: str: name prefix of checkpoint files.
    step_type: type: type for steps, int (default) or float.

  Returns:
    Sorted list of available steps or empty list if no checkpoints were found.
  """
  checkpoint_files = _all_checkpoints(ckpt_dir, prefix)

  checkpoint_steps = []

  for file in checkpoint_files:
    prefix_idx = file.rfind(prefix)
    checkpoint_steps += [step_type(file[prefix_idx + len(prefix) :])]

  return checkpoint_steps


def restore_checkpoint(
  ckpt_dir: str | os.PathLike,
  target: Any | None,
  step: int | float | None = None,
  prefix: str = 'checkpoint_',
  parallel: bool = True,
  gda_manager: GlobalAsyncCheckpointManager | None = None,
  allow_partial_mpa_restoration: bool = False,
  orbax_checkpointer: ocp.Checkpointer | None = None,
  orbax_transforms: dict | None = None,
) -> PyTree:
  """Restore last/best checkpoint from checkpoints in path.

  Sorts the checkpoint files naturally, returning the highest-valued
  file, e.g.:

  *  ``ckpt_1, ckpt_2, ckpt_3 --> ckpt_3``

  *  ``ckpt_0.01, ckpt_0.1, ckpt_0.001 --> ckpt_0.1``

  *  ``ckpt_-1.0, ckpt_1.0, ckpt_1e5 --> ckpt_1e5``

  Example usage::

    >>> from flax.training import checkpoints
    >>> import jax.numpy as jnp
    >>> import tempfile
    ...
    >>> with tempfile.TemporaryDirectory() as dir_path:
    ...   test_object = {
    ...     'a': jnp.array([1, 2, 3], jnp.int32),
    ...     'b': jnp.array([1, 1, 1], jnp.int32),
    ...   }
    ...   file_path = checkpoints.save_checkpoint(
    ...     dir_path, target=test_object, step=0, prefix='test_', keep=1
    ...   )
    ...   restored_object = checkpoints.restore_checkpoint(
    ...     file_path, target=None
    ...   )
    >>> restored_object
    {'a': Array([1, 2, 3], dtype=int32), 'b': Array([1, 1, 1], dtype=int32)}

  Args:
    ckpt_dir: str: checkpoint file or directory of checkpoints to restore from.
    target: matching object to rebuild via deserialized state-dict. If None, the
      deserialized state-dict is returned as-is.
    step: int or float: step number to load or None to load latest. If
      specified, ckpt_dir must be a directory.
    prefix: str: name prefix of checkpoint files.
    parallel: bool: whether to load seekable checkpoints in parallel, for speed.
    gda_manager: required if checkpoint contains a multiprocess array
      (GlobalDeviceArray or jax Array from pjit). Will read the arrays from the
      separate subdirectory with postfix "_gda".
    allow_partial_mpa_restoration: If true, the given ``target`` doesn't have to
      contain all valid multiprocess arrays. As a result, the restored Pytree
      may have some MPAs not restored correctly. Use this if you cannot provide
      a fully valid ``target`` and don't need all the MPAs in the checkpoint to
      be restored.
    orbax_checkpointer: the ``ocp.Checkpointer`` that handles the underlying
      restore, if the given checkpoint is saved with ocp.
    orbax_transforms: the Orbax transformations that will be passed into
      ``orbax_checkpointer.restore()`` call.

  Returns:
    Restored ``target`` updated from checkpoint file, or if no step specified
    and
    no checkpoint files present, returns the passed-in ``target`` unchanged.
    If a file path is specified and is not found, the passed-in ``target`` will
    be
    returned. This is to match the behavior of the case where a directory path
    is specified but the directory has not yet been created.
  """
  jax.monitoring.record_event('/jax/flax/checkpoint/restore')
  start_time = time.time()
  # Make sure any previous work is done before checking files.
  if orbax_checkpointer and isinstance(
    orbax_checkpointer, ocp.AsyncCheckpointer
  ):
    orbax_checkpointer.wait_until_finished()

  ckpt_dir = os.fspath(ckpt_dir)  # Pathlib -> str
  ckpt_dir = safe_normpath(ckpt_dir)
  if step is not None:
    ckpt_path = _checkpoint_path(ckpt_dir, step, prefix)
    if not io.exists(ckpt_path):
      raise ValueError(f'Matching checkpoint not found: {ckpt_path}')
  else:
    if not io.exists(ckpt_dir):
      logging.info('Found no checkpoint directory at %s', ckpt_dir)
      return target
    if io.isdir(ckpt_dir):
      # This means the given dir is an orbax checkpoint.
      if _is_orbax_checkpoint(ckpt_dir):
        ckpt_path = ckpt_dir
      else:
        ckpt_path = latest_checkpoint(ckpt_dir, prefix)  # type: ignore
        if not ckpt_path:
          logging.info(
            'Found no checkpoint files in %s with prefix %s', ckpt_dir, prefix
          )
          return target
    else:
      ckpt_path = ckpt_dir

  # Restore the checkpoint with Orbax if needed.
  is_orbax = _is_orbax_checkpoint(ckpt_path)
  ckpt_type = 'orbax' if is_orbax else 'legacy Flax'
  logging.info(f'Restoring {ckpt_type} checkpoint from {ckpt_path}')
  if is_orbax:
    if not orbax_checkpointer:
      orbax_checkpointer = ocp.Checkpointer(
        ocp.PyTreeCheckpointHandler()
      )

    restore_kwargs = {}
    if target is not None:
      restore_kwargs['restore_args'] = orbax_utils.restore_args_from_target(
        target
      )
      if isinstance(orbax_checkpointer._handler, ocp.PyTreeCheckpointHandler):  # pylint: disable=protected-access
        restore_kwargs[
          'transforms'
        ] = orbax_utils.maybe_construct_transformations(
          target, orbax_transforms
        )
    restored = orbax_checkpointer.restore(
      ckpt_path, item=target, **restore_kwargs
    )
    restored = serialization.to_state_dict(restored)
    if target is not None:
      restored = serialization.from_state_dict(target, restored)
    end_time = time.time()
    monitoring.record_event_duration_secs(
      _READ_CHECKPOINT_EVENT, end_time - start_time
    )
    return restored

  # Legacy Flax checkpoint restoration.
  ckpt_size = io.getsize(ckpt_path)
  with io.GFile(ckpt_path, 'rb') as fp:
    if parallel and fp.seekable():
      buf_size = 128 << 20  # 128M buffer.
      num_bufs = ckpt_size / buf_size
      logging.debug('num_bufs: %d', num_bufs)
      checkpoint_contents = bytearray(ckpt_size)

      def read_chunk(i):
        # NOTE: We have to re-open the file to read each chunk, otherwise the
        # parallelism has no effect. But we could reuse the file pointers
        # within each thread.
        with io.GFile(ckpt_path, 'rb') as f:
          f.seek(i * buf_size)
          buf = f.read(buf_size)
          if buf:
            checkpoint_contents[i * buf_size : i * buf_size + len(buf)] = buf
          return len(buf) / buf_size

      pool_size = 32
      pool = thread.ThreadPoolExecutor(pool_size)
      results = pool.map(read_chunk, range(int(num_bufs) + 1))
      pool.shutdown(wait=False)
      logging.debug(f'results: {list(results)}')
    else:
      checkpoint_contents = fp.read()

  state_dict = serialization.msgpack_restore(checkpoint_contents)
  state_dict = _restore_mpas(
    state_dict,
    target,
    ckpt_path,
    step,
    gda_manager,
    allow_partial_mpa_restoration,
  )

  if target is None:
    restored_checkpoint = state_dict
  else:
    restored_checkpoint = serialization.from_state_dict(target, state_dict)

  end_time = time.time()
  monitoring.record_event_duration_secs(
    _READ_CHECKPOINT_EVENT, end_time - start_time
  )

  return restored_checkpoint


def convert_pre_linen(params: PyTree) -> PyTree:
  """Converts a pre-Linen parameter pytree.

  In pre-Linen API submodules were numbered incrementally, independent of the
  submodule class. With Linen this behavior has changed to keep separate
  submodule counts per module class.

  Consider the following module::

    class Model(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = nn.Conv(1, 1)(x)
        x = nn.Dense(1)(x)
        return x

  In pre-Linen the resulting params would have had the structure:

    ``{'Conv_0': { ... }, 'Dense_1': { ... } }``

  With Linen the resulting params would instead have had the structure:

    ``{'Conv_0': { ... }, 'Dense_0': { ... } }``

  To convert from pre-Linen format to Linen simply call::

    params = convert_pre_linen(pre_linen_params)

  Note that you can also use this utility to convert pre-Linen collections
  because they're following the same module naming. Note though that collections
  were "flat" in pre-Linen and first need to be unflattened before they can be
  used with this function::

    batch_stats = convert_pre_linen(flax.traverse_util.unflatten_dict({
        tuple(k.split('/')[1:]): v
        for k, v in pre_linen_model_state.as_dict().items()
    }))

  Then Linen variables can be defined from these converted collections::

    variables = {'params': params, 'batch_stats': batch_stats}

  Args:
    params: Parameter pytree in pre-Linen format. If the pytree is already in
      Linen format, then the returned pytree is unchanged (i.e. this function
      can safely be called on any loaded checkpoint for use with Linen).

  Returns:
    Parameter pytree with Linen submodule naming.
  """
  if not isinstance(params, (dict, core.FrozenDict)):
    return params
  params_renamed = {}
  counts: dict[Any, Any] = {}
  names = natural_sort(params.keys())
  for name in names:
    value = params[name]
    match = MODULE_NUM_RE.match(name)
    if match:
      module = match.group(1)
      num = counts.get(module, 0)
      name = f'{module}_{num}'
      counts[module] = num + 1
    params_renamed[name] = convert_pre_linen(value)

  if isinstance(params, core.FrozenDict):
    params_renamed = core.freeze(params_renamed)  # type: ignore
  return params_renamed
