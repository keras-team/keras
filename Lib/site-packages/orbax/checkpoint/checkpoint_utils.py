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

"""High-level checkpoint utils provided for user convenience."""

import contextlib
import time
from typing import Any, Callable, Iterator, Optional

from absl import logging
from etils import epath
import jax
from jax.experimental import layout
import numpy as np
from orbax.checkpoint import utils
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import step as step_lib
from orbax.checkpoint._src.path.snapshot import snapshot as snapshot_lib
from orbax.checkpoint._src.serialization import type_handlers


PyTree = Any
STANDARD_ARRAY_TYPES = (int, float, np.ndarray, jax.Array)
_SNAPSHOTS = '_SNAPSHOTS'
Layout = layout.Layout


def _init_step_name_format(
    step_name_format: Optional[step_lib.NameFormat[step_lib.Metadata]] = None,
    step_prefix: Optional[str] = None,
    step_format_fixed_length: Optional[int] = None,
):
  return step_name_format or step_lib.standard_name_format(
      step_prefix=step_prefix,
      step_format_fixed_length=step_format_fixed_length,
  )


def get_snapshot_dir_from_step_dir(
    step_dir: epath.Path, snapshot_dir: Optional[epath.Path] = None
) -> epath.Path:
  """Returns the snapshot directory from the step directory."""
  if snapshot_dir is None:
    snapshot_dir = step_dir.parent / _SNAPSHOTS
  new_path = snapshot_dir / step_dir.name
  return new_path


def _snapshot_checkpoint(
    checkpoint_dir: epath.Path,
    step: int,
    step_name_format: step_lib.NameFormat[step_lib.Metadata],
    snapshot_dir: Optional[epath.Path] = None,
):
  """Uses `Snapshot` class to create a cheap "copy" of the checkpoint."""
  if multihost.process_index() == 0:
    logging.info('Snpashotting step: %d.', step)
    step_dir = step_name_format.find_step(checkpoint_dir, step).path
    if not step_dir.exists():
      raise ValueError(f'Step directory {step_dir} does not exist.')

    if snapshot_dir is None:
      snapshot_dir = checkpoint_dir / _SNAPSHOTS
    if not snapshot_dir.exists():
      try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
      except OSError as e:
        raise ValueError(
            f'Failed to create snapshot directory {snapshot_dir}. Please'
            ' provide a snapshot directory instead.'
        ) from e

    snapshot_path = get_snapshot_dir_from_step_dir(step_dir, snapshot_dir)
    if epath.Path(snapshot_path).exists():
      return True
    snapshot_impl = snapshot_lib.create_instance(str(snapshot_path))
    dst_path = snapshot_impl.create_snapshot(str(step_dir), str(snapshot_path))
    if str(snapshot_path) == dst_path:
      return True
  return False


def _release_snapshot(
    checkpoint_dir: epath.Path,
    step: int,
    step_name_format: step_lib.NameFormat[step_lib.Metadata],
    snapshot_dir: Optional[epath.Path] = None,
):
  """Releases snapshot by deleting the snapshot of the checkpoint."""
  if multihost.process_index() == 0:
    logging.info('Releasing snapshot at step: %d.', step)
    if snapshot_dir is None:
      snapshot_dir = checkpoint_dir / _SNAPSHOTS
    snapshot_path = snapshot_dir / step_name_format.build_name(step)
    snapshot_impl = snapshot_lib.create_instance(str(snapshot_path))
    snapshot_impl.release_snapshot(str(snapshot_path))


def _reached_desired_step(step: int, until_step: Optional[int]) -> bool:
  if step is None:
    return False
  elif until_step is None:
    return True
  elif step >= until_step:
    return True
  return False


def _wait_for_new_checkpoint(
    checkpoint_dir: epath.Path,
    *,
    step_name_format: step_lib.NameFormat[step_lib.Metadata],
    until_step: Optional[int] = None,
    seconds_to_sleep: int = 1,
    timeout: Optional[int] = None,
    timeout_fn: Optional[Callable[[], bool]] = None,
    snapshot_dir: Optional[epath.Path] = None,
) -> int:
  """See documentation for wait_for_new_checkpoint."""
  start = time.time()
  stop_time = start + timeout if timeout is not None else None

  def _sleep_and_maybe_exit():
    if stop_time is not None and time.time() + seconds_to_sleep > stop_time:
      if timeout_fn is None:
        return True
      elif timeout_fn():  # Only exit when timeout_fn indicates completion.
        return True
    logging.info('Sleeping for %d seconds.', seconds_to_sleep)
    time.sleep(seconds_to_sleep)
    return False

  log_str = f'Waiting for new checkpoint at {checkpoint_dir}. '
  if until_step is not None:
    log_str += f'Waiting until step {until_step} is reached. '
  if timeout is not None:
    log_str += f'Will time out after {timeout} seconds. '
  logging.info(log_str)

  result = -1
  if multihost.process_index() == 0:
    while True:
      if not checkpoint_dir.exists():
        if _sleep_and_maybe_exit():
          break
        continue  # continue waiting until directory creation or timeout.

      steps = utils.checkpoint_steps(checkpoint_dir)
      checkpoint_step = max(steps) if steps else None
      if _reached_desired_step(checkpoint_step, until_step):
        if not _snapshot_checkpoint(
            checkpoint_dir, checkpoint_step, step_name_format, snapshot_dir
        ):
          continue
        result = checkpoint_step
        break
      elif _sleep_and_maybe_exit():
        break

  result = multihost.broadcast_one_to_all(np.int32(result)).item()
  wait_duration = time.time() - start
  jax.monitoring.record_event_duration_secs(
      '/jax/orbax/checkpoint_utils/wait_duration', wait_duration
  )

  if result == -1:
    logging.info('Timed out waiting for new checkpoint. Returning -1.')
  else:
    logging.info('Found new checkpoint step: %d.', result)
  return result


@contextlib.contextmanager
def wait_for_new_checkpoint(
    checkpoint_dir: epath.Path,
    *,
    until_step: Optional[int] = None,
    seconds_to_sleep: int = 1,
    timeout: Optional[int] = None,
    timeout_fn: Optional[Callable[[], bool]] = None,
    step_prefix: Optional[str] = None,
    step_format_fixed_length: Optional[int] = None,
    step_name_format: Optional[step_lib.NameFormat[step_lib.Metadata]] = None,
    snapshot_dir: Optional[epath.Path] = None,
):
  """Waits until a new checkpoint file is found.

  Automatically snapshots any checkpoint that is returned, and releases the
  snapshot of the checkpoint when execution returns to this function.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    until_step: If specified, waits until a step greater than or equal to
      `until_step` has been found. If set to None (default), returns the first
      step found.
    seconds_to_sleep: The number of seconds to sleep for before looking for a
      new checkpoint.
    timeout: The maximum number of seconds to wait. If left as `None`, then the
      process will wait indefinitely.
    timeout_fn: Optional function to call after a timeout.  If the function
      returns True, then it means that no new checkpoints will be generated and
      the iterator will exit.  The function is called with no arguments.
    step_prefix: A prefix applied to step numbers (e.g. <prefix>_42).
    step_format_fixed_length: Expects to find checkpoint step directories with
      exactly this number of digits (leading zeros if necessary).
    step_name_format: Step NameFormat used to find step under given root
      directory. If provided, `step_prefix` and `step_format_fixed_length` are
      ignored.
    snapshot_dir: The directory in which snapshots are saved. If not provided,
      the snapshot directory is created as `checkpoint_dir / _SNAPSHOTS`.

  Yields:
    a new checkpoint step, or -1 if the timeout was reached.
  """
  step_name_format = _init_step_name_format(
      step_name_format=step_name_format,
      step_prefix=step_prefix,
      step_format_fixed_length=step_format_fixed_length,
  )
  step = _wait_for_new_checkpoint(
      checkpoint_dir,
      step_name_format=step_name_format,
      until_step=until_step,
      seconds_to_sleep=seconds_to_sleep,
      timeout=timeout,
      timeout_fn=timeout_fn,
      snapshot_dir=snapshot_dir,
  )
  try:
    yield step
  finally:
    # Release snapshot on the checkpoint step.
    if step != -1:
      logging.info(
          'Releasing snapshot for step: %d after releasing control.', step
      )
      _release_snapshot(checkpoint_dir, step, step_name_format, snapshot_dir)


def checkpoints_iterator(
    checkpoint_dir: epath.PathLike,
    *,
    min_interval_secs: int = 0,
    seconds_to_sleep: int = 1,
    timeout: Optional[int] = None,
    timeout_fn: Optional[Callable[[], bool]] = None,
    step_prefix: Optional[str] = None,
    step_format_fixed_length: Optional[int] = None,
    step_name_format: Optional[step_lib.NameFormat[step_lib.Metadata]] = None,
    snapshot_dir: Optional[epath.Path] = None,
) -> Iterator[int]:
  """Continuously yield new checkpoint files as they appear.

  Based on the equivalent TF method.

  The iterator only checks for new checkpoints when control flow has been
  reverted to it. This means it can miss checkpoints if your code takes longer
  to run between iterations than `min_interval_secs` or the interval at which
  new checkpoints are written.

  Warning: If CheckpointManager is running in a different process for training
  and is cleaning up old checkpoints (via the `max_to_keep` argument), steps
  returned by this function may not be valid after being clean up by another
  process. In this case, `max_to_keep` should be increased (suggested value: 5)

  The `timeout` argument is the maximum number of seconds to block waiting for
  a new checkpoint.  It is used in combination with the `timeout_fn` as
  follows:

  * If the timeout expires and no `timeout_fn` was specified, the iterator
    stops yielding.
  * If a `timeout_fn` was specified, that function is called and if it returns
    a true boolean value the iterator stops yielding.
  * If the function returns a false boolean value then the iterator resumes the
    wait for new checkpoints.  At this point the timeout logic applies again.

  This behavior gives control to callers on what to do if checkpoints do not
  come fast enough or stop being generated.  For example, if callers have a way
  to detect that the training has stopped and know that no new checkpoints
  will be generated, they can provide a `timeout_fn` that returns `True` when
  the training has stopped.  If they know that the training is still going on
  they return `False` instead.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    min_interval_secs: The minimum number of seconds between yielding
      checkpoints.
    seconds_to_sleep: Seconds to sleep if a checkpoint is not found. Note the
      difference with min_interval_secs, which puts a lower bound on how when a
      new checkpoint will be looked for after yielding one checkpoint.
      seconds_to_sleep instead specifies how we should sleep for if no new
      checkpoints are found. Note that the timeout is only checked when not
      sleeping, so a `seconds_to_sleep` longer than the timeout would result in
      timing out after `seconds_to_sleep` seconds rather than `timeout` seconds.
    timeout: The maximum number of seconds to wait between checkpoints. The
      function will time out if `timeout` seconds have passed since a new
      checkpoint step was found. If left as `None`, then the process will wait
      indefinitely.
    timeout_fn: Optional function called after a timeout. If the function
      returns True, then it means that no new checkpoints will be generated and
      the iterator will exit. The function is called with no arguments.
    step_prefix: A prefix applied to step numbers (e.g. <prefix>_42).
    step_format_fixed_length: Expects to find checkpoint step directories with
      exactly this number of digits (leading zeros if necessary).
    step_name_format: Step NameFormat used to find step under given root
      directory. If provided, `step_prefix` and `step_format_fixed_length` are
      ignored.
    snapshot_dir: The directory in which snapshots are saved. If not provided,
      the snapshot directory is created as `checkpoint_dir / _SNAPSHOTS`.

  Yields:
    Integer step numbers of the latest checkpoints as they arrive.
  """
  checkpoint_dir = epath.Path(checkpoint_dir)
  step_name_format = _init_step_name_format(
      step_name_format=step_name_format,
      step_prefix=step_prefix,
      step_format_fixed_length=step_format_fixed_length,
  )
  snapshot_impl = snapshot_lib.create_instance(str(checkpoint_dir))
  if snapshot_dir is None:
    snapshot_dir = checkpoint_dir / _SNAPSHOTS
  if snapshot_dir.exists():
    for step_dir in snapshot_dir.iterdir():
      snapshot_impl.release_snapshot(str(step_dir))
  checkpoint_step = None
  while True:
    until_step = checkpoint_step + 1 if checkpoint_step is not None else None
    with wait_for_new_checkpoint(
        checkpoint_dir,
        until_step=until_step,
        seconds_to_sleep=seconds_to_sleep,
        timeout=timeout,
        timeout_fn=timeout_fn,
        step_name_format=step_name_format,
        snapshot_dir=snapshot_dir,
    ) as new_checkpoint_step:
      if new_checkpoint_step == -1:
        if not timeout_fn:
          logging.info('Timed-out waiting for a checkpoint.')
          return
        if timeout_fn():
          # The timeout_fn indicated that we are truly done.
          return
        else:
          # The timeout_fn indicated that more checkpoints may come.
          continue
      start = time.time()
      checkpoint_step = new_checkpoint_step
      yield checkpoint_step
    time_to_next_eval = start + min_interval_secs - time.time()
    if time_to_next_eval > 0:
      time.sleep(time_to_next_eval)


def _python_type_from_dtype(dtype):
  if hasattr(dtype, 'type'):
    return type(dtype.type(0).item())
  else:
    return type(dtype(0).item())


def construct_restore_args(
    target: PyTree,
    sharding_tree: Optional[PyTree] = None,
    set_global_shape: bool = True,
    support_layout: bool = False,
) -> PyTree:
  """Creates restore_args given a target PyTree.

  This method should be used in conjunction with a CheckpointManager or
  Checkpointer that wraps a PyTreeCheckpointHandler.

  For example::

    mngr = CheckpointManager(path, Checkpointer(PyTreeCheckpointHandler()))
    restore_args = construct_restore_args(train_state, train_state_sharding)
    restore_kwargs = {'restore_args': restore_args}
    mngr.restore(..., restore_kwargs=restore_kwargs)

  OR::

    mngr = CheckpointManager(path, {
        'train_state': Checkpointer(PyTreeCheckpointHandler())
    })
    restore_args = construct_restore_args(train_state, train_state_sharding)
    restore_kwargs = {'train_state': {'restore_args': restore_args} }
    mngr.restore(..., restore_kwargs=restore_kwargs)

  OR::

    ckptr = Checkpointer(PyTreeCheckpointHandler())
    restore_args = construct_restore_args(train_state, train_state_sharding)
    ckptr.restore(..., restore_args=restore_args)

  If a leaf in target is a np.ndarray, or int, or string, for example, a
  corresponding value for that leaf must be provided in axes_tree, but will be
  ignored.

  Args:
    target: The returned PyTree will match the structure of `target`. `target`
      may contain `value_metadata.Metadata`, real scalar or array values, or may
      contain jax.ShapeDtypeStruct.
    sharding_tree: A PyTree matching `target` which will be used to set the
      restoration sharding. If not provided, sharding will default to the
      shardings specified by `target`.
    set_global_shape: If true, set the `global_shape` field of ArrayRestoreArgs.
    support_layout: If true, layout is extracted from jax.Array or
      jax.ShapeDtypeStruct.

  Returns:
    A PyTree matching target of RestoreArgs (or ArrayRestoreArgs) objects.
  """

  def _array_restore_args(
      value: Any,
      sharding: Optional[jax.sharding.Sharding | Layout],
      dtype: Optional[np.dtype] = None,
  ) -> type_handlers.ArrayRestoreArgs:
    return type_handlers.ArrayRestoreArgs(
        restore_type=jax.Array,
        sharding=sharding,
        global_shape=value.shape if set_global_shape else None,
        dtype=dtype,
    )

  def _restore_args(
      value: Any,
      sharding: Optional[jax.sharding.Sharding],
  ) -> type_handlers.RestoreArgs:
    if isinstance(value, jax.ShapeDtypeStruct):
      if sharding is None:
        return type_handlers.RestoreArgs(dtype=value.dtype)
      else:
        return _array_restore_args(value, sharding, value.dtype)
    elif isinstance(value, value_metadata.Metadata):
      if isinstance(value, value_metadata.StringMetadata):
        return type_handlers.RestoreArgs(restore_type=str)
      elif isinstance(value, value_metadata.ScalarMetadata):
        return type_handlers.RestoreArgs(
            restore_type=_python_type_from_dtype(value.dtype), dtype=value.dtype
        )
      elif isinstance(value, value_metadata.ArrayMetadata):
        if sharding is None:
          return type_handlers.RestoreArgs(
              restore_type=np.ndarray, dtype=value.dtype
          )
        else:
          return _array_restore_args(value, sharding, value.dtype)
      else:
        raise ValueError(f'Unsupported value_metadata class: {type(value)}.')
    elif isinstance(value, STANDARD_ARRAY_TYPES):
      if isinstance(value, np.ndarray):
        return type_handlers.RestoreArgs(
            restore_type=type(value), dtype=value.dtype
        )
      elif isinstance(value, jax.Array):
        return _array_restore_args(value, sharding, value.dtype)
      else:
        return type_handlers.RestoreArgs(restore_type=type(value))
    elif isinstance(value, str):
      return type_handlers.RestoreArgs(restore_type=str)
    else:
      raise ValueError(f'Unsupported type: {type(value)}')

  def _get_sharding_or_layout(value):
    if hasattr(value, 'sharding'):
      if (
          support_layout
          and hasattr(value, 'layout')
          and value.layout.device_local_layout
      ):
        # value is a jax.Array or a jax.ShapeDtypeStruct.
        return value.layout
      else:
        return value.sharding
    else:
      return None

  if sharding_tree is None:
    sharding_tree = jax.tree.map(_get_sharding_or_layout, target)
  if isinstance(target, tree_metadata.TreeMetadata):
    return tree_metadata.build_default_tree_metadata(
        jax.tree.map(_restore_args, target.tree, sharding_tree.tree),
        custom_metadata=target.custom_metadata,
    )
  else:
    return jax.tree.map(_restore_args, target, sharding_tree)
