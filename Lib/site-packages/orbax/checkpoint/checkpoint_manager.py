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

"""A class providing functionalities for managing a series of checkpoints."""

from __future__ import annotations

import concurrent
import dataclasses
import datetime
import threading
import time
import typing
from typing import Any, Callable, Container, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union, overload

from absl import logging
from etils import epath
from etils import epy
import jax
from jax.experimental.array_serialization import serialization as jax_serialization
from orbax.checkpoint import abstract_checkpoint_manager
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import options as options_lib
from orbax.checkpoint import utils
from orbax.checkpoint._src import threading as threading_lib
from orbax.checkpoint._src.checkpoint_managers import save_decision_policy as save_decision_policy_lib
from orbax.checkpoint._src.checkpointers import abstract_checkpointer
from orbax.checkpoint._src.checkpointers import async_checkpointer
from orbax.checkpoint._src.checkpointers import checkpointer as checkpointer_lib
from orbax.checkpoint._src.handlers import checkpoint_handler
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
from orbax.checkpoint._src.handlers import handler_registration
from orbax.checkpoint._src.handlers import json_checkpoint_handler
from orbax.checkpoint._src.handlers import proto_checkpoint_handler
from orbax.checkpoint._src.logging import abstract_logger
from orbax.checkpoint._src.logging import standard_logger
from orbax.checkpoint._src.logging import step_statistics
from orbax.checkpoint._src.metadata import checkpoint
from orbax.checkpoint._src.metadata import checkpoint_info
from orbax.checkpoint._src.metadata import root_metadata_serialization
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint._src.path import deleter
from orbax.checkpoint._src.path import step as step_lib
from orbax.checkpoint._src.path import utils as path_utils
from typing_extensions import Self  # for Python version < 3.11



PyTree = Any
CheckpointDirs = Tuple[str, str]
SaveParams = Mapping[str, Any]
RestoreParams = SaveParams
CheckpointInfo = checkpoint_info.CheckpointInfo
AbstractCheckpointer = abstract_checkpointer.AbstractCheckpointer
CheckpointersDict = Mapping[str, AbstractCheckpointer]
AbstractCheckpointManager = (
    abstract_checkpoint_manager.AbstractCheckpointManager
)
StepMetadata = checkpoint.StepMetadata
RootMetadata = checkpoint.RootMetadata
ItemMetadata = checkpoint.CompositeItemMetadata | checkpoint.SingleItemMetadata
AsyncCheckpointer = async_checkpointer.AsyncCheckpointer
Checkpointer = checkpointer_lib.Checkpointer
JsonCheckpointHandler = json_checkpoint_handler.JsonCheckpointHandler
ProtoCheckpointHandler = proto_checkpoint_handler.ProtoCheckpointHandler
CompositeCheckpointHandler = (
    composite_checkpoint_handler.CompositeCheckpointHandler
)
CheckpointHandler = checkpoint_handler.CheckpointHandler
CheckpointArgs = checkpoint_args.CheckpointArgs
CheckpointHandlersDict = Mapping[str, CheckpointHandler]
CheckpointHandlerRegistry = handler_registration.CheckpointHandlerRegistry
serialize_root_metadata = root_metadata_serialization.serialize
deserialize_root_metadata = root_metadata_serialization.deserialize

AsyncOptions = options_lib.AsyncOptions
MultiprocessingOptions = options_lib.MultiprocessingOptions
FileOptions = options_lib.FileOptions

DEFAULT_ITEM_NAME = 'default'
DESCRIPTOR_ITEM_NAME = 'descriptor'
METRIC_ITEM_NAME = 'metrics'
METADATA_ITEM_NAME = 'metadata'
RESERVED_ITEM_NAMES = [DESCRIPTOR_ITEM_NAME, METRIC_ITEM_NAME]

_INIT_TIME = datetime.datetime.now(tz=datetime.timezone.utc)


def _metrics_file_exists(metrics_item_path: epath.Path) -> bool:
  """True if item directory AND actual file both exist."""
  return (
      metrics_item_path.exists()
      and (metrics_item_path / METRIC_ITEM_NAME).exists()
  )


class StepAlreadyExistsError(ValueError):
  """Raised when a step is already present for a save request."""

  pass


class _FinalizeThread(threading.Thread):
  """Thread wrapper that raises an exception if encountered."""

  exception = None

  def __init__(
      self,
      step: int,
      target: Callable[..., object],
      name: str,
      args=(),
      kwargs=None,
      *,
      daemon=None,
  ):
    super().__init__(
        target=target,
        name=name,
        args=args,
        kwargs=kwargs,
        daemon=daemon,
    )
    self._step = step
    self._thread_local = threading.local()

  def step(self) -> int:
    return self._step

  def run(self):
    try:
      super().run()
    except BaseException as e:  # pylint:disable=broad-exception-caught
      self.exception = e

  def join(self, *args, **kwargs):
    super().join(*args, **kwargs)

    # Keep track of whether the calling thread should raise the exception.
    if not hasattr(self._thread_local, 'should_raise'):
      self._thread_local.should_raise = True
    if self.exception and self._thread_local.should_raise:
      # Each thread should only raise once.
      self._thread_local.should_raise = False
      raise self.exception


@dataclasses.dataclass
class _ShouldSaveFnPolicy(save_decision_policy_lib.SaveDecisionPolicy):
  """A policy that uses a user-provided should_save_fn."""

  should_save_fn: Callable[[int, Optional[int]], bool]

  def should_save(
      self,
      step: checkpoint_info.CheckpointInfo,
      previous_steps: Sequence[checkpoint_info.CheckpointInfo],
      *,
      context: save_decision_policy_lib.DecisionContext,
  ) -> bool:
    return self.should_save_fn(
        step.step, previous_steps[-1].step if previous_steps else None
    )


def _get_default_save_decision_policy(
    options: CheckpointManagerOptions,
) -> save_decision_policy_lib.SaveDecisionPolicy:
  """Creates a default policy from CheckpointManagerOptions."""
  save_interval_policies = []
  if options.should_save_fn is not None:
    save_interval_policies.append(_ShouldSaveFnPolicy(options.should_save_fn))
    save_interval_policies.append(
        save_decision_policy_lib.PreemptionCheckpointingPolicy()
    )
  else:
    if options.save_interval_steps is not None:
      save_interval_policies.append(
          save_decision_policy_lib.FixedIntervalPolicy(
              options.save_interval_steps
          )
      )
    if options.save_on_steps is not None:
      save_interval_policies.append(
          save_decision_policy_lib.SpecificStepsPolicy(options.save_on_steps)
      )
    save_interval_policies.append(
        save_decision_policy_lib.PreemptionCheckpointingPolicy()
    )
    save_interval_policies.append(save_decision_policy_lib.InitialSavePolicy())
  return save_decision_policy_lib.AnySavePolicy(save_interval_policies)


# TODO(b/268051457) Clean up when no longer depended upon by internal users.
def is_async_checkpointer(checkpointer: AbstractCheckpointer):
  return isinstance(
      checkpointer, async_checkpointer.AsyncCheckpointer
  ) or isinstance(
      checkpointer,
      jax_serialization.GlobalAsyncCheckpointManagerBase,
  )


# TODO(b/309965339) Set todelete_subdir defaults if directory is on CNS.
@dataclasses.dataclass
class CheckpointManagerOptions:
  """Optional arguments for CheckpointManager.

  save_interval_steps:
    The interval at which checkpoints should be saved.
    Ensures checkpoints will only be saved every n steps. Defaults to 1.
  max_to_keep:
    If provided, specifies the maximum number of checkpoints to
    keep. Older checkpoints are removed. By default, does not remove any old
    checkpoints. Must be None or non-negative. When set, checkpoints
    may be considered for deletion when there are more than `max_to_keep`
    checkpoints present. Checkpoints are kept if they meet any of the conditions
    below, such as `keep_time_interval`, `keep_period`, etc. Any remaining
    checkpoints that do not meet these conditions are garbage-collected.
  keep_time_interval:
    When more than max_to_keep checkpoints are present,
    an older checkpoint that would ordinarily be deleted will be preserved if it
    has been at least `keep_time_interval` since the previous preserved
    checkpoint. The default setting of `None` does not preserve any checkpoints
    in this way. For example, this may be used to ensure checkpoints are
    retained at a frequency of approximately than one per hour.
  keep_period:
    If set, any existing checkpoints matching checkpoint_step % keep_period == 0
    will not be deleted.
  should_keep_fn:
    WARNING: It is an experimental feature and may change without notice.
    If set then this Callable overrides the behavior of `keep_period`, which is
    ignored. It does not change the behavior of `keep_time_interval` and
    `keep_checkpoints_without_metrics`. It is a predicate with signature:
    `Callable[[int], bool]`, where int is the step number of the checkpoint.
  best_fn:
    If set, maintains checkpoints based on the quality of given
    metrics rather than recency. The function should accept a PyTree of metrics,
    and return a scalar value that can be used to determine the quality score
    of the checkpoint. If `max_to_keep` is also set, then the retained
    checkpoints will be kept based on their quality, as measured by this
    function.
  best_mode:
    One of ['max', 'min']. The best metric is determine on the basis of this
    value.
  keep_checkpoints_without_metrics:
    If False, checkpoints without metrics present
    are eligible for cleanup. Otherwise, they will never be deleted.
  step_prefix:
    If provided, step directories will take the form
    f'{step_prefix}_<step>'. Otherwise, they will simply be an integer <step>.
  step_format_fixed_length:
    If set, formats step with n digits (leading zeros).
    This makes sorting steps easier. Otherwise, step has no leading zeros.
  step_name_format:
    NameFormat to build or find steps under input root directory. If provided,
    `step_prefix`, `step_format_fixed_length` are ignored.
  create:
    If True, creates the top-level directory if it does not already exist.
  cleanup_tmp_directories:
    If True, cleans up any existing temporary directories
    on CheckpointManager creation.
  save_on_steps:
    Optional set of steps at which checkpoints should be saved.
    Useful to save checkpoints on a fixed set of steps that are not multiple of
    `save_interval_steps`.
  single_host_load_and_broadcast:
    If True, calling `all_steps(read=True)` will load on only a single host, and
    will then be broadcast to other hosts. Otherwise, I/O will be performed on
    every host. This can be helpful to reduce QPS to the filesystem if there
    are a large number of hosts.
  todelete_subdir: If set, checkpoints to be deleted will be only renamed into a
    subdirectory with the provided string. Otherwise, they will be directly
    deleted from the file system. Useful if checkpoint deletion is time
    consuming. By default, delete the checkpoint assets. Ignored if file system
    is Google Cloud Storage (directory is prefixed with gs://)
  enable_background_delete: If True, old checkpoint deletions will be done in a
    background thread, otherwise, it will be done at the end of each save.  When
    it's enabled, make sure to call CheckpointManager.close() or use context to
    make sure all old steps are deleted before exit.
  read_only: If True, then checkpoints save and delete are skipped. However,
    checkpoints restore works as usual.
  enable_async_checkpointing:
    If True, enables async checkpointing.
  async_options:
    Used to configure properties of async behavior. See above.
  multiprocessing_options: MultiprocessingOptions instance to configure
    multiprocessing behavior.
  should_save_fn:
    Predicate callable to check if given step can be saved. This callable
    accepts step number and optional latest step number as param and returns
    bool. If present then `save_interval_steps` and `save_on_steps` options are
    ignored.
  file_options: Options to configure checkpoint directories and files.
    default=FileOptions().
  save_root_metadata: If True, saves root-level metadata about checkpoints.
    This metadata is not step-specific and is written only once.
  temporary_path_class:
    Optional. The concrete `atomicity_types.TemporaryPath` class to be used by
    the underlying `Checkpointer`.
  save_decision_policy: An object used to determine when a checkpoint should be
    saved. If provided, overrides any other options dealing with this subject,
    including `save_interval_steps`, `save_on_steps`, and `should_save_fn`, and
    is the sole means of determining when a checkpoint should be saved. If not
    provided, these other options are used instead. Prefer to use this option
    over others.
  """

  save_interval_steps: int = 1
  max_to_keep: Optional[int] = None
  keep_time_interval: Optional[datetime.timedelta] = None
  keep_period: Optional[int] = None
  should_keep_fn: Optional[Callable[[int], bool]] = None
  best_fn: Optional[Callable[[PyTree], float]] = None
  best_mode: str = 'max'
  keep_checkpoints_without_metrics: bool = True
  step_prefix: Optional[str] = None
  step_format_fixed_length: Optional[int] = None
  step_name_format: Optional[step_lib.NameFormat[step_lib.Metadata]] = None
  create: bool = True
  cleanup_tmp_directories: bool = False
  save_on_steps: Optional[Container[int]] = None
  single_host_load_and_broadcast: bool = False
  todelete_subdir: Optional[str] = None
  enable_background_delete: bool = False
  read_only: bool = False
  enable_async_checkpointing: bool = True
  async_options: Optional[AsyncOptions] = None
  multiprocessing_options: MultiprocessingOptions = dataclasses.field(
      default_factory=MultiprocessingOptions
  )
  should_save_fn: Optional[Callable[[int, Optional[int]], bool]] = None
  file_options: FileOptions = dataclasses.field(default_factory=FileOptions)
  save_root_metadata: bool = True
  temporary_path_class: Optional[Type[atomicity_types.TemporaryPath]] = None
  save_decision_policy: Optional[
      save_decision_policy_lib.SaveDecisionPolicy
  ] = None

  def __post_init__(self):
    if self.best_mode not in ('min', 'max'):
      msg = (
          "`CheckpointManagerOptions.best_mode` must be one of None, 'min' "
          f"or 'max'. Got {self.best_mode}."
      )
      raise ValueError(msg)
    if self.max_to_keep is not None and self.max_to_keep < 0:
      raise ValueError('Setting of `max_to_keep` must be None or non-negative.')
    if self.read_only and self.save_interval_steps > 0:
      self.save_interval_steps = 0
      logging.warning(
          'CheckpointManagerOptions.read_only=True, setting'
          ' save_interval_steps=0.'
      )
    if self.read_only and self.max_to_keep is not None:
      self.max_to_keep = None
      logging.warning(
          'CheckpointManagerOptions.read_only=True, setting max_to_keep=None.'
      )
    if self.read_only and self.keep_time_interval is not None:
      self.keep_time_interval = None
      logging.warning(
          'CheckpointManagerOptions.read_only=True, setting'
          ' keep_time_interval=None.'
      )
    if self.read_only and self.keep_period is not None:
      self.keep_period = None
      self.should_keep_fn = None
      logging.warning(
          'CheckpointManagerOptions.read_only=True, setting keep_period=None'
          ' and should_keep_fn=None.'
      )
    if self.read_only and self.create:
      self.create = False
      logging.warning(
          'CheckpointManagerOptions.read_only=True, setting create=False.'
      )
    if self.read_only and self.cleanup_tmp_directories:
      self.cleanup_tmp_directories = False
      logging.warning(
          'CheckpointManagerOptions.read_only=True, setting'
          ' cleanup_tmp_directories=False.'
      )
    if self.read_only and self.save_on_steps:
      self.save_on_steps = None
      logging.warning(
          'CheckpointManagerOptions.read_only=True, setting save_on_steps=None.'
      )
    if self.read_only and self.todelete_subdir is not None:
      self.todelete_subdir = None
      logging.warning(
          'CheckpointManagerOptions.read_only=True, setting'
          ' todelete_subdir=None.'
      )
    if self.read_only and self.should_save_fn is not None:
      self.should_save_fn = None
      logging.warning(
          'CheckpointManagerOptions.read_only=True, setting'
          ' should_save_fn=None.'
      )
    if self.should_keep_fn is not None:
      logging.warning(
          'CheckpointManagerOptions.should_keep_fn is set, setting'
          ' keep_period=None (was %s).',
          self.keep_period,
      )
      self.keep_period = None
    self.save_on_steps = frozenset(self.save_on_steps or ())


def _get_args_for_key(
    handler: CheckpointHandler, item_name: str
) -> Tuple[Type[CheckpointArgs], Type[CheckpointArgs]]:
  """Returns the (save, restore) args for the given item name."""

  if not isinstance(handler, CompositeCheckpointHandler):
    raise ValueError(
        'Expected handler to be a `CompositeCheckpointHandler`, but got'
        f' {type(handler)}.'
    )

  registry = handler._handler_registry  # pylint: disable=protected-access
  for (registered_item, _), handler in registry.get_all_entries().items():
    if registered_item == item_name:
      return checkpoint_args.get_registered_args_cls(handler)
  raise ValueError(f'Unknown key "{item_name}" in CompositeCheckpointHandler.')


def _create_root_directory(
    directory: epath.PathLike,
    multiprocessing_options: MultiprocessingOptions,
    file_options: Optional[FileOptions] = None,
) -> None:
  """Creates the top-level directory if it does not already exist."""
  if multiprocessing_options.active_processes is not None:
    raise NotImplementedError(
        'Option `create=True` with `active_processes` set is not'
        ' supported. Please create the root directory yourself.'
    )
  directory = epath.Path(directory)
  if not directory.exists() and utils.is_primary_host(
      multiprocessing_options.primary_host
  ):
    # exists_ok=True is required, see b/362903314.
    directory.mkdir(parents=True, exist_ok=True)
    logging.info('Created directory=%s', directory)
  multihost.sync_global_processes(
      multihost.unique_barrier_key(
          'CheckpointManager:create_directory',
          prefix=multiprocessing_options.barrier_sync_key_prefix,
          # suffix=None,
      ),
      timeout=multihost.DIRECTORY_CREATION_TIMEOUT,
      processes=multiprocessing_options.active_processes,
  )


def determine_default_item_mode_from_args(
    args: args_lib.CheckpointArgs,
) -> bool:
  if isinstance(args, args_lib.Composite):
    return False
  else:
    return True


def _determine_default_item_mode_from_directory(step_path: epath.Path) -> bool:
  return (step_path / DEFAULT_ITEM_NAME).exists()


class CheckpointManager(AbstractCheckpointManager, epy.ContextManager):
  """A generic, synchronous AbstractCheckpointManager implementation."""

  def __init__(
      self,
      directory: epath.PathLike,
      checkpointers: Optional[
          Union[AbstractCheckpointer, CheckpointersDict]
      ] = None,
      options: Optional[CheckpointManagerOptions] = None,
      metadata: Optional[Mapping[str, Any]] = None,
      item_names: Optional[Sequence[str]] = None,
      item_handlers: Optional[
          Union[CheckpointHandler, CheckpointHandlersDict]
      ] = None,
      logger: Optional[abstract_logger.AbstractLogger] = None,
      handler_registry: Optional[CheckpointHandlerRegistry] = None,
  ):
    """CheckpointManager constructor.

    IMPORTANT: `CheckpointManager` has been refactored to provide a new API.
    Please ensure you have migrated all existing use cases to the newer style by
    August 1st, 2024. Please see
    https://orbax.readthedocs.io/en/latest/guides/checkpoint/api_refactor.html
    for technical details.

    The `CheckpointManager` is ultimately backed by a single `Checkpointer`, to
    which saving and restoring is delegated. Behind step management options,
    metrics-related logic, and other frills, saving and restoring with
    `CheckpointManager` is quite similar to using
    `Checkpointer(CompositeCheckpointHandler)`.

    Example::

      # Multiple items.
      with CheckpointManager(
        'path/to/dir/',
        metadata={'version': 1.1, 'lang': 'en'},
      ) as mngr:
        mngr.save(0, args=args.Composite(
            train_state=args.StandardSave(train_state),
            custom_metadata=args.JsonSave(custom_metadata),
          )
        )
        restored = mngr.restore(0)
        print(restored.train_state)
        print(restored.custom_metadata)
        restored = mngr.restore(0, args=args.Composite(
            train_state=args.StandardRestore(abstract_train_state),
          )
        )
        print(restored.train_state)
        print(restored.custom_metadata)  # Error, not restored

      # Single, unnamed (default) item.
      with CheckpointManager(
          'path/to/dir/',
          options = CheckpointManagerOptions(max_to_keep=5, ...),
        ) as mngr:
        mngr.save(0, args=StandardSave(train_state))
        train_state = mngr.restore(0)
        train_state = mngr.restore(0,
        args=StandardRestore(abstract_train_state))

    IMPORTANT: Don't forget to use the keyword `args=...` for save and restore!
    Otherwise you will get the legacy API. This will not be necessary forever,
    but only until the legacy API is removed.

    IMPORTANT: The CheckpointManager is designed to be used as a context
    manager. Use `with CheckpointManager` schematic for automatic cleanup. If
    you can't use a context manager, always call `close()` to release resources
    properly.  Otherwise, background operations such as deleting old checkpoints
    might not finish before your program exits.

    CheckpointManager:
      - is NOT thread-safe.
      - IS multi-process-safe.
      - is NOT multi-job-safe.
    This means that CheckpointManager is intended to be created and called
    across all processes within a single job, where each process is
    single-threaded, but is not safe to use when multiple jobs each have
    CheckpointManager instances pointing to the same root directory. Concretely,
    this means that if you have a trainer job and one or more evaluator jobs,
    the CheckpointManager should be created and called across all processes
    in the trainer, but a CheckpointManager cannot be used in the evaluators.
    Instead, utilities used during evaluation can be found in
    `checkpoint_utils` (see
    https://orbax.readthedocs.io/en/latest/api_reference/checkpoint.checkpoint_utils.html).

    Args:
      directory: the top level directory in which to save all files.
      checkpointers: deprecated, do not use. use `handler_registry` instead.
      options: CheckpointManagerOptions. May be provided to specify additional
        arguments. If None, uses default values of CheckpointManagerOptions.
      metadata: High-level metadata that does not depend on step number. If
        `directory` is write enabled then given metadata is saved only once. A
        new CheckpointManager instance with that `directory` does not overwrite
        the existing metadata and ignores the current given metadata. If
        `directory` is read-only then the current given metadata is not saved as
        expected. A CheckpointManager instance with a read-only `directory` uses
        the metadata if already present, otherwise always uses the current given
        metadata.
      item_names: deprecated, do not use. use `handler_registry` instead.
      item_handlers: deprecated, do not use. use `handler_registry` instead.
      logger: A logger to log checkpointing events.
      handler_registry: A registry of handlers to use for checkpointing. This
        option is mutually exclusive with `checkpointers`,`item_handlers`, and
        'item_names'. See :py:class:`CheckpointHandlerRegistry` for more
        details.
    """
    jax.monitoring.record_event('/jax/orbax/checkpoint_manager/init')
    logging.info(
        '[process=%s][thread=%s] CheckpointManager init: checkpointers=%s,'
        ' item_names=%s, item_handlers=%s, handler_registry=%s',
        multihost.process_index(),
        threading.current_thread().name,
        checkpointers,
        item_names,
        item_handlers,
        handler_registry,
    )

    self._options = options or CheckpointManagerOptions()
    self._multiprocessing_options = self._options.multiprocessing_options
    self._save_decision_policy = (
        self._options.save_decision_policy
        or _get_default_save_decision_policy(self._options)
    )

    if self._options.best_mode not in ['min', 'max']:
      raise ValueError('`best_mode` must be one of: "min", "max"')

    self._logger = logger or standard_logger.StandardLogger()

    if checkpointers and item_names:
      raise ValueError(
          '`item_names` and `checkpointers` are mutually exclusive - do not use'
          ' together.'
      )
    if checkpointers and item_handlers:
      raise ValueError(
          '`item_handlers` and `checkpointers` are mutually exclusive - do not'
          ' use together.'
      )
    if item_names and isinstance(item_handlers, CheckpointHandler):
      raise ValueError(
          '`item_handlers` in default item mode and `item_names` should not be'
          ' provided together.'
      )
    if checkpointers is not None and handler_registry is not None:
      raise ValueError(
          'Deprecated `checkpointers` can not be used with `handler_registry`.'
          ' Please follow the instructions at'
          ' https://orbax.readthedocs.io/en/latest/api_refactor.html to'
          ' migrate.'
      )

    if item_handlers is not None and handler_registry is not None:
      raise ValueError(
          '`item_handlers` and `handler_registry` are mutually exclusive -'
          ' prefer configuring the handler registry.'
      )

    if item_names is not None and handler_registry is not None:
      raise ValueError(
          '`item_names` and `handler_registry` are mutually exclusive - prefer'
          ' configuring the handler registry.'
      )

    # For async_checkpointer.
    self._non_blocking_metadata_store = checkpoint.metadata_store(
        enable_write=True
    )
    # For metadata checkpointer and regular checkpointer.
    self._blocking_metadata_store = checkpoint.metadata_store(
        enable_write=True, blocking_write=True
    )

    self._default_item = threading_lib.OptionalRef[bool]()
    if checkpointers is not None:
      jax.monitoring.record_event(
          '/jax/orbax/deprecation/checkpoint_manager_legacy_init'
      )
      logging.warning(
          'Configured `CheckpointManager` using deprecated legacy API. Please'
          ' follow the instructions at'
          ' https://orbax.readthedocs.io/en/latest/api_refactor.html to'
          ' migrate.'
      )
      self._default_item.set(isinstance(checkpointers, AbstractCheckpointer))
      self._checkpointer = self._configure_checkpointer_legacy_init(
          checkpointers, self._options
      )
    elif handler_registry is not None:
      # There is no way to know if this is a single, unnamed (default) item or
      # not, detemine this lazily instead on the first call to `save`, `restore`
      # or `item_metadata`. Once locked-in, the value of `_default_item` will
      # not change.
      self._default_item.set(None)
      self._checkpointer = self._configure_checkpointer_from_handler_registry(
          handler_registry,
          self._options,
      )
    elif item_names is None and item_handlers is None:
      # In this case, we can just default construct the
      # CheckpointHandlerRegistry and allow the user to lazily specify default
      # vs. multi-item mode.
      self._default_item.set(None)
      handler_registry = handler_registration.DefaultCheckpointHandlerRegistry()
      self._checkpointer = self._configure_checkpointer_from_handler_registry(
          handler_registry,
          self._options,
      )
    else:
      default_item = isinstance(item_handlers, CheckpointHandler)
      self._default_item.set(default_item)
      self._checkpointer = (
          self._configure_checkpointer_from_item_names_and_handlers(
              item_names,
              item_handlers,
              self._options,
              default_item,
          )
      )

    if (
        self._options.async_options is not None
        and self._options.async_options.post_finalization_callback is not None
        and not isinstance(self._checkpointer, AsyncCheckpointer)
    ):
      raise ValueError(
          'AsyncOptions.post_finalization_callback is only supported with'
          ' AsyncCheckpointer. But final resolved checkpointer is: '
          f' {self._checkpointer}'
      )

    self._directory = epath.Path(directory)
    if self._options.read_only:
      logging.warning('Given directory is read only=%s', self._directory)
    if self._options.create:
      _create_root_directory(
          self._directory,
          self._multiprocessing_options,
          self._options.file_options,
      )


    # Cleanup directories from previous runs that may not have been finalized.
    if self._options.cleanup_tmp_directories:
      self._cleanup_tmp_directories()

    self._step_name_format = (
        self._options.step_name_format
        or step_lib.standard_name_format(
            step_prefix=self._options.step_prefix,
            step_format_fixed_length=self._options.step_format_fixed_length,
        )
    )

    self._checkpoints = checkpoint_info.CheckpointInfos(
        self._load_checkpoint_infos()
    )

    self._metadata_dir = self.directory / METADATA_ITEM_NAME
    if self._options.read_only and not self._metadata_dir.exists():
      custom_metadata = {} if metadata is None else dict(metadata)
    else:
      custom_metadata = None
    self._root_metadata = threading_lib.Ref(
        RootMetadata(
            custom_metadata=custom_metadata,
        )
    )

    self._maybe_save_root_metadata(metadata)

    # TODO: b/359854428 - Move Finalize biz logic to a separate class/module.
    self._finalize_thread = threading_lib.OptionalRef[_FinalizeThread]()

    self._checkpoint_deleter: deleter.CheckpointDeleter = (
        deleter.create_checkpoint_deleter(
            self._multiprocessing_options.primary_host,
            self._directory,
            self._options.todelete_subdir,
            self._step_name_format,
            self._options.enable_background_delete,
        )
    )

    logging.info(
        '[process=%s][thread=%s] CheckpointManager created,  primary_host=%s,'
        ' CheckpointManagerOptions=%s, root_directory=%s: %s',
        multihost.process_index(),
        threading.current_thread().name,
        self._multiprocessing_options.primary_host,
        self._options,
        self.directory,
        self,
    )

  def _create_thread_safe_barrier_sync_fn(self) -> Callable[[str], None]:
    """Returns a barrier sync function to be called from threads.

    The function accepts a key, but the timeout is already set up using
    `AsyncOptions.timeout_secs` attribute.

    The Jax based barrier sync util, `sync_global_devices`, should not be called
    concurrently. Otherwise, it may cause a deadlock.

    In general, any Jax function with `collectives` should not be called
    concurrently to avoid deadlocks.
    """
    async_options = self._options.async_options or AsyncOptions()
    timeout_secs = async_options.timeout_secs
    barrier_sync_fn = (
        async_options.barrier_sync_fn
        or multihost.get_barrier_sync_fn(
            processes=self._multiprocessing_options.active_processes
        )
    )
    return lambda key: barrier_sync_fn(key=key, timeout_ms=timeout_secs * 1000)

  def _configure_checkpointer_common(
      self,
      handler: CompositeCheckpointHandler,
      options: CheckpointManagerOptions,
      use_async: bool,
  ) -> Checkpointer:
    if use_async:
      return async_checkpointer.AsyncCheckpointer(
          handler,
          multiprocessing_options=options.multiprocessing_options,
          async_options=options.async_options or AsyncOptions(),
          file_options=options.file_options,
          checkpoint_metadata_store=self._non_blocking_metadata_store,
          temporary_path_class=options.temporary_path_class,
      )
    else:
      return Checkpointer(
          handler,
          multiprocessing_options=options.multiprocessing_options,
          file_options=options.file_options,
          checkpoint_metadata_store=self._blocking_metadata_store,
          temporary_path_class=options.temporary_path_class,
      )

  def _configure_checkpointer_legacy_init(
      self,
      checkpointers: Union[AbstractCheckpointer, CheckpointersDict],
      options: CheckpointManagerOptions,
  ) -> Checkpointer:
    """Initializes _CompositeCheckpointer with legacy style checkpointers."""
    if self._multiprocessing_options.primary_host != 0:
      raise ValueError(
          f'`primary_host`={self._multiprocessing_options.primary_host} is not'
          ' supported in legacy API.'
      )

    item_handlers = {}
    if isinstance(checkpointers, Checkpointer):
      use_async = is_async_checkpointer(checkpointers)
      if isinstance(checkpointers, async_checkpointer.AsyncCheckpointer):
        async_timeout = checkpointers._async_manager._timeout_secs  # pylint: disable=protected-access
      else:
        async_timeout = None
      item_handlers[DEFAULT_ITEM_NAME] = checkpointers.handler
    elif isinstance(checkpointers, dict):
      individual_use_async = []
      async_timeout = 0
      for item_name, checkpointer in checkpointers.items():
        if not isinstance(checkpointer, Checkpointer):
          raise ValueError(
              f'Value corresponding to {item_name} in `checkpointers` is not a'
              f' Checkpointer. Found {type(checkpointer)}.'
          )
        individual_use_async.append(is_async_checkpointer(checkpointer))
        if isinstance(checkpointer, async_checkpointer.AsyncCheckpointer):
          async_timeout = max(
              async_timeout, checkpointer._async_manager._timeout_secs  # pylint: disable=protected-access
          )
        if item_name in RESERVED_ITEM_NAMES:
          raise ValueError(
              f'Found {item_name} in `checkpointers`; this is a reserved key.'
          )
        item_handlers[item_name] = checkpointer.handler
      if any(individual_use_async) and not all(individual_use_async):
        logging.error(
            'Orbax `CheckpointManager` is transitioning toward using'
            ' asynchronous saving logic under the hood in all cases. Users that'
            ' configure `CheckpointManager` with some `Checkpointer`s and some'
            ' `AsyncCheckpointer`s will now see asynchronous logic used to save'
            ' all items. This may result in breakages if the code is assuming'
            ' that certain objects will be available immediately after saving.'
            ' Ensure that if you depend on the result of `save` being fully'
            ' written at a particular moment, use `wait_until_finished()`.'
        )
      use_async = any(individual_use_async)
      async_timeout = async_timeout or None
    else:
      raise ValueError(
          f'Invalid type for `checkpointers`. Found {checkpointers}.'
      )

    # if options.best_fn:
    item_handlers[METRIC_ITEM_NAME] = self._metrics_handler
    if options.async_options is None:
      options.async_options = (
          AsyncOptions(timeout_secs=async_timeout)
          if async_timeout is not None
          else AsyncOptions()
      )
    return self._configure_checkpointer_common(
        CompositeCheckpointHandler(
            composite_options=composite_checkpoint_handler.CompositeOptions(
                multiprocessing_options=options.multiprocessing_options,
                file_options=options.file_options,
                async_options=options.async_options,
            ),
            **item_handlers,
        ),
        options,
        use_async,
    )


  def _configure_checkpointer_from_item_names_and_handlers(
      self,
      item_names: Optional[Sequence[str]],
      item_handlers: Optional[Union[CheckpointHandler, CheckpointHandlersDict]],
      options: CheckpointManagerOptions,
      default_item: bool,
  ) -> Checkpointer:
    """Initializes _CompositeCheckpointer given `item_names`."""
    if (
        self._multiprocessing_options.primary_host is None
        and item_handlers is None
    ):
      raise ValueError(
          'When primary_host is set to None, item_handlers must be provided to'
          ' match with the primary_host setting.'
      )
    if default_item:
      item_handler = (
          item_handlers
          if isinstance(item_handlers, CheckpointHandler)
          else None
      )
      all_item_handlers = {DEFAULT_ITEM_NAME: item_handler}
    else:
      # Initialize all_item_handlers with None or empty.
      if item_names:
        all_item_handlers = {item_name: None for item_name in item_names}
      else:
        all_item_handlers = {}
      # Update all_item_handlers with provided CheckpointHandlers.
      if item_handlers and isinstance(item_handlers, Mapping):
        for item_name, handler in item_handlers.items():
          all_item_handlers[item_name] = handler

    for item_name in all_item_handlers:
      if item_name in RESERVED_ITEM_NAMES:
        raise ValueError(
            f'Found {item_name} in `checkpointers`; this is a reserved key.'
        )
    all_item_handlers[METRIC_ITEM_NAME] = self._metrics_handler
    # CompositeCheckpointHandler defers per-item handler creation until
    # save/restore time.
    return self._configure_checkpointer_common(
        CompositeCheckpointHandler(
            composite_options=composite_checkpoint_handler.CompositeOptions(
                multiprocessing_options=options.multiprocessing_options,
                file_options=options.file_options,
                async_options=options.async_options,
            ),
            **all_item_handlers,
        ),
        options,
        options.enable_async_checkpointing,
    )

  def _configure_checkpointer_from_handler_registry(
      self,
      handler_registry: CheckpointHandlerRegistry,
      options: CheckpointManagerOptions,
  ) -> Checkpointer:
    """Initializes _CompositeCheckpointer given a `handler_registry`."""
    metrics_handler = self._metrics_handler
    if not handler_registry.has(METRIC_ITEM_NAME, args_lib.JsonSave):
      handler_registry.add(METRIC_ITEM_NAME, args_lib.JsonSave, metrics_handler)
    if not handler_registry.has(METRIC_ITEM_NAME, args_lib.JsonRestore):
      handler_registry.add(
          METRIC_ITEM_NAME, args_lib.JsonRestore, metrics_handler
      )

    # CompositeCheckpointHandler defers per-item handler creation until
    # save/restore time.
    return self._configure_checkpointer_common(
        CompositeCheckpointHandler(
            composite_options=composite_checkpoint_handler.CompositeOptions(
                multiprocessing_options=options.multiprocessing_options,
                async_options=options.async_options,
            ),
            handler_registry=handler_registry,
        ),
        options,
        options.enable_async_checkpointing,
    )

  @property
  def directory(self) -> epath.Path:
    """Returns the top-level directory containing checkpoints for all items."""
    return self._directory

  def all_steps(self, read: bool = False) -> Sequence[int]:
    """Returns all steps tracked by the manager.

    Args:
      read: If True, forces a read directly from the storage location.
        Otherwise, a cached result can be returned.

    Returns:
      A sequence of steps (integers)
    """
    if read:
      logging.warning(
          '`read` option is deprecated. Use `reload` to read from disk.'
      )
      self._checkpoints.set(self._load_checkpoint_infos())
    return [ckpt.step for ckpt in self._checkpoints]

  def latest_step(self) -> Optional[int]:
    """Returns the latest step saved.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    latest = self._checkpoints.latest()
    return latest.step if latest else None

  def best_step(self) -> Optional[int]:
    """Returns the best step saved, as defined by `options.best_fn`.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    if not self._track_best:
      return self.latest_step()
    if self._checkpoints.empty():
      return None
    _, sorted_checkpoints = self._sort_checkpoints_by_metrics(self._checkpoints)
    if not sorted_checkpoints:
      return None
    return sorted_checkpoints[-1].step

  def reload(self):
    """Reloads internal properties.

    Resets internal cache of checkpoint steps, in case the directory managed
    by this object has been updated externally.
    """
    self._checkpoints.set(self._load_checkpoint_infos())

  def reached_preemption(self, step: int) -> bool:
    """Returns True if a preemption sync point has been reached."""
    return utils.reached_preemption(step)

  def should_save(self, step: int) -> bool:
    """Returns True if a checkpoint should be saved for the current step.

    This depends the previous step and SaveDecisionPolicy.

    Args:
      step: int

    Returns:
      True if the checkpoint should be saved.
    """
    if self._options.read_only:
      logging.warning('%s is read only, save will be skipped', self.directory)
      return False
    last_checkpoint_step = self.latest_step()
    # Ensure current step is between the last step and next step (accounting for
    # save interval).
    if last_checkpoint_step is not None and last_checkpoint_step >= step:
      return False

    is_saving_in_progress = self.is_saving_in_progress()
    reached_preemption = self.reached_preemption(step)
    current_step_info = checkpoint_info.CheckpointInfo(
        step=step, time=datetime.datetime.now(tz=datetime.timezone.utc),
        metrics=None
    )
    context = save_decision_policy_lib.DecisionContext(
        is_saving_in_progress=is_saving_in_progress,
        reached_preemption=reached_preemption,
    )
    return self._save_decision_policy.should_save(
        current_step_info, previous_steps=self._checkpoints, context=context
    )

  def _get_save_directory(
      self,
      step: int,
      directory: epath.Path,
  ) -> epath.Path:
    """Returns the standardized path to a save directory for a single item."""

    return step_lib.build_step_path(directory, self._step_name_format, step)

  def _get_write_step_directory(
      self, step: int, root_dir: epath.Path
  ) -> epath.Path:
    return self._get_save_directory(step, root_dir)

  def _get_read_step_directory(
      self, step: int, root_dir: epath.Path
  ) -> epath.Path:
    if self._options.step_name_format is not None:
      return self._options.step_name_format.find_step(root_dir, step).path
    else:
      return self._get_save_directory(step, root_dir)

  def delete(self, step: int):
    """See superclass documentation.

    Delete can be run asynchronously if
    CheckpointManagerOptions.enable_background_delete is set to True.

    Args:
      step: The step to delete.
    """
    if self._options.read_only:
      logging.warning('%s is read only, delete will be skipped', self.directory)
      return
    if step not in self.all_steps():
      raise ValueError(f'Requested deleting a non-existent step: {step}.')
    self._checkpoint_deleter.delete(step)
    multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'CheckpointManager:deleted_step',
            prefix=self._multiprocessing_options.barrier_sync_key_prefix,
            # suffix=str(step),
        ),
        timeout=multihost.DIRECTORY_DELETION_TIMEOUT,
        processes=self._multiprocessing_options.active_processes,
    )
    self._checkpoints.delete_if(lambda info: info.step == step)

  def _validate_args(
      self,
      items: Optional[Union[Any, Mapping[str, Any]]],
      args: Optional[args_lib.CheckpointArgs],
  ):
    if isinstance(items, args_lib.CheckpointArgs):
      raise ValueError(
          'Found an instance of `CheckpointArgs` provided for `items`. This may'
          ' be due to misuse of the newer API - make sure to specify the'
          ' argument keyword (e.g. `args=args`).'
      )
    if args is not None:
      if not isinstance(args, args_lib.CheckpointArgs):
        raise ValueError(
            f'Expected args of type `CheckpointArgs`; found {type(args)}.'
        )
      if self._default_item.get():
        if isinstance(args, args_lib.Composite):
          raise ValueError(
              'Cannot provide `args` of type `Composite` when dealing with a'
              ' single, unnamed (default) checkpointable object.'
          )
      else:
        if not isinstance(args, args_lib.Composite):
          raise ValueError(
              'Must provide `args` of type `Composite` when dealing with'
              ' multiple checkpointable objects.'
          )

  def save(
      self,
      step: int,
      items: Optional[Union[Any, Mapping[str, Any]]] = None,
      save_kwargs: Optional[Union[SaveParams, Mapping[str, SaveParams]]] = None,
      metrics: Optional[PyTree] = None,
      force: Optional[bool] = False,
      args: Optional[args_lib.CheckpointArgs] = None,
      custom_metadata: dict[str, Any] | None = None,
  ) -> bool:
    """Saves the provided items.

    This method should be called by all hosts - process synchronization and
    actions that need to be performed on only one host are managed internally.

    NOTE: The `items` and `save_kwargs` arguments are deprecated, use `args`
    instead. Make sure to configure `CheckpointManager` with `item_names`.

    `args` should be a subclass of
    `orbax.checkpoint.args.CheckpointArgs`, the specific type of which is used
    to indicate what logic is used to save the object. For a typical, PyTree of
    arrays, use `StandardSave`/`StandardRestore`.

    When constructing the `CheckpointManager`, if no `item_names` were provided,
    it is assumed that we are managing a single object. If `item_names` were
    provided, it is assumed that we are managing multiple objects, and `args`
    must be `orbax.checkpoint.args.CompositeArgs`. See below for details.

    Example::

      # Single item
      mngr = ocp.CheckpointManager(directory)
      mngr.save(step, args=ocp.args.StandardSave(my_train_state))

      # Multiple items
      mngr = ocp.CheckpointManager(directory, item_names=('state', 'meta'))
      mngr.save(step, args=ocp.args.Composite(
          state=ocp.args.StandardSave(my_train_state),
          meta=ocp.args.JsonSave(my_metadata)
      ))

    Args:
      step: current step, int
      items: a savable object, or a dictionary of object name to savable object.
      save_kwargs: save kwargs for a single Checkpointer, or a dictionary of
        object name to kwargs needed by the Checkpointer implementation to save
        the object.
      metrics: a dictionary of metric name (string) to numeric value to be
        tracked along with this checkpoint. Required if `options.best_fn` is
        set. Allows users to specify a metric value to determine which
        checkpoints are best and should be kept (in conjunction with
        `options.max_to_keep`).
      force: if `True`, this method will attempt to save a checkpoint regardless
        of the result of `AbstractCheckpointManager.should_save(step)`. By
        default, `save` will only write a checkpoint to disk when the options
        permit, e.g. when `step` is in `options.save_interval_steps` or
        `options.save_on_steps`. Setting `force=True` will not overwrite
        existing checkpoints.
      args: `CheckpointArgs` which is used to save checkpointable objects with
        the appropriate logic.
      custom_metadata: a dictionary of custom metadata to be written to the
        checkpoint directory via StepMetadata.

    Returns:
      bool indicating whether a save operation was performed.
    Raises:
      ValueError: if `track_best` was indicated but `metrics` is not provided.
      ValueError: directory creation failed.
      ValueError: if an item is provided for which no `Checkpointer` is
      found.
      ValueError: if the checkpoint already exists.
    """
    process_index = multihost.process_index()
    step_stats = step_statistics.SaveStepStatistics()
    step_stats.step = step
    step_stats.checkpoint_manager_blocking_start_time = time.time()
    step_stats.directory = str(self.directory)

    multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'CheckpointManager:save_start',
            prefix=self._multiprocessing_options.barrier_sync_key_prefix,
        ),
        processes=self._multiprocessing_options.active_processes,
        record_event_name=(
            '/jax/orbax/write/checkpoint_start_sync_duration_secs'
        ),
    )

    if items is None and args is None:
      raise ValueError('Must provide `args` for `save`.')
    self._default_item.set_if_none(determine_default_item_mode_from_args(args))
    self._validate_args(items, args)
    if not force and not self.should_save(step):
      return False
    if self.reached_preemption(step):
      logging.info(
          '[process=%s] Saving checkpoint at step %d due to preemption.',
          process_index,
          step,
      )
      step_stats.reached_preemption = True
      step_stats.preemption_received_at = time.time()

    # Wait for ongoing saves to complete. Only applicable if some of the
    # checkpointers are AsyncCheckpointers.
    # Must happen after `should_save` to avoid blocking callers.
    step_stats.wait_for_prev_start_time = time.time()
    self.wait_until_finished()
    step_stats.wait_for_prev_duration_secs = (
        time.time() - step_stats.wait_for_prev_start_time
    )

    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/write/wait_for_prev_duration_secs',
        step_stats.wait_for_prev_duration_secs,
    )

    if step in self.all_steps():
      raise StepAlreadyExistsError(
          f'Checkpoint for step {step} already exists.'
      )

    if items is None:
      items = {}
    if save_kwargs is None:
      save_kwargs = {}
    if self._default_item.get():
      items = {DEFAULT_ITEM_NAME: items}
      save_kwargs = {DEFAULT_ITEM_NAME: save_kwargs}

    if self._track_best and metrics is None:
      logging.warning('Requested `tracked_metric`; did not provide metrics.')

    if args is None:
      args_dict = {}
      for key, item in items.items():
        save_ckpt_arg_cls, _ = _get_args_for_key(
            self._checkpointer.handler,
            key,
        )
        extra_args = save_kwargs[key] if key in save_kwargs else {}
        extra_args = extra_args or {}
        args_dict[key] = save_ckpt_arg_cls(item, **extra_args)  # pylint: disable=too-many-function-args  # pytype: disable=wrong-arg-count
      args = args_lib.Composite(**args_dict)
    else:
      if self._default_item.get():
        args = args_lib.Composite(**{DEFAULT_ITEM_NAME: args})
      else:
        if not isinstance(args, args_lib.Composite):
          raise ValueError(
              f'Expected args of type `Composite`; found {type(args)}.'
          )
        args = typing.cast(args_lib.Composite, args)

    args_dict = dict(args.items())
    if any(
        [item_name in RESERVED_ITEM_NAMES for item_name in args_dict.keys()]
    ):
      raise ValueError(
          'Some provided items have prohibited reserved names:'
          f' {args_dict.keys()}. Reserved names: {RESERVED_ITEM_NAMES}.'
      )
    if metrics is not None and self._track_best:
      args_dict['metrics'] = args_lib.JsonSave(metrics)
    args = args_lib.Composite(**args_dict)

    save_directory = self._get_write_step_directory(step, self.directory)
    # If a folder for the step to save exists and is not finalized, remove the
    # existing folder.
    if step_lib.is_gcs_path(self.directory):
      if (
          utils.is_primary_host(self._multiprocessing_options.primary_host)
          and save_directory.exists()
          and utils.is_tmp_checkpoint(save_directory)
      ):
        logging.warning(
            '[process=%s] Attempting to save on GCS at step %s which has an'
            ' unfinalized checkpoint from previous runs. Removing the'
            ' unfinalized checkpoint before saving.',
            process_index,
            step,
        )
        # make sure to use a synchronous deleter here
        deleter.create_checkpoint_deleter(
            self._multiprocessing_options.primary_host,
            self._directory,
            self._options.todelete_subdir,
            self._step_name_format,
            enable_background_delete=False,  # no background thread
        ).delete(step)
      multihost.sync_global_processes(
          multihost.unique_barrier_key(
              'CheckpointManager:delete_unfinalized_step_gcs',
              prefix=self._multiprocessing_options.barrier_sync_key_prefix,
          ),
          timeout=multihost.DIRECTORY_DELETION_TIMEOUT,
          processes=self._multiprocessing_options.active_processes,
      )
    logging.info(
        '[process=%s] Saving checkpoint at step %d', process_index, step
    )
    step_stats.checkpointer_blocking_start_time = time.time()
    self._checkpointer.save(
        save_directory, args=args, custom_metadata=custom_metadata
    )
    step_stats.checkpointer_blocking_duration_secs = (
        time.time() - step_stats.checkpointer_blocking_start_time
    )

    self._add_checkpoint_info(step, metrics)
    step_stats.get_old_steps_start_time = time.time()
    steps_to_remove = self._get_old_steps_to_remove()
    step_stats.get_old_steps_duration_secs = (
        time.time() - step_stats.get_old_steps_start_time
    )

    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/write/get_old_steps_duration_secs',
        step_stats.get_old_steps_duration_secs,
    )

    self._checkpoints.delete_if(lambda info: info.step in steps_to_remove)
    # Sync needed to ensure that old steps to remove are retrieved before
    # actually deleting them during finalize, since retrieval can involve
    # looking at the directory.
    multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'CheckpointManager:old_steps_to_remove',
            prefix=self._multiprocessing_options.barrier_sync_key_prefix,
            # suffix=str(step),
        ),
        processes=self._multiprocessing_options.active_processes,
    )

    current_thread = threading.current_thread()
    if is_async_checkpointer(self._checkpointer):

      def launch_finalize_thread() -> _FinalizeThread:
        assert not self.is_saving_in_progress(), (
            'Save finalization already in progress for'
            f' step={self._finalize_thread.get_not_none().step()}'
        )
        finalize_thread_name = 'save_finalize'
        logging.info(
            '[process=%s][thread=%s][step=%s] Starting CheckpointManager Save'
            ' Finalize thread=%s',
            process_index,
            current_thread.name,
            step,
            finalize_thread_name,
        )
        finalize_thread = _FinalizeThread(
            step=step,
            name=finalize_thread_name,
            target=self._finalize,
            args=(step, steps_to_remove),
        )
        finalize_thread.start()
        return finalize_thread

      self._finalize_thread.set_from(launch_finalize_thread)

    else:
      self._finalize(step, steps_to_remove)
      logging.info(
          '[process=%s][thread=%s][step=%s] Finished synchronous save.',
          process_index,
          current_thread.name,
          step,
      )

    step_stats.synchronous = not is_async_checkpointer(self._checkpointer)
    step_stats.checkpoint_manager_blocking_duration_secs = (
        time.time() - step_stats.checkpoint_manager_blocking_start_time
    )
    self._logger.log_entry(dataclasses.asdict(step_stats))
    return True

  def _maybe_get_default_item(
      self, composite_result: args_lib.Composite
  ) -> Union[Any, args_lib.Composite]:
    if self._default_item.get():
      if DEFAULT_ITEM_NAME not in composite_result:
        raise ValueError(
            'Unable to retrieve default item. Please ensure that a handler for'
            ' the default item is registered using `handler_registry` when'
            ' initializing the `CheckpointManager`.'
        )
      return composite_result[DEFAULT_ITEM_NAME]
    return composite_result

  def restore(
      self,
      step: Optional[int],
      items: Optional[Union[Any, Mapping[str, Any]]] = None,
      restore_kwargs: Optional[
          Union[RestoreParams, Mapping[str, RestoreParams]]
      ] = None,
      directory: Optional[epath.PathLike] = None,
      args: Optional[args_lib.CheckpointArgs] = None,
  ) -> Union[Any, Mapping[str, Any]]:
    """Restores from the given step and provided items.

    This method should be called by all hosts - process synchronization and
    actions that need to be performed on only one host are managed internally.

    NOTE: The `items` and `restore_kwargs` arguments are deprecated, use `args`
    instead. Make sure to configure `CheckpointManager` with `item_names`.
    See `save` docstring for additional details.

    Example::

      # Single item
      mngr = ocp.CheckpointManager(directory)
      mngr.restore(step, args=ocp.args.StandardRestore(abstract_train_state))

      # Multiple items
      mngr = ocp.CheckpointManager(directory, item_names=('state', 'meta'))
      mngr.restore(step, args=ocp.args.Composite(
          state=ocp.args.StandardRestore(abstract_train_state),
          meta=ocp.args.JsonRestore(),
      ))
      # If it is acceptable to restore without providing additional arguments,
      # and if a save has already been performed, it is ok to do the following:
      mngr.restore(step, args=ocp.args.Composite(state=None, meta=None))
      # If a save has not already been performed, there is no way for Orbax to
      # know how to restore the objects. If a save has already been performed,
      # it remembers the logic used to save the objects.

    Args:
      step: current step, int
      items: a restoreable object, or a dictionary of object name to restorable
        object.
      restore_kwargs: restore kwargs for a single Checkpointer, or a dictionary
        of object name to kwargs needed by the Checkpointer implementation to
        restore the object.
      directory: if provided, uses the given directory rather than the
        `directory` property of this class. Can be used to restore checkpoints
        from an independent location.
      args: `CheckpointArgs` which is used to restore checkpointable objects
        with the appropriate logic.

    Returns:
      If managing a single item, returns a single checkpointable object.
      If managing multiple items, returns ocp.args.Composite, where the keys
      are item names, and values are checkpointable objects.

    Raises:
      FileNotFoundError: If no steps are found in the directory.
    """
    if step is None:
      step = self.latest_step()
      if step is None:
        raise FileNotFoundError(f'No steps found in {self.directory}.')
    directory = directory or self.directory
    directory = epath.Path(directory)
    step_stats = step_statistics.RestoreStepStatistics()
    step_stats.step = step
    step_stats.checkpoint_manager_start_time = time.time()
    step_stats.directory = str(directory)

    self._default_item.set_if_none(
        _determine_default_item_mode_from_directory(
            self._get_read_step_directory(step, directory)
        )
    )
    self._validate_args(items, args)

    if items is None:
      items = {}
    elif self._default_item.get():
      items = {DEFAULT_ITEM_NAME: items}
    if restore_kwargs is None:
      restore_kwargs = {}
    elif self._default_item.get():
      restore_kwargs = {DEFAULT_ITEM_NAME: restore_kwargs}

    if args is None:
      args_dict = {}
      item_keys = set(items.keys()) | set(restore_kwargs.keys())
      for key in item_keys:
        _, restore_ckpt_arg_cls = _get_args_for_key(
            self._checkpointer.handler,
            key,
        )
        item = items[key] if key in items else None
        extra_args = restore_kwargs[key] if key in restore_kwargs else {}
        extra_args = extra_args or {}
        args_dict[key] = restore_ckpt_arg_cls(item, **extra_args)  # pylint: disable=too-many-function-args  # pytype: disable=wrong-arg-count
      args = args_lib.Composite(**args_dict)
    else:
      if self._default_item.get():
        args = args_lib.Composite(**{DEFAULT_ITEM_NAME: args})
      else:
        args = typing.cast(args_lib.Composite, args)

    restore_directory = self._get_read_step_directory(step, directory)
    step_stats.checkpointer_start_time = time.time()
    restored = self._checkpointer.restore(restore_directory, args=args)
    step_stats.checkpointer_duration_secs = (
        time.time() - step_stats.checkpointer_start_time
    )

    step_stats.checkpoint_manager_duration_secs = (
        time.time() - step_stats.checkpoint_manager_start_time
    )
    self._logger.log_entry(dataclasses.asdict(step_stats))

    return self._maybe_get_default_item(restored)

  def item_metadata(
      self, step: int
  ) -> Union[Any, args_lib.Composite, ItemMetadata]:
    """Retrieves metadata for all known items.

    Important note: This method will soon be deprecated in favor of
    `metadata().item_metadata`. Please use that method instead.

    Note that metadata will only be returned for items that can actually be
    interpreted. If an item is present in the checkpoint but not registered
    (using a prior save or restore, or with `handler_registry` at init), the
    item will not be returned.

    Args:
      step: The step to retrieve metadata for.

    Returns:
      Either metadata for the item itself, if in default-item mode, or a
      Composite of metadata for each item.
    """
    return self.metadata(step).item_metadata

  # TODO(b/370812224): Deprecate in favor of StepMetadata.metrics
  def metrics(self, step: int) -> Optional[PyTree]:
    if self._track_best:
      try:
        # Use handler directly, since this happens in a background thread and
        # barriers cannot be used. This usage pattern is not
        # recommended in other contexts.
        return self._metrics_handler.restore(
            self._get_read_step_directory(step, self.directory)
            / METRIC_ITEM_NAME
        )
      except FileNotFoundError as e:
        logging.warning('Missing metrics for step %d', step)
        logging.error(e)
        return None
    else:
      return None

  @property
  def _metrics_handler(self) -> CheckpointHandler:
    return JsonCheckpointHandler(
        filename=METRIC_ITEM_NAME,
        multiprocessing_options=self._multiprocessing_options,
    )

  @property
  def _track_best(self):
    """Returns true if we should track the best checkpoints by given metric."""
    return self._options.best_fn is not None

  def _load_checkpoint_infos(self) -> List[CheckpointInfo]:
    """Loads a list of CheckpointInfo for existing checkpoints.

    If none are present, returns empty list.

    Returns:
      a list of CheckpointInfo, sorted by increasing step.
    """
    if not self.directory.exists():
      raise ValueError(
          f'Checkpoint root directory {self.directory} does not exist.'
      )
    start = time.time()
    step_metadatas = self._step_name_format.find_all(self.directory)

    def build_checkpoint_info(step_metadata):
      return CheckpointInfo(
          step=step_metadata.step,
          time=step_metadata.commit_timestamp,
          metrics=self.metrics(step_metadata.step),
      )

    with concurrent.futures.ThreadPoolExecutor() as executor:
      checkpoint_infos = list(
          executor.map(build_checkpoint_info, step_metadatas)
      )
      checkpoint_infos.sort(key=lambda x: x.step)  # Prefer in-place sort.
      jax.monitoring.record_event_duration_secs(
          '/jax/checkpoint/read/load_all_step_metadata_duration_secs',
          time.time() - start,
      )
      logging.info(
          'Found %d checkpoint steps in %s',
          len(checkpoint_infos),
          self.directory,
      )
      return checkpoint_infos

  def _get_interval_preserved_checkpoints(
      self, checkpoints: checkpoint_info.CheckpointInfos
  ) -> List[CheckpointInfo]:
    """Gets which checkpoints should be kept based on keep_time_interval."""
    if checkpoints.empty():
      return []
    interval_preserved_checkpoints = [checkpoints[0]]
    if self._options.keep_time_interval is not None:
      for info in checkpoints[1:]:
        if info.time >= (
            interval_preserved_checkpoints[-1].time
            + self._options.keep_time_interval
        ):
          interval_preserved_checkpoints.append(info)
    return interval_preserved_checkpoints

  def _add_checkpoint_info(self, step: int, metrics: Optional[PyTree]):
    self._checkpoints.append(
        CheckpointInfo(
            step, datetime.datetime.now(tz=datetime.timezone.utc), metrics
        )
    )

  def _root_metadata_file_path(self, legacy: bool = False) -> epath.Path:
    if not self._metadata_dir.exists():
      raise ValueError(
          f'Root Metadata directory is not initialized: {self._metadata_dir}'
      )
    return checkpoint.root_metadata_file_path(self._metadata_dir, legacy=legacy)

  def _maybe_save_root_metadata(
      self, custom_metadata: Mapping[str, Any] | None
  ):
    """Saves CheckpointManager level metadata, skips if already present."""
    if self._options.save_root_metadata:
      if (
          custom_metadata is not None
          and not self._options.read_only
          and utils.is_primary_host(self._multiprocessing_options.primary_host)
      ):
        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._root_metadata_file_path()
        if not file_path.exists():  # May have been created by a previous run.
          metadata_to_save = self._root_metadata.get()
          if custom_metadata is not None:
            metadata_to_save.custom_metadata = dict(custom_metadata)
          self._blocking_metadata_store.write(
              file_path, serialize_root_metadata(metadata_to_save)
          )

      multihost.sync_global_processes(
          multihost.unique_barrier_key(
              'CheckpointManager:save_root_metadata',
              prefix=self._multiprocessing_options.barrier_sync_key_prefix,
          ),
          processes=self._multiprocessing_options.active_processes,
      )

  def _get_step_metadata(self, step: int) -> StepMetadata:
    infos = [info for info in self._checkpoints if info.step == step]
    if not infos:
      metrics = None
    else:
      if len(infos) > 1:
        logging.warning(
            'Multiple CheckpointInfos found for step %d. Using the first one.',
            step,
        )
      metrics = infos[0].metrics

    step_metadata = self._checkpointer.metadata(
        self._get_read_step_directory(step, self.directory),
    )

    self._default_item.set_if_none(
        _determine_default_item_mode_from_directory(
            self._get_read_step_directory(step, self.directory)
        )
    )
    step_metadata.item_metadata = self._maybe_get_default_item(
        step_metadata.item_metadata
    )

    if metrics is not None:
      validated_metrics = step_metadata_serialization.deserialize(
          {}, metrics=dict(metrics)
      ).metrics
      step_metadata = dataclasses.replace(
          step_metadata,
          metrics=validated_metrics,
      )

    return step_metadata

  def _get_root_metadata(self) -> RootMetadata:
    if self._root_metadata.get().custom_metadata is None:
      if self._metadata_dir.exists():
        file_path = self._root_metadata_file_path()
        if not file_path.exists():
          logging.warning(
              'New root metadata file not found in directory: %s. '
              'Will try to read legacy file.',
              self._metadata_dir,
          )
          file_path = self._root_metadata_file_path(legacy=True)
        serialized_metadata = self._blocking_metadata_store.read(file_path)
        if serialized_metadata is None:
          raise IOError(f'Failed to read metadata from {file_path}')
        self._root_metadata.set(
            root_metadata_serialization.deserialize(serialized_metadata)
        )
      else:
        self._root_metadata.get().custom_metadata = {}
    return self._root_metadata.get()

  @overload
  def metadata(self, step: None = None) -> RootMetadata:
    ...

  @overload
  def metadata(self, step: int) -> StepMetadata:
    ...

  def metadata(self, step: int | None = None) -> RootMetadata | StepMetadata:
    """See superclass documentation."""
    if step is not None:
      return self._get_step_metadata(step)
    return self._get_root_metadata()

  def _sort_checkpoints_by_metrics(
      self, checkpoints: checkpoint_info.CheckpointInfos
  ) -> Tuple[List[CheckpointInfo], List[CheckpointInfo]]:
    """Sorts `checkpoints` in order of increasing metric quality.

    Checkpoints without corresponding metrics set will be at the beginning.

    Args:
      checkpoints: CheckpointInfos to sort.

    Returns:
      Tuple of CheckpointInfo lists:
      (checkpoints_without_metrics, checkpoints_sorted_by_metrics)
    """
    without_metrics = [info for info in checkpoints if info.metrics is None]
    with_metrics = [info for info in checkpoints if info.metrics is not None]

    return without_metrics, sorted(
        with_metrics,
        key=lambda info: self._options.best_fn(info.metrics),
        reverse=(self._options.best_mode == 'min'),
    )

  def _cleanup_tmp_directories(self):
    utils.cleanup_tmp_directories(
        self.directory,
        primary_host=self._multiprocessing_options.primary_host,
        active_processes=self._multiprocessing_options.active_processes,
        barrier_sync_key_prefix=self._multiprocessing_options.barrier_sync_key_prefix,
    )

  def _get_old_steps_to_remove(self) -> List[int]:
    """Returns checkpoints that should be deleted."""
    # Must have set max_to_keep in order to remove any checkpoints.
    if self._options.max_to_keep is None:
      return []
    # Not enough checkpoints accumulated to consider deletion.
    if self._checkpoints.size() <= self._options.max_to_keep:
      return []

    # This isn't a duration but there isn't a general counter that we can use so
    # we abuse a duration metric to count the number of steps examined.
    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/write/old_steps_examined_count',
        self._checkpoints.size(),
    )

    if self._track_best:
      # Best steps (to keep) are at the end, after sorting.
      (
          checkpoints_without_metrics,
          sorted_checkpoints,
      ) = self._sort_checkpoints_by_metrics(self._checkpoints)
    else:
      # checkpoints already sorted by ascending step
      checkpoints_without_metrics = []
      sorted_checkpoints = [info for info in self._checkpoints]

    keep = int(self._options.max_to_keep)
    if self._options.keep_checkpoints_without_metrics:
      maybe_delete = (
          sorted_checkpoints[:-keep] if keep > 0 else sorted_checkpoints
      )
      active_checkpoints = set(
          checkpoints_without_metrics
          + (sorted_checkpoints[-keep:] if keep > 0 else [])
      )
    else:
      all_checkpoints = checkpoints_without_metrics + sorted_checkpoints
      maybe_delete = all_checkpoints[:-keep] if keep > 0 else sorted_checkpoints
      active_checkpoints = set(all_checkpoints[-keep:] if keep > 0 else [])

    interval_preserved_checkpoints = self._get_interval_preserved_checkpoints(
        self._checkpoints
    )
    kept_checkpoints = set()
    for info in maybe_delete:
      if info.is_locked:
        logging.info(
            'Preserving %s: (Reason: checkpoint is locked).',
            info,
        )
        kept_checkpoints.add(info)
        continue
      if (
          self._options.keep_time_interval is not None
          and interval_preserved_checkpoints
      ):
        if info in interval_preserved_checkpoints:
          logging.info(
              'Preserving %s: (Reason: older falling on keep_time_interval).',
              info,
          )
          kept_checkpoints.add(info)
          continue
        elif info.time >= (
            interval_preserved_checkpoints[-1].time
            + self._options.keep_time_interval
        ):
          interval_preserved_checkpoints.append(info)
          logging.info(
              'Preserving %s: (Reason: latest falling on keep_time_interval).',
              info,
          )
          kept_checkpoints.add(info)
          continue
      if (
          self._options.should_keep_fn is not None
          and self._options.should_keep_fn(info.step)
      ):
        logging.info(
            'Preserving %s: (Reason: on should_keep_fn callback).', info
        )
        kept_checkpoints.add(info)
        continue
      if (
          self._options.keep_period is not None
          and info.step % self._options.keep_period == 0
      ):
        logging.info(
            'Preserving %s: (Reason: on keep_period=%s).',
            info,
            self._options.keep_period,
        )
        kept_checkpoints.add(info)
        continue

    kept_checkpoints.update(active_checkpoints)

    steps_to_remove = []
    for info in self._checkpoints:
      if info not in kept_checkpoints:
        reason = 'worse metric' if self._track_best else 'old checkpoint'
        logging.info('Deleting %s: (Reason: %s).', info, reason)
        steps_to_remove.append(info.step)
    return steps_to_remove

  def _wait_for_checkpointers(self):
    if is_async_checkpointer(self._checkpointer):
      self._checkpointer.wait_until_finished()  # pytype: disable=attribute-error

  def wait_until_finished(self):
    """Blocks until any incomplete save operations are completed.

    Note that this method will typically be a no-op if all checkpointers are
    synchronous, since old checkpoints are already cleaned up immediately after
    completing `save`, and there is no background thread to wait for.

    If some checkpointers are of type AsyncCheckpointer, however, this method
    will wait until each of these checkpointers is finished.
    """
    process_index = multihost.process_index()
    current_thread = threading.current_thread()

    if self._finalize_thread.map(
        lambda t: t is None or (not t.is_alive() and t.exception is None)
    ):
      logging.info(
          '[process=%s][thread=%s][wait_until_finished] No Save Finalize'
          ' thread to wait for. Returning.',
          process_index,
          current_thread.name,
      )
      return

    step = self._finalize_thread.get_not_none().step()
    finalize_thread_name = self._finalize_thread.get_not_none().name
    try:
      logging.info(
          '[process=%s][thread=%s][step=%s][wait_until_finished] Waiting for'
          ' Save Finalize thread (%s) to complete.',
          process_index,
          current_thread.name,
          step,
          finalize_thread_name,
      )
      # Let all threads join and wait for the finalize thread to complete.
      # Don't call join() with a lock otherwise we will end up serializing the
      # access to the finalize thread.
      self._finalize_thread.get_not_none().join()
      logging.info(
          '[process=%s][thread=%s][step=%s][wait_until_finished] Done'
          ' waiting for Save Finalize thread (%s) running at step=%d.',
          process_index,
          current_thread.name,
          step,
          finalize_thread_name,
          step,
      )
    except BaseException:  # pylint:disable=broad-exception-caught
      logging.exception(
          '[process=%s][thread=%s][step=%s][wait_until_finished] Save'
          ' Finalize thread (%s) failed.',
          process_index,
          current_thread.name,
          step,
          finalize_thread_name,
      )
      self._checkpoints.delete_if(lambda info: info.step == step)
      raise

  def is_saving_in_progress(self) -> bool:
    """Returns whether a checkpoint save is in progress."""
    return self._finalize_thread.map(lambda t: t is not None and t.is_alive())

  def check_for_errors(self):
    """Checks for any outstanding errors in completed asynchronous save operations.

    Delegates to underlying Checkpointer.
    """
    if is_async_checkpointer(self._checkpointer):
      self._checkpointer.check_for_errors()  # pytype: disable=attribute-error

  def _finalize_checkpoint(self, step: int):
    """Executes final actions just before the checkpoint write completes.

    * Logs error if any.
    * Records duration saved due to preemption if any.

    Args:
      step: finalized checkpoint step.
    """
    if utils.is_primary_host(self._multiprocessing_options.primary_host):
      try:
        self.check_for_errors()
      except Exception:  # pylint: disable=broad-except
        logging.exception(
            (
                '[step=%s] Checkpointer failed: one or more items may not be'
                ' finalized. Skipping finalization of step checkpoint.'
            ),
            step,
        )
        return None
      # If at a preemption step, record the time since the previous checkpoint.
      # This represents training time that would otherwise have been wasted.
      # If another checkpoint has not been previously saved, measures the time
      # since program start.
      if self.reached_preemption(step):
        if self._checkpoints.size() > 1:
          previous_time = self._checkpoints[-2].time
        else:
          previous_time = _INIT_TIME
        assert not self._checkpoints.empty()
        duration = self._checkpoints[-1].time - previous_time
        jax.monitoring.record_event_duration_secs(
            '/jax/checkpoint/write/preempt/duration_saved_secs',
            duration.total_seconds(),
        )

  def _finalize(self, step: int, steps_to_remove: List[int]):
    """Finalizes individual items and starts garbage collection."""
    process_index = multihost.process_index()
    current_thread = threading.current_thread()
    self._non_blocking_metadata_store.wait_until_finished()
    self._wait_for_checkpointers()
    # If an error is encountered while waiting for commit futures to complete,
    # we will not proceed past this point.
    self._finalize_checkpoint(step)
    remove_steps_start_time = time.time()
    self._checkpoint_deleter.delete_steps(steps_to_remove)
    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/write/remove_steps_duration_secs',
        time.time() - remove_steps_start_time,
    )
    logging.info(
        '[process=%s][thread=%s][step=%s] CheckpointManager Save Finalize is'
        ' syncing with other hosts...',
        process_index,
        current_thread.name,
        step,
    )
    barrier_sync_fn = self._create_thread_safe_barrier_sync_fn()
    barrier_sync_fn(
        multihost.unique_barrier_key(
            'CheckpointManager:finalize',
            prefix=self._multiprocessing_options.barrier_sync_key_prefix,
            suffix=str(step),
        )
    )
    logging.info(
        '[process=%s][thread=%s][step=%s] CheckpointManager Save Finalize is'
        ' done on all hosts.',
        process_index,
        current_thread.name,
        step,
    )

  def close(self):
    """Waits for outstanding operations to finish and closes internal objects."""
    self.wait_until_finished()
    self._checkpointer.close()
    # Call after checkpointer.close().
    self._non_blocking_metadata_store.close()
    self._blocking_metadata_store.close()
    self._checkpoint_deleter.close()

  def __contextmanager__(
      self,
  ) -> Iterable[Self]:
    try:
      yield self
    finally:
      self.close()
