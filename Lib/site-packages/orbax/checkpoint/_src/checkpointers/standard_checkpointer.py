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

"""Shorthand for `AsyncCheckpointer(StandardCheckpointHandler())`."""

from typing import Any, Optional, Type
from etils import epath
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.checkpointers import async_checkpointer
from orbax.checkpoint._src.handlers import standard_checkpoint_handler
from orbax.checkpoint._src.metadata import checkpoint
from orbax.checkpoint._src.path import atomicity_types


StandardCheckpointHandler = (
    standard_checkpoint_handler.StandardCheckpointHandler
)
StandardSave = standard_checkpoint_handler.StandardSaveArgs
StandardRestore = standard_checkpoint_handler.StandardRestoreArgs
PyTree = Any


class StandardCheckpointer(async_checkpointer.AsyncCheckpointer):
  """Shorthand class.

  Note that this `Checkpointer` saves asynchronously.

  Initialization::

    # Instead of: 
    with AsyncCheckpointer(StandardCheckpointHandler()) as ckptr:
      ...
    # We can use:
    with StandardCheckpointer() as ckptr:
      ...

  This class is convenient because `ocp.args` does not need to specified when
  saving and restoring. Saving/restoring::

    # Instead of:
    with AsyncCheckpointer(StandardCheckpointHandler()) as ckptr:
      ckptr.save(directory, args=StandardSave(state, save_args))
      ckptr.restore(directory, args=StandardRestore(abstract_target))
    # We can use:
    with StandardCheckpointer() as ckptr:
      ckptr.save(directory, state, save_args=save_args)
      ckptr.restore(directory, abstract_target)
  """

  def __init__(
      self,
      *,
      async_options: options_lib.AsyncOptions = options_lib.AsyncOptions(),
      multiprocessing_options: options_lib.MultiprocessingOptions = options_lib.MultiprocessingOptions(),
      file_options: options_lib.FileOptions = options_lib.FileOptions(),
      checkpoint_metadata_store: Optional[checkpoint.MetadataStore] = None,
      temporary_path_class: Optional[
          Type[atomicity_types.TemporaryPath]
      ] = None,
      **kwargs,
  ):
    """Constructor.

    Args:
      async_options: See superclass documentation.
      multiprocessing_options: See superclass documentation.
      file_options: See superclass documentation.
      checkpoint_metadata_store: See superclass documentation.
      temporary_path_class: See superclass documentation.
      **kwargs: Additional init args passed to StandardCHeckpointHandler. See
        orbax.checkpoint.standard_checkpoint_handler.StandardCheckpointHandler.
    """
    super().__init__(
        standard_checkpoint_handler.StandardCheckpointHandler(
            multiprocessing_options=multiprocessing_options,
            **kwargs,
        ),
        async_options=async_options,
        multiprocessing_options=multiprocessing_options,
        file_options=file_options,
        checkpoint_metadata_store=checkpoint_metadata_store,
        temporary_path_class=temporary_path_class,
    )

  def save(
      self,
      directory: epath.PathLike,
      state: PyTree,
      *,
      save_args: Optional[PyTree] = None,
      force: bool = False,
      custom_metadata: dict[str, Any] | None = None,
  ):
    """Saves a checkpoint asynchronously (does not block).

    Args:
      directory: Path where the checkpoint will be saved.
      state: a PyTree of arrays to be saved.
      save_args: a PyTree with the same structure of `item`, which consists of
        `ocp.SaveArgs` objects as values. `None` can be used for values where no
        `SaveArgs` are specified. Only necessary for fine-grained customization
        of saving behavior for individual parameters.
      force: See superclass documentation.
      custom_metadata: a dictionary of custom metadata to be written to the
        checkpoint directory via StepMetadata.
    """
    super().save(
        directory,
        args=StandardSave(state, save_args),
        force=force,
        custom_metadata=custom_metadata,
    )

  def restore(
      self,
      directory: epath.PathLike,
      target: Optional[PyTree] = None,
      *,
      strict: bool = True,
  ) -> Any:
    """Restores a checkpoint.

    Args:
      directory: Path where the checkpoint will be saved.
      target: a PyTree representing the expected structure of the checkpoint.
        Values may be either real array or scalar values, or they may be
        jax.ShapeDtypeStruct. If real values are provided, that value will be
        restored as the given type, with the given properties. If
        jax.ShapeDtypeStruct is provided, the value will be restored as
        np.ndarray, unless `sharding` is specified. If `item` is a custom PyTree
        class, the tree will be restored with the same structure as provided. If
        not provided, restores as a serialized nested dict representation of the
        custom class.
      strict: if False, restoration allows silent truncating/padding of arrays
        if the stored array shape does not match the target shape. Otherwise,
        raises an error.

    Returns:
      The restored checkpoint.
    """
    return super().restore(
        directory, args=StandardRestore(target, strict=strict)
    )
