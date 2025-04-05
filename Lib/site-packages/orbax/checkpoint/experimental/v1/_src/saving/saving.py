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

"""Defines free-function interface for saving."""

# pylint: disable=protected-access

import threading

from etils import epath
import orbax.checkpoint as ocp
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


def save_pytree(
    directory: path_types.PathLike,
    pytree: tree_types.PyTreeOf[tree_types.LeafType],
    *,
    force: bool = False,
    custom_metadata: ocp.tree.JsonType | None = None,
):
  """Saves a PyTree.

  The operation blocks until complete. For improved performance, consider using
  `save_async` instead.

  Args:
    directory: The directory to save the checkpoint to.
    pytree: The PyTree to save. This may be any JAX PyTree (including custom
      objects registered as PyTrees) consisting of supported leaf types. Default
      supported leaf types include `jax.Array`, `np.ndarray`, simple types like
      `int`, `float`, `str`, and empty nodes. Support for custom leaves is also
      possible by implementing a `LeafTypeHandler`.
    force: Whether to allow the save to proceed even if it would fully overwrite
      an existing checkpoint.
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.
  """
  context = context_lib.get_context()

  handler_registry = ocp.handlers.create_default_handler_registry(
      pytree=pytree_handler.create_v0_handler(context)
  )
  ckptr = ocp.Checkpointer(
      ocp.CompositeCheckpointHandler(handler_registry=handler_registry)
  )
  args = ocp.args.Composite(
      pytree=pytree_handler.create_v0_save_args(context, pytree)
  )
  ckptr.save(directory, args=args, force=force, custom_metadata=custom_metadata)


class _SaveResponse(async_types.AsyncResponse[None]):
  """An `AsyncResponse` representing the result of `save_pytree_async`.

  TODO(cpgaffney): Note that a memory leak is possible if the user does not
  call `result`.
  """

  def __init__(self, checkpointer: ocp.AsyncCheckpointer):
    self._checkpointer = checkpointer
    self._thread = threading.Thread(target=self._wait_for_save)
    self._thread.start()

  def _wait_for_save(self):
    self._checkpointer.wait_until_finished()

  def result(self, timeout: float | None = None) -> None:
    self._thread.join()
    self._checkpointer.close()


def save_pytree_async(
    directory: path_types.PathLike,
    pytree: tree_types.PyTreeOf[tree_types.LeafType],
    *,
    force: bool = False,
    custom_metadata: ocp.tree.JsonType | None = None,
) -> async_types.AsyncResponse[None]:
  """Saves a PyTree asynchronously.

  Unlike `save`, this function returns immediately after the save operation is
  scheduled (except for certain operations, like device-to-host copying of
  on-device arrays, which must happen on the main thread). Further writing
  operations continue in a background thread. An `AsyncResponse` is returned
  that can be used to block until the save is complete (using
  `response.result()`). Make sure to wait for completion before attempting to
  load the checkpoint or exiting the program.

  Args:
    directory: The directory to save the checkpoint to.
    pytree: The PyTree to save. This may be any JAX PyTree (including custom
      objects registered as PyTrees) consisting of supported leaf types. Default
      supported leaf types include `jax.Array`, `np.ndarray`, simple types like
      `int`, `float`, `str`, and empty nodes. Support for custom leaves is also
      possible by implementing a `LeafTypeHandler`.
    force: Whether to allow the save to proceed even if it would fully overwrite
      an existing checkpoint.
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.

  Returns:
    An `AsyncResponse` that can be used to block until the save is complete.
    Blocking can be done using `response.result()`, which returns `None`.
  """
  context = context_lib.get_context()

  handler_registry = ocp.handlers.create_default_handler_registry(
      pytree=pytree_handler.create_v0_handler(context)
  )
  ckptr = ocp.AsyncCheckpointer(
      ocp.CompositeCheckpointHandler(handler_registry=handler_registry)
  )
  args = ocp.args.Composite(
      pytree=pytree_handler.create_v0_save_args(context, pytree)
  )
  directory = epath.Path(directory)
  ckptr.save(directory, args=args, force=force, custom_metadata=custom_metadata)
  return _SaveResponse(ckptr)
