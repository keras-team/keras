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

"""Orbax context for customized checkpointing."""

from __future__ import annotations

from collections.abc import Iterable
import contextvars
import dataclasses

from etils import epy
from orbax.checkpoint.experimental.v1._src.context import options as options_lib


# Each Thread will have its own copy of `Context` object.
# Task and groups will have their own copy of `Context` object.
_CONTEXT: contextvars.ContextVar[Context] = contextvars.ContextVar(
    "orbax_context", default=None
)


def get_context(default: Context | None = None) -> Context:
  """Returns the current `Context` or `default` or `Context()` if not set."""
  default = default or Context()
  return _CONTEXT.get(default)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Context(epy.ContextManager):
  """Context for customized checkpointing.

  Usage example::

    with ocp.Context(...):
      ocp.save_pytree(...)

  NOTE: The context is not shared across threads. In other words, the whole
  context block must be executed in the same thread. Following example will
  not work as expected::

    executor = ThreadPoolExecutor()
    with ocp.Context(...):  # Thread #1 creates Context A.
      executor.submit(ocp.save_pytree, ...)  # Thread #2 sees "default" Context.


  Attributes:
    pytree_options: Options for PyTree checkpointing.
    array_options: Options for saving and loading array (and array-like
      objects).
    async_options: Options for controlling asynchronous behavior.
    multiprocessing_options: Options for multiprocessing behavior.
    file_options: Options for working with the file system.
  """

  pytree_options: options_lib.PyTreeOptions = dataclasses.field(
      default_factory=options_lib.PyTreeOptions
  )
  array_options: options_lib.ArrayOptions = dataclasses.field(
      default_factory=options_lib.ArrayOptions
  )
  async_options: options_lib.AsyncOptions = dataclasses.field(
      default_factory=options_lib.AsyncOptions
  )
  multiprocessing_options: options_lib.MultiprocessingOptions = (
      dataclasses.field(default_factory=options_lib.MultiprocessingOptions)
  )
  file_options: options_lib.FileOptions = dataclasses.field(
      default_factory=options_lib.FileOptions
  )

  def __contextmanager__(self) -> Iterable[Context]:
    token = _CONTEXT.set(self)
    try:
      yield self
    finally:
      _CONTEXT.reset(token)
