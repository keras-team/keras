# Copyright 2024 The Treescope Authors.
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

"""Helpers for scoped contextual value providers."""
from __future__ import annotations

import contextlib
import dataclasses
from typing import Generic, TypeVar

from treescope._internal import docs_util

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class _NoScopedValue:
  """Sentinel value for when no scoped value is set."""

  pass


@dataclasses.dataclass(init=False)
class ContextualValue(Generic[T]):
  """A global value which can be modified in a scoped context.

  Mutable global values can be difficult to reason about, but can also be very
  convenient for reducing boilerplate and focusing on the important parts if
  they are used carefully. Moreover, it's common to only want to change a global
  value within a particular delimited scope, without affecting anything beyond
  that scope.

  This class manages a global value that can be read anywhere, and modified
  either globally or within a delimited scope. Local values always take
  precedence over global values.

  When used within a function, it is usually recommended to use scoped updates.
  If you have many values to set, or you want to set them conditionally,
  consider using a `contextlib.ExitStack`, e.g.

  ::

      with contextlib.ExitStack() as stack:
        stack.enter_context(contextual_one.set_scoped(True))
        if foo:
          stack.enter_context(contextual_two.set_scoped(some_value))
        # logic that uses those values
      # all contexts are exited here

  or just

  ::

      stack = contextlib.ExitStack()
      stack.enter_context(contextual_one.set_scoped(True))
      if foo:
        stack.enter_context(contextual_two.set_scoped(some_value))
      # ... do something ...
      # ... then eventually:
      stack.close()

  Attributes:
    __module__: The module where this contextual value is defined.
    __qualname__: The fully-qualified name of the contextual value within its
      module.
    _raw_global_value: The current value at the global level. Should not be used
      directly.
    _raw_scoped_value: The current value at the scoped level. Should not be used
      directly.
  """

  __module__: str | None
  __qualname__: str | None
  _raw_global_value: T
  _raw_scoped_value: T | _NoScopedValue

  def __init__(
      self,
      initial_value: T,
      module: str | None = None,
      qualname: str | None = None,
  ):
    """Creates a contextual value.

    Args:
      initial_value: A value to use outside of `set_scoped` contexts.
      module: The module where this contextual value is defined. Usually this
        will be `__name__` (for whichever model it was defined in).
      qualname: The fully-qualified name of the contextual value within its
        module.
    """
    self.__module__ = module
    self.__qualname__ = qualname
    self._raw_global_value = initial_value
    self._raw_scoped_value = _NoScopedValue()

  def get(self) -> T:
    """Retrieves the current value."""
    if isinstance(self._raw_scoped_value, _NoScopedValue):
      return self._raw_global_value
    else:
      return self._raw_scoped_value

  @contextlib.contextmanager
  def set_scoped(self, new_value: T):
    # pylint: disable=g-doc-return-or-yield
    """Returns a context manager in which the value will be modified.

    This allows scoped modifications to the value, using something like

    ::

      contextual = ContextualValue(10)

      with contextual.set_scoped(3):
        v1 = contextual.get()  # v1 is 3

      v2 = contextual.get()  # v2 is 10

    If you want to conditionally set the value, consider using
    `contextlib.ExitStack`:

    ::

      with contextlib.ExitStack() as exit_stack:
        if condition:
          exit_stack.enter_context(contextual.set_scoped(3))

        # use contextual.get() here

    Args:
      new_value: The new value to use in this context.

    Returns:
      A context manager in which ``new_value`` is set.
    """
    # pylint: enable=g-doc-return-or-yield
    old_value = self._raw_scoped_value
    self._raw_scoped_value = new_value
    try:
      yield
    finally:
      assert self._raw_scoped_value is new_value
      self._raw_scoped_value = old_value

  def set_globally(self, new_value: T) -> None:
    """Sets the value at the global level.

    Updates at the global level will be overridden by any scoped updates.

    Args:
      new_value: The new value to use.
    """
    self._raw_global_value = new_value

  @docs_util.skip_automatic_documentation
  def set_interactive(self, new_value: T) -> None:
    """Alias for `set_globally`, for consistency with older Penzai API.

    Args:
      new_value: The new value to use.
    """
    self.set_globally(new_value)
