# Copyright 2024 The etils Authors.
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

"""Contextmanager utils."""

from __future__ import annotations

import abc
from collections.abc import Iterable
import contextlib
import typing
from typing import Any, Generic, TypeVar

_T = TypeVar('_T')

# TODO(epot): Support
# * Per-instance (done)
# * Inheritance with `super()`
# * Multi-thread
# * Covariable
# * Re-entry
# * Immutable classes


class ContextManager(abc.ABC, Generic[_T]):
  """ContextManager allows to define contextmanager class using yield-syntax.

  Example:

  ```python
  class A(epy.ContextManager):

    def __contextmanager__(self) -> Iterable[A]:
      yield self


  with A() as a:
    pass
  ```

  One the code is more mature, this could be merged to `contextlib` directly.
  https://discuss.python.org/t/yield-based-contextmanager-for-classes/8453

  """

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)

    # Check whether the class is already wrapped in a CM
    if not hasattr(cls.__contextmanager__, '_cm_added'):
      cls.__contextmanager__ = contextlib.contextmanager(cls.__contextmanager__)
      cls.__contextmanager__._cm_added = (
          True  # pylint: disable=protected-access
      )

  @abc.abstractmethod
  def __contextmanager__(self) -> Iterable[_T]:
    pass

  __contextmanager__._cm_added = True  # pylint: disable=protected-access

  def __enter__(self) -> _T:
    # object.__setattr__ to support frozen dataclasses
    object.__setattr__(self, '_epy_cm', self.__contextmanager__())
    return self._epy_cm.__enter__()  # pytype: disable=attribute-error

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    return self._epy_cm.__exit__(exc_type, exc_value, traceback)  # pytype: disable=attribute-error


# Should use `contextlib.nested` instead if outputs are required?
class ExitStack(ContextManager[contextlib.ExitStack]):
  """Like `contextlib.ExitStack` but allows to set contextmanagers at init time.

  ```python
  with epy.ExitStack([A(), B()]) as stack:
    pass
  ```

  Outputs are not available.

  See:
  https://discuss.python.org/t/add-init-self-contexts-iterable-contextmanager-to-contextlib-exitstack/19906
  """

  def __init__(self, contexts: Iterable[typing.ContextManager[Any]]):
    self._contexts = list(contexts)

  def __contextmanager__(self) -> Iterable[contextlib.ExitStack]:
    with contextlib.ExitStack() as stack:
      for context in self._contexts:
        stack.enter_context(context)
      yield stack
