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

"""Reusable threading constructs."""

from __future__ import annotations

from collections.abc import Callable
import threading
import time
from typing import Generic, TypeVar

_T = TypeVar('_T')
_R = TypeVar('_R')


class TimeoutRLock:
  """An RLock that can time out when used as a context manager.

  Unforunately, `threading.RLock` does not support a timeout when used as a
  context manager.

  NOTE: Always blocks until the timeout.

  NOTE: Must be used as a context manager.

  Usage:
    ```
    # Blocks until the timeout of 10 seconds.
    with _TimeoutRLock(timeout=10.0):
      ...
    ```
  """

  def __init__(self, timeout: float = 5.0):
    """Initializes the RLock with the given timeout in seconds."""
    self._lock = threading.RLock()
    self._timeout = timeout
    self._acquired = False
    self._start = time.time()

  def __enter__(self) -> TimeoutRLock:
    self._start = time.time()
    self._acquired = self._lock.acquire(timeout=self._timeout)
    if self._acquired:
      return self
    raise TimeoutError(
        f'Thread {threading.current_thread().name} failed to acquire reentrant'
        f' lock. timeout={self._timeout}s,'
        f' time_elapsed={time.time() - self._start}s: {self._lock}'
    )

  def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
    if self._acquired:
      self._lock.release()
    return False  # Raise the exception if any.


class OptionalRef(Generic[_T]):
  """A thread-safe reference to a value that may be None."""

  def __init__(self, value: _T | None = None):
    """Initializes the thread-safe reference with `value`."""
    self._lock = threading.RLock()
    with self._lock:
      self._value = value

  def set(self, value: _T | None) -> OptionalRef[_T]:
    """Sets `value` and returns self."""
    with self._lock:
      self._value = value
    return self

  def set_from(self, get_value: Callable[[], _T | None]) -> OptionalRef[_T]:
    """Sets `value` from `get_value` and returns self."""
    with self._lock:
      self._value = get_value()
    return self

  def set_if_none(self, value: _T) -> OptionalRef[_T]:
    """Sets `value` if current value is None and returns self."""
    with self._lock:
      if self._value is None:
        self._value = value
    return self

  def get(self) -> _T | None:
    """Returns the value."""
    with self._lock:
      return self._value

  def get_not_none(self) -> _T:
    """Returns the value, or raises assertion exception if the value is None."""
    with self._lock:
      assert self._value is not None
      return self._value

  def map(self, func: Callable[[_T], _R]) -> _R:
    """Applies `func` to the value and returns the result."""
    with self._lock:
      return func(self._value)


class Ref(Generic[_T]):
  """A thread-safe reference to a value that should never be None."""

  def __init__(self, value: _T):
    """Initializes the thread-safe reference with `value`."""
    self._lock = threading.RLock()
    with self._lock:
      self._value = value

  def set(self, value: _T) -> Ref[_T]:
    """Sets `value` and returns self."""
    with self._lock:
      self._value = value
    return self

  def get(self) -> _T:
    """Returns the value."""
    with self._lock:
      return self._value
