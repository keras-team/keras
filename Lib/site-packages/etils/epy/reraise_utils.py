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

"""Utils to reraise."""

from __future__ import annotations

import contextlib
from typing import Callable, Iterator, NoReturn, Optional, Union

# TODO(epot):
# * Better stacktrace:
#   * Do not set `__suppress_context__` if the original exception has a
#     context (careful about nested exceptions).
#   * Do not have the reraise appear in the stacktrace ?
# * Design `epy.ReraiseMsg()` for better representation when nesting errors
#   (error in `x/y/z/`, rather than `error in x/: error in y/: error in z/:`)
# * Should automatically detect the current exception ?
# * Should unify `maybe_reraise` & `reraise` ?

# prefix/suffix can be:
# * A string
# * A lazy string (only computed if exception is reraised)
_Str = Union[str, Callable[[], str]]


def reraise(
    e: Exception,
    prefix: Optional[_Str] = None,
    suffix: Optional[_Str] = None,
) -> NoReturn:
  """Reraise an exception with an additional message.

  Benefit: Contrary to `raise ... from ...` and
  `raise Exception().with_traceback(tb)`, this function will:

  * Keep the original exception type, attributes,...
  * Avoid multi-nested `During handling of the above exception, another
    exception occurred`. Only the single original stacktrace is displayed.

  This result in cleaner and more compact error messages.

  Usage:

  ```
  try:
    fn(x)
  except Exception as e:
    epy.reraise(e, prefix=f'Error for {x}: ')
  ```

  Args:
    e: Exception to reraise
    prefix: Prefix to add to the exception message.
    suffix: Suffix to add to the exception message.
  """
  # TODO(epot): Mutate the `e.__traceback__` when not in IPython.
  # Hide the function from the traceback. Is supported by Pytest and IPython 7
  __tracebackhide__ = True  # pylint: disable=unused-variable,invalid-name

  # Lazy-evaluate functions
  prefix = prefix() if callable(prefix) else prefix
  suffix = suffix() if callable(suffix) else suffix
  prefix = prefix or ''
  suffix = '\n' + suffix if suffix else ''
  msg = f'{prefix}{e}{suffix}'

  # Dynamically create an exception for:
  # * Compatibility with caller core (e.g. `except OriginalError`)

  class WrappedException(type(e)):
    """Exception proxy with additional message."""

    def __init__(self, msg):
      # We explicitly bypass super() as the `type(e).__init__` constructor
      # might have special kwargs
      Exception.__init__(self, msg)  # pylint: disable=non-parent-init-called

    def __getattr__(self, name: str):
      # Capture `e` through closure. We do not pass e through __init__
      # to bypass `Exception.__new__` magic which add `__str__` artifacts.
      return getattr(e, name)

    # The wrapped exception might have overwritten `__str__` & cie, so
    # use the base exception ones.
    __repr__ = BaseException.__repr__
    __str__ = BaseException.__str__

  WrappedException.__name__ = type(e).__name__
  WrappedException.__qualname__ = type(e).__qualname__
  WrappedException.__module__ = type(e).__module__
  new_exception = WrappedException(msg)

  # Propagate the exception:
  # * `with_traceback` will propagate the original stacktrace
  # * `from e.__cause__` will:
  #   * Propagate the original `__cause__` (likely `None`)
  #   * Set `__suppress_context__` to True, so `__context__` isn't displayed
  #     This avoid multiple `During handling of the above exception, another
  #     exception occurred:` messages when nesting `reraise`
  raise new_exception.with_traceback(e.__traceback__) from e.__cause__


@contextlib.contextmanager
def maybe_reraise(
    prefix: Optional[_Str] = None,
    suffix: Optional[_Str] = None,
) -> Iterator[None]:
  """Context manager which reraise exceptions with an additional message.

  Benefit: Contrary to `raise ... from ...` and
  `raise Exception().with_traceback(tb)`, this function will:

  * Keep the original exception type, attributes,...
  * Avoid multi-nested `During handling of the above exception, another
    exception occurred`. Only the single original stacktrace is displayed.

  This result in cleaner and more compact error messages.

  Usage:

  ```python
  with epy.maybe_reraise(prefix=f'Error for {x}:'):
    fn(x)
  ```

  Args:
    prefix: Prefix to add to the exception message. Can be a function for
      lazy-evaluation.
    suffix: Suffix to add to the exception message. Can be a function for
      lazy-evaluation.

  Yields:
    None
  """
  # Hide the function from the traceback. Is supported by Pytest and IPython 7
  __tracebackhide__ = True  # pylint: disable=unused-variable,invalid-name

  try:
    yield
  except Exception as e:  # pylint: disable=broad-except
    reraise(e, prefix=prefix, suffix=suffix)
