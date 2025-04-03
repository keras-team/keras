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

"""`etils` internal utils."""

from collections.abc import Callable
import contextlib
import functools
from typing import Any, Iterator, TypeVar

from etils.epy import reraise_utils

_FnT = TypeVar('_FnT')


@contextlib.contextmanager
def check_missing_deps() -> Iterator[None]:
  """Raise a better error message in case of `ImportError`.

  Usage:

  ```python
  from etils.epy import _internal

  with _internal.check_missing_deps():
    # pylint: disable=g-import-not-at-top
    import xyz
    # pylint: enable=g-import-not-at-top
  ```

  Yields:
    None
  """
  try:
    yield
  except ImportError as e:
    reraise_utils.reraise(
        e,
        suffix=(
            '\nEach etils sub-modules require deps to be installed separately '
            '(e.g. `from etils import ecolab` -> `pip install etils[ecolab]`)'
        ),
    )


def unwrap_on_reload(fn: _FnT) -> _FnT:
  """Unwrap the function to support colab module reload."""
  return getattr(fn, '__original_fn__', fn)


def wraps_with_reload(fn: Callable[..., Any]) -> Callable[[_FnT], _FnT]:
  """Wrap the function to support colab module reload."""

  def decorator(fn_to_wrap):
    fn_to_wrap = functools.wraps(fn)(fn_to_wrap)
    fn_to_wrap.__original_fn__ = fn
    return fn_to_wrap

  return decorator
