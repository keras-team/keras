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

"""Pytest utils."""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Any, Iterator

import pytest


# ==== Hermetic tests ===

_SKIP_NON_HERMETIC = False

# Non hermetic tests are explicitly marked and skipped if `_SKIP_NON_HERMETIC`
# is True.
non_hermetic = pytest.mark.skipif(
    _SKIP_NON_HERMETIC,
    reason='Non-hermetic test skipped.',
)

# ==== Subtests ===

_curr_context = None


@dataclasses.dataclass
class _SubtestContext:
  """Context of a test using `subtests` fixture.

  Attributes:
    subtests: Reference to the original `subtests` fixture output
    names: Stack of current nested `subtests.test`
  """

  subtests: Any
  names: list[str] = dataclasses.field(default_factory=list)


@contextlib.contextmanager
def subtest(name: str) -> Iterator[None]:
  """Contextmanager for a new subtest. To use with `with_subtests` fixture."""
  if not _curr_context:
    raise AssertionError(
        '`epy.testing.subtest` can only be called inside a '
        '`with_subtests` context.'
    )
  name = str(name)
  _curr_context.names.append(name)
  subtest_name = '/'.join(_curr_context.names)
  try:
    with _curr_context.subtests.test(msg=subtest_name):
      yield
  finally:
    out_name = _curr_context.names.pop()
    assert out_name == name  # Sanity check


@pytest.fixture
def with_subtests(subtests) -> Iterator[None]:
  """Fixture which activate subtests for global usage.

  This fixture is a small wrapper around `subtests` pytest extension fixing
  2 issues:

  * Global usage: https://github.com/pytest-dev/pytest-subtests/issues/44
  * Nested report: https://github.com/pytest-dev/pytest-subtests/issues/45

  Usage:

  ```python
  with_subtests = epy.testing.with_subtests  # Required to register the fixture

  @pytest.mark.usefixtures('with_subtests')
  def my_test():
    with epy.testing.subtest('a'):
      with epy.testing.subtest('b'):
        assert False
  ```

  Args:
    subtests: Subtest fixture

  Yields:
    None
  """
  global _curr_context
  if _curr_context is not None:
    raise AssertionError('Conflicting `subtests` context.')
  new_context = _SubtestContext(subtests=subtests)
  try:
    _curr_context = new_context
    yield
  finally:
    _curr_context = None
