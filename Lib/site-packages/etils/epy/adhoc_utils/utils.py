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

"""Adhoc utils."""

from collections.abc import Iterator
import contextlib
import os

from etils.epy import py_utils


def normalize_restrict_and_reload(
    restrict: py_utils.StrOrStrList,
    reload: py_utils.StrOrStrList,
    *,
    restrict_reload: bool = True,
) -> tuple[list[str], list[str]]:
  """Normalize restrict and reload."""
  if isinstance(reload, bool):
    raise ValueError(
        f"reload={reload} is deprecated. Instead use reload='my_module'"
    )

  restrict = py_utils.normalize_str_to_list(restrict)
  reload = py_utils.normalize_str_to_list(reload)

  # Restrict also include the reload
  # This allow to call `adhoc(reload='visu3d')` without explicitly set restrict
  if restrict_reload:
    restrict = _remove_duplicate(restrict + reload)

  return restrict, reload


def _remove_duplicate(x: list[str]) -> list[str]:
  return list(dict.fromkeys(x))


@contextlib.contextmanager
def skip_disable_tf2() -> Iterator[None]:
  """Set environment variable."""
  # Allow TF to conditionally detect if they are running inside adhoc (fix
  # b/322775800)
  prev_value = os.environ.get('SKIP_DISABLE_TF2')
  try:
    os.environ['SKIP_DISABLE_TF2'] = '1'
    yield
  finally:
    if prev_value is None:
      del os.environ['SKIP_DISABLE_TF2']
    else:
      os.environ['SKIP_DISABLE_TF2'] = prev_value
