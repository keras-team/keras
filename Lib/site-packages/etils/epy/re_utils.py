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

"""Re utils."""

import functools
import re


def reverse_fstring(pattern: str, string: str) -> dict[str, str] | None:
  """Reverse f-string.

  Example:

  ```python
  epy.reverse_fstring(
      '/home/{user}/projects/{project}',
      '/home/conchylicultor/projects/menhir'
  ) == {
      'user': 'conchylicultor',
      'project': 'menhir',
  }
  ```

  Args:
    pattern: The f-string pattern (can only contained named group)
    string: The string to search

  Returns:
    The extracted info
  """
  pattern = _pattern_cache(pattern)
  if m := pattern.fullmatch(string):
    return m.groupdict()
  else:
    return None


@functools.cache
def _pattern_cache(pattern: str) -> re.Pattern[str]:
  pattern = re.sub(r'\{(?P<name>\w+)\}', r'(?P<\g<name>>.+)', pattern)
  return re.compile(pattern)
