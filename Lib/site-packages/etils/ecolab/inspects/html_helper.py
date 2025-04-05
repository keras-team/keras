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

"""Mini library to help building html code."""

from __future__ import annotations

from collections.abc import Callable
import functools


def tag(name: str, **attributes: str | list[str] | None) -> Callable[..., str]:
  """Create a html tag.

  Usage:

  ```python
  tag('div', id='x')('content') == '<div id="x">content</div>'
  ```

  Args:
    name: Tag name
    **attributes: Attributes of the tag

  Returns:
    The HTML string
  """
  # Could be much more optimized by first building the graph of nested
  # element, then joining individual parts

  attributes = _format_tag_attributes(attributes)

  def apply(*content: str) -> str:
    content = ''.join(content)
    return f'<{name}{attributes}>{content}</{name}>'

  return apply


def _format_tag_attributes(attrs: dict[str, str | list[str]]) -> str:
  """Format the tag attributes."""
  out = ['']
  for k, v in attrs.items():
    if v is None:
      continue
    if k == 'class_':  # `class` is a forbidden Python keyword for arg name
      k = 'class'

    if isinstance(v, str):
      v = v.split()
    elif not isinstance(v, list):
      raise TypeError(f'Unexpected attribute: {k}={v!r}')

    # To avoid collisions, we prefix all classes with `etils-`
    if k == 'class':
      v = [f'etils-{v_}' for v_ in v]

    v = ' '.join(v)

    out.append(f'{k}="{v}"')
  return ' '.join(out)


span = functools.partial(tag, 'span')
ul = functools.partial(tag, 'ul')
li = functools.partial(tag, 'li')
