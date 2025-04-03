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

"""Text utils."""

from __future__ import annotations

import contextlib
import dataclasses
import difflib
import inspect
import reprlib
import sys
import textwrap
from typing import Any, Iterable, Iterator, Union

from etils.epy import py_utils

_BRACE_TO_BRACES = {
    '(': ('(', ')'),
    '[': ('[', ']'),
    '{': ('{', '}'),
}


@dataclasses.dataclass
class _Line:
  """Line item."""

  content: str
  indent_lvl: int
  indent_size: int


class Lines:
  """Util to build multi-line text.

  Useful for pretty-print tools and human readable `__repr__`.

  Example:

  ```python
  d = {'a': 1, 'b': 2}

  lines = epy.Lines()
  lines += 'dict('
  with lines.indent():
    for k, v in d.items():
      lines += f'{k}={v},'
  lines += ')'
  text = lines.join()
  ```

  Output:

  ```
  dict(
      a=1,
      b=2,
  )
  ```

  """

  def __init__(self, *, indent: int = 4):
    self._lines: list[_Line] = []
    self._indent_size = indent
    self._indent_lvl = 0

  def append(self, line: str) -> None:
    """Append a new line `str`."""
    if not isinstance(line, str):
      raise TypeError(f'Lines should be added with `str`. Got {line!r}')

    self._lines.append(
        _Line(
            content=line,
            indent_lvl=self._indent_lvl,
            indent_size=self._indent_size,
        ),
    )

  def extend(self, iterable: Iterable[str]) -> None:
    """Append all the new line `str` from the iterable."""
    for line in iterable:
      self.append(line)

  def __iadd__(self, line: str) -> Lines:
    """Append a new line `str`."""
    self.append(line)
    return self

  @contextlib.contextmanager
  def indent(self) -> Iterator[None]:
    self._indent_lvl += 1
    try:
      yield
    finally:
      self._indent_lvl -= 1

  def join(self, *, collapse: bool = False) -> str:
    """Returns the lines.

    Args:
      collapse: If `True`, all lines are merged together in a single line.

    Returns:
      text: All lines merged together
    """
    lines = []
    for line in self._lines:
      content = line.content
      if not collapse:
        # Add the indentation to all the sub-lines
        indentation = ' ' * line.indent_lvl * line.indent_size
        content = textwrap.indent(content, indentation)
      lines.append(content)

    if collapse:
      token = ''
    else:
      token = '\n'
    return token.join(lines)

  @classmethod
  def make_block(
      cls,
      header: str = '',
      content: str | dict[str, Any] | list[Any] | tuple[Any, ...] = (),
      *,
      braces: Union[str, tuple[str, str]] = '(',
      equal: str = '=',
      limit: int = 20,
  ) -> str:
    """Util function to create a code block.

    Example:

    ```python
    epy.Lines.make_block('A', {}) == 'A()'
    epy.Lines.make_block('A', {'x': '1'}) == 'A(x=1)'
    epy.Lines.make_block('A', {'x': '1', 'y': '2'}) == '''A(
        x=1,
        y=2,
    )'''
    ```

    Pattern is as:

    ```
    {header}{braces[0]}
        {k}={v},
        ...
    {braces[1]}
    ```

    Args:
      header: Prefix before the brace
      content: Dict of key to values. One line will be displayed per item if
        `len(content) > 1`. Otherwise the code is collapsed
      braces: Brace type (`(`, `[`, `{`), can be tuple for custom open/close.
      equal: The separator (`=`, `: `)
      limit: Strings smaller than this will be collapsed

    Returns:
      The block string
    """
    if isinstance(braces, str):
      braces = _BRACE_TO_BRACES[braces]
    brace_start, brace_end = braces

    if isinstance(content, str):
      content = [content]

    if isinstance(content, dict):
      parts = [f'{k}{equal}{pretty_repr(v)}' for k, v in content.items()]
    elif isinstance(content, (list, tuple)):
      parts = [pretty_repr(v) for v in content]
    else:
      raise TypeError(f'Invalid fields {type(content)}')

    collapse = len(parts) <= 1
    if any('\n' in p for p in parts):
      collapse = False
    # Also collapse string which are small
    elif sum(len(p) for p in parts) <= limit:
      collapse = True

    lines = cls()
    lines += f'{header}{brace_start}'
    with lines.indent():
      if collapse:
        lines += ', '.join(parts)
      else:
        for p in parts:
          lines += f'{p},'
    lines += f'{brace_end}'

    return lines.join(collapse=collapse)


def pprint(obj: Any) -> None:
  """Pretty print `obj`."""
  print(pretty_repr(obj))


@reprlib.recursive_repr()
def pretty_repr(obj: Any) -> str:
  """Pretty `repr(obj)` for nested list, dict, dataclasses,..."""
  return pretty_repr_top_level(obj)


def pretty_repr_top_level(obj: Any, *, force: bool = False) -> str:
  """Pretty `repr(obj)` for nested list, dict, dataclasses,...

  This version do not use `@reprlib.recursive_repr()` to avoid bug when used
  inside `__repr__`:

  ```python
  class A:
    def __repr__(self):
      return epy.pretty_repr_top_level(self)

  epy.pretty_repr(A())  # Do not display `...`
  ```

  Args:
    obj: Object to display
    force: Force the pretty_repr, even if the object has a custom `__repr__`.
      This is useful when the `__repr__` implementation itself want to call
      `pretty_repr(self)`.

  Returns:
    Repr
  """
  # TODO(epot): Should still somehow register `self` with the `recursive_repr`,
  # should support both:
  # pretty_repr(a) == A(recursive=...)
  # a.__repr__() == A(recursive=...)

  if isinstance(obj, str):
    return repr(obj)
  elif py_utils.is_namedtuple(obj):
    # TODO(epot): Could check if obj has custom `__repr__`
    return Lines.make_block(
        header=obj.__class__.__name__,
        content={
            field_name: getattr(obj, field_name)
            for field_name in type(obj)._fields
        },
    )
  elif type(obj) in (list, tuple):  # Skip sub-class as could have custom repr
    lines = Lines.make_block(
        content=obj,
        braces='[' if isinstance(obj, list) else '(',
    )
    # Singleton tuple have a trailing `,`
    if isinstance(obj, tuple) and len(obj) == 1:
      lines = lines.removesuffix(')') + ',)'
    return lines
  elif type(obj) is dict:  # pylint: disable=unidiomatic-typecheck
    return Lines.make_block(
        content={repr(k): v for k, v in obj.items()},
        braces='{',
        equal=': ',
    )
  elif _is_datclass(obj, force=force):
    all_fields = dataclasses.fields(obj)

    return Lines.make_block(
        header=obj.__class__.__name__,
        content={
            field.name: getattr(obj, field.name)
            for field in all_fields
            if field.repr
        },
    )
  elif _is_attr(obj, force=force):
    import attr  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    all_fields = attr.fields_dict(type(obj))

    return Lines.make_block(
        header=obj.__class__.__name__,
        content={
            field.name: getattr(obj, field.name)
            for field in all_fields.values()
            if field.repr
        },
    )
  else:
    return repr(obj)


class _Repr(str):
  """Display `str.__repr__` without quotes `'`."""

  def __repr__(self):
    return str(self)


def has_default_repr(cls: Any) -> bool:
  """Returns `True` if the dataclass do not overwrite `__repr__`."""
  repr_fn = inspect.unwrap(cls.__repr__)
  return repr_fn.__qualname__ == '__create_fn__.<locals>.__repr__'


def _is_datclass(obj: Any, *, force: bool = False) -> bool:
  """Returns `True` if the object is a dataclass."""
  if isinstance(obj, type):  # Class are not pretty-print
    return False
  if not dataclasses.is_dataclass(obj):
    return False
  if force:
    return True
  if not obj.__dataclass_params__.repr:  # dataclass(repr=False)
    return False
  # TODO(epot): Better support for recursive `pretty_repr` to avoid infinite
  # loops.
  if has_default_repr(type(obj)) or type(obj).__repr__ in (
      pretty_repr,
      pretty_repr_top_level,
  ):
    return True
  return False


def _is_attr(obj: Any, *, force: bool = False) -> bool:
  """Returns `True` if the object is a `attr` dataclass."""
  if 'attr' not in sys.modules:
    return False
  import attr  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  if not attr.has(type(obj)):
    return False
  if force:
    return True
  if not (doc := type(obj).__repr__.__doc__):
    return False
  if not doc.startswith('Method generated by attrs'):
    return False
  return True


def _is_fiddle(obj: Any) -> bool:
  """Returns `True` if the object is a `fiddle` config object."""
  if 'fiddle' not in sys.modules:
    return False
  import fiddle  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  return isinstance(obj, fiddle.Config)


def dedent(text: str) -> str:
  r"""Wrapper around `textwrap.dedent` which also `strip()` the content.

  Before:

  ```python
  text = textwrap.dedent(
      \"\"\"\\
      A(
         x=1,
      )\"\"\"
  )
  ```

  After:

  ```python
  text = epy.dedent(
      \"\"\"
      A(
         x=1,
      )
      \"\"\"
  )
  ```

  Args:
    text: The text to dedent

  Returns:
    The dedented text
  """
  return textwrap.dedent(text).strip()


def diff_str(a: str | object, b: str | object) -> str:
  """Pretty diff between 2 objects.

  Args:
    a: Object/str to compare
    b: Object/str to compare

  Returns:
    The diff string
  """
  if not isinstance(a, str):
    a = pretty_repr(a).split('\n')
  if not isinstance(b, str):
    b = pretty_repr(b).split('\n')

  diff = difflib.ndiff(a, b)
  return '\n'.join(diff)
