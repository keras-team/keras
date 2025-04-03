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

"""Auto display statements ending with `;` on Colab."""

from __future__ import annotations

import ast
import dataclasses
import functools
import re
import traceback
import typing
from typing import Any, Callable, TypeVar

from etils import epy
from etils.ecolab import array_as_img
from etils.ecolab import highlight_util
from etils.ecolab.inspects import core as inspects
from etils.etree import jax as etree  # pylint: disable=g-importing-member
import IPython
import packaging.version

_T = TypeVar('_T')
_DisplayFn = Callable[[Any], None]

# Old API (before IPython 7.0)
_IS_LEGACY_API = packaging.version.parse(
    IPython.__version__
) < packaging.version.parse('7')


class _Options(epy.StrEnum):
  """Available options."""

  SPEC = 's'
  INSPECT = 'i'
  ARRAY = 'a'
  PPRINT = 'p'
  SYNTAX_HIGHLIGHT = 'h'
  LINE = 'l'
  QUIET = 'q'

  @classmethod
  @property
  def all_letters(cls) -> set[str]:
    return {option.value for option in cls}


def auto_display(activate: bool = True) -> None:
  r"""Activate auto display statements ending with `;` (activated by default).

  Add a trailing `;` to any statement (assignment, expression, return
  statement) to display the current line.
  This call `IPython.display.display()` for pretty display.

  This change the default IPython behavior where `;` added to the last statement
  of the cell still silence the output.

  ```python
  x = my_fn();  # Display `my_fn()`

  my_fn();  # Display `my_fn()`
  ```

  `;` behavior can be disabled with `ecolab.auto_display(False)`

  Format:

  *   `my_obj;`: Alias for `IPython.display.display(x)`
  *   `my_obj;s`: (`spec`) Alias for
      `IPython.display.display(etree.spec_like(x))`
  *   `my_obj;i`: (`inspect`) Alias for `ecolab.inspect(x)`
  *   `my_obj;a`: (`array`) Alias for `media.show_images(x)` /
      `media.show_videos(x)` (`ecolab.auto_plot_array` behavior)
  *   `my_obj;p`: (`pretty_display`) Alias for `print(epy.pretty_repr(x))`.
      Can be combined with `s`. Used for pretty print `dataclasses` or print
      strings containing new lines (rather than displaying `\n`).
  *   `my_obj;h`: (`syntax_highlight`) Add Python code syntax highlighting (
      using `ecolab.highlight_html`)
  *   `my_obj;q`: (`quiet`) Don't display the line (e.g. last line)
  *   `my_obj;l`: (`line`) Also display the line (can be combined with previous
      statements). Has to be at the end (`;sl` is valid but not `;ls`).

  `p`, `s`, `h`, `l` can be combined.

  A functional API exists:

  ```python
  ecolab.disp(obj, mode='sp')  # Equivalent to `obj;sp`
  ```

  Args:
    activate: Allow to disable `auto_display`
  """
  if not epy.is_notebook():  # No-op outside Colab
    return

  # Register the transformation
  ip = IPython.get_ipython()
  shell = ip.kernel.shell

  # Clear previous transformation to support reload and disable.
  _clear_transform(shell.ast_transformers)
  if _IS_LEGACY_API:
    _clear_transform(shell.input_transformer_manager.python_line_transforms)
    _clear_transform(shell.input_splitter.python_line_transforms)
  else:
    _clear_transform(ip.input_transformers_post)

  if not activate:
    return

  # Register the transformation.
  # The `ast` do not contain the trailing `;` information, and there's no way
  # to access the original source code from `shell.ast_transformers`, so
  # the logic has 2 steps:
  # 1) Record the source code string using `python_line_transforms`
  # 2) Parse and add the `display` the source code with `ast_transformers`

  lines_recorder = _RecordLines()

  if _IS_LEGACY_API:
    _append_transform(
        shell.input_splitter.python_line_transforms, lines_recorder
    )
    _append_transform(
        shell.input_transformer_manager.python_line_transforms, lines_recorder
    )
  else:
    _append_transform(ip.input_transformers_post, lines_recorder)
  _append_transform(
      shell.ast_transformers, _AddDisplayStatement(lines_recorder)
  )


def _clear_transform(list_):
  # Support `ecolab` adhoc `reload`
  for elem in list(list_):
    if hasattr(elem, '__is_ecolab__'):
      list_.remove(elem)


def _append_transform(list_, transform):
  """Append `transform` to `list_` (with `ecolab` reload support)."""
  transform.__is_ecolab__ = True  # Marker to support `ecolab` adhoc `reload`
  list_.append(transform)


if _IS_LEGACY_API and not typing.TYPE_CHECKING:

  class _RecordLines(IPython.core.inputtransformer.InputTransformer):
    """Record lines."""

    def __init__(self):
      self._lines = []
      self.last_lines = []

      self.trailing_stmt_line_nums = {}
      super().__init__()

    def push(self, line):
      """Add a line."""
      self._lines.append(line)
      return line

    def reset(self):
      """Reset."""
      if self._lines:  # Save the last recorded lines
        # Required because reset is called multiple times on empty output,
        # so always keep the last non-empty output
        self.last_lines = []
        for line in self._lines:
          self.last_lines.extend(line.split('\n'))
      self._lines.clear()

      self.trailing_stmt_line_nums.clear()
      return

else:

  class _RecordLines:
    """Record lines."""

    def __init__(self):
      self.last_lines = []
      # Additional state (reset at each cell) to keep track of which lines
      # contain trailing statements
      self.trailing_stmt_line_nums = {}

    def __call__(self, lines: list[str]) -> list[str]:
      self.last_lines = [l.rstrip('\n') for l in lines]
      self.trailing_stmt_line_nums = {}
      return lines


# TODO(epot): During the first parsing stage, could implement a fault-tolerant
# parsing to support `line;=` Rather than `line;l`
# Something like that but that would chunk the code in blocks of valid
# statements
# def fault_tolerant_parsing(code: str):
#   lines = code.split('\n')
#   for i in range(len(lines)):
#     try:
#       last_valid = ast.parse('\n'.join(lines[:i]))
#     except SyntaxError:
#       break

#   return ast.unparse(last_valid)


def _reraise_error(fn: _T) -> _T:
  @functools.wraps(fn)
  def decorated(self, node: ast.AST):
    try:
      return fn(self, node)  # pytype: disable=wrong-arg-types
    except Exception as e:  # pylint: disable=broad-exception-caught
      code = '\n'.join(self.lines_recorder.last_lines)
      print(f'Error for code:\n-----\n{code}\n-----')
      traceback.print_exception(e)

  return decorated


class _AddDisplayStatement(ast.NodeTransformer):
  """Transform the `ast` to add the `IPython.display.display` statements."""

  def __init__(self, lines_recorder: _RecordLines):
    self.lines_recorder = lines_recorder
    super().__init__()

  def _maybe_display(
      self, node: ast.Assign | ast.AnnAssign | ast.Expr
  ) -> ast.AST:
    """Wrap the node in a `display()` call."""
    if self._is_alias_stmt(node):  # Alias statements should be no-op
      return ast.Pass()

    line_info = _has_trailing_semicolon(self.lines_recorder.last_lines, node)
    if line_info.has_trailing:
      if node.value is None:  # `AnnAssign().value` can be `None` (`a: int`)
        pass
      else:
        options = ''.join([o.value for o in line_info.options])
        fn_kwargs = [
            ast.keyword('options', ast.Constant(options)),
        ]
        if line_info.print_line:
          fn_kwargs.append(ast.keyword('line_code', _unparse_line(node)))

        node.value = ast.Call(  # pytype: disable=wrong-arg-types
            func=_parse_expr('ecolab.auto_display_utils._display_and_return'),
            args=[node.value],
            keywords=fn_kwargs,
        )
        node = ast.fix_missing_locations(node)
        self.lines_recorder.trailing_stmt_line_nums[line_info.line_num] = (
            line_info
        )

    return node

  def _is_alias_stmt(self, node: ast.AST) -> bool:
    match node:
      case ast.Expr(value=ast.Name(id=name)):
        pass
      case _:
        return False
    if any(l not in _Options.all_letters for l in name):
      return False
    # The alias is not in the same line as a trailing `;`
    if node.end_lineno - 1 not in self.lines_recorder.trailing_stmt_line_nums:
      return False
    return True

  @_reraise_error
  def visit_Assert(self, node: ast.Assert) -> None:  # pylint: disable=invalid-name
    # Wrap assert so the `node.value` match the expected API
    node = _WrapAssertNode(node)
    node = self._maybe_display(node)  # pytype: disable=wrong-arg-types
    assert isinstance(node, _WrapAssertNode)
    node = node._node  # Unwrap  # pylint: disable=protected-access
    return node

  # pylint: disable=invalid-name
  visit_Assign = _reraise_error(_maybe_display)
  visit_AnnAssign = _reraise_error(_maybe_display)
  visit_Expr = _reraise_error(_maybe_display)
  visit_Return = _reraise_error(_maybe_display)
  # pylint: enable=invalid-name


def _parse_expr(code: str) -> ast.AST:
  return ast.parse(code, mode='eval').body


class _WrapAssertNode(ast.Assert):
  """Like `Assert`, but rename `node.test` to `node.value`."""

  def __init__(self, node: ast.Assert) -> None:
    self._node = node

  def __getattribute__(self, name: str) -> Any:
    if name in ('value', '_node'):
      return super().__getattribute__(name)
    return getattr(self._node, name)

  @property
  def value(self) -> ast.AST:
    return self._node.test

  @value.setter
  def value(self, value: ast.AST) -> None:
    self._node.test = value


@dataclasses.dataclass(frozen=True)
class _LineInfo:
  has_trailing: bool
  options: set[_Options]
  line_num: int

  @property
  def print_line(self) -> bool:
    return _Options.LINE in self.options


def _has_trailing_semicolon(
    code_lines: list[str],
    node: ast.AST,
) -> _LineInfo:
  """Check if `node` has trailing `;`."""
  if isinstance(node, ast.AnnAssign) and node.value is None:
    # `AnnAssign().value` can be `None` (`a: int`), do not print anything
    return _LineInfo(
        has_trailing=False,
        options=set(),
        line_num=-1,
    )

  # Extract the lines of the statement
  line_num = node.end_lineno - 1  # pytype: disable=attribute-error
  last_line = code_lines[line_num]  # lineno starts at `1`

  # `node.end_col_offset` is in bytes, so UTF-8 characters count 3.
  last_part_of_line = last_line.encode('utf-8')
  last_part_of_line = last_part_of_line[node.end_col_offset :]  # pytype: disable=attribute-error
  last_part_of_line = last_part_of_line.decode('utf-8')

  # Check if the last character is a `;` token
  has_trailing = False
  options = set()
  if match := _detect_trailing_regex().match(last_part_of_line):
    has_trailing = True
    if match.group('options'):
      options = match.group('options')
      options = {_Options(o) for o in options}

  return _LineInfo(
      has_trailing=has_trailing,
      options=options,
      line_num=line_num,
  )


@functools.cache
def _detect_trailing_regex() -> re.Pattern[str]:
  """Check if the last character is a `;` token."""
  # Match:
  # * `; a`
  # * `; a # Some comment`
  # * `; # Some comment`
  # Do not match:
  # * `; a; b`
  # * `; a=1`

  available_letters = ''.join(sorted(_Options.all_letters))  # pytype: disable=wrong-arg-types
  return re.compile(
      ' *; *'  # Trailing `;` (surrounded by spaces)
      f'(?P<options>[{available_letters}]*)?'  # Optionally a `option` letter
      ' *(?:#.*)?$'  # Line can end by a `# comment`
  )


def _unparse_line(node: ast.AST) -> ast.Constant:
  """Extract the line code."""
  if isinstance(node, ast.Assign):
    node = node.targets
  elif isinstance(node, ast.AnnAssign):
    node = node.target
  return ast.Constant(ast.unparse(node))


def disp(obj: Any, *, mode: str = '') -> None:
  """Display the object.

  This is the functional API for the `;` auto display magic.

  Args:
    obj: The object to display
    mode: Any mode supported by `ecolab.auto_display()`
  """
  if _Options.LINE in mode:
    raise NotImplementedError('Line mode not supported in `disp()`')
  # Do not return anything so the object is not displayed twice at the last
  # instuction of a cell
  _display_and_return(obj, options=mode)


def _display_and_return(
    x: _T,
    *,
    options: str,
    line_code: str | None = None,
) -> _T:
  """Print `x` and return `x`."""
  x_origin = x
  options = {_Options(o) for o in options}

  if _Options.QUIET in options:  # Do not display anything
    return x_origin

  if _Options.SPEC in options:  # Convert to spec
    x = etree.spec_like(x)

  repr_fn = repr
  display_fn = IPython.display.display
  if line_code and _Options.SYNTAX_HIGHLIGHT not in options:
    print(line_code + ' = ', end='')
    # When the next element is a `IPython.display`, the next element is
    # displayed on a new line. This is because `display()` create a new
    # <div> section. So use standard `print` when line is displayed.
    display_fn = lambda x: print(repr(x))

  if _Options.PPRINT in options:
    repr_fn = epy.pretty_repr
    display_fn = _pretty_display

  if _Options.INSPECT in options:
    inspects.inspect(x)
    return x_origin

  if _Options.ARRAY in options:
    html = array_as_img.array_repr_html(x)
    if html is None:
      print(f'Invalid array to display: {type(x)}')
    else:
      _html_display(html)
    return x_origin

  if _Options.SYNTAX_HIGHLIGHT in options:
    x_repr = repr_fn(x)
    if line_code:
      x_repr = f'{line_code} = {x_repr}'
    _html_display(highlight_util.highlight_html(x_repr))
    return x_origin

  display_fn(x)
  return x_origin


def _html_display(html):
  IPython.display.display(IPython.display.HTML(html))


def _pretty_display(x):
  """Print `x` and return `x`."""
  # 2 main use-case:
  # * Print strings (including `\n`)
  # * Pretty-print dataclasses
  if isinstance(x, str):
    print(x)
  else:
    print(epy.pretty_repr(x))
