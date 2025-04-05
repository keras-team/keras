# Copyright 2024 The Flax Authors.
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

import dataclasses
import os
import sys
import threading
import typing as tp

A = tp.TypeVar('A')
B = tp.TypeVar('B')


def supports_color() -> bool:
  """
  Returns True if the running system's terminal supports color, and False otherwise.
  """
  try:
    from IPython import get_ipython

    ipython_available = get_ipython() is not None
  except ImportError:
    ipython_available = False

  supported_platform = sys.platform != 'win32' or 'ANSICON' in os.environ
  is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
  return (supported_platform and is_a_tty) or ipython_available


class Color(tp.NamedTuple):
  TYPE: str
  ATTRIBUTE: str
  SEP: str
  PAREN: str
  COMMENT: str
  INT: str
  STRING: str
  FLOAT: str
  BOOL: str
  NONE: str
  END: str


NO_COLOR = Color(
  TYPE='',
  ATTRIBUTE='',
  SEP='',
  PAREN='',
  COMMENT='',
  INT='',
  STRING='',
  FLOAT='',
  BOOL='',
  NONE='',
  END='',
)


# Use python vscode theme colors
if supports_color():
  COLOR = Color(
    TYPE='\x1b[38;2;79;201;177m',
    ATTRIBUTE='\033[38;2;156;220;254m',
    SEP='\x1b[38;2;212;212;212m',
    PAREN='\x1b[38;2;255;213;3m',
    # COMMENT='\033[38;2;87;166;74m',
    COMMENT='\033[38;2;105;105;105m',  # Dark gray
    INT='\x1b[38;2;182;207;169m',
    STRING='\x1b[38;2;207;144;120m',
    FLOAT='\x1b[38;2;182;207;169m',
    BOOL='\x1b[38;2;86;156;214m',
    NONE='\x1b[38;2;86;156;214m',
    END='\x1b[0m',
  )
else:
  COLOR = NO_COLOR


@dataclasses.dataclass
class ReprContext(threading.local):
  current_color: Color = COLOR


REPR_CONTEXT = ReprContext()


def colorized(x, /):
  c = REPR_CONTEXT.current_color
  if isinstance(x, list):
    return f'{c.PAREN}[{c.END}{", ".join(map(lambda i: colorized(i), x))}{c.PAREN}]{c.END}'
  elif isinstance(x, tuple):
    if len(x) == 1:
      return f'{c.PAREN}({c.END}{colorized(x[0])},{c.PAREN}){c.END}'
    return f'{c.PAREN}({c.END}{", ".join(map(lambda i: colorized(i), x))}{c.PAREN}){c.END}'
  elif isinstance(x, dict):
    open, close = '{', '}'
    return f'{c.PAREN}{open}{c.END}{", ".join(f"{c.STRING}{k!r}{c.END}: {colorized(v)}" for k, v in x.items())}{c.PAREN}{close}{c.END}'
  elif isinstance(x, set):
    open, close = '{', '}'
    return f'{c.PAREN}{open}{c.END}{", ".join(map(lambda i: colorized(i), x))}{c.PAREN}{close}{c.END}'
  elif isinstance(x, type):
    return f'{c.TYPE}{x.__name__}{c.END}'
  elif isinstance(x, bool):
    return f'{c.BOOL}{x}{c.END}'
  elif isinstance(x, int):
    return f'{c.INT}{x}{c.END}'
  elif isinstance(x, str):
    return f'{c.STRING}{x!r}{c.END}'
  elif isinstance(x, float):
    return f'{c.FLOAT}{x}{c.END}'
  elif x is None:
    return f'{c.NONE}{x}{c.END}'
  elif isinstance(x, Representable):
    return get_repr(x)
  else:
    return repr(x)


@dataclasses.dataclass
class Object:
  type: tp.Union[str, type]
  start: str = '('
  end: str = ')'
  kv_sep: str = '='
  indent: str = '  '
  empty_repr: str = ''
  comment: str = ''
  same_line: bool = False

  @property
  def elem_sep(self):
    return ', ' if self.same_line else ',\n'


@dataclasses.dataclass
class Attr:
  key: str
  value: tp.Union[str, tp.Any]
  start: str = ''
  end: str = ''
  use_raw_value: bool = False
  use_raw_key: bool = False


class Representable:
  __slots__ = ()

  def __nnx_repr__(self) -> tp.Iterator[tp.Union[Object, Attr]]:
    raise NotImplementedError

  def __repr__(self) -> str:
    current_color = REPR_CONTEXT.current_color
    REPR_CONTEXT.current_color = NO_COLOR
    try:
      return get_repr(self)
    finally:
      REPR_CONTEXT.current_color = current_color

  def __str__(self) -> str:
    return get_repr(self)


def get_repr(obj: Representable) -> str:
  if not isinstance(obj, Representable):
    raise TypeError(f'Object {obj!r} is not representable')

  c = REPR_CONTEXT.current_color
  iterator = obj.__nnx_repr__()
  config = next(iterator)

  if not isinstance(config, Object):
    raise TypeError(f'First item must be Config, got {type(config).__name__}')

  kv_sep = f'{c.SEP}{config.kv_sep}{c.END}'

  def _repr_elem(elem: tp.Any) -> str:
    if not isinstance(elem, Attr):
      raise TypeError(f'Item must be Elem, got {type(elem).__name__}')

    value_repr = elem.value if elem.use_raw_value else colorized(elem.value)
    value_repr = value_repr.replace('\n', '\n' + config.indent)
    key = elem.key if elem.use_raw_key else f'{c.ATTRIBUTE}{elem.key}{c.END}'
    indent = '' if config.same_line else config.indent

    return f'{indent}{elem.start}{key}{kv_sep}{value_repr}{elem.end}'

  elems = config.elem_sep.join(map(_repr_elem, iterator))

  if elems:
    if config.same_line:
      elems_repr = elems
      comment = ''
    else:
      elems_repr = '\n' + elems + '\n'
      comment = f'{c.COMMENT}{config.comment}{c.END}'
  else:
    elems_repr = config.empty_repr
    comment = ''

  type_repr = (
    config.type if isinstance(config.type, str) else config.type.__name__
  )
  type_repr = f'{c.TYPE}{type_repr}{c.END}' if type_repr else ''
  start = f'{c.PAREN}{config.start}{c.END}' if config.start else ''
  end = f'{c.PAREN}{config.end}{c.END}' if config.end else ''

  out = f'{type_repr}{start}{comment}{elems_repr}{end}'
  return out

class MappingReprMixin(Representable):
  def __nnx_repr__(self):
    yield Object(type='', kv_sep=': ', start='{', end='}')

    for key, value in self.items():  # type: ignore
      yield Attr(colorized(key), value, use_raw_key=True)

@dataclasses.dataclass(repr=False)
class PrettyMapping(Representable):
  mapping: tp.Mapping

  def __nnx_repr__(self):
    yield Object(type=type(self), kv_sep=': ', start='({', end='})')

    for key, value in self.mapping.items():
      yield Attr(colorized(key), value, use_raw_key=True)


@dataclasses.dataclass(repr=False)
class SequenceReprMixin(Representable):
  def __nnx_repr__(self):
    yield Object(type=type(self), kv_sep='', start='([', end='])')

    for value in self:  # type: ignore
      yield Attr('', value, use_raw_key=True)


@dataclasses.dataclass(repr=False)
class PrettySequence(Representable):
  sequence: tp.Sequence

  def __nnx_repr__(self):
    yield Object(type=type(self), kv_sep='', start='([', end='])')

    for value in self.sequence:
      yield Attr('', value, use_raw_key=True)