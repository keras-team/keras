# Copyright 2020 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import dataclasses
import functools
import itertools
import os.path
import re
import sys
import sysconfig
import threading
import types
from typing import NamedTuple

import jax.version
from jax._src.lib import xla_client

from jax._src import traceback_util
traceback_util.register_exclusion(__file__)


Traceback = xla_client.Traceback

class Frame(NamedTuple):
  file_name: str
  function_name: str
  start_line: int
  start_column: int
  end_line: int
  end_column: int


_exclude_paths: list[str] = [
    # Attach the separator to make sure that .../jax does not end up matching
    # .../jax_triton and other packages that might have a jax prefix.
    os.path.dirname(jax.version.__file__) + os.sep,
    # Also exclude stdlib as user frames. In a non-standard Python runtime,
    # the following two may be different.
    sysconfig.get_path('stdlib'),
    os.path.dirname(sysconfig.__file__)
]

@functools.cache
def _exclude_path_regex() -> re.Pattern[str]:
  # The regex below would not handle an empty set of exclusions correctly.
  assert len(_exclude_paths) > 0
  return re.compile('|'.join(f'^{re.escape(path)}' for path in _exclude_paths))


def register_exclusion(path: str):
  _exclude_paths.append(path)
  _exclude_path_regex.cache_clear()
  is_user_filename.cache_clear()


# Explicit inclusions take priority over exclude paths.
_include_paths: list[str] = []

@functools.cache
def _include_path_regex() -> re.Pattern[str]:
  patterns = [f'^{re.escape(path)}' for path in _include_paths]
  patterns.append('_test.py$')
  return re.compile('|'.join(patterns))

def register_inclusion(path: str):
  _include_paths.append(path)
  _include_path_regex.cache_clear()
  is_user_filename.cache_clear()


class Scope(NamedTuple):
  name: str

  def wrap(self, stack: tuple[str, ...]) -> tuple[str, ...]:
    return (self.name, *stack)

class Transform(NamedTuple):
  name: str

  def wrap(self, stack: tuple[str, ...]) -> tuple[str, ...]:
    if stack:
      return (f'{self.name}({stack[0]})', *stack[1:])
    else:
      return ()

@dataclasses.dataclass(frozen=True)
class NameStack:
  stack: tuple[Scope | Transform, ...] = ()

  def extend(self, name: tuple[str, ...] | str) -> NameStack:
    if not isinstance(name, tuple):
      name = (name,)
    scopes = tuple(map(Scope, name))
    return NameStack(self.stack + scopes)

  def wrap_name(self, name: str) -> str:
    if not self.stack:
      return name
    return f'{self}/{name}'

  def transform(self, transform_name: str) -> NameStack:
    return NameStack((*self.stack, Transform(transform_name)))

  def __getitem__(self, idx: slice) -> NameStack:
    return NameStack(self.stack[idx])

  def __len__(self):
    return len(self.stack)

  def __add__(self, other: NameStack) -> NameStack:
    return NameStack(self.stack + other.stack)

  def __radd__(self, other: NameStack) -> NameStack:
    return NameStack(other.stack + self.stack)

  def __str__(self) -> str:
    scope: tuple[str, ...] = ()
    for elem in self.stack[::-1]:
      scope = elem.wrap(scope)
    return '/'.join(scope)


def new_name_stack(name: str = '') -> NameStack:
  name_stack = NameStack()
  if name:
    name_stack = name_stack.extend(name)
  return name_stack


class SourceInfo:
  traceback: Traceback | None
  name_stack: NameStack

  # It's slightly faster to use a class with __slots__ than a NamedTuple.
  __slots__ = ['traceback', 'name_stack']

  def __init__(self, traceback: Traceback | None, name_stack: NameStack):
    self.traceback = traceback
    self.name_stack = name_stack

  def replace(self, *, traceback: Traceback | None = None,
      name_stack: NameStack | None = None) -> SourceInfo:
    return SourceInfo(
        self.traceback if traceback is None else traceback,
        self.name_stack if name_stack is None else name_stack
    )

def new_source_info() -> SourceInfo:
  return SourceInfo(None, NameStack())

@functools.cache
def is_user_filename(filename: str) -> bool:
  """Heuristic that guesses the identity of the user's code in a stack trace."""
  return (_include_path_regex().search(filename) is not None
          or _exclude_path_regex().search(filename) is None)

if sys.version_info >= (3, 11):
  def raw_frame_to_frame(code: types.CodeType, lasti: int) -> Frame:
    loc = xla_client.Traceback.code_addr2location(code, lasti)
    start_line, start_column, end_line, end_column = loc
    return Frame(file_name=code.co_filename,
                function_name=code.co_qualname,
                start_line=start_line, start_column=start_column,
                end_line=end_line, end_column=end_column)
else:
  def raw_frame_to_frame(code: types.CodeType, lasti: int) -> Frame:
    # pre-3.11 co_qualname does not exist, use co_name
    return Frame(file_name=code.co_filename,
                function_name=code.co_name,
                start_line=xla_client.Traceback.code_addr2line(code, lasti),
                start_column=0, end_line=0, end_column=0)

def user_frames(source_info: SourceInfo) -> Iterator[Frame]:
  """Iterator over the user's frames, filtering jax-internal frames."""
  # Guess the user's frame is the innermost frame not in the jax source tree or
  # Python stdlib. We don't use traceback_util.path_starts_with because that
  # incurs filesystem access, which may be slow; we call this function when
  # e.g. adding source provenance annotations to XLA lowerings, so we don't
  # want to incur the cost. We consider files that end with _test.py as user
  # frames, to allow testing this mechanism from tests.
  traceback = source_info.traceback
  code, lasti = traceback.raw_frames() if traceback else ([], [])
  return (raw_frame_to_frame(code[i], lasti[i]) for i in range(len(code))
          if is_user_filename(code[i].co_filename))

@functools.lru_cache(maxsize=64)
def user_frame(source_info: SourceInfo) -> Frame | None:
  return next(user_frames(source_info), None)

def _summarize_frame(frame: Frame) -> str:
  if frame.start_column != 0:
    return (f"{frame.file_name}:{frame.start_line}:{frame.start_column} "
            f"({frame.function_name})")
  else:
    return f"{frame.file_name}:{frame.start_line} ({frame.function_name})"

def summarize(source_info: SourceInfo, num_frames=1) -> str:
  frames = itertools.islice(user_frames(source_info), num_frames)
  frame_strs = [_summarize_frame(frame) if frame else "unknown"
                for frame in frames]
  return '\n'.join(reversed(frame_strs))

class _SourceInfoContext(threading.local):
  context: SourceInfo

  def __init__(self):
    self.context = new_source_info()

_source_info_context = _SourceInfoContext()

def current() -> SourceInfo:
  source_info = _source_info_context.context
  if not source_info.traceback:
    source_info = source_info.replace(traceback=xla_client.Traceback.get_traceback())
  return source_info

class JaxStackTraceBeforeTransformation(Exception): pass

_message = (
    'The preceding stack trace is the source of the JAX operation that, once '
    'transformed by JAX, triggered the following exception.\n'
    '\n--------------------')

def has_user_context(e):
  while e is not None:
    if isinstance(e, JaxStackTraceBeforeTransformation):
      return True
    e = e.__cause__
  return False

@contextlib.contextmanager
def user_context(c: Traceback | None, *, name_stack: NameStack | None = None):
  prev = _source_info_context.context
  _source_info_context.context = _source_info_context.context.replace(
      traceback=c, name_stack=name_stack)
  filtered_tb = None
  try:
    yield
  except Exception as e:
    if c is None or has_user_context(e):
      raise
    filtered_tb = traceback_util.filter_traceback(c.as_python_traceback())
    if filtered_tb:
      msg = traceback_util.format_exception_only(e)
      msg = f'{msg}\n\n{_message}'
      exp = JaxStackTraceBeforeTransformation(msg).with_traceback(filtered_tb)
      exp.__context__ = e.__context__
      exp.__cause__ = e.__cause__
      exp.__suppress_context__ = e.__suppress_context__
      e.__context__ = None
      e.__cause__ = exp
    raise
  finally:
    _source_info_context.context = prev
    del filtered_tb

def current_name_stack() -> NameStack:
  return _source_info_context.context.name_stack

@contextlib.contextmanager
def extend_name_stack(name: str) -> Iterator[NameStack]:
  prev_context = _source_info_context.context
  curr_name_stack = prev_context.name_stack
  new_context = prev_context.replace(name_stack=curr_name_stack.extend(name))
  _source_info_context.context = new_context
  try:
    yield _source_info_context.context.name_stack
  finally:
    _source_info_context.context = prev_context

@contextlib.contextmanager
def set_name_stack(name_stack: NameStack) -> Iterator[None]:
  prev_context = _source_info_context.context
  new_context = prev_context.replace(name_stack=name_stack)
  _source_info_context.context = new_context
  try:
    yield
  finally:
    _source_info_context.context = prev_context

@contextlib.contextmanager
def reset_name_stack() -> Iterator[None]:
  with set_name_stack(NameStack()):
    yield

@contextlib.contextmanager
def transform_name_stack(name: str) -> Iterator[NameStack]:
  prev_context = _source_info_context.context
  curr_name_stack = prev_context.name_stack
  new_context = prev_context.replace(name_stack=curr_name_stack.transform(name))
  _source_info_context.context = new_context
  try:
    yield _source_info_context.context.name_stack
  finally:
    _source_info_context.context = prev_context
