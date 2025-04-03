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

"""Abstract path."""

from __future__ import annotations

from collections.abc import Callable
import os
import pathlib
import typing
from typing import Any, AnyStr, Iterator, Optional, Type, TypeVar

from etils.epath import register
from etils.epath import stat_utils
from etils.epath.typing import PathLike  # pylint: disable=g-importing-member

_T = TypeVar('_T')


# Ideally, `Path` should be `abc.ABC`. However this trigger pytype errors
# when calling `Path()` (can't instantiate abstract base class)
# Also this allow path childs to only partially implement the Path API (e.g.
# read only path)
def abstractmethod(fn: _T) -> _T:
  return fn


class Path(pathlib.PurePosixPath):
  """Abstract base class for pathlib.Path-like API.

  See [pathlib.Path](https://docs.python.org/3/library/pathlib.html)
  documentation.
  """

  # TODO(epot): With 3.12, might be able to inherit from `pathlib.PosixPath`
  # directly so some of those methods are automatically implemented.

  def __new__(cls: Type[_T], *args: PathLike) -> _T:
    """Create a new path.

    ```python
    path = abcpath.Path()
    ```

    We use __new__ instead of __init__ to allow subclassing, even though the
    usage of __init__ is possible from python>=3.12.

    Args:
      *args: Paths to create

    Returns:
      path: The registered path
    """

    if cls == Path:
      if not args:
        return register.make_path('.')
      root, *parts = args
      return register.make_path(root).joinpath(*parts)
    else:
      return super().__new__(cls, *args)

  # ====== Pure paths ======

  def format(self: _T, *args: Any, **kwargs: Any) -> _T:
    """Apply `str.format()` to the path."""
    return type(self)(os.fspath(self).format(*args, **kwargs))  # pytype: disable=not-instantiable

  # ====== Read-only methods ======

  @abstractmethod
  def exists(self) -> bool:
    """Returns True if self exists."""
    raise NotImplementedError

  @abstractmethod
  def is_dir(self) -> bool:
    """Returns True if self is a dir."""
    raise NotImplementedError

  def is_file(self) -> bool:
    """Returns True if self is a file."""
    return not self.is_dir()

  @abstractmethod
  def iterdir(self: _T) -> Iterator[_T]:
    """Iterates over the directory."""
    raise NotImplementedError

  @abstractmethod
  def glob(self: _T, pattern: str) -> Iterator[_T]:
    """Yields all matching files (of any kind)."""
    # Might be able to implement using `iterdir` (recursivelly for `rglob`).
    raise NotImplementedError

  def rglob(self: _T, pattern: str) -> Iterator[_T]:
    """Yields all matching files recursively (of any kind)."""
    return self.glob(f'**/{pattern}')

  @abstractmethod
  def walk(
      self: _T,
      *,
      top_down: bool = True,
      on_error: Callable[[OSError], object] | None = None,
  ) -> Iterator[tuple[_T, list[str], list[str]]]:
    raise NotImplementedError

  def expanduser(self: _T) -> _T:
    """Returns a new path with expanded `~` and `~user` constructs."""
    if '~' not in self.parts:  # pytype: disable=attribute-error
      return self
    raise NotImplementedError

  @abstractmethod
  def resolve(self: _T, strict: bool = False) -> _T:
    """Returns the absolute path."""
    raise NotImplementedError

  @abstractmethod
  def open(
      self,
      mode: str = 'r',
      encoding: Optional[str] = None,
      errors: Optional[str] = None,
      **kwargs: Any,
  ) -> typing.IO[AnyStr]:
    """Opens the file."""
    raise NotImplementedError

  def read_bytes(self) -> bytes:
    """Reads contents of self as bytes."""
    with self.open('rb') as f:
      return f.read()

  def read_text(self, encoding: Optional[str] = None) -> str:
    """Reads contents of self as a string."""
    with self.open('r', encoding=encoding) as f:
      return f.read()

  @abstractmethod
  def stat(self) -> stat_utils.StatResult:
    """Returns metadata for the file/directory."""
    raise NotImplementedError

  # ====== Write methods ======

  @abstractmethod
  def mkdir(
      self,
      mode: int = 0o777,
      parents: bool = False,
      exist_ok: bool = False,
  ) -> None:
    """Create a new directory at this given path."""
    raise NotImplementedError

  @abstractmethod
  def rmdir(self) -> None:
    """Remove the empty directory at this given path."""
    raise NotImplementedError

  @abstractmethod
  def rmtree(self, missing_ok: bool = False) -> None:
    """Remove the directory, including all sub-files."""
    raise NotImplementedError

  @abstractmethod
  def unlink(self, missing_ok: bool = False) -> None:
    """Remove this file or symbolic link."""
    raise NotImplementedError

  def write_bytes(self, data: bytes) -> int:
    """Writes content as bytes."""
    with self.open('wb') as f:
      return f.write(data)

  def write_text(
      self,
      data: str,
      encoding: Optional[str] = None,
      errors: Optional[str] = None,
  ) -> int:
    """Writes content as str."""
    if encoding and encoding.lower() not in {'utf8', 'utf-8'}:
      raise NotImplementedError(f'Non UTF-8 encoding not supported for {self}')
    if errors:
      raise NotImplementedError(f'Error not supported for writing {self}')
    with self.open('w') as f:
      return f.write(data)

  def touch(self, mode: int = 0o666, exist_ok: bool = True) -> None:
    """Create a file at this given path."""
    if mode != 0o666:
      raise NotImplementedError(f'Only mode=0o666 supported for {self}')
    if self.exists():
      if exist_ok:
        return
      else:
        raise FileExistsError(f'{self} already exists.')
    self.write_text('')

  # pytype: disable=bad-return-type
  @abstractmethod
  def rename(self: _T, target: PathLike) -> _T:
    """Renames the path."""

  @abstractmethod
  def replace(self: _T, target: PathLike) -> _T:
    """Overwrites the destination path."""

  @abstractmethod
  def copy(self: _T, dst: PathLike, overwrite: bool = False) -> _T:
    """Copy the current file to the given destination."""

  # pytype: enable=bad-return-type

  # ====== Other ======

  @classmethod
  def __get_pydantic_core_schema__(cls, source_type, handler):
    """Handles pydantic serialization similarly to a pathlib.PurePosixPath.

    See https://docs.pydantic.dev/latest/concepts/types

    Args:
      source_type: Any
      handler: pydantic.GetCoreSchemaHandler

    Returns:
      pydantic_core.CoreSchema
    """
    from pydantic_core import core_schema  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    del source_type  # Unused in this implementation.
    del handler  # Unused in this implementation.

    # This schema is based on Pydantic's schema for pathlib.Path types:
    # https://github.com/pydantic/pydantic/blob/23246fe642e35c2972dd9559af905e9424f22438/pydantic/_internal/_std_types_schema.py#L66
    # But we can't delegate to this, because we must not invoke the pathlib
    # constructors, because they break paths containing a schema
    # like 'gs://foo', whereas epath.Path handles these correctly.

    # Schema to construct from a 'str' instance.
    from_str_schema = core_schema.no_info_after_validator_function(
        cls, core_schema.str_schema()
    )
    return core_schema.json_or_python_schema(
        # When constructing from JSON, require a string as input, and pass it
        # to the class constructor.
        json_schema=from_str_schema,
        # When constructing from Python, accept either an instance of the class,
        # or a string.
        python_schema=core_schema.union_schema([
            core_schema.is_instance_schema(cls),
            from_str_schema,
        ]),
        serialization=core_schema.to_string_ser_schema(),
    )
