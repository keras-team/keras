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

"""Register path."""

from __future__ import annotations

import os
import typing
from typing import Callable, TypeVar

from etils.epath import abstract_path
from etils.epath import gpath
from etils.epath.typing import PathLike  # pylint: disable=g-multiple-import,g-importing-member

_T = TypeVar('_T')

# Classes and uri are registered in `gpath.py`
_PATHLIKE_CLS: tuple[type[abstract_path.Path], ...] = ()
_URI_PREFIXES_TO_CLS: dict[str, type[abstract_path.Path]] = {}


@typing.overload
def register_path_cls(
    path_cls_or_uri_prefix: str | list[str] | tuple[str, ...]
) -> Callable[[_T], _T]:
  ...


@typing.overload
def register_path_cls(path_cls_or_uri_prefix: _T) -> _T:
  ...


def register_path_cls(path_cls_or_uri_prefix):
  """Register the pathlib-like class.

  ```python
  @epath.register_path_cls('my_path://')
  class MyPath(pathlib.PurePosixPath):
    ...

  my_path = epath.Path('my_path://some-path')
  assert isinstance(my_path, MyPath)
  ```

  Args:
    path_cls_or_uri_prefix: If a uri prefix is given, then passing calling
      `tfds.core.as_path('prefix://path')` will call the decorated class.

  Returns:
    The decorator or decoratorated class
  """
  global _PATHLIKE_CLS
  if isinstance(path_cls_or_uri_prefix, (str, list, tuple)):

    def register_path_cls_decorator(cls: _T) -> _T:
      if isinstance(path_cls_or_uri_prefix, str):
        _URI_PREFIXES_TO_CLS[path_cls_or_uri_prefix] = cls
      elif isinstance(path_cls_or_uri_prefix, (list, tuple)):
        for uri_prefix in path_cls_or_uri_prefix:
          _URI_PREFIXES_TO_CLS[uri_prefix] = cls
      return register_path_cls(cls)

    return register_path_cls_decorator
  else:
    _PATHLIKE_CLS = _PATHLIKE_CLS + (path_cls_or_uri_prefix,)
    return path_cls_or_uri_prefix


def make_path(path: PathLike) -> abstract_path.Path:
  """Create a generic `pathlib.Path`-like abstraction.

  Depending on the input (e.g. `gs://`, `github://`, `ResourcePath`,...), the
  system (Windows, Linux,...), the function will create the right pathlib-like
  abstraction.

  Args:
    path: Pathlike object.

  Returns:
    path: The `pathlib.Path`-like abstraction.
  """
  is_windows = os.name == 'nt'
  if isinstance(path, str):
    uri_splits = path.split('://', maxsplit=1)
    if len(uri_splits) > 1:  # str is URI (e.g. `gs://`, `github://`,...)
      # On windows, `PosixGPath` is created for `gs://` paths
      return _URI_PREFIXES_TO_CLS[uri_splits[0] + '://'](path)  # pytype: disable=bad-return-type
    elif is_windows:
      return gpath.WindowsGPath(path)
    else:
      return gpath.PosixGPath(path)
  elif isinstance(path, _PATHLIKE_CLS):
    return path  # Forward resource path, gpath,... as-is  # pytype: disable=bad-return-type
  elif isinstance(path, os.PathLike):  # Other `os.fspath` compatible objects
    path_cls = gpath.WindowsGPath if is_windows else gpath.PosixGPath
    return path_cls(path)
  else:
    raise TypeError(f'Invalid path type: {path!r}')
