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

"""Utils to handle resources."""

from __future__ import annotations

import itertools
import pathlib
import posixpath
import sys
import types
import typing
from typing import Union
import zipfile

from etils.epath import abstract_path
from etils.epath import register
from etils.epath.typing import PathLike  # pylint: disable=g-importing-member
import importlib_resources


@register.register_path_cls
class ResourcePath(zipfile.Path):
  """Wrapper around `zipfile.Path` compatible with `os.PathLike`.

  Note: Calling `os.fspath` on the path will extract the file so should be
  discouraged.
  """

  def __fspath__(self) -> str:
    """Path string for `os.path.join`, `open`,...

    compatibility.

    Note: Calling `os.fspath` on the path extract the file, so should be
    discouraged. Prefer using `read_bytes`,... This only works for files,
    not directories.

    Returns:
      the extracted path string.
    """
    raise NotImplementedError('zipapp not supported. Please send us a PR.')

  # zipfile.Path do not define `__eq__` nor `__hash__`. See:
  # https://discuss.python.org/t/missing-zipfile-path-eq-and-zipfile-path-hash/16519
  def __eq__(self, other) -> bool:
    return (
        type(self) == type(other)  # pylint: disable=unidiomatic-typecheck
        and self.root == other.root  # pytype: disable=attribute-error
        and self.at == other.at  # pytype: disable=attribute-error
    )

  def __hash__(self) -> int:
    return hash((self.root, self.at))  # pytype: disable=attribute-error

  if sys.version_info < (3, 10):
    # Required due to: https://bugs.python.org/issue42043
    def _next(self, at) -> 'ResourcePath':  # pylint: disable=g-wrong-blank-lines
      return type(self)(self.root, at)  # pytype: disable=attribute-error

    # Before 3.10, joinpath only accept a single arg
    def joinpath(self, *other):
      """Overwrite `joinpath` to be consistent with `pathlib.Path`."""
      next_ = posixpath.join(self.at, *other)  # pytype: disable=attribute-error
      return self._next(self.root.resolve_dir(next_))  # pytype: disable=attribute-error

  if sys.version_info < (3, 11):

    @property
    def suffix(self):
      return pathlib.Path(self.at).suffix or self.filename.suffix  # pytype: disable=attribute-error


def resource_path(package: Union[str, types.ModuleType]) -> abstract_path.Path:
  """Returns read-only root directory path of the module.

  Used to access module resource files.

  Usage:

  ```python
  path = epath.resource_path('tensorflow_datasets') / 'README.md'
  content = path.read_text()
  ```

  This is compatible with everything, including zipapp (`.par`).

  Resource files should be in the `data=` of the `py_library(` (when using
  bazel).

  To write to your project (e.g. automatically update your code), read-only
  resource paths can be converted to read-write paths with
  `epath.to_write_path(path)`.

  Args:
    package: Module or module name.

  Returns:
    The read-only path to the root module directory
  """
  try:
    path = importlib_resources.files(package)  # pytype: disable=module-attr
  except AttributeError:
    is_adhoc = True
  else:
    if isinstance(
        path, importlib_resources._adapters.CompatibilityFiles.SpecPath  # pylint: disable=protected-access
    ):
      is_adhoc = True
    else:
      is_adhoc = False

  if is_adhoc:
    # TODO(b/260333695): `importlib_resources` fail with adhoc imports
    # When module are imported with adhoc, `importlib_resources.files` returns
    # a non-path object, so convert manually.
    # Note this is not the true path (`/google_src/` vs
    # `/export/.../server/ml_notebook.runfiles`), but should be equivalent.
    if isinstance(package, types.ModuleType):
      # TODO(b/390190120): We cannot use `.__name__` directly as adhoc import
      # behave inconsistently.
      package = getattr(package.__spec__, 'name', package.__name__)  # pytype: disable=attribute-error
    path = pathlib.Path(sys.modules[package].__file__)
    if path.name == '__init__.py':
      path = path.parent

  # pylint: disable=undefined-variable
  if isinstance(path, pathlib.Path):
    # TODO(etils): To ensure compatibility with zipfile.Path, we should ensure
    # that the returned `pathlib.Path` isn't missused. More specifically:
    # * `os.fspath` should only be called on files (not directories)
    # * `str(path)` should be forbidden (only `__format__` allowed).
    # In practice, it is trickier to do as `__fspath__` and `__str__` are
    # called internally.
    # Convert to `GPath` for consistency and compatibility with `MockFs`.
    return abstract_path.Path(path)
  elif isinstance(path, zipfile.Path):
    path = ResourcePath(path.root, path.at)
    return typing.cast(abstract_path.Path, path)
  elif isinstance(path, importlib_resources.abc.Traversable):
    # Is seems like `importlib_resources.files` can return additional types,
    # like `MultiplexedPath`.
    # Fallback to avoid failure, however those objects might not implement
    # `__fspath__`, so might fail later.
    return typing.cast(abstract_path.Path, path)
  else:
    raise TypeError(f'Unknown resource path: {type(path)}: {path}')
  # pylint: enable=undefined-variable


def to_write_path(path: abstract_path.Path) -> abstract_path.Path:
  """Cast the `epath.resource_path` to a read-write Path."""
  return path
