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

"""Adhoc imports utils."""

from __future__ import annotations

import functools
import sys
import types
import typing
from typing import NoReturn

from etils.epy import py_utils


@functools.cache
def path_to_module_name(path: str) -> str:
  """Normalizes paths to module names."""
  if '/' not in path:  # Already a module name
    return path
  path = path.lstrip('/')
  return (
      path
      .removesuffix('.py')
      .removesuffix('/__init__')
      .replace('/', '.')
  )


def get_module_names(
    restrict: py_utils.StrOrStrList, *, recursive: bool = True
) -> list[str]:
  """Returns all `sys.modules` matching the restrict name."""

  modules = py_utils.normalize_str_to_list(restrict)
  assert all('/' not in module for module in modules)

  # List all the currently loaded modules matching `modules`
  if recursive:
    modules = tuple(modules)
    return [m for m in sys.modules if m.startswith(modules)]
  else:
    modules = set(modules)
    return [m for m in sys.modules if m in modules]


def clear_cached_modules(
    modules: py_utils.StrOrStrList,
    *,
    recursive: bool = True,
    verbose: bool = False,
    invalidate: bool = True,
) -> None:
  """Clear the `sys.modules` cache.

  Helpful for interactive development to reload from Jupyter notebook the
  code we're currently editing (without having to restart the notebook kernel).

  Usage:

  ```python
  ecolab.clear_cached_modules(['visu3d', 'other_module.submodule'])

  import visu3d
  import other_module.submodule
  ```

  Args:
    modules: List of modules to clear
    recursive: Whether submodules are cleared too
    verbose: Whether to display the list of modules cleared.
    invalidate: If `True` (default), the instances of the module will raise an
      error when used (to avoid using 2 versions of a module at the same time)
  """
  modules_to_clear = get_module_names(modules, recursive=recursive)
  if not modules_to_clear:
    return

  modules = set(py_utils.normalize_str_to_list(modules))

  for module_name in modules_to_clear:
    if verbose:
      print(f'Clearing {module_name}')

    # We do not invalidate ecolab
    # Note that `reload=['etils']` will still clear `ecolab` from `sys.modules`
    # even if it is not reloaded. In practice, this is fine as `ecolab`
    # should only be imported once in the colab.
    invalidate_curr = invalidate and not module_name.startswith('etils')

    # Only the top-most attribute should be cleared. when `invalidate=False`
    # Otherwise, childs are not re-imported
    # Clear the parent ref to the module
    if invalidate_curr or module_name in modules:
      _clear_parent_module_attr(module_name)

    if invalidate_curr:
      # Mutate the existing modules to raise an error if accessed
      _invalidate_module(sys.modules[module_name])

    del sys.modules[module_name]

  # The typing module has side effect by caching `A[B]` from the old modules
  # but thankfully they expose the cleanup method.
  for cleanup in typing._cleanups:  # pytype: disable=module-attr  # pylint: disable=protected-access
    cleanup()


def _clear_parent_module_attr(module_name: str) -> None:
  """Remove parent reference (e.g. `path.to.child` -> `path.to`."""
  if '.' not in module_name:
    return
  parent_module_name, child_name = module_name.rsplit('.', 1)
  parent_module = sys.modules.get(parent_module_name)
  if not parent_module:
    return
  if hasattr(parent_module, child_name):
    delattr(parent_module, child_name)


def _invalidate_module(module: types.ModuleType) -> None:
  """Invalidate the module in-place."""
  module_name = module.__name__

  # https://peps.python.org/pep-0562/
  def __getattr__(name: str) -> NoReturn:  # pylint: disable=invalid-name
    """All module attribute access will raise error."""
    raise AttributeError(
        f'Cannot access {module_name}.{name} on the old module'
        f' instance.\n{module_name} was reloaded, so the new module should be'
        ' used instead.\nYou can pass `invalidate=False` to keep both old and'
        ' reloaded modules.'
    )

  module.__dict__.clear()
  module.__getattr__ = __getattr__
  # Allow `adhoc_error` to detect invalidated modules and raise better errors
  module.__etils_invalidated__ = True
