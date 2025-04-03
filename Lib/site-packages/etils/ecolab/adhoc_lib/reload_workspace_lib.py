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

"""Reload workspace util."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import contextlib
import importlib
import inspect
import re
import sys
import types
import typing

from etils.ecolab import inplace_reload
from etils.ecolab.adhoc_lib import module_deps_utils
from etils.epy import py_utils
from etils.epy.adhoc_utils import module_utils
import IPython


if typing.TYPE_CHECKING:
  # Do not import here to avoid circular deps:
  # `g3_utils` uses `epy.lazy_imports` that call `get_curr_adhoc_kwargs()` from
  # `ecolab`.
  from etils import g3_utils  # pylint: disable=g-bad-import-order

_SKIP_RELOAD = 'etils'


def reload_workspace(
    source: g3_utils.Source | None = None,
    *,
    restrict: py_utils.StrOrStrList | None = None,
    verbose: bool = False,
    reload_mode: (
        inplace_reload.ReloadMode | str
    ) = inplace_reload.ReloadMode.INVALIDATE,
) -> None:
  """Reload all modified files in the current workspace.

  This function look at all edited files in the given workspace which are also
  imported and reload them.

  This is a no-op if the source is not a user workspace.

  Args:
    source: Same as `ecolab.adhoc`
    restrict: Same as `ecolab.adhoc`
    verbose: Same as `ecolab.adhoc`
    reload_mode: Same as `ecolab.adhoc`
  """

  from etils import g3_utils  # pylint: disable=g-import-not-at-top
  from etils.ecolab import adhoc_imports  # pylint: disable=g-import-not-at-top

  # Normalize inputs
  citc_info = g3_utils.citc_info_from_source(source)
  if isinstance(citc_info, g3_utils.PendingCl):
    citc_info = citc_info.workspace
  restrict = py_utils.normalize_str_to_list(restrict)

  # TODO(epot): When loading from workspace, then from HEAD, how to detect
  # which files should be reloaded ?
  # Especially when there's 2 different `ecolab.adhoc` scope!
  # Could detect all modules that are adhoc-imported (through the
  # `module.__file__` and reload them)
  if not isinstance(citc_info, g3_utils.Workspace):
    if prev_modules := _find_modules_imported_in_different_source(source):
      prev_module_name = next(iter(prev_modules))
      prev_source = sys.modules[prev_module_name]._etils_workspace_reload_source  # pylint: disable=protected-access
      raise ValueError(
          f'Source was changed from {prev_source!r} to {source!r}.'
          ' `reload_workspace=True` cannot auto-infer which modules to reload.'
          ' Please restart your kernel.'
      )
    else:
      return

  # Step 1: List all modules to reload
  modules_to_reload = _get_modules_to_reload(
      restrict=restrict,
      citc_info=citc_info,
  )
  if not modules_to_reload:
    return

  # Step 2: Reload
  with adhoc_imports.adhoc(
      source,
      reload=modules_to_reload,
      restrict=restrict,
      restrict_reload=False,
      reload_recursive=False,
      reload_mode=reload_mode,
      verbose=verbose,
      collapse_prefix=f'Reload workspace ({len(modules_to_reload)} modules): ',
  ):
    for module_name in modules_to_reload:
      module = importlib.import_module(module_name)
      module._etils_workspace_reload_source = source  # pylint: disable=protected-access

  # Step 3: Replace the reloaded modules in the Colab kernel.
  update_global_namespace(reload=modules_to_reload, verbose=verbose)


def _get_modules_to_reload(
    *,
    citc_info: g3_utils.Workspace,
    restrict: list[str],
) -> list[str]:
  """Find all modules to reload."""
  # Find the modules to reload:
  # * Load all files openned in the workspace
  # * Only keep the `.py` files which are in the `sys.modules`
  # * Check if the module has been changed since the import time.
  # * Traverse the tree of import dependencies to find out all modules affected
  #   by the changes.
  opened_files = _all_opened_files(citc_info)
  opened_module_paths = _filter_only_imported_modules(opened_files)
  edited_modules = _filter_recently_edited(
      module_paths=opened_module_paths,
      citc_info=citc_info,
  )
  modules_to_reload = _find_all_affected_modules(
      module_names=edited_modules,
      restrict=restrict,
  )
  return list(modules_to_reload)


def _filter_only_imported_modules(opened_files: Iterable[str]) -> list[str]:
  """Restrict the list of files to Python modules already in sys.modules."""
  all_opened_modules = []
  for path in opened_files:
    # TODO(epot): supports proto
    if not path.endswith('.py'):
      continue
    if module_utils.path_to_module_name(path) not in sys.modules:
      continue
    all_opened_modules.append(path)

  return all_opened_modules


# Note that reloading etils will reset the cache, triggering a full reload of
# all edited modules. This is likely the best behavior.
# Otherwise, we could try to attach the cache to the `sys.modules` directly.
_MODULES_EDIT_TIME = {}


def _filter_recently_edited(
    *,
    citc_info: g3_utils.Workspace,
    module_paths: list[str],
) -> list[str]:
  """Only keep modules that have been updated."""
  recently_edited = []

  for module_path in module_paths:
    # TODO(epot): could also check that `sys.modules[].__file__` matches
    # `citc_info.g3_path`, otherwise, should force reload.
    module_name = module_utils.path_to_module_name(module_path)
    curr_time = (citc_info.g3_path.parent / module_path).stat().mtime
    prev_time = _MODULES_EDIT_TIME.get(module_name)

    if prev_time == curr_time:  # File already cached. No updates
      continue
    _MODULES_EDIT_TIME[module_name] = curr_time
    recently_edited.append(module_name)
  return recently_edited


def _find_all_affected_modules(
    *,
    module_names: list[str],
    restrict: list[str],
) -> set[str]:
  """Find all affected modules."""
  if not module_names:
    return set()

  all_module_deps = module_deps_utils.get_all_module_deps()

  restrict_exact = set(restrict)
  restrict_prefix = tuple(f'{r}.' for r in restrict)

  def _filter_restrict(names: Iterable[str]) -> list[str]:
    # etils modifications are not reloaded (etils is used in too many places)
    names = [n for n in names if not n.startswith(_SKIP_RELOAD)]
    if not restrict:
      return names
    names = [
        n for n in names if n in restrict_exact or n.startswith(restrict_prefix)
    ]
    return names

  module_names = _filter_restrict(module_names)

  affected_modules = set(module_names)
  modules_to_process = set(module_names)
  visited = set()

  while modules_to_process:
    module_name = modules_to_process.pop()
    if module_name in visited:
      continue
    visited.add(module_name)
    module_deps = all_module_deps[module_name]

    imported_in = _filter_restrict(module_deps.imported_in)
    modules_to_process.update(imported_in)
    affected_modules.update(imported_in)
  return affected_modules


def _find_modules_imported_in_different_source(
    source: g3_utils.Source,
) -> set[str]:
  """Extract all modules imported in a different source."""
  sentinel = object()
  other_previous_modules = set()
  for module_name, module in sys.modules.items():
    old_source = inspect.getattr_static(
        module, '_etils_workspace_reload_source', sentinel
    )
    if old_source is sentinel:
      continue
    if old_source != source:
      other_previous_modules.add(module_name)
  return other_previous_modules


@contextlib.contextmanager
def mark_adhoc_imported_modules(
    source: None | g3_utils.Source,
) -> Iterator[None]:
  """Mark modules that were adhoc imported from this context.

  This allow `reload_workspace=True` to raise an error if the source change.

  Args:
    source: The source of the adhoc import

  Yields:
    None
  """
  old_modules = set(sys.modules)
  try:
    yield
  finally:
    adhoc_imported_modules = set(sys.modules) - old_modules
    for module_name in adhoc_imported_modules:
      # TODO(epot): Could check that the `source` and `.__file__` match
      sys.modules[module_name]._etils_workspace_reload_source = source  # pylint: disable=protected-access


def update_global_namespace(
    *,
    reload: list[str],
    verbose: bool,
) -> None:
  """Overwrite the imported modules in the current Colab global namespace."""
  reload = set(reload)

  ip = IPython.get_ipython()
  user_ns = ip.kernel.shell.user_ns

  # Filter only the modules
  # This means that `from module import function` or `from module import *`
  # won't be reloaded
  # Note that we overwrite all modules which match, not only the ones defined
  # inside the `adhoc` contextmanager. It's not trivial to detect when a module
  # is re-imported, like:
  #
  # import module
  # with ecolab.adhoc():
  #   import module  # < globals() not modified, difficult to detect
  #
  for name, module in dict(user_ns).items():
    # We look at all globals, not just the ones defined inside the
    # contextmanager.
    # The solution would be to mock `__import__` to capture all statements
    # but over-engineered for now.
    if not isinstance(module, types.ModuleType):
      continue  # The object is not a module

    # `getattr_static` for `lazy_imports` modules
    module_name = inspect.getattr_static(module, '__name__', None)
    if module_name not in reload:
      continue  # The module not reloaded
    if re.fullmatch(r'_+(\d+)?', name):
      continue  # Internal IPython variables (`_`, `__`, `_12`)

    if verbose:
      print(f'Overwrting Colab global {name!r} to new module ({module_name!r})')

    reloaded_module = sys.modules[module_name]
    user_ns[name] = reloaded_module
