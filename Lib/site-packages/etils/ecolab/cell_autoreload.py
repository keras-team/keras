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

"""Cell auto-reload."""

from __future__ import annotations

import functools
import importlib
import inspect
import sys

from etils import epath
from etils.ecolab import adhoc_imports
from etils.ecolab import ip_utils
from etils.ecolab.adhoc_lib import reload_workspace_lib
from etils.epy.adhoc_utils import module_utils


def _create_module_graph(nodes: set[str]) -> dict[str, set[str]]:
  graph = {}

  for source in nodes:
    deps = set()
    for val in sys.modules[source].__dict__.values():
      if inspect.ismodule(val) and val.__name__ in nodes:
        deps.add(val.__name__)
    graph[source] = deps

  return graph


class _ModuleSearch:
  """Graph of module dependencies that can be queried."""

  def __init__(self, targets: set[str], graph: dict[str, set[str]]):
    self._graph = graph
    self._cache: dict[str, bool] = {}
    self._targets = targets

  def _reaches_targets(self, source: str) -> bool:
    """Check if a module references other modules directly or indirectly."""
    queue = [source]
    visited = set(queue)

    while queue:
      m = queue.pop(0)

      if (reaches := self._cache.get(m)) is not None:
        if reaches:
          # If m is known to reach target -> source reaches targets
          return True
        else:
          # Otherwise, no need to search the neighbours of this node either.
          continue

      if m in self._targets:
        return True

      for neighbour in self._graph.get(m, set()):
        if neighbour not in visited:
          visited.add(neighbour)
          queue.append(neighbour)

    return False

  def reaches_targets(self, source: str) -> bool:
    """Check if a module references other modules directly or indirectly."""
    ret = self._reaches_targets(source)
    self._cache[source] = ret
    return ret


class ModuleReloader:
  """Module reloader."""

  def __init__(self, **adhoc_kwargs):
    self.adhoc_kwargs = adhoc_kwargs
    self._last_updates: dict[str, int | None] = {}

  @functools.cached_property
  def reload(self) -> tuple[str, ...]:
    return tuple(self.adhoc_kwargs['reload'])

  @property
  def verbose(self) -> bool:
    return self.adhoc_kwargs['verbose']

  def register(self) -> None:
    if not self.reload:
      raise ValueError('`cell_autoreload=True` require to set `reload=`')

    # Keep a value for each module. If a file is updated, trigger a reload.
    for module in module_utils.get_module_names(self.reload):
      self._last_updates[module] = _get_last_module_update(module)

    # Currently, only a single auto-reload can be set at the time.
    # Probably a good idea as it's unclear how to differentiate between
    # registering 2 cell_autoreload and overwriting cell_autoreload params.
    ip_utils.register_once(
        'pre_run_cell',
        # Cannot use `self.method` because bound methods do not support
        # set attribute.
        functools.partial(type(self)._pre_run_cell_maybe_reload, self),
        'is_cell_auto_reload',
    )

  def _pre_run_cell_maybe_reload(
      self,
      *args,
  ) -> None:
    """Check if workspace is modified, then eventually reload modules."""
    del args  # Future version of IPython will have a `info` arg

    # TODO(epot): This function could be unified with `reload_workspace`

    # If any of the modules has been updated, trigger a reload

    # Find which modules are dirty.
    dirty_modules: set[str] = set()
    for module in module_utils.get_module_names(self.reload):
      prev_mtime = self._last_updates.get(module)
      new_mtime = _get_last_module_update(module)
      if prev_mtime is None or (
          new_mtime is not None and new_mtime > prev_mtime
      ):
        dirty_modules.add(module)
      self._last_updates[module] = new_mtime

    if not dirty_modules:
      return

    # Get set of all modules we could potentially reload.
    reload_set = set(module_utils.get_module_names(self.reload))
    graph = _create_module_graph(reload_set)
    search = _ModuleSearch(dirty_modules, graph)

    # Narrow it down to modules that are dirty or reference a dirty module.
    modules_to_reload = [
        mod for mod in reload_set if search.reaches_targets(mod)
    ]

    # Only reload exactly the modules we know are dirty. reload_recursive
    # is an undocumented flag in adhoc for now.
    adhoc_kwargs = self.adhoc_kwargs | {
        'reload': modules_to_reload,
        'reload_recursive': False,
        'collapse_prefix': f'Autoreload ({len(modules_to_reload)} modules): ',
    }
    with adhoc_imports.adhoc(**adhoc_kwargs):
      for module in modules_to_reload:
        importlib.import_module(module)

      # Update globals in user namespace with reloaded modules
      reload_workspace_lib.update_global_namespace(
          reload=modules_to_reload,
          verbose=self.verbose,
      )


def _get_last_module_update(module_name: str) -> int | None:
  """Get the last update for one module."""
  module = sys.modules.get(module_name, None)
  if module is None:
    return None
  if module.__name__ == '__main__':
    return None

  module_file = getattr(module, '__file__', None)
  if not module_file:
    return None

  module_file = epath.Path(module_file)

  try:
    return module_file.stat().mtime
  except OSError:
    return None
