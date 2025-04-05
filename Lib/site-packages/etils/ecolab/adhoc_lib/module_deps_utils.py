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

"""Utils to compute module deps."""

from __future__ import annotations

import dataclasses
import inspect
import sys
import types

from etils import epy


@dataclasses.dataclass(kw_only=True, slots=True)
class _ModuleDeps:
  name: str
  importing: dict[str, _ModuleDeps] = dataclasses.field(default_factory=dict)
  imported_in: dict[str, _ModuleDeps] = dataclasses.field(default_factory=dict)

  def __repr__(self) -> str:
    return epy.Lines.make_block(
        header=type(self).__name__,
        content={
            'name': self.name,
            # Only display the list to avoid infinite recursion
            'importing': list(self.importing),
            'imported_in': list(self.imported_in),
        },
    )


def get_all_module_deps() -> dict[str, _ModuleDeps]:
  """Construct the graph of the imported nodes."""
  # Track all modules deps
  all_module_deps = {}

  # TODO(epot): This should be optimized with caching (no need to recompute
  # the full graph at each reload). However be careful:
  # * Make sure syncing workspace re-trigger a full graph computation
  # * How to make sure the lazy-imports are properly updated
  # * Should cache be shared across adhoc calls ?

  for k, module in sys.modules.items():
    # Filter built-in, lazy-imports
    if inspect.getattr_static(module, '__file__', None) is None:
      continue

    module_deps = _get_module_deps(all_module_deps, k)

    for imported_module in module.__dict__.values():
      if not _ismodule(imported_module):
        continue

      imported_module_name = inspect.getattr_static(
          imported_module, '__name__', None
      )
      if imported_module_name is None:  # Filter lazy-imports
        continue

      imported_module_deps = _get_module_deps(
          all_module_deps, imported_module_name
      )
      module_deps.importing[imported_module_deps.name] = imported_module_deps
      imported_module_deps.imported_in[module_deps.name] = module_deps
  return all_module_deps


def _ismodule(obj: object) -> bool:
  # Do not use `isinstance` as it trigger `.__getattribute__('__class__')`
  # implemented by some lazy objects (like TFDS `LazyBuilderImport`)
  return issubclass(type(obj), types.ModuleType)


def _get_module_deps(
    all_module_deps: dict[str, _ModuleDeps], module_name: str
) -> _ModuleDeps:
  if module_name not in all_module_deps:
    all_module_deps[module_name] = _ModuleDeps(name=module_name)
  return all_module_deps[module_name]
