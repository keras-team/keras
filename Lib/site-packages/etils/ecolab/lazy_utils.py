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

"""lazy imports implementation."""

from __future__ import annotations

import builtins
import contextlib
import dataclasses
import importlib
import traceback
import types
from typing import Any, Iterator, Optional

from etils import epy


# Attributes which will be updated after the module is loaded.
_MODULE_ATTR_NAMES = [
    '__builtins__',
    '__cached__',
    '__doc__',
    '__file__',
    '__loader__',
    '__name__',
    '__package__',
    '__path__',
    '__spec__',
]


@dataclasses.dataclass(eq=False)
class LazyModuleState:
  """State of the lazy module.

  We store the state in a separate object to:

  1) Reduce the risk of collision
  2) Avoid infinite recursion error when typo on a attribute
  3) `@property`, `functools.cached_property` fail when the class is changed

  Attributes:
    module_name: E.g. `jax.numpy`
    alias: E.g. `jnp`
    is_std: Whether the module is part of the standard library (to order
      imports)
    host: `LazyModule` attached to the state
    extra_imports: Additional extra imports to trigger (e.g. `concurrent`
      trigger `concurrent.futures` import)
    _module: Cached original imported module
    trace_repr: Track the trace which trigger the import (Helpful to debug) E.g.
      `Colab` call '.getdoc' on the background, which trigger import.
  """

  module_name: str
  alias: str
  is_std: bool = dataclasses.field(repr=False, default=False)
  host: LazyModule = dataclasses.field(repr=False, default=None)
  extra_imports: list[str] = dataclasses.field(default_factory=list)
  _module: Optional[types.ModuleType] = None
  # Track the trace which trigger the import
  # Helpful to debug.
  # E.g. `Colab` call '.getdoc' on the background, which trigger import.
  trace_repr: Optional[str] = dataclasses.field(default=None, repr=False)

  @property
  def module(self) -> types.ModuleType:
    """Returns the module."""
    if not self.module_loaded:  # Load on first call
      # Keep track of attributes which triggered import
      # Used to track ipython internals (e.g. `<module>.get_traits` gets called
      # internally when ipython inspect the object)
      # So writing `<module>.` trigger module loading & auto-completion even if
      # the module was never used before.
      self.trace_repr = ''.join(traceback.format_stack())

      self._module = _load_module(
          self.module_name,
          extra_imports=self.extra_imports,
      )
      # Update the module.__doc__, module.__file__,...
      self._mutate_host()
    return self._module

  @property
  def module_loaded(self) -> bool:
    """Returns `True` if the module is loaded."""
    return self._module is not None

  @property
  def import_statement(self) -> str:
    """Returns the `import xyz` statement."""
    # Possible cases:
    # `import abc.xyz`
    # `import abc.xyz as def`
    # `from abc import xyz`
    # `from abc import xyz as def` (currently, never used)
    if self.module_name == self.alias:
      return f'import {self.module_name}'

    if '.' in self.module_name:
      left_import, right_import = self.module_name.rsplit('.', maxsplit=1)
      if right_import == self.alias:
        return f'from {left_import} import {right_import}'

    # TODO(epot): Also add extra imports ?
    return f'import {self.module_name} as {self.alias}'

  def _mutate_host(self) -> None:
    """When the module is first loaded, update `__doc__`, `__file__`,..."""
    assert self.module_loaded
    missing = object()
    for attr_name in _MODULE_ATTR_NAMES:
      attr_value = getattr(self.module, attr_name, missing)
      if attr_value is not missing:
        object.__setattr__(self.host, attr_name, attr_value)


# Class name has to be `module` for Colab compatibility (colab hardcodes class
# name instead of checking the instance)
class module(types.ModuleType):  # pylint: disable=invalid-name
  """Lazy module which auto-loads on first attribute call."""

  _etils_state: LazyModuleState

  def __init__(self, state: LazyModuleState):
    # We set `__file__` to None, to avoid `colab_import_.reload_package(etils)`
    # to trigger a full reload of all modules here.
    object.__setattr__(self, '__file__', None)
    object.__setattr__(self, '_etils_state', state)
    assert state.host is None
    state.host = self

  def __getattr__(self, name: str) -> Any:
    if not self._etils_state.module_loaded and name in {
        'getdoc',
        '__wrapped__',
    }:
      # IPython dynamically inspect the object when hovering the symbol:
      # This can trigger a slow import which then disable rich annotations:
      # So raising attribute error avoid lazy-loading the module.
      # There might be a more long term fix but this should cover the most
      # common cases.
      raise AttributeError
    return getattr(self._etils_state.module, name)

  def __setattr__(self, name: str, value: Any) -> None:
    # Overwrite the module attribute
    setattr(self._etils_state.module, name, value)

  def __dir__(self) -> list[str]:  # Used for Colab auto-completion
    return dir(self._etils_state.module)

  def __repr__(self) -> str:
    if not self._etils_state.module_loaded:
      return f'LazyModule({self._etils_state.module_name!r})'
    else:
      module_ = self._etils_state.module
      if hasattr(module_, '__file__'):
        file = module_.__file__
      else:
        file = '(built-in)'
      return f'<lazy_module {module_.__name__!r} from {file!r}>'


# Create alias to avoid confusion
LazyModule = module
del module


# Modules here are imported from head (missing from the Brain Kernel)
_PACKAGE_RESTRICT = [
    'dataclass_array',
    'etils',
    'lark',
    'sunds',
    'visu3d',
    'imageio',
    'mediapy',
    'pycolmap',
]


# TODO(epot): Rather than hardcoding which modules are adhoc-imported, this
# could be a argument.
def _load_module(
    module_name: str,
    *,
    extra_imports: list[str],
) -> types.ModuleType:
  """Load the module, eventually using adhoc-import."""
  adhoc_cm = contextlib.suppress()

  # First time, load the module
  with adhoc_cm:
    for extra_import in extra_imports:
      # Hardcoded hack to not import tqdm.notebook on non-Colab env
      if extra_import == 'tqdm.notebook' and not epy.is_notebook():
        continue
      importlib.import_module(extra_import)
    return importlib.import_module(module_name)


class LazyImportsBuilder:
  """Capture import statements and replace them by lazy-import equivalement."""

  def __init__(self, globals_):
    self._globals = globals_
    self.lazy_modules: dict[str, LazyModule] = {}

  @contextlib.contextmanager
  def replace_imports(self, *, is_std: bool) -> Iterator[None]:
    """Replace import statement by their lazy equivalent."""
    # Step 1: Capture all imports by `_ModuleImportProxy`.

    # Need to mock `__import__` (instead of `sys.meta_path`, as we do not want
    # to modify the `sys.modules` cache in any way)
    original_import = builtins.__import__
    try:
      builtins.__import__ = _lazy_import
      yield
    finally:
      builtins.__import__ = original_import

    # Step 1: Replace all `_ModuleImportProxy` by the actual lazy `LazyModule`.

    # We need 2 steps otherwise we have no way of knowing the alias used,
    # for example to discriminating between:
    # `import concurrent.futures` => `LazyModule('concurent')`
    # `import concurrent.futures as xxx` => `LazyModule('concurent.future')`

    for k, v in list(self._globals.items()):  # List to allow mutating `globals`
      if isinstance(v, _ModuleImportProxy):
        state = LazyModuleState(
            module_name=v.qualname,
            alias=k,
            extra_imports=v.leaves_qualnames,
            is_std=is_std,
        )
        lazy_module = LazyModule(state)
        self.lazy_modules[k] = lazy_module
        self._globals[k] = lazy_module


def _lazy_import(
    name: str,
    globals_=None,
    locals_=None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
):
  """Mock of `builtins.__import__`."""
  del globals_, locals_  # Unused
  if level:
    raise ValueError(f'Relative import statements not supported ({name}).')

  root_name, *parts = name.split('.')
  root = _ModuleImportProxy(name=root_name)

  # Extract inner-most module
  child = root
  for name in parts:
    child = getattr(child, name)

  if fromlist:
    # from x.y.z import a, b
    return child  # return the inner-most module (`x.y.z`)
  else:
    # import x.y.z
    # import x.y.z as z
    return root  # return the top-level module (`x`)


@dataclasses.dataclass(eq=False)
class _ModuleImportProxy:
  """`_ModuleImportProxy` replace all modules during import statement.

  ```python
  with LazyImportsBuilder().replace_imports():
    import abc.def
    assert isinstance(abc.def, _ModuleImportProxy)
  ```
  """

  name: str
  parent: Optional[_ModuleImportProxy] = None
  children: dict[str, _ModuleImportProxy] = dataclasses.field(
      default_factory=dict
  )

  @property
  def qualname(self) -> str:
    if not self.parent:
      return self.name
    else:
      return f'{self.parent.qualname}.{self.name}'

  @property
  def leaves_qualnames(self) -> list[str]:
    """Extract all qualnames of leaves children."""
    all_children = []
    for children in self.children.values():
      all_children.extend(
          leaves_qualnames
          if (leaves_qualnames := children.leaves_qualnames)
          else [children.qualname]  # Child is a leave
      )
    return all_children

  def __repr__(self) -> str:
    if self.leaves_qualnames:
      child_arg = f', children={self.leaves_qualnames}'
    else:
      child_arg = ''
    return f'{type(self).__name__}({self.qualname}{child_arg})'

  def __getattr__(self, name: str):
    if name not in self.children:
      self.children[name] = type(self)(
          name=name,
          parent=self,
      )
    return self.children[name]


def current_import_statements(lazy_modules: dict[str, LazyModule]) -> str:
  """Returns the lazy import statement string."""
  lines = []

  lazy_modules = [m._etils_state for m in lazy_modules.values()]  # pylint: disable=protected-access
  used_lazy_modules = [
      # For convenience, we do not add the `lazy_imports` import
      m
      for m in lazy_modules
      if m.module_loaded and m.alias != 'lazy_imports'
  ]
  std_modules = [m.import_statement for m in used_lazy_modules if m.is_std]
  non_std_modules = [
      m.import_statement for m in used_lazy_modules if not m.is_std
  ]

  # Import standard python module first, then other modules
  lines.extend(std_modules)
  if std_modules and non_std_modules:
    lines.append('')  # Empty line
  lines.extend(non_std_modules)  # pylint: disable=protected-access
  return '\n'.join(lines)
