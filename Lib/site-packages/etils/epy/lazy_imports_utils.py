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

"""Lazy import utils."""

# Forked from TFDS

# TODO(epot): Could try to unify with
# - etils/ecolab/lazy_utils.py
# - kauldron/konfig/fake_import_utils.py
# - visu3d/utils/py_utils.py

from __future__ import annotations

import builtins
import collections
import contextlib
import dataclasses
import functools
import importlib
import sys
import threading
import types
from typing import Any, Callable, ContextManager, Iterator

from etils.epy import reraise_utils
from etils.epy.adhoc_utils import curr_args


_ErrorCallback = Callable[[Exception], None]
_SuccessCallback = Callable[[str], None]

# Store a lock per module to avoid problems with multiple threads trying to
# import the same module at the same time.
_LOCK_PER_MODULE = collections.defaultdict(threading.Lock)


@dataclasses.dataclass(kw_only=True)
class LazyModule:
  """Module loaded lazily during first call."""

  module_name: str
  adhoc_kwargs: dict[str, Any] | None
  error_callback: str | _ErrorCallback | None
  success_callback: _SuccessCallback | None
  _submodules: dict[str, LazyModule] = dataclasses.field(default_factory=dict)

  def __post_init__(self):
    if self.adhoc_kwargs is not None:
      self.adhoc_kwargs = dict(self.adhoc_kwargs)
      # Whe the lazy import is triggered, do not reload modules, nor trigger a
      # full build.
      self.adhoc_kwargs.pop("reload", None)
      self.adhoc_kwargs.pop("build_targets", None)
      self.adhoc_kwargs.pop("reload_workspace", None)
      self.adhoc_kwargs.pop("cell_autoreload", None)
      self.adhoc_kwargs.pop("restrict_reload", None)

  @functools.cached_property
  def _module(self) -> types.ModuleType:
    """Resolve the module."""
    # Try to import the module, eventually replaying the adhoc scope
    with self._maybe_adhoc():
      try:
        # When multiple threads try to import the same module, make sure only
        # one of them is importing it at the same time.
        with _LOCK_PER_MODULE[self.module_name]:
          module = importlib.import_module(self.module_name)
      except ImportError as e:
        if self.error_callback is not None:
          if isinstance(self.error_callback, str):
            reraise_utils.reraise(e, suffix=f"\n{self.error_callback}")
          else:
            self.error_callback(e)
        raise
      except AttributeError as e:
        # If `self._module` raises an `AttributeError`, this will trigger
        # `self.__getattr__('_module')`, triggering an infinite recursion.
        # To avoid this, we re-raise the `AttributeError` as an `ImportError`.
        raise ImportError(f"Error importing {self.module_name}: {e!r}") from e
    if self.success_callback is not None:
      self.success_callback(self.module_name)
    return module

  def _maybe_adhoc(self) -> ContextManager[None]:
    """Recreate the adhoc import context used during the original import."""
    if self.adhoc_kwargs is None or self.module_name in sys.modules:
      adhoc = contextlib.nullcontext()
    else:
      # If the lazy-import is resolved from within an adhoc, keep the adhoc.
      # Is it the right thing to do ?
      if curr_args.get_curr_adhoc_kwargs() is not None:
        adhoc = contextlib.nullcontext()
      else:
        adhoc = curr_args.replay_adhoc_ctx(
            **self.adhoc_kwargs,
            collapse_prefix=f"Lazy-import {self.module_name!r}: ",
        )

    return adhoc

  def __getattr__(self, name: str) -> Any:
    if name in self._submodules:
      # known submodule accessed. Do not trigger import
      return self._submodules[name]
    else:
      return getattr(self._module, name)

  # TODO(epot): Also support __setattr__


def _register_submodule(module: LazyModule, name: str) -> LazyModule:
  child_module = LazyModule(
      module_name=f"{module.module_name}.{name}",
      adhoc_kwargs=module.adhoc_kwargs,
      error_callback=module.error_callback,
      success_callback=module.success_callback,
  )
  module._submodules[name] = child_module  # pylint: disable=protected-access
  return child_module


@contextlib.contextmanager
def lazy_imports(
    *,
    error_callback: str | _ErrorCallback | None = None,
    success_callback: _SuccessCallback | None = None,
) -> Iterator[None]:
  """Context Manager which lazy loads packages.

  Their import is not executed immediately, but is postponed to the first
  call of one of their attributes.

  Limitation:

  - You can only lazy load modules (`from x import y` will not work if `y` is a
    constant or a function or a class).

  Usage:

  ```python
  with epy.lazy_imports():
    import tensorflow as tf

  with epy.lazy_imports(success_callback=check_tf_version):
    import tensorflow as tf
  ```

  When using type annotations, make sure to also use
  `from __future__ import annotations`, otherwise the lazy-import will be
  triggered at import time when used in typing annotations:

  ```pthon
  with epy.lazy_imports():
    import tensorflow as tf

  # !!! Resolve the lazy-import when `__future__.annotations` isn't present
  def get_dataset() -> tf.data.Dataset:
  ```

  This support `ecolab.adhoc` imports: When the lazy-import is resolved,
  the original `ecolab.adhoc` context is re-created to import the lazy module.

  Args:
    error_callback: A additional message to append to the `ImportError` if the
      import fails. Can also be a `Callable[[Exception], None]`. The exception
      is passed as an arg, so user can use `epy.reraise(e, 'Additional
      message')`.
    success_callback: a callback to trigger when an import succeeds. The
      callback is passed the name of the imported module as an arg.

  Yields:
    None
  """
  # Need to mock `__import__` (instead of `sys.meta_path`, as we do not want
  # to modify the `sys.modules` cache in any way)
  original_import = builtins.__import__
  try:
    builtins.__import__ = functools.partial(
        _lazy_import,
        error_callback=error_callback,
        success_callback=success_callback,
    )
    yield
  finally:
    builtins.__import__ = original_import


def _lazy_import(
    name: str,
    globals_=None,
    locals_=None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
    *,
    error_callback: str | _ErrorCallback | None,
    success_callback: _SuccessCallback | None,
):
  """Mock of `builtins.__import__`."""
  del globals_, locals_  # Unused

  if level:
    raise ValueError(f"Relative import statements not supported ({name}).")

  root_name, *parts = name.split(".")
  root = LazyModule(
      module_name=root_name,
      adhoc_kwargs=curr_args.get_curr_adhoc_kwargs(),
      error_callback=error_callback,
      success_callback=success_callback,
  )

  # Extract inner-most module
  child = root
  for name in parts:
    child = _register_submodule(child, name)

  if fromlist:
    # from x.y.z import a, b

    for fl in fromlist:
      _register_submodule(child, fl)

    return child  # return the inner-most module (`x.y.z`)
  else:
    # import x.y.z
    # import x.y.z as z
    return root  # return the top-level module (`x`)
