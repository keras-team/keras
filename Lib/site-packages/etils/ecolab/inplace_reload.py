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

"""Hot reload."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import enum
import functools
import gc
import inspect
import sys
import time
import types
import typing
from typing import Any, Iterator
import weakref

from etils import epy
from etils.epy.adhoc_utils import module_utils


class ReloadMode(epy.StrEnum):
  """Invalidate mode.

  When reloading a module, indicate what to do with the old module instance.

  Attributes:
    UPDATE_INPLACE: Update the old module instances with the new reloaded values
    INVALIDATE: Clear the old module so it cannot be reused.
    KEEP_OLD: Do not do anything, so 2 versions of the module coohexist at the
      same time.
  """
  UPDATE_INPLACE = enum.auto()
  INVALIDATE = enum.auto()
  KEEP_OLD = enum.auto()


@functools.cache
def get_reloader() -> _InPlaceReloader:
  """Returns the singleton in-place reloader."""
  return _InPlaceReloader()


def _has_update_rule(obj):
  return isinstance(obj, (type, types.MethodType, types.FunctionType, property))


class _ObjectUpdater:
  """Update various objects & keeps track of type remapping.

  Call update_instances after updating all class types to remap all
  instances to their new class.
  """

  def __init__(self):
    self._type_updates: dict[type, type] = {}  # pylint: disable=g-bare-generic

  def _update_class(self, old: type, new: type):  # pylint: disable=g-bare-generic
    """Update the class."""
    self._type_updates[old] = new

    for key in list(old.__dict__.keys()):
      old_obj = getattr(old, key)

      try:
        new_obj = getattr(new, key)
      except AttributeError:
        # obsolete attribute: remove it
        try:
          delattr(old, key)
        except (AttributeError, TypeError):
          pass
        continue

      self.update(old_obj, new_obj)

      try:
        setattr(old, key, getattr(new, key))
      except (AttributeError, TypeError):
        pass  # skip non-writable attributes

  def _update_enum(self, old: type, new: type):  # pylint: disable=g-bare-generic
    """Update enum types."""
    self._update_class(old, new)
    # Horrible hack: Enums are normally compared as singletons. However, after
    # reloading, the old and new singletons no longer match. There is no good
    # way to globally replace all the old singletons with the new singletons.
    # Instead, enums are defined to compare equal if their class & name match.

    # Pytype doesn't understand that this method is a class method.
    # cast it to silene warnings.
    cur_eq = typing.cast(Any, new.__eq__)

    def enum_eq(x, y):
      eq = cur_eq(x, y)
      # Normal enums compare by singleton and dont implement eq. For these,
      # override it to our special eq. If eq is implemented (eg. for
      # ReprEnum's) just leave it alone.
      if cur_eq(x, y) == NotImplemented:
        return x is y or x.__class__ == y.__class__ and x.name == y.name
      return eq

    new.__eq__ = enum_eq

  def _update_function(self, old: types.FunctionType, new: types.FunctionType):
    """Upgrade the code object of a function."""
    for name in [
        "__code__",
        "__defaults__",
        "__doc__",
        "__closure__",
        "__globals__",
        "__dict__",
    ]:
      try:
        setattr(old, name, getattr(new, name))
      except (AttributeError, TypeError, ValueError):
        pass

  def _update_property(self, old: property, new: property):
    """Replace get/set/del functions of a property."""
    self._update_function(old.fdel, new.fdel)
    self._update_function(old.fget, new.fget)
    self._update_function(old.fset, new.fset)

  def update(self, old, new):
    """Updates a function/class/method/property to a new definition."""
    # Stop replacement if the 2 objects are the same
    if old is new:
      return

    match old, new:
      case enum.EnumType(), enum.EnumType():
        return self._update_enum(old, new)
      case type(), type():
        return self._update_class(old, new)
      case types.FunctionType(), types.FunctionType():
        return self._update_function(old, new)
      case types.MethodType(), types.MethodType():
        return self._update_function(old.__func__, new.__func__)  # pytype: disable=wrong-arg-types
      case property(), property():
        return self._update_property(old, new)

  def update_instances(self):
    """Backport of `update_instances`."""
    if len(self._type_updates) == 0:  # pylint: disable=g-explicit-length-test
      return

    refs = gc.get_referrers(*self._type_updates.keys())
    for ref in refs:
      if (new := self._type_updates.get(type(ref))) is not None:
        object.__setattr__(ref, "__class__", new)


@dataclasses.dataclass(frozen=True)
class _ModuleRefs:
  """Reference on the previous module/object instances."""

  modules: list[weakref.ref[types.ModuleType]] = dataclasses.field(
      default_factory=list
  )
  objs: dict[str, list[weakref.ref[Any]]] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list)
  )

  def save_module(self, module: types.ModuleType) -> None:
    """Save reference on all previous modules/objects."""
    self.modules.append(weakref.ref(module))

    for name, old_obj in module.__dict__.items():
      # Only update objects part of the module (filter other imported symbols)
      if not _belong_to_module(old_obj, module) or not _has_update_rule(
          old_obj
      ):
        continue

      self.objs[name].append(weakref.ref(old_obj))

  def update_refs_with_new_module(
      self,
      new_module: types.ModuleType,
      updater: _ObjectUpdater,
      *,
      verbose: bool = False,
  ) -> _ModuleRefs:
    """Update all old reference to previous objects."""
    # Resolve all weakrefs
    modules = [m for mod in self.modules if (m := mod())]

    for old_module in modules:
      _update_old_module(old_module, new_module)

    # Update all objects
    live_old_objects = collections.defaultdict(list)

    for name, new_obj in new_module.__dict__.items():
      old_obj_refs = self.objs.get(name)

      if old_obj_refs is None:
        continue

      live_old_objects[name] = [r for ref in old_obj_refs if (r := ref())]

      for old_obj in live_old_objects[name]:
        # TODO(epot): Support cycles
        updater.update(old_obj, new_obj)

    if verbose:
      obj_count = lambda x: sum((len(l) for l in x.values()))
      print(
          f"Updated refs of module {new_module.__name__}. Updated"
          f" {len(self.modules)} versions (retained {len(modules)}). Updated"
          f" {obj_count(self.objs)} objects"
          f" ({obj_count(live_old_objects)} retained)."
      )
    live_old_refs = collections.defaultdict(
        list,
        {
            name: [weakref.ref(obj) for obj in objs]
            for name, objs in live_old_objects.items()
        },
    )
    return _ModuleRefs([weakref.ref(m) for m in modules], objs=live_old_refs)


class _InPlaceReloader:
  """Global manager which track all reloaded modules."""

  def __init__(self):
    # Previously imported modules / objects
    # When a new module is `reloaded` with `UPDATE_INPLACE`, all previous
    # modules are replaced in-place.
    self._previous_modules: dict[str, _ModuleRefs] = collections.defaultdict(
        _ModuleRefs
    )

  @contextlib.contextmanager
  def update_old_modules(
      self,
      *,
      reload: list[str],
      verbose: bool,
      reload_mode: ReloadMode,
      recursive: bool,
  ) -> Iterator[None]:
    """Eventually update old modules."""
    # Track imported modules before they are removed from cache (to update them
    # after reloading)
    self._save_objs(reload=reload, recursive=recursive)

    # We clear the module cache to trigger a full reload import.
    # This is better than `colab_import.reload_package` as it support
    # reloading modules with complex import dependency tree.
    # The drawback is that module is duplicated between old module instance
    # and re-loaded instance. To make sure the user don't accidentally use
    # previous instances, we're invalidating all previous modules.
    module_utils.clear_cached_modules(
        modules=reload,
        verbose=verbose,
        invalidate=True if reload_mode == ReloadMode.INVALIDATE else False,
        recursive=recursive,
    )
    try:
      yield
    finally:
      # After reloading, try to update the reloaded modules
      if reload_mode == ReloadMode.UPDATE_INPLACE:
        _update_old_modules(
            reload=reload,
            previous_modules=self._previous_modules,
            verbose=verbose,
            recursive=recursive,
        )

  def _save_objs(self, *, reload: list[str], recursive) -> None:
    """Save all modules/objects."""
    # Save all modules
    for module_name in module_utils.get_module_names(
        reload, recursive=recursive
    ):
      module = sys.modules.get(module_name)
      if module is None:
        continue

      # Save module with all objects from the module
      self._previous_modules[module_name].save_module(module)


def _update_old_modules(
    *,
    reload: list[str],
    previous_modules: dict[str, _ModuleRefs],
    verbose: bool,
    recursive: bool,
) -> None:
  """Update all old modules."""
  # Don't spend time updating types that are already dead anyway.
  gc.collect()

  start_time = time.time()

  updater = _ObjectUpdater()

  for module_name in module_utils.get_module_names(reload, recursive=recursive):
    new_module = sys.modules[module_name]
    old_module_refs = previous_modules.get(module_name)
    if old_module_refs is not None:
      previous_modules[module_name] = (
          old_module_refs.update_refs_with_new_module(
              new_module, updater, verbose=verbose
          )
      )

  # Finally update all existing instances to their new class.
  updater.update_instances()

  if verbose:
    print(
        "Inplace reloading old modules took"
        f" {time.time() - start_time:.2} seconds."
    )


def _update_old_module(
    old_module: types.ModuleType,
    new_module: types.ModuleType,
) -> None:
  """Mutate the old module version with the new dict.

  This also try to update the class, functions,... from the old module (so
  instances are updated in-place).

  Args:
    old_module: Old module to update
    new_module: New module
  """
  # Replace the old dict by the new module content
  old_module.__dict__.clear()
  old_module.__dict__.update(new_module.__dict__)


def _belong_to_module(obj: Any, module: types.ModuleType) -> bool:
  """Returns `True` if the instance, class, function belong to module."""
  return hasattr(obj, "__module__") and obj.__module__ == module.__name__


def _wrap_fn(old_fn, new_fn):
  # Recover the original function (to support colab reload)
  old_fn = inspect.unwrap(old_fn)

  @functools.wraps(old_fn)
  def decorated(*args, **kwargs):
    return new_fn(old_fn, *args, **kwargs)

  return decorated
