# Copyright 2024 The Treescope Authors.
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

"""Global registries for adding treescope support to external types.

This module defines global registries which can be used to add treescope support
for new types that it does not natively support, or types defined in libraries
that may not be installed.

These registries are intended to be used by either the module that defines the
objects being registered, or by `treescope` itself. If a type already has
a global registry entry, you should generally avoid modifying it. This is
because the registries are defined as global variables, without a mechanism for
resolving conflicts between multiple entries. If you would like to customize the
rendering of a type that treescope already supports, you should generally either
define your own treescope renderer object and use it directly, or override the
default renderer or autovisualizer (`treescope.active_renderer` and
`treescope.active_autovisualizer`) using the `set_scoped` and `set_globally`
methods. This will take precedence over any global registry entries.
"""

from __future__ import annotations

import abc
import importlib
import sys
import types
from typing import Any, TypeVar

from treescope import ndarray_adapters
from treescope import renderers
from treescope._internal import object_inspection

T = TypeVar("T")


NDARRAY_ADAPTER_REGISTRY: dict[
    type[Any], ndarray_adapters.NDArrayAdapter[Any]
] = {}
"""Global registry of NDArray adapters, keyed by type.

The value for a given type should be an instance of `NDArrayAdapter`, and will
be used to render any arrays of that type.

If a type is not present in this registry, the entries of that type's `__mro__`
will also be searched. Additionally, virtual base classes will be checked if
the abstract base class is in `VIRTUAL_BASE_CLASSES`.

NDArray adapters are usually looked up using the `lookup_ndarray_adapter`
function, which will also check for the __treescope_ndarray_adapter__ method
on the type.
"""

TREESCOPE_HANDLER_REGISTRY: dict[type[Any], renderers.TreescopeNodeHandler] = {}
"""Global registry of custom treescope handlers, keyed by type.

If a type is not present in this registry, the entries of that type's `__mro__`
will also be searched. Additionally, virtual base classes will be checked if
the abstract base class is in `VIRTUAL_BASE_CLASSES`.

The handler itself will be passed the object, and can either return a treescope
rendering or the `NotImplemented` sentinel, just like an ordinary treescope
handler.

This registry is primarily intended to add treescope support to custom types
without requiring the type to be modified. If you can modify the type, you can
instead define the `__treescope_repr__` method on the type itself; this has
precedence over the registry.
"""

VIRTUAL_BASE_CLASSES: list[type[abc.ABC]] = []
"""List of abstract base classes that should be checked for virtual subclasses.

This list should contain a list of abstract base classes that have virtual
subclasses (defined using the ``.register`` method), and which appear in the
global type registries `NDARRAY_ADAPTER_REGISTRY` or
`TREESCOPE_HANDLER_REGISTRY`. If a type is a subclass of one of these base
classes, the corresponding registry entry will be used.
"""

IMMUTABLE_TYPES_REGISTRY: dict[type[Any], bool] = {
    types.FunctionType: True,
    types.MethodType: True,
    types.ModuleType: True,
    type: True,
    type(None): True,
    type(NotImplemented): True,
    type(Ellipsis): True,
}
"""Global registry of non-hashable types that are considered immutable.

By default, treescope will detect repeated values of any non-hashable type and
render a warning that they are shared across different parts of the tree. This
is intended to help catch bugs in which a value is accidentally shared between
different parts of a tree, which could cause problems when the tree is mutated.

Some types are not hashable, but are still immutable. For instance, `jax.Array`
is immutable and can be safely shared. This set is used to suppress the
"shared value" warning for these types.
"""

_LAZY_MODULE_SETUP_FUNCTIONS: dict[str, tuple[str, str]] = {
    # Note: Numpy is always imported because it is used by the core array
    # rendering system, but we define its setup function here as well for
    # consistency with the other array modules.
    "numpy": (
        "treescope.external.numpy_support",
        "set_up_treescope",
    ),
    "jax": (
        "treescope.external.jax_support",
        "set_up_treescope",
    ),
    "torch": (
        "treescope.external.torch_support",
        "set_up_treescope",
    ),
    "omegaconf": (
        "treescope.external.omegaconf_support",
        "set_up_omegaconf",
    ),
    "pydantic": (
        "treescope.external.pydantic_support",
        "set_up_pydantic",
    ),
}
"""Delayed setup functions that run only once a module is imported.

This dictionary maps module name keys to a ``(setup_module, setup_attribute)``
tuple, where ``setup_module`` is the name of a module and ``setup_attribute``
is the name of a zero-argument function in that module that can be used to set
up support for the key module.

When `update_registries_for_imports` is called (usually at the start of
rendering an object), if any of the key modules have already been imported,
the corresponding setup module will be imported as well
and the setup attribute will be called. This function can then modify the
global values `NDARRAY_ADAPTER_REGISTRY`, `TREESCOPE_HANDLER_REGISTRY`,
`IMMUTABLE_TYPES_SET`, or `VIRTUAL_BASE_CLASSES` to add support for this module.
It can also register the public API of the module using
`treescope.canonical_aliases`, if applicable.

After being called, the setup function will be removed from this dictionary.
"""


def update_registries_for_imports():
  """Updates registries by running setup logic for newly-imported modules."""
  for module_name, (setup_module, setup_attribute) in tuple(
      _LAZY_MODULE_SETUP_FUNCTIONS.items()
  ):
    if module_name in sys.modules:
      module = importlib.import_module(setup_module)
      setup_fn = getattr(module, setup_attribute)
      setup_fn()
      del _LAZY_MODULE_SETUP_FUNCTIONS[module_name]


def _lookup_by_mro(
    registry: dict[type[Any], T], candidate_type: type[Any]
) -> T | None:
  """Looks up the given type in the given registry, or in its base classes.

  This function looks up the given type in the given registry, or in the
  registry for any base class of the given type, in method resolution order.

  If no concrete base class is found in the registry, each of the entries of
  `VIRTUAL_BASE_CLASSES` will be checked to see if it is a virtual base class.
  The first such base class that has an entry in the registry will be used.

  Args:
    registry: The registry to look up in.
    candidate_type: The type to look up.

  Returns:
    The value associated with the given type (or a base class of it) in the
    given registry, or None if no entry was found.
  """
  for supertype in candidate_type.__mro__:
    if supertype in registry:
      return registry[supertype]
  for base_class in VIRTUAL_BASE_CLASSES:
    if issubclass(candidate_type, base_class) and base_class in registry:
      return registry[base_class]
  return None


def lookup_immutability_for_type(candidate_type: type[Any]) -> bool:
  """Checks if an object is marked as immutable in the global registry.

  This function can be used to look up whether an object should be considered
  immutable for treescope according to `IMMUTABLE_TYPES_REGISTRY`. The status
  will be looked up by the type of the given object, any of its base classes, or
  any virtual base classes listed in `VIRTUAL_BASE_CLASSES`.

  Args:
    candidate_type: The type to check immutability of.

  Returns:
    True if this type is registered as immutable, False otherwise.
  """
  return bool(_lookup_by_mro(IMMUTABLE_TYPES_REGISTRY, candidate_type))


def lookup_treescope_handler_for_type(
    candidate_type: type[Any],
) -> renderers.TreescopeNodeHandler | None:
  """Looks up a treescope handler for the given type.

  This function can be used to look up a treescope handler for an object using
  the global registry `TREESCOPE_HANDLER_REGISTRY`. The handler will be looked
  up by the type of the given object, any of its base classes, or any virtual
  base classes listed in `VIRTUAL_BASE_CLASSES`.

  This function does NOT check for methods on the type; those should be checked
  separately.

  Args:
    candidate_type: The type to look up a handler for.

  Returns:
    A treescope handler for the given type, or None if no handler was found.
  """
  return _lookup_by_mro(TREESCOPE_HANDLER_REGISTRY, candidate_type)


def lookup_ndarray_adapter(
    possible_array: Any,
) -> ndarray_adapters.NDArrayAdapter[Any] | None:
  """Looks up an NDArray adapter for the given type.

  This function looks for an NDArray adapter by first checking for the
  `__treescope_ndarray_adapter__` method on the type, and then by looking up
  the type in the global registry `NDARRAY_ADAPTER_REGISTRY`.

  Args:
    possible_array: The object to look up an adapter for.

  Returns:
    An NDArray adapter for the given type, or None if no adapter was found.
  """
  has_adapter_method = object_inspection.safely_get_real_method(
      possible_array, "__treescope_ndarray_adapter__"
  )
  if has_adapter_method:
    return has_adapter_method()
  else:
    return _lookup_by_mro(NDARRAY_ADAPTER_REGISTRY, type(possible_array))
