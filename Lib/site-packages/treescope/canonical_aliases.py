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

"""Registry of certain "well-known" objects, such as module functions.

Taking the `repr` of a function or callable usually produces something like

  <function vmap at 0x7f98bf556440>

and in some cases produces something like

  <jax.custom_derivatives.custom_jvp object at 0x7f98c0b5f130>

This can make it hard to determine what object this actually is from a user
perspective, and it would likely be more user-friendly to just output `jax.vmap`
or `jax.nn.relu` as the representation of this object.

Many functions and classes store a reference to the location where they were
originally defined (in their __module__ and __qualname__ attributes), which
can be used to find an alias for them. However, this may not be the "canonical"
alias, because some modules re-export private symbols under a public namespace
(in particular, JAX and Equinox both do this).

This module contains a registry of canonical paths for specific functions and
other objects, so that they can be rendered in a predictable way by treescope.
The intended purpose is primarily for interactive printing and debugging,
although it also helps for reifying objects into executable code through
round-trippable pretty printing, which can enable a simple form of
serialization.

This module also supports walking the public API of a package to automatically
set aliases; this is done by default for JAX and a few other libraries to ensure
the pretty-printed outputs avoid private module paths whenever possible. This
is intended as a heuristic to construct readable aliases for common objects on a
best-effort basis. It is not guaranteed that these inferred aliases will always
be stable across different versions of the external libraries.
"""

import collections
import contextlib
import dataclasses
import inspect
import sys
import types
from typing import Any, Callable, Literal, Mapping
import warnings

from treescope import context


@dataclasses.dataclass(frozen=True)
class ModuleAttributePath:
  """Expected path where we can find a particular object in a module.

  Attributes:
    module_name: Fully-qualified name of the module (the key of the module in
      sys.modules) in which this object can be found.
    attribute_path: Sequence of attributes identifying this object, separated by
      dots ("."). For instance, if this is ["foo", "bar"] then we expect to find
      the object at "{module_name}.foo.bar".
  """

  module_name: str
  attribute_path: tuple[str, ...]

  def __str__(self):
    module_name = self.module_name
    attribute_path_str = "".join(f".{attr}" for attr in self.attribute_path)
    if module_name == "__main__":
      assert attribute_path_str.startswith(".")
      return attribute_path_str[1:]

    return f"{module_name}{attribute_path_str}"

  def retrieve(self, forgiving: bool = False) -> Any:
    """Retrieves the object at this path.

    Args:
      forgiving: If True, return None on failure instead of raising an error.

    Returns:
      The retrieved object, or None if it wasn't found and `forgiving` was True.

    Raises:
      ValueError: If the object wasn't found and `forgiving` is False.
    """
    if self.module_name not in sys.modules:
      if forgiving:
        return None
      raise ValueError(
          f"Invalid alias {self} pointing to a non-imported module"
          f" {self.module_name}"
      )
    try:
      the_module = sys.modules[self.module_name]
    except KeyError:
      if forgiving:
        return None
      else:
        raise

    current_object = the_module
    for attr in self.attribute_path:
      if hasattr(current_object, attr):
        current_object = getattr(current_object, attr)
      else:
        if forgiving:
          return None
        raise ValueError(
            f"Can't retrieve {self}: {self.module_name} does"
            f" not expose an attribute {self.attribute_path!r}."
        )

    return current_object


@dataclasses.dataclass(frozen=True)
class LocalNamePath:
  """Expected path where we can find a particular object in a local scope.

  The "local scope" can be any dictionary of values with string keys, but it
  is usually either the locals() or globals() for a particular scope (or the
  union of these). A local name path is only valid relative to a particular
  scope that was used to create it.

  Attributes:
    local_name: Name of the variable in the local scope that we are retrieving.
    attribute_path: Sequence of attributes identifying this object, separated by
      dots ("."). For instance, if this is ["foo", "bar"] then we expect to find
      the object at "{local_name}.foo.bar".
  """

  local_name: str
  attribute_path: tuple[str, ...]

  def __str__(self):
    attribute_path_str = "".join(f".{attr}" for attr in self.attribute_path)
    return f"{self.local_name}{attribute_path_str}"

  def retrieve(
      self, local_scope: dict[str, Any], forgiving: bool = False
  ) -> Any:
    """Retrieves the object at this path.

    Args:
      local_scope: The scope in which we should retrieve this value.
      forgiving: If True, return None on failure instead of raising an error.

    Returns:
      The retrieved object, or None if it wasn't found and `forgiving` was True.

    Raises:
      KeyError, AttributeError: If the object wasn't found and `forgiving` is
        False.
    """
    try:
      current_object = local_scope[self.local_name]
    except KeyError:
      if forgiving:
        return None
      else:
        raise

    for attr in self.attribute_path:
      try:
        current_object = getattr(current_object, attr)
      except AttributeError:
        if forgiving:
          return None
        else:
          raise

    return current_object


@dataclasses.dataclass(frozen=True)
class CanonicalAliasEnvironment:
  """An environment that defines a set of canonical aliases.

  Attributes:
    aliases: A mapping from id(some_object) to the path where we expect to find
      that object.
  """

  aliases: dict[int, ModuleAttributePath]


_alias_environment: context.ContextualValue[CanonicalAliasEnvironment] = (
    context.ContextualValue(
        module=__name__,
        qualname="_alias_environment",
        initial_value=CanonicalAliasEnvironment({}),
    )
)
"""The current environment for module-level canonical aliases.

All alias mutation and lookups occur relative to the current
environment. Usually, this will not need to be modified, but it can
be useful to modify for tests or other local modifications.
"""


def add_alias(
    the_object: Any,
    path: ModuleAttributePath,
    on_conflict: Literal["ignore", "overwrite", "warn", "error"] = "warn",
):
  """Adds an alias to this object.

  Args:
    the_object: Object to add an alias to.
    path: The path where we expect to find this object.
    on_conflict: What to do if we try to add an alias for an object that already
      has one.

  Raises:
    ValueError: If overwrite is False and this object already has an alias,
      or if this object is not accessible at this path.
  """
  alias_env = _alias_environment.get()
  if id(the_object) in alias_env.aliases:
    if on_conflict == "ignore":
      return
    elif on_conflict == "overwrite":
      pass  # Continue adding an alias
    elif on_conflict == "warn":
      warnings.warn(
          f"Not defining alias {path} for {the_object!r}: it already has an"
          f" alias {alias_env.aliases[id(the_object)]}."
      )
    elif on_conflict == "error":
      raise ValueError(
          f"Can't define alias {path} for {the_object!r}: it already has an"
          f" alias {alias_env.aliases[id(the_object)]}."
      )

  retrieved = path.retrieve()
  if retrieved is not the_object:
    raise ValueError(
        f"Can't define alias {path} for {the_object!r}: {path} "
        f" is a different object {retrieved!r}."
    )
  # OK, it's probably safe to add this object as a well-known alias.
  alias_env.aliases[id(the_object)] = path


_current_scope_for_local_aliases: context.ContextualValue[
    dict[int, LocalNamePath] | None
] = context.ContextualValue(
    module=__name__,
    qualname="_current_scope_for_local_aliases",
    initial_value=None,
)
"""A mapping from IDs to local names.

Should only be used or modified by `local_alias_names`.
"""


def lookup_alias(
    the_object: Any,
    infer_from_attributes: bool = True,
    allow_outdated: bool = False,
    allow_relative: bool = False,
) -> ModuleAttributePath | LocalNamePath | None:
  """Retrieves an alias for this object, if possible.

  This function checks if an alias has been registered for this object, and also
  makes sure the object is still available at that alias. It optionally also
  tries to infer a fallback alias using __module__ and __qualname__ attributes.

  Args:
    the_object: Object to get a well-known alias for.
    infer_from_attributes: Whether to use __module__ and __qualname__ attributes
      to infer an alias if no explicit path is registered.
    allow_outdated: If True, return old aliases even if the object is no longer
      accessible at this path. (For instance, if the module was reloaded after
      this class/function was defined.)
    allow_relative: If True, return aliases that are local to the current
      `relative_alias_names` context if possible.

  Returns:
    A path at which we can find this object, or None if we do not have a path
    for it (or if the path is no longer correct).
  """
  if the_object is None:
    # None itself should never have an alias. Checking for it here lets us
    # easily catch the "broken alias" case in `retrieve` below.
    return None

  alias_env = _alias_environment.get()
  # Is this object in the local aliases? If so, return that.
  if allow_relative:
    local_aliases = _current_scope_for_local_aliases.get()
  else:
    local_aliases = None

  if local_aliases and id(the_object) in local_aliases:
    return local_aliases[id(the_object)]

  # Check for a global canonical alias.
  if id(the_object) in alias_env.aliases:
    alias = alias_env.aliases[id(the_object)]
  elif infer_from_attributes:
    # Try to unwrap it, in case it's a function-like object wrapping a function.
    unwrapped = inspect.unwrap(the_object)
    if (
        hasattr(unwrapped, "__module__")
        and hasattr(unwrapped, "__qualname__")
        and unwrapped.__qualname__ is not None
        and "<" not in unwrapped.__qualname__
    ):
      alias = ModuleAttributePath(
          unwrapped.__module__, tuple(unwrapped.__qualname__.split("."))
      )
    elif isinstance(the_object, types.ModuleType):
      alias = ModuleAttributePath(the_object.__name__, ())
    else:
      # Can't infer an alias.
      return None
  else:
    return None

  if not allow_outdated:
    if isinstance(the_object, types.MethodType):
      # Methods get different IDs on each access.
      if alias.retrieve(forgiving=True) != the_object:
        return None
    else:
      if alias.retrieve(forgiving=True) is not the_object:
        return None

  if local_aliases:
    # Check if any of the attributes along this path have a local alias.
    for split_point in reversed(range(len(alias.attribute_path))):
      parent_object = ModuleAttributePath(
          alias.module_name, alias.attribute_path[:split_point]
      ).retrieve(forgiving=True)
      if id(parent_object) in local_aliases:
        local_path_to_parent = local_aliases[id(parent_object)]
        return LocalNamePath(
            local_name=local_path_to_parent.local_name,
            attribute_path=(
                local_path_to_parent.attribute_path
                + alias.attribute_path[split_point:]
            ),
        )

    # Check if any parent module of the module in which this global alias is
    # defined has a local alias that we should use instead (e.g. using `np`
    # instead of `numpy`).
    module_parts = alias.module_name.split(".")
    submodule_path_reversed = []
    while module_parts:
      # Get an equivalent path from the parent module.
      parent_module_alias = ModuleAttributePath(
          ".".join(module_parts),
          tuple(reversed(submodule_path_reversed)) + alias.attribute_path,
      )
      # Is this a real parent module? (It probably should be unless someone
      # mucked with import paths.)
      if parent_module_alias.module_name in sys.modules:
        module_id = id(sys.modules[parent_module_alias.module_name])
        # Do we have a local alias for this parent module? And, if so, can we
        # access this particular object through the parent module in the
        # expected way?
        if (
            module_id in local_aliases
            and parent_module_alias.retrieve(forgiving=True) is the_object
        ):
          # First lookup the module in the local scope, then lookup the value
          # in the module.
          local_path_to_module = local_aliases[module_id]
          return LocalNamePath(
              local_name=local_path_to_module.local_name,
              attribute_path=(
                  local_path_to_module.attribute_path
                  + parent_module_alias.attribute_path
              ),
          )
      submodule_path_reversed.append(module_parts.pop())

  return alias


def maybe_local_module_name(module: types.ModuleType) -> str:
  """Returns a name for this module, possibly looking up local aliases."""
  alias = lookup_alias(module, allow_outdated=True, allow_relative=True)
  assert alias is not None
  return str(alias)


def default_well_known_filter(
    the_object: Any, path: ModuleAttributePath | LocalNamePath
) -> bool:
  """Checks if an object looks like something we want to define an alias for."""

  is_function_after_unwrap = isinstance(
      inspect.unwrap(the_object), types.FunctionType
  )

  # Only define aliases for objects that are either (a) mutable or (b)
  # classes/functions/modules.
  if (
      (hasattr(the_object, "__hash__") and the_object.__hash__ is not None)
      and not isinstance(the_object, (types.ModuleType, type))
      and not is_function_after_unwrap
  ):
    return False

  # Don't allow classes and functions to be assigned to any name other than the
  # name they were given at creation time.
  if isinstance(the_object, type) or (
      is_function_after_unwrap and hasattr(the_object, "__name__")
  ):
    expected_name = the_object.__name__
    if path.attribute_path and expected_name != path.attribute_path[-1]:
      return False

  # Assume any name that starts with an underscore is private.
  if any(attr.startswith("_") for attr in path.attribute_path) or (
      isinstance(path, LocalNamePath) and path.local_name.startswith("_")
  ):
    return False

  return True


def relative_alias_names(
    relative_scope: Mapping[str, Any] | Literal["magic"],
    predicate: Callable[[Any, LocalNamePath], bool] = default_well_known_filter,
) -> contextlib.AbstractContextManager[None]:
  """Context manager that makes `lookup_alias` return relative aliases.

  Args:
    relative_scope: A dictionary mapping in-scope names to their values, e.g.
      `globals()` or `{**globals(), **locals()}`. Objects that are reachable
      from this scope will have aliases that reference the keys in this dict.
      Alternatively, can be the string "magic", in which case we will walk the
      stack and use the `{**globals(), **locals()}` of the caller. ("magic" is
      only recommended for interactive usage.)
    predicate: A filter function to check if an object should be given an alias.

  Returns:
    A context manager in which `lookup_alias` will try to return relative
    aliases when `allow_relative=True`.
  """
  if relative_scope == "magic":
    # Infer the scope by walking the stack. 1 is the caller's frame.
    caller_frame_info = inspect.stack()[1]
    relative_scope = collections.ChainMap(
        caller_frame_info.frame.f_globals,
        caller_frame_info.frame.f_locals,
    )

  local_alias_map = {}

  assert isinstance(relative_scope, Mapping)
  for key, value in relative_scope.items():
    path = LocalNamePath(key, ())
    if predicate(value, path):
      local_alias_map[id(value)] = path

  return _current_scope_for_local_aliases.set_scoped(local_alias_map)


def populate_from_public_api(
    module: types.ModuleType,
    predicate: Callable[
        [Any, ModuleAttributePath], bool
    ] = default_well_known_filter,
):
  """Populates canonical aliases with all public symbols in a module.

  Attempts to walk this module and its submodules to extract well-known symbols.
  Symbols that already have an alias defined will be ignored.

  If the module defines __all__, we assume the symbols in __all__ are the
  well-known symbols in this module. (See
  https://docs.python.org/3/reference/simple_stmts.html#the-import-statement)

  If the module does not define __all__, we look for all names that do not
  start with "_".

  We then additionally filter down this set to the set of objects for which
  the `predicate` argument returns True.

  This function should only be called on modules with a well-defined public API,
  that exports only functions defined in the module itself. If a module
  re-exports symbols from another external module (e.g. importing `partial`
  from `functools`), we might otherwise end up using the unrelated module as
  the "canonical" source of that object. (The `prefix_filter` function below
  tries to prevent this if possible when used as a predicate.)

  Args:
    module: The module we will collect symbols from.
    predicate: A filter function to check if an object should be given an alias.
  """
  if hasattr(module, "__all__"):
    public_names = module.__all__
  else:
    public_names = [
        key for key in module.__dict__.keys() if not key.startswith("_")
    ]

  for name in public_names:
    try:
      value = getattr(module, name)
    except AttributeError:
      # Possibly a misspecified __all__?
      continue
    path = ModuleAttributePath(module.__name__, (name,))
    if isinstance(value, types.ModuleType):
      if (
          value.__name__.startswith(module.__name__)
          and value.__name__ != module.__name__
      ):
        # Process submodules of this module also.
        populate_from_public_api(value, predicate)
      # Don't process external modules that are being re-exported.
    elif predicate(value, path):
      add_alias(value, path, on_conflict="ignore")


def prefix_filter(include: str, excludes: tuple[str, ...] = ()):
  """Builds a filter that only defines aliases within a given prefix."""

  def is_under_prefix(path: str, prefix: str):
    return path == prefix or path.startswith(prefix + ".")

  def predicate(the_object: Any, path: ModuleAttributePath) -> bool:
    if not default_well_known_filter(the_object, path):
      return False
    if not is_under_prefix(str(path), include):
      return False
    if any(is_under_prefix(str(path), exclude) for exclude in excludes):
      return False
    if (
        hasattr(the_object, "__module__")
        and the_object.__module__
        and not is_under_prefix(the_object.__module__, include)
    ):
      return False
    return True

  return predicate
