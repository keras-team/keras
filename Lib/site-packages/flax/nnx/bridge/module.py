# Copyright 2025 The Flax Authors.
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

from __future__ import annotations

from collections import defaultdict
import dataclasses
import functools
import inspect
import threading
import typing as tp
import jax
import typing_extensions as tpe

from flax import errors
from flax.core import meta
from flax.core.frozen_dict import FrozenDict
from flax.nnx import graph, rnglib, statelib, traversals
from flax.nnx import variablelib
import flax.nnx.module as nnx_module
from flax.nnx.object import Object
from flax.nnx import variablelib
from flax.nnx.bridge import variables as bridge_variables
import jax.numpy as jnp

A = tp.TypeVar('A')
M = tp.TypeVar('M', bound='Module')
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])


@dataclasses.dataclass
class ModuleStackEntry:
  module: Module
  in_compact: bool
  type_counter: defaultdict[type, int] = dataclasses.field(
    default_factory=lambda: defaultdict(int)
  )


@dataclasses.dataclass
class ModuleContext(threading.local):
  module_stack: list[ModuleStackEntry | None] = dataclasses.field(
    default_factory=lambda: [None]
  )


MODULE_CONTEXT = ModuleContext()


class ModuleState(statelib.State):
  pass


class Scope(Object):
  def __init__(self, rngs: rnglib.Rngs):
    self.rngs = rngs

  def copy(self):
    return Scope(self.rngs)


class _HasSetup(tp.Protocol):
  def setup(self) -> None: ...


def has_setup(x: tp.Any) -> tp.TypeGuard[_HasSetup]:
  return hasattr(x, 'setup')


def _maybe_call_setup(module: Module):
  if (
    has_setup(module)
    and isinstance(module, Module)
    and not module._object__state.is_setup
  ):
    # void parent context
    MODULE_CONTEXT.module_stack.append(
      ModuleStackEntry(module, in_compact=False)
    )
    try:
      module.setup()  # type: ignore[attribute-error]
      module._object__state._is_setup = True
    finally:
      MODULE_CONTEXT.module_stack.pop()


def _bind_module(parent: Module, module: Module) -> Module:
  assert parent.scope is not None

  for _, value in reversed(list(graph.iter_graph(module))):
    if isinstance(value, Module):
      if module.scope is None:
        value.scope = parent.scope.copy()  # type: ignore[attribute-error]
      _maybe_call_setup(value)
  return module


class ModuleMeta(nnx_module.ModuleMeta):
  if not tp.TYPE_CHECKING:

    def __call__(cls, *args, **kwargs):
      return _module_meta_call(cls, *args, **kwargs)

  def _object_meta_construct(cls, self, *args, **kwargs):
    vars(self)['scope'] = None
    super()._object_meta_construct(self, *args, **kwargs)


def _module_meta_call(cls: type[M], *args, **kwargs) -> M:
  # compact behavior
  parent_ctx = MODULE_CONTEXT.module_stack[-1]
  parent = None
  module: M

  name = None
  if parent_ctx is not None:
    if not parent_ctx.in_compact and 'name' in kwargs:
      raise ValueError(
        f"'name' can only be set in @compact functions. If in setup(), "
          "use parent's `self.<attr_name> to set the submodule name.")

    if parent_ctx.in_compact:
      if 'parent' in kwargs:
        parent = kwargs.pop('parent')
        if parent is not None:
          raise ValueError(
            f"'parent' can only be set to None, got {type(parent).__name__}"
          )
      else:
        type_index = parent_ctx.type_counter[cls]
        parent_ctx.type_counter[cls] += 1

        if 'name' in kwargs:
          name = kwargs.pop('name')
          if not isinstance(name, str):
            raise ValueError(f"'name' must be a 'str', got {type(name).__name__}")
        else:
          name = f'{cls.__name__}_{type_index}'
        parent = parent_ctx.module

  module = nnx_module.ModuleMeta.__call__(cls, *args, **kwargs)
  module.scope = None

  if parent is not None:
    assert parent.scope is not None
    assert name is not None
    setattr(parent, name, module)

  return module  # type: ignore


class ModuleBase:
  if tp.TYPE_CHECKING:
    scope: Scope | None


@tpe.dataclass_transform(field_specifiers=(dataclasses.field,))  # type: ignore[not-supported-yet]
class Module(nnx_module.Module, ModuleBase, metaclass=ModuleMeta):
  def __init_subclass__(cls, experimental_pytree: bool = False) -> None:
    super().__init_subclass__(experimental_pytree)

    cls = dataclasses.dataclass(repr=False)(cls)
    cls.__hash__ = object.__hash__  # type: ignore[method-assign]

  def __getattribute__(self, name: str):
    return type(self)._getattr(self, name)

  def _getattr(self, name: str) -> tp.Any:
    value = super().__getattribute__(name)
    if isinstance(value, ModuleState):
      raise AttributeError
    return value

  def _setattr(self, name: str, value: tp.Any) -> None:
    if self.scope is not None:
      if name in vars(self) and isinstance(
        state := vars(self)[name], ModuleState
      ):
        graph.update(value, state)
      for leaf in jax.tree.leaves(value):
        if isinstance(leaf, Module):
          leaf._object__state._initializing = self.is_initializing()
          _bind_module(self, leaf)
    super()._setattr(name, value)

  def make_rng(self, name: str = 'default') -> jax.Array:
    if self.scope is None:
      raise ValueError("Can't use RNGs on unbound modules")
    return self.scope.rngs[name]()  # type: ignore[attribute-error]

  def param(  # type: ignore[invalid-annotation]
    self,
    name: str,
    init_fn: tp.Callable[..., A],
    *init_args,
    unbox: bool = True,
    **init_kwargs,
  ) -> variablelib.Param[A]:
    # TODO(cgarciae): implement same condition as linen
    if self.scope is None:
      raise ValueError(
        'Parameters must be initialized in `setup()` or in a method '
        'wrapped in `@compact`'
      )
    if hasattr(self, name):
      value = getattr(self, name)
      # TODO(cgarciae): implement reservations
      # if self._name_taken(name):
      #   raise errors.NameInUseError('param', name, self.__class__.__name__)

      if isinstance(value, variablelib.Variable):
        if not isinstance(value, variablelib.Param):
          raise ValueError(
            f"Expected '{name}' to be a Param, got {type(value).__name__}"
          )
        return value

      abs_value = jax.eval_shape(
        lambda: init_fn(jax.random.key(0), *init_args, **init_kwargs)
      )
      abs_value_flat = jax.tree_util.tree_leaves(abs_value)
      value_flat = jax.tree_util.tree_leaves(value)
      for val, abs_val in zip(value_flat, abs_value_flat):
        if jnp.shape(val) != jnp.shape(abs_val):
          raise errors.ScopeParamShapeError(
            name, '', jnp.shape(abs_val), jnp.shape(val)
          )

      if isinstance(abs_value, variablelib.VariableMetadata):
        abs_value.raw_value = value
        value = abs_value

      variable = variablelib.Param(value)
    else:
      value = init_fn(self.make_rng('params'), *init_args, **init_kwargs)
      variable = variablelib.Param(value)

    setattr(self, name, variable)
    return variable

  def variable(  # type: ignore[invalid-annotation]
    self,
    col: str,
    name: str,
    init_fn: tp.Callable[..., A] | None = None,
    *init_args,
    unbox: bool = True,
    **init_kwargs,
  ) -> variablelib.Variable[A]:
    variable_type = variablelib.variable_type_from_name(
      col, allow_register=True
    )
    if self.scope is None:
      raise ValueError(
        'Variables must be initialized in `setup()` or in a method '
        'wrapped in `@compact`'
      )

    if hasattr(self, name):
      value = getattr(self, name)
      # TODO(cgarciae): implement reservations
      # if self._name_taken(name):
      #   raise errors.NameInUseError('param', name, self.__class__.__name__)

      if isinstance(value, variablelib.Variable):
        return value

      if init_fn is None:
        raise ValueError(f"Expected 'init_fn' to be a callable, got None")

      abs_value = jax.eval_shape(lambda: init_fn(*init_args, **init_kwargs))
      abs_value_flat = jax.tree_util.tree_leaves(abs_value)
      value_flat = jax.tree_util.tree_leaves(value)
      for val, abs_val in zip(value_flat, abs_value_flat):
        if jnp.shape(val) != jnp.shape(abs_val):
          raise errors.ScopeParamShapeError(
            name, '', jnp.shape(abs_val), jnp.shape(val)
          )

      if isinstance(abs_value, variablelib.VariableMetadata):
        abs_value.raw_value = value
        value = abs_value

      variable = variable_type(value)
    else:
      if init_fn is None:
        raise ValueError(f"Expected 'init_fn' to be a callable, got None")

      value = init_fn(*init_args, **init_kwargs)
      variable = variable_type(value)

    setattr(self, name, variable)
    return variable

  def _get_variables(self) -> tp.Mapping:
    state = graph.state(self)
    _variables: dict = {}

    variable_state: variablelib.VariableState
    for path, variable_state in statelib.to_flat_state(state):

      if issubclass(variable_state.type, rnglib.RngState):
        # Don't return RNG states, since Linen doesn't have them.
        continue

      try:
        collection = variablelib.variable_name_from_type(variable_state.type)
      except ValueError:
        collection = variable_state.type.__name__

      if collection not in _variables:
        _variables[collection] = {}

      if (
        isinstance(variable_state, variablelib.VariableState)
        and not variable_state._var_metadata
      ):
        leaf = variable_state.value
      else:
        leaf = bridge_variables.to_linen_var(variable_state)

      _variables[collection][path] = leaf

    _variables = {
      collection: traversals.unflatten_mapping(flat_state)
      for collection, flat_state in _variables.items()
    }

    return _variables

  @property
  def variables(self):
    _variables = FrozenDict(self._get_variables())
    return _variables

  def apply(
    self,
    variables: dict[str, tp.Mapping],
    *args,
    rngs: int | jax.Array | dict[str, jax.Array] | rnglib.Rngs | None = None,
    method: tp.Callable[..., tp.Any] | str = '__call__',
    mutable: tp.Any = False,
    _initialize: bool = False,
    **kwargs,
  ) -> tp.Any:
    module = graph.clone(self)

    # create variables
    real_variables = dict(variables)
    for col_name, linen_collection in variables.items():

      def to_variable(value):
        return bridge_variables.to_nnx_var(col_name, value)

      linen_collection = jax.tree.map(
        to_variable,
        linen_collection,
        is_leaf=lambda x: isinstance(x, meta.AxisMetadata),
      )
      real_variables[col_name] = linen_collection

    states = ({},) if not real_variables else real_variables.values()
    state = statelib.merge_state(*states, cls=ModuleState)
    graph.update(module, state)

    if rngs is None:
      rngs = rnglib.Rngs()
    elif isinstance(rngs, jax.Array | int):
      rngs = rnglib.Rngs(rngs)
    elif isinstance(rngs, dict):
      rngs = rnglib.Rngs(**rngs)

    # get method
    _method: tp.Callable[..., tp.Any]
    if isinstance(method, str):
      attribute_name = method
      _method = getattr(module, attribute_name)
      if not callable(_method):
        class_name = type(module).__name__
        raise TypeError(
          f"'{class_name}.{attribute_name}' must be a callable, got"
          f' {type(_method)}.'
        )
      # if the `method` string is a submodule, we create a lambda function
      # that calls the submodule, forwarding all arguments.
      if isinstance(_method, Module):
        _method = lambda module, *args, **kwargs: getattr(
          module, attribute_name
        )(*args, **kwargs)
    else:
      _method = method
    _method = _get_unbound_fn(_method)

    # set temporary state
    for _, value in graph.iter_graph(module):
      if isinstance(value, Object):
        value._object__state._initializing = _initialize
      if isinstance(value, Module):
        value.scope = Scope(rngs)
        _maybe_call_setup(value)

    MODULE_CONTEXT.module_stack.append(
      ModuleStackEntry(module, in_compact=False)
    )
    try:
      out = _method(module, *args, **kwargs)
    finally:
      MODULE_CONTEXT.module_stack.pop()
      # reset temporary state
      for _, value in graph.iter_graph(module):
        if isinstance(value, Object):
          value._object__state._initializing = False
        if isinstance(value, Module):
          value.scope = None

    _variables: tp.Mapping = module._get_variables()

    if mutable is False:
      return out
    else:
      return out, _variables

  def init(
    self,
    rngs: int | jax.Array | dict[str, jax.Array] | rnglib.Rngs | None = None,
    *args,
    method: tp.Callable[..., tp.Any] | str = '__call__',
    **kwargs,
  ):
    out, variables = self.apply(
      {},
      *args,
      _initialize=True,
      mutable=True,
      rngs=rngs,
      method=method,
      **kwargs,
    )
    return variables

  def init_with_output(
    self,
    rngs: int | jax.Array | dict[str, jax.Array] | rnglib.Rngs | None = None,
    *args,
    method: tp.Callable[..., tp.Any] | str = '__call__',
    mutable: tp.Any = False,
    # capture_intermediates: bool | Callable[['Module', str], bool] = False,
    **kwargs,
  ) -> tuple[tp.Any, dict[str, tp.Mapping]]:
    return self.apply(
      {},
      *args,
      rngs=rngs,
      method=method,
      mutable=True,
      _initialize=True,
      **kwargs,
    )

  def is_initializing(self) -> bool:
    return self._object__state._initializing


def compact(f: F) -> F:
  @functools.wraps(f)
  def compact_wrapper(self, *args, **kwargs):
    if not isinstance(self, Module):
      raise ValueError(
        f"Expected 'self' to be a nnx.bridge.Module, got {type(self).__name__}"
      )

    MODULE_CONTEXT.module_stack.append(ModuleStackEntry(self, in_compact=True))

    try:
      return f(self, *args, **kwargs)
    finally:
      MODULE_CONTEXT.module_stack.pop()

  return compact_wrapper  # type: ignore


def _get_unbound_fn(method_or_fn: tp.Callable) -> tp.Callable:
  if inspect.ismethod(method_or_fn) and isinstance(
    method_or_fn.__self__, Module
  ):  # pytype: disable=attribute-error
    method_or_fn = method_or_fn.__func__  # pytype: disable=attribute-error
  if (
    not callable(method_or_fn)
    or len(inspect.signature(method_or_fn).parameters) < 1
  ):
    raise errors.ApplyModuleInvalidMethodError(method_or_fn)

  return method_or_fn
