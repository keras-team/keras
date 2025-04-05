# Copyright 2024 The Flax Authors.
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

import dataclasses
import typing as tp
from typing import Any

from flax import linen
from flax import nnx
from flax.core import FrozenDict
from flax.core import meta
from flax.nnx import graph
from flax.nnx import variablelib
from flax.nnx.bridge import variables as bv
from flax.nnx.module import Module
from flax.nnx.object import Object
from flax.nnx.rnglib import Rngs
from flax.nnx.statelib import State
import jax
from jax import tree_util as jtu

M = tp.TypeVar('M', bound=Module)


# Flax-like style is NNX
@dataclasses.dataclass
class Functional(tp.Generic[M]):
  module_type: tp.Type[M]
  graphdef: tp.Optional[graph.NodeDef[M]]
  args: tuple[tp.Any, ...]
  kwargs: dict[str, tp.Any]

  def init(self, *, rngs: tp.Optional[Rngs] = None) -> State:
    kwargs = {}
    if rngs is not None:
      kwargs['rngs'] = rngs
    module = self.module_type(*self.args, **self.kwargs, **kwargs)
    graphdef, state = nnx.split(module)
    assert type(graphdef) is graph.NodeDef
    self.graphdef = graphdef
    return state

  def apply(self, *states: tp.Any):
    assert self.graphdef is not None
    return self.graphdef.apply(*states)


def functional(cls: tp.Type[M]) -> tp.Callable[..., Functional[M]]:
  def _functional_constructor(*args: tp.Any, **kwargs: tp.Any) -> Functional[M]:
    return Functional(cls, None, args, kwargs)

  return _functional_constructor


def _set_initializing(module: Module, initializing: bool):
  for k, value in graph.iter_graph(module):
    if isinstance(value, Object):
      value._object__state._initializing = initializing


def lazy_init(fn: Module | tp.Callable[..., tp.Any], *args, **kwargs):
  """To run through an arbitrary nnx.Module method and initialize all its needed state.

  Here used to trigger initialization of all `LinenToNNX` module variables."""
  if isinstance(fn, Module):
    module = fn
    assert callable(fn)
  else:
    if not hasattr(fn, '__self__') and isinstance(fn.__self__, Module):
      raise ValueError(f'{fn = } needs to be a method of an NNX Module.')
    module = fn.__self__
  _set_initializing(module, True)
  try:
    _ = fn(*args, **kwargs)
  finally:
    _set_initializing(module, False)
  return fn


class ToNNX(Module):
  """A wrapper to turn any Linen module into an NNX module.

  The result NNX module can be used standalone with all NNX APIs, or as a submodule of
  another NNX module.

  Since Linen module initialization requires a sample input, you need to call `lazy_init`
  with an argument to initialize the variables.

  Example::

    >>> from flax import linen as nn, nnx
    >>> import jax
    >>> linen_module = nn.Dense(features=64)
    >>> x = jax.numpy.ones((1, 32))
    >>> # Like Linen init(), initialize with a sample input
    >>> model = nnx.bridge.ToNNX(linen_module, rngs=nnx.Rngs(0)).lazy_init(x)
    >>> # Like Linen apply(), but using NNX's direct call method
    >>> y = model(x)
    >>> model.kernel.shape
    (32, 64)

  Args:
    module: The Linen Module instance.
    rngs: The `nnx.Rngs` instance being passed to any NNX module.

  Returns:
    A stateful NNX module that behaves the same as the wrapped Linen module.
  """
  def __init__(
    self,
    module: linen.Module,
    rngs: tp.Optional[Rngs] = None,
  ):
    self.module = module
    self.rngs = rngs
    self.linen_attributes: tuple[str, ...] = ()

  def lazy_init(self, *args, **kwargs):
    """A shortcut of calling `nnx.bridge.lazy_init()` upon this module."""
    return lazy_init(self, *args, **kwargs)

  def __call__(
    self, *args: Any, rngs: tp.Optional[Rngs] = None,
    method: tp.Callable[..., Any] | str | None = None, **kwargs: Any
  ) -> Any:

    # Shape-based lazy init of the flax variables
    if not rngs:
      rngs = self.rngs
    if self._object__state.initializing:
      _rngs = (
        {name: stream.key.raw_value for name, stream in rngs.items()}
        if rngs
        else {}
      )
      # rename default to params
      if 'params' not in _rngs and 'default' in _rngs:
        _rngs['params'] = _rngs.pop('default')
      out, variables = self.module.init_with_output(_rngs, *args, method=method, **kwargs)

      nnx_attrs = bv.linen_vars_to_nnx_attrs(variables)
      linen_attributes = set(self.linen_attributes)
      for attr_name, value in nnx_attrs.items():
        setattr(self, attr_name, value)
        linen_attributes.add(attr_name)
      self.linen_attributes = tuple(linen_attributes)  # make it hashable

    else:
      nnx_attrs = {name: getattr(self, name) for name in self.linen_attributes}
      variables = bv.nnx_attrs_to_linen_vars(nnx_attrs)

      _rngs = (
        {name: stream() for name, stream in rngs.items()} if rngs else {}
      )
      out = self.module.apply(variables, *args, rngs=_rngs, method=method, **kwargs)

    # Split out the updates if `mutable` is passed into the Flax module
    if kwargs.get('mutable', False) != False:
      out, updates = out
      nnx_attrs = bv.linen_vars_to_nnx_attrs(updates)
      linen_attributes = set(self.linen_attributes)
      for attr_name, value in nnx_attrs.items():
        linen_attributes.add(attr_name)
        if hasattr(self, attr_name) and isinstance(value, dict):
          original_tree = getattr(self, attr_name)
          setattr(self, attr_name, original_tree | value)
        else:
          setattr(self, attr_name, value)

      self.linen_attributes = tuple(linen_attributes)  # make it hashable

    return out


def linen_rngs_dict(linen_module: linen.Module) -> tp.Mapping[str, jax.Array]:
  """Given a module, split out one of its every active RNG key collections."""
  assert linen_module.scope is not None, 'linen_rngs_dict() must be called inside a Linen module.'
  return {name: linen_module.make_rng(name)
          for name in linen_module.scope.rngs.keys()}


class ToLinen(linen.Module):
  """A wrapper to turn any NNX module into a Linen module.

  The result Linen module can be used standalone with all Linen APIs, or as a submodule of
  another Linen module.

  Since NNX modules are stateful and owns the state, we only create it once during init
  time, and will track its state and static data as separate variables.

  Example::

    >>> from flax import linen as nn, nnx
    >>> import jax
    >>> model = nnx.bridge.ToLinen(nnx.Linear, args=(32, 64))
    >>> x = jax.numpy.ones((1, 32))
    >>> y, variables = model.init_with_output(jax.random.key(0), x)
    >>> y.shape
    (1, 64)
    >>> variables['params']['kernel'].shape
    (32, 64)
    >>> # The static GraphDef of the underlying NNX module
    >>> variables.keys()
    dict_keys(['nnx', 'params'])
    >>> type(variables['nnx']['graphdef'])
    <class 'flax.nnx.graph.NodeDef'>

  Args:
    nnx_class: The NNX Module class (not instance!).
    args: The arguments that normally would be passed in to create the NNX module.
    kwargs: The keyword arguments that normally would be passed in to create the NNX module.
    skip_rng: True if this NNX module doesn't need `rngs` arg during initialization (not common).

  Returns:
    A stateful NNX module that behaves the same as the wrapped Linen module.
  """
  nnx_class: tp.Callable[..., Module]
  args: tp.Sequence = ()
  kwargs: tp.Mapping[str, tp.Any] = FrozenDict({})
  skip_rng: bool = False
  metadata_type: tp.Type = bv.NNXMeta

  @linen.compact
  def __call__(self, *args, **kwargs):
    # init codepath
    if self.is_initializing():
      module_kwargs = dict(self.kwargs)
      if not self.skip_rng:
        module_kwargs |= dict(rngs=nnx.Rngs(**linen_rngs_dict(self)))
      module = self.nnx_class(*self.args, **module_kwargs)
      # TODO: add lazy_init here in case there's an `ToNNX` submodule under `module`.
      self._update_variables(module)
      return module(*args, **kwargs)

    # apply codepath
    gdef = self.get_variable('nnx', 'graphdef')
    assert gdef, 'GraphDef not found in variables. Was the collection "nnx" dropped somewhere?'
    variables = {col: v for col, v in self.variables.items() if col != 'nnx'}
    states = jtu.tree_map_with_path(
        lambda kp, x: bv.to_nnx_var(bv.get_col_name(kp), x).to_state(),
        variables, is_leaf=lambda x: isinstance(x, meta.AxisMetadata))
    states = [State(v) for v in states.values()]
    nnx_state = nnx.merge_state(*states) if states else nnx.GraphState({})
    module = nnx.merge(gdef, nnx_state)
    nnx.reseed(module, **linen_rngs_dict(self))  # reseed with keys from linen apply call.
    out = module(*args, **kwargs)
    self._update_variables(module)
    return out

  def _update_variables(self, module):
    """Store the NNX module's graph def and state inside Linen module variables."""
    gdef, state = nnx.split(module)
    # Save the graph def.
    if self.is_mutable_collection('nnx'):
      self.put_variable('nnx', 'graphdef', gdef)
    # Sort all the variable types.
    types = set(jax.tree.leaves(
      jax.tree.map(lambda x: x.type, state,
                    is_leaf=lambda x: isinstance(x, nnx.VariableState))))
    types = bv.sort_variable_types(types)
    _, *state_by_types = nnx.split(module, *types)
    # Each variable type goes to its own linen collection, and
    # each attribute goes to its own linen variable
    for typ, state in zip(types, state_by_types):
      collection = variablelib.variable_name_from_type(typ, allow_register=True)
      if self.is_mutable_collection(collection):
        for k, v in state.raw_mapping.items():
          v = jax.tree.map(bv.to_linen_var, v,
                           is_leaf=lambda x: isinstance(x, nnx.VariableState))
          self.put_variable(collection, k, v)


def to_linen(nnx_class: tp.Callable[..., Module], *args,
             name: str | None = None, **kwargs):
  """Shortcut of `nnx.bridge.ToLinen` if user is not changing any of its default fields."""
  return ToLinen(nnx_class, args=args, kwargs=FrozenDict(kwargs), name=name)
