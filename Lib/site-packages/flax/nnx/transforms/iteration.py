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
# pytype: skip-file

from collections import deque
import dataclasses
import functools
import typing as tp

from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.nnx import extract, filterlib, graph, spmd, variablelib
from flax.nnx import statelib
from flax.nnx.module import Module
from flax.nnx.statelib import State
from flax.nnx.transforms.transforms import resolve_kwargs
from flax.typing import Leaf, Missing, PytreeDeque
import jax
import jax.core
import jax.numpy as jnp
import jax.stages
import numpy as np

A = tp.TypeVar('A')
C = tp.TypeVar('C')
B = tp.TypeVar('B')
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])
G = tp.TypeVar('G', bound=tp.Callable[..., tp.Any])
M = tp.TypeVar('M', bound=Module)
MA = tp.TypeVar('MA', bound=Module)
N = tp.TypeVar('N', bound=Module)
T = tp.TypeVar('T')
StrInt = tp.TypeVar('StrInt', str, int)
AxisName = tp.Hashable
Leaves = tp.List[Leaf]
Index = int


class Carry:
  pass


# -------------------------------
# vmap
# -------------------------------


class StateAxes(extract.PrefixMapping):

  def __init__(
    self,
    filter_axes: (
      statelib.State
      | tp.Mapping[filterlib.Filter, Index | type[Carry] | None]
      | tp.Iterable[tuple[filterlib.Filter, Index | type[Carry] | None]]
    ),
    /,
  ):
    if isinstance(filter_axes, statelib.State):
      filter_axes = statelib.create_path_filters(filter_axes)  # type: ignore

    iterable = tuple(
        filter_axes.items()
        if isinstance(filter_axes, tp.Mapping)
        else filter_axes
    )
    self._filters = tuple(filter for filter, _ in iterable)
    self._axes = tuple(axis for _, axis in iterable)

  @property
  def filters(self) -> tuple[filterlib.Filter, ...]:
    return self._filters

  @property
  def axes(self) -> tuple[Index | type[Carry] | None, ...]:
    return self._axes

  def map_prefix(
    self, path: variablelib.PathParts, variable: variablelib.Variable
  ) -> tp.Any:
    for filter, axis in zip(self.filters, self.axes):
      predicate = filterlib.to_predicate(filter)
      if predicate(path, variable):
        return axis
    raise ValueError(f'No axis found for {path=}, {variable=}')

  def __repr__(self):
    return f'StateAxes({dict(self.items())})'

  def items(self):
    return zip(self.filters, self.axes)

  def __eq__(self, other):
    return (
        isinstance(other, StateAxes)
        and self.filters == other.filters
        and self.axes == other.axes
    )

  def __hash__(self):
    return hash((self.filters, self.axes))


AxisFn = tp.Callable[[graph.GraphState, int, tp.Mapping], graph.GraphState]


def _update_variable_sharding_metadata(
    tree, transform_metadata, axis_fn: AxisFn
):
  def _update_axes_fn(node_states):
    if isinstance(node_states, extract.NodeStates) and isinstance(
      node_states.metadata, (StateAxes, int)
    ):
      if isinstance(node_states.metadata, int):
        state = node_states.state
        assert isinstance(state, State)
        state = axis_fn(state, node_states.metadata, transform_metadata)
        return node_states.replace(states=(state,))
      else:
        states_out: list[graph.GraphState] = []
        for state, axis in zip(node_states.states, node_states.metadata.axes):
          assert isinstance(state, graph.State)
          if isinstance(axis, int):
            state = axis_fn(state, axis, transform_metadata)
          states_out.append(state)
        return node_states.replace(states=tuple(states_out))
    return node_states

  return jax.tree.map(
    _update_axes_fn, tree, is_leaf=lambda x: isinstance(x, extract.NodeStates)
  )


def _vmap_split_fn(ctx: graph.SplitContext, path, prefix, x):
  if isinstance(prefix, StateAxes):
    return extract.NodeStates.from_split(
      *ctx.split(x, *prefix.filters), metadata=prefix
    )
  return extract.NodeStates.from_split(*ctx.split(x), metadata=prefix)


@dataclasses.dataclass(eq=False)
class VmapFn:
  f: tp.Callable[..., tp.Any]
  transform_metadata: tp.Mapping[str, tp.Any]
  in_axes: tp.Any
  out_axes: tp.Any

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  def __call__(self, *pure_args: tuple[tp.Any, ...]):
    if spmd.PARTITION_NAME in self.transform_metadata:
      pure_args = _update_variable_sharding_metadata(
          pure_args, self.transform_metadata, spmd.remove_axis
      )
    args = extract.from_tree(pure_args, ctxtag='vmap', is_inner=True)

    out = self.f(*args)

    args_out = extract.clear_non_graph_nodes(args)
    pure_args_out, pure_out = extract.to_tree(
        (args_out, out),
        prefix=(self.in_axes, self.out_axes),
        split_fn=_vmap_split_fn,
        ctxtag='vmap',
    )
    if spmd.PARTITION_NAME in self.transform_metadata:
      pure_args_out, pure_out = _update_variable_sharding_metadata(
          (pure_args_out, pure_out), self.transform_metadata, spmd.add_axis
      )
    return pure_args_out, pure_out


@tp.overload
def vmap(
    *,
    in_axes: int | None | tp.Sequence[tp.Any] = 0,
    out_axes: tp.Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    # nnx specific
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> tp.Callable[[F], F]:
  ...


@tp.overload
def vmap(
    f: F,
    *,
    in_axes: int | None | tp.Sequence[tp.Any] = 0,
    out_axes: tp.Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    # nnx specific
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F:
  ...


def vmap(
  f: F | type[Missing] = Missing,
  *,
  in_axes: int | None | tp.Sequence[tp.Any] = 0,
  out_axes: tp.Any = 0,
  axis_name: AxisName | None = None,
  axis_size: int | None = None,
  spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
  # nnx specific
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F | tp.Callable[[F], F]:
  """Reference-aware version of `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__.

  Args:
    f: Function to be mapped over additional axes.
    in_axes: An integer, None, or sequence of values specifying which input
      array axes to map over (see `jax.vmap
      <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__). In
      addition to integers and None, :class:`StateAxes`  can be used to control
      how graph nodes like Modules are vectorized by specifying the axes to be
      applied to substates of the graph node given a `Filter
      <https://flax.readthedocs.io/en/latest/nnx/filters_guide.html>`__.
    out_axes: An integer, None, or pytree indicating where the mapped axis
      should appear in the output (see `jax.vmap
      <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__).
    axis_name: Optional, a hashable Python object used to identify the mapped
      axis so that parallel collectives can be applied.
    axis_size: Optional, an integer indicating the size of the axis to be
      mapped. If not provided, the mapped axis size is inferred from arguments.

  Returns:
    Batched/vectorized version of ``f`` with arguments that correspond to
    those of ``f``, but with extra array axes at positions indicated by
    ``in_axes``, and a return value that corresponds to that of ``f``, but
    with extra array axes at positions indicated by ``out_axes``.

  Example::

    >>> from flax import nnx
    >>> from jax import random, numpy as jnp
    ...
    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> x = jnp.ones((5, 2))
    ...
    >>> @nnx.vmap(in_axes=(None, 0), out_axes=0)
    ... def forward(model, x):
    ...   return model(x)
    ...
    >>> y = forward(model, x)
    >>> y.shape
    (5, 3)

  >>> class LinearEnsemble(nnx.Module):
  ...   def __init__(self, num, rngs):
  ...     self.w = nnx.Param(jax.random.uniform(rngs(), (num, 2, 3)))
  ...
  >>> model = LinearEnsemble(5, rngs=nnx.Rngs(0))
  >>> x = jnp.ones((2,))
  ...
  >>> @nnx.vmap(in_axes=(0, None), out_axes=0)
  ... def forward(model, x):
  ...   return jnp.dot(x, model.w.value)
  ...
  >>> y = forward(model, x)
  >>> y.shape
  (5, 3)

  To control control how graph node substates are vectorized, ``StateAxes``
  can be passed to ``in_axes`` and ``out_axes`` specifying the axes to be
  applied to each substate given a filter. The following example shows how to
  share the parameters between the ensemble members which keeping different
  batch statistics and dropout random state::

    >>> class Foo(nnx.Module):
    ...   def __init__(self):
    ...     self.a = nnx.Param(jnp.arange(4))
    ...     self.b = nnx.BatchStat(jnp.arange(4))
    ...
    >>> state_axes = nnx.StateAxes({nnx.Param: 0, nnx.BatchStat: None})
    >>> @nnx.vmap(in_axes=(state_axes,), out_axes=0)
    ... def mul(foo):
    ...   return foo.a * foo.b
    ...
    >>> foo = Foo()
    >>> y = mul(foo)
    >>> y
    Array([[0, 0, 0, 0],
           [0, 1, 2, 3],
           [0, 2, 4, 6],
           [0, 3, 6, 9]], dtype=int32)
  """
  if f is Missing:
    return functools.partial(
        vmap,
        in_axes=in_axes,
        out_axes=out_axes,
        axis_name=axis_name,
        axis_size=axis_size,
        spmd_axis_name=spmd_axis_name,
        transform_metadata=transform_metadata,
    )  # type: ignore[return-value]

  jax_in_axes = jax.tree.map(
    lambda x: extract.NodeStates.from_prefixes(x.axes, metadata=x)
    if isinstance(x, StateAxes)
    else x,
    in_axes,
  )
  jax_out_axes = jax.tree.map(
    lambda x: extract.NodeStates.from_prefixes(x.axes, metadata=x)
    if isinstance(x, StateAxes)
    else x,
    out_axes,
  )
  vmapped_fn = jax.vmap(
      VmapFn(f, transform_metadata, in_axes, out_axes),
      in_axes=jax_in_axes,
      out_axes=(jax_in_axes, jax_out_axes),
      axis_name=axis_name,
      axis_size=axis_size,
      spmd_axis_name=spmd_axis_name,
  )

  @functools.wraps(f)
  @graph.update_context('vmap')
  def vmap_wrapper(*args, **kwargs):
    args = resolve_kwargs(f, args, kwargs)
    pure_args = extract.to_tree(
        args, prefix=in_axes, split_fn=_vmap_split_fn, ctxtag='vmap'
    )
    pure_args_out, pure_out = vmapped_fn(*pure_args)
    _args_out, out = extract.from_tree(
      (pure_args_out, pure_out), ctxtag='vmap', is_inner=False
    )
    return out

  return vmap_wrapper  # type: ignore


# -------------------------------
# pmap
# -------------------------------


@dataclasses.dataclass(eq=False)
class PmapFn:
  f: tp.Callable[..., tp.Any]
  transform_metadata: tp.Mapping[str, tp.Any]
  in_axes: tp.Any
  out_axes: tp.Any

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  def __call__(self, *pure_args: tuple[tp.Any, ...]):
    if spmd.PARTITION_NAME in self.transform_metadata:
      pure_args = _update_variable_sharding_metadata(
          pure_args, self.transform_metadata, spmd.remove_axis
      )
    args = extract.from_tree(pure_args, ctxtag='pmap', is_inner=True)

    out = self.f(*args)

    args_out = extract.clear_non_graph_nodes(args)
    pure_args_out, pure_out = extract.to_tree(
        (args_out, out),
        prefix=(self.in_axes, self.out_axes),
        split_fn=_vmap_split_fn,
        ctxtag='pmap',
    )
    if spmd.PARTITION_NAME in self.transform_metadata:
      pure_args_out, pure_out = _update_variable_sharding_metadata(
          (pure_args_out, pure_out), self.transform_metadata, spmd.add_axis
      )
    return pure_args_out, pure_out


@tp.overload
def pmap(
    *,
    axis_name: AxisName | None = None,
    in_axes: tp.Any = 0,
    out_axes: tp.Any = 0,
    static_broadcasted_argnums: int | tp.Iterable[int] = (),
    devices: tp.Sequence[jax.Device] | None = None,  # noqa: F811
    backend: str | None = None,
    axis_size: int | None = None,
    donate_argnums: int | tp.Iterable[int] = (),
    global_arg_shapes: tuple[tuple[int, ...], ...] | None = None,
    # nnx specific
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> tp.Callable[[F], F]:
  ...


@tp.overload
def pmap(
    f: F,
    *,
    axis_name: AxisName | None = None,
    in_axes: tp.Any = 0,
    out_axes: tp.Any = 0,
    static_broadcasted_argnums: int | tp.Iterable[int] = (),
    devices: tp.Sequence[jax.Device] | None = None,  # noqa: F811
    backend: str | None = None,
    axis_size: int | None = None,
    donate_argnums: int | tp.Iterable[int] = (),
    global_arg_shapes: tuple[tuple[int, ...], ...] | None = None,
    # nnx specific
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F:
  ...


def pmap(
  f: F | type[Missing] = Missing,
  *,
  axis_name: AxisName | None = None,
  in_axes: tp.Any = 0,
  out_axes: tp.Any = 0,
  static_broadcasted_argnums: int | tp.Iterable[int] = (),
  devices: tp.Sequence[jax.Device] | None = None,  # noqa: F811
  backend: str | None = None,
  axis_size: int | None = None,
  donate_argnums: int | tp.Iterable[int] = (),
  global_arg_shapes: tuple[tuple[int, ...], ...] | None = None,
  # nnx specific
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F | tp.Callable[[F], F]:
  """Reference-aware version of `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__.

  Args:
    f: Function to be mapped over additional axes.
    in_axes: An integer, None, or sequence of values specifying which input
      array axes to map over (see `jax.vmap
      <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__). In
      addition to integers and None, :class:`StateAxes`  can be used to control
      how graph nodes like Modules are vectorized by specifying the axes to be
      applied to substates of the graph node given a `Filter
      <https://flax.readthedocs.io/en/latest/nnx/filters_guide.html>`__.
    out_axes: An integer, None, or pytree indicating where the mapped axis
      should appear in the output (see `jax.vmap
      <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__).
    axis_name: Optional, a hashable Python object used to identify the mapped
      axis so that parallel collectives can be applied.
    axis_size: Optional, an integer indicating the size of the axis to be
      mapped. If not provided, the mapped axis size is inferred from arguments.

  Returns:
    Batched/vectorized version of ``f`` with arguments that correspond to
    those of ``f``, but with extra array axes at positions indicated by
    ``in_axes``, and a return value that corresponds to that of ``f``, but
    with extra array axes at positions indicated by ``out_axes``.

  Example::

    >>> from flax import nnx
    >>> from jax import random, numpy as jnp
    ...
    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> x = jnp.ones((5, 2))
    ...
    >>> @nnx.vmap(in_axes=(None, 0), out_axes=0)
    ... def forward(model, x):
    ...   return model(x)
    ...
    >>> y = forward(model, x)
    >>> y.shape
    (5, 3)

  >>> class LinearEnsemble(nnx.Module):
  ...   def __init__(self, num, rngs):
  ...     self.w = nnx.Param(jax.random.uniform(rngs(), (num, 2, 3)))
  ...
  >>> model = LinearEnsemble(5, rngs=nnx.Rngs(0))
  >>> x = jnp.ones((2,))
  ...
  >>> @nnx.vmap(in_axes=(0, None), out_axes=0)
  ... def forward(model, x):
  ...   return jnp.dot(x, model.w.value)
  ...
  >>> y = forward(model, x)
  >>> y.shape
  (5, 3)

  To control control how graph node substates are vectorized, ``StateAxes``
  can be passed to ``in_axes`` and ``out_axes`` specifying the axes to be
  applied to each substate given a filter. The following example shows how to
  share the parameters between the ensemble members which keeping different
  batch statistics and dropout random state::

    >>> class Foo(nnx.Module):
    ...   def __init__(self):
    ...     self.a = nnx.Param(jnp.arange(4))
    ...     self.b = nnx.BatchStat(jnp.arange(4))
    ...
    >>> state_axes = nnx.StateAxes({nnx.Param: 0, nnx.BatchStat: None})
    >>> @nnx.vmap(in_axes=(state_axes,), out_axes=0)
    ... def mul(foo):
    ...   return foo.a * foo.b
    ...
    >>> foo = Foo()
    >>> y = mul(foo)
    >>> y
    Array([[0, 0, 0, 0],
           [0, 1, 2, 3],
           [0, 2, 4, 6],
           [0, 3, 6, 9]], dtype=int32)
  """
  if f is Missing:
    return functools.partial(
        pmap,
        axis_name=axis_name,
        in_axes=in_axes,
        out_axes=out_axes,
        static_broadcasted_argnums=static_broadcasted_argnums,
        devices=devices,
        backend=backend,
        axis_size=axis_size,
        donate_argnums=donate_argnums,
        global_arg_shapes=global_arg_shapes,
        transform_metadata=transform_metadata,
    )  # type: ignore[return-value]

  jax_in_axes = jax.tree.map(
    lambda x: extract.NodeStates.from_prefixes(x.axes, metadata=x)
    if isinstance(x, StateAxes)
    else x,
    in_axes,
  )
  jax_out_axes = jax.tree.map(
    lambda x: extract.NodeStates.from_prefixes(x.axes, metadata=x)
    if isinstance(x, StateAxes)
    else x,
    out_axes,
  )
  pmapped_fn = jax.pmap(
      PmapFn(f, transform_metadata, in_axes, out_axes),
      axis_name=axis_name,
      in_axes=jax_in_axes,
      out_axes=(jax_in_axes, jax_out_axes),
      static_broadcasted_argnums=static_broadcasted_argnums,
      devices=devices,
      backend=backend,
      axis_size=axis_size,
      donate_argnums=donate_argnums,
      global_arg_shapes=global_arg_shapes,
  )

  @functools.wraps(f)
  @graph.update_context('pmap')
  def vmap_wrapper(*args):
    pure_args = extract.to_tree(
        args, prefix=in_axes, split_fn=_vmap_split_fn, ctxtag='pmap'
    )
    pure_args_out, pure_out = pmapped_fn(*pure_args)
    _args_out, out = extract.from_tree(
      (pure_args_out, pure_out), ctxtag='pmap', is_inner=False
    )
    return out

  return vmap_wrapper  # type: ignore


# -------------------------------
# scan
# -------------------------------


class Broadcasted(struct.PyTreeNode):
  data: tp.Any

def _get_carry_argnum(axes, is_in_axes: bool):
  if axes is Carry:
    return 'all'
  elif isinstance(axes, int) or axes is None:
    return None

  obj_repr = 'in_axes' if is_in_axes else 'out_axes'
  carry_argnum: int | None = None
  prev_key: tp.Any = None
  for key, x in jax.tree_util.tree_leaves_with_path(axes):
    if x is not Carry:
      continue
    assert isinstance(key[0], jax.tree_util.SequenceKey)
    i = key[0].idx
    if len(key) >= 2:
      raise ValueError(
        f'Carry must at the top-level, it cannot be nested. Found {axes=}'
      )
    if carry_argnum is not None:
      raise ValueError(
        f'Found multiple Carry definitions at '
        f'{obj_repr}{jax.tree_util.keystr(prev_key)} and '
        f'{obj_repr}{jax.tree_util.keystr(key)}'
      )
    carry_argnum = i
    prev_key = key

  return carry_argnum


def _check_out_axes(out_axes):
  for key, x in jax.tree_util.tree_leaves_with_path(
    out_axes, is_leaf=lambda x: x is None
  ):
    if x is None:
      raise ValueError(
        f'Cannot broadcast output state. '
        f'Got out_axes=None at: out_axes{jax.tree_util.keystr(key)}'
      )
    elif isinstance(x, StateAxes):
      for filter, value in x.items():
        if value is None:
          raise ValueError(
            f'Cannot broadcast output state. '
            f'Got StateAxes({{{filter}: None}}) at: out_axes'
            f'{jax.tree_util.keystr(key)}'
          )
        elif value is Carry:
          raise ValueError(
            f'Cannot carry output state. '
            f'Got StateAxes({{{filter}: Carry}}) at: out_axes'
            f'{jax.tree_util.keystr(key)}'
          )
def _check_carry_same_references(carry_arg, carry_arg_out):
  def check_carry_same_references(key_path, arg, out):
    if (
      not isinstance(arg, jax.Array) or not isinstance(out, jax.Array)
    ) and arg is not out:
      raise ValueError(
        'Carry references must be the same between iterations. '
        f'Got {arg=} with id={id(arg)} and {out=} with id={id(out)} '
        f'at carry{jax.tree_util.keystr(key_path)}'
      )

  jax.tree_util.tree_map_with_path(
    check_carry_same_references, carry_arg, carry_arg_out
  )

def _extract_nodedefs(
  pure_carry_arg_out, carry_nodedefs: list[graph.NodeDef], /
):
  def extract_index_mappings(x):
    if isinstance(x, extract.NodeStates) and isinstance(
      x._graphdef, graph.NodeDef
    ):
      nodedef = x._graphdef
      assert nodedef.outer_index is not None
      carry_nodedefs.append(nodedef)
      x = x.replace(_graphdef=nodedef.with_no_outer_index())
    return x

  pure_carry_arg_out = jax.tree.map(
    extract_index_mappings,
    pure_carry_arg_out,
    is_leaf=lambda x: isinstance(x, extract.NodeStates),
  )

  return pure_carry_arg_out

def _insert_nodedefs(
  pure_carry_arg_out,
  carry_nodedefs: deque[graph.NodeDef],
  /,
):
  def insert_index_mappings(x):
    if isinstance(x, extract.NodeStates) and isinstance(
      x._graphdef, graph.NodeDef
    ):
      nodedef = carry_nodedefs.popleft()
      x = x.replace(_graphdef=nodedef)
    return x

  pure_carry_arg_out = jax.tree.map(
    insert_index_mappings,
    pure_carry_arg_out,
    is_leaf=lambda x: isinstance(x, extract.NodeStates),
  )
  return pure_carry_arg_out


def _scan_split_in(
    carry_deque: PytreeDeque[list[State]],
    broadcast_deque: PytreeDeque[list[State]],
    broadcast_arrays: PytreeDeque[Broadcasted],
    /,
    ctx: graph.SplitContext,
    path,
    prefix,
    x,
):
  if graph.is_graph_node(x):
    vectorized_states: list[State] = []
    carry_states: list[State] = []
    broadcast_states: list[State] = []
    if isinstance(prefix, StateAxes):
      graphdef, *states = ctx.split(x, *prefix.filters)

      for state, axis in zip(states, prefix.axes):
        if axis is None:
          broadcast_states.append(state)
        elif isinstance(axis, int):
          state = jax.tree.map(lambda x: jnp.moveaxis(x, axis, 0), state)
          vectorized_states.append(state)
        else:  # axis is Carry
          carry_states.append(state)

      if not vectorized_states:
        vectorized_states.append(State({}))
      carry_deque.append(carry_states)
      broadcast_deque.append(broadcast_states)
      return extract.NodeStates.from_split(
        graphdef, *vectorized_states, metadata=prefix
      )
    elif isinstance(prefix, int):
      graphdef, state = ctx.split(x)
      state = jax.tree.map(lambda x: jnp.moveaxis(x, prefix, 0), state)
      vectorized_states.append(state)
    elif prefix is None:
      graphdef, state = ctx.split(x)
      broadcast_states.append(state)
      vectorized_states.append(State({}))
    elif prefix is Carry:
      graphdef, state = ctx.split(x)
      carry_states.append(state)
      vectorized_states.append(State({}))
    else:
      raise ValueError(
        f'Invalid axes {prefix} args{jax.tree_util.keystr(path)}'
      )

    if not vectorized_states:
      vectorized_states.append(State({}))
    carry_deque.append(carry_states)
    broadcast_deque.append(broadcast_states)
    return extract.NodeStates.from_split(
      graphdef, *vectorized_states, metadata=prefix
    )
  else:
    if isinstance(prefix, StateAxes):
      raise ValueError(
        'Cannot use StateAxes on non-graph nodes, '
        f'found {prefix} args{jax.tree_util.keystr(path)}'
      )
    elif prefix is Carry:
      return x
    elif prefix is None:
      broadcast_arrays.append(Broadcasted(x))
      return Broadcasted(None)
    elif isinstance(prefix, int):
      if not isinstance(x, (jax.Array, np.ndarray)):
        raise ValueError(
          f'Expected an array, got {type(x).__name__} args'
          f'{jax.tree_util.keystr(path)}'
        )
      return jnp.moveaxis(x, prefix, 0)
    else:
      raise ValueError(
        f'Invalid axes {prefix} args{jax.tree_util.keystr(path)}'
      )


def _scan_split_out(
    carry_deque: PytreeDeque[list[State]],
    broadcast_deque: PytreeDeque[list[State]],
    /,
    ctx: graph.SplitContext,
    path: extract.KeyPath,
    prefix,
    x,
):
  assert isinstance(path[0], jax.tree_util.SequenceKey)
  is_input_arg = path[0].idx == 0

  if graph.is_graph_node(x):
    vectorized_states: list[State] = []
    carry_states: list[State] = []
    broadcast_states: list[State] = []
    if isinstance(prefix, StateAxes):
      graphdef, *states = ctx.split(x, *prefix.filters)

      for state, filter, axis in zip(states, prefix.filters, prefix.axes):
        if axis is None:
          assert is_input_arg  # validated by _check_out_axes
          broadcast_states.append(state)
        elif isinstance(axis, int):
          vectorized_states.append(state)
        elif axis is Carry:
          assert is_input_arg  # validated by _check_out_axes
          carry_states.append(state)
        else:
          obj_repr = 'args' if is_input_arg else 'out'
          raise ValueError(
            f'Invalid axes {axis} for filter {filter} at '
            f'{obj_repr}{jax.tree_util.keystr(path)}'
          )

      if not vectorized_states:
        vectorized_states.append(State({}))
      if is_input_arg:
        carry_deque.append(carry_states)
        broadcast_deque.append(broadcast_states)
      return extract.NodeStates.from_split(
        graphdef, *vectorized_states, metadata=prefix
      )
    elif isinstance(prefix, int):
      graphdef, state = ctx.split(x)
      vectorized_states.append(state)
    elif prefix is None:
      assert is_input_arg  # validated by _check_out_axes
      graphdef, state = ctx.split(x)
      broadcast_states.append(state)
      vectorized_states.append(State({}))
    elif prefix is Carry:
      assert is_input_arg  # validated by _check_out_axes
      graphdef, state = ctx.split(x)
      carry_states.append(state)
      vectorized_states.append(State({}))
    else:
      obj_repr = 'args' if is_input_arg else 'out'
      raise ValueError(
        f'Invalid axes {prefix} at {obj_repr}{jax.tree_util.keystr(path)}'
      )
    if not vectorized_states:
      vectorized_states.append(State({}))
    if is_input_arg:
      carry_deque.append(carry_states)
      broadcast_deque.append(broadcast_states)
    return extract.NodeStates.from_split(
      graphdef, *vectorized_states, metadata=prefix
    )
  else:
    if isinstance(prefix, StateAxes):
      obj_repr = 'args' if is_input_arg else 'out'
      raise ValueError(
        'Cannot use StateAxes on non-graph nodes, '
        f'found {prefix} at {obj_repr}{jax.tree_util.keystr(path)}'
      )
    elif prefix is Carry:
      return x
    elif prefix is None:
      assert not is_input_arg  # validated by _check_out_axes
      return Broadcasted(None)
    elif isinstance(prefix, int):
      return x
    else:
      obj_repr = 'args' if is_input_arg else 'out'
      raise ValueError(
        f'Invalid axes {prefix} at {obj_repr}{jax.tree_util.keystr(path)}'
      )


def _scan_merge_in(
    carry_deque: PytreeDeque[list[State]],
    broadcast_deque: PytreeDeque[list[State]],
    broadcast_arrays: PytreeDeque[Broadcasted],
    /,
    ctx: graph.MergeContext,
    path,
    prefix,
    x,
):
  if isinstance(x, extract.NodeStates):
    carry_states = carry_deque.popleft()
    broadcast_states = broadcast_deque.popleft()
    return ctx.merge(x.graphdef, *x.states, *carry_states, *broadcast_states)
  elif isinstance(x, Broadcasted):
    assert x.data is None
    return broadcast_arrays.popleft().data
  else:
    return x


def _scan_merge_out(
    carry_deque: PytreeDeque[list[State]],
    broadcast_deque: PytreeDeque[list[State]],
    /,
    ctx: graph.MergeContext,
    path,
    prefix,
    x,
):
  assert isinstance(path[0], jax.tree_util.SequenceKey)
  is_input_arg = path[0].idx == 0

  if isinstance(x, extract.NodeStates):
    states: list[State] = []
    if is_input_arg:
      carry_states = deque(carry_deque.popleft())
      broadcast_states = deque(broadcast_deque.popleft())
    else:
      carry_states = deque[State]()
      broadcast_states = deque[State]()
    if isinstance(prefix, StateAxes):
      vectorized_states = deque(x.states)
      for axis in prefix.axes:
        if isinstance(axis, int):
          state = vectorized_states.popleft()
          state = jax.tree.map(lambda x: jnp.moveaxis(x, 0, axis), state)
          states.append(state)
        elif axis is None:
          states.append(broadcast_states.popleft())
        else:  # axis is Carry
          states.append(carry_states.popleft())
      assert not carry_states and not broadcast_states
      assert not vectorized_states or (
        len(vectorized_states) == 1 and not vectorized_states[0]
      )
    elif isinstance(prefix, int):
      state = jax.tree.map(lambda x: jnp.moveaxis(x, 0, prefix), x.state)
      states.extend((state, *carry_states, *broadcast_states))
    elif prefix is None:
      assert is_input_arg
      states.extend(broadcast_states)
    elif prefix is Carry:
      assert is_input_arg
      states.extend(carry_states)
    else:
      obj_repr = 'args' if is_input_arg else 'out'
      raise ValueError(
        f'Invalid axes {prefix} at {obj_repr}{jax.tree_util.keystr(path)}'
      )

    return ctx.merge(x.graphdef, *states)
  else:
    if isinstance(prefix, StateAxes):
      obj_repr = 'args' if is_input_arg else 'out'
      raise ValueError(
        'Cannot use StateAxes on non-graph nodes, '
        f'found {prefix} at {obj_repr}{jax.tree_util.keystr(path)}'
      )
    elif prefix is Carry:
      return x
    elif prefix is None:
      return x
    elif isinstance(prefix, int):
      if not isinstance(x, (jax.Array, np.ndarray)):
        obj_repr = 'args' if is_input_arg else 'out'
        raise ValueError(
          f'Expected an array, got {type(x).__name__} at '
          f'{obj_repr}{jax.tree_util.keystr(path)}'
        )
      return jnp.moveaxis(x, 0, prefix)
    else:
      obj_repr = 'args' if is_input_arg else 'out'
      raise ValueError(
        f'Invalid axes {prefix} at {obj_repr}{jax.tree_util.keystr(path)}'
      )


@dataclasses.dataclass(eq=False)
class ScanFn:
  f: tp.Callable[..., tp.Any]
  input_carry_argnum: int | None | tp.Literal['all']
  output_carry_argnum: int | None | tp.Literal['all']
  in_axes: tp.Any
  out_axes: tp.Any
  transform_metadata: tp.Mapping[str, tp.Any]

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  def __call__(
    self,
    carry: tuple[
      tp.Any,  # carry_arg
      PytreeDeque[list[State]],  # carry_deque
      PytreeDeque[list[State]],  # broadcast_deque
      PytreeDeque[Broadcasted],  # broadcast_arrays
    ],
    pure_args: tuple[tp.Any, ...],
  ):
    pure_carry_arg, carry_deque, broadcast_deque, broadcast_arrays = carry
    broadcast_deque_out = PytreeDeque(broadcast_deque)
    broadcast_arrays_out = PytreeDeque(broadcast_arrays)

    if self.input_carry_argnum == 'all':
      assert pure_args == ()
      pure_args = (pure_carry_arg,)
    elif isinstance(self.input_carry_argnum, int):
      assert pure_args[self.input_carry_argnum] is None
      _pure_args = list(pure_args)
      _pure_args[self.input_carry_argnum] = pure_carry_arg
      pure_args = tuple(_pure_args)
    else:
      assert self.input_carry_argnum is None
      assert pure_carry_arg is None

    if spmd.PARTITION_NAME in self.transform_metadata:
      pure_args = _update_variable_sharding_metadata(
          pure_args, self.transform_metadata, spmd.remove_axis
      )

    args: tuple = extract.from_tree(
      pure_args,
      prefix=self.in_axes,
      merge_fn=functools.partial(
        _scan_merge_in, carry_deque, broadcast_deque, broadcast_arrays
      ),
      is_leaf=lambda x: isinstance(x, (extract.NodeStates, Broadcasted)),
      map_non_graph_nodes=True,
      ctxtag='scan',
      is_inner=True,
    )
    assert not carry_deque and not broadcast_deque and not broadcast_arrays

    out = self.f(*args)

    # extract the carry from the args
    if self.input_carry_argnum == 'all':
      carry_arg = args[0]
    elif isinstance(self.input_carry_argnum, int):
      carry_arg = args[self.input_carry_argnum]
    else:
      assert self.input_carry_argnum is None
      carry_arg = None

    # extract the carry from the output
    if self.output_carry_argnum == 'all':
      carry_arg_out = out
      out = None
    elif isinstance(self.output_carry_argnum, int):
      assert isinstance(out, tuple)
      carry_arg_out = out[self.output_carry_argnum]
      _out = list(out)
      _out[self.output_carry_argnum] = None
      out = tuple(_out)
    else:
      assert self.output_carry_argnum is None
      carry_arg_out = None

    # TODO(cgarciae): allowing new references might lead to inconsistencies with
    # scan's looping semantics and we would also need to propagate the input
    _check_carry_same_references(carry_arg, carry_arg_out)

    args_out: tuple = extract.clear_non_graph_nodes(args)

    # replace the carry from the input args with the carry from the output
    if self.input_carry_argnum == 'all':
      args_out = (carry_arg_out,)
    elif isinstance(self.input_carry_argnum, int):
      _args_out = list(args_out)
      _args_out[self.input_carry_argnum] = carry_arg_out
      args_out = tuple(_args_out)
    else:
      assert self.input_carry_argnum is None
      assert carry_arg_out is None

    carry_deque_out = PytreeDeque[list[State]]()
    _broadcast_deque_out_tmp = PytreeDeque[list[State]]()  # discarded
    pure_args_out: tuple
    pure_args_out, pure_out = extract.to_tree(
      (args_out, out),
      prefix=(self.in_axes, self.out_axes),
      split_fn=functools.partial(
        _scan_split_out, carry_deque_out, _broadcast_deque_out_tmp
      ),
      map_non_graph_nodes=True,
      ctxtag='scan',
    )
    if spmd.PARTITION_NAME in self.transform_metadata:
      pure_args_out, pure_out = _update_variable_sharding_metadata(
        (pure_args_out, pure_out),
        self.transform_metadata,
        spmd.add_axis,
      )

    # extract the pure carry from the pure args
    if self.input_carry_argnum == 'all':
      pure_carry_arg_out = pure_args_out[0]
      pure_args_out = ()
    elif isinstance(self.input_carry_argnum, int):
      pure_carry_arg_out = pure_args_out[self.input_carry_argnum]
      _pure_args_out = list(pure_args_out)
      _pure_args_out[self.input_carry_argnum] = None
      pure_args_out = tuple(_pure_args_out)
    else:
      assert self.input_carry_argnum is None
      pure_carry_arg_out = None

    # next we have to remove all the index_mappings from the NodeDefs
    # in the carry outputs because they are not present in the inputs
    carry_nodedefs: list[graph.NodeDef] = []
    pure_carry_arg_out = _extract_nodedefs(pure_carry_arg_out, carry_nodedefs)

    carry_arg_out = (
      pure_carry_arg_out,
      carry_deque_out,
      broadcast_deque_out,
      broadcast_arrays_out,
    )
    scan_out = (
      carry_nodedefs,
      pure_args_out,
      pure_out,
    )
    return carry_arg_out, scan_out


@tp.overload
def scan(
  *,
  length: int | None = None,
  reverse: bool = False,
  unroll: int | bool = 1,
  _split_transpose: bool = False,
  # extended api
  in_axes: int | None | type[Carry] | tuple[tp.Any, ...] = (Carry, 0),
  out_axes: tp.Any = (Carry, 0),
  # nnx specific
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> tp.Callable[[F], F]:
  ...


@tp.overload
def scan(
  f: F,
  *,
  length: int | None = None,
  reverse: bool = False,
  unroll: int | bool = 1,
  _split_transpose: bool = False,
  # extended api
  in_axes: int | None | type[Carry] | tuple[tp.Any, ...] = (Carry, 0),
  out_axes: tp.Any = (Carry, 0),
  # nnx specific
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F:
  ...


def scan(
  f: F | type[Missing] = Missing,
  *,
  length: int | None = None,
  reverse: bool = False,
  unroll: int | bool = 1,
  _split_transpose: bool = False,
  # extended api
  in_axes: int | None | type[Carry] | tuple[tp.Any, ...] = (Carry, 0),
  out_axes: tp.Any = (Carry, 0),
  # nnx specific
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F | tp.Callable[[F], F]:
  if f is Missing:
    return functools.partial(
        scan,
        length=length,
        reverse=reverse,
        unroll=unroll,
        _split_transpose=_split_transpose,
        in_axes=in_axes,
        out_axes=out_axes,
        transform_metadata=transform_metadata,
    )  # type: ignore[return-value]

  _check_out_axes(out_axes)

  input_carry_argnum = _get_carry_argnum(in_axes, is_in_axes=True)
  output_carry_argnum = _get_carry_argnum(out_axes, is_in_axes=False)

  if (input_carry_argnum is None and output_carry_argnum is not None) or (
    input_carry_argnum is not None and output_carry_argnum is None
  ):
    raise ValueError(
      'If one of in_axes or out_axes has Carry, the other must also have Carry. '
      f'Got {in_axes=!r} and {out_axes=!r}'
    )

  scan_fn = ScanFn(
    f,
    input_carry_argnum,
    output_carry_argnum,
    in_axes,
    out_axes,
    transform_metadata,
  )

  @functools.wraps(f)
  @graph.update_context('scan')
  def scan_wrapper(*args, **kwargs):
    args = resolve_kwargs(f, args, kwargs)

    if in_axes is Carry and len(args) != 1:
      raise ValueError(
        f'When in_axes=Carry, the function must take exactly one argument, '
        f'got {len(args)} arguments.'
      )

    carry_deque = PytreeDeque()
    broadcast_deque = PytreeDeque()
    broadcast_arrays = PytreeDeque()
    pure_args: tuple = extract.to_tree(
      args,
      prefix=in_axes,
      split_fn=functools.partial(
        _scan_split_in, carry_deque, broadcast_deque, broadcast_arrays
      ),
      map_non_graph_nodes=True,
      ctxtag='scan',
    )
    if isinstance(input_carry_argnum, int):
      pure_carry_arg = pure_args[input_carry_argnum]
      _pure_args = list(pure_args)
      _pure_args[input_carry_argnum] = None
      pure_args = tuple(_pure_args)
    elif input_carry_argnum == 'all':
      pure_carry_arg = pure_args[0]
      pure_args = ()
    else:
      assert input_carry_argnum is None
      pure_carry_arg = None

    carry = (pure_carry_arg, carry_deque, broadcast_deque, broadcast_arrays)

    carry_out, scan_out = jax.lax.scan(
      scan_fn,
      carry,
      pure_args,
      length=length,
      reverse=reverse,
      unroll=unroll,
      _split_transpose=_split_transpose,
    )
    (
        pure_carry_arg_out,
        carry_deque_out,
        broadcast_deque_out,
        broadcast_arrays_out,
    ) = carry_out
    (
      carry_nodedefs,
      pure_args_out,
      pure_out,
    ) = scan_out

    # next we have to insert all the index_mappings back into the NodeDefs
    # in the carry outputs
    pure_carry_arg_out = _insert_nodedefs(
      pure_carry_arg_out, deque(carry_nodedefs)
    )

    # insert pure carry into pure_args_out
    if input_carry_argnum == 'all':
      pure_args_out = (pure_carry_arg_out,)
    elif isinstance(input_carry_argnum, int):
      _pure_args_out = list(pure_args_out)
      _pure_args_out[input_carry_argnum] = pure_carry_arg_out
      pure_args_out = tuple(_pure_args_out)
    else:
      assert input_carry_argnum is None
      assert pure_carry_arg_out is None

    args_out, out = extract.from_tree(
      (pure_args_out, pure_out),
      prefix=(in_axes, out_axes),
      merge_fn=functools.partial(
        _scan_merge_out, carry_deque_out, broadcast_deque_out
      ),
      is_leaf=lambda x: isinstance(x, (extract.NodeStates, Broadcasted)),
      map_non_graph_nodes=True,
      ctxtag='scan',
      is_inner=False,
    )

    # extract the carry from args_out
    if input_carry_argnum == 'all':
      carry_arg = args_out[0]
    elif isinstance(input_carry_argnum, int):
      carry_arg = args_out[input_carry_argnum]
    else:
      assert input_carry_argnum is None
      carry_arg = None

    # insert carry into the output
    if output_carry_argnum == 'all':
      out = carry_arg
    elif isinstance(output_carry_argnum, int):
      _out = list(out)
      _out[output_carry_argnum] = carry_arg
      out = tuple(_out)
    else:
      assert output_carry_argnum is None
      assert carry_arg is None

    return out

  return scan_wrapper  # type: ignore





# -------------------------------
# while_loop
# -------------------------------


@dataclasses.dataclass(eq=False)
class WhileLoopCondFn:
  f: tp.Callable[..., tp.Any]

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  def __call__(self, pure_val):
    val = extract.from_tree(pure_val)
    out = self.f(val)
    return out


def _add_fake_index_mapping(tree: tp.Any):
  def per_node_state(node_state: extract.NodeStates | tp.Any):
    if not isinstance(node_state, extract.NodeStates) or not isinstance(
      node_state._graphdef, graph.NodeDef
    ):
      return node_state

    return dataclasses.replace(
      node_state, _graphdef=node_state._graphdef.with_same_outer_index()
    )

  return jax.tree.map(per_node_state, tree,
                      is_leaf=lambda x: isinstance(x, extract.NodeStates))


def _remove_index_mapping(tree: tp.Any):
  """Remove a fake outer_index for the input to match that of the output."""

  def per_node_state(node_state: extract.NodeStates | tp.Any):
    if not isinstance(node_state, extract.NodeStates) or not isinstance(
      node_state._graphdef, graph.NodeDef
    ):
      return node_state
    assert isinstance(node_state._graphdef, graph.NodeDef)
    node_state = dataclasses.replace(
      node_state, _graphdef=node_state._graphdef.with_no_outer_index()
    )
    return node_state

  return jax.tree.map(per_node_state, tree,
                      is_leaf=lambda x: isinstance(x, extract.NodeStates))


@dataclasses.dataclass(eq=False)
class WhileLoopBodyFn:
  f: tp.Callable[..., tp.Any]

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  @graph.update_context('while_loop_body')
  def __call__(self, pure_val):
    # Removing the dummy index mapping being added outside of body function.
    pure_val_in = _remove_index_mapping(pure_val)

    val = extract.from_tree(
      pure_val_in, ctxtag='while_loop_body', is_inner=True
    )
    out = self.f(val)
    pure_out = extract.to_tree(out, ctxtag='while_loop_body')

    try:
      jax.tree.map(lambda a, b: None, pure_val, pure_out)
    except ValueError as e:
      msg = (
        "nnx.while_loop requires body function's input and output to "
        'have the same reference and pytree structure, but they differ. '
        'If the mismatch comes from `outer_index` field, you might '
        'have modified reference structure within the body function, '
        'which is not allowed.'
        f'Detail of the mismatch: \n {str(e)}'
      )
      raise ValueError(msg)

    return pure_out


@graph.update_context('while_loop')
def while_loop(cond_fun: tp.Callable[[T], tp.Any],
               body_fun: tp.Callable[[T], T],
               init_val: T) -> T:
  """A Flax NNX transformation of `jax.lax.while_loop <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html>`_.

  Caution: for the NNX internal reference tracing mechanism to work, you cannot
  change the variable reference structure of ``init_val`` inside ``body_fun``.

  Example::

    >>> import jax
    >>> from flax import nnx
    >>> def fwd_fn(input):
    ...   module, x, count = input
    ...   return module, module(x), count - 1.0

    >>> module = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    >>> x = jax.random.normal(jax.random.key(0), (10,))
    >>> # `module` will be called three times
    >>> _, y, _ = nnx.while_loop(
    ...   lambda input: input[-1] > 0, fwd_fn, (module, x, 3.0))


  Args:
    cond_fun: A function for the continue condition of the while loop, taking a
      single input of type ``T`` and outputting a boolean.
    body_fun: A function that takes an input of type ``T`` and outputs an ``T``.
      Note that both data and modules of ``T`` must have the same reference
      structure between inputs and outputs.
    init_val: The initial input for ``cond_fun`` and ``body_fun``. Must be of type ``T``.

  """

  pure_init_val = extract.to_tree(init_val, ctxtag='while_loop')

  # Adding the expected reference mapping to `pure_init_val` to match
  # `body_fun`'s output pytree structure, to make JAX while_loop happy.
  pure_init_val = _add_fake_index_mapping(pure_init_val)

  pure_out = jax.lax.while_loop(
    WhileLoopCondFn(cond_fun),
    WhileLoopBodyFn(body_fun),
    pure_init_val,
  )
  out = extract.from_tree(pure_out, ctxtag='while_loop', is_inner=False)
  return out


@dataclasses.dataclass(eq=False)
class ForiLoopBodyFn:
  f: tp.Callable[..., tp.Any]

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  @graph.update_context('fori_loop_body')
  def __call__(self, i, pure_val):
    # Removing the dummy index mapping being added outside of body function.
    pure_val_in = _remove_index_mapping(pure_val)

    val = extract.from_tree(pure_val_in, ctxtag='fori_loop_body', is_inner=True)
    out = self.f(i, val)
    pure_out = extract.to_tree(out, ctxtag='fori_loop_body')

    try:
      jax.tree.map(lambda a, b: None, pure_val, pure_out)
    except ValueError as e:
      msg = (
        "nnx.fori_loop requires body function's input and output to "
        'have the same reference and pytree structure, but they differ. '
        'If the mismatch comes from `outer_index` field, you might '
        'have modified reference structure within the body function, '
        'which is not allowed. '
        f'Detail of the mismatch: \n {str(e)}'
      )
      raise ValueError(msg)

    return pure_out


@graph.update_context('fori_loop')
def fori_loop(lower: int, upper: int,
              body_fun: tp.Callable[[int, T], T],
              init_val: T,
              *,
              unroll: int | bool | None = None) -> T:
  """A Flax NNX transformation of `jax.lax.fori_loop <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html>`_.

  Caution: for the NNX internal reference tracing mechanism to work, you cannot
  change the variable reference structure of `init_val` inside `body_fun`.

  Example::

    >>> import jax
    >>> from flax import nnx

    >>> def fwd_fn(i, input):
    ...   m, x = input
    ...   m.kernel.value = jnp.identity(10) * i
    ...   return m, m(x)

    >>> module = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    >>> x = jax.random.normal(jax.random.key(0), (10,))
    >>> _, y = nnx.fori_loop(2, 4, fwd_fn, (module, x))
    >>> np.testing.assert_array_equal(y, x * 2 * 3)


  Args:
    lower: An integer representing the loop index lower bound (inclusive).
    upper: An integer representing the loop index upper bound (exclusive).
    body_fun: a function that takes an input of type ``T`` and outputs an ``T``.
      Note that both data and modules of ``T`` must have the same reference
      structure between inputs and outputs.
    init_val: the initial input for body_fun. Must be of type ``T``.
    unroll: An optional integer or boolean that determines how much to unroll
      the loop. If an integer is provided, it determines how many unrolled
      loop iterations to run within a single rolled iteration of the loop. If a
      boolean is provided, it will determine if the loop is competely unrolled
      (i.e. ``unroll=True``) or left completely unrolled (i.e. ``unroll=False``).
      This argument is only applicable if the loop bounds are statically known.

  Returns:
    A loop value from the final iteration, of type ``T``.

  """

  pure_init_val = extract.to_tree(init_val, ctxtag='fori_loop')

  # Adding the expected reference mapping to `pure_init_val` to match
  # `body_fun`'s output pytree structure, to make JAX happy.
  pure_init_val = _add_fake_index_mapping(pure_init_val)

  pure_out = jax.lax.fori_loop(lower, upper,
                               ForiLoopBodyFn(body_fun), pure_init_val,
                               unroll=unroll)
  out = extract.from_tree(pure_out, ctxtag='fori_loop', is_inner=False)
  return out
