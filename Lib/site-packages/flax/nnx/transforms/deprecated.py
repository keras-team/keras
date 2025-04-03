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
from __future__ import annotations

import dataclasses
import functools
import typing as tp

from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.nnx import extract, filterlib, graph, rnglib, spmd, variablelib
from flax.nnx import statelib
from flax.nnx.module import GraphDef, Module
from flax.nnx.proxy_caller import DelayedAccessor
from flax.nnx.statelib import State
from flax.nnx.transforms.transforms import LiftedModule
from flax.typing import MISSING, Leaf, Missing
import jax
from jax._src.tree_util import broadcast_prefix
import jax.core
import jax.numpy as jnp
import jax.stages
from flax import nnx

A = tp.TypeVar('A')
C = tp.TypeVar('C')
B = tp.TypeVar('B')
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])
G = tp.TypeVar('G', bound=tp.Callable[..., tp.Any])
M = tp.TypeVar('M', bound=Module)
MA = tp.TypeVar('MA', bound=Module)
N = tp.TypeVar('N', bound=Module)
StrInt = tp.TypeVar('StrInt', str, int)
AxisName = tp.Hashable
Leaves = tp.List[Leaf]
Index = int

def _normalize_sequence(
  x: StrInt | tp.Iterable[StrInt] | None, /
) -> tuple[StrInt, ...]:
  if x is None:
    return ()
  elif isinstance(x, (str, int)):
    return (x,)  # type: ignore
  else:
    return tuple(x)


# -------------------------------
# vmap
# -------------------------------
class _VmapForkStates(tp.NamedTuple):
  split_keys: State
  split_counts: State
  broadcast_keys: State
  broadcast_counts: State


def _get_axis_sizes(pytree, axes):
  axes = broadcast_prefix(axes, pytree, is_leaf=lambda x: x is None)
  leaves = jax.tree_util.tree_leaves(pytree)
  axis_sizes = {
    leaf.shape[axis] for axis, leaf in zip(axes, leaves) if axis is not None
  }
  return axis_sizes


def _fork_vmap_keys(
  state: State,
  split_filter: filterlib.Filter,
  num_splits: int,
) -> _VmapForkStates:
  split_keys, split_counts, broadcast_keys, broadcast_counts = (
    statelib.split_state(
      state,
      filterlib.All(split_filter, rnglib.RngKey),
      filterlib.All(split_filter, rnglib.RngCount),
      rnglib.RngKey,
      rnglib.RngCount,
    )
  )

  def split_key(key: tp.Any, count: tp.Any) -> jax.Array:
    if not isinstance(key, jax.Array):
      raise TypeError(f'key must be a jax.Array, got {type(key)}')
    if not isinstance(count, jax.Array):
      raise TypeError(f'count must be a jax.Array, got {type(count)}')

    key = jax.random.fold_in(key, count)
    return jax.random.split(key, num_splits)

  split_keys_leaves, split_keys_treedef = jax.tree.flatten(split_keys)
  split_counts_leaves, split_counts_treedef = jax.tree.flatten(split_counts)

  if len(split_keys_leaves) != len(split_counts_leaves):
    raise ValueError(
      'split_keys and split_counts must have the same number of leaves',
      f'got {len(split_keys_leaves)} and {len(split_counts_leaves)}',
    )

  split_keys_leaves = [
    split_key(key, count)
    for key, count in zip(split_keys_leaves, split_counts_leaves)
  ]
  split_counts_leaves = [
    jnp.full((num_splits,), 0, dtype=jnp.uint32) for _ in split_counts_leaves
  ]
  split_keys = jax.tree.unflatten(split_keys_treedef, split_keys_leaves)
  split_counts = jax.tree.unflatten(split_counts_treedef, split_counts_leaves)

  return _VmapForkStates(
    split_keys, split_counts, broadcast_keys, broadcast_counts
  )


def _backup_vmap_keys(node: tp.Any, /):
  backups: list[
    tuple[graph.PathParts, rnglib.RngStream, jax.Array, jax.Array]
  ] = []
  for path, stream in graph.iter_graph(node):
    if isinstance(stream, rnglib.RngStream):
      backups.append((path, stream, stream.key.value, stream.count.value))
  return backups


def _restore_vmap_keys(
  backups: list[tuple[graph.PathParts, rnglib.RngStream, jax.Array, jax.Array]],
  split_rngs: filterlib.Filter,
  /,
):
  predicate_fn = filterlib.to_predicate(split_rngs)
  for path, stream, key, count in backups:
    stream.key.value = key
    count_path = (*path, 'count')
    if predicate_fn(count_path, stream.count.to_state()):
      # restore count only if it was split
      # add 1 to reflect the split
      stream.count.value = count + 1


def vmap_fn(
  args: tuple[tp.Any, ...],
  kwargs: dict[str, tp.Any],
  graphdef: GraphDef[tuple[tp.Any, ...]],
  split_keys: State,
  split_counts: State,
  broadcast_keys: State,
  broadcast_counts: State,
  vectorized_states: list[State],
  broadcast_state: State,
  transform_metadata: tp.Mapping[str, tp.Any],
  state_axes_: list[tuple[filterlib.Filter, int]],
  f: tp.Callable[..., tp.Any],
  filters: tp.Tuple[filterlib.Filter, ...],
  split_rngs: filterlib.Filter,
):
  ctx = graph.current_update_context('vmap')
  state_axes = dict(state_axes_)
  # remove metadata axis name from Variable.sharding
  if spmd.PARTITION_NAME in transform_metadata:
    vectorized_states = [
      spmd.remove_axis(state, index, transform_metadata)
      for state, index in zip(vectorized_states, state_axes.values())
    ]

  # merge module state
  input_graph_nodes = ctx.merge(
    graphdef,
    *vectorized_states,
    broadcast_state,
    split_keys,
    split_counts,
    broadcast_keys,
    broadcast_counts,
  )

  (args, kwargs) = extract.insert_graph_nodes((args, kwargs), input_graph_nodes)

  out = f(*args, **kwargs)

  out, output_graph_nodes = extract.extract_graph_nodes(out)

  # split module state
  (
    graphdef_out,
    rng_state_out,
    *vectorized_states_out,
    broadcast_state_out,
  ) = ctx.split(  # type: ignore[misc]
    (input_graph_nodes, output_graph_nodes),
    rnglib.RngState,
    *filters,
  )

  split_keys_out, broadcast_keys_out = statelib.split_state(
    rng_state_out, split_rngs, ...
  )

  broadcast_state_out = statelib.merge_state(
    broadcast_state_out, broadcast_keys_out
  )

  # add metadata axis name to Variable.sharding
  if spmd.PARTITION_NAME in transform_metadata:
    vectorized_states_out = [
      spmd.add_axis(state, index, transform_metadata)
      for state, index in zip(vectorized_states_out, state_axes.values())
    ]

  return (
    graphdef_out,
    broadcast_state_out,
    vectorized_states_out,
    split_keys_out,
    out,
  )


@tp.overload
def vmap(
  *,
  in_axes: int | None | tp.Sequence[tp.Any] = 0,
  out_axes: tp.Any = 0,
  axis_name: AxisName | None = None,
  axis_size: int | None = None,
  spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
  # nnx specific
  in_axes_kwargs: tp.Any = 0,
  state_axes: tp.Mapping[filterlib.Filter, int | None] = FrozenDict({...: 0}),
  split_rngs: filterlib.Filter = ...,
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> tp.Callable[[F], F]: ...
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
  in_axes_kwargs: tp.Any = 0,
  state_axes: tp.Mapping[filterlib.Filter, int | None] = FrozenDict({...: 0}),
  split_rngs: filterlib.Filter = ...,
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F: ...
def vmap(
  f: F | Missing = MISSING,
  *,
  in_axes: int | None | tp.Sequence[tp.Any] = 0,
  out_axes: tp.Any = 0,
  axis_name: AxisName | None = None,
  axis_size: int | None = None,
  spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
  # nnx specific
  in_axes_kwargs: tp.Any = 0,
  state_axes: tp.Mapping[filterlib.Filter, int | None] = FrozenDict({...: 0}),
  split_rngs: filterlib.Filter = ...,
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F | tp.Callable[[F], F]:
  if isinstance(f, Missing):
    return functools.partial(
      vmap,
      in_axes=in_axes,
      out_axes=out_axes,
      axis_name=axis_name,
      axis_size=axis_size,
      spmd_axis_name=spmd_axis_name,
      in_axes_kwargs=in_axes_kwargs,
      state_axes=state_axes,
      split_rngs=split_rngs,
      transform_metadata=transform_metadata,
    )  # type: ignore[return-value]

  vectorized_states_axes = list(state_axes.values())
  vmapped_fn = jax.vmap(
    vmap_fn,
    in_axes=(
      in_axes,  # args
      in_axes_kwargs,  # kwargs
      None,  # graphdef
      0,  # split_keys
      0,  # split_counts
      None,  # broadcast_keys
      None,  # broadcast_counts
      vectorized_states_axes,  # vectorized_states
      None,  # broadcast_state
      None,  # transform_metadata
      None,  # states_axes
      None,  # f
      None,  # vectorized_states_filters
      None,  # split_rngs
    ),
    out_axes=(
      None,  # graphdef_out
      None,  # broadcast_state
      vectorized_states_axes,
      0,  # keys_out
      out_axes,  # out_axes
    ),
    axis_name=axis_name,
    axis_size=axis_size,
    spmd_axis_name=spmd_axis_name,
  )

  @functools.wraps(f)
  @graph.update_context('vmap')
  def vmap_wrapper(*args, **kwargs):
    ctx = graph.current_update_context('vmap')

    (args, kwargs), input_graph_nodes = extract.extract_graph_nodes(
      (args, kwargs)
    )
    input_rng_streams = _backup_vmap_keys(input_graph_nodes)

    # split module state
    filters = (*state_axes.keys(), ...)
    graphdef, rng_state, *vectorized_states, broadcast_state = ctx.split(  # type: ignore[misc]
      input_graph_nodes, rnglib.RngState, *filters
    )

    # infer length
    axis_sizes: tp.Set[int] = set()
    axis_sizes.update(_get_axis_sizes(args, in_axes))
    axis_sizes.update(_get_axis_sizes(kwargs, in_axes_kwargs))
    for state, state_axis in zip(vectorized_states, state_axes.values()):
      axis_sizes.update(_get_axis_sizes(state, state_axis))

    if len(axis_sizes) > 1:
      raise ValueError(
        'Inconsistent lengths between state_axes states and '
        f'arguments: {axis_sizes}'
      )
    elif len(axis_sizes) == 0:
      if axis_size is None:
        raise ValueError(
          'Cannot infer length from state_axes states or axes_arg, '
          'please specify `length`'
        )
      _axis_size = axis_size
    else:
      _axis_size = axis_sizes.pop()
      if axis_size is not None and axis_size != _axis_size:
        raise ValueError(
          f'Specified axis_size {axis_size} is not the same as the'
          f' inferred length {_axis_size}'
        )

    split_keys, split_counts, broadcast_keys, broadcast_counts = (
      _fork_vmap_keys(
        rng_state,
        split_rngs,
        _axis_size,
      )
    )

    (
      graphdef_out,
      broadcast_state,
      vectorized_states,
      split_keys_out,
      out,
    ) = vmapped_fn(
      args,
      kwargs,
      graphdef,
      split_keys,
      split_counts,
      broadcast_keys,
      broadcast_counts,
      vectorized_states,
      broadcast_state,
      transform_metadata,
      list(state_axes.items()),
      f,
      filters,
      split_rngs,
    )

    _, output_graph_nodes = ctx.merge(
      graphdef_out,
      *vectorized_states,
      broadcast_state,
      split_keys_out,
    )

    out = extract.insert_graph_nodes(out, output_graph_nodes)

    _restore_vmap_keys(input_rng_streams, split_rngs)

    return out

  return vmap_wrapper  # type: ignore


class Vmap(tp.Generic[M], LiftedModule[M]):
  @staticmethod
  def constructor(
    module_constructor: tp.Callable[..., MA],
    *,
    in_axes: int | None | tp.Sequence[tp.Any] = 0,
    out_axes: tp.Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    # nnx specific
    in_axes_kwargs: tp.Any = 0,
    state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
    split_rngs: filterlib.Filter = ...,
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
  ) -> tp.Callable[..., Vmap[MA]]:
    def _create_vmap(*args, **kwargs):
      return Vmap(
        module_constructor=module_constructor,
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=axis_size,
        axis_name=axis_name,
        spmd_axis_name=spmd_axis_name,
        # nnx specific
        in_axes_kwargs=in_axes_kwargs,
        state_axes=state_axes,
        split_rngs=split_rngs,
        transform_metadata=transform_metadata,
        # submodule args
        module_init_args=args,
        module_init_kwargs=kwargs,
      )

    return _create_vmap

  def __init__(
    self,
    module_constructor: tp.Callable[..., M],
    *,
    in_axes: int | None | tp.Sequence[tp.Any] = 0,
    out_axes: tp.Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    # nnx specific
    in_axes_kwargs: tp.Any = 0,
    state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
    split_rngs: filterlib.Filter = ...,
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
    # submodule args
    module_init_args: tuple[tp.Any, ...],
    module_init_kwargs: dict[str, tp.Any],
  ):
    self.module_constructor = module_constructor

    @functools.partial(
      vmap,
      in_axes=None,
      out_axes=None,
      axis_name=axis_name,
      axis_size=axis_size,
      spmd_axis_name=spmd_axis_name,
      in_axes_kwargs=None,
      state_axes=state_axes,
      split_rngs=split_rngs,
      transform_metadata=transform_metadata,
    )
    def vmap_init(*args, **kwargs):
      return module_constructor(*args, **kwargs)

    self.vmap_module = vmap_init(*module_init_args, **module_init_kwargs)

    @functools.partial(
      vmap,
      in_axes=in_axes,
      out_axes=out_axes,
      axis_name=axis_name,
      axis_size=axis_size,
      spmd_axis_name=spmd_axis_name,
      in_axes_kwargs=in_axes_kwargs,
      state_axes=state_axes,
      split_rngs=split_rngs,
      transform_metadata=transform_metadata,
    )
    def vmap_call(module, *args, _nnx_vmap_accessor: DelayedAccessor, **kwargs):
      method = _nnx_vmap_accessor(module)
      return method(*args, **kwargs)

    self.vmap_call = vmap_call

  @property
  def _submodule(self) -> M:
    return self.vmap_module

  def _call(self, accessor: DelayedAccessor, *args, **kwargs):
    return self.vmap_call(
      self._submodule, *args, _nnx_vmap_accessor=accessor, **kwargs
    )  # type: ignore[type-var, call-arg]


# -------------------------------
# pmap
# -------------------------------
@struct.dataclass
class PmapInputs:
  transform_metadata: tp.Mapping[str, tp.Any] = struct.field(pytree_node=False)
  state_axes: tp.Mapping[filterlib.Filter, int] = struct.field(
    pytree_node=False
  )
  f: tp.Callable[..., tp.Any] = struct.field(pytree_node=False)
  filters: tp.Tuple[filterlib.Filter, ...] = struct.field(pytree_node=False)
  split_rngs: filterlib.Filter = struct.field(pytree_node=False)


def pmap_fn(
  args: tuple[tp.Any, ...],
  kwargs: dict[str, tp.Any],
  graphdef: GraphDef[tuple[tp.Any, ...]],
  split_keys: State,
  split_counts: State,
  broadcast_keys: State,
  broadcast_counts: State,
  vectorized_states: list[State],
  broadcast_state: State,
  pmap_inputs: PmapInputs,
):
  transform_metadata = pmap_inputs.transform_metadata
  state_axes = pmap_inputs.state_axes
  f = pmap_inputs.f
  filters = pmap_inputs.filters
  split_rngs = pmap_inputs.split_rngs
  ctx = graph.current_update_context('pmap')
  # remove metadata axis name from Variable.sharding
  if spmd.PARTITION_NAME in transform_metadata:
    vectorized_states = [
      spmd.remove_axis(state, index, transform_metadata)
      for state, index in zip(vectorized_states, state_axes.values())
    ]

  # merge module state
  input_graph_nodes = ctx.merge(
    graphdef,
    *vectorized_states,
    broadcast_state,
    split_keys,
    split_counts,
    broadcast_keys,
    broadcast_counts,
  )

  (args, kwargs) = extract.insert_graph_nodes((args, kwargs), input_graph_nodes)

  out = f(*args, **kwargs)

  out, output_graph_nodes = extract.extract_graph_nodes(out)

  # split module state
  (
    graphdef_out,
    rng_state_out,
    *vectorized_states_out,
    broadcast_state_out,
  ) = ctx.split(  # type: ignore[misc]
    (input_graph_nodes, output_graph_nodes),
    rnglib.RngState,
    *filters,
  )

  not_keys_out, split_keys_out, broadcast_keys_out = statelib.split_state(
    rng_state_out, rnglib.NotKey, split_rngs, ...
  )

  broadcast_state_out = statelib.merge_state(
    broadcast_state_out, broadcast_keys_out, not_keys_out
  )

  # add metadata axis name to Variable.sharding
  if spmd.PARTITION_NAME in transform_metadata:
    vectorized_states_out = [
      spmd.add_axis(state, index, transform_metadata)
      for state, index in zip(vectorized_states_out, state_axes.values())
    ]

  return (
    graphdef_out,
    broadcast_state_out,
    vectorized_states_out,
    split_keys_out,
    out,
  )


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
  in_axes_kwargs: tp.Any = 0,
  state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
  split_rngs: filterlib.Filter = ...,
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> tp.Callable[[F], F]: ...
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
  in_axes_kwargs: tp.Any = 0,
  state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
  split_rngs: filterlib.Filter = ...,
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F: ...
def pmap(
  f: F | Missing = MISSING,
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
  in_axes_kwargs: tp.Any = 0,
  state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
  split_rngs: filterlib.Filter = ...,
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F | tp.Callable[[F], F]:
  if isinstance(f, Missing):
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
      in_axes_kwargs=in_axes_kwargs,
      state_axes=state_axes,
      split_rngs=split_rngs,
      transform_metadata=transform_metadata,
    )  # type: ignore[return-value]
  if static_broadcasted_argnums:
    raise NotImplementedError(
      'static_broadcasted_argnums is not yet supported in nnx.pmap'
    )
  if donate_argnums != ():
    raise NotImplementedError('donate_argnums is not yet supported in nnx.pmap')

  if global_arg_shapes is not None:
    raise NotImplementedError(
      'global_arg_shapes is not yet supported in nnx.pmap'
    )

  vectorized_states_axes = list(state_axes.values())

  pmapped_fn = jax.pmap(
    pmap_fn,
    axis_name=axis_name,
    in_axes=(
      in_axes,  # args_axes
      in_axes_kwargs,  # kwargs_axes
      None,  # graphdef_axes
      0,  # split_keys_axes
      None,  # split_counts_axes
      None,  # broadcast_keys_axes
      None,  # broadcast_counts_axes
      vectorized_states_axes,  # vectorized_states_axes
      None,  # broadcast_state_axes
      None,  # pmap_inputs_axes
    ),  # type: ignore
    out_axes=(
      None,  # graphdef_out_axes
      None,  # broadcast_state_axes
      vectorized_states_axes,
      0,  # keys_axes_out
      out_axes,  # out_axes
    ),  # type: ignore
    devices=devices,
    backend=backend,
    axis_size=axis_size,
  )

  @functools.wraps(f)
  @graph.update_context('pmap')
  def pmap_wrapper(*args, **kwargs):
    ctx = graph.current_update_context('pmap')

    (args, kwargs), input_graph_nodes = extract.extract_graph_nodes(
      (args, kwargs)
    )
    input_rng_streams = rnglib.backup_keys(input_graph_nodes)

    # split module state
    filters = (*state_axes.keys(), ...)
    graphdef, rng_state, *vectorized_states, broadcast_state = ctx.split(  # type: ignore[misc]
      input_graph_nodes, rnglib.RngState, *filters
    )

    # infer length
    axis_sizes: tp.Set[int] = set()
    axis_sizes.update(_get_axis_sizes(args, in_axes))
    axis_sizes.update(_get_axis_sizes(kwargs, in_axes_kwargs))
    for state, state_axis in zip(vectorized_states, state_axes.values()):
      axis_sizes.update(_get_axis_sizes(state, state_axis))

    if len(axis_sizes) > 1:
      raise ValueError(
        'Inconsistent lengths between state_axes states and '
        f'arguments: {axis_sizes}'
      )
    elif len(axis_sizes) == 0:
      if axis_size is None:
        raise ValueError(
          'Cannot infer length from state_axes states or axes_arg, '
          'please specify `length`'
        )
      _axis_size = axis_size
    else:
      _axis_size = axis_sizes.pop()
      if axis_size is not None and axis_size != _axis_size:
        raise ValueError(
          f'Specified axis_size {axis_size} is not the same as the'
          f' inferred length {_axis_size}'
        )

    split_keys, split_counts, broadcast_keys, broadcast_counts = rnglib.fork(
      rng_state,
      split_rngs,
      _axis_size,
    )

    (
      graphdef_out,
      broadcast_state,
      vectorized_states,
      split_keys_out,
      out,
    ) = pmapped_fn(
      args,
      kwargs,
      graphdef,
      split_keys,
      split_counts,
      broadcast_keys,
      broadcast_counts,
      vectorized_states,
      broadcast_state,
      PmapInputs(
        transform_metadata=transform_metadata,
        state_axes=state_axes,
        f=f,
        filters=filters,
        split_rngs=split_rngs,
      ),
    )

    _, output_graph_nodes = ctx.merge(
      graphdef_out,
      *vectorized_states,
      broadcast_state,
      split_keys_out,
    )

    out = extract.insert_graph_nodes(out, output_graph_nodes)

    rnglib.restore_rngs(input_rng_streams)

    return out

  return pmap_wrapper  # type: ignore


class Pmap(tp.Generic[M], LiftedModule[M]):
  @staticmethod
  def constructor(
    module_constructor: tp.Callable[..., MA],
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
    in_axes_kwargs: tp.Any = 0,
    state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
    split_rngs: filterlib.Filter = ...,
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
  ) -> tp.Callable[..., Pmap[MA]]:
    def _create_pmap(*args, **kwargs):
      return Pmap(
        module_constructor=module_constructor,
        axis_name=axis_name,
        in_axes=in_axes,
        out_axes=out_axes,
        static_broadcasted_argnums=static_broadcasted_argnums,
        devices=devices,
        backend=backend,
        axis_size=axis_size,
        # nnx specific
        in_axes_kwargs=in_axes_kwargs,
        state_axes=state_axes,
        split_rngs=split_rngs,
        transform_metadata=transform_metadata,
        # submodule args
        module_init_args=args,
        module_init_kwargs=kwargs,
      )

    return _create_pmap

  def __init__(
    self,
    module_constructor: tp.Callable[..., M],
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
    in_axes_kwargs: tp.Any = 0,
    state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
    split_rngs: filterlib.Filter = ...,
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
    # submodule args
    module_init_args: tuple[tp.Any, ...],
    module_init_kwargs: dict[str, tp.Any],
  ):
    self.module_constructor = module_constructor

    @pmap(
      axis_name=axis_name,
      in_axes=None,
      out_axes=None,
      static_broadcasted_argnums=static_broadcasted_argnums,
      devices=devices,
      backend=backend,
      axis_size=axis_size,
      donate_argnums=(),
      global_arg_shapes=None,
      in_axes_kwargs=None,
      state_axes=state_axes,
      split_rngs=split_rngs,
      transform_metadata=transform_metadata,
    )
    def pmap_init(*args, **kwargs):
      return module_constructor(*args, **kwargs)

    self.pmap_module = pmap_init(*module_init_args, **module_init_kwargs)

    @pmap(
      axis_name=axis_name,
      in_axes=in_axes,
      out_axes=out_axes,
      static_broadcasted_argnums=static_broadcasted_argnums,
      devices=devices,
      backend=backend,
      axis_size=axis_size,
      donate_argnums=donate_argnums,
      global_arg_shapes=global_arg_shapes,
      in_axes_kwargs=in_axes_kwargs,
      state_axes=state_axes,
      split_rngs=split_rngs,
      transform_metadata=transform_metadata,
    )
    def pmap_call(module, *args, _nnx_vmap_accessor: DelayedAccessor, **kwargs):
      method = _nnx_vmap_accessor(module)
      return method(*args, **kwargs)

    self.pmap_call = pmap_call

  @property
  def _submodule(self) -> M:
    return self.pmap_module

  def _call(self, accessor: DelayedAccessor, *args, **kwargs):
    return self.pmap_call(
      self._submodule, *args, _nnx_vmap_accessor=accessor, **kwargs
    )


# -------------------------------
# scan
# -------------------------------


@dataclasses.dataclass(frozen=True)
class FlatDef(tp.Generic[A]):
  type: type[A]
  treedef: jax.tree_util.PyTreeDef
  flat_axes: list[int | None]


jax.tree_util.register_static(FlatDef)


def _transpose_tree(tree: A, axes, /, *, move_front: bool) -> A:
  flatdef, flat_transposes, _ = _transpose_and_split(
    tree, axes, allow_none=False, move_front=move_front
  )
  return flatdef.treedef.unflatten(flat_transposes)


def _transpose_and_split(
  tree: A, axes, /, *, allow_none: bool = True, move_front: bool = True
) -> tuple[
  FlatDef[A],
  list[jax.Array | None],
  list[tp.Any],
]:
  flat_axes: list[int | None] = broadcast_prefix(
    axes, tree, is_leaf=lambda x: x is None
  )
  flat_tree, treedef = jax.tree.flatten(tree)

  flat_broadcasts: list[tp.Any] = []
  flat_transposes: list[jax.Array | None] = []

  for i, (axis, node) in enumerate(zip(flat_axes, flat_tree)):
    if axis is None:
      if not allow_none:
        raise ValueError('None axis not allowed')

      flat_broadcasts.append(node)
      flat_transposes.append(None)
    else:
      if not isinstance(node, jax.Array):
        raise TypeError(
          f'Expected a jax.Array, got {type(node).__name__} for axis {axis}'
        )
      # normalize axis
      if axis < 0:
        if axis < -len(node.shape):
          raise ValueError(
            f'Axis {axis} out of bounds for array with shape {node.shape}'
          )
        axis = len(node.shape) + axis
        flat_axes[i] = axis

      if node.shape == ():
        raise ValueError(f'Cannot map over a scalar array, got {node}')
      elif axis >= len(node.shape):
        raise ValueError(
          f'Axis {axis} out of bounds for array with shape {node.shape}'
        )

      if move_front:
        node = jnp.moveaxis(node, axis, 0)
      else:
        node = jnp.moveaxis(node, 0, axis)
      flat_broadcasts.append(None)
      flat_transposes.append(node)

  flatdef = FlatDef(type(tree), treedef, flat_axes)

  return flatdef, flat_transposes, flat_broadcasts


def _unflatten_splits(
  flatdef: FlatDef[A],
  flat_transposes: list[jax.Array | None],
  flat_broadcasts: list[tp.Any] | None = None,
  /,
  *,
  allow_none: bool = True,
) -> A:
  flat_axes = flatdef.flat_axes
  treedef = flatdef.treedef
  if flat_broadcasts is None:
    if allow_none:
      raise ValueError('flat_broadcasts must be provided if allow_none is True')
    flat_broadcasts = [None] * len(flat_axes)

  flat_tree = []
  for axis, transpose, broadcast in zip(
    flat_axes, flat_transposes, flat_broadcasts
  ):
    if axis is None:
      if not allow_none:
        raise ValueError('None axis not allowed')
      flat_tree.append(broadcast)
    else:
      if transpose is None:
        raise ValueError('None transpose not allowed')
      flat_tree.append(transpose)

  tree = treedef.unflatten(flat_tree)
  return tree


def _extract_carry_arg(
  args: tuple[tp.Any, ...], carry_argnum: int, /
) -> tuple[tp.Any, tuple[tp.Any, ...]]:
  # extract carry arg
  if len(args) < carry_argnum + 1:
    raise TypeError(
      f'Expected at least {carry_argnum + 1} positional arguments, '
      f'got {len(args)}'
    )

  args_ = list(args)
  carry_arg = args_[carry_argnum]
  args_[carry_argnum] = None
  args = tuple(args_)

  return carry_arg, args


def _insert_carry_arg(
  args: tuple[tp.Any, ...], carry_argnum: int, carry_arg: tp.Any, /
) -> tuple[tp.Any, ...]:
  args_ = list(args)
  args_[carry_argnum] = carry_arg
  args = tuple(args_)

  return args


@struct.dataclass
class ScanBroadcasts(tp.Generic[C, B]):
  flatdef: FlatDef[
    tuple[tuple[tp.Any, ...], dict[str, tp.Any], list[State]]
  ] = struct.field(pytree_node=False)
  flat_carry: list[tp.Any] = struct.field(pytree_node=True)
  graphdef: GraphDef[tuple[tp.Any, ...]] = struct.field(pytree_node=False)
  filters: tuple[filterlib.Filter, ...] = struct.field(pytree_node=False)
  f: tp.Callable[..., tuple[C, B] | C] = struct.field(pytree_node=False)
  # options
  carry_argnum: int = struct.field(pytree_node=False)
  state_axes: tp.Mapping[filterlib.Filter, int] = struct.field(
    pytree_node=False
  )
  split_rngs: filterlib.Filter = struct.field(pytree_node=False)
  transform_metadata: tp.Mapping[str, tp.Any] = struct.field(pytree_node=False)
  scan_output: bool = struct.field(pytree_node=False)


def scan_fn(
  carry: tuple[
    State,  # split_rng_state
    State,  # broadcast_rng_state
    State,  # carry_state
    tp.Any,  # carry_arg
    ScanBroadcasts[C, B],  # broadcasts
  ],
  scan: tuple[
    list[jax.Array | None],  # flat_scan
  ],
):
  split_rng_state, broadcast_rng_state, carry_state, carry_arg, broadcasts = (
    carry
  )
  (flat_scan,) = scan
  flatdef = broadcasts.flatdef
  flat_carry = broadcasts.flat_carry
  graphdef, filters = broadcasts.graphdef, broadcasts.filters
  f = broadcasts.f
  ctx = graph.current_update_context('scan')

  # merge args and kwargs
  args, kwargs, scan_states = _unflatten_splits(flatdef, flat_scan, flat_carry)
  # remove metadata axis name from Variable.sharding
  if spmd.PARTITION_NAME in broadcasts.transform_metadata:
    scan_states = [
      spmd.remove_axis(state, index, broadcasts.transform_metadata)
      for state, index in zip(scan_states, broadcasts.state_axes.values())
    ]

  # insert carry arg
  args = _insert_carry_arg(args, broadcasts.carry_argnum, carry_arg)

  # merge module state
  input_graph_nodes = ctx.merge(
    graphdef, *scan_states, carry_state, split_rng_state, broadcast_rng_state
  )
  (args, kwargs) = extract.insert_graph_nodes((args, kwargs), input_graph_nodes)

  out = f(*args, **kwargs)

  if broadcasts.scan_output:
    if not isinstance(out, tuple) or len(out) != 2:
      raise ValueError(
        'Expected a tuple of length 2 as the output of the scan function, '
        f'got {out}'
      )
    out = tp.cast(tuple[C, B], out)  # type: ignore[invalid-annotation]
    carry_arg_out, scan_args_out = out
  else:
    out = tp.cast(C, out)  # type: ignore[invalid-annotation]
    carry_arg_out = out
    scan_args_out = None

  ((carry_arg_out, scan_args_out), output_graph_nodes) = (
    extract.extract_graph_nodes((carry_arg_out, scan_args_out))
  )

  # split module state
  (
    graphdef_out,
    rng_state_out,
    *scan_states_out,
    carry_state_out,
  ) = ctx.split(  # type: ignore[misc]
    (input_graph_nodes, output_graph_nodes),
    rnglib.RngState,
    *filters,
  )

  split_rng_state_out, broadcast_rng_state_out = statelib.split_state(
    rng_state_out, broadcasts.split_rngs, ...
  )

  def _extract_carry_state(state: State, /):
    if 1 in state:
      raise ValueError(
        f'Cannot add new carry state during scan, got {state[1]}'
      )
    if 0 in state:
      _state = state[0]
      assert isinstance(_state, State)
      state = _state

    return state

  carry_state_out = _extract_carry_state(carry_state_out)
  split_rng_state_out = _extract_carry_state(split_rng_state_out)
  broadcast_rng_state_out = _extract_carry_state(broadcast_rng_state_out)

  # override  broadcast_rng_state_out to keep the same state
  # for the next iteration
  broadcast_rng_state_out = broadcast_rng_state

  # add metadata axis name to Variable.sharding
  if spmd.PARTITION_NAME in broadcasts.transform_metadata:
    scan_states_out = [
      spmd.add_axis(state, index, broadcasts.transform_metadata)
      for state, index in zip(scan_states_out, broadcasts.state_axes.values())
    ]

  carry_out = (
    split_rng_state_out,
    broadcast_rng_state_out,
    carry_state_out,
    carry_arg_out,
    broadcasts,
  )
  scan_out = (graphdef_out, scan_args_out, scan_states_out)

  return carry_out, scan_out


@tp.overload
def scan(
  *,
  length: int | None = None,
  reverse: bool = False,
  unroll: int | bool = 1,
  _split_transpose: bool = False,
  # extended api
  in_axes: int | None | tp.Sequence[tp.Any] = 0,
  in_axes_kwargs: tp.Any = 0,
  out_axes: tp.Any = 0,
  carry_argnum: int = 0,
  # nnx specific
  state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
  split_rngs: filterlib.Filter = ...,
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
  scan_output: bool = True,
) -> tp.Callable[[F], F]: ...
@tp.overload
def scan(
  f: F,
  *,
  length: int | None = None,
  reverse: bool = False,
  unroll: int | bool = 1,
  _split_transpose: bool = False,
  # extended api
  in_axes: int | None | tp.Sequence[tp.Any] = 0,
  in_axes_kwargs: tp.Any = 0,
  out_axes: tp.Any = 0,
  carry_argnum: int = 0,
  # nnx specific
  state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
  split_rngs: filterlib.Filter = ...,
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
  scan_output: bool = True,
) -> F: ...
def scan(
  f: F | Missing = MISSING,
  *,
  length: int | None = None,
  reverse: bool = False,
  unroll: int | bool = 1,
  _split_transpose: bool = False,
  # extended api
  in_axes: int | None | tp.Sequence[tp.Any] = 0,
  in_axes_kwargs: tp.Any = 0,
  out_axes: tp.Any = 0,
  carry_argnum: int = 0,
  # nnx specific
  state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
  split_rngs: filterlib.Filter = ...,
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
  scan_output: bool = True,
) -> F | tp.Callable[[F], F]:
  if isinstance(f, Missing):
    return functools.partial(
        scan,
        length=length,
        reverse=reverse,
        unroll=unroll,
        _split_transpose=_split_transpose,
        in_axes=in_axes,
        in_axes_kwargs=in_axes_kwargs,
        out_axes=out_axes,
        carry_argnum=carry_argnum,
        state_axes=state_axes,
        split_rngs=split_rngs,
        transform_metadata=transform_metadata,
        scan_output=scan_output,
    )  # type: ignore[return-value]

  @functools.wraps(f)
  @graph.update_context('scan')
  def scan_apply_wrapper(*args, **kwargs):
    # extract nodes
    (args, kwargs), input_graph_nodes = extract.extract_graph_nodes(
      (args, kwargs)
    )
    input_rng_streams = rnglib.backup_keys(input_graph_nodes)

    # extract carry arg
    carry_arg, args = _extract_carry_arg(args, carry_argnum)

    ctx = graph.current_update_context('scan')
    # split module state
    filters = (*state_axes.keys(), ...)
    graphdef, rng_state, *scan_states, carry_state = ctx.split(  # type: ignore[misc]
      input_graph_nodes, rnglib.RngState, *filters
    )

    # transpose axes arg
    flatdef, flat_scan, flat_carry = _transpose_and_split(
      (args, kwargs, scan_states),
      (in_axes, in_axes_kwargs, list(state_axes.values())),
    )

    # infer length
    lengths: set[int] = {
      x.shape[0]  # type: ignore
      for x, axis in zip(flat_scan, flatdef.flat_axes)
      if axis is not None
    }

    if len(lengths) > 1:
      raise ValueError(
        'Inconsistent lengths between state_axes states and '
        f'arguments: {lengths}'
      )
    elif len(lengths) == 0:
      if length is None:
        raise ValueError(
          'Cannot infer length from state_axes states or axes_arg, '
          'please specify `length`'
        )
      infered_length = length
    else:
      infered_length = lengths.pop()
      if length is not None and length != infered_length:
        raise ValueError(
          f'Specified length {length} is not the same as the inferred '
          f'length {infered_length}'
        )

    # split rng state
    split_rng_state, broadcast_rng_state = statelib.split_state(
      rng_state, split_rngs, ...
    )

    broadcasts = ScanBroadcasts(
      flatdef,
      flat_carry,
      graphdef,
      filters,
      f,
      # options
      carry_argnum,
      state_axes,
      split_rngs,
      transform_metadata,
      scan_output,
    )
    carry = (
      split_rng_state,
      broadcast_rng_state,
      carry_state,
      carry_arg,
      broadcasts,
    )
    scan = (flat_scan,)

    carry_out, scan_out = jax.lax.scan(
      scan_fn,
      carry,
      scan,
      length=infered_length,
      reverse=reverse,
      unroll=unroll,
      _split_transpose=_split_transpose,
    )
    (
      split_rng_state_out,
      broadcast_rng_state_out,
      carry_state_out,
      carry_arg_out,
      broadcasts,
    ) = carry_out
    graphdef_out, scan_args_out, scan_states_out = scan_out

    scan_args_out, scan_states_out = _transpose_tree(
      (scan_args_out, scan_states_out),
      (out_axes, list(state_axes.values())),
      move_front=False,
    )

    if carry_state_out:
      carry_state_out = State({0: carry_state_out._mapping})
    if split_rng_state_out:
      split_rng_state_out = State({0: split_rng_state_out._mapping})
    if broadcast_rng_state_out:
      broadcast_rng_state_out = State({0: broadcast_rng_state_out._mapping})

    _, output_graph_nodes = ctx.merge(
      graphdef_out,
      *scan_states_out,
      carry_state_out,
      split_rng_state_out,
      broadcast_rng_state_out,
    )

    carry_arg_out, scan_args_out = extract.insert_graph_nodes(
      (carry_arg_out, scan_args_out), output_graph_nodes
    )

    rnglib.restore_rngs(input_rng_streams)

    if scan_output:
      scan_args_out = tp.cast(B, scan_args_out)
      return carry_arg_out, scan_args_out
    else:
      return carry_arg_out

  return scan_apply_wrapper  # type: ignore


class Scan(tp.Generic[M], LiftedModule[M]):
  @staticmethod
  def constructor(
    module_constructor: tp.Callable[..., MA],
    *,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
    # extended api
    in_axes: int | None | tp.Sequence[tp.Any] = 0,
    in_axes_kwargs: tp.Any = 0,
    out_axes: tp.Any = 0,
    carry_argnum: int = 1,
    # nnx specific
    state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
    split_rngs: filterlib.Filter = ...,
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
    scan_output: bool = True,
  ) -> tp.Callable[..., Scan[MA]]:
    def _create_scan(*args, **kwargs):
      return Scan(
        module_constructor=module_constructor,
        module_init_args=args,
        module_init_kwargs=kwargs,
        # base api
        length=length,
        reverse=reverse,
        unroll=unroll,
        _split_transpose=_split_transpose,
        # extended api
        in_axes=in_axes,
        in_axes_kwargs=in_axes_kwargs,
        out_axes=out_axes,
        carry_argnum=carry_argnum,
        # nnx specific
        state_axes=state_axes,
        split_rngs=split_rngs,
        transform_metadata=transform_metadata,
        scan_output=scan_output,
      )

    return _create_scan

  def __init__(
    self,
    module_constructor: tp.Callable[..., M],
    *,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
    # extended api
    in_axes: int | None | tp.Sequence[tp.Any] = 0,
    in_axes_kwargs: tp.Any = 0,
    out_axes: tp.Any = 0,
    carry_argnum: int = 1,
    # nnx specific
    state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
    split_rngs: filterlib.Filter = ...,
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
    scan_output: bool = True,
    # submodule args
    module_init_args: tuple[tp.Any, ...],
    module_init_kwargs: dict[str, tp.Any],
  ):
    self.module_constructor = module_constructor
    # use Vmap to handle initialisation
    vmapped_module = Vmap.constructor(
      module_constructor,
      in_axes=in_axes,
      out_axes=None,
      axis_name=None,
      axis_size=length,
      spmd_axis_name=None,
      state_axes=state_axes,
      split_rngs=split_rngs,
      in_axes_kwargs=in_axes_kwargs,
      transform_metadata=transform_metadata,
    )(*module_init_args, **module_init_kwargs)
    self.scan_module = vmapped_module.vmap_module

    @functools.partial(
      scan,
      length=length,
      reverse=reverse,
      unroll=unroll,
      _split_transpose=_split_transpose,
      in_axes=in_axes,
      in_axes_kwargs=in_axes_kwargs,
      out_axes=out_axes,
      carry_argnum=carry_argnum,
      state_axes=state_axes,
      split_rngs=split_rngs,
      transform_metadata=transform_metadata,
      scan_output=scan_output,
    )
    def scan_call(module, *args, _nnx_scan_accessor: DelayedAccessor, **kwargs):
      method = _nnx_scan_accessor(module)
      return method(*args, **kwargs)

    self.scan_call = scan_call

  @property
  def _submodule(self) -> M:
    return self.scan_module

  def _call(
    self, accessor: DelayedAccessor, *args, **kwargs
  ) -> tuple[tp.Any, tp.Any]:
    return self.scan_call(
      self._submodule, *args, _nnx_scan_accessor=accessor, **kwargs
    )  # type: ignore[call-arg, return-value, type-var]


# -------------------------------
# remat
# -------------------------------


class Remat(tp.Generic[M], LiftedModule[M]):
  @staticmethod
  def constructor(
    module_constructor: tp.Callable[..., MA],
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: tp.Callable[..., bool] | None = None,
  ) -> tp.Callable[..., Remat[MA]]:
    def create_remat(*args, **kwargs):
      return Remat(
        module_constructor=module_constructor,
        module_init_args=args,
        module_init_kwargs=kwargs,
        prevent_cse=prevent_cse,
        static_argnums=static_argnums,
        policy=policy,
      )

    return create_remat

  def __init__(
    self,
    *,
    module_constructor: tp.Callable[..., M],
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: tp.Callable[..., bool] | None = None,
    # submodule args
    module_init_args: tuple[tp.Any, ...],
    module_init_kwargs: dict[str, tp.Any],
  ):
    self.module_constructor = module_constructor
    self.remat_module = self.module_constructor(
      *module_init_args, **module_init_kwargs
    )

    @nnx.remat(
      prevent_cse=prevent_cse, static_argnums=static_argnums, policy=policy
    )
    def remat_call(module, *args):
      accessor: DelayedAccessor
      *args, accessor = args
      method = accessor(module)
      return method(*args)

    self.rem_call = remat_call

  @property
  def _submodule(self) -> M:
    return self.remat_module

  def _call(self, accessor: DelayedAccessor, *args) -> tp.Any:
    return self.rem_call(self._submodule, *args, accessor)


# -------------------------------
# grad
# -------------------------------


def grad_fn(*args):
  f: tp.Callable[..., tp.Any]
  graphdef: GraphDef[tuple[dict[int, tp.Any], tuple[tp.Any, ...]]]
  non_diff_state: State
  has_aux: bool
  diff_args: list[int]
  ctx = graph.current_update_context('grad')
  *args, f, graphdef, non_diff_state, has_aux, diff_args = args

  # rebuild diff_state from substates in args
  diff_state = State({})
  for i in diff_args:
    diff_state[i] = args[i]
  diff_state: graph.GraphState = State({0: diff_state.raw_mapping})

  diff_graph_nodes, input_nodes = ctx.merge(
    graphdef, diff_state, non_diff_state
  )

  # add nodes to the args
  for i, arg in diff_graph_nodes.items():
    args[i] = arg

  # add other nodes to the args
  args = extract.insert_graph_nodes(args, input_nodes)

  out = f(*args)

  out, out_nodes = extract.extract_graph_nodes(out)

  graphdef_out, state_out = ctx.split((input_nodes, out_nodes))

  if has_aux:
    loss, aux = out
    out = (loss, (graphdef_out, state_out, aux))
  else:
    out = (out, (graphdef_out, state_out))

  return out


def _grad_general(
  f: tp.Callable[..., tp.Any],
  argnums: int | tp.Sequence[int],
  has_aux: bool,
  holomorphic: bool,
  allow_int: bool,
  reduce_axes: tp.Sequence[AxisName],
  wrt: filterlib.Filter,
  return_value: bool,
) -> tp.Callable[..., tp.Any]:
  @graph.update_context('grad')
  def grad_wrapper(*args):
    ctx: graph.UpdateContext = graph.current_update_context('grad')
    _argnums = _normalize_sequence(argnums)
    diff_graph_nodes: dict[int, tp.Any] = {
      i: arg
      for i, arg in enumerate(args)
      if i in _argnums and graph.is_node(arg)
    }
    args, input_nodes = extract.extract_graph_nodes(args)
    args = list(args)

    def only_diff(path: tuple, value: tp.Any) -> bool:
      # diff_graph_nodes is the first element in the tuple
      return path[0] == 0

    graphdef, diff_state, non_diff_state = ctx.split(
      (diff_graph_nodes, input_nodes), filterlib.All(wrt, only_diff), ...
    )  # type: ignore[misc]

    # extract diff_state substates into the args
    diff_args: list[int] = []
    if 0 in diff_state:
      for i, diff_substate in diff_state[0].items():  # type: ignore
        assert isinstance(i, int)
        args[i] = diff_substate
        diff_args.append(i)
    transform = jax.value_and_grad if return_value else jax.grad

    _argnums = _argnums[0] if len(_argnums) == 1 else _argnums

    out = transform(
      grad_fn,
      argnums=_argnums,
      has_aux=True,
      holomorphic=holomorphic,
      allow_int=allow_int,
      reduce_axes=reduce_axes,
    )(*args, f, graphdef, non_diff_state, has_aux, diff_args)

    if return_value:
      if has_aux:
        (loss, (graphdef_out, state_out, aux)), grads = out
        out = (loss, aux), grads
      else:
        (loss, (graphdef_out, state_out)), grads = out
        out = loss, grads
    else:
      if has_aux:
        grads, (graphdef_out, state_out, aux) = out
        out = grads, aux
      else:
        out, (graphdef_out, state_out) = out

    input_nodes, out_nodes = ctx.merge(graphdef_out, state_out)

    out = extract.insert_graph_nodes(out, out_nodes)
    return out

  return grad_wrapper


def grad(
  f: tp.Callable[..., tp.Any],
  argnums: int | tp.Sequence[int] = 0,
  has_aux: bool = False,
  holomorphic: bool = False,
  allow_int: bool = False,
  reduce_axes: tp.Sequence[AxisName] = (),
  *,
  wrt: filterlib.Filter = variablelib.Param,
) -> tp.Callable[..., tp.Any]:
  """Lifted version of ``jax.grad`` that can handle Modules / graph nodes as
  arguments.

  The differentiable state of each graph node is defined by the `wrt` filter,
  which by default is set to `nnx.Param`. Internally the ``State`` of
  graph nodes is extracted, filtered according to `wrt` filter, and
  passed to the underlying ``jax.grad`` function. The gradients
  of graph nodes are of type ``State``.

  Example::

    >>> from flax import nnx
    >>> import jax.numpy as jnp
    ...
    >>> m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> x = jnp.ones((1, 2))
    >>> y = jnp.ones((1, 3))
    ...
    >>> loss_fn = lambda m, x, y: jnp.mean((m(x) - y) ** 2)
    >>> grad_fn = nnx.transforms.deprecated.grad(loss_fn, wrt=nnx.Param)
    ...
    >>> grads = grad_fn(m, x, y)
    >>> jax.tree.map(jnp.shape, grads)
    State({
      'bias': VariableState(
        type=Param,
        value=(3,)
      ),
      'kernel': VariableState(
        type=Param,
        value=(2, 3)
      )
    })

  Args:
    fun: Function to be differentiated. Its arguments at positions specified by
      ``argnums`` should be arrays, scalars, graph nodes or standard Python
      containers. Argument arrays in the positions specified by ``argnums`` must
      be of inexact (i.e., floating-point or complex) type. It should return a
      scalar (which includes arrays with shape ``()`` but not arrays with shape
      ``(1,)`` etc.)
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default 0).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. If True, inputs and outputs must be complex. Default False.
    allow_int: Optional, bool. Whether to allow differentiating with
      respect to integer valued inputs. The gradient of an integer input will
      have a trivial vector-space dtype (float0). Default False.
    reduce_axes: Optional, tuple of axis names. If an axis is listed here, and
      ``fun`` implicitly broadcasts a value over that axis, the backward pass
      will perform a ``psum`` of the corresponding gradient. Otherwise, the
      gradient will be per-example over named axes. For example, if ``'batch'``
      is a named batch axis, ``grad(f, reduce_axes=('batch',))`` will create a
      function that computes the total gradient while ``grad(f)`` will create
      one that computes the per-example gradient.
    wrt: Optional, filterlib.Filter. Filter to extract the differentiable state
      of each graph node. Default is `nnx.Param`.

  """

  return _grad_general(
    f,
    argnums,
    has_aux,
    holomorphic,
    allow_int,
    reduce_axes,
    wrt,
    return_value=False,
  )


def value_and_grad(
  f: tp.Callable[..., tp.Any],
  argnums: int | tp.Sequence[int] = 0,
  has_aux: bool = False,
  holomorphic: bool = False,
  allow_int: bool = False,
  reduce_axes: tp.Sequence[AxisName] = (),
  *,
  wrt: filterlib.Filter = variablelib.Param,
) -> tp.Callable[..., tp.Any]:
  return _grad_general(
    f,
    argnums,
    has_aux,
    holomorphic,
    allow_int,
    reduce_axes,
    wrt,
    return_value=True,
  )


class Grad(tp.Generic[M], LiftedModule[M]):
  @staticmethod
  def constructor(
    module_constructor: tp.Callable[..., MA],
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
    return_value: bool = False,
    *,
    wrt: filterlib.Filter = variablelib.Param,
  ) -> tp.Callable[..., Grad[MA]]:
    def _create_grad(*args, **kwargs):
      return Grad(
        module_constructor=module_constructor,
        wrt=wrt,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        reduce_axes=reduce_axes,
        return_value=return_value,
        # submodule args
        module_init_args=args,
        module_init_kwargs=kwargs,
      )

    return _create_grad

  def __init__(
    self,
    module_constructor: tp.Callable[..., M],
    argnums: int | tp.Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
    *,
    wrt: filterlib.Filter = variablelib.Param,
    # submodule args
    module_init_args: tuple[tp.Any, ...],
    module_init_kwargs: dict[str, tp.Any],
  ):
    self.module_constructor = module_constructor
    self.grad_module = self.module_constructor(
      *module_init_args, **module_init_kwargs
    )

    @functools.partial(
      grad,
      argnums=argnums,
      has_aux=has_aux,
      holomorphic=holomorphic,
      allow_int=allow_int,
      reduce_axes=reduce_axes,
      wrt=wrt,
    )
    def grad_call_apply(module, *args):
      *args, accessor = args
      method = accessor(module)
      return method(*args)

    self.grad_apply = grad_call_apply

  @property
  def _submodule(self) -> M:
    return self.grad_module

  def _call(self, accessor: DelayedAccessor, *args) -> tp.Any:
    return self.grad_apply(self.grad_module, *args, accessor)


# -------------------------------
# jit
# -------------------------------


class Jit(tp.Generic[M], LiftedModule[M]):
  @staticmethod
  def constructor(
    module_constructor: tp.Callable[..., MA],
    *,
    in_shardings: tp.Any = None,
    out_shardings: tp.Any = None,
    static_argnums: int | tp.Sequence[int] | None = None,
    static_argnames: str | tp.Iterable[str] | None = None,
    donate_argnums: int | tp.Sequence[int] | None = None,
    donate_argnames: str | tp.Iterable[str] | None = None,
    keep_unused: bool = False,
    device: tp.Optional[jax.Device] = None,
    backend: tp.Optional[str] = None,
    inline: bool = False,
    abstracted_axes: tp.Optional[tp.Any] = None,
  ) -> tp.Callable[..., Jit[MA]]:
    def _create_jit(*args, **kwargs):
      return Jit(
        module_constructor=module_constructor,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        donate_argnames=donate_argnames,
        keep_unused=keep_unused,
        device=device,
        backend=backend,
        inline=inline,
        abstracted_axes=abstracted_axes,
        # submodule args
        module_init_args=args,
        module_init_kwargs=kwargs,
      )

    return _create_jit

  def __init__(
    self,
    module_constructor: tp.Callable[..., M],
    *,
    in_shardings: tp.Any = None,
    out_shardings: tp.Any = None,
    static_argnums: int | tp.Sequence[int] | None = None,
    static_argnames: str | tp.Iterable[str] | None = None,
    donate_argnums: int | tp.Sequence[int] | None = None,
    donate_argnames: str | tp.Iterable[str] | None = None,
    keep_unused: bool = False,
    device: tp.Optional[jax.Device] = None,
    backend: tp.Optional[str] = None,
    inline: bool = False,
    abstracted_axes: tp.Optional[tp.Any] = None,
    # submodule args
    module_init_args: tuple[tp.Any, ...],
    module_init_kwargs: dict[str, tp.Any],
  ):
    @functools.partial(
      nnx.jit,
      in_shardings=in_shardings,
      out_shardings=out_shardings,
      static_argnums=static_argnums,
      static_argnames=static_argnames,
      donate_argnums=donate_argnums,
      donate_argnames=donate_argnames,
      keep_unused=keep_unused,
      device=device,
      backend=backend,
      inline=inline,
      abstracted_axes=abstracted_axes,
    )
    def jit_call_module(
      module, *args, _nnx_jit_accessor: DelayedAccessor, **kwargs
    ):
      method = _nnx_jit_accessor(module)
      return method(*args, **kwargs)

    self.jitted_fn = jit_call_module
    self.module_constructor = module_constructor
    self.jit_module = self.module_constructor(
      *module_init_args, **module_init_kwargs
    )

  @property
  def _submodule(self) -> M:
    return self.jit_module

  def _call(self, accessor: DelayedAccessor, *args, **kwargs) -> tp.Any:
    out = self.jitted_fn(
      self.jit_module, *args, _nnx_jit_accessor=accessor, **kwargs
    )  # type: ignore[call-arg, type-var]
    return out
