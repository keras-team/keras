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

import abc
import typing as tp

import jax
# from jax._src.tree_util import broadcast_prefix

from flax import struct
from flax.nnx.object import Object
from flax.typing import Missing, PathParts
from flax.nnx import graph, variablelib


A = tp.TypeVar('A')
Index = int
KeyEntry = tp.TypeVar('KeyEntry', bound=tp.Hashable)
KeyPath = tuple[KeyEntry, ...]
Prefix = tp.Any
Leaf = tp.Any


class ExtractionIndex(struct.PyTreeNode):
  """Index of a graph node in a Pytree structure."""

  index: Index = struct.field(pytree_node=False)


@tp.overload
def extract_graph_nodes(
  pytree: A,
  /,
  *,
  validate_fn: tp.Callable[[KeyPath, Prefix, Leaf], None] | None = None,
) -> tuple[A, tuple[tp.Any, ...]]: ...
@tp.overload
def extract_graph_nodes(
  pytree: A,
  /,
  *,
  prefix: tp.Any,
  validate_fn: tp.Callable[[KeyPath, Prefix, Leaf], None] | None = None,
) -> tuple[A, tuple[tp.Any, ...], tuple[tp.Any, ...]]: ...
def extract_graph_nodes(
  pytree: A,
  /,
  *,
  prefix: tp.Any = Missing,
  validate_fn: tp.Callable[[KeyPath, Prefix, Leaf], None] | None = None,
) -> (
  tuple[A, tuple[tp.Any, ...]]
  | tuple[A, tuple[tp.Any, ...], tuple[tp.Any, ...]]
):
  """Extracts all graph nodes from a pytree."""
  nodes: dict[tp.Any, Index] = {}
  node_prefixes = []
  leaves = []

  prefix_leaves = broadcast_prefix(
    prefix,
    pytree,
    prefix_is_leaf=lambda x: x is None,
  )
  key_leaves, treedef = jax.tree_util.tree_flatten_with_path(pytree)

  assert len(key_leaves) == len(prefix_leaves)

  for (keypath, leaf), prefix_leaf in zip(key_leaves, prefix_leaves):
    if validate_fn:
      validate_fn(keypath, prefix_leaf, leaf)
    if graph.is_graph_node(leaf):
      if leaf not in nodes:
        index = nodes[leaf] = len(nodes)
        node_prefixes.append(prefix_leaf)
      else:
        index = nodes[leaf]
        # check consistent aliasing
        if prefix_leaf != node_prefixes[index]:
          path_str = jax.tree_util.keystr(keypath)
          raise ValueError(
            f'Inconsistent aliasing detected. Node {type(leaf)} at path {path_str} '
            f'has different prefixes: {prefix_leaf} and {node_prefixes[index]}.'
          )
      leaves.append(ExtractionIndex(index))
    else:
      leaves.append(leaf)

  pytree_out = jax.tree.unflatten(treedef, leaves)

  if prefix is Missing:
    return pytree_out, tuple(nodes)  # type: ignore[bad-return-type]
  else:
    return pytree_out, tuple(nodes), tuple(node_prefixes)  # type: ignore[bad-return-type]


def insert_graph_nodes(pytree: A, nodes: tuple[tp.Any, ...], /) -> A:
  """Inserts graph nodes into a pytree."""

  def _maybe_insert(x):
    if isinstance(x, ExtractionIndex):
      return nodes[x.index]
    return x

  return jax.tree.map(
    _maybe_insert, pytree, is_leaf=lambda x: isinstance(x, ExtractionIndex)
  )

class PrefixMapping(abc.ABC):
  @abc.abstractmethod
  def map_prefix(
    self,
    path: variablelib.PathParts,
    variable: variablelib.Variable,
    /,
  ) -> tp.Any: ...

def check_consistent_aliasing(
  node: tuple[tp.Any, ...],
  prefix: tuple[tp.Any, ...],
  /,
  *,
  node_prefixes: dict[tp.Any, list[tuple[PathParts, tp.Any]]] | None = None,
):
  if node_prefixes is None:
    node_prefixes = {}

  # collect all paths and prefixes for each node
  for path, value in graph.iter_graph(node):
    if graph.is_graph_node(value) or isinstance(value, graph.Variable):
      if isinstance(value, Object):
        value._check_valid_context(
          lambda: f'Trying to extract graph node from different trace level, got {value!r}'
        )
      if isinstance(value, graph.Variable):
        if not value._trace_state.is_valid():
          raise ValueError(
            f'Cannot extract graph node from different trace level, got {value!r}'
          )
        if isinstance(prefix, PrefixMapping):
          variable_prefix = prefix.map_prefix(path, value)
        else:
          variable_prefix = prefix

        if value in node_prefixes:
          paths_prefixes = node_prefixes[value]
          paths_prefixes.append((path, variable_prefix))
        else:
          node_prefixes[value] = [(path, variable_prefix)]

  # check for inconsistent aliasing
  node_msgs = []
  for node, paths_prefixes in node_prefixes.items():
    unique_prefixes = {prefix for _, prefix in paths_prefixes}
    if len(unique_prefixes) > 1:
      path_prefix_repr = '\n'.join(
        f'  {"/".join(map(str,path)) if path else "<root>"}: {prefix}'
        for path, prefix in paths_prefixes
      )
      nodes_msg = f'Node: {type(node)}\n{path_prefix_repr}'
      node_msgs.append(nodes_msg)

  if node_msgs:
    raise ValueError(
      'Inconsistent aliasing detected. The following nodes have different prefixes:\n'
      + '\n'.join(node_msgs)
    )

# -----------------------------
# to_tree/from_tree
# -----------------------------

def broadcast_prefix(
  prefix_tree: tp.Any,
  full_tree: tp.Any,
  prefix_is_leaf: tp.Callable[[tp.Any], bool] | None = None,
  tree_is_leaf: tp.Callable[[tp.Any], bool] | None = None,
) -> list[tp.Any]:
  # If prefix_tree is not a tree prefix of full_tree, this code can raise a
  # ValueError; use prefix_errors to find disagreements and raise more precise
  # error messages.
  result = []
  num_leaves = lambda t: jax.tree_util.tree_structure(
    t, is_leaf=tree_is_leaf
  ).num_leaves
  add_leaves = lambda x, subtree: result.extend([x] * num_leaves(subtree))
  jax.tree.map(add_leaves, prefix_tree, full_tree, is_leaf=prefix_is_leaf)
  return result


class GraphDefState(struct.PyTreeNode):
  graphdef: graph.GraphDef[tp.Any] = struct.field(pytree_node=False)
  state: graph.GraphState = struct.field(pytree_node=True)

S = tp.TypeVar(
  'S', bound=graph.GraphState | graph.GraphFlatState | list[tp.Any]
)

class NodeStates(struct.PyTreeNode, tp.Generic[S]):
  _graphdef: graph.GraphDef[tp.Any] | None
  states: tuple[S, ...]
  metadata: tp.Any = struct.field(pytree_node=False)

  @property
  def graphdef(self) -> graph.GraphDef[tp.Any]:
    if self._graphdef is None:
      raise ValueError('No graphdef available')
    return self._graphdef

  @property
  def state(self) -> S:
    if len(self.states) != 1:
      raise ValueError(
        f'Expected exactly one GraphDefState, got {len(self.states)}'
      )
    return self.states[0]

  @classmethod
  def from_split(
    cls,
    graphdef: graph.GraphDef[tp.Any],
    state: S,
    /,
    *states: S,
    metadata: tp.Any = None,
  ):
    return cls(_graphdef=graphdef, states=(state, *states), metadata=metadata)

  @classmethod
  def from_states(
    cls,
    state: S,
    *states: S,
  ):
    return cls(_graphdef=None, states=(state, *states), metadata=None)

  @classmethod
  def from_prefixes(
    cls,
    prefixes: tp.Iterable[tp.Any],
    /,
    *,
    metadata: tp.Any = None,
  ):
    return cls(_graphdef=None, states=tuple(prefixes), metadata=metadata)


def default_split_fn(
  ctx: graph.SplitContext, path: KeyPath, prefix: Prefix, leaf: Leaf
) -> tp.Any:
  return NodeStates.from_split(*ctx.split(leaf))


def to_tree(
  tree,
  /,
  *,
  prefix: tp.Any = Missing,
  split_fn: tp.Callable[
    [graph.SplitContext, KeyPath, Prefix, Leaf], tp.Any
  ] = default_split_fn,
  map_non_graph_nodes: bool = False,
  ctxtag: tp.Hashable | None = None,
  check_aliasing: bool = True,
) -> tp.Any:
  if prefix is Missing or prefix is None:
    # fast path, no need for prefix broadcasting or consistent aliasing checks
    with graph.split_context(ctxtag) as split_ctx:
      return jax.tree.map(
        lambda x: split_fn(split_ctx, (), prefix, x)
        if map_non_graph_nodes or graph.is_graph_node(x)
        else x,
        tree,
      )
  leaf_prefixes = broadcast_prefix(
    prefix,
    tree,
    prefix_is_leaf=lambda x: x is None,
  )
  leaf_keys, treedef = jax.tree_util.tree_flatten_with_path(tree)

  assert len(leaf_keys) == len(leaf_prefixes)
  leaves_out = []
  node_prefixes: dict[tp.Any, list[tuple[PathParts, tp.Any]]] = {}

  with graph.split_context(ctxtag) as split_ctx:
    for (keypath, leaf), leaf_prefix in zip(leaf_keys, leaf_prefixes):
      if graph.is_graph_node(leaf):
        if check_aliasing:
          check_consistent_aliasing(
            leaf, leaf_prefix, node_prefixes=node_prefixes
          )
        tree_node = split_fn(split_ctx, keypath, leaf_prefix, leaf)
        leaves_out.append(tree_node)
      else:
        if map_non_graph_nodes:
          leaf = split_fn(split_ctx, keypath, leaf_prefix, leaf)
        leaves_out.append(leaf)

  pytree_out = jax.tree.unflatten(treedef, leaves_out)
  return pytree_out


def merge_tree_node(
  ctx: graph.MergeContext, path: KeyPath, prefix: Prefix, leaf: Leaf
) -> tp.Any:
  if not isinstance(leaf, NodeStates):
    raise ValueError(f'Expected TreeNode, got {type(leaf)} at path {path}')
  return ctx.merge(leaf.graphdef, *leaf.states)


def is_tree_node(x):
  return isinstance(x, NodeStates)


def from_tree(
  tree: tp.Any,
  /,
  *,
  prefix: tp.Any = Missing,
  merge_fn: tp.Callable[
    [graph.MergeContext, KeyPath, Prefix, Leaf], tp.Any
  ] = merge_tree_node,
  is_node_leaf: tp.Callable[[Leaf], bool] = is_tree_node,
  is_leaf: tp.Callable[[Leaf], bool] = is_tree_node,
  map_non_graph_nodes: bool = False,
  is_inner: bool | None = None,
  ctxtag: tp.Hashable | None = None,
) -> tp.Any:
  if prefix is Missing or prefix is None:
    # fast path, no need for prefix broadcasting or consistent aliasing checks
    with graph.merge_context(is_inner, ctxtag) as merge_ctx:
      return jax.tree.map(
        lambda x: merge_fn(merge_ctx, (), prefix, x)
        if map_non_graph_nodes or is_node_leaf(x)
        else x,
        tree,
        is_leaf=is_leaf,
      )
  leaf_prefixes = broadcast_prefix(
    prefix,
    tree,
    prefix_is_leaf=lambda x: x is None or is_leaf(x),
    tree_is_leaf=is_leaf,
  )
  leaf_keys, treedef = jax.tree_util.tree_flatten_with_path(
    tree, is_leaf=is_leaf
  )
  assert len(leaf_keys) == len(leaf_prefixes)
  leaves_out = []

  with graph.merge_context(is_inner, ctxtag) as merge_ctx:
    for (keypath, leaf), leaf_prefix in zip(leaf_keys, leaf_prefixes):
      if map_non_graph_nodes or is_node_leaf(leaf):
        leaf = merge_fn(merge_ctx, keypath, leaf_prefix, leaf)
      leaves_out.append(leaf)

  pytree_out = jax.tree.unflatten(treedef, leaves_out)
  return pytree_out

def clear_non_graph_nodes(tree):
  return jax.tree.map(lambda x: x if graph.is_graph_node(x) else None, tree)