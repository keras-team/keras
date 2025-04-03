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

from collections import defaultdict
from typing import Any, TypeVar
import typing as tp

import jax
from flax import struct
from flax.core import meta
from flax.nnx import graph
from flax.nnx import spmd
from flax.nnx import traversals
from flax.nnx import variablelib
from flax.typing import LogicalNames


A = TypeVar('A')
B = TypeVar('B')


def sort_variable_types(types: tp.Iterable[type]):
  def _variable_parents_count(t: type):
    return sum(1 for p in t.mro() if issubclass(p, variablelib.Variable))
  parent_count = {t: _variable_parents_count(t) for t in types}
  return sorted(types, key=lambda t: -parent_count[t])


#############################################
### NNX Variable <-> Linen metadata boxes ###
#############################################


class NNXMeta(struct.PyTreeNode, meta.AxisMetadata[A]):
  """Default Flax metadata class for `nnx.VariableState`."""

  var_type: type[variablelib.Variable[tp.Any]] = struct.field(pytree_node=False)
  value: Any = struct.field(pytree_node=True)
  metadata: dict[str, tp.Any] = struct.field(pytree_node=False)

  def unbox(self) -> A:
    return self.value

  def replace_boxed(self, val: B) -> 'NNXMeta[B]':
    return self.replace(value=val)  # type: ignore

  def add_axis(self, index: int, params: dict[Any, Any]) -> 'NNXMeta[A]':
    # TODO: implement this, supporting hooks
    return self

  def remove_axis(self, index: int, params: dict[Any, Any]) -> 'NNXMeta[A]':
    # TODO: implement this, supporting hooks
    return self

  def get_partition_spec(self) -> jax.sharding.PartitionSpec:
    """Returns the ``Partitionspec`` for this partitioned value."""
    nnx_var = self.to_nnx_variable().to_state()
    return spmd.get_partition_spec(nnx_var).value

  def to_nnx_variable(self) -> variablelib.Variable:
    return self.var_type(self.value, **self.metadata)


def is_vanilla_variable(vs: variablelib.VariableState) -> bool:
  """A variables state is vanilla if its metadata is essentially blank.

  Returns False only if it has non-empty hooks or any non-built-in attribute.
  """
  for key, value in vs.get_metadata().items():
    if key.endswith('_hooks'):
      if value != ():
        return False
    else:
      return False
  return True


def to_linen_var(vs: variablelib.VariableState) -> meta.AxisMetadata:
  metadata = vs.get_metadata()
  if 'linen_meta_type' in metadata:
    linen_type = metadata['linen_meta_type']
    if hasattr(linen_type, 'from_nnx_metadata'):
      return linen_type.from_nnx_metadata({'value': vs.value, **metadata})
    return linen_type(vs.value, **metadata)
  if is_vanilla_variable(vs):
    return vs.value
  return NNXMeta(vs.type, vs.value, metadata)


def get_col_name(keypath: tp.Sequence[Any]) -> str:
  """Given the keypath of a Flax variable type, return its Linen collection name."""
  # Infer variable type from the leaf's path, which contains its Linen collection name
  assert isinstance(keypath[0], jax.tree_util.DictKey)
  return str(keypath[0].key)


def to_nnx_var(col: str, x: meta.AxisMetadata | Any) -> variablelib.Variable:
  """Convert a Linen variable to an NNX variable."""
  vtype = variablelib.variable_type_from_name(col, allow_register=True)
  if isinstance(x, NNXMeta):
    assert vtype == x.var_type, f'Type stored in NNXMeta {x.var_type} != type inferred from collection name {vtype}'
    return x.to_nnx_variable()
  if isinstance(x, meta.AxisMetadata):
    x_metadata = vars(x)
    if hasattr(x, 'to_nnx_metadata'):
      x_metadata = x.to_nnx_metadata()
    assert hasattr(x, 'value')
    return vtype(**x_metadata, linen_meta_type=type(x))
  return vtype(x)


def _recursive_merge(dict1, dict2):
  """Recursively merge two dicts."""
  flat_map = traversals.flatten_mapping(dict1)
  flat_map |= traversals.flatten_mapping(dict2)
  return traversals.unflatten_mapping(flat_map)


def linen_vars_to_nnx_attrs(variables: tp.Mapping[str, Any]) -> dict[str, Any]:
  """Convert a dict of Linen-style variables to NNX variables."""
  nnx_vars = jax.tree_util.tree_map_with_path(
    lambda kp, x: to_nnx_var(get_col_name(kp), x),
    variables, is_leaf=lambda x: isinstance(x, meta.AxisMetadata))
  nnx_attrs: dict[str, Any] = defaultdict(dict)
  for _, col_tree in nnx_vars.items():
    assert isinstance(col_tree, dict)
    for attr_name, value in col_tree.items():
      assert isinstance(attr_name, str)
      if isinstance(value, tp.Mapping):  # it's a sublayer
        nnx_attrs[attr_name] = _recursive_merge(nnx_attrs[attr_name], value)
      else:
        nnx_attrs[attr_name] = value     # it's a variable on this layer
  return dict(nnx_attrs)


def nnx_attrs_to_linen_vars(nnx_attrs: dict) -> dict:
  """Convert a dict of NNX variables (or variable states) to Linen-style variables."""
  linen_structured = {}
  for kp, v in traversals.flatten_mapping(nnx_attrs).items():
    if isinstance(v, variablelib.Variable):
      col_name = variablelib.variable_name_from_type(type(v))
      v = to_linen_var(v.to_state())
    elif isinstance(v, variablelib.VariableState):
      col_name = variablelib.variable_name_from_type(v.type)
      v = to_linen_var(v)
    elif isinstance(v, graph.NodeDef) or isinstance(v, graph.NodeRef):
      col_name = 'nnx'  # an nnx.GraphDef for some ToLinen submodule
    else:
      raise ValueError(f'Cannot infer collection name from value: {v}')
    linen_structured[(col_name, *kp)] = v
  variables = traversals.unflatten_mapping(linen_structured)
  return variables



def with_partitioning(
    fn: tp.Callable[..., tp.Any],
    names: LogicalNames,
    mesh: jax.sharding.Mesh | None = None,
) -> tp.Callable[..., meta.Partitioned[tp.Any]]:
  """Same interface as Linen, but calls NNX `with_partitioning` within."""
  return spmd.with_partitioning(fn, names, mesh,
                                linen_meta_type=meta.Partitioned)