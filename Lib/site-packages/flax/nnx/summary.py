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
import dataclasses
import functools
import inspect
import io
import typing as tp
from types import MappingProxyType

import jax
import numpy as np
import rich.console
import rich.table
import rich.text
import yaml
import jax.numpy as jnp

from flax import nnx
from flax import typing
from flax.nnx import graph, statelib, variablelib

try:
  from IPython import get_ipython

  in_ipython = get_ipython() is not None
except ImportError:
  in_ipython = False

class SizeBytes(typing.SizeBytes):
  def __repr__(self) -> str:
    bytes_repr = _bytes_repr(self.bytes)
    return f'{self.size:,} [dim]({bytes_repr})[/dim]'

class ObjectInfo(tp.NamedTuple):
  path: statelib.PathParts
  stats: dict[type[variablelib.Variable], SizeBytes]
  variable_groups: defaultdict[
    type[variablelib.Variable], defaultdict[typing.Key, variablelib.Variable]
  ]

NodeStats = dict[int, ObjectInfo | None]

def _collect_stats(
  path: statelib.PathParts,
  node: tp.Any,
  node_stats: NodeStats,
  object_types: set[type],
):
  if not graph.is_node(node) and not isinstance(node, variablelib.Variable):
    raise ValueError(f'Expected a graph node or Variable, got {type(node)!r}.')

  if id(node) in node_stats:
    return

  stats: dict[type[variablelib.Variable], SizeBytes] = {}
  variable_groups: defaultdict[
    type[variablelib.Variable], defaultdict[typing.Key, variablelib.Variable]
  ] = defaultdict(lambda: defaultdict())
  node_stats[id(node)] = ObjectInfo(path, stats, variable_groups)

  if isinstance(node, nnx.Object):
    node._nnx_tabulate_id = id(node)  # type: ignore
    object_types.add(type(node))

  node_impl = graph.get_node_impl(node)
  assert node_impl is not None
  node_dict = node_impl.node_dict(node)
  for key, value in node_dict.items():
    if id(value) in node_stats:
      continue
    elif isinstance(value, variablelib.Variable):
      var_type = type(value)
      if issubclass(var_type, nnx.RngState):
        var_type = nnx.RngState
      size_bytes = SizeBytes.from_any(value.value)
      if var_type in stats:
        stats[var_type] += size_bytes
      else:
        stats[var_type] = size_bytes
      variable_groups[var_type][key] = value
      node_stats[id(value)] = None
    elif graph.is_node(value):
      _collect_stats((*path, key), value, node_stats, object_types)
      # accumulate stats from children
      child_info = node_stats[id(value)]
      assert child_info is not None
      for var_type, size_bytes in child_info.stats.items():
        if var_type in stats:
          stats[var_type] += size_bytes
        else:
          stats[var_type] = size_bytes

@dataclasses.dataclass(frozen=True, repr=False)
class ArrayRepr:
  shape: tuple[int, ...]
  dtype: tp.Any

  @classmethod
  def from_array(cls, x: jax.Array | np.ndarray):
    return cls(jnp.shape(x), jnp.result_type(x))

  def __str__(self):
    shape_repr = ','.join(str(x) for x in self.shape)
    return f'[dim]{self.dtype}[/dim][{shape_repr}]'


@dataclasses.dataclass
class CallInfo:
  object_id: int
  type: type
  path: statelib.PathParts
  input_args: tuple[tp.Any, ...]
  input_kwargs: dict[str, tp.Any]
  outputs: tp.Any

class SimpleObjectRepr:
  def __init__(self, obj: tp.Any):
    self.type = type(obj)

  def __str__(self):
    return f'{self.type.__name__}(...)'

  def __repr__(self):
    return f'{self.type.__name__}(...)'


def get_method_wrapper(
  rows: list[CallInfo],
  node_stats: NodeStats,
  method: tp.Callable,
) -> tp.Callable:
  method_name = method.__name__
  @functools.wraps(method)
  def method_wrapper(obj, *args, **kwargs):
    def _to_dummy_array(x):
      if isinstance(x, jax.Array | np.ndarray):
        return ArrayRepr.from_array(x)
      elif graph.is_graph_node(x):
        return SimpleObjectRepr(x)
      else:
        return x

    object_id: int = getattr(obj, '_nnx_tabulate_id')
    input_args_info, input_kwargs_info = jax.tree.map(
      _to_dummy_array, (args, kwargs)
    )
    node_info = node_stats[object_id]
    assert node_info is not None
    path = node_info.path
    if method_name != '__call__':
      path = (*path, method_name)
    call_info = CallInfo(
      object_id=object_id,
      type=type(obj),
      path=path,
      input_args=input_args_info,
      input_kwargs=input_kwargs_info,
      outputs=None,
    )
    rows.append(call_info)
    out = method(obj, *args, **kwargs)
    call_info.outputs = jax.tree.map(_to_dummy_array, out)
    return out

  return method_wrapper


def _call_obj(
  obj,
  input_args: tuple[tp.Any, ...],
  input_kwargs: dict[str, tp.Any],
  *,
  call_method: str,
  rows: list[CallInfo],
  node_stats: NodeStats,
  object_types: set[type],
):
  original_methods: dict[type, dict[str, tp.Callable]] = {}
  try:
    for obj_type in object_types:
      methods: dict[str, tp.Callable] = {}
      for name, top_method in inspect.getmembers(obj_type, inspect.isfunction):
        if not name.startswith('_') or name == '__call__':
          methods[name] = top_method
          method_wrapper = get_method_wrapper(rows, node_stats, top_method)
          setattr(obj_type, name, method_wrapper)

      original_methods[obj_type] = methods

    top_method = getattr(type(obj), call_method)
    top_method(obj, *input_args, **input_kwargs)
  finally:
    for obj_type, methods in original_methods.items():
      for name, top_method in methods.items():
        setattr(obj_type, name, top_method)


def filter_rng_streams(row: CallInfo):
  return not issubclass(row.type, nnx.RngStream)

def tabulate(
  obj,
  *input_args,
  depth: int | None = None,
  method: str = '__call__',
  row_filter: tp.Callable[[CallInfo], bool] = filter_rng_streams,
  table_kwargs: tp.Mapping[str, tp.Any] = MappingProxyType({}),
  column_kwargs: tp.Mapping[str, tp.Any] = MappingProxyType({}),
  console_kwargs: tp.Mapping[str, tp.Any] = MappingProxyType({}),
  **input_kwargs,
) -> str:
  """Creates a summary of the graph object represented as a table.

  The table summarizes the object's state and metadata. The table is
  structured as follows:

  - The first column represents the path of the object in the graph.
  - The second column represents the type of the object.
  - The third column represents the input arguments passed to the object's
    method.
  - The fourth column represents the output of the object's method.
  - The following columns provide information about the object's state,
    grouped by Variable types.

  Example::

    >>> from flax import nnx
    ...
    >>> class Block(nnx.Module):
    ...   def __init__(self, din, dout, rngs: nnx.Rngs):
    ...     self.linear = nnx.Linear(din, dout, rngs=rngs)
    ...     self.bn = nnx.BatchNorm(dout, rngs=rngs)
    ...     self.dropout = nnx.Dropout(0.2, rngs=rngs)
    ...
    ...   def __call__(self, x):
    ...     return nnx.relu(self.dropout(self.bn(self.linear(x))))
    ...
    >>> class Foo(nnx.Module):
    ...   def __init__(self, rngs: nnx.Rngs):
    ...     self.block1 = Block(32, 128, rngs=rngs)
    ...     self.block2 = Block(128, 10, rngs=rngs)
    ...
    ...   def __call__(self, x):
    ...     return self.block2(self.block1(x))
    ...
    >>> foo = Foo(nnx.Rngs(0))
    >>> # print(nnx.tabulate(foo, jnp.ones((1, 32))))

                                                          Foo Summary
    ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
    ┃ path           ┃ type      ┃ inputs         ┃ outputs        ┃ BatchStat          ┃ Param                   ┃ RngState ┃
    ┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
    │                │ Foo       │ float32[1,32]  │ float32[1,10]  │ 276 (1.1 KB)       │ 5,790 (23.2 KB)         │ 2 (12 B) │
    ├────────────────┼───────────┼────────────────┼────────────────┼────────────────────┼─────────────────────────┼──────────┤
    │ block1         │ Block     │ float32[1,32]  │ float32[1,128] │ 256 (1.0 KB)       │ 4,480 (17.9 KB)         │ 2 (12 B) │
    ├────────────────┼───────────┼────────────────┼────────────────┼────────────────────┼─────────────────────────┼──────────┤
    │ block1/linear  │ Linear    │ float32[1,32]  │ float32[1,128] │                    │ bias: float32[128]      │          │
    │                │           │                │                │                    │ kernel: float32[32,128] │          │
    │                │           │                │                │                    │                         │          │
    │                │           │                │                │                    │ 4,224 (16.9 KB)         │          │
    ├────────────────┼───────────┼────────────────┼────────────────┼────────────────────┼─────────────────────────┼──────────┤
    │ block1/bn      │ BatchNorm │ float32[1,128] │ float32[1,128] │ mean: float32[128] │ bias: float32[128]      │          │
    │                │           │                │                │ var: float32[128]  │ scale: float32[128]     │          │
    │                │           │                │                │                    │                         │          │
    │                │           │                │                │ 256 (1.0 KB)       │ 256 (1.0 KB)            │          │
    ├────────────────┼───────────┼────────────────┼────────────────┼────────────────────┼─────────────────────────┼──────────┤
    │ block1/dropout │ Dropout   │ float32[1,128] │ float32[1,128] │                    │                         │ 2 (12 B) │
    ├────────────────┼───────────┼────────────────┼────────────────┼────────────────────┼─────────────────────────┼──────────┤
    │ block2         │ Block     │ float32[1,128] │ float32[1,10]  │ 20 (80 B)          │ 1,310 (5.2 KB)          │          │
    ├────────────────┼───────────┼────────────────┼────────────────┼────────────────────┼─────────────────────────┼──────────┤
    │ block2/linear  │ Linear    │ float32[1,128] │ float32[1,10]  │                    │ bias: float32[10]       │          │
    │                │           │                │                │                    │ kernel: float32[128,10] │          │
    │                │           │                │                │                    │                         │          │
    │                │           │                │                │                    │ 1,290 (5.2 KB)          │          │
    ├────────────────┼───────────┼────────────────┼────────────────┼────────────────────┼─────────────────────────┼──────────┤
    │ block2/bn      │ BatchNorm │ float32[1,10]  │ float32[1,10]  │ mean: float32[10]  │ bias: float32[10]       │          │
    │                │           │                │                │ var: float32[10]   │ scale: float32[10]      │          │
    │                │           │                │                │                    │                         │          │
    │                │           │                │                │ 20 (80 B)          │ 20 (80 B)               │          │
    ├────────────────┼───────────┼────────────────┼────────────────┼────────────────────┼─────────────────────────┼──────────┤
    │ block2/dropout │ Dropout   │ float32[1,10]  │ float32[1,10]  │                    │                         │          │
    ├────────────────┼───────────┼────────────────┼────────────────┼────────────────────┼─────────────────────────┼──────────┤
    │                │           │                │          Total │ 276 (1.1 KB)       │ 5,790 (23.2 KB)         │ 2 (12 B) │
    └────────────────┴───────────┴────────────────┴────────────────┴────────────────────┴─────────────────────────┴──────────┘

                                                Total Parameters: 6,068 (24.3 KB)

  Note that ``block2/dropout`` is not shown in the table because it shares the
  same ``RngState`` with ``block1/dropout``.

  Args:
    obj: A object to summarize. It can a pytree or a graph objects
      such as nnx.Module or nnx.Optimizer.
    *input_args: Positional arguments passed to the object's method.
    **input_kwargs: Keyword arguments passed to the object's method.
    depth: The depth of the table.
    method: The method to call on the object. Default is ``'__call__'``.
    row_filter: A callable that filters the rows to be displayed in the table.
      By default, it filters out rows with type ``nnx.RngStream``.
    table_kwargs: An optional dictionary with additional keyword arguments
      that are passed to ``rich.table.Table`` constructor.
    column_kwargs: An optional dictionary with additional keyword arguments
      that are passed to ``rich.table.Table.add_column`` when adding columns to
      the table.
    console_kwargs: An optional dictionary with additional keyword arguments
      that are passed to `rich.console.Console` when rendering the table.
      Default arguments are  ``'force_terminal': True``, and ``'force_jupyter'``
      is set to ``True`` if the code is running in a Jupyter notebook, otherwise
      it is set to ``False``.

  Returns:
    A string summarizing the object.
  """
  _console_kwargs = {'force_terminal': True, 'force_jupyter': in_ipython}
  _console_kwargs.update(console_kwargs)

  obj = graph.clone(obj)  # create copy to avoid side effects
  node_stats: NodeStats = {}
  object_types: set[type] = set()
  _collect_stats((), obj, node_stats, object_types)
  _variable_types: set[type] = {
    nnx.RngState  # type: ignore[misc]
    if issubclass(variable_state.type, nnx.RngState)
    else variable_state.type
    for _, variable_state in nnx.to_flat_state(nnx.state(obj))
  }
  variable_types: list[type] = sorted(_variable_types, key=lambda t: t.__name__)

  rows: list[CallInfo] = []
  eval_fn = functools.partial(
    _call_obj,
    call_method=method,
    rows=rows,
    node_stats=node_stats,
    object_types=object_types,
  )
  nnx.eval_shape(eval_fn, obj, input_args, input_kwargs)

  if depth is not None:
    rows = [row for row in rows if len(row.path) <= depth and row_filter(row)]
  else:
    rows = [row for row in rows if row_filter(row)]

  rich_table = rich.table.Table(
    show_header=True,
    show_lines=True,
    show_footer=True,
    title=f'{type(obj).__name__} Summary',
    **table_kwargs,
  )

  rich_table.add_column('path', **column_kwargs)
  rich_table.add_column('type', **column_kwargs)
  rich_table.add_column('inputs', **column_kwargs)
  rich_table.add_column('outputs', **column_kwargs)

  for var_type in variable_types:
    rich_table.add_column(var_type.__name__, **column_kwargs)

  for row in rows:
    node_info = node_stats[row.object_id]
    assert node_info is not None
    col_reprs: list[str] = []
    path_str = '/'.join(map(str, row.path))
    col_reprs.append(path_str)
    col_reprs.append(row.type.__name__)
    inputs_repr = ''
    if row.input_args:
      input_args = row.input_args
      if len(row.input_args) == 1 and not row.input_kwargs:
        input_args = row.input_args[0]
      inputs_repr += _as_yaml_str(input_args)
      if inputs_repr and row.input_kwargs:
        inputs_repr += '\n'
    if row.input_kwargs:
      inputs_repr += _as_yaml_str(row.input_kwargs)
    col_reprs.append(inputs_repr)
    col_reprs.append(_as_yaml_str(row.outputs))

    for var_type in variable_types:
      attributes = {}
      for name, variable in node_info.variable_groups[var_type].items():
        value = variable.value
        value_repr = _render_array(value) if _has_shape_dtype(value) else ''
        metadata = variable.get_metadata()

        if metadata:
          attributes[name] = {
            'value': value_repr,
            **metadata,
          }
        elif value_repr:
          attributes[name] = value_repr

      if attributes:
        col_repr = _as_yaml_str(attributes) + '\n\n'
      else:
        col_repr = ''

      size_bytes = node_info.stats.get(var_type)  # type: ignore[call-overload]
      if size_bytes:
        col_repr += f'[bold]{size_bytes}[/bold]'
      col_reprs.append(col_repr)

    rich_table.add_row(*col_reprs)

  rich_table.columns[3].footer = rich.text.Text.from_markup(
    'Total', justify='right'
  )
  node_info = node_stats[id(obj)]
  assert node_info is not None
  for i, var_type in enumerate(variable_types):
    size_bytes = node_info.stats[var_type]
    rich_table.columns[i + 4].footer = str(size_bytes)

  rich_table.caption_style = 'bold'
  total_size = sum(node_info.stats.values(), SizeBytes(0, 0))
  rich_table.caption = f'\nTotal Parameters: {total_size}'

  return _get_rich_repr(rich_table, _console_kwargs)


def _get_rich_repr(obj, console_kwargs):
  f = io.StringIO()
  console = rich.console.Console(file=f, **console_kwargs)
  console.print(obj)
  return f.getvalue()


def _size_and_bytes(pytree: tp.Any) -> tuple[int, int]:
  leaves = jax.tree.leaves(pytree)
  size = sum(x.size for x in leaves if hasattr(x, 'size'))
  num_bytes = sum(
    x.size * x.dtype.itemsize for x in leaves if hasattr(x, 'size')
  )
  return size, num_bytes


def _size_and_bytes_repr(size: int, num_bytes: int) -> str:
  if not size:
    return ''
  bytes_repr = _bytes_repr(num_bytes)
  return f'{size:,} [dim]({bytes_repr})[/dim]'


def _bytes_repr(num_bytes):
  count, units = (
    (f'{num_bytes / 1e9:,.1f}', 'GB')
    if num_bytes > 1e9
    else (f'{num_bytes / 1e6:,.1f}', 'MB')
    if num_bytes > 1e6
    else (f'{num_bytes / 1e3:,.1f}', 'KB')
    if num_bytes > 1e3
    else (f'{num_bytes:,}', 'B')
  )

  return f'{count} {units}'


def _has_shape_dtype(value):
  return hasattr(value, 'shape') and hasattr(value, 'dtype')


def _normalize_values(x):
  if isinstance(x, type):
    return f'type[{x.__name__}]'
  elif isinstance(x, ArrayRepr | SimpleObjectRepr):
    return str(x)
  else:
    return x

def _maybe_pytree_to_dict(pytree: tp.Any):
  path_leaves = jax.tree_util.tree_flatten_with_path(pytree)[0]
  path_leaves = [
    (tuple(map(graph._key_path_to_key, path)), value)
    for path, value in path_leaves
  ]
  if len(path_leaves) < 1:
    return pytree
  elif len(path_leaves) == 1 and path_leaves[0][0] == ():
    return pytree
  else:
    return _unflatten_to_simple_structure(path_leaves)


def _unflatten_to_simple_structure(xs: list[tuple[tuple[tp.Any, ...], tp.Any]]):
  result: list | dict = (
    [] if len(xs) > 0 and isinstance(xs[0][0][0], int) else {}
  )
  for path, value in xs:
    cursor = result
    for i, key in enumerate(path[:-1]):
      if isinstance(cursor, list):
        if key == len(cursor):
          next_key = path[i + 1]
          if isinstance(next_key, int):
            cursor.append([])
          else:
            cursor.append({})
      else:
        if key not in cursor:
          next_key = path[i + 1]
          if isinstance(next_key, int):
            cursor[key] = []
          else:
            cursor[key] = {}
      cursor = cursor[key]
    cursor[path[-1]] = value
  return result

def _as_yaml_str(value) -> str:
  if (hasattr(value, '__len__') and len(value) == 0) or value is None:
    return ''

  value = jax.tree.map(_normalize_values, value)
  value = _maybe_pytree_to_dict(value)

  file = io.StringIO()
  yaml.safe_dump(
    value,
    file,
    default_flow_style=False,
    indent=2,
    sort_keys=False,
    explicit_end=False,
  )
  return file.getvalue().replace('\n...', '').replace("'", '').strip()


def _render_array(x):
  shape, dtype = jnp.shape(x), jnp.result_type(x)
  shape_repr = ','.join(str(x) for x in shape)
  return f'[dim]{dtype}[/dim][{shape_repr}]'


def _sort_variable_types(types: tp.Iterable[type]) -> list[type]:
  def _variable_parents_count(t: type):
    return sum(1 for p in t.mro() if issubclass(p, variablelib.Variable))

  type_sort_key = {t: (-_variable_parents_count(t), t.__name__) for t in types}
  return sorted(types, key=lambda t: type_sort_key[t])
