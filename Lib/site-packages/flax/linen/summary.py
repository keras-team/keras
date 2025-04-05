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

"""Flax Module summary library."""

import dataclasses
import enum
import io
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import (
  Any,
)
from collections.abc import Callable, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import rich.console
import rich.table
import rich.text
import yaml

import flax.linen.module as module_lib
from flax.core import meta, unfreeze
from flax.core.scope import (
  CollectionFilter,
  DenyList,
  LazyRng,
)
from flax.typing import (
  Array,
  PRNGKey,
  RNGSequences,
  FrozenVariableDict,
  MutableVariableDict,
  LogicalNames,
)


class _ValueRepresentation(ABC):
  """A class that represents a value in the summary table."""

  @abstractmethod
  def render(self) -> str:
    ...


@dataclasses.dataclass
class _ArrayRepresentation(_ValueRepresentation):
  shape: tuple[int, ...]
  dtype: Any

  @classmethod
  def from_array(cls, x: Array) -> '_ArrayRepresentation':
    return cls(jnp.shape(x), jnp.result_type(x))

  @classmethod
  def render_array(cls, x) -> str:
    return cls.from_array(x).render()

  def render(self):
    shape_repr = ','.join(str(x) for x in self.shape)
    return f'[dim]{self.dtype}[/dim][{shape_repr}]'


@dataclasses.dataclass
class _PartitionedArrayRepresentation(_ValueRepresentation):
  array_representation: _ArrayRepresentation
  names: LogicalNames

  @classmethod
  def from_partitioned(
    cls, partitioned: meta.Partitioned
  ) -> '_PartitionedArrayRepresentation':
    return cls(
      _ArrayRepresentation.from_array(partitioned.value), partitioned.names
    )

  def render(self):
    return self.array_representation.render() + f' [dim]P[/dim]{self.names}'


@dataclasses.dataclass
class _ObjectRepresentation(_ValueRepresentation):
  obj: Any

  def render(self):
    return repr(self.obj)


@dataclasses.dataclass
class Row:
  """Contains the information about a single row in the summary table.

  Attributes:
    path: A tuple of strings that represents the path to the module.
    module_copy: A copy of the module being summarized.
    method: method of the module called.
    inputs: inputs to the module.
    outputs: Output of the Module as reported by `capture_intermediates`.
    module_variables: Dictionary of variables in the module (no submodules
      included).
    counted_variables: Dictionary of variables that should be counted for this
      row, if no summarization is done (e.g. `depth=None` in `module_summary`)
      then this field is the same as `module_variables`, however if a
      summarization is done then this dictionary potentially contains parameters
      from submodules depending on the depth of the Module in question.
    flops: FLOPs cost of calling the module method.
    vjp_flops: FLOPs cost of calling the VJP of the module method.
  """

  path: tuple[str, ...]
  module_copy: module_lib.Module
  method: str
  inputs: Any
  outputs: Any
  module_variables: dict[str, dict[str, Any]]
  counted_variables: dict[str, dict[str, Any]]
  flops: int
  vjp_flops: int

  def __post_init__(self):
    self.inputs = self.inputs
    self.outputs = self.outputs
    self.module_variables = self.module_variables
    self.counted_variables = self.counted_variables

  def size_and_bytes(
    self, collections: Iterable[str]
  ) -> dict[str, tuple[int, int]]:
    return {
      col: (
        _size_and_bytes(self.counted_variables[col])
        if col in self.counted_variables
        else (0, 0)
      )
      for col in collections
    }


class Table(list[Row]):
  """A list of Row objects.

  Table inherits from `List[Row]` so it has all the methods of a list, however
  it also contains some additional fields:

  * `module`: the module that this table is summarizing
  * `collections`: a list containing the parameter collections (e.g. 'params', 'batch_stats', etc)
  """

  def __init__(
    self,
    module: module_lib.Module,
    collections: Sequence[str],
    rows: Iterable[Row],
  ):
    super().__init__(rows)
    self.module = module
    self.collections = collections


def tabulate(
  module: module_lib.Module,
  rngs: PRNGKey | RNGSequences,
  depth: int | None = None,
  show_repeated: bool = False,
  mutable: CollectionFilter = DenyList('intermediates'),
  console_kwargs: Mapping[str, Any] | None = None,
  table_kwargs: Mapping[str, Any] = MappingProxyType({}),
  column_kwargs: Mapping[str, Any] = MappingProxyType({}),
  compute_flops: bool = False,
  compute_vjp_flops: bool = False,
  **kwargs,
) -> Callable[..., str]:
  """Returns a function that creates a summary of the Module represented as a table.

  This function accepts most of the same arguments and internally calls
  `Module.init`, except that it returns a function of the form
  `(*args, **kwargs) -> str` where `*args` and `**kwargs` are passed to
  `method` (e.g. `__call__`) during the forward pass.

  `tabulate` uses `jax.eval_shape` under the hood to run the forward computation
  without consuming any FLOPs or allocating memory.

  Additional arguments can be passed into the `console_kwargs` argument, for
  example, `{'width': 120}`. For a full list of `console_kwargs` arguments, see:
  https://rich.readthedocs.io/en/stable/reference/console.html#rich.console.Console

  Example::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> class Foo(nn.Module):
    ...   @nn.compact
    ...   def __call__(self, x):
    ...     h = nn.Dense(4)(x)
    ...     return nn.Dense(2)(h)

    >>> x = jnp.ones((16, 9))
    >>> tabulate_fn = nn.tabulate(
    ...     Foo(), jax.random.key(0), compute_flops=True, compute_vjp_flops=True)

    >>> # print(tabulate_fn(x))

  This gives the following output::

                                           Foo Summary
    ┏━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
    ┃ path    ┃ module ┃ inputs        ┃ outputs       ┃ flops ┃ vjp_flops ┃ params          ┃
    ┡━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
    │         │ Foo    │ float32[16,9] │ float32[16,2] │ 1504  │ 4460      │                 │
    ├─────────┼────────┼───────────────┼───────────────┼───────┼───────────┼─────────────────┤
    │ Dense_0 │ Dense  │ float32[16,9] │ float32[16,4] │ 1216  │ 3620      │ bias:           │
    │         │        │               │               │       │           │ float32[4]      │
    │         │        │               │               │       │           │ kernel:         │
    │         │        │               │               │       │           │ float32[9,4]    │
    │         │        │               │               │       │           │                 │
    │         │        │               │               │       │           │ 40 (160 B)      │
    ├─────────┼────────┼───────────────┼───────────────┼───────┼───────────┼─────────────────┤
    │ Dense_1 │ Dense  │ float32[16,4] │ float32[16,2] │ 288   │ 840       │ bias:           │
    │         │        │               │               │       │           │ float32[2]      │
    │         │        │               │               │       │           │ kernel:         │
    │         │        │               │               │       │           │ float32[4,2]    │
    │         │        │               │               │       │           │                 │
    │         │        │               │               │       │           │ 10 (40 B)       │
    ├─────────┼────────┼───────────────┼───────────────┼───────┼───────────┼─────────────────┤
    │         │        │               │               │       │     Total │ 50 (200 B)      │
    └─────────┴────────┴───────────────┴───────────────┴───────┴───────────┴─────────────────┘

                                   Total Parameters: 50 (200 B)


  **Note**: rows order in the table does not represent execution order,
  instead it aligns with the order of keys in `variables` which are sorted
  alphabetically.

  **Note**: `vjp_flops` returns `0` if the module is not differentiable.

  Args:
    module: The module to tabulate.
    rngs: The rngs for the variable collections as passed to `Module.init`.
    depth: controls how many submodule deep the summary can go. By default its
      `None` which means no limit. If a submodule is not shown because of the
      depth limit, its parameter count and bytes will be added to the row of its
      first shown ancestor such that the sum of all rows always adds up to the
      total number of parameters of the Module.
    show_repeated: If `True`, repeated calls to the same module will be shown
      in the table, otherwise only the first call will be shown. Default is
      `False`.
    mutable: Can be bool, str, or list. Specifies which collections should be
      treated as mutable: ``bool``: all/no collections are mutable. ``str``: The
      name of a single mutable collection. ``list``: A list of names of mutable
      collections. By default all collections except 'intermediates' are
      mutable.
    console_kwargs: An optional dictionary with additional keyword arguments
      that are passed to `rich.console.Console` when rendering the table.
      Default arguments are `{'force_terminal': True, 'force_jupyter': False}`.
    table_kwargs: An optional dictionary with additional keyword arguments that
      are passed to `rich.table.Table` constructor.
    column_kwargs: An optional dictionary with additional keyword arguments that
      are passed to `rich.table.Table.add_column` when adding columns to the
      table.
    compute_flops: whether to include a `flops` column in the table listing the
      estimated FLOPs cost of each module forward pass. Does incur actual
      on-device computation / compilation / memory allocation, but still
      introduces overhead for large modules (e.g. extra 20 seconds for a
      Stable Diffusion's UNet, whereas otherwise tabulation would finish in 5
      seconds).
    compute_vjp_flops: whether to include a `vjp_flops` column in the table
      listing the estimated FLOPs cost of each module backward pass. Introduces
      a compute overhead of about 2-3X of `compute_flops`.
    **kwargs: Additional arguments passed to `Module.init`.

  Returns:
    A function that accepts the same `*args` and `**kwargs` of the forward pass
    (`method`) and returns a string with a tabular representation of the
    Modules.
  """
  # add non-default arguments to kwargs, this prevents some issue we overloading init
  # see: https://github.com/google/flax/issues/3299
  if mutable != DenyList('intermediates'):
    kwargs['mutable'] = mutable

  def _tabulate_fn(*fn_args, **fn_kwargs):
    table_fn = _get_module_table(
      module,
      depth=depth,
      show_repeated=show_repeated,
      compute_flops=compute_flops,
      compute_vjp_flops=compute_vjp_flops,
    )

    table = table_fn(rngs, *fn_args, **fn_kwargs, **kwargs)

    non_param_cols = [
      'path',
      'module',
      'inputs',
      'outputs',
    ]

    if compute_flops:
      non_param_cols.append('flops')
    if compute_vjp_flops:
      non_param_cols.append('vjp_flops')

    return _render_table(
      table, console_kwargs, table_kwargs, column_kwargs, non_param_cols
    )

  return _tabulate_fn


def _get_flops(fn, *args, **kwargs):
  e = jax.jit(fn).lower(*args, **kwargs)
  cost = e.cost_analysis()
  if cost is None:
    return 0
  flops = int(cost['flops']) if 'flops' in cost else 0
  return flops


def _get_call_flops(
  c: module_lib._CallInfo,
  compute_flops: bool,
  compute_vjp_flops: bool,
) -> tuple[int, int]:
  """Return the FLOPs of executing the call `c` in the call stack.

  Does not perform actual computation / compilation / memory allocation, but
  still introduces overhead for large modules.

  Args:
    c: ``_CallInfo``.
    compute_flops: whether to compute forward pass FLOPs. Return `-1` otherwise.
    compute_vjp_flops: whether to compute backward pass FLOPs. Return `-1`
      otherwise.

  Returns:
    FLOPs of executing forward pass of `c`, and its VJP.
  """

  if not compute_flops and not compute_vjp_flops:
    return -1, -1

  rngs = jax.tree_util.tree_map(
      lambda x: x.rng, c.rngs, is_leaf=lambda x: isinstance(x, LazyRng)
  )

  args = jax.tree_util.tree_map(_from_value_representation, c.args)
  kwargs = jax.tree_util.tree_map(_from_value_representation, c.kwargs)

  leaves, treedef = jax.tree_util.tree_flatten((args, kwargs))
  dynamic_leaves = []
  dynamic_idxs = []
  for i, arg in enumerate(leaves):
    if isinstance(arg, jax.ShapeDtypeStruct):
      dynamic_leaves.append(arg)
      dynamic_idxs.append(i)

  def _get_inputs(dynamic_leaves):
    new_leaves: list[Any] = leaves.copy()
    for i, arg in zip(dynamic_idxs, dynamic_leaves):
      new_leaves[i] = arg
    return treedef.unflatten(new_leaves)

  def init(rngs, dynamic_leaves):
    """`c.module.init` closed over static keyword arguments."""
    args, kwargs = _get_inputs(dynamic_leaves)
    return c.module.init(
      rngs,
      *args,
      method=c.method,
      mutable=c.mutable,
      **kwargs,
    )

  variables = jax.eval_shape(init, rngs, dynamic_leaves)

  def apply(variables, rngs, dynamic_leaves):
    """`c.module.apply` closed over static keyword arguments."""
    args, kwargs = _get_inputs(dynamic_leaves)
    return c.module.apply(
      variables,
      *args,
      rngs=rngs,
      method=c.method,
      mutable=c.mutable,
      **kwargs,
    )

  # Forward pass FLOPs
  if compute_flops:
    flops = _get_flops(apply, variables, rngs, dynamic_leaves)
  else:
    flops = -1

  if compute_vjp_flops:
    # Backward pass FLOPs
    def apply_vjp(variables, rngs, dynamic_leaves):
      """VJP of `c.module.apply` closed over static keyword arguments."""
      out, vjp_fn = jax.vjp(apply, variables, rngs, dynamic_leaves)
      return vjp_fn(out)

    vjp_flops = _get_flops(apply_vjp, variables, rngs, dynamic_leaves)
  else:
    vjp_flops = -1

  return flops, vjp_flops


def _get_module_table(
  module: module_lib.Module,
  depth: int | None,
  show_repeated: bool,
  compute_flops: bool,
  compute_vjp_flops: bool,
) -> Callable[..., Table]:
  """A function that takes a Module and returns function with the same signature
  as `init` but returns the Table representation of the Module."""

  def _get_table_fn(*args, **kwargs):
    with module_lib._tabulate_context():

      def _get_variables():
        return module.init(*args, **kwargs)
      # TODO(cgarciae): is it possible to avoid leaking tracers for summaries?
      with jax.check_tracer_leaks(False):
        variables = jax.eval_shape(_get_variables)
      calls = module_lib._context.call_info_stack[-1].calls
      calls.sort(key=lambda c: c.index)

    collections: set[str] = set(variables.keys())
    rows = []
    all_paths: set[tuple[str, ...]] = {call.path for call in calls}
    visited_paths: set[tuple[str, ...]] = set()

    for c in calls:
      call_depth = len(c.path)
      inputs = _process_inputs(c.args, c.kwargs)

      if c.path in visited_paths or not hasattr(c.module, c.method):
        if not show_repeated:
          continue
        module_vars = {}
        counted_vars = {}
      elif depth is not None:
        if call_depth > depth:
          continue
        module_vars, _ = _get_module_variables(c.path, variables, all_paths)
        if call_depth == depth:
          counted_vars = _get_path_variables(c.path, variables)
        else:
          counted_vars = module_vars
      else:
        module_vars, _ = _get_module_variables(c.path, variables, all_paths)
        counted_vars = module_vars

      visited_paths.add(c.path)
      rows.append(
        Row(
          c.path,
          c.module.copy(parent=None),
          c.method,
          inputs,
          c.outputs,
          module_vars,
          counted_vars,
          *_get_call_flops(c, compute_flops, compute_vjp_flops),
        )
      )

    return Table(module, tuple(collections), rows)

  return _get_table_fn


def _get_module_variables(
  path: tuple[str, ...],
  variables: FrozenVariableDict,
  all_paths: set[tuple[str, ...]],
) -> tuple[MutableVariableDict, Any]:
  """A function that takes a path and variables structure and returns a

  (module_variables, submodule_variables) tuple for that path.
  _get_module_variables
  uses the `all_paths` set to determine if a variable belongs to a submodule or
  not.
  """
  module_variables = _get_path_variables(path, variables)
  submodule_variables: Any = {collection: {} for collection in module_variables}
  all_keys = {
    key for collection in module_variables.values() for key in collection
  }

  for key in all_keys:
    submodule_path = path + (key,)
    if submodule_path in all_paths:
      for collection in module_variables:
        if key in module_variables[collection]:
          submodule_variables[collection][key] = module_variables[
            collection
          ].pop(key)

  return module_variables, submodule_variables


def _get_path_variables(
  path: tuple[str, ...], variables: FrozenVariableDict
) -> MutableVariableDict:
  """A function that takes a path and a variables structure and returns the
  variable structure at that path.
  """
  path_variables = {}

  for collection in variables:
    collection_variables = variables[collection]
    for name in path:
      if name not in collection_variables:
        collection_variables = None
        break
      collection_variables = collection_variables[name]

    if collection_variables is not None:
      path_variables[collection] = unfreeze(collection_variables)

  return path_variables


def _process_inputs(args, kwargs) -> Any:
  """A function that normalizes the representation of the ``args`` and
  ``kwargs`` for the ``inputs`` column.
  """
  if args and kwargs:
    input_values = (*args, kwargs)
  elif args and not kwargs:
    input_values = args[0] if len(args) == 1 else args
  elif kwargs and not args:
    input_values = kwargs
  else:
    input_values = ()

  return input_values


def _render_table(
  table: Table,
  console_extras: Mapping[str, Any] | None,
  table_kwargs: Mapping[str, Any],
  column_kwargs: Mapping[str, Any],
  non_params_cols: list[str],
) -> str:
  """A function that renders a Table to a string representation using rich."""
  console_kwargs = {'force_terminal': True, 'force_jupyter': False}
  if console_extras is not None:
    console_kwargs.update(console_extras)

  rich_table = rich.table.Table(
    show_header=True,
    show_lines=True,
    show_footer=True,
    title=f'{table.module.__class__.__name__} Summary',
    **table_kwargs,
  )

  for c in non_params_cols:
    rich_table.add_column(c, **column_kwargs)

  for col in table.collections:
    rich_table.add_column(col, **column_kwargs)

  for row in table:
    collections_size_repr = []

    for collection, size_bytes in row.size_and_bytes(table.collections).items():
      col_repr = ''

      if collection in row.module_variables:
        module_variables = _represent_tree(row.module_variables[collection])
        module_variables = _normalize_structure(module_variables)
        col_repr += _as_yaml_str(
          _summary_tree_map(_maybe_render, module_variables)
        )
        if col_repr:
          col_repr += '\n\n'

      col_repr += f'[bold]{_size_and_bytes_repr(*size_bytes)}[/bold]'
      collections_size_repr.append(col_repr)

    no_show_methods = {'__call__', '<lambda>'}
    path_repr = '/'.join(row.path)
    method_repr = (
      f' [dim]({row.method})[/dim]' if row.method not in no_show_methods else ''
    )
    rich_table.add_row(
      path_repr,
      type(row.module_copy).__name__ + method_repr,
      *(
        _as_yaml_str(
          _summary_tree_map(
            _maybe_render, _normalize_structure(getattr(row, c))
          )
        )
        for c in non_params_cols[2:]
      ),
      *collections_size_repr,
    )

  # add footer with totals
  n_non_params_cols = len(non_params_cols)
  rich_table.columns[n_non_params_cols - 1].footer = rich.text.Text.from_markup(
    'Total', justify='right'
  )

  # get collection totals
  collection_total = {col: (0, 0) for col in table.collections}
  for row in table:
    for col, size_bytes in row.size_and_bytes(table.collections).items():
      collection_total[col] = (
        collection_total[col][0] + size_bytes[0],
        collection_total[col][1] + size_bytes[1],
      )

  # add totals to footer
  for i, col in enumerate(table.collections):
    rich_table.columns[n_non_params_cols + i].footer = _size_and_bytes_repr(
      *collection_total[col]
    )

  # add final totals to caption
  caption_totals = (0, 0)
  for size, num_bytes in collection_total.values():
    caption_totals = (
      caption_totals[0] + size,
      caption_totals[1] + num_bytes,
    )

  rich_table.caption_style = 'bold'
  rich_table.caption = (
    f'\nTotal Parameters: {_size_and_bytes_repr(*caption_totals)}'
  )

  return '\n' + _get_rich_repr(rich_table, console_kwargs) + '\n'


def _summary_tree_map(f, tree, *rest):
  return jax.tree_util.tree_map(f, tree, *rest, is_leaf=lambda x: x is None)


def _size_and_bytes_repr(size: int, num_bytes: int) -> str:
  if not size:
    return ''
  bytes_repr = _bytes_repr(num_bytes)
  return f'{size:,} [dim]({bytes_repr})[/dim]'


def _size_and_bytes(pytree: Any) -> tuple[int, int]:
  leaves = jax.tree_util.tree_leaves(pytree)
  size = sum(x.size for x in leaves if hasattr(x, 'size'))
  num_bytes = sum(
    x.size * x.dtype.itemsize for x in leaves if hasattr(x, 'size')
  )
  return size, num_bytes


def _get_rich_repr(obj, console_kwargs):
  f = io.StringIO()
  console = rich.console.Console(file=f, **console_kwargs)
  console.print(obj)
  return f.getvalue()


def _as_yaml_str(value) -> str:
  if (hasattr(value, '__len__') and len(value) == 0) or value is None:
    return ''

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


def _normalize_structure(obj):
  if isinstance(obj, _ValueRepresentation):
    return obj
  if isinstance(obj, (tuple, list)):
    return tuple(map(_normalize_structure, obj))
  elif isinstance(obj, Mapping):
    return {
      _normalize_structure(k): _normalize_structure(v) for k, v in obj.items()
    }
  elif dataclasses.is_dataclass(obj):
    return {
      f.name: _normalize_structure(getattr(obj, f.name))
      for f in dataclasses.fields(obj)
    }
  elif isinstance(obj, enum.Enum):
    # `yaml.safe_dump` does not support Enum key types so extract the underlying value
    return obj.value
  else:
    return obj


def _bytes_repr(num_bytes):
  count, units = (
    (f'{num_bytes / 1e9 :,.1f}', 'GB')
    if num_bytes > 1e9
    else (f'{num_bytes / 1e6 :,.1f}', 'MB')
    if num_bytes > 1e6
    else (f'{num_bytes / 1e3 :,.1f}', 'KB')
    if num_bytes > 1e3
    else (f'{num_bytes:,}', 'B')
  )

  return f'{count} {units}'


def _get_value_representation(x: Any) -> _ValueRepresentation:
  if isinstance(x, (int, float, bool, type(None))) or (
    isinstance(x, np.ndarray) and np.isscalar(x)
  ):
    return _ObjectRepresentation(x)
  elif isinstance(x, meta.Partitioned):
    return _PartitionedArrayRepresentation.from_partitioned(x)
  try:
    return _ArrayRepresentation.from_array(x)
  except:
    return _ObjectRepresentation(x)


def _from_value_representation(x: _ValueRepresentation) -> Any:
  if isinstance(x, _ArrayRepresentation):
    return jax.ShapeDtypeStruct(x.shape, x.dtype)

  elif isinstance(x, _PartitionedArrayRepresentation):
    return jax.ShapeDtypeStruct(
      x.array_representation.shape, x.array_representation.dtype
    )

  elif isinstance(x, _ObjectRepresentation):
    return x.obj

  raise TypeError(x, type(x))


def _represent_tree(x):
  """Returns a tree with the same structure as `x` but with each leaf replaced
  by a `_ValueRepresentation` object."""
  return jax.tree_util.tree_map(
    _get_value_representation,
    x,
    is_leaf=lambda x: x is None or isinstance(x, meta.Partitioned),
  )


def _maybe_render(x):
  return x.render() if hasattr(x, 'render') else repr(x)
