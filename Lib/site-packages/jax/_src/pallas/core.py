# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for pallas-core functionality."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
import contextlib
import copy
import dataclasses
import enum
import functools
import itertools
import threading
from typing import Any, ClassVar, Hashable, Protocol, Union, runtime_checkable

import jax
from jax._src import api_util
from jax._src import config
from jax._src import core as jax_core
from jax._src import dtypes
from jax._src import linear_util as lu
from jax._src import state
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.state import discharge as state_discharge
from jax._src.state import types as state_types
from jax._src.state.types import TransformedRef
import jax.numpy as jnp

class DynamicGridDim:
  def __repr__(self):
    return "DynamicGridDim"
dynamic_grid_dim = DynamicGridDim()


partial = functools.partial
GridElement = int | jax_core.Array
GridName = Hashable
GridNames = tuple[Hashable, ...] | None
NamedGrid = tuple[tuple[GridName, int], ...]
TupleGrid = tuple[GridElement, ...]
Grid = Union[NamedGrid, TupleGrid]
StaticGrid = tuple[int, ...]
GridMappingGrid = tuple[int | DynamicGridDim, ...]
OriginStr = str  # The origin of a block spec, e.g. input[2]["field"]

# Datatype for semaphore values in interpret mode.
# For now, we choose a relatively uncommon datatype (i16) so it is more easily
# identifiable in kernels.
# TODO(justinfu): Handle semaphores with a custom extended dtype.
SEMAPHORE_INTERPRET_DTYPE = jnp.int16
SEMAPHORE_MAX_VALUE = jnp.iinfo(SEMAPHORE_INTERPRET_DTYPE).max


@runtime_checkable
class CompilerParams(Protocol):
  """Base class for compiler parameters."""
  PLATFORM: ClassVar[str]

  # Subclasses must be dataclasses.
  __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]


@dataclasses.dataclass(frozen=True)
class NameAndSrcInfo:
  #: The name of the pallas_call or the name of the kernel function.
  name: str
  #: the source info, and the name of kernel function if not in `name`.`
  src_info: str

  def __str__(self):
    return f"{self.name}{' ' if self.src_info else ''}{self.src_info}"
  __repr__ = __str__

  replace = dataclasses.replace


  @staticmethod
  def from_pallas_call(pallas_call_name: str | None,
                       src_info : str | None) -> NameAndSrcInfo:
    """Formats the name and the source info.

    Args:
      pallas_call_name: The `name` argument to pallas_call.
      src_info: The result of `api_util.fun_source_info(kernel)`, in the form
        "{function_name} at {file_name}:{line_number}".
    """
    if pallas_call_name is not None:
      pallas_call_name = mlir._module_name_regex.sub("_", pallas_call_name)
    if src_info is None:
      return NameAndSrcInfo(
          "unknown" if pallas_call_name is None else pallas_call_name,
          "")
    if pallas_call_name is not None:
      return NameAndSrcInfo(pallas_call_name,
                            f"for kernel function {src_info}")
    src_info_parts = src_info.split(" ")
    return NameAndSrcInfo(src_info_parts[0],
                          " ".join(src_info_parts[1:]))


split_list = util.split_list

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


class ShapedArrayWithMemorySpace(jax_core.ShapedArray):
  __slots__ = ["memory_space"]

  def __init__(self, shape, dtype, weak_type=False, sharding=None,
               memory_space=None):
    super().__init__(shape, dtype, weak_type=weak_type, sharding=sharding)
    self.memory_space = memory_space

  def __eq__(self, other):
    return super().__eq__(other) and self.memory_space == other.memory_space

  def __hash__(self):
    return hash((
        self.shape,
        self.dtype,
        self.weak_type,
        getattr(self, "sharding", None),
        self.memory_space,
    ))

  def str_short(self, short_dtypes=False):
    dt_str = \
        dtypes.short_dtype_name(self.dtype) if short_dtypes else self.dtype.name
    dt_str = dt_str.replace("void", "float0")
    shapestr = ",".join(map(str, self.shape))
    if hasattr(self, "sharding"):
      sharding_str = f"{dt_str}[{shapestr}]({self.sharding})"
    else:
      sharding_str = ""
    memoryspace_str = (
        "" if self.memory_space is None else f"<{self.memory_space}>"
    )
    return f"{dt_str}{memoryspace_str}[{shapestr}]{sharding_str}"

  def update(
      self,
      shape=None,
      dtype=None,
      weak_type=None,
      sharding=None,
      memory_space=None,
  ):
    if shape is None:
      shape = self.shape
    if dtype is None:
      dtype = self.dtype
    if weak_type is None:
      weak_type = self.weak_type
    if sharding is None:
      sharding = getattr(self, "sharding", None)
    if memory_space is None:
      memory_space = self.memory_space
    return ShapedArrayWithMemorySpace(
        shape, dtype, weak_type, sharding=sharding, memory_space=memory_space
    )
mlir.ir_type_handlers[ShapedArrayWithMemorySpace] = mlir._array_ir_types


@dataclasses.dataclass(frozen=True)
class MemoryRef:
  """Like jax.ShapeDtypeStruct but with memory spaces."""
  shape: tuple[int, ...]
  dtype: jnp.dtype
  # TODO(b/368122763): Unify memory space types across backends
  memory_space: Any

  def get_array_aval(self) -> jax_core.ShapedArray:
    dtype = self.dtype
    if not isinstance(dtype, (jnp.dtype, dtypes.ExtendedDType)):
      dtype = jnp.dtype(dtype)
    return ShapedArrayWithMemorySpace(
        self.shape, dtype, memory_space=self.memory_space
    )

  def get_ref_aval(self) -> TransformedRef | AbstractMemoryRef:
    # TODO(sharadmv): Clean this up. ShapedArrayWithMemorySpace fails when we
    # try to apply JAX ops to it.
    return AbstractMemoryRef(
        jax_core.ShapedArray(self.shape, self.dtype), self.memory_space)


class AbstractMemoryRef(state.AbstractRef):
  __slots__ = ["inner_aval", "memory_space"]

  inner_aval: jax_core.ShapedArray

  def __init__(self, inner_aval: jax_core.ShapedArray, memory_space: Any):
    if isinstance(inner_aval, ShapedArrayWithMemorySpace):
      if inner_aval.memory_space is not None:
        assert inner_aval.memory_space == memory_space, (
            f"Mismatched memory spaces: {inner_aval.memory_space=},"
            f" {memory_space=}"
        )
    self.inner_aval = inner_aval
    self.memory_space = memory_space

  def __repr__(self) -> str:
    return f'MemRef<{self.memory_space}>{{{self.inner_aval.str_short()}}}'

  @property
  def sharding(self):
    return self.inner_aval.sharding

  def update_weak_type(self, weak_type):
    return AbstractMemoryRef(
        self.inner_aval.update_weak_type(weak_type), self.memory_space)

  def update(self, inner_aval=None, memory_space=None):
    inner_aval = self.inner_aval if inner_aval is None else inner_aval
    memory_space = self.memory_space if memory_space is None else memory_space
    return AbstractMemoryRef(inner_aval, memory_space)

  def to_tangent_aval(self):
    return AbstractMemoryRef(
        self.inner_aval.to_tangent_aval(), self.memory_space)

  # TODO(dougalm, sharadmv): figure out how to avoid needing this
  def normalize(self):
    return state.AbstractRef(self.inner_aval).normalize()

  def __eq__(self, other):
    return (type(self) is type(other) and self.inner_aval == other.inner_aval
            and self.memory_space == other.memory_space)

  def __hash__(self):
    return hash((self.__class__, self.inner_aval, self.memory_space))


class MemorySpace(enum.Enum):
  """Logical, device-agnostic memory spaces.

  Each memory space will be translated to a device-specific memory
  type during lowering.
  """
  ANY = "any"  # Unrestricted memory space (usually HBM)
  ERROR = "error"  # Memory space for checkify errors.
  INDEX = "index"  # Memory space for scalar prefetch arguments.

  def __str__(self) -> str:
    return self.value


@dataclasses.dataclass(frozen=True)
class PallasGridContext:
  grid: GridMappingGrid
  mapped_dims: tuple[int, ...]

  def size(self, axis: int) -> int | DynamicGridDim:
    valid_grid = tuple(self.grid)
    try:
      size = valid_grid[axis]
    except IndexError as e:
      raise ValueError(
          f"Axis {axis} is out of bounds for grid {self.grid}"
      ) from e
    return size


@dataclasses.dataclass
class PallasTracingEnv(threading.local):
  grid_context: PallasGridContext | None = None
  grid_env_stack: list[GridEnv] = dataclasses.field(default_factory=list)
  is_interpret_mode: bool = False
  dynamic_shapes: bool = False
  module_export_fn: Callable[[mlir.ir.Module], None] | None = None

_pallas_tracing_env = PallasTracingEnv()


def axis_frame() -> PallasGridContext:
  # This is like jax_core.axis_frame, except there should only ever be one
  # active PallasGridAxisName for a particular main_trace because we cannot
  # nest pallas_calls.
  env = _pallas_tracing_env
  assert env.grid_context is not None
  return env.grid_context


@dataclasses.dataclass(frozen=True)
class GridAxis:
  index: jax.Array
  size: int

# Stores the kernel execution position and the size along grid axes.
GridEnv = Sequence[GridAxis]

@contextlib.contextmanager
def grid_env(env: GridEnv) -> Iterator[None]:
  _pallas_tracing_env.grid_env_stack.append(env)
  try:
    yield
  finally:
   _pallas_tracing_env.grid_env_stack.pop()


def current_grid_env() -> GridEnv | None:
  if not _pallas_tracing_env.grid_env_stack:
    return None
  return _pallas_tracing_env.grid_env_stack[-1]


@contextlib.contextmanager
def interpret_mode_env(interpret_mode: bool) -> Iterator[None]:
  prev_interpret = _pallas_tracing_env.is_interpret_mode
  if interpret_mode:
    _pallas_tracing_env.is_interpret_mode = True
  try:
    yield
  finally:
    if interpret_mode:
      _pallas_tracing_env.is_interpret_mode = prev_interpret

def is_interpret_mode() -> bool:
  """Returns whether the kernel is executing in interpret mode."""
  return _pallas_tracing_env.is_interpret_mode


class Mapped:
  """Used as a block shape dimension to denote a mapped dimension.
  A mapped dimension behaves like `1` except it is squeezed from the block.
  See :ref:`pallas_blockspec` for more details.
  """
  def __repr__(self):
    return "Mapped"
mapped = Mapped()


@dataclasses.dataclass(frozen=True)
class Unblocked:
  padding: tuple[tuple[int, int], ...] | None = None

  def __repr__(self):
    return f"Unblocked(padding={self.padding})"
unblocked = Unblocked()


class Blocked:
  def __repr__(self):
    return "Blocked"
blocked = Blocked()


IndexingMode = Union[Blocked, Unblocked]


@dataclasses.dataclass
class BlockSpec:
  """Specifies how an array should be sliced for each invocation of a kernel.

  See :ref:`pallas_blockspec` for more details.
  """
  # An internal canonicalized version is in BlockMapping.
  block_shape: Sequence[int | None] | None = None
  index_map: Callable[..., Any] | None = None
  memory_space: Any | None = dataclasses.field(kw_only=True, default=None)
  indexing_mode: IndexingMode = dataclasses.field(kw_only=True, default=blocked)

  def to_block_mapping(
      self,
      origin: OriginStr,
      array_aval: jax_core.ShapedArray,
      *,
      # Inputs for the index_map
      index_map_avals: Sequence[jax_core.AbstractValue],
      index_map_tree: tree_util.PyTreeDef,
      grid: GridMappingGrid,
      mapped_dims: tuple[int, ...],
  ) -> BlockMapping:
    if self.index_map is None:
      index_map_func = lambda *args: (0,) * len(array_aval.shape)
    else:
      index_map_func = self.index_map
    if self.block_shape is None:
      block_shape = array_aval.shape
    else:
      block_shape = self.block_shape
      if len(array_aval.shape) != len(block_shape):
        raise ValueError(
            f"Block shape for {origin} (= {block_shape}) "
            "must have the same number of dimensions as the "
            f"array shape {array_aval.shape}."
        )

    unmapped_block_shape = tuple(s for s in block_shape if s is not None)
    block_array_aval = array_aval.update(shape=unmapped_block_shape)
    if isinstance(array_aval, jax_core.DShapedArray):
      # Get the "max" shape for the ragged array.
      block_array_aval = jax_core.ShapedArray(
          block_array_aval.shape,
          block_array_aval.dtype,
          block_array_aval.weak_type,
      )
    block_aval = AbstractMemoryRef(block_array_aval, self.memory_space)

    if (
        not jax_core.is_constant_shape(block_aval.shape)
        and not dynamic_shapes_export_enabled()
    ):
      raise ValueError(
          "shape polymorphism for Pallas does not support "
          "dynamically-shaped blocks. "
          f"Block spec for {origin} has block_shape: {block_aval.shape}"
      )

    flat_index_map_fun, index_map_out_tree_thunk = api_util.flatten_fun(
        lu.wrap_init(index_map_func), index_map_tree
    )
    debug = pe.tracing_debug_info(
        index_map_func,
        index_map_tree,
        index_map_out_tree_thunk,
        False,
        "pallas_call index_map",
    )
    index_map_src_info = NameAndSrcInfo.from_pallas_call(
        None, debug.func_src_info  # type: ignore
    )
    with tracing_grid_env(grid, mapped_dims):
      jaxpr, out_avals, consts, () = pe.trace_to_jaxpr_dynamic(
          flat_index_map_fun, index_map_avals, debug_info=debug
      )
    mapped_block_shape = tuple(mapped if s is None else s for s in block_shape)
    if len(out_avals) != len(block_shape):
      raise ValueError(
          f"Index map function {index_map_src_info} for "
          f"{origin} must return "
          f"{len(block_shape)} values to match {block_shape=}. "
          f"Currently returning {len(out_avals)} values."
      )
    for i, ov in enumerate(out_avals):
      if ov.shape or ov.dtype not in [jnp.int32, jnp.int64]:
        raise ValueError(
            f"Index map function {index_map_src_info} for "
            f"{origin} must return integer scalars. Output[{i}] has type "
            f"{ov}."
        )

    if consts:
      raise ValueError(
          f"Index map function {index_map_src_info} for "
          f"{origin} must not capture constants: {consts}"
      )

    array_aval_shape = _max_shape_from_aval(array_aval)

    mapping = BlockMapping(
        block_shape=mapped_block_shape,
        transformed_block_aval=block_aval,  # There are no transforms by default
        index_map_jaxpr=jax_core.ClosedJaxpr(jaxpr, consts),
        index_map_src_info=index_map_src_info,
        indexing_mode=self.indexing_mode,
        array_shape_dtype=jax.ShapeDtypeStruct(
            array_aval_shape, array_aval.dtype
        ),
        origin=origin,
    )
    mapping.check_invariants()
    return mapping

  replace = dataclasses.replace


class NoBlockSpec:
  def __repr__(self):
    return "NoBlockSpec"
no_block_spec = NoBlockSpec()


# A PyTree of BlockSpec | NoBlockSpec.
# BlockSpecTree = Sequence[BlockSpec | NoBlockSpec, ...] | NoBlockSpec
BlockSpecTree = Any


class MemoryRefTransform(Protocol):
  """Transforms a memory reference on load or store."""

  def undo(self, ref: TransformedRef) -> TransformedRef:
    raise NotImplementedError("Abstract evaluation not implemented.")


@dataclasses.dataclass(frozen=True)
class BlockMapping:
  """An internal canonicalized version of BlockSpec.

  See the `check_invariants` method for precise specification.
  """
  # TODO(apaszke,sharadmv): Replace mapped dims in block_shape with a transform.
  # After all, it's just indexing out singleton dimensions.
  block_shape: tuple[Mapped | int, ...]
  transformed_block_aval: AbstractMemoryRef
  index_map_jaxpr: jax_core.ClosedJaxpr
  index_map_src_info: NameAndSrcInfo
  indexing_mode: IndexingMode
  array_shape_dtype: jax.ShapeDtypeStruct  # The whole array
  origin: OriginStr
  transforms: Sequence[MemoryRefTransform] = ()

  def check_invariants(self) -> None:
    if not config.enable_checks.value: return

    unmapped_block_shape = tuple(s for s in self.block_shape if s is not mapped)
    assert unmapped_block_shape == self.ref_aval.shape, (
        self.block_shape, self.ref_aval.shape)
    assert len(self.block_shape) == len(self.array_shape_dtype.shape), (
        self.block_shape, self.array_shape_dtype
    )

    assert not self.index_map_jaxpr.consts
    assert len(self.block_shape) == len(self.index_map_jaxpr.out_avals), (
        self.block_shape,
        self.index_map_jaxpr.out_avals,
    )
    assert all(ov.shape == () and
               (ov.dtype == jnp.int32 or ov.dtype == jnp.int64)
               for ov in self.index_map_jaxpr.out_avals), (
               self.index_map_jaxpr.out_avals)

  def replace(self, **kwargs):
    new_self = dataclasses.replace(self, **kwargs)
    new_self.check_invariants()
    return new_self

  @property
  def block_aval(self) -> AbstractMemoryRef:
    # If you hit this, make sure you take transforms into account and use either
    # ref_aval or transformed_block_aval.
    assert not self.transforms, "Lowering failed to handle transforms"
    return self.transformed_block_aval

  @property
  def ref_aval(self) -> AbstractMemoryRef | TransformedRef:
    """Returns the abstract value of the Ref after transformations."""
    if not self.transforms:
      return self.transformed_block_aval
    ref = TransformedRef(self.transformed_block_aval, ())
    for transform in reversed(self.transforms):
      ref = transform.undo(ref)
    return ref

  def compute_start_indices_interpret(self, loop_idx, *args):
    discharged_jaxpr, discharged_consts = state_discharge.discharge_state(
        self.index_map_jaxpr.jaxpr, self.index_map_jaxpr.consts
    )
    jaxpr = jax_core.ClosedJaxpr(discharged_jaxpr, discharged_consts)
    block_indices_and_rest = jax_core.jaxpr_as_fun(jaxpr)(*loop_idx, *args)
    # Since we're passing in `Ref`s potentially, we need to split out their
    # updated values since we only care about the return values.
    block_indices, _ = split_list(block_indices_and_rest,
                                  [len(self.block_shape)])
    if isinstance(self.indexing_mode, Blocked):
      return tuple(i if b is mapped else b * i
                  for b, i in zip(self.block_shape, block_indices))
    elif isinstance(self.indexing_mode, Unblocked):
      return block_indices
    else:
      raise RuntimeError(f"Unknown indexing mode: {self.indexing_mode}")

  def has_trivial_window(self):
    """If block shape is same as the array shape and index_map returns 0s."""
    for b, s in zip(self.block_shape, self.array_shape_dtype.shape):
      if b != s and not (b is mapped and s == 1):
        return False
    for atom in self.index_map_jaxpr.jaxpr.outvars:
      if not (isinstance(atom, jax_core.Literal) and atom.val == 0):
        return False
    return True


@contextlib.contextmanager
def tracing_grid_env(grid: GridMappingGrid, mapped_dims: tuple[int, ...]):
  if dynamic_shapes_export_enabled():
    assert all(i is dynamic_grid_dim or jax_core.is_dim(i) for i in grid)
  else:
    assert all(i is dynamic_grid_dim or isinstance(i, int) for i in grid)
  old_grid_context = _pallas_tracing_env.grid_context
  try:
    _pallas_tracing_env.grid_context = PallasGridContext(grid, mapped_dims)
    yield
  finally:
    _pallas_tracing_env.grid_context = old_grid_context


@contextlib.contextmanager
def pallas_export_experimental(dynamic_shapes: bool):
  old_dynamic_shapes = _pallas_tracing_env.dynamic_shapes
  try:
    _pallas_tracing_env.dynamic_shapes = dynamic_shapes
    yield
  finally:
    _pallas_tracing_env.dynamic_shapes = old_dynamic_shapes


def dynamic_shapes_export_enabled() -> bool:
  return _pallas_tracing_env.dynamic_shapes


@dataclasses.dataclass(frozen=True)
class GridMapping:
  """An internal canonicalized version of GridSpec.

  Encodes the calling conventions of the pallas_call primitive, the kernel,
  and the index maps.

  The pallas_call is invoked with: ``*dynamic_grid_sizes, *index, *inputs``.
  The ``index`` operands are for the scalar prefetch.

  The kernel function is invoked with:
  ``*index, *inputs, *scratch``.

  The index map functions are invoked with:
  ``*program_ids, *index``.

  See the `check_invariants` method for a more precise specification.
  """
  grid: GridMappingGrid
  grid_names: tuple[Hashable, ...] | None

  # Block mappings for: *inputs, *outputs
  block_mappings: tuple[BlockMapping, ...]
  # The inputs for tracing the index map: the tree and the flat avals
  index_map_tree: tree_util.PyTreeDef
  index_map_avals: tuple[jax_core.AbstractValue]
  # Which dimensions in `grid` are vmapped.
  vmapped_dims: tuple[int, ...]

  num_index_operands: int
  num_inputs: int
  num_outputs: int
  num_scratch_operands: int
  get_grid_indices: Callable | None = None
  local_grid_env: Callable | None = None

  def check_invariants(self) -> None:
    if not config.enable_checks.value: return
    assert (len(self.block_mappings) == self.num_inputs + self.num_outputs), (
        self.num_inputs, self.num_outputs,
        self.block_mappings
    )
    # index_map_avals = int32[] * len(self.grid) + index_operands
    assert len(self.index_map_avals) == len(self.grid) + self.num_index_operands, (
        self.index_map_avals,
        self.grid,
        self.num_index_operands,
    )
    # Check that we can put together the avals and the tree.
    index_map_args, index_map_kwargs = self.index_map_tree.unflatten(
        self.index_map_avals)
    assert not index_map_kwargs
    assert len(index_map_args) >= len(self.grid)
    for i in range(len(self.grid)):
      index_map_arg = index_map_args[i]
      assert index_map_arg.shape == (), f"index_map_arg: {index_map_arg}"
      assert index_map_arg.dtype == jnp.int32, f"index_map_arg: {index_map_arg}"

    assert len(self.vmapped_dims) <= len(self.grid)
    for i in self.vmapped_dims:
      assert 0 <= i < len(self.grid)

    if self.grid_names is not None:
      assert len(self.grid) == len(self.grid_names), (self.grid, self.grid_names)

    for bm in self.block_mappings:
      bm.check_invariants()
      assert tuple(self.index_map_avals) == tuple(
          bm.index_map_jaxpr.in_avals
      ), (
          self.index_map_avals,
          "|",
          bm.index_map_jaxpr.in_avals,
      )

  def replace(self, **kwargs) -> GridMapping:
    new_self = dataclasses.replace(self, **kwargs)
    new_self.check_invariants()
    return new_self

  @property
  # TODO(necula): deprecate and then remove this property.
  def mapped_dims(self) -> tuple[int, ...]:
    return self.vmapped_dims

  @property
  def num_dynamic_grid_bounds(self):
    return sum(b is dynamic_grid_dim for b in self.grid)

  @property
  def static_grid(self) -> StaticGrid:
    if self.num_dynamic_grid_bounds:
      raise ValueError("Expected a grid with fully static bounds")
    return self.grid  # type: ignore

  @contextlib.contextmanager
  def trace_env(self):
    if self.grid_names is None:
      axis_env_ctx = contextlib.nullcontext()
    else:
      axis_env_ctx = jax_core.extend_axis_env_nd(
          zip(self.grid_names, self.grid)
      )
    with tracing_grid_env(self.grid, self.vmapped_dims), axis_env_ctx:
      yield

  @property
  def slice_index_ops(self):
    """Returns a slice object to select the index operands to a kernel.
    This works on a sequence that contains *index, *ins, *outs, *scratch.
    """
    return slice(0, self.num_index_operands)

  @property
  def slice_block_ops(self):
    """Returns a slice to select the block operands to a kernel.

    The block operands are: *ins, *outs, the same for which we
    have `self.block_mappings`.
    This works on a sequence that contains *index, *ins, *outs, *scratch.
    """
    return slice(self.num_index_operands,
                 self.num_index_operands + len(self.block_mappings))

  @property
  def slice_scratch_ops(self):
    """Returns a slice object to select the scratch operands to a kernel.
    This works on a sequence that contains *index, *ins, *outs, *scratch.
    """
    if self.num_scratch_operands:
      return slice(-self.num_scratch_operands, None)
    else:
      return slice(0, 0)

  @property
  def in_shapes(self) -> Iterable[jax.ShapeDtypeStruct]:
    """The shapes of *index, *inputs."""
    index_shapes = (
        jax.ShapeDtypeStruct(ia.shape, ia.dtype)
        for ia in self.index_map_avals[len(self.grid) :]
    )
    inputs_shapes = (
        bm.array_shape_dtype
        for bm in self.block_mappings[:self.num_inputs])
    return itertools.chain(index_shapes, inputs_shapes)

  @property
  def block_mappings_output(self) -> Iterable[BlockMapping]:
    return itertools.islice(
        self.block_mappings,
        self.num_inputs,
        self.num_inputs + self.num_outputs)

  @property
  def out_shapes(self) -> Iterable[jax.ShapeDtypeStruct]:
    return tuple(
        bm.array_shape_dtype for bm in self.block_mappings_output)


def _is_valid_grid_dim(dim: int | jax.Array) -> bool:
  if isinstance(dim, jax.Array):
    return True
  return jax_core.is_dim(dim)


def _max_shape_from_aval(array_aval: jax_core.ShapedArray):
  array_aval_shape = list(array_aval.shape)
  for i, s in enumerate(array_aval.shape):
    try:
      aval = jax_core.get_aval(s)
      if isinstance(aval, jax_core.DShapedArray):
        array_aval_shape[i] = aval.dtype.bound
    except OverflowError as e:
      # Note - there are annoying cases where on 32 bit hardware,
      # a flattened index space may overflow - for these cases,
      # we just take the shape as is.
      # In most places, this is totally sound to do.
      # For ragged/jumble inputs, this will fail downstream.
      return array_aval.shape

  return tuple(array_aval_shape)


def _convert_block_spec_to_block_mapping(
    block_spec: BlockSpec,
    origin: OriginStr,
    array_aval: jax_core.ShapedArray,
    *,
    # Inputs for the index_map
    index_map_avals: Sequence[jax_core.AbstractValue],
    index_map_tree: tree_util.PyTreeDef,
    grid: GridMappingGrid,
    mapped_dims: tuple[int, ...],
) -> BlockMapping:
  if block_spec is no_block_spec:
    block_spec = BlockSpec(None, None)
  return block_spec.to_block_mapping(
      origin,
      array_aval,
      index_map_avals=index_map_avals,
      index_map_tree=index_map_tree,
      grid=grid,
      mapped_dims=mapped_dims,
  )

index_map_grid_aval = jax_core.ShapedArray((), jnp.int32)


class ScratchShape(Protocol):
  def get_array_aval(self) -> jax_core.AbstractValue:
    ...
  def get_ref_aval(self) -> state.AbstractRef:
    ...


ScratchShapeTree = Sequence[Union[ScratchShape, "ScratchShapeTree"]]


@dataclasses.dataclass(init=False, kw_only=True)
class GridSpec:
  """Encodes the grid parameters for :func:`jax.experimental.pallas.pallas_call`.

  See the documentation for :func:`jax.experimental.pallas.pallas_call`,
  and also :ref:`pallas_grids_and_blockspecs` for a more detailed
  description of the parameters.
  """
  # A canonicalized internal version is in GridMapping.
  grid: TupleGrid
  grid_names: tuple[Hashable, ...] | None
  in_specs: BlockSpecTree
  out_specs: BlockSpecTree
  scratch_shapes: ScratchShapeTree = ()

  def __init__(
      self,
      grid: Grid = (),
      in_specs: BlockSpecTree = no_block_spec,
      out_specs: BlockSpecTree = no_block_spec,
      scratch_shapes: ScratchShapeTree = (),
  ):
    # Be more lenient for in/out_specs
    if isinstance(in_specs, list):
      in_specs = tuple(in_specs)
    elif in_specs is not no_block_spec and not isinstance(in_specs, Sequence):
      raise ValueError(f"`in_specs` must be a tuple or a list. Found: {in_specs}")
    if isinstance(out_specs, list):
      out_specs = tuple(out_specs)

    self.in_specs = in_specs
    self.out_specs = out_specs
    self.scratch_shapes = tuple(scratch_shapes)

    grid_names = None
    if isinstance(grid, int):
      grid = (grid,)
    elif grid and isinstance(grid[0], tuple):  # Check if we have a named grid
      grid_names, grid = util.unzip2(grid)  # type: ignore

    # TODO(b/353730556): allow NumPy scalars in grids
    if not all(_is_valid_grid_dim(g) for g in grid):  # type: ignore
      raise ValueError(
          f"Grid must be a tuple of integers or jax.Array, got {grid}"
      )
    self.grid = grid  # type: ignore
    self.grid_names = grid_names

  def _make_scalar_ref_aval(self, aval):
    assert False  # Not needed in GridSpec


def get_grid_mapping(
    grid_spec: GridSpec,
    in_avals: Sequence[jax_core.AbstractValue],
    in_tree: tree_util.PyTreeDef,
    in_origins: Sequence[OriginStr],
    out_avals: Sequence[jax_core.AbstractValue],
    out_tree: tree_util.PyTreeDef,
    out_origins: Sequence[OriginStr],
) -> tuple[tuple[jax_core.AbstractValue, ...],
           GridMapping]:
  if dynamic_shapes_export_enabled():
    dim_check : Any = jax_core.is_dim  # type: ignore[no-redef]
  else:
    dim_check : Any = jax_core.is_constant_dim  # type: ignore[no-redef]
  assert all(i is None or dim_check(i) for i in grid_spec.grid)
  grid_mapping_grid = tuple(
      dynamic_grid_dim if d is None else d for d in grid_spec.grid
  )
  # The inputs for the index maps
  index_map_avals = (
      (index_map_grid_aval.update(sharding=None),) * len(grid_spec.grid))
  index_map_tree = tree_util.tree_structure((index_map_avals, {}))

  num_scalar_prefetch: int = getattr(grid_spec, "num_scalar_prefetch", 0)
  if num_scalar_prefetch:
    all_avals = tree_util.tree_unflatten(in_tree, in_avals)
    scalar_avals, unflat_in_avals = split_list(
        all_avals, [num_scalar_prefetch])
    flat_scalar_avals, scalar_tree = tree_util.tree_flatten(scalar_avals)
    num_flat_scalar_prefetch = len(flat_scalar_avals)
    scalar_ref_avals = [
        grid_spec._make_scalar_ref_aval(aval)
        for aval in flat_scalar_avals]
    jaxpr_scalar_ref_avals = tree_util.tree_unflatten(
        scalar_tree, scalar_ref_avals)
    in_avals, in_tree = tree_util.tree_flatten(tuple(unflat_in_avals))
    index_map_tree = tree_util.tree_structure(((*index_map_avals,
                                                *scalar_avals), {}))
    index_map_avals = (*index_map_avals, *scalar_ref_avals)
    del scalar_ref_avals, flat_scalar_avals, scalar_tree
    del scalar_avals, unflat_in_avals, all_avals
  else:
    num_flat_scalar_prefetch = 0
    jaxpr_scalar_ref_avals = ()
  if grid_spec.scratch_shapes:
    flat_scratch_shapes, scratch_tree = tree_util.tree_flatten(
        grid_spec.scratch_shapes)
    flat_scratch_avals = map(lambda s: s.get_ref_aval(), flat_scratch_shapes)
    num_flat_scratch_operands = len(flat_scratch_avals)
    jaxpr_scratch_avals = tree_util.tree_unflatten(
        scratch_tree, flat_scratch_avals)
    if not isinstance(jaxpr_scratch_avals, (tuple, list)):
      jaxpr_scratch_avals = (jaxpr_scratch_avals,)
    del flat_scratch_avals, flat_scratch_shapes, scratch_tree
  else:
    num_flat_scratch_operands = 0
    jaxpr_scratch_avals = ()

  if grid_spec.in_specs is not no_block_spec:
    flat_in_specs, in_specs_tree = tree_util.tree_flatten(grid_spec.in_specs)
    if in_specs_tree != in_tree:
      raise ValueError(
          pytreedef_mismatch_err_msg("`in_specs`", in_specs_tree,
                                     "inputs", in_tree))
  else:
    flat_in_specs = [no_block_spec] * len(in_avals)

  in_block_mappings = map(
      partial(
          _convert_block_spec_to_block_mapping,
          index_map_avals=index_map_avals,
          index_map_tree=index_map_tree,
          grid=grid_mapping_grid,  # type: ignore[arg-type]
          mapped_dims=(),
      ),
      flat_in_specs,
      in_origins[num_flat_scalar_prefetch:],
      in_avals,
  )

  if grid_spec.out_specs is not no_block_spec:
    flat_out_specs, out_specs_tree = tree_util.tree_flatten(grid_spec.out_specs)
    if out_specs_tree != out_tree:
      raise ValueError(
          pytreedef_mismatch_err_msg("`out_specs`", out_specs_tree,
                                     "`out_shape`", out_tree))
  else:
    flat_out_specs = [no_block_spec] * len(out_avals)

  out_block_mappings = map(
      partial(
          _convert_block_spec_to_block_mapping,
          index_map_avals=index_map_avals,
          index_map_tree=index_map_tree,
          grid=grid_mapping_grid,  # type: ignore[arg-type]
          mapped_dims=(),
      ),
      flat_out_specs,
      out_origins,
      out_avals,
  )
  grid_mapping = GridMapping(
      grid=grid_mapping_grid,  # type: ignore[arg-type]
      grid_names=grid_spec.grid_names,
      block_mappings=(*in_block_mappings, *out_block_mappings),
      index_map_avals=index_map_avals,  # type: ignore[arg-type]
      index_map_tree=index_map_tree,
      vmapped_dims=(),
      num_index_operands=num_flat_scalar_prefetch,
      num_inputs=len(flat_in_specs),
      num_outputs=len(flat_out_specs),
      num_scratch_operands=num_flat_scratch_operands,
  )
  grid_mapping.check_invariants()
  in_ref_avals = [bm.ref_aval for bm in in_block_mappings]
  jaxpr_in_ref_avals = tree_util.tree_unflatten(in_tree, in_ref_avals)
  jaxpr_in_avals = (*jaxpr_scalar_ref_avals,
                    *jaxpr_in_ref_avals)
  out_ref_avals = [bm.ref_aval for bm in out_block_mappings]
  jaxpr_out_avals = tree_util.tree_unflatten(out_tree, out_ref_avals)
  if not isinstance(jaxpr_out_avals, (tuple, list)):
    jaxpr_out_avals = (jaxpr_out_avals,)
  return (*jaxpr_in_avals, *jaxpr_out_avals,
          *jaxpr_scratch_avals), grid_mapping


def unzip_dynamic_grid_bounds(
    grid_spec: GridSpec) -> tuple[GridSpec, tuple[Any, ...]]:
  if dynamic_shapes_export_enabled():
    new_grid : Any = grid_spec.grid  # type: ignore[no-redef]
  else:
    new_grid : Any = tuple(d if isinstance(d, int) else None for d in grid_spec.grid)  # type: ignore[no-redef]
  dynamic_bounds = tuple(d for d in grid_spec.grid if not isinstance(d, int))
  # We can't use dataclasses.replace, because our fields are incompatible
  # with __init__'s signature.
  static_self = copy.copy(grid_spec)
  static_self.grid = new_grid  # type: ignore
  return static_self, dynamic_bounds


def pytreedef_mismatch_err_msg(
    what1: str, tree1: tree_util.PyTreeDef,
    what2: str, tree2: tree_util.PyTreeDef) -> str:
  errs = list(tree_util.equality_errors_pytreedef(tree1, tree2))
  msg = []
  msg.append(
      f"Pytree for {what1} and {what2} do not match. "
      f"There are {len(errs)} mismatches, including:")
  for path, thing1, thing2, explanation in errs:
    where = f"at {tree_util.keystr(path)}, " if path else ""
    msg.append(
        f"    * {where}{what1} is a {thing1} but"
        f" {what2} is a {thing2}, so {explanation}")
  return "\n".join(msg)


@dataclasses.dataclass(frozen=True)
class CostEstimate:
  flops: int
  transcendentals: int
  bytes_accessed: int

  def to_json(self) -> bytes:
    return (
        f'{{"flops": {self.flops}, "transcendentals": {self.transcendentals},'
        f' "bytes_accessed": {self.bytes_accessed}}}'
    ).encode("ascii")


core_map_p = jax_core.Primitive("core_map")
core_map_p.multiple_results = True


def core_map(
    mesh,
    *,
    compiler_params: Any | None = None,
    interpret: bool = False,
    debug: bool = False,
    cost_estimate: CostEstimate | None = None,
):
  """Runs a function on a mesh, mapping it over the devices in the mesh.

  The function should be stateful in that it takes in no inputs and returns
  no outputs but can mutate closed-over Refs, for example.

  Args:
    mesh: The mesh to run the function on.
    compiler_params: The compiler parameters to pass to the backend.
    interpret: Whether to run the function in interpret mode.
    debug: Whether or not to out helpful debugging information.
    cost_estimate: The cost estimate of the function.
  """
  def wrapped(f):
    flat_args, in_tree = tree_util.tree_flatten(((), {}))
    flat_fun, out_tree_thunk = api_util.flatten_fun(lu.wrap_init(f), in_tree)
    with jax_core.extend_axis_env_nd(mesh.shape.items()):
      jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(flat_fun, flat_args)
    out = core_map_p.bind(*consts, jaxpr=jaxpr, mesh=mesh,
                          compiler_params=compiler_params,
                          interpret=interpret,
                          debug=debug,
                          cost_estimate=cost_estimate)
    if out:
      raise ValueError("core_map-ped functions must not return any outputs.")
    return tree_util.tree_unflatten(out_tree_thunk(), out)
  return wrapped


@core_map_p.def_effectful_abstract_eval
def _core_map_abstract_eval(*args, jaxpr, mesh, **_):
  del args
  if jaxpr.outvars:
    raise ValueError("core_map must not return any outputs.")
  effs = set()
  for eff in jaxpr.effects:
    if mesh.discharges_effect(eff):
      continue
    if not isinstance(eff, jax_core.NamedAxisEffect):
      effs.add(eff)
      continue
    if eff.name not in mesh.shape:
      effs.add(eff)
  return [], effs


_core_map_mesh_rules: dict[type[Any], Callable[..., Any]] = {}


def default_mesh_discharge_rule(
    in_avals,
    out_avals,
    *args,
    grid,
    compiler_params,
    backend,
    jaxpr,
    debug,
    interpret,
    cost_estimate,
):
  """Discharges a ``core_map`` over a mesh to a ``pallas_call``."""
  del out_avals  # Unused.

  def body(*args):
    # Due to aliasing, ``args`` contains aliased inputs and outputs so we
    # remove outputs.
    in_refs = args[:len(in_avals)]
    jax_core.eval_jaxpr(jaxpr, in_refs)

  assert len(jaxpr.outvars) == 0
  modified_idxs = sorted(
      eff.input_index
      for eff in jaxpr.effects
      if isinstance(eff, state_types.WriteEffect)
  )
  any_spec = BlockSpec(memory_space=MemorySpace.ANY)
  from jax._src.pallas import pallas_call  # Avoid circular dependency.
  outs = pallas_call.pallas_call(
      body,
      out_shape=[in_avals[idx] for idx in modified_idxs],
      in_specs=[any_spec] * len(in_avals),
      out_specs=[any_spec] * len(modified_idxs),
      input_output_aliases={
          in_idx: out_idx for out_idx, in_idx in enumerate(modified_idxs)
      },
      grid=grid,
      compiler_params=compiler_params,
      backend=backend,
      interpret=interpret,
      debug=debug,
      cost_estimate=cost_estimate,
  )(*args)
  # ``outs`` lacks the unmodified inputs. Add them back in.
  all_outs = [None] * len(args)
  for out_idx, in_idx in enumerate(modified_idxs):
    all_outs[in_idx] = outs[out_idx]
  return all_outs, ()


@state_discharge.register_discharge_rule(core_map_p)
def _core_map_discharge_rule(in_avals, out_avals, *args_flat, jaxpr, mesh, **kwargs):
  if type(mesh) not in _core_map_mesh_rules:
    raise NotImplementedError(f"Mesh type {type(mesh)} not supported.")
  return _core_map_mesh_rules[type(mesh)](
      in_avals, out_avals, *args_flat, jaxpr=jaxpr, mesh=mesh, **kwargs
  )


def _core_map_typecheck_rule(_, *in_atoms, jaxpr, mesh, **kwargs):
  del in_atoms, kwargs
  with jax_core.extend_axis_env_nd(tuple(mesh.shape.items())):
    jax_core.check_jaxpr(jaxpr)
  effs = set()
  for eff in jaxpr.effects:
    if mesh.discharges_effect(eff):
      continue
    if not isinstance(eff, jax_core.NamedAxisEffect):
      effs.add(eff)
      continue
    if eff.name not in mesh.shape:
      effs.add(eff)
  return [], effs
jax_core.custom_typechecks[core_map_p] = _core_map_typecheck_rule


def lower_as_mlir(f, *args, dynamic_shapes=False, **kwargs) -> mlir.ir.Module:
  with pallas_export_experimental(dynamic_shapes):
    lowered = jax.jit(f).lower(*args, **kwargs)
    stablehlo = lowered.compiler_ir(dialect="stablehlo")  # type: ignore[return-value]

  return stablehlo  # type: ignore[return-value]
