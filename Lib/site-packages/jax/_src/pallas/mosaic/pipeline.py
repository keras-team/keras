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

"""Module for emitting custom TPU pipelines within a Pallas call."""
from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
import dataclasses
import enum
import functools
import itertools
import operator
from typing import Any, Union

import jax
from jax import lax
from jax import tree_util
from jax._src import util as jax_util
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives as primitives
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import primitives as tpu_primitives
from jax.experimental import pallas as pl
from jax.extend.backend import get_default_device
import jax.numpy as jnp
import numpy as np


SMEM = tpu_core.TPUMemorySpace.SMEM
VMEM = tpu_core.TPUMemorySpace.VMEM
DMA = tpu_core.SemaphoreType.DMA
REF = pallas_core.MemoryRef
SemaphoreType = tpu_core.SemaphoreType
SemaphoreTuple = jax.Array
ArrayRef = Union[REF, jax.Array]

GridIndices = tuple[jax.Array, ...]
CondVal = Union[jax.Array, bool]
PipelineBlockSpecs = Union[Sequence[pallas_core.BlockSpec], Any]
PipelineRefs = Union[Sequence[REF], Any]


# TODO(sharadmv): make this a parameter and make it queryable from the Device.
_TILING = (8, 128)

def _broadcast_pytree_to(from_pytree, to_pytree):
  """Broadcast a prefix pytree to a given full tree."""
  proxy = object()
  treedef = tree_util.tree_structure(to_pytree)
  broadcast_leaves = []
  def add_leaves(i, x):
    broadcast_leaves.extend(
        [i] * tree_util.tree_structure(x).num_leaves)
  try:
    tree_util.tree_map(add_leaves, from_pytree, to_pytree,
                       is_leaf=lambda x: x is None)
  except ValueError:
    raise ValueError(f"Cannot broadcast tree {from_pytree} "
                     f"to full tree structure {treedef}.") from None
  broadcast_leaves = [None if a is proxy else a for a in broadcast_leaves]
  assert len(broadcast_leaves) == treedef.num_leaves
  return tree_util.tree_unflatten(treedef, broadcast_leaves)


@jax_util.cache(trace_context_in_key=False)
def _get_tpu_generation() -> int:
  kind = get_default_device().device_kind
  if kind.endswith(' lite'):
    kind = kind[:-len(' lite')]
  assert kind[:5] == "TPU v", kind
  return int(kind[5])

def _make_tiling(shape: tuple[int, ...], dtype: np.dtype) -> tuple[int, ...]:
  # For a n-dimensional shape, returns (8, 128) for the last 2 dimensions
  # and 1 for the leading n - 2. For example, (256, 256) -> (8, 128) and
  # (2, 3, 128, 128) -> (1, 1, 8, 128).
  if len(shape) < 2:
    raise ValueError(f"Shape must have at least 2 dimensions: {shape=}")
  leading_dims, final_dims = shape[:-2], shape[-2:]
  # We want to find the minimum power of 2 that fits the second-minor dimension
  # of shape, with maximum value 8.
  second_minor, _ = final_dims
  packing = 4 // dtype.itemsize
  max_tiling = _TILING[0]
  second_minor_tiling = (1 + int(_get_tpu_generation() < 4)) * packing
  while second_minor_tiling < min(second_minor, max_tiling):
    second_minor_tiling *= 2
  return (*(1,) * len(leading_dims), second_minor_tiling, _TILING[1])


def _round_up_to_nearest_multiple(s: int, multiple: int) -> int:
  if s % multiple == 0:
    return s
  # Subtract off the remainder, then add multiple
  return s - s % multiple + multiple


def _make_ds(
    idx: jax.Array | int, size: jax.Array | int
) -> pl.Slice:
  """Make a DMA slice with mosaic size hints."""
  out = pl.ds(idx * size, size)
  assert isinstance(out, pl.Slice)
  return out


def _make_block_slice(
    block_index: jax.Array, block_size: int, size: int, tiling: int
) -> pl.Slice | slice:
  # Computes a slice given a block index and block size. In the default case,
  # we return slice(block_index * block_size, (block_index + 1) * block_size).
  # However, if the total size of the ref does not divide block size and we are
  # selecting the last block, we need to pick the lowest tiling size multiple
  # that contains the block.
  if size % block_size == 0:
    return _make_ds(block_index, block_size)
  if block_size % tiling != 0:
    raise ValueError(f"Block size must divide tiling: {block_size=}, {tiling=}")
  num_blocks = pl.cdiv(size, block_size)
  is_last = block_index == num_blocks - 1
  rounded_size = jnp.where(
      is_last,
      _round_up_to_nearest_multiple(size % block_size, tiling),
      block_size,
  )
  rounded_size = pl.multiple_of(rounded_size, tiling)
  return pl.ds(block_index * block_size, rounded_size)


def _tuples_differ(xs, ys):
  """Dynamic index-tuple comparison calculation."""
  differences = jax.tree.map(lambda x, y: x != y, xs, ys)
  return functools.reduce(lambda x, y: x | y, differences, False)


def _grid_size(grid):
  """Dynamic grid size calculation."""
  size = jnp.array(1, jnp.int32)
  for dim in grid:
    size *= dim
  return size


def _get_indices(step, grid, offsets):
  """Get indices for a given step and grid."""
  # TODO(enriqueps): Implement using bitwise ops, avoid div/rem since they are
  # expensive.
  extended_grid = grid + (1,)
  strides = tuple(
      itertools.accumulate(extended_grid[::-1], func=operator.mul))[::-1]
  indices = tuple(
      lax.div(lax.rem(step, a), b)
      for a, b in zip(strides[:-1], strides[1:])
  )
  return tuple(a + b for a, b in zip(indices, offsets, strict=True))


class BufferType(enum.Enum):
  """Buffer type for the arguments to an emitted pipeline."""
  INPUT = 1
  OUTPUT = 2
  ACCUMULATOR = 3
  INPUT_OUTPUT = 4

  MANUAL = 5


@tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class BufferedRef:
  """A helper class to automate VMEM double buffering in pallas pipelines.

  Attributes:
    spec: pallas blockspec.
    dtype: dtype for buffers.
    buffer_type: enum indicating whether this is an input, output, or in/out
      accumulator buffered reference.
    window_ref: a double-buffer to hold a working buffer and a dirty buffer used
      to copy into and out of.  In the case of a BufferedRef targeting a VMEM
      reference, this simply points to the existing ref.
    accum_ref: accumulating buffer used by accumulator BufferedRefs.
    current_slot: current slot index to the working buffer.
    next_slot: slot that will point to the working buffer in the next iteration.
    sem_recvs: Double buffered semaphores for input DMAs.
    sem_sends: Double buffered semaphores for output DMAs.
    block_shape: passthrough property for the BlockSpec's block_shape.
    compute_index: passthrough property for the BlockSpec's compute_index.
    memory_space: passthrough property for the BlockSpec's memory_space.
    current_ref: points to the current working slice of the double-buffer.
    is_input: whether this BufferedRef acts as a pipeline input.
    is_output: whether this BufferedRef acts as a pipeline output.
    is_accumulator: whether this BufferedRef is an accumulator.
    is_input_output: whether this BufferedRef is an input/output without
      automatic accumulation.
  """
  spec: pl.BlockSpec       # static metadata
  dtype: Any               # static metadata
  buffer_type: BufferType  # static metadata
  window_ref: REF | None
  accum_ref: REF | None
  current_slot: ArrayRef | None
  next_slot: ArrayRef | None
  sem_recvs: SemaphoreTuple | None
  sem_sends: SemaphoreTuple | None

  def tree_flatten(self):
    return (
        (
            self.window_ref,
            self.accum_ref,
            self.current_slot,
            self.next_slot,
            self.sem_recvs,
            self.sem_sends,
        ),
        (self.spec, self.dtype, self.buffer_type),
    )

  @classmethod
  def tree_unflatten(cls, meta, data):
    return cls(*meta, *data)

  @staticmethod
  def buffer_types() -> type[BufferType]:
    return BufferType

  @classmethod
  def create(cls, spec, dtype, buffer_type) -> BufferedRef:
    """Create a BufferedRef.

    Args:
      spec: pallas blockspec.
      dtype: dtype for buffers.
      buffer_type: enum indicating whether this is an input, output, or in/out
        accumulator buffered reference.

    Returns:
      Initialized BufferedRef
    """
    block_shape = tuple(1 if x is None else x for x in spec.block_shape)
    if buffer_type is BufferType.ACCUMULATOR:
      accum_ref = VMEM(block_shape, dtype)
    else:
      accum_ref = None
    if spec.memory_space == VMEM:
      # We don't need to do any double-buffering in the case that our pipeline
      # reference is already in VMEM, we just need allocate the accumulation
      # buffer and we will refer to the original reference slices directly.
      return cls(
          spec=spec,
          dtype=dtype,
          buffer_type=buffer_type,
          window_ref=None,  # to be bound to existing ref by the pipeline routine
          accum_ref=accum_ref,
          current_slot=None,
          next_slot=None,
          sem_recvs=None,
          sem_sends=None,
      )
    else:
      memory_space = SMEM if spec.memory_space == SMEM else VMEM
      return cls(
          spec=spec,
          dtype=dtype,
          buffer_type=buffer_type,
          window_ref=memory_space((2,) + block_shape, dtype),
          accum_ref=accum_ref,
          current_slot=SMEM((1,), jnp.int32),
          next_slot=SMEM((1,), jnp.int32),
          sem_recvs=(
              None
              if buffer_type is BufferType.OUTPUT
              else SemaphoreType.DMA((2,))
          ),
          sem_sends=(
              None
              if buffer_type is BufferType.INPUT
              else SemaphoreType.DMA((2,))
          ),
      )

  @classmethod
  def input(cls, spec, dtype):
    return cls.create(spec, dtype, BufferType.INPUT)

  @classmethod
  def output(cls, spec, dtype):
    return cls.create(spec, dtype, BufferType.OUTPUT)

  @classmethod
  def accumulator(cls, spec, dtype):
    return cls.create(spec, dtype, BufferType.ACCUMULATOR)

  @classmethod
  def input_output(cls, spec, dtype):
    return cls.create(spec, dtype, BufferType.INPUT_OUTPUT)

  @property
  def block_shape(self):
    return self.spec.block_shape

  @property
  def compute_index(self):
    return self.spec.index_map

  @property
  def memory_space(self):
    return self.spec.memory_space

  @property
  def current_ref(self):
    buffer_slice = tuple(
        0 if x is None else slice(None) for x in self.block_shape)
    if self.memory_space == VMEM:
      return self.window_ref.at[buffer_slice]
    else:
      return self.window_ref.at[(self.current_slot[0], *buffer_slice)]

  @property
  def is_input(self):
    return self.buffer_type in [
        BufferType.INPUT,
        BufferType.ACCUMULATOR,
        BufferType.INPUT_OUTPUT,
    ]

  @property
  def is_output(self):
    return self.buffer_type in [
        BufferType.OUTPUT,
        BufferType.ACCUMULATOR,
        BufferType.INPUT_OUTPUT,
    ]

  @property
  def is_accumulator(self):
    return self.buffer_type == BufferType.ACCUMULATOR

  @property
  def is_input_output(self):
    return self.buffer_type == BufferType.INPUT_OUTPUT

  def bind_existing_ref(self, window_ref, indices):
    """For handling VMEM references, the pipeline aliases the existing ref."""
    if self.memory_space == VMEM:
      return dataclasses.replace(
          self, window_ref=window_ref.at[self.compute_slice(indices)]
      )
    return self

  def compute_slice(self, grid_indices):
    """Compute DMA slice from grid indices."""
    block_shape = tuple(1 if x is None else x for x in self.block_shape)
    indices = self.compute_index(*grid_indices)
    return jax.tree.map(_make_ds, indices, block_shape)

  def init_slots(self):
    """Initialize slot indices."""
    if self.memory_space == VMEM: return
    self.current_slot[0] = 0
    self.next_slot[0] = 0

  def swap_slots(self):
    """Switch to the next slot."""
    if self.memory_space == VMEM: return
    self.current_slot[0] = self.next_slot[0]

  def get_dma_slice(self, src_shape, src_dtype, grid_indices):
    # We need to handle blocks that might go OOB in the src array. An in bounds
    # block looks like this (for array shape (600, 600) and block shape
    # (256, 256)):
    #
    # +--------------+------------------|
    # | Block (0,0)  |                  |
    # | (256, 256)   |                  |
    # +--------------+                  |
    # |    A (600, 600)                 |
    # |                                 |
    # +---------------------------------+
    #
    # For in-bounds blocks, we don't need to do anything special.
    # An out-of-bounds block looks like this:
    #
    # +--------------+------------------|
    # |                                 |
    # |                                 |
    # +                                 |
    # |    A (600, 600)                 |
    # +--------------+                  |
    # | Block (2,0)  |                  |
    # + --------------------------------|
    # | XXXXXXXXXX   |
    # +--------------+
    # where the X's indicate where the block is out of bounds.
    #
    # When we have an out of bounds block like this, we need to truncate it to
    # a tile boundary (tiles are (8, 128) along the two minormost dimensions).
    # In this case, we'll have a block that is indexing the
    # 512:768 elements of A along the first dimension. We need to convert 768
    # into 600 (600 % 8 == 0), so our indexing will look like this:

    # +--------------+------------------|
    # |                                 |
    # |                                 |
    # +                                 |
    # |    A (600, 600)                 |
    # +--------------+                  |
    # | Block (2,0)  |                  |
    # + --------------------------------|
    # where it is now a (88, 256) sized block.
    #
    # Suppose A is now (601, 600), instead of picking a (88, 256)-sized block
    # for the last iteration on that dimension, we will pick the next highest
    # tile multiple, i.e. (96, 256).
    if len(src_shape) < 2:
      raise NotImplementedError("Must use >1D values.")

    tiling = _make_tiling(src_shape, src_dtype)
    block_shape = tuple(1 if b is None else b for b in self.block_shape)
    block_indices = self.compute_index(*grid_indices)
    return jax.tree.map(
        _make_block_slice, block_indices, block_shape, src_shape, tiling
    )

  def copy_in(self, src_ref, grid_indices):
    """Starts copy of HBM dma slice into the current slot."""
    assert self.is_input
    if self.memory_space == VMEM: return
    next_slot = lax.rem(self.current_slot[0] + 1, 2)
    self.next_slot[0] = next_slot
    src_slice = self.get_dma_slice(src_ref.shape, src_ref.dtype, grid_indices)
    dst_slice = tuple(pl.ds(0, s.size) for s in src_slice)
    tpu_primitives.make_async_copy(
        src_ref.at[src_slice],
        self.window_ref.at[next_slot].at[dst_slice],
        self.sem_recvs.at[next_slot],
    ).start()

  def copy_out(self, dst_ref, grid_indices):
    """Starts copy of HBM dma slice from the current slot."""
    assert self.is_output
    if self.memory_space == VMEM: return
    slot = self.current_slot[0]
    self.next_slot[0] = lax.rem(slot + 1, 2)
    dst_slice = self.get_dma_slice(dst_ref.shape, dst_ref.dtype, grid_indices)
    src_slice = tuple(pl.ds(0, s.size) for s in dst_slice)
    tpu_primitives.make_async_copy(
        self.window_ref.at[slot].at[src_slice],
        dst_ref.at[dst_slice],
        self.sem_sends.at[slot],
    ).start()

  def wait_in(self, src_ref, grid_indices):
    """Waits for input copy to finish."""
    assert self.is_input
    if self.memory_space == VMEM: return
    src_slice = self.get_dma_slice(src_ref.shape, src_ref.dtype, grid_indices)
    dst_slice = tuple(pl.ds(0, s.size) for s in src_slice)
    current_slot = self.current_slot[0]
    tpu_primitives.make_async_copy(
        src_ref.at[src_slice],  # nb: doesn't matter
        self.window_ref.at[current_slot].at[
            dst_slice
        ],  # only dst shape is important
        self.sem_recvs.at[current_slot],
    ).wait()

  def wait_out(self, dst_ref, grid_indices):
    """Waits for output copy to finish."""
    assert self.is_output
    if self.memory_space == VMEM: return
    prev_slot = lax.rem(self.current_slot[0] + 1, 2)
    dst_slice = self.get_dma_slice(dst_ref.shape, dst_ref.dtype, grid_indices)
    src_slice = tuple(pl.ds(0, s.size) for s in dst_slice)
    tpu_primitives.make_async_copy(
        self.window_ref.at[prev_slot].at[src_slice],  # nb: doesn't matter
        dst_ref.at[dst_slice],  # only dst shape is important
        self.sem_sends.at[prev_slot],
    ).wait()

  # Accumulator methods
  #
  # Accumulating inline in VMEM saves half the HBM<->VMEM bandwidth cost of
  # doing another full loop around HBM to do a reduction, at the current cost
  # of allocating another VMEM buffer.
  #
  # NB: there's no actual need to have an additional accumulation buffer, if
  # we just rewrote inner kernels to handle the initial-zero-init and output
  # reduction, we don't need to waste VMEM.  Consider removing this magic
  # init and reduce support.

  def set_accumulator(self, init=False):
    """Set accumulator or zero it out to initialize."""
    assert self.is_accumulator
    if self.accum_ref is not None:
      def _init():
        self.accum_ref[...] = jnp.zeros_like(self.accum_ref[...])
      def _set():
        self.accum_ref[...] = self.current_ref[...].astype(self.accum_ref.dtype)
      lax.cond(init, _init, _set)

  def accumulate(self):
    """Add into the current slot."""
    assert self.is_accumulator
    if self.accum_ref is not None:
      accum_dtype = jnp.float32
      if self.window_ref.dtype == jnp.int32:
        accum_dtype = jnp.int32
      # TODO(levskaya): we could generalize init and reduction functions,
      # could it ever be useful to support more generic monoids?
      self.current_ref[...] = (
          self.current_ref[...].astype(accum_dtype)
          + self.accum_ref[...].astype(accum_dtype)
      ).astype(self.window_ref.dtype)


# Helper to tree map over BufferedRefs as leaves.
map_brefs = functools.partial(
    jax.tree.map,
    is_leaf=lambda x: isinstance(x, BufferedRef))


def _filter_indices(
    indices: tuple[int | jax.Array, ...], grid: tuple[int | jax.Array, ...]
) -> tuple[int | jax.Array, ...]:
  return tuple(
      0 if isinstance(g, int) and g == 1 else i
      for i, g in zip(indices, grid, strict=True)
  )


def _next_index(
    indices: tuple[int | jax.Array, ...], grid: tuple[int | jax.Array, ...]
) -> tuple[int | jax.Array, ...]:
  out = []
  carry: bool | jax.Array = True
  for i, g in reversed(list(zip(indices, grid, strict=True))):
    inc = jax.lax.select(carry, i + 1, i)
    carry = inc == g
    out.append(jax.lax.select(carry, 0, inc))
  return _filter_indices(tuple(reversed(out)), grid)


def _prev_index(
    indices: tuple[int | jax.Array, ...], grid: tuple[int | jax.Array, ...]
) -> tuple[int | jax.Array, ...]:
  out = []
  borrow: bool | jax.Array = True
  for i, g in reversed(list(zip(indices, grid, strict=True))):
    dec = jax.lax.select(borrow, i - 1, i)
    borrow = dec == -1
    out.append(jax.lax.select(borrow, g - 1, dec))
  return _filter_indices(tuple(reversed(out)), grid)


class Scheduler:
  """Sequences input and output copies and waits for a pipeline."""

  def __init__(
      self,
      step: jax.Array,
      indices: tuple[int | jax.Array, ...],
      grid: tuple[int | jax.Array, ...],
      grid_offsets: tuple[int | jax.Array, ...],
      first_cycle=None,
      last_cycle=None,
      init_accumulators=None,
      trace_scopes=True,
  ):
    """Initializes scheduler.

    Args:
      step: inner step number.
      indices: current grid indices.
      grid: pallas grid for BufferedRefs.
      grid_offsets: offsets for grid indices (used for megacore).
      first_cycle: whether this is the first invocation of the pipeline.
      last_cycle: whether this is the last invocation of the pipeline.
      init_accumulators: do we zero-initialize accumulator state for this
        invocation of the pipeline.
      trace_scopes: whether to use named_scope to trace blocks in the pipeline.
    """
    self.step = step
    self.grid = grid
    self.first_cycle = first_cycle
    self.last_cycle = last_cycle
    self.init_accumulators = init_accumulators
    self.trace_scopes = trace_scopes

    # Total number of linear steps.
    self.num_steps = _grid_size(grid)

    # First and last inner step conditionals.
    self.first_step = step == 0
    self.last_step = step == self.num_steps - 1

    # First and last total step conditionals.
    self.first_step_ever = first_cycle & self.first_step
    self.last_step_ever = last_cycle & self.last_step

    # Derived grid indices for present, previous, and next steps.
    self.indices = tuple(
        i + j for i, j in zip(indices, grid_offsets, strict=True)
    )

    self.prev_indices = tuple(
        i + j
        for i, j in zip(_prev_index(indices, grid), grid_offsets, strict=True)
    )
    self.next_indices = tuple(
        i + j
        for i, j in zip(_next_index(indices, grid), grid_offsets, strict=True)
    )

  @contextmanager
  def _named_scope(self, name):
    if self.trace_scopes:
      with jax.named_scope(name):
        yield
    else:
      yield

  def grid_env(self):
    return pallas_core.grid_env(
        list(map(pallas_core.GridAxis, self.indices, self.grid)))

  def has_changed(self, buffered_ref):
    indices = buffered_ref.compute_index(*self.indices)
    prev_indices = buffered_ref.compute_index(*self.prev_indices)
    return _tuples_differ(indices, prev_indices)

  def will_change(self, buffered_ref):
    indices = buffered_ref.compute_index(*self.indices)
    next_indices = buffered_ref.compute_index(*self.next_indices)
    return _tuples_differ(indices, next_indices)

  def alias_local_refs(self, buffered_ref, ref):
    return buffered_ref.bind_existing_ref(ref, self.indices)

  # SCHEDULE ----------------------------------------------------------------

  # Below is the sequence of conditional waits and copies used for inputs,
  # outputs, and in-out accumulators.

  def initialize(self, buffered_ref, src_ref, schedule=None):
    if schedule is None:
      schedule = _default_schedule
    pred = schedule["prologue_copy_in"](self, buffered_ref, src_ref)

    with self._named_scope("ep_initialize"):
      @pl.when(self.first_step_ever)
      def _init_slots():
        buffered_ref.init_slots()

      @pl.when(pred)
      def _start():
        if buffered_ref.is_input:
          buffered_ref.copy_in(src_ref, self.indices)

      # In the prologue this makes it so we wait on the prologue copy to finish.
      # In other iterations this is the regular swap.
      buffered_ref.swap_slots()

  def wait_in(self, buffered_ref, src_ref, schedule=None):
    if schedule is None:
      schedule = _default_schedule
    pred = schedule["wait_in"](self, buffered_ref, src_ref)

    @self._named_scope("ep_wait_in")
    def _wait():
      if buffered_ref.is_input:
        buffered_ref.wait_in(src_ref, self.indices)
      if buffered_ref.is_accumulator:
        # In most cases we won't be waiting when init_accumulators is True,
        # so this is usually just setting what we just copied.
        buffered_ref.set_accumulator(self.init_accumulators)

    @self._named_scope("ep_set_accum")
    def _no_wait():
      if buffered_ref.is_accumulator:

        @pl.when(self.first_step | self.has_changed(buffered_ref))
        def _set_accumulator():
          # In most cases we will skip waiting when init_accumulators is True,
          # so this is usually just setting the accumulator to 0.
          buffered_ref.set_accumulator(self.init_accumulators)
    lax.cond(pred, _wait, _no_wait)

  def copy_in(self, buffered_ref, src_ref, schedule=None):
    if schedule is None:
      schedule = _default_schedule
    pred = schedule['copy_in'](self, buffered_ref, src_ref)

    @pl.when(pred)
    @self._named_scope("ep_copy_in")
    def _send():
      if buffered_ref.is_input:
        # We skip the last step because that's what prefetch is for.
        @pl.when(~self.last_step)
        def _copy_in():
          buffered_ref.copy_in(src_ref, self.next_indices)

  # --> Call prefetch here to grab the first inputs of next cycle.

  # convenience method for prefetch callbacks.
  def prefetch(self, buffered_ref, src_ref, schedule=None):
    if schedule is None:
      schedule = _default_schedule
    pred = schedule['prefetch'](self, buffered_ref, src_ref)

    @pl.when(pred)
    @self._named_scope("ep_prefetch")
    def _send():
      if buffered_ref.is_input:
        # Prefetch should only run on the last step.
        @pl.when(self.last_step)
        def _prefetch_in():
          buffered_ref.copy_in(src_ref, self.next_indices)

  def wait_out(self, buffered_ref, dst_ref, schedule=None):
    if schedule is None:
      schedule = _default_schedule
    pred = schedule['wait_out'](self, buffered_ref, dst_ref)

    @pl.when(pred)
    @self._named_scope("ep_wait_out")
    def _wait():
      if buffered_ref.is_output:
        buffered_ref.wait_out(dst_ref, self.prev_indices)

  # --> Call "postyeet" here, after last output copy is finished from previous
  #     cycle

  def copy_out(self, buffered_ref, dst_ref, schedule=None):
    if schedule is None:
      schedule = _default_schedule
    pred = schedule['copy_out'](self, buffered_ref, dst_ref)

    @self._named_scope("ep_copy_out")
    def _copy_out_and_accumulate():
      if buffered_ref.is_accumulator:
        buffered_ref.accumulate()
      if buffered_ref.is_output:
        buffered_ref.copy_out(dst_ref, self.indices)

    @self._named_scope("ep_accum")
    def _just_accumulate():
      if buffered_ref.is_accumulator:
        # We accumulate on the last step because we will set the accumulator
        # on the next first step. We can optimize this away if it becomes
        # a problem, but it is probably not worth the complexity to support
        # chains of different pipelines that want to reuse the accumulator with
        # slightly different schedules.
        @pl.when(self.last_step)
        def _accumulate():
          buffered_ref.accumulate()
    lax.cond(pred, _copy_out_and_accumulate, _just_accumulate)

  def finalize(self, buffered_ref, dst_ref, schedule=None):
    if schedule is None:
      schedule = _default_schedule
    pred = schedule['epilogue_wait_out'](self, buffered_ref, dst_ref)

    @pl.when(pred)
    @self._named_scope("ep_finalize")
    def _end():
      if buffered_ref.is_output:
        buffered_ref.swap_slots()  # formally correct, not actually necessary.
        buffered_ref.wait_out(dst_ref, self.indices)

  # END SCHEDULE --------------------------------------------------------------


# Scheduling overrides.

# When trying to fuse across pipelines that use accumulator arguments, we
# sometimes need to mess with the default scheduling above to avoid data-races
# or to maximize performance.  A schedule is simply a set of functions that
# calculate predicates for whether or not the pipeline input and output
# BufferedRefs should do copies and waits.


# Copy of the default pipeline schedule.  The default schedule tacitly assumes
# that the source and target HBM Refs change with each cycle.
_default_schedule = dict(
    prologue_copy_in=lambda s, bref, _: s.first_step_ever,
    # We assume that the source ref changed for prefetch.
    wait_in=lambda s, bref, _: s.has_changed(bref) | s.first_step,
    copy_in=lambda s, bref, _: s.will_change(bref) & ~s.last_step_ever,
    # We assume that the source ref changed. E.g. because of a CM DMA.
    prefetch=lambda s, bref, _: (
        (s.will_change(bref) | s.last_step) & ~s.last_step_ever
    ),
    # We assume that the target ref changed. E.g. because of a CM DMA.
    wait_out=lambda s, bref, _: (
        (s.has_changed(bref) | s.first_step) & ~s.first_step_ever
    ),
    # We assume that the target ref is changing. E.g. because of a CM DMA.
    copy_out=lambda s, bref, _: s.will_change(bref) | s.last_step,
    epilogue_wait_out=lambda s, bref, _: s.last_step_ever,
)


# Alternative schedule needed for accumulators reading and writing to a fixed
# HBM reference to avoid HBM data races for trivially small grids: only
# read/write when tiles change or at the very beginning or end of a fused
# pipeline schedule.
_fixed_schedule = dict(
    prologue_copy_in=lambda s, bref, _: s.first_step_ever,
    # We don't assume that the source ref changed for prefetch.
    wait_in=lambda s, bref, _: s.has_changed(bref) | s.first_step_ever,
    copy_in=lambda s, bref, _: s.will_change(bref) & ~s.last_step_ever,
    # We don't assume that the source ref changed.
    prefetch=lambda s, bref, _: s.will_change(bref) & ~s.last_step_ever,
    # We don't assume that the target ref changed.
    wait_out=lambda s, bref, _: s.has_changed(bref) & ~s.first_step_ever,
    # We don't assume that the target ref is changing.
    copy_out=lambda s, bref, _: s.will_change(bref) | s.last_step_ever,
    epilogue_wait_out=lambda s, bref, _: s.last_step_ever,
)


def skip_input_copies_when_init_accumulators(schedule) -> Any:
  """Skip input copies in schedule when init_accumulators is True."""
  new_schedule = {**schedule}
  for k in ["prologue_copy_in", "wait_in", "copy_in"]:

    def new_pred(original_pred_fn, *a):
      pred = original_pred_fn(*a)
      if a[1].is_accumulator or a[1].is_input_output:
        pred &= ~a[0].init_accumulators
      return pred

    new_schedule[k] = functools.partial(
        new_pred,
        schedule[k],
    )
  return new_schedule


_default_schedule = skip_input_copies_when_init_accumulators(_default_schedule)
_fixed_schedule = skip_input_copies_when_init_accumulators(_fixed_schedule)

def get_pipeline_schedule(schedule) -> Any:
  """Retrieve a named pipeline schedule or pass through fully specified one."""
  predefined_schedules = {
      'default': _default_schedule,
      'fixed': _fixed_schedule
  }
  if isinstance(schedule, str):
    return predefined_schedules[schedule].copy()
  return schedule


# Main pipeline methods


def make_pipeline_allocations(
    *refs,
    in_specs=None,
    out_specs=None,
    should_accumulate_out=False,
):
  """Create BufferedRefs for the pipeline.

  This function creates buffered refs for an inner pipeline that can be
  created at the top-level of a pallas call such that they may be reused across
  multiple invocations of the inner pipeline.

  Args:
    in_specs: input pallas block specs
    out_specs: output pallas block specs
    should_accumulate_out: booleans to indicate which outputs should be treated
      as accumulators.

  Returns:
    A list of BufferedRefs, one corresponding to each ref specified in the
    in_specs and out_specs.
  """
  # TODO(levskaya): generalize argument tree handling here and in emit_pipeline.
  num_in_specs = len(in_specs)
  if not isinstance(in_specs, (list, tuple)):
    in_specs = (in_specs,)
  if not isinstance(out_specs, (list, tuple)):
    out_specs = (out_specs,)
  if isinstance(in_specs, list):
    in_specs = tuple(in_specs)
  if isinstance(out_specs, list):
    out_specs = tuple(out_specs)
  in_refs = refs[:num_in_specs]
  out_refs = refs[num_in_specs:]
  def make_input_bref(in_spec, in_ref):
    return BufferedRef.input(in_spec, in_ref.dtype)
  in_brefs = jax.tree.map(make_input_bref, in_specs, in_refs)
  def make_output_bref(out_spec, out_ref, accumulate):
    if accumulate:
      return BufferedRef.accumulator(out_spec, out_ref.dtype)
    return BufferedRef.output(out_spec, out_ref.dtype)
  out_brefs = jax.tree.map(
      make_output_bref, out_specs, out_refs, should_accumulate_out)
  return (*in_brefs, *out_brefs)


class GridDimensionSemantics:
  pass
PARALLEL = GridDimensionSemantics()
ARBITRARY = GridDimensionSemantics()


def _partition_grid(
    grid: tuple[int | jax.Array, ...],
    core_axis: int | None,
    dimension_semantics: tuple[GridDimensionSemantics, ...] | None,
) -> tuple[tuple[int | jax.Array, ...], tuple[int | jax.Array, ...]]:
  if core_axis is None:
    # We aren't partitioning the grid
    return grid, (0,) * len(grid)
  num_cores = pl.num_programs(core_axis)
  # Check that num_cores is statically known
  if not isinstance(num_cores, int):
    raise NotImplementedError(
        f"Cannot partition grid over dynamic number of cores: {core_axis=}"
    )
  if num_cores == 1:
    # We aren't partitioning the grid
    return grid, (0,) * len(grid)

  # If dimension_semantics aren't provided, we assume it is all arbitrary.
  if dimension_semantics is None:
    dimension_semantics = (ARBITRARY,) * len(grid)
  if len(dimension_semantics) != len(grid):
    raise ValueError("dimension_semantics must be the same length as grid.")

  parallel_dimensions = {i for i, d in enumerate(dimension_semantics)
                         if d == PARALLEL}
  # If there are no parallel dimensions, we can't partition the grid
  if not parallel_dimensions:
    # TODO(sharadmv): enable running kernel on just one core
    raise NotImplementedError(
        "Cannot partition over cores without parallel grid dimensions:"
        f" {dimension_semantics=}"
    )
  if all(not isinstance(grid[i], int) for i in parallel_dimensions):
    raise NotImplementedError(
        f"Cannot partition cores over only dynamic grid dimensions: {grid=}"
    )
  # Try to find a divisible dimension to partition the grid on
  divisible_dimensions = {
      i for i in parallel_dimensions
      if isinstance(grid[i], int) and grid[i] % num_cores == 0
  }
  if divisible_dimensions:
    first_divisible_dimension, *_ = (
        i for i in range(len(dimension_semantics)) if i in divisible_dimensions
    )
    partitioned_dim_size = grid[first_divisible_dimension] // num_cores
    partitioned_dim_offset = pl.program_id(core_axis) * partitioned_dim_size
    new_grid = jax_util.tuple_update(
        grid, first_divisible_dimension, partitioned_dim_size
    )
    offsets = jax_util.tuple_update(
        (0,) * len(grid), first_divisible_dimension, partitioned_dim_offset
    )
  else:
    # No divisible dimensions, so we can't evenly partition the grid. Let's pick
    # the largest dimension and try to divide it as evenly as possible.
    # TODO(sharadmv): take the product of many nondivisible dimensions to
    # potentially divide it more evenly
    largest_parallel_dimension = max(grid[i] for i in parallel_dimensions
                                     if isinstance(grid[i], int))  # type: ignore
    partition_dimension, *_ = (
        i
        for i, d in enumerate(grid)
        if isinstance(d, int) and d == largest_parallel_dimension
    )
    base_num_iters, rem = divmod(grid[partition_dimension], num_cores)
    assert rem > 0, rem
    # We have some remainder iterations that we need to assign somewhere. We
    # know that rem < num_cores, so we can assign one extra iteration to each
    # core except for the last (num_cores - rem).
    core_index = pl.program_id(core_axis)
    num_iters = jnp.where(core_index < rem, base_num_iters + 1,
                          base_num_iters)
    new_grid = jax_util.tuple_update(grid, partition_dimension, num_iters)
    # Ordinarily, we would compute the offset as:
    #   grid_offset = pl.program_id(core_axis) * num_iters
    # However, since we have some cores that don't have an extra iteration, we
    # need to adjust the offset by `rem`.
    grid_offset = jnp.where(
        core_index < rem,
        core_index * num_iters,
        core_index * base_num_iters + rem,
    )
    offsets = jax_util.tuple_update(
        (0,) * len(grid), partition_dimension, grid_offset
    )
  return new_grid, offsets


def emit_pipeline(
    body,
    *,
    grid: tuple[int | jax.Array, ...],
    in_specs=None,
    out_specs=None,
    should_accumulate_out=False,
    core_axis: int | None = None,
    dimension_semantics: tuple[GridDimensionSemantics, ...] | None = None,
    trace_scopes: bool = True,
):
  """Creates a function to emit a manual pallas pipeline.

  This has the same semantics as pallas_call but is meant to be called inside
  pallas_call for nesting grids. This is useful when you need to have separate
  windowing strategies for communication and computation.

  The new argument `should_accumulate_out` can be used to specify which outputs
  we should accumulate into automatically within and across pipeline
  invocations.

  Args:
    body: pallas kernel to set up pipeline for.
    grid: a pallas grid definition.
    in_specs: input pallas block specs
    out_specs: output pallas block specs
    should_accumulate_out: booleans to indicate which outputs should be treated
      as accumulators.
    core_axis: optional int, indicates whether or not to partition the grid
      along the core axis.
    dimension_semantics: optional tuple of GridDimensionSemantics (e.g. PARALLEL
      or ARBITRARY).
    trace_scopes: optional bool, indicates whether to annotate each region in
      the pipeline using named_scope.
  """
  if any(not isinstance(d, (int, jax.Array)) for d in grid):
    grid_types = tuple(type(d) for d in grid)
    raise ValueError(
        f"Grid must consist of Python integers and JAX Arrays: {grid_types}"
    )
  grid, grid_offsets = _partition_grid(grid, core_axis, dimension_semantics)

  num_steps = _grid_size(grid)
  if not isinstance(in_specs, (list, tuple)):
    in_specs = (in_specs,)
  if not isinstance(out_specs, (list, tuple)):
    out_specs = (out_specs,)
  if isinstance(in_specs, list):
    in_specs = tuple(in_specs)
  if isinstance(out_specs, list):
    out_specs = tuple(out_specs)
  should_accumulate_out = _broadcast_pytree_to(should_accumulate_out, out_specs)

  def pipeline(
    *refs: Any,
    scratches=None,
    allocations=None,
    first_cycle: CondVal = True,
    last_cycle: CondVal = True,
    init_accumulators: CondVal = False,
    prefetch=None,
    postyeet=None,
    schedule=None,
    body_prologue=None,
  ):
    """
    Run the pipeline.

    Args:
      *ref_args: a list of pallas refs (or more generally a list of pytrees of
        pallas refs)
      scratches: scratch buffers for the inner kernel
      allocations: a list of BufferedRefs, one corresponding to each ref
      first_cycle: boolean indicating if this is the first invocation of the
        inner pipeline cycle.
      last_cycle: boolean indicating if this is the last invocation of the
        inner pipeline cycle.
      init_accumulators: whether to zero-init accumulators during this cycle.
      prefetch: callback called as fn(*brefs, scheduler) that is used to fetch
        the next cycle invocations first inputs.  Called during the inputs phase
        in the final inner step.
      postyeet: callback called as fn(*brefs, scheduler) that is used to finish
        any writes or transfers from the last output of the previous cycle.
        Called during the outputs phase in the first inner step.
      schedule: manually specified pipeline schedules for brefs, None indicates
        default schedule.
      body_prologue: For running code within the grid environment before the
        body is run. Useful for updating manual refs.
    """
    if scratches is None:
      scratches = ()
    if allocations is None:
      # run with inline scoped allocations
      return primitives.run_scoped(
          lambda allocations: pipeline(
              *refs,
              scratches=scratches,
              allocations=allocations,
              first_cycle=first_cycle,
              last_cycle=last_cycle,
              init_accumulators=init_accumulators,
              prefetch=prefetch,
              postyeet=postyeet,
              schedule=schedule,
          ),
          make_pipeline_allocations(
              *refs,
              in_specs=in_specs,
              out_specs=out_specs,
              should_accumulate_out=should_accumulate_out),
      )
    if isinstance(allocations, list):
      allocations = tuple(allocations)
    # Normalize custom schedule arguments.
    if schedule is None:
      schedule = map_brefs(lambda x: None, allocations)
    if not isinstance(schedule, (list, tuple)):
      schedule = map_brefs(lambda x: schedule, allocations)
    if isinstance(schedule, list):
      schedule = tuple(schedule)
    schedule = map_brefs(
        lambda _, x: get_pipeline_schedule(x), allocations, schedule)

    def loop_body(step, indices):
      nonlocal allocations
      scheduler = Scheduler(
          step,
          indices,
          grid,
          grid_offsets=grid_offsets,
          first_cycle=first_cycle,
          last_cycle=last_cycle,
          init_accumulators=init_accumulators,
          trace_scopes=trace_scopes,
      )

      # prepare any local VMEM aliases
      brefs = map_brefs(scheduler.alias_local_refs, allocations, refs)

      # loop input handling phase
      map_brefs(scheduler.initialize, brefs, refs, schedule)
      map_brefs(scheduler.copy_in, brefs, refs, schedule)
      map_brefs(scheduler.wait_in, brefs, refs, schedule)

      # prefetch inputs for the *next* invocation of this pipeline
      with scheduler._named_scope("ep_prefetch"):
        if prefetch is not None:
          lax.cond(step == num_steps - 1,
                  lambda: prefetch(*brefs, scheduler),
                  lambda: None)

      # run the kernel!
      if body_prologue is not None:
        with scheduler.grid_env():
          body_prologue()
      current_refs = map_brefs(lambda x: x.current_ref, brefs)
      with scheduler._named_scope("ep_run_kernel"):
        with scheduler.grid_env():
          body(*current_refs, *scratches)

      # loop output handling phase
      map_brefs(scheduler.copy_out, brefs, refs, schedule)
      map_brefs(scheduler.wait_out, brefs, refs, schedule)
      # handle writes for the *last* invocation of this pipeline's outputs
      with scheduler._named_scope("ep_postyeet"):
        if postyeet is not None:
          lax.cond(step == 0,
                  lambda: postyeet(*brefs, scheduler),
                  lambda: None)
      map_brefs(scheduler.finalize, brefs, refs, schedule)

      return _next_index(indices, grid)

    # run pipeline
    lax.fori_loop(0, num_steps, loop_body, (0,) * len(grid))

  return pipeline


def emit_pipeline_with_allocations(
    body,
    *,
    grid,
    in_specs=None,
    out_specs=None,
    should_accumulate_out=False,
):
  """Creates pallas pipeline and top-level allocation preparation functions.

  Args:
    body: pallas kernel to set up pipeline for.
    grid: a pallas grid definition.
    in_specs: input pallas block specs
    out_specs: output pallas block specs
    should_accumulate_out: booleans to indicate which outputs should be treated
      as accumulators.

  Returns:
    (emit_pipeline, make_allocations) function pair, where:
    emit_pipeline is the pallas pipeline function.
    make_allocations is a function to create buffered refs for the inner
      pipeline that can be created at the top-level of a pallas call to be
      reused across multiple invocations of the inner pipeline.

  """
  make_allocations = functools.partial(make_pipeline_allocations,
                    in_specs=in_specs,
                    out_specs=out_specs,
                    should_accumulate_out=should_accumulate_out)
  pipeline = emit_pipeline(
      body,
      grid=grid,
      in_specs=in_specs,
      out_specs=out_specs,
      should_accumulate_out=should_accumulate_out)

  return pipeline, make_allocations
