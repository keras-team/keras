# Copyright 2024 The JAX Authors.
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

"""Module for emitting custom GPU pipelines within a Pallas kernel."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import functools
import itertools as it
import math
from typing import Any

import jax
from jax import lax
from jax._src import core
from jax._src import linear_util as lu
from jax._src import util
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import primitives as gpu_primitives
from jax.experimental import pallas as pl
import jax.numpy as jnp


map = util.safe_map
zip = util.safe_zip


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BufferedRef:
  spec: pallas_core.BlockSpec = dataclasses.field(metadata={"static": True})
  is_index_invariant: bool = dataclasses.field(metadata={"static": True})
  gmem_ref: pallas_core.AbstractMemoryRef
  # ``None`` if the ref is pinned to GMEM; otherwise, has shape
  # [num_slots, *spec.block_shape].
  smem_ref: pallas_core.AbstractMemoryRef | None

  def get_ref_for_slot(
      self, slot: int | jax.Array
  ) -> pallas_core.AbstractMemoryRef:
    if self.smem_ref is None:
      return self.gmem_ref
    return self.smem_ref.at[slot]

  def compute_gmem_slice(self, grid_indices) -> tuple[pl.Slice, ...]:
    index_map = self.spec.index_map
    assert index_map is not None
    return tuple(
        pl.Slice(idx * size, size)  # type: ignore[arg-type]
        for idx, size in zip(
            index_map(*grid_indices), self.spec.block_shape  # type: ignore[arg-type]
        )
    )

  def copy_in(self, slot, grid_indices, barrier_ref):
    if not _in_smem(self.spec):
      return
    assert self.smem_ref is not None
    gmem_slices = self.compute_gmem_slice(grid_indices)
    gpu_primitives.copy_gmem_to_smem(
        self.gmem_ref.at[gmem_slices],  # pytype: disable=unsupported-operands
        self.smem_ref.at[slot],
        barrier_ref.at[slot],
    )

  def copy_out(self, slot, grid_indices, predicate=None):
    if not _in_smem(self.spec):
      return
    assert self.smem_ref is not None
    gmem_slices = self.compute_gmem_slice(grid_indices)
    gpu_primitives.copy_smem_to_gmem(
        self.smem_ref.at[slot],
        self.gmem_ref.at[gmem_slices],  # pytype: disable=unsupported-operands
        predicate=predicate,
    )


def _uses_arguments(
    index_map: Callable[..., Any], num_args: int
) -> Sequence[bool]:
  jaxpr, _, _, () = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(index_map), (core.ShapedArray((), jnp.int32),) * num_args
  )
  _, used_inputs = pe.dce_jaxpr(jaxpr, used_outputs=[True] * len(jaxpr.outvars))
  return used_inputs


def _is_index_invariant(
    spec: pallas_core.BlockSpec, grid: pallas_core.StaticGrid
) -> bool:
  if (index_map := spec.index_map) is None:
    return True
  return not any(_uses_arguments(index_map, len(grid)))


def _inc_grid_by_1(
    indices: tuple[jax.Array, ...], grid: Sequence[int]
) -> tuple[jax.Array, ...]:
  next_indices = []
  carry: bool | jax.Array = True
  for idx, size in reversed(list(zip(indices, grid))):
    next_idx = lax.select(carry, idx + 1, idx)
    carry = next_idx == size
    next_indices.append(lax.select(carry, 0, next_idx).astype(idx.dtype))
  return tuple(reversed(next_indices))


def _in_smem(spec: pallas_core.BlockSpec) -> bool:
  return spec.memory_space in (None, gpu_core.SMEM)


# ``pl.Slice`` uses a different pytree encoding, depending on whether the
# start/size are static or dynamic. This leads to pytree structure mismatch
# in the pipeline body. So, we define a different ``Slice`` class below.


@dataclasses.dataclass(frozen=True)
class _Slice:
  start: int | jax.Array
  size: int | jax.Array

  def __eq__(self, other: _Slice) -> jax.Array:  # type: ignore
    return lax.bitwise_and(self.start == other.start, self.size == other.size)


jax.tree_util.register_dataclass(
    _Slice, data_fields=["start", "size"], meta_fields=[]
)


def emit_pipeline(
    body: Callable[..., None],
    *,
    grid: pallas_core.StaticGrid,
    in_specs: Sequence[pallas_core.BlockSpec] = (),
    out_specs: Sequence[pallas_core.BlockSpec] = (),
    max_concurrent_steps: int = 1,
    delay_release: int = 0,
):
  """Creates a function to emit a manual pipeline within a Pallas kernel.

  Args:
    body: The pipeline body.
    grid: The grid to use for the pipeline.
    in_specs: The block specs for the inputs.
    out_specs: The block specs for the outputs.
    max_concurrent_steps: The maximum number of sequential stages that are
      active concurrently. Defaults to 1.
    delay_release: The number of steps to wait before reusing the input/output
      references. Defaults to 0, and must be strictly smaller than
      ``max_concurrent_steps``. Generally, you'll want to set it to 1 if you
      don't await the WGMMA in the body.
  """
  num_steps = math.prod(grid)

  if max_concurrent_steps <= delay_release:
    raise ValueError(
        "max_concurrent_steps must be greater than delay_release, but"
        f" {max_concurrent_steps=}, {delay_release=}"
    )

  # Shrink ``max_concurrent_steps`` if the total number of steps is lower to
  # reduce the size of the refs allocated in SMEM.
  if max_concurrent_steps > num_steps:
    max_concurrent_steps = num_steps
    delay_release = 0  # No need to delay anything.

  def pipeline(*gmem_refs: pallas_core.AbstractMemoryRef):
    in_gmem_refs, out_gmem_refs = util.split_list(gmem_refs, [len(in_specs)])
    in_smem_refs, out_smem_refs = util.split_list(
        [
            gpu_core.SMEM(
                (max_concurrent_steps, *spec.block_shape),  # type: ignore
                ref.dtype,
                transforms=tuple(
                    t.batch(1) for t in getattr(spec, "transforms", ())
                ),
            )
            if _in_smem(spec)
            else None
            for spec, ref in zip(it.chain(in_specs, out_specs), gmem_refs)
        ],
        [len(in_specs)],
    )
    return pl.run_scoped(
        functools.partial(
            scoped_pipeline,
            in_gmem_refs=in_gmem_refs,
            out_gmem_refs=out_gmem_refs,
        ),
        in_smem_refs=in_smem_refs,
        out_smem_refs=out_smem_refs,
        barrier_ref=gpu_core.Barrier(
            # TODO(slebedev): Change this to arrive only once.
            sum(map(_in_smem, in_specs)),
            num_barriers=max_concurrent_steps,
        ),
    )

  def scoped_pipeline(
      *, in_gmem_refs, out_gmem_refs, in_smem_refs, out_smem_refs, barrier_ref
  ):
    in_brefs: Sequence[BufferedRef] = [
        BufferedRef(spec, _is_index_invariant(spec, grid), gmem_ref, smem_ref)
        for spec, gmem_ref, smem_ref in zip(
            in_specs, in_gmem_refs, in_smem_refs
        )
    ]
    out_brefs: Sequence[BufferedRef] = [
        BufferedRef(spec, _is_index_invariant(spec, grid), gmem_ref, smem_ref)
        for spec, gmem_ref, smem_ref in zip(
            out_specs, out_gmem_refs, out_smem_refs
        )
    ]

    for step, indices in enumerate(
        it.islice(it.product(*map(range, grid)), max_concurrent_steps)
    ):
      map(lambda bref: bref.copy_in(step, indices, barrier_ref), in_brefs)

    def loop_body(step, carry):
      slot = step % max_concurrent_steps
      indices, fetch_indices, last_store_slices = carry

      if in_specs:
        # Wait for the current GMEM->SMEM copy to complete.
        gpu_primitives.barrier_wait(barrier_ref.at[slot])
      # Wait for the previous output SMEM->GMEM copy to complete.
      gpu_primitives.wait_smem_to_gmem(
          max_concurrent_steps - (1 + delay_release), wait_read_only=True
      )

      with pallas_core.grid_env(map(pallas_core.GridAxis, indices, grid)):
        body(*(
            bref.get_ref_for_slot(slot)
            for bref in it.chain(in_brefs, out_brefs)
        ))

      if not all(bref.is_index_invariant for bref in out_brefs):
        gpu_primitives.commit_smem()

      # Copy the output from SMEM to GMEM.
      new_store_slices = last_store_slices[:]
      for idx, bref in enumerate(out_brefs):
        if bref.is_index_invariant:
          assert last_store_slices[idx] is None
          continue
        assert last_store_slices[idx] is not None
        new_store_slices[idx] = tuple(
            _Slice(s.start, s.size) for s in bref.compute_gmem_slice(indices)
        )
        are_same_slices = map(
            lambda old, new: old == new,
            last_store_slices[idx],
            new_store_slices[idx],
        )
        slices_changed = ~functools.reduce(lax.bitwise_and, are_same_slices)
        is_last_step = step == num_steps - 1
        # TODO(apaszke,slebedev): This still diverges significantly from the
        # TPU semantics in that it will move on to the next SMEM output slice
        # even if it's not storing the previous one.
        bref.copy_out(
            slot,
            indices,
            predicate=lax.bitwise_or(slices_changed, is_last_step),
        )

      fetch_step = step + (max_concurrent_steps - delay_release)
      fetch_slot = slot  # (x + y) % y == x % y
      jax.lax.cond(
          lax.bitwise_and(fetch_step >= delay_release, fetch_step < num_steps),
          lambda: map(
              lambda bref: bref.copy_in(fetch_slot, fetch_indices, barrier_ref),
              in_brefs,
          ),
          lambda: [None] * len(in_brefs),
      )

      return (
          _inc_grid_by_1(indices, grid),
          _inc_grid_by_1(fetch_indices, grid),
          new_store_slices,
      )

    indices = (jnp.asarray(0, dtype=lax.dtype(0)),) * len(grid)
    fetch_indices = indices
    for _ in range(max_concurrent_steps):
      fetch_indices = _inc_grid_by_1(fetch_indices, grid)
    # TODO(justinfu): Only store base pointer instead of all indices.
    last_store_slices = [
        None
        if bref.is_index_invariant
        else (_Slice(-1, -1),) * len(bref.spec.block_shape)
        for bref in out_brefs
    ]
    last_indices, _, _ = lax.fori_loop(
        0, num_steps, loop_body, (indices, fetch_indices, last_store_slices)
    )

    # Outputs invariant to the sequential axis are never written from inside the
    # loop. This is the only place where we store them.
    if all(bref.is_index_invariant for bref in out_brefs):
      gpu_primitives.commit_smem()
    last_slot = (num_steps - 1) % max_concurrent_steps
    for bref in out_brefs:
      if bref.is_index_invariant:
        bref.copy_out(last_slot, last_indices, predicate=None)

    # Finalize the pipeline.
    gpu_primitives.wait_smem_to_gmem(0)

  return pipeline

def emit_pipeline_warp_specialized(
    body: Callable[..., None],
    *,
    grid: pallas_core.StaticGrid,
    memory_registers: int,
    in_specs: Sequence[gpu_core.GPUBlockSpec] = (),
    out_specs: Sequence[gpu_core.GPUBlockSpec] = (),
    max_concurrent_steps: int = 2,
    wg_axis: str,
    num_compute_wgs: int,
    carry_coroutine: Any | None = None,
    memory_thread_idx: int | None = None,
):
  """Creates a function to emit a warp-specialized pipeline.

  Args:
    body: The pipeline body.
    grid: The grid to use for the pipeline.
    memory_registers: The number of registers to reserve for the memory thread.
      For H100 GPUs, 40 is a reasonable value.
    in_specs: The block specs for the inputs.
    out_specs: The block specs for the outputs.
    max_concurrent_steps: The maximum number of sequential stages that are
      active concurrently. Defaults to 2.
    wg_axis: The axis name for the warp group axis.
    num_compute_wgs: The number of compute warpgroups
    carry_coroutine: If specified, enables carries in the pipeline.
      The signature of the body function will be modified such that the last
      argument will be the current carry and it must return the next carry.
      The coroutine itself should yield the initial carry, and the
      yield statement will return the final value of the carry.
    memory_thread_idx: The index of the memory thread. If not specified,
      defaults to the last thread.
  """
  # TODO(justinfu): Factor out common code between warp-specialized and
  # normal pipelines.
  # TODO(justinfu): Allow passing consumed_barrier into body.

  if memory_thread_idx is None:
    memory_thread_idx = num_compute_wgs
  if memory_thread_idx != num_compute_wgs:
    # TODO(justinfu): Indexing calculations for buffers assume the memory
    # thread is the last thread.
    raise NotImplementedError("Memory thread must be the last thread.")

  has_carry = carry_coroutine is not None

  # Trace the index maps to determine if they depend on the grid.
  # Grid-independent values will not be multiple-buffered.
  in_spec_has_seq_axis = [
      ~_is_index_invariant(spec, grid) for spec in in_specs]
  out_spec_has_seq_axis = [
      ~_is_index_invariant(spec, grid) for spec in out_specs]
  spec_has_seq_axis = [*in_spec_has_seq_axis, *out_spec_has_seq_axis]

  num_pipeline_steps = math.prod(grid)

  def _get_slot(step, has_seq_dim):
    """Returns the buffer slot given the pipeline step."""
    if has_seq_dim:
      return step
    else:
      return 0

  # Shrink ``max_concurrent_steps`` if the total number of steps is lower to
  # reduce the size of the refs allocated in SMEM.
  if max_concurrent_steps > num_pipeline_steps:
    max_concurrent_steps = num_pipeline_steps

  def pipeline(*gmem_refs: pallas_core.AbstractMemoryRef):
    in_gmem_refs, out_gmem_refs = util.split_list(gmem_refs, [len(in_specs)])
    if len(out_gmem_refs) != len(out_specs):
      raise ValueError(
          "Number of output refs does not match number of output specs."
      )
    smem_allocs = []
    for spec, has_seq_dim, gmem_ref in zip(
        it.chain(in_specs, out_specs),
        spec_has_seq_axis,
        gmem_refs):
      slots = max_concurrent_steps if has_seq_dim else 1
      smem_allocs.append(
          gpu_core.SMEM(
              (slots, *spec.block_shape),   # type: ignore
              gmem_ref.dtype,
              transforms=spec.transforms,
          )
      )
    in_smem_refs, out_smem_refs = util.split_list(
        smem_allocs, [len(in_specs)])

    in_smem_barriers = []
    for has_seq_dim in in_spec_has_seq_axis:
      num_barriers = max_concurrent_steps if has_seq_dim else 1
      in_smem_barriers.append(
          gpu_core.Barrier(
          num_arrivals=1,
          num_barriers=num_barriers))
    return pl.run_scoped(
        functools.partial(
            scoped_pipeline,
            in_gmem_refs=in_gmem_refs,
            out_gmem_refs=out_gmem_refs,
        ),
        in_smem_refs=in_smem_refs,
        out_smem_refs=out_smem_refs,
        in_smem_barrier_refs=in_smem_barriers,
        consumed_barrier_ref=gpu_core.Barrier(
            num_arrivals=num_compute_wgs,
            num_barriers=max_concurrent_steps,
        ),
    )

  def scoped_pipeline(
      *,
      in_gmem_refs,
      out_gmem_refs,
      in_smem_refs,
      out_smem_refs,
      in_smem_barrier_refs,
      consumed_barrier_ref,
  ):
    in_brefs: Sequence[BufferedRef] = [
        BufferedRef(spec, ~has_seq_axis, gmem_ref, smem_ref)
        for spec, has_seq_axis, gmem_ref, smem_ref in zip(
            in_specs, in_spec_has_seq_axis, in_gmem_refs, in_smem_refs
        )
    ]
    out_brefs: Sequence[BufferedRef] = [
        BufferedRef(spec, ~has_seq_axis, gmem_ref, smem_ref)
        for spec, has_seq_axis, gmem_ref, smem_ref in zip(
            out_specs, out_spec_has_seq_axis, out_gmem_refs, out_smem_refs
        )
    ]

    def compute_block():
      gpu_primitives.set_max_registers(
          _compute_registers(memory_registers, num_compute_wgs),
          action="increase")

      def compute_loop_body(step, carry):
        indices, last_store_slices, prev_body_carry = carry
        slot = step % max_concurrent_steps
        # Wait for the current GMEM->SMEM copies to complete.
        for in_barrier, has_seq_dim in zip(
            in_smem_barrier_refs, in_spec_has_seq_axis):
          # TODO(justinfu): Use a single barrier with
          # num_arrivals=len(in_smem_barrier_refs)
          gpu_primitives.barrier_wait(
              in_barrier.at[_get_slot(slot, has_seq_dim)])

        # Wait for the previous output SMEM->GMEM copy to complete.
        gpu_primitives.wait_smem_to_gmem(max_concurrent_steps - 1)

        with pallas_core.grid_env(map(pallas_core.GridAxis, indices, grid)):
          body_refs = []
          for bref in it.chain(in_brefs, out_brefs):
            buf_slot = _get_slot(slot, ~bref.is_index_invariant)
            body_refs.append(bref.get_ref_for_slot(buf_slot))

          if has_carry:
            next_body_carry = body(*body_refs, prev_body_carry)
          else:
            body(*body_refs)
            next_body_carry = None
        gpu_primitives.barrier_arrive(consumed_barrier_ref.at[slot])
        # Copy the output from SMEM to GMEM.
        if not all(bref.is_index_invariant for bref in out_brefs):
          gpu_primitives.commit_smem()

        new_store_slices = last_store_slices[:]
        for idx, bref in enumerate(out_brefs):
          if bref.is_index_invariant:
            assert last_store_slices[idx] is None
            continue
          assert last_store_slices[idx] is not None
          new_store_slices[idx] = tuple(
              _Slice(s.start, s.size) for s in bref.compute_gmem_slice(indices)
          )
          are_same_slices = map(
              lambda old, new: old == new,
              last_store_slices[idx],
              new_store_slices[idx],
          )
          slices_changed = ~functools.reduce(lax.bitwise_and, are_same_slices)
          bref.copy_out(_get_slot(slot, ~bref.is_index_invariant),
                        indices,
                        predicate=slices_changed)
        next_indices = _inc_grid_by_1(indices, grid)
        return (next_indices, new_store_slices, next_body_carry)
      init_indices = (jnp.asarray(0, dtype=lax.dtype(0)),) * len(grid)
      # TODO(justinfu): Only store base pointer instead of all indices.
      last_store_slices = [
          None
          if bref.is_index_invariant
          else (_Slice(-1, -1),) * len(bref.spec.block_shape)
          for bref in out_brefs
      ]

      if has_carry:
        _carry = carry_coroutine()
        try:
          carry_init = next(_carry)
        except StopIteration:
          raise ValueError("carry_coroutine must yield the initial carry.")  # pylint: disable=raise-missing-from
      else:
        _carry = None
        carry_init = None
      init_loop_carry = (init_indices, last_store_slices, carry_init)
      last_indices, _, final_body_carry = lax.fori_loop(0,
                    num_pipeline_steps,
                    compute_loop_body,
                    init_loop_carry)
      if has_carry:
        try:
          _carry.send(final_body_carry)  # pytype: disable=attribute-error
          raise ValueError("carry_coroutine must only yield once.")
        except StopIteration:
          pass

      # Handle index_invariant outputs after the loop. They are not
      # written in the main pipeline loop.
      if all(bref.is_index_invariant for bref in out_brefs):
        gpu_primitives.commit_smem()
      last_slot = (num_pipeline_steps - 1) % max_concurrent_steps
      for bref in out_brefs:
        if bref.is_index_invariant:
          bref.copy_out(last_slot, last_indices, predicate=None)

      # Finalize the pipeline.
      gpu_primitives.wait_smem_to_gmem(0)

    # The memory thread executes this block which issues all pipelined DMAs.
    def memory_block():
      gpu_primitives.set_max_registers(memory_registers, action="decrease")
      indices = (jnp.asarray(0, dtype=lax.dtype(0)),) * len(grid)

      # Begin initial copies.
      for step in range(max_concurrent_steps):
        for bref, barrier in zip(in_brefs, in_smem_barrier_refs):
          buf_slot = _get_slot(step, ~bref.is_index_invariant)
          bref.copy_in(buf_slot, indices, barrier)
        indices = _inc_grid_by_1(indices, grid)

      def memory_loop_body(step, carry):
        indices, = carry
        slot = step % max_concurrent_steps
        fetch_slot = slot  # (x + y) % y == x % y
        gpu_primitives.barrier_wait(consumed_barrier_ref.at[slot])
        for bref, barrier in zip(in_brefs, in_smem_barrier_refs):
          bref.copy_in(
              _get_slot(fetch_slot, ~bref.is_index_invariant), indices, barrier)
        next_indices = _inc_grid_by_1(indices, grid)
        return (next_indices,)
      lax.fori_loop(0, num_pipeline_steps - max_concurrent_steps,
                    memory_loop_body, (indices,))

    wg_idx = lax.axis_index(wg_axis)
    lax.cond(
        wg_idx != memory_thread_idx,
        compute_block,
        memory_block
    )
  return pipeline

def _compute_registers(
    memory_registers: int,
    num_compute_wgs: int,
) -> int:
  """Returns the number of registers to use for the compute thread."""
  # TODO(justinfu): Configure this per-platform.
  n_registers = (512 - memory_registers) / num_compute_wgs
  # Round down to the nearest multiple of 8.
  return int((n_registers // 8) * 8)
