# Copyright 2024 The JAX Authors. All Rights Reserved.
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
# ==============================================================================

import contextlib
import itertools
import json
import math
from typing import Callable, ParamSpec, TypeVar
import warnings

import jax
from jax._src.lib import xla_client
import jax.numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import scf
import numpy as np

from .utils import *  # noqa: F403

try:
  from jax._src.lib import mosaic_gpu as mosaic_gpu_lib
except ImportError:
  has_registrations = False
else:
  # TODO(slebedev): Remove the if once the minimum jaxlib is 0.4.36.
  has_registrations = hasattr(mosaic_gpu_lib._mosaic_gpu_ext, "registrations")
  if has_registrations:
    for name, handler in mosaic_gpu_lib._mosaic_gpu_ext.registrations():
      xla_client.register_custom_call_target(
          name, handler, platform="CUDA", api_version=1
      )

# ruff: noqa: F405
# mypy: ignore-errors

T = TypeVar("T")
P = ParamSpec("P")

def _event_record(args, *, copy_before):
  flat_args, treedef = jax.tree.flatten(args)
  event, *flat_outs = jax.ffi.ffi_call(
      "mgpu_event_record",
      result_shape_dtypes=(jax.core.ShapedArray((), jnp.uint64), *flat_args),
      input_output_aliases={i: i + 1 for i in range(len(flat_args))},
  )(*flat_args, copy_before=copy_before)
  return event, treedef.unflatten(flat_outs)


def _event_elapsed(start_event, end_event):
  return jax.ffi.ffi_call(
      "mgpu_event_elapsed",
      result_shape_dtypes=jax.core.ShapedArray((), jnp.float32),
  )(start_event, end_event)


def _measure_events(
    f: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> tuple[T, float]:
  if not has_registrations:
    raise RuntimeError(
        "This function requires jaxlib >=0.4.36 with CUDA support."
    )

  if not (args or kwargs):
    # We require at least one argument and at least one output to ensure
    # that there is a data dependency between `_event_record` calls in
    # the resulting HLO program.
    raise ValueError("Can only measure functions with arguments")

  @jax.jit
  def run(*args, **kwargs):
    start_event, (args, kwargs) = _event_record(
        (args, kwargs), copy_before=True
    )
    end_event, outs = _event_record(f(*args, **kwargs), copy_before=False)
    if jax.tree.structure(outs).num_leaves == 0:
      raise ValueError("Can only measure functions with at least one output")
    return outs, _event_elapsed(start_event, end_event)

  jax.block_until_ready(run(*args, **kwargs))  # Warmup.
  outs, elapsed = run(*args, **kwargs)
  return outs, float(elapsed)


def _measure_cupti(f, aggregate):
  def wrapper(*args, **kwargs):
    mosaic_gpu_lib._mosaic_gpu_ext._cupti_init()
    try:
      results = jax.block_until_ready(jax.jit(f)(*args, **kwargs))
    finally:
      timings = mosaic_gpu_lib._mosaic_gpu_ext._cupti_get_timings()
    if not timings:
      return results, None
    elif aggregate:
      return results, sum(item[1] for item in timings)
    else:
      return results, timings
  return wrapper


def measure(f: Callable, *, mode: str = "events", aggregate: bool = True
) -> Callable:
  """Sets up a function ``f`` for profiling on GPU.

  ``measure`` is a higher-order function that augments the argument ``f`` to
  return GPU runtime in milliseconds, in addition to its proper outputs.

  Args:
    f: The function to measure. It must accept at least one argument and return
      at least one output to be measurable.
    mode: The mode of operation. Possible values are:

      - "cupti", for CUPTI-based profiling.
      - "events", for CUDA events-based profiling.

      The two modes use different measurement methodologies and should not be
      treated as interchangeable backends. See the Notes section for important
      discussion.
    aggregate: Whether to report an aggregate runtime. When ``False`` (only
      supported by ``mode="cupti"``), the per-kernel timings are returned as a
      list of tuples ``(<kernel name>, <runtime in ms>)``.

  Returns:
    A new function ``g`` that returns the measured GPU runtime as its last
    additional output. Otherwise ``g`` accepts the same inputs and returns the
    same outputs as ``f``.

  Notes:
    `CUPTI (CUDA Profiling Tools Interface)
    <https://docs.nvidia.com/cupti/index.html>`_ is a high-accuracy,
    high-precision profiling and tracing API, used in particular by Nsight
    Systems and Nsight Compute. When using ``measure`` with ``mode="cupti"``,
    device (GPU) execution runtimes are recorded for each kernel launched
    during the execution of the function. In that mode, setting
    ``aggregate=True`` will sum the individual kernel runtimes to arrive at an
    aggregate measurement. The "gaps" between the kernels when the device is
    idle are not included in the aggregate.

    The CUPTI API only allows a single "subscriber". This means that the
    CUPTI-based profiler will fail when the program is run using tools that
    make use of CUPTI, such as CUDA-GDB, Compute Sanitizer, Nsight Systems, or
    Nsight Compute.

    ``mode="events"`` uses a different approach: a CUDA event is recorded
    before and after the function ``f`` is executed. The reported runtime is
    the time elapsed between the two events. In particular, included in the
    measurement are:

    - any potential "gaps" between the kernels when the device is idle
    - any potential "gaps" between the "before" event and the start of the
      first kernel, or between the end of the last kernel and the "after" event

    In an attempt to minimize the second effect, internally the events-based
    implementation may execute ``f`` more than once to "warm up" and exclude
    compilation time from the measurement.
  """
  match mode:
    case "cupti":
      return _measure_cupti(f, aggregate)
    case "events":
      if not aggregate:
        raise ValueError(f"{aggregate=} is not supported with {mode=}")
      def measure_events_wrapper(*args, **kwargs):
        return _measure_events(f, *args, **kwargs)
      return measure_events_wrapper
    case _:
      raise ValueError(f"Unrecognized profiler mode {mode}")


class ProfilerSpec:
  ENTER = 0
  EXIT = 1 << 31

  def __init__(self, entries_per_warpgroup: int):
    self.entries_per_warpgroup = entries_per_warpgroup
    self.interned_names = {}

  def _num_warpgroups(
      self, grid: tuple[int, ...], block: tuple[int, ...]
  ) -> int:
    if math.prod(block) % WARPGROUP_SIZE:
      raise ValueError("Block size is not a multiple of warpgroup size")
    return math.prod(grid) * math.prod(block) // WARPGROUP_SIZE

  def mlir_buffer_type(
      self, grid: tuple[int, ...], block: tuple[int, ...]
  ) -> ir.Type:
    return ir.MemRefType.get(
        (self._num_warpgroups(grid, block) * self.entries_per_warpgroup,),
        ir.IntegerType.get_signless(32),
    )

  def jax_buffer_type(
      self, grid: tuple[int, ...], block: tuple[int, ...]
  ) -> ir.Type:
    return jax.ShapeDtypeStruct(
        (self._num_warpgroups(grid, block) * self.entries_per_warpgroup,),
        jnp.uint32,
    )

  def smem_i32_elements(self, block: tuple[int, ...]):
    num_warpgroups = self._num_warpgroups((), block)
    return int(num_warpgroups * self.entries_per_warpgroup)

  def smem_bytes(self, block: tuple[int, ...]):
    bytes_per_entry = 4
    return self.smem_i32_elements(block) * bytes_per_entry

  def intern_name(self, name: str) -> int:
    if (name_id := self.interned_names.get(name, None)) is not None:
      return name_id
    name_id = self.interned_names[name] = len(self.interned_names)
    if name_id & self.EXIT:
      raise RuntimeError("Allocated too many names")
    return name_id

  def dump(self, buffer, f, grid: tuple[int, ...], block: tuple[int, ...]):
    buffer = np.asarray(buffer)
    num_blocks = math.prod(grid)
    warpgroups_per_block = self._num_warpgroups((), block)
    entries = buffer.reshape(
        num_blocks, warpgroups_per_block, self.entries_per_warpgroup
    )
    start_times = entries[..., 0]
    sm_ids = entries[..., 1]
    entries_used = entries[..., 2]
    if np.any(entries_used > self.entries_per_warpgroup - 2):
      raise RuntimeError("Insufficient space to capture a full trace")
    traces = entries[..., 3:]
    unintern = {v: k for k, v in self.interned_names.items()}
    events = []
    for block_idx, wg_idx in np.ndindex(num_blocks, warpgroups_per_block):
      valid_entries = entries_used[block_idx, wg_idx] - 3
      local_clock_offset = None
      assert valid_entries % 2 == 0, valid_entries
      start_time = start_times[block_idx, wg_idx]
      block_events = []
      last_time = float("-inf")
      for i in range(0, valid_entries, 2):
        tag = traces[block_idx, wg_idx, i]
        time = traces[block_idx, wg_idx, i + 1]
        if local_clock_offset is None:
          local_clock_offset = time
        time -= local_clock_offset
        time -= i * 6  # Account for the overhead of profiling.
        if time < 0:
          break  # Detect a timer wraparound
        name_id = tag
        begin = True
        if name_id & ProfilerSpec.EXIT:
          name_id = name_id ^ ProfilerSpec.EXIT
          begin = False
        name = unintern[name_id]
        if last_time >= time:
          if last_time - time > 10:
            warnings.warn(
                "Profiler clock went significantly backwards for event"
                f" {'start' if begin else 'end'} `{name}`: {last_time} ->"
                f" {time}"
            )
          time = last_time + 1
        last_time = time
        block_events.append({
            "name": name,
            "ph": "B" if begin else "E",
            "ts": float(start_time + time) / 1e3,
            "pid": 1 + int(sm_ids[block_idx, wg_idx]),
            "tid": 1 + wg_idx + warpgroups_per_block * block_idx,
        })
      else:  # If we didn't break
        if block_events:
          events.append(block_events)
    events = sorted(events, key=lambda x: x[0]["ts"])
    flat_events = list(itertools.chain.from_iterable(events))
    return json.dump({"displayTimeUnit": "ns", "traceEvents": flat_events}, f)


class OnDeviceProfiler:

  def __init__(self, spec: ProfilerSpec, smem_buffer: ir.Value, gmem_buffer: ir.Value):
    self.spec = spec
    self.start = globaltimer("low")
    i32 = ir.IntegerType.get_signless(32)
    index = ir.IndexType.get()
    self.entries_per_wg = spec.entries_per_warpgroup
    wg_idx = warpgroup_idx(sync=False)
    self.smem_buffer = memref_slice(
        smem_buffer,
        ds(
            arith.index_cast(
                index, arith.muli(wg_idx, c(self.entries_per_wg, i32))
            ),
            self.entries_per_wg,
        ),
    )
    self.smem_buffer_ptr = memref_ptr(self.smem_buffer, memory_space=3)
    self.gmem_buffer = gmem_buffer
    self.is_profiling_thread = arith.cmpi(
        arith.CmpIPredicate.eq,
        arith.remui(thread_idx(), c(WARPGROUP_SIZE, i32)),
        c(0, i32),
    )
    # Hopefully mem2reg will remove the allocation.
    self.offset = memref.alloca(ir.MemRefType.get((), i32), [], [])
    memref.store(c(0, i32), self.offset, [])

  @contextlib.contextmanager
  def record(self, name: str):
    i32 = ir.IntegerType.get_signless(32)
    name_id = self.spec.intern_name(name)
    def store(modifier):
      cur = memref.load(self.offset, [])
      i64 = ir.IntegerType.get_signless(64)
      base_addr = arith.addi(
          llvm.ptrtoint(i64, self.smem_buffer_ptr),
          arith.extui(i64, arith.muli(cur, c(4, i32))),
      )
      llvm.inline_asm(
          ir.Type.parse("!llvm.void"),
          [self.is_profiling_thread, base_addr, c(modifier | name_id, i32)],
          """
          @$0 st.shared.v2.u32 [$1], {$2, %clock};
          """,
          "b,l,r",
          has_side_effects=True,
      )
      memref.store(
          arith.addi(cur, c(2, cur.type)),
          self.offset,
          [],
      )
    store(ProfilerSpec.ENTER)
    yield
    store(ProfilerSpec.EXIT)

  def finalize(self, grid: tuple[int, ...], block: tuple[int, ...]):
    index = ir.IndexType.get()
    i32 = ir.IntegerType.get_signless(32)

    gpu.barrier()   # Make sure all warpgroups are done.

    block_idx = c(0, index)
    for dim in gpu.Dimension:  # pytype: disable=wrong-arg-types
      block_idx = arith.addi(
          arith.muli(block_idx, gpu.grid_dim(dim)), gpu.block_id(dim)
      )
    wg_idx = warpgroup_idx(sync=False)
    wg_per_block = math.prod(block) // WARPGROUP_SIZE
    global_wg_idx = arith.addi(
        arith.muli(block_idx, c(wg_per_block, index)),
        arith.index_cast(index, wg_idx),
    )
    start_offset = arith.muli(global_wg_idx, c(self.entries_per_wg, index))
    wg_gmem_buffer = memref.subview(
        self.gmem_buffer, [start_offset], [self.entries_per_wg], [1],
        result_type=ir.Type.parse(
            f"memref<{self.entries_per_wg}xi32, strided<[1], offset: ?>>"
        ),
    )
    thread_in_wg = arith.remui(thread_idx(), c(128, i32))
    if_first = scf.IfOp(
        arith.cmpi(arith.CmpIPredicate.eq, thread_in_wg, c(0, i32))
    )
    with ir.InsertionPoint(if_first.then_block):
      memref.store(self.start, wg_gmem_buffer, [c(0, index)])
      memref.store(smid(), wg_gmem_buffer, [c(1, index)])
      memref.store(
          arith.addi(memref.load(self.offset, []), c(3, i32)),
          wg_gmem_buffer,
          [c(2, index)],
      )

      for_op = scf.ForOp(
          c(0, index),
          c(self.entries_per_wg - 3, index),
          c(1, index),
      )
      with ir.InsertionPoint(for_op.body):
        x = memref.load(self.smem_buffer, [for_op.induction_variable])
        memref.store(
            x,
            wg_gmem_buffer,
            [arith.addi(for_op.induction_variable, c(3, index))],
        )
        scf.yield_([])
      scf.yield_([])
