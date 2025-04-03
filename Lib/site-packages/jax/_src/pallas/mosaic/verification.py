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

import contextlib
import dataclasses
import io
import itertools
import math
import textwrap
from typing import Any, Sequence
from jax import lax
from jax._src import core as jax_core
from jax._src import tree_util
from jax._src.lib import tpu
from jax._src.pallas.mosaic import lowering
from jax._src.pallas.mosaic import primitives
from jax._src.util import split_list, unzip2
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import func
from jaxlib.mlir.passmanager import PassManager

_UNSPECIFIED = object()
Var = str

# TODO(apaszke): Add checks that semaphores are always left at 0.
# TODO(apaszke): Add checks that no remote resources are used while the remote
# device is not in the kernel (both before and after).
# TODO(apaszke): Model 0-sized DMAs faithfully.


PREAMBLE = """
#define buf_readers(index, device, core) _buf_readers[(index)*(NDEVICE*NCORE) + (device)*NCORE + core]
#define buf_written(index, device, core) _buf_written[(index)*(NDEVICE*NCORE) + (device)*NCORE + core]
#define sems(index, device, core) _sems[(index)*(NDEVICE*NCORE) + (device)*NCORE + core]
#define barrier_sems(device, core) _barrier_sems[(device)*NCORE + core]

#ifndef NDMA
#define NDMA 2
#endif

mtype = { DMA };
chan dma_queue = [NDMA] of { mtype, int, int, int, int, int, int, int, int, int, int };
"""

DMA_PROCESS = """
active [NDMA] proctype DmaEngine() {
  int src_dev, src_core, src_sem, src_buf_base, src_buf_len;
  int dst_dev, dst_core, dst_sem, dst_buf_base, dst_buf_len;
  do
    :: skip;
       end: dma_queue?DMA(src_dev, src_core, src_sem, src_buf_base, src_buf_len, dst_dev, dst_core, dst_sem, dst_buf_base, dst_buf_len);
       d_step {
         printf("DMA read done: [%d, %d)@{%d, %d} (%d++)\\n", src_buf_base, src_buf_base + src_buf_len, src_dev, src_core, src_sem);
         int i;
         for (i : src_buf_base .. src_buf_base + src_buf_len - 1) {
           buf_readers(i, src_dev, src_core)--;
         }
         sems(src_sem, src_dev, src_core)++;
       }  // Read complete
       d_step {
         printf("DMA write done: [%d, %d)@{%d, %d} (%d++)\\n", dst_buf_base, dst_buf_base + dst_buf_len, dst_dev, dst_core, dst_sem);
         int i;
         for (i : dst_buf_base .. dst_buf_base + dst_buf_len - 1) {
           buf_written(i, dst_dev, dst_core)--;
         }
         sems(dst_sem, dst_dev, dst_core)++;
       }  // Write complete
  od
}
"""

class PrintCtx:
  MAX_REF_UNROLL = 8

  def __init__(self, iteration_bounds):
    self.level = 1
    self.num_semaphores = 0
    self.num_buffers = 0
    self.locals = []
    self.counter = itertools.count()
    self.env: dict[ir.Value, Var | int] = {}
    self.program_ids = tuple(f"pid{i}" for i in range(len(iteration_bounds)))
    self.device_id = "dev_id"

    # TODO(apaszke): Clean up core_id! This is not a visible detail in the Mosaic
    # programming model.
    self.emit(None, "int core_id = 0")
    # Reconstruct device id and program ids from the pid.
    self.emit(None, "int dev_id")
    if iteration_bounds:
      self.emit(None, f"int {', '.join(self.program_ids)}")
    with self.block("d_step {", "}"):
      idx = "_pid"
      program_ids = []
      for i, b in reversed(list(enumerate(iteration_bounds))):
        program_ids.append(self.emit(None, f"pid{i} = {idx} % {b}"))
        idx = self.emit("int", f"{idx} / {b}")
      self.emit(None, f"dev_id = {idx}")

  def emit_global_ref(self, shape: Sequence[int]):
    slots = 1
    if shape and shape[0] <= self.MAX_REF_UNROLL:
      slots = shape[0]
    base = self.num_buffers
    self.num_buffers += slots
    return GlobalRefModel(base, slots)

  def emit_global_semaphore_ref(self, shape: Sequence[int]):
    count = math.prod(shape)
    base = self.num_semaphores
    self.num_semaphores += count
    return GlobalSemaphoreModel(base, count)

  def _indent(self, text: str) -> str:
    return textwrap.indent(text, "  " * self.level)

  def emit(self, ty, expr):
    name = None
    if ty is not None:
      name = "l" + str(next(self.counter))
      expr = f"{ty} {name} = {expr}"
    self.locals.append(self._indent(expr) + ";\n")
    return name

  def comment(self, comment):
    self.locals.append(self._indent(f"/* {comment} */\n"))

  @contextlib.contextmanager
  def block(self, begin: str, end: str):
    self.locals.append(self._indent(begin) + "\n")
    self.level += 1
    yield
    self.level -= 1
    self.locals.append(self._indent(end) + "\n")

  @contextlib.contextmanager
  def comment_if_emitted(self, comment):
    self.comment(comment)
    yield
    self.comment(comment)
    if self.locals[-1] == self.locals[-2]:
      self.locals.pop()
    self.locals.pop()

  def get(self, value: ir.Value, default: Any = _UNSPECIFIED):
    if default is _UNSPECIFIED:
      return self.env[value]
    else:
      return self.env.get(value, default)

  def set(self, value: ir.Value, model_value: Any):
    self.env[value] = model_value

  def get_model(
      self,
      has_barrier_sems: bool,
      num_devices: int,
      num_cores: int,
      parallel_iteration_bounds: Sequence[int],
  ) -> str:
    result = io.StringIO()
    result.write(f"#define NDEVICE {num_devices}\n")
    result.write("#define NCORE 1\n")
    result.write(f"byte _buf_readers[{self.num_buffers}*NDEVICE*NCORE] = 0;\n")
    result.write(f"bool _buf_written[{self.num_buffers}*NDEVICE*NCORE] = 0;\n")
    result.write(f"byte _sems[{self.num_semaphores}*NDEVICE*NCORE] = 0;\n")
    if has_barrier_sems:
      result.write("byte _barrier_sems[NDEVICE*NCORE] = 0;\n")
    result.write(PREAMBLE)
    result.write("\n")
    parallel_threads = math.prod(parallel_iteration_bounds)
    result.write(f"active [NDEVICE*{parallel_threads}] proctype Kernel() {{\n")
    for l in self.locals:
      result.write(l)
    result.write("}\n")
    result.write(DMA_PROCESS)
    return result.getvalue()


def resolve_location(location):
  if location is None:
    location = [None, None]
  else:
    location = list(location)
  if location[0] is None:
    location[0] = "dev_id"
  if location[1] is None:
    location[1] = "core_id"
  return tuple(location)


@dataclasses.dataclass(frozen=True)
class GlobalRefModel:
  """A model of a memory reference.

  When a reference has a small leading dimension, it might be represented by
  multiple slots in the reference array. Its region starts at base (that can be
  dynamic) and has the given length (always static).
  """
  base: Any
  length: int

  def readers_at(self, location):
    dev, core = resolve_location(location)
    return [f"buf_readers({self.base} + {i}, {dev}, {core})" for i in range(self.length)]

  def written_at(self, location):
    dev, core = resolve_location(location)
    return [f"buf_written({self.base} + {i}, {dev}, {core})" for i in range(self.length)]


@dataclasses.dataclass(frozen=True)
class GlobalSemaphoreModel:
  """A model of a semaphore reference.

  Semaphore arrays are always fully unrolled and are represented by a contiguous
  subset of the global semaphore array.
  """
  base: Any
  length: int

  def at(self, location):
    dev, core = resolve_location(location)
    return f"sems({self.base}, {dev}, {core})"


@dataclasses.dataclass(frozen=True)
class GlobalBarrierSemaphoreModel:
  def at(self, location):
    dev, core = resolve_location(location)
    return f"barrier_sems({dev}, {core})"


def _print_op(ctx, op):
  match op.name:
    case "tpu.region":
      _print_block(ctx, op.body)
    case "tpu.device_id":
      return ctx.device_id
    case "arith.constant":
      if ir.IntegerType.isinstance(op.result.type):
        return str(ir.IntegerAttr(op.value).value)
      else:
        return
    case "tpu.sem_signal":
      location = resolve_location((ctx.get(op.device_id, None), ctx.get(op.core_id, None)))
      sem_model = ctx.get(op.semaphore)
      sem = sem_model.at(location)
      amount = ctx.get(op.amount)
      if isinstance(sem_model, GlobalBarrierSemaphoreModel):
        ctx.emit(None, f'printf("Signal: BARRIER@{{%d, %d}} += %d\\n", {location[0]}, {location[1]}, {amount})')
      else:
        ctx.emit(None, f'printf("Signal: %d@{{%d, %d}} += %d\\n", {sem_model.base}, {location[0]}, {location[1]}, {amount})')
      ctx.emit(None, f"d_step {{ {sem} = {sem} + {amount} }}")
    case "tpu.sem_wait":
      sem_model = ctx.get(op.semaphore)
      sem = sem_model.at(location=None)
      amount = ctx.get(op.amount)
      ctx.emit(None, f"atomic {{ {sem} >= {amount}; {sem} = {sem} - {amount} }}")
      if isinstance(sem_model, GlobalBarrierSemaphoreModel):
        ctx.emit(None, f'printf("Wait done: BARRIER -= %d\\n", {amount})')
      else:
        ctx.emit(None, f'printf("Wait done: %d -= %d\\n", {sem_model.base}, {amount})')
    case "tpu.enqueue_dma":
      dst_location = resolve_location((ctx.get(op.device_id, None), ctx.get(op.core_id, None)))
      src = ctx.get(op.source)
      src_sem = ctx.get(op.source_semaphore)
      dst = ctx.get(op.target)
      dst_sem = ctx.get(op.target_semaphore)
      src_readonly = "\n    && ".join(is_written + " == 0" for is_written in src.written_at(None))
      dst_unused = "\n    && ".join(
          is_written + " == 0"
          for is_written in itertools.chain(
              dst.written_at(dst_location), dst.readers_at(dst_location)
          )
      )
      ctx.emit(
          None,
          'printf("DMA: [%d, %d)@{%d, %d} -> [%d, %d)@{%d, %d}\\n",'
          f" {src.base}, {src.base} + {src.length}, dev_id, core_id,"
          f" {dst.base}, {dst.base} + {dst.length}, {dst_location[0]},"
          f" {dst_location[1]})",
      )
      with ctx.block("d_step {", "}"):
        ctx.emit(None, f"assert({src_readonly});  // Source is not written to.")
        ctx.emit(None, f"assert({dst_unused});  // Destination is unused.")
        for r in src.readers_at(None):
          ctx.emit(None, f"{r}++")
          for w in dst.written_at(dst_location):
            ctx.emit(None, f"{w} = 1")
      ctx.emit(
          None,
          f"dma_queue!DMA(dev_id, core_id, {src_sem.base}, {src.base},"
          f" {src.length}, {dst_location[0]}, {dst_location[1]},"
          f" {dst_sem.base}, {dst.base}, {dst.length})",
      )
    case "tpu.wait_dma":
      sem_model = ctx.get(op.semaphore)
      sem = sem_model.at(location=None)
      ctx.emit(None, f"atomic {{ {sem} >= 1; {sem} = {sem} - 1 }}")
      ctx.emit(None, f'printf("Awaited DMA: %d\\n", {sem_model.base})')
    case "tpu.sem_barrier":
      return GlobalBarrierSemaphoreModel()
    case "tpu.memref_slice":
      result = ctx.get(op.mem_ref, None)
      if result is None:
        return NotImplemented
      src_shape = ir.MemRefType(op.mem_ref.type).shape
      dst_shape = ir.MemRefType(op.result.type).shape
      dynamic = ir.ShapedType.get_dynamic_size()
      # We always unroll semaphore references entirely, and we need to be
      # faithful when slicing them.
      if isinstance(result, GlobalSemaphoreModel):
        # We only support contiguous slices of semaphore arrays at the moment.
        seen_nontrivial_unequal = False
        for s, d in zip(src_shape, dst_shape):
          if d == 1:
            continue
          if s != d:
            if seen_nontrivial_unequal:
              raise NotImplementedError("Non-contiguous slices of semaphore arrays")
            seen_nontrivial_unequal = True
        strides = []
        stride = 1
        for s in src_shape[::-1]:
          strides.append(stride)
          stride *= s
        strides = reversed(strides)
        indices = [ctx.get(idx) for idx in op.base_idx]
        linear_offset = " + ".join(f"{idx} * {s}" for idx, s in zip(indices, strides))
        return GlobalSemaphoreModel(
            base=f"{result.base} + {linear_offset}", length=math.prod(dst_shape)
        )
      else:
        assert isinstance(result, GlobalRefModel)
        major_idx = ctx.get(op.base_idx[0], None)
        if (not src_shape or src_shape[0] == dynamic or dst_shape[0] == dynamic
            or result.length == 1 or major_idx is None):
          return result
        return GlobalRefModel(f"{result.base} + {major_idx}", dst_shape[0])
    case "tpu.memref_squeeze":
      result = ctx.get(op.input, None)
      return NotImplemented if result is None else result
    case "tpu.assume_multiple":
      result = ctx.get(op.value, None)
      return NotImplemented if result is None else result
    case "arith.addi":
      return bin_op(ctx, "int", "+", *op.operands)
    case "arith.subi":
      return bin_op(ctx, "int", "-", *op.operands)
    case "arith.muli":
      return bin_op(ctx, "int", "*", *op.operands)
    case "arith.remsi":
      # TODO(apaszke): Make sure this has right semantics for negative integers.
      return bin_op(ctx, "int", "%", *op.operands)
    case "arith.divsi":
      return bin_op(ctx, "int", "/", *op.operands)
    case "arith.andi":
      return bin_op(ctx, _model_type(op.result.type), "&", *op.operands)
    case "arith.select":
      cond, if_true, if_false = map(lambda o: ctx.get(o, None), op.operands)
      if cond is None or if_true is None or if_false is None:
        return NotImplemented
      result_ty = _model_type(op.result.type)
      return ctx.emit(result_ty, f"({cond} -> {if_true} : {if_false})")
    case "arith.index_cast":
      model = ctx.get(op.operands[0], None)
      return ctx.emit("int", model) if model is not None else NotImplemented
    case "arith.cmpi":
      match op.predicate.value:
        case arith.CmpIPredicate.eq:
          return bin_op(ctx, "bool", "==", *op.operands)
        case arith.CmpIPredicate.ne:
          return bin_op(ctx, "bool", "!=", *op.operands)
        case arith.CmpIPredicate.slt:
          return bin_op(ctx, "bool", "<", *op.operands)
        case arith.CmpIPredicate.sle:
          return bin_op(ctx, "bool", "<=", *op.operands)
        case arith.CmpIPredicate.sgt:
          return bin_op(ctx, "bool", ">", *op.operands)
        case arith.CmpIPredicate.sge:
          return bin_op(ctx, "bool", ">=", *op.operands)
      return bin_op(ctx, "bool", "/", *op.operands)
    case "tpu.trace_start":
      ctx.comment(op.message.value)
    case "tpu.assume_multiple":
      # TODO(apaszke): Add an assertion
      return ctx.get(op.value, NotImplemented)
    case "verification.pretend":
      read_refs = []
      for o in op.operands:
        if (model := ctx.get(o, None)) is None:
          raise ValueError(f"Could not model the read of {o}")
        read_refs.append(model)
      with ctx.block("d_step {", "}"):  # Start reading
        for r in read_refs:
          for loc in r.written_at(None):
            ctx.emit(None, f"assert(!{loc})")
          for loc in r.readers_at(None):
            ctx.emit(None, f"{loc}++")
      with ctx.block("d_step {", "}"):  # Stop reading
        for r in read_refs:
          for loc in r.readers_at(None):
            ctx.emit(None, f"{loc}--")
    case "vector.load":
      ref = ctx.get(op.operands[0])
      assert isinstance(ref, GlobalRefModel)
      if (first_idx := ctx.get(op.operands[1], None)) is not None:
        leading_load_len = ir.VectorType(op.result.type).shape[0]
        ref = GlobalRefModel(f"{ref.base} + {first_idx}", leading_load_len)
      with ctx.block("d_step {", "}"):  # Start reading
        for loc in ref.written_at(None):
          ctx.emit(None, f"assert(!{loc})")
        for loc in ref.readers_at(None):
          ctx.emit(None, f"{loc}++")
      with ctx.block("d_step {", "}"):  # Stop reading
        for loc in ref.readers_at(None):
          ctx.emit(None, f"{loc}--")
      return NotImplemented  # We don't model the result of the load.
    case "vector.store":
      ref = ctx.get(op.operands[1])  # Stored value goes first
      assert isinstance(ref, GlobalRefModel)
      if (first_idx := ctx.get(op.operands[2], None)) is not None:
        leading_store_len = ir.VectorType(op.operands[0].type).shape[0]
        ref = GlobalRefModel(f"{ref.base} + {first_idx}", leading_store_len)
      with ctx.block("d_step {", "}"):  # Start writing
        for loc in ref.readers_at(None):
          ctx.emit(None, f"assert(!{loc})")
        for loc in ref.written_at(None):
          ctx.emit(None, f"assert(!{loc})")
          ctx.emit(None, f"{loc} = 1")
      with ctx.block("d_step {", "}"):  # Stop reading
        for loc in ref.written_at(None):
          ctx.emit(None, f"{loc} = 0")
    case "scf.for":
      carrys = [
          ctx.emit("int", ctx.get(arg))
          if ir.IntegerType.isinstance(arg.type) else None
          for arg in op.initArgs
      ]
      bounds = (op.lowerBound, op.upperBound, op.step)
      lower, upper, step = bound_models = map(ctx.get, bounds)
      for model, v in zip(bound_models, bounds):
        if model is None:
          raise ValueError(f"Could not model loop bound or step: {v}")
      induction_var = ctx.emit("int", lower)
      with ctx.block("do", "od"):
        ctx.emit(None, f":: {induction_var} < {upper}; ")
        ctx.set(op.induction_variable, induction_var)
        for c, arg in zip(carrys, op.inner_iter_args, strict=True):
          if c is not None:
            ctx.set(arg, c)
        _print_block(ctx, op.body)
        terminator = op.body.operations[len(op.body.operations) - 1]
        new_carrys = terminator.operands
        with ctx.block("d_step {", "}"):
          for c, new in zip(carrys, new_carrys, strict=True):
            if c is not None:
              ctx.emit(None, f"{c} = {ctx.get(new)}")
          ctx.emit(None, f"{induction_var} = {induction_var} + {step}")
        ctx.emit(None, ":: else -> break")
      ctx.emit(None, "skip")  # To avoid "Jump into d_step sequence errors"
      if len(carrys) == 1:
        return carrys[0]
      else:
        return tuple(carrys)
    case "scf.if":
      if op.results:
        raise NotImplementedError
      if (condition := ctx.get(op.condition, None)) is None:
        raise ValueError(f"Could not model branch condition: {op.condition}")
      with ctx.block("if", "fi"):
        ctx.emit(None, f":: ({condition})")
        _print_block(ctx, op.then_block)
        if op.regions[1].blocks:
          ctx.emit(None, ":: else")
          _print_block(ctx, op.else_block)
        else:
          ctx.emit(None, ":: else -> skip")
    case _:
      if not op.regions:
        return NotImplemented
      raise NotImplementedError("Must handle all ops with regions")


def bin_op(ctx, result_ty, op, lhs, rhs):
  lhs = ctx.get(lhs, None)
  rhs = ctx.get(rhs, None)
  if lhs is None or rhs is None:
    return NotImplemented
  return ctx.emit(result_ty, f"{lhs} {op} {rhs}")


def _model_type(ty):
  if ir.IntegerType.isinstance(ty):
    if ir.IntegerType(ty).width == 1:
      return "bool"
    else:
      return "int"
  else:
    raise NotImplementedError(ty)


def _print_block(ctx, block):
  for op in block:
    try:
      with ctx.comment_if_emitted(op.OPERATION_NAME):
        results = _print_op(ctx, op)
    except Exception as e:
      raise RuntimeError(f"Failed to print op: {op}") from e
    if results is NotImplemented:
      continue
    if not op.results:
      assert results is None or results == ()
    elif len(op.results) > 1:
      raise NotImplementedError(op)
    else:
      ctx.set(op.result, results)


def export_promela_model(
    module, num_devices: int, num_cores_per_device: int
) -> str:
  with module.context:
    _, uses_barrier_semaphores = tpu.private_has_communication(module.operation)
    # Clone the module and simplify it to make the model smaller and simpler.
    module = ir.Module.parse(module.operation.get_asm(binary=True))
    passes = ["canonicalize", "cse"]
    pipeline = PassManager.parse(f"builtin.module({','.join(passes)})")
    pipeline.run(module.operation)
    main_str_attr = ir.StringAttr.get("main")
    for f in module.body:
      if getattr(f, "name", None) == main_str_attr:
        break
    else:
      raise ValueError("No main function found")
    assert isinstance(f, func.FuncOp)

    iteration_bounds: Sequence[int] = ()
    if "iteration_bounds" in f.attributes:
      iteration_bounds = ir.DenseI64ArrayAttr(f.attributes["iteration_bounds"])  # type: ignore
      dynamic = ir.ShapedType.get_dynamic_size()
      if any(b == dynamic for b in iteration_bounds):
        raise ValueError("Dynamic iteration bounds not supported")

      dimension_semantics = ir.ArrayAttr(f.attributes["dimension_semantics"])

      parallel = ir.Attribute.parse("#tpu.dimension_semantics<parallel>")
      if any(s != parallel for s in dimension_semantics):
        raise NotImplementedError("Non-parallel dimensions not supported")

    num_scalar_prefetch = 0
    if "scalar_prefetch" in f.attributes:
      num_scalar_prefetch = ir.IntegerAttr(f.attributes["scalar_prefetch"]).value

    (entry_block,) = f.body
    ctx = PrintCtx(iteration_bounds)
    sem_ty = ir.Type.parse("!tpu.semaphore")
    dma_sem_ty = ir.Type.parse("!tpu.dma_semaphore")
    program_id_args, prefetch_args, other_args = split_list(
        entry_block.arguments, [len(iteration_bounds), num_scalar_prefetch]
    )
    for arg, model in zip(program_id_args, ctx.program_ids, strict=True):
      ctx.set(arg, model)
    del prefetch_args  # We ignore prefetch_args
    for arg in other_args:
      if ir.MemRefType.isinstance(arg.type):
        ty = ir.MemRefType(arg.type)
        if ty.element_type == sem_ty or ty.element_type == dma_sem_ty:
          ctx.set(arg, ctx.emit_global_semaphore_ref(ty.shape))
        else:
          ctx.set(arg, ctx.emit_global_ref(ty.shape))
    _print_block(ctx, entry_block)
    return ctx.get_model(
        uses_barrier_semaphores, num_devices, num_cores_per_device, iteration_bounds
    )


assume_p = jax_core.Primitive("assume_for_verification")
assume_p.def_impl(lambda x, y: x)

@assume_p.def_abstract_eval
def _assume_abstract_eval(x, y):
  assert jax_core.typematch(x, y)
  return x

def _assume_lowering(ctx: lowering.LoweringRuleContext, x, y):
  return y if ctx.lowering_context.for_verification else x

lowering.lowering_rules[assume_p] = _assume_lowering  # type: ignore

def assume(normally, *, when_verifying):
  return assume_p.bind(normally, when_verifying)


pretend_p = jax_core.Primitive("pretend_for_verification")
pretend_p.multiple_results = True

@pretend_p.def_abstract_eval
def _pretend_abstract_eval(*_, **params):
  del params  # Unused.
  return ()

def _pretend_lowering(ctx: lowering.LoweringRuleContext, *flat_args, tree):
  if ctx.lowering_context.for_verification:
    (base_read_refs, transforms) = tree_util.tree_unflatten(tree, flat_args)
    read_ref_avals, _ = tree_util.tree_unflatten(tree, ctx.avals_in)
    block_shapes, _ = tree_util.tree_unflatten(tree, ctx.block_shapes)
    read_refs = [
        lowering._index_ref(ref, aval, block_shape, indexer)[0]
        for ref, aval, block_shape, indexer in zip(
            base_read_refs,
            read_ref_avals,
            block_shapes,
            transforms,
            strict=True,
        )
    ]
    ir.Operation.create("verification.pretend", operands=read_refs)
  return ()

lowering.lowering_rules[pretend_p] = _pretend_lowering  # type: ignore

def pretend(read_refs):
  refs, transforms = unzip2(
      primitives._get_ref_and_transforms(r) for r in read_refs
  )
  flat_args, tree = tree_util.tree_flatten((refs, transforms))
  return pretend_p.bind(*flat_args, tree=tree)


def skip(f):
  """Skips the verification of the given function."""
  def wrapper(*args, **kwargs):
    is_not_verifying = assume(normally=1, when_verifying=0)
    lax.cond(is_not_verifying)(lambda: f(*args, **kwargs))
  return wrapper


def define_model(model):
  """Replaces a function with its simplified model during verification."""
  def decorator(f):
    def wrapper(*args, **kwargs):
      lax.cond(
          assume(normally=1, when_verifying=0),
          lambda: f(*args, **kwargs),
          lambda: model(*args, **kwargs),
      )
    return wrapper
  return decorator
