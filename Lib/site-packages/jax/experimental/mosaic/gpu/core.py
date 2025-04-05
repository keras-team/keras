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

from collections.abc import Sequence
import contextlib
import ctypes
import dataclasses
import enum
import hashlib
import math
import os
import pathlib
import time
from typing import Any, Generic, TypeVar
import weakref

import jax
from jax._src.interpreters import mlir
from jax._src.lib import mosaic_gpu_dialect as dialect
from jaxlib.mlir import ir
from jaxlib.mlir import passmanager
from jaxlib.mlir.dialects import builtin
from jaxlib.mlir.dialects import func
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import nvvm
import numpy as np

if dialect is not None:
  from . import dialect_lowering
  from . import layout_inference
else:
  dialect_lowering = None
  layout_inference = None

from . import profiler
from . import utils
from . import launch_context

# mypy: ignore-errors

# MLIR can't find libdevice unless we point it to the CUDA path
# TODO(apaszke): Unify with jax._src.lib.cuda_path
CUDA_ROOT = "/usr/local/cuda"
if os.environ.get("CUDA_ROOT") is None:
  os.environ["CUDA_ROOT"] = CUDA_ROOT
else:
  CUDA_ROOT = os.environ["CUDA_ROOT"]

PTXAS_PATH = os.path.join(CUDA_ROOT, "bin/ptxas")
NVDISASM_PATH = os.path.join(CUDA_ROOT, "bin/nvdisasm")

# This tracks the latest Mosaic GPU IR version with a monthly delay.
FWD_COMPAT_IR_VERSION = 1

c = utils.c  # This is too common to fully qualify.


RUNTIME_PATH = None
try:
  from jax._src.lib import mosaic_gpu as mosaic_gpu_lib

  RUNTIME_PATH = (
      pathlib.Path(mosaic_gpu_lib._mosaic_gpu_ext.__file__).parent
      / "libmosaic_gpu_runtime.so"
  )
except ImportError:
  pass

if RUNTIME_PATH and RUNTIME_PATH.exists():
  # Set this so that the custom call can find it
  os.environ["MOSAIC_GPU_RUNTIME_LIB_PATH"] = str(RUNTIME_PATH)


mosaic_gpu_p = jax._src.core.Primitive("mosaic_gpu_p")
mosaic_gpu_p.multiple_results = True


@mosaic_gpu_p.def_abstract_eval
def _mosaic_gpu_abstract_eval(*_, module, out_types):
  del module  # Unused.
  return [jax._src.core.ShapedArray(t.shape, t.dtype) for t in out_types]

# TODO(apaszke): Implement a proper system for managing kernel lifetimes
KNOWN_KERNELS = {}


def _mosaic_gpu_lowering_rule(
    ctx,
    *args,
    module,
    out_types,
    input_output_aliases: tuple[tuple[int, int], ...] = (),
):
  assert len(out_types) == len(ctx.avals_out)
  module = _run_serde_pass(
      module,
      serialize=True,
      ir_version=FWD_COMPAT_IR_VERSION if ctx.is_forward_compat() else None,
  )
  module_asm = module.operation.get_asm(binary=True, enable_debug_info=True)
  kernel_id = hashlib.sha256(module_asm).digest()
  # Note that this is technically only a half measure. Someone might load a
  # compiled module with a hash collision from disk. But that's so unlikely with
  # SHA256 that it shouldn't be a problem.
  if (kernel_text := KNOWN_KERNELS.get(kernel_id, None)) is not None:
    if kernel_text != module_asm:
      raise RuntimeError("Hash collision!")
  else:
    KNOWN_KERNELS[kernel_id] = module_asm
  op = mlir.custom_call(
      "mosaic_gpu",
      result_types=[mlir.aval_to_ir_type(aval) for aval in ctx.avals_out],
      operands=args,
      operand_layouts=[list(reversed(range(a.ndim))) for a in ctx.avals_in],
      result_layouts=[list(reversed(range(a.ndim))) for a in ctx.avals_out],
      backend_config=kernel_id + module_asm,
      operand_output_aliases=dict(input_output_aliases),
  )
  return op.results


mlir.register_lowering(mosaic_gpu_p, _mosaic_gpu_lowering_rule, "cuda")


# ShapeTrees currently can not contain unions.
ShapeTree = Any
RefTree = Any
T = TypeVar('T')


@dataclasses.dataclass(frozen=True)
class Union(Generic[T]):
  members: Sequence[T]

  def __iter__(self):
    return iter(self.members)

@dataclasses.dataclass(frozen=True)
class TMABarrier:
  num_barriers: int = 1

@dataclasses.dataclass(frozen=True)
class Barrier:
  arrival_count: int
  num_barriers: int = 1

@dataclasses.dataclass(frozen=True)
class ClusterBarrier:
  collective_dims: Sequence[gpu.Dimension]
  num_barriers: int = 1


def _count_buffer_bytes(shape_dtype: jax.ShapeDtypeStruct) -> int:
  return np.prod(shape_dtype.shape) * np.dtype(shape_dtype.dtype).itemsize


class ThreadSemantics(enum.Enum):
  """Semantics for the kernel's instruction stream."""

  Lane = enum.auto()
  Warpgroup = enum.auto()


def _construct_smem_reftree(
    cluster_shape: tuple[int, int, int],
    dynamic_smem: ir.Value,
    smem_buffers: ShapeTree,
    dynamic_smem_offset: int = 0,
) -> RefTree:
  index = ir.IndexType.get()
  i8 = ir.IntegerType.get_signless(8)
  smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
  flat_ref_tys, smem_buffer_tree = jax.tree.flatten(
      smem_buffers, is_leaf=lambda x: isinstance(x, Union)
  )
  smem_refs = []
  for ref_ty in flat_ref_tys:
    def get_barrier_ptr(num_barriers: int) -> ir.Value:
      nonlocal dynamic_smem_offset
      workgroup_nvptx_address_space = (
          utils.gpu_address_space_to_nvptx(gpu.AddressSpace.Workgroup)
      )
      smem_base_ptr = utils.memref_ptr(
          dynamic_smem, memory_space=workgroup_nvptx_address_space
      )
      smem_ptr_ty = ir.Type.parse(f"!llvm.ptr<{workgroup_nvptx_address_space}>")
      barrier_base_ptr = llvm.getelementptr(
          smem_ptr_ty, smem_base_ptr, [], [dynamic_smem_offset], i8
      )
      dynamic_smem_offset += num_barriers * utils.MBARRIER_BYTES
      return barrier_base_ptr
    match ref_ty:
      case Union(members):
        member_trees = [
            _construct_smem_reftree(cluster_shape, dynamic_smem, m, dynamic_smem_offset)
            for m in members
        ]
        # TODO(apaszke): This is quadratic, but it shouldn't matter for now...
        dynamic_smem_offset += _smem_tree_size(ref_ty)
        ref = Union(member_trees)
      case TMABarrier(num_barriers):
        ref = utils.BarrierRef.initialize(
            get_barrier_ptr(num_barriers), num_barriers, arrival_count=1
        )
      case Barrier(arrival_count, num_barriers):
        ref = utils.BarrierRef.initialize(
            get_barrier_ptr(num_barriers),
            num_barriers,
            arrival_count=arrival_count,
        )
      case ClusterBarrier(collective_dims, num_barriers):
        ref = utils.CollectiveBarrierRef.initialize(
            get_barrier_ptr(num_barriers),
            num_barriers,
            collective_dims,
            cluster_shape,
        )
      case _:
        mlir_dtype = utils.dtype_to_ir_type(ref_ty.dtype)
        tile_smem = memref.view(
            ir.MemRefType.get(ref_ty.shape, mlir_dtype, memory_space=smem),
            dynamic_smem, c(dynamic_smem_offset, index), [],
        )
        dynamic_smem_offset += _count_buffer_bytes(ref_ty)
        ref = tile_smem
    smem_refs.append(ref)
  return jax.tree.unflatten(smem_buffer_tree, smem_refs)


def _smem_tree_size(smem_buffers: ShapeTree) -> int:
  leaves = jax.tree.leaves(
      smem_buffers, is_leaf=lambda x: isinstance(x, Union)
  )
  size = 0
  for l in leaves:
    match l:
      case Union(members):
        size += max(_smem_tree_size(s) for s in members)
      case (
          TMABarrier(num_barriers)
          | ClusterBarrier(_, num_barriers=num_barriers)
          | Barrier(_, num_barriers=num_barriers)
      ):
        if size % utils.MBARRIER_BYTES:
          raise NotImplementedError("Misaligned barrier allocation")
        size += num_barriers * utils.MBARRIER_BYTES
      case _:
        size += _count_buffer_bytes(l)
  return size


# TODO(apaszke): Inline this
@contextlib.contextmanager
def _launch(
    token,
    grid: tuple[int, int, int],
    cluster: tuple[int, int, int],
    block: tuple[int, int, int],
    scratch_arr,
    smem_buffers: ShapeTree | Union[ShapeTree],
    profiler_spec: profiler.ProfilerSpec | None = None,
    maybe_prof_buffer: ir.Value | None = None,
):
  if (profiler_spec is None) != (maybe_prof_buffer is None):
    raise ValueError
  index = ir.IndexType.get()
  i32 = ir.IntegerType.get_signless(32)
  i8 = ir.IntegerType.get_signless(8)
  grid_vals = [c(i, index) for i in grid]
  block_vals = [c(i, index) for i in block]

  user_smem_bytes = _smem_tree_size(smem_buffers)

  smem_bytes = user_smem_bytes
  if profiler_spec is not None:
    smem_bytes += profiler_spec.smem_bytes(block=block)

  # TODO(cperivol): Query the shared memory size programmatically.
  if smem_bytes > 228 * 1024:
    raise ValueError(f"Mosaic GPU kernel exceeds available shared memory {smem_bytes=} > 228000")
  if math.prod(cluster) != 1:
    if len(cluster) != 3:
      raise ValueError("Clusters must be 3D")
    cluster_kwargs = {
        "clusterSize" + d: c(s, index) for s, d in zip(cluster, "XYZ")
    }
    for d, grid_size, cluster_size in zip("xyz", grid, cluster):
      if grid_size % cluster_size != 0:
        raise ValueError(
            f"Grid dimension {d} must be divisible by cluster dimension:"
            f" {grid_size} % {cluster_size} != 0"
        )
  else:
    cluster_kwargs = {}
  launch_op = gpu.LaunchOp(
      token.type, [token], *grid_vals, *block_vals,
      dynamicSharedMemorySize=c(smem_bytes, i32), **cluster_kwargs)
  launch_op.body.blocks.append(*([index] * (12 + 2 * len(cluster_kwargs))))  # Append an empty block
  smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
  with ir.InsertionPoint(launch_op.body.blocks[0]):
    dynamic_smem = gpu.dynamic_shared_memory(
        ir.MemRefType.get(
            (ir.ShapedType.get_dynamic_size(),), i8, memory_space=smem
        )
    )

    if profiler_spec:
      prof_smem = memref.view(
          ir.MemRefType.get(
              (profiler_spec.smem_i32_elements(block=block),),
              i32, memory_space=smem,
          ),
          dynamic_smem, c(user_smem_bytes, index), [],
      )
      prof = profiler.OnDeviceProfiler(
          profiler_spec, prof_smem, maybe_prof_buffer
      )
    else:
      prof = None

    ptr_ty = ir.Type.parse("!llvm.ptr")
    scratch_ptr = builtin.unrealized_conversion_cast([ptr_ty], [scratch_arr])
    ctx = launch_context.LaunchContext(launch_op, scratch_ptr, cluster, prof)
    with ctx.named_region("Init"):
      smem_ref_tree = _construct_smem_reftree(
          cluster, dynamic_smem, smem_buffers
      )
      # TODO(apaszke): Skip the following if no barriers were initialized.
      nvvm.fence_mbarrier_init()
      if math.prod(cluster) != 1:
        nvvm.cluster_arrive_relaxed(aligned=ir.UnitAttr.get())
        nvvm.cluster_wait(aligned=ir.UnitAttr.get())
      gpu.barrier()

    yield ctx, smem_ref_tree
    if prof is not None:
      prof.finalize(grid=grid, block=block)
    gpu.terminator()


def _lower_as_gpu_kernel(
    body,
    grid: tuple[int, int, int],
    cluster: tuple[int, int, int],
    block: tuple[int, int, int],
    in_shapes: tuple[Any, ...],
    out_shape,
    smem_scratch_shape: ShapeTree | Union[ShapeTree],
    module_name: str,
    kernel_name: str | None = None,
    prof_spec: profiler.ProfilerSpec | None = None,
):
  ptr_ty = ir.Type.parse("!llvm.ptr")
  token_ty = ir.Type.parse("!gpu.async.token")
  i32 = ir.IntegerType.get_signless(32)
  i64 = ir.IntegerType.get_signless(64)

  def _shape_to_ref_ty(shape: jax.ShapeDtypeStruct) -> ir.MemRefType:
    return ir.MemRefType.get(shape.shape, utils.dtype_to_ir_type(shape.dtype))

  in_ref_tys = [_shape_to_ref_ty(t) for t in in_shapes]

  unwrap_output_tuple = False
  if isinstance(out_shape, list):
    out_shape = tuple(out_shape)
  elif not isinstance(out_shape, tuple):
    out_shape = (out_shape,)
    unwrap_output_tuple = True
  out_ref_tys = [_shape_to_ref_ty(t) for t in out_shape]
  if prof_spec is not None:
    out_shape = (*out_shape, prof_spec.jax_buffer_type(grid, block))
    out_ref_tys.append(prof_spec.mlir_buffer_type(grid, block))

  module = ir.Module.create()
  attrs = module.operation.attributes
  attrs["sym_name"] = ir.StringAttr.get(module_name)
  if kernel_name is None:
    kernel_name = getattr(body, "__name__", "anonymous")

  # These are needed as nonlocal below.
  launch_ctx, scratch_arr = None, None
  with ir.InsertionPoint(module.body):
    _declare_runtime_functions()
    global_scratch = llvm.GlobalOp(
        ir.Type.parse("!llvm.array<0 x i8>"),  # We don't know the shape yet.
        "global_scratch",
        ir.Attribute.parse("#llvm.linkage<external>"),
        addr_space=ir.IntegerAttr.get(i32, 4),  # GPU constant memory.
    )
    @func.FuncOp.from_py_func(ptr_ty, ptr_ty, name=f"mosaic_gpu_{kernel_name}")
    def main(token_ptr, buffers):
      nonlocal launch_ctx, scratch_arr
      token = builtin.unrealized_conversion_cast([token_ty], [token_ptr])
      arg_refs = []
      for i, ref_ty in enumerate([*in_ref_tys, *out_ref_tys]):
        ptr = llvm.LoadOp(ptr_ty, llvm.GEPOp(ptr_ty, buffers, [], [i], ptr_ty))
        arg_refs.append(utils.ptr_as_memref(ptr, ir.MemRefType(ref_ty)))
      in_refs = arg_refs[:len(in_ref_tys)]
      out_refs = arg_refs[len(in_ref_tys):]
      prof_buffer = out_refs.pop() if prof_spec is not None else None
      empty_arr_ty = ir.Type.parse("!llvm.array<0 x i8>")
      scratch_alloc = llvm.AllocaOp(
          ptr_ty, c(1, i64), empty_arr_ty,
          alignment=launch_context.TMA_DESCRIPTOR_ALIGNMENT
      )
      scratch_arr = llvm.load(empty_arr_ty, scratch_alloc.result)
      with _launch(
          token, grid, cluster, block, scratch_arr, smem_scratch_shape,
          prof_spec, prof_buffer
      ) as (_launch_ctx, smem_refs):
        nonlocal launch_ctx
        launch_ctx = _launch_ctx
        body(launch_ctx, *in_refs, *out_refs, smem_refs)
    main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
  sym_tab = ir.SymbolTable(module.operation)
  sym_tab.insert(main.func_op)
  sym_tab.insert(global_scratch)
  module.operation.verify()

  return module, out_shape, unwrap_output_tuple, launch_ctx, scratch_arr


def _run_serde_pass(
    module: ir.Module, *, serialize: bool, ir_version: int | None = None
) -> ir.Module:
  module = ir.Module.parse(
      module.operation.get_asm(binary=True, enable_debug_info=True),
      context=module.context,
  )
  pipeline = passmanager.PassManager.parse(
      "builtin.module(mosaic_gpu-serde{serialize="
      + str(serialize).lower()
      + (f" target-version={ir_version}" if ir_version is not None else "")
      + "})",
      module.context,
  )
  allow_unregistered_dialects = module.context.allow_unregistered_dialects
  module.context.allow_unregistered_dialects = True
  try:
    pipeline.run(module.operation)
    module.operation.verify()
  finally:
    module.context.allow_unregistered_dialects = allow_unregistered_dialects
  return module


def _initialize_scratch(
    launch_ctx : launch_context.LaunchContext,
    scratch_arr: ir.Value,
    ):
  """
  Allocates and initializes the host buffer right before the launch. This needs
  to be done after all TMA descriptors have been recorded by the launch context.
  Only then we know what the scratch contains.

  When using the Mosaic GPU dialect, the necessary information is known only
  after the lowering passes have run.
  """
  with ir.InsertionPoint(scratch_arr.owner):
    gmem_scratch_bytes = launch_ctx.next_scratch_offset
    scratch_alloc_op = scratch_arr.owner.opview.addr.owner.opview
    scratch_arr_ty = ir.Type.parse(f"!llvm.array<{gmem_scratch_bytes} x i8>")
    scratch_alloc_op.elem_type = ir.TypeAttr.get(scratch_arr_ty)
    scratch_arr.set_type(scratch_arr_ty)
    for init_callback in launch_ctx.host_scratch_init:
      init_callback(scratch_alloc_op.result)

def _declare_runtime_functions():
  """Declares the runtime functions that can be used by the generated code."""
  ptr_ty = ir.Type.parse("!llvm.ptr")
  i64 = ir.IntegerType.get_signless(64)
  arg_tys = [ptr_ty, ptr_ty, i64, i64, ptr_ty, ptr_ty, i64, ptr_ty]
  init_tma_desc_type = ir.FunctionType.get(arg_tys, [])
  func.FuncOp(
      "mosaic_gpu_init_tma_desc", init_tma_desc_type, visibility="private"
  )


def as_gpu_kernel(
    body,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    in_shape,
    out_shape,
    smem_scratch_shape: ShapeTree | Union[ShapeTree],
    prof_spec: profiler.ProfilerSpec | None = None,
    cluster: tuple[int, int, int] = (1, 1, 1),
    module_name: str = "unknown",
    kernel_name: str | None = None,
    ir_version: int | None = None,
    thread_semantics: ThreadSemantics = ThreadSemantics.Lane,
):
  if isinstance(in_shape, list):
    in_shape = tuple(in_shape)
  elif not isinstance(in_shape, tuple):
    in_shape = (in_shape,)

  module, out_shape, unwrap_output_tuple, launch_ctx, scratch_arr = (
      _lower_as_gpu_kernel(
          body, grid, cluster, block, in_shape, out_shape, smem_scratch_shape,
          module_name, kernel_name, prof_spec
      )
  )

  if thread_semantics == ThreadSemantics.Warpgroup and dialect is not None:
    # Run Python lowering passes. The remaining passes will be run in C++ in
    # jax/jaxlib/mosaic/gpu/custom_call.cc
    layout_inference.infer_layout(module)  # pytype: disable=attribute-error
    dialect_lowering.lower_mgpu_dialect(module, launch_ctx)  # pytype: disable=attribute-error

  _initialize_scratch(launch_ctx, scratch_arr)
  module.operation.verify()

  expected_arg_treedef = jax.tree.structure(in_shape)
  def _check_args(*args):
    arg_treedef = jax.tree.structure(args)
    if arg_treedef != expected_arg_treedef:
      raise ValueError(
          f"Invalid argument structure: expected {expected_arg_treedef}, got"
          f" {arg_treedef}, ({args=})"
      )

  def bind(*args) -> Any:
    return mosaic_gpu_p.bind(*args, module=module, out_types=out_shape)

  if prof_spec is not None:
    @jax.jit
    def prof_kernel(*args):
      _check_args(*args)
      *results, prof_buffer = bind(*args)
      def dump_profile(prof_buffer):
        out_file = os.path.join(
            os.getenv("TEST_UNDECLARED_OUTPUTS_DIR"),
            f"{time.time_ns()}-trace.json",
        )
        try:
          with open(out_file, "x") as f:
            prof_spec.dump(prof_buffer, f, grid=grid, block=block)
        except FileExistsError:
          pass  # TODO: Retry
      jax.debug.callback(dump_profile, prof_buffer)
      return results[0] if unwrap_output_tuple else results
    return prof_kernel
  else:
    @jax.jit
    def kernel(*args):
      _check_args(*args)
      results = bind(*args)
      return results[0] if unwrap_output_tuple else results
    return kernel


def as_torch_gpu_kernel(
    body,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    in_shape,
    out_shape,
    smem_scratch_shape: ShapeTree | Union[ShapeTree],
    prof_spec: profiler.ProfilerSpec | None = None,
    cluster: tuple[int, int, int] = (1, 1, 1),
    module_name: str = "unknown",
    kernel_name: str | None = None,
    thread_semantics: ThreadSemantics = ThreadSemantics.Lane,
):
  try:
    import torch
  except ImportError:
    raise RuntimeError("as_torch_gpu_kernel requires PyTorch")
  torch.cuda.init()  # Make sure CUDA context is set up.

  if isinstance(in_shape, list):
    in_shape = tuple(in_shape)
  elif not isinstance(in_shape, tuple):
    in_shape = (in_shape,)

  flat_out_types, out_treedef = jax.tree.flatten(out_shape)
  expected_arg_treedef = jax.tree.structure(in_shape)

  module, out_shape, unwrap_output_tuple, launch_ctx, scratch_arr = (
      _lower_as_gpu_kernel(
          body, grid, cluster, block, in_shape, out_shape, smem_scratch_shape,
          module_name, kernel_name, prof_spec
      )
  )

  if thread_semantics == ThreadSemantics.Warpgroup and dialect is not None:
    # Run Python lowering passes. The remaining passes will be run in C++ in
    # jax/jaxlib/mosaic/gpu/custom_call.cc
    layout_inference.infer_layout(module)  # pytype: disable=attribute-error
    dialect_lowering.lower_mgpu_dialect(module, launch_ctx)  # pytype: disable=attribute-error

  _initialize_scratch(launch_ctx, scratch_arr)
  module.operation.verify()

  # Get our hands on the compilation and unload functions
  try:
    import jax_plugins.xla_cuda12 as cuda_plugin
  except ImportError:
    raise RuntimeError("as_torch_gpu_kernel only works with recent jaxlib builds "
                       "that use backend plugins")
  dll = ctypes.CDLL(cuda_plugin._get_library_path())
  compile_func = dll.MosaicGpuCompile
  compile_func.argtypes = [ctypes.c_void_p]
  compile_func.restype = ctypes.POINTER(ctypes.c_void_p)
  unload_func = dll.MosaicGpuUnload
  unload_func.argtypes = [compile_func.restype]
  unload_func.restype = None

  module_asm = module.operation.get_asm(binary=True, enable_debug_info=True)
  compiled = compile_func(ctypes.c_char_p(module_asm))
  if compiled is None:
    raise RuntimeError("Failed to compile the module")
  ctx, launch_ptr = compiled[0], compiled[1]
  ctx_ptr_ptr = ctypes.pointer(ctypes.c_void_p(ctx))
  launch = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(launch_ptr)

  def as_torch_dtype(dtype):
    # torch contains NumPy-compatible dtypes in its top namespace
    return getattr(torch, np.dtype(dtype).name)

  def apply(*args):
    flat_args, arg_treedef = jax.tree.flatten(args)
    if arg_treedef != expected_arg_treedef:
      raise ValueError(
          f"Invalid argument structure: expected {expected_arg_treedef}, got"
          f" {arg_treedef}, ({args=})"
      )

    # Construct a device pointer list like in the XLA calling convention
    buffers = (ctypes.c_void_p * (arg_treedef.num_leaves + out_treedef.num_leaves))()
    i = -1  # Define i in case there are no args
    device = 'cuda'
    for i, arg in enumerate(flat_args):
      buffers[i] = arg.data_ptr()
      device = arg.device
    flat_outs = []
    for i, t in enumerate(flat_out_types, i + 1):
      out = torch.empty(t.shape, dtype=as_torch_dtype(t.dtype), device=device)
      flat_outs.append(out)
      buffers[i] = out.data_ptr()
    # Allocate another buffer for args of the host-side program. This is sadly
    # the default MLIR calling convention.
    args_ptr = (ctypes.POINTER(ctypes.c_void_p) * 3)()
    args_ptr[0] = ctx_ptr_ptr
    args_ptr[1] = ctypes.pointer(torch.cuda.default_stream(device)._as_parameter_)
    args_ptr[2] = ctypes.cast(ctypes.pointer(ctypes.pointer(buffers)),
                              ctypes.POINTER(ctypes.c_void_p))
    launch(args_ptr)
    return jax.tree.unflatten(out_treedef, flat_outs)

  # Unload the compiled code when the Python function is destroyed.
  def unload(_):
    unload_func(compiled)
  apply.destructor = weakref.ref(apply, unload)

  return apply
