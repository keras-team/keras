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

import dataclasses
import enum
import functools
import itertools

import jax
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import vector
from jaxlib.mlir.dialects import nvvm
import numpy as np

import jax.experimental.mosaic.gpu as mgpu
from . import utils

# mypy: ignore-errors

c = mgpu.c
bytewidth = mgpu.bytewidth


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class WGMMAAccumulator:
  """A FragmentedArray that has is synchronized with the async proxy.

  This implies that it requires no additional synchronization when passed in
  as a WGMMA accumulator. In particular, when created from a
  FragmentedArray, the necessary synchronization is inserted at construction.
  """
  value: mgpu.FragmentedArray

  def __init__(self, *, _value: mgpu.FragmentedArray, _sync: bool = True):
    if _value.layout != mgpu.WGMMA_LAYOUT:
      raise ValueError("Only WGMMA layouts supported in WGMMAAccumulator")
    self.value = _value
    if _sync:
      self.value = wgmma_fence(_value)

  @classmethod
  def zero(cls, m, n, dtype=None, *, is_signed: bool | None = None):
    if m % 64 or n % 8:
      raise ValueError
    if is_signed is False:
      raise TypeError("PTX does not support unsigned WGMMA accumulators")
    f32 = ir.F32Type.get()
    if dtype is None:
      dtype = f32
    zero = arith.constant(dtype, ir.FloatAttr.get(dtype, 0.0))
    return cls(
        _value=mgpu.FragmentedArray.splat(
            zero, (m, n), mgpu.WGMMA_LAYOUT, is_signed=is_signed
        )
    )

  @classmethod
  def from_registers(cls, registers):
    return cls(_value=registers)

  def tree_flatten(self):
    return (self.value,), ()

  @classmethod
  def tree_unflatten(cls, aux, value):
    del aux
    return cls(_value=value[0], _sync=False)


def wgmma_encode(x: int):
  result = (x & 0x3FFFF) >> 4
  if result << 4 != x:
    raise ValueError("Cannot encode value in a WGMMA descriptor")
  return result


def llvm_add(x, y):
  return llvm.add(x, y, overflow_flags=llvm.IntegerOverflowFlags.none)


def create_descriptor(
    memref_arg,
    leading_byte_offset: int,
    stride_byte_offset: int,
    swizzle: int | None,
    memory_space: int | None = None,
):
  i64 = ir.IntegerType.get_signless(64)
  ptr_val = llvm.ptrtoint(i64, utils.memref_ptr(memref_arg, memory_space))
  if swizzle is None:
    swizzle_encoding = 0
  elif swizzle == 128:
    swizzle_encoding = 1
  elif swizzle == 64:
    swizzle_encoding = 2
  elif swizzle == 32:
    swizzle_encoding = 3
  else:
    raise NotImplementedError(swizzle)
  encoded_base_addr = llvm.LShrOp(
      llvm.AndOp(ptr_val, c(0x3FFFF, i64)), c(4, i64)
  )
  # We ignore the offset
  desc_const = (
      (wgmma_encode(leading_byte_offset) << 16)
      | (wgmma_encode(stride_byte_offset) << 32)
  )
  desc = llvm.or_(
      arith.shli(c(swizzle_encoding, i64), c(62, i64)), c(desc_const, i64)
  )
  desc = llvm.or_(encoded_base_addr, desc)
  return desc


def _unpack_i32(vec_ty, r):
  i32 = ir.IntegerType.get_signless(32)
  return vector.bitcast(
      vec_ty, vector.splat(ir.VectorType.get((1,), i32), r)
  )


def _supported_wgmma_types(dtype, abtype) -> bool:
  input_types_are = lambda ty: ty.isinstance(abtype)
  if ir.F32Type.isinstance(dtype):
    return any(input_types_are(ty) for ty in (ir.FloatTF32Type, ir.BF16Type, ir.F16Type))
  elif ir.F16Type.isinstance(dtype):
    return input_types_are(ir.F16Type)
  else:
    return False


def wgmma_m64(
    acc: np.ndarray,  # of register Values
    a,
    b_descriptor: ir.Value,
    a_transpose: bool | None,
    b_transpose: bool,
    a_k_stride: int | None,
    b_k_stride: int,
    n: int,
    swizzle: int,
    element_type: ir.Type,
):
  out_ty = ir.VectorType(acc.flat[0].type).element_type
  if not _supported_wgmma_types(out_ty, element_type):
    raise ValueError(f"Usupported wgmma types {(out_ty, element_type)=}")
  if n % 8:
    raise ValueError

  i32 = ir.IntegerType.get_signless(32)
  i64 = ir.IntegerType.get_signless(64)
  index = ir.IndexType.get()
  if b_k_stride % 16:
    raise ValueError
  # Only 16-bit types support transposes
  supports_transpose = bytewidth(element_type) == 2
  if not supports_transpose and (a_transpose or b_transpose):
    raise ValueError("Only f16 WGMMA supports transposes")
  if a_in_regs := isinstance(a, mgpu.FragmentedArray):
    if a.mlir_dtype != ir.F16Type.get() and a.mlir_dtype != ir.BF16Type.get():
      raise ValueError(f"Unsupported A register array dtype: {a.mlir_dtype}")
    # Column count must be equal to swizzle // bytewidth.
    if a.layout != mgpu.WGMMA_LAYOUT or a.shape != (64, swizzle // 2):
      raise ValueError("Unsupported A register array layout")
    if a_k_stride is not None or a_transpose is not None:
      raise ValueError("Unsupported WGMMA features with A in registers")
  else:
    if a_k_stride is None or a_k_stride % 16:
      raise ValueError
    if a_transpose is None:
      raise ValueError

  if ir.F32Type.isinstance(out_ty):
    num_acc_regs = n // 2
    out_ty_field = out_ty
    acc_regs = [  # pylint: disable=g-complex-comprehension
        vector.extractelement(reg, position=c(pos, index))
        for reg in acc.flat
        for pos in range(2)
    ]
    to_acc_vec_regs = functools.partial(_as_fragmented_reg_ndarray, dtype=out_ty, shape=acc.shape)
    acc_constraint = "f"
  elif ir.F16Type.isinstance(out_ty):
    num_acc_regs = n // 4
    out_ty_field = i32
    acc_regs = [_as_i32_reg(reg) for reg in acc.flat]
    vec_ty = ir.VectorType(acc.flat[0].type)
    to_acc_vec_regs = lambda regs : np.array([_unpack_i32(vec_ty, reg) for reg in regs]).reshape(acc.shape)
    acc_constraint = "r"
  else:
    raise ValueError(f"WGMMA instruciton only supports f32 and f16 out (got {out_ty})")

  num_imm_regs = 4 if supports_transpose else 2

  if a_in_regs:
    a_reg_constraints = ["r"] * 4  # 4x f16x2 registers
    num_imm_regs -= 1  # transpose not supported for a in registers
  else:
    a_reg_constraints = ["l"]  # descriptor
  # Reference for i/o aliasing: https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html
  # Seems like it's not actually documented in LLVM IR docs.
  reg_constraints_list = (
      [f"={acc_constraint}"] * num_acc_regs  # accumulator registers
      + [str(i) for i in range(num_acc_regs)]  # we alias outputs as inputs, too.
      + a_reg_constraints  # a descriptor / registers
      + ["l"] * 1  # b descriptor
      + ["n"] * (1 + num_imm_regs)  # literal constants
  )
  reg_constraints = ",".join(reg_constraints_list)

  reg_count = itertools.count()

  def take_regs(n):
    return (f"${i}" for i in itertools.islice(reg_count, n))

  acc_reg_vector = "{" + ",".join(take_regs(num_acc_regs)) + "}"
  for _ in take_regs(num_acc_regs):  # Ignore next entries: aliasing.
    pass
  if a_in_regs:
    a_regs = "{" + ",".join(take_regs(len(a_reg_constraints))) + "}"
  else:
    a_regs, = take_regs(1)
  b_desc_reg, use_out_reg = take_regs(2)
  imm_regs = ", ".join(take_regs(num_imm_regs))  # Immediate regs (scale, ...).
  assert next(reg_count) == len(reg_constraints_list)
  el_ty = element_type
  k_instr = 32 // bytewidth(element_type)
  wgmma_instr = (
      f"wgmma.mma_async.sync.aligned.m64n{n}k{k_instr}.{out_ty}.{el_ty}.{el_ty} "
      f"{acc_reg_vector}, {a_regs}, {b_desc_reg}, p, {imm_regs};"
  )
  ptx = f"{{ .reg .pred p; setp.ne.b32 p, {use_out_reg}, 0; {wgmma_instr} }}\n"

  def lc(x):
    return llvm.ConstantOp(i32, ir.IntegerAttr.get(i32, x)).result

  use_out = scale_a = scale_b = lc(1)
  imms = [use_out, scale_a, scale_b]
  if supports_transpose and a_transpose is not None:
    imms += [lc(int(a_transpose)), lc(int(b_transpose))]
  elif supports_transpose:
    imms += [lc(int(b_transpose))]
  if acc.ndim != 4 or acc.shape[0] != 1 or acc.shape[2:] != (2, 1):
    raise ValueError(acc.shape)
  acc_struct_type = ir.Type.parse(
      f"!llvm.struct<({','.join(str(out_ty_field) for _ in acc_regs)})>"
  )
  for i in range((swizzle // bytewidth(element_type)) // k_instr):
    # Slice out the relevant part of A or advance the A descriptor.
    if a_in_regs:
      a_slice = a[:, (i * 16) : ((i + 1) * 16)]
      a_args = [_as_i32_reg(v) for v in a_slice.registers.flat]
    else:
      if i > 0:
        a = llvm_add(
            a,
            llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, a_k_stride >> 4)),
        )
      a_args = [a]
    # Advance the B descriptor.
    if i > 0:
      b_descriptor = llvm_add(
          b_descriptor,
          llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, b_k_stride >> 4)),
      )
    assert len(a_args) == len(a_reg_constraints)
    acc_struct = llvm.inline_asm(
        acc_struct_type,
        [*acc_regs, *a_args, b_descriptor, *imms],
        ptx,
        reg_constraints,
        asm_dialect=0,
        has_side_effects=True,
    )
    acc_regs = [
        llvm.extractvalue(out_ty_field, acc_struct, [i]) for i in range(len(acc_regs))
    ]
  return to_acc_vec_regs(acc_regs)


class WGMMALayout(enum.Enum):
  ROW_MAJOR = enum.auto()
  COL_MAJOR = enum.auto()


# TODO(apaszke): Remove WGMMALayout. Make input shapes logical and infer
# transpositions from memref strides.
def wgmma(
    acc: WGMMAAccumulator,
    a,
    b,
    *,
    swizzle: int  = 128,
    # Order only applies within each tile!
    a_order: WGMMALayout | None = None,
    b_order: WGMMALayout = WGMMALayout.ROW_MAJOR,
):
  if a_in_regs := isinstance(a, mgpu.FragmentedArray):
    a_element_type = a.mlir_dtype
    a_shape = a.shape
  else:
    a_ty = ir.MemRefType(a.type)
    a_element_type = a_ty.element_type
    a_shape = a_ty.shape
  b_ty = ir.MemRefType(b.type)
  supported_types = {ir.F16Type.get(), ir.BF16Type.get(), ir.F32Type.get()}
  if a_element_type not in supported_types:
    raise ValueError(a_element_type)
  if b_ty.element_type not in supported_types:
    raise ValueError(b_ty.element_type)
  if (element_type := a_element_type) != b_ty.element_type:
    raise ValueError
  element_bytewidth = bytewidth(element_type)
  kn_tile = swizzle // element_bytewidth

  groups_k, groups_n = b_ty.shape[:2]
  k_group_size, n_group_size = (
      b_ty.shape[2:] if b_order == WGMMALayout.ROW_MAJOR else b_ty.shape[:1:-1]
  )
  # Note that while this technically allows n to be smaller than kn_tile,
  # the stride checks below will still enforce that the memory region is padded.
  # It might be possible to relax that requirement, but I haven't tested it.
  if n_group_size > kn_tile and n_group_size % kn_tile:
    raise ValueError(n_group_size, kn_tile)
  if k_group_size != kn_tile:
    raise ValueError(b_ty.shape)

  if a_in_regs:
    if a_element_type != ir.F16Type.get() and a_element_type != ir.BF16Type.get():
      raise ValueError(a_element_type)
    if a_shape[0] % 64 or a_shape[1] % kn_tile:
      raise ValueError(a_shape)
    if a_shape[1] // kn_tile != groups_k:
      raise ValueError(a_shape[1] // kn_tile, groups_k)
    groups_m = a_shape[0] // 64
    if a_order is not None:
      raise ValueError(
          "a_order can only be specified when A is in shared memory"
      )
  else:
    groups_m = a_shape[0]
    if a_shape[1] != groups_k:
      raise ValueError(a_shape[1], groups_k)
    if a_shape[2:] != [64, kn_tile]:
      raise ValueError(a_shape)
    if a_order is None:
      a_order = WGMMALayout.ROW_MAJOR

  if a_order == WGMMALayout.COL_MAJOR and swizzle != 128:
    # Not sure what the layout is like, since the tiles aren't square.
    raise NotImplementedError
  expected_acc_shape = (groups_m * 64, groups_n * n_group_size)
  if acc.value.shape != expected_acc_shape:
    raise ValueError(
        f"Accumulator shape mismatch: expected {expected_acc_shape}, got"
        f" {acc.value.shape}"
    )

  row_major = WGMMALayout.ROW_MAJOR
  col_major = WGMMALayout.COL_MAJOR
  tnsp_lbo = swizzle * (swizzle // 32)
  sbo = swizzle // 2
  a_desc_fields = dict(
      leading_byte_offset=(1 if a_order == row_major else tnsp_lbo) << 4,
      stride_byte_offset=sbo << 4,
      swizzle=swizzle,
      memory_space=3,
  )
  b_desc_fields = dict(
      leading_byte_offset=(tnsp_lbo if b_order == row_major else 1) << 4,
      stride_byte_offset=sbo << 4,
      swizzle=swizzle,
      memory_space=3,
  )
  wgmma_params = dict(
      a_transpose=a_order == col_major,
      b_transpose=b_order == row_major,
      a_k_stride=(2 if a_order == row_major else 128) << 4,
      b_k_stride=(swizzle if b_order == row_major else 2) << 4,
      n=(groups_n * n_group_size),
      swizzle=swizzle,
      element_type=ir.FloatTF32Type.get()
      if ir.F32Type.isinstance(element_type)
      else element_type,
  )
  if a_in_regs:
    wgmma_params["a_k_stride"] = wgmma_params["a_transpose"] = None

  if a_in_regs:
    a = wgmma_fence(a)  # Make sure the registers are ready.
    a_m_byte_stride = a_k_byte_stride = a_desc_base = None  # Silence pytype.
  else:
    a_desc_base = create_descriptor(a, **a_desc_fields)
    a_strides, _ = ir.MemRefType(a.type).get_strides_and_offset()
    a_byte_strides = [s * element_bytewidth for s in a_strides]
    a_m_byte_stride, a_k_byte_stride = a_byte_strides[:2]
    if a_byte_strides[2:] != [swizzle, element_bytewidth]:
      raise ValueError(a_byte_strides)
  b_desc_base = create_descriptor(b, **b_desc_fields)
  b_strides, _ = b_ty.get_strides_and_offset()
  b_byte_strides = [s * element_bytewidth for s in b_strides]
  b_k_byte_stride = b_byte_strides[0]
  if b_byte_strides[1:] != [swizzle * kn_tile, swizzle, element_bytewidth]:
    raise ValueError(b_byte_strides)

  i64 = ir.IntegerType.get_signless(64)
  new_acc_regs = acc.value.registers.copy()
  for mi in range(groups_m):
    for ki in range(groups_k):
      if a_in_regs:
        a_mk = a[mi * 64 : (mi + 1) * 64, ki * kn_tile : (ki + 1) * kn_tile]
      else:
        a_mk = llvm_add(
            a_desc_base,
            c(wgmma_encode(mi * a_m_byte_stride + ki * a_k_byte_stride), i64),
        )
      b_k = llvm_add(b_desc_base, c(wgmma_encode(ki * b_k_byte_stride), i64))
      new_acc_regs[mi : mi + 1] = wgmma_m64(
          new_acc_regs[mi : mi + 1], a_mk, b_k, **wgmma_params
      )
  return WGMMAAccumulator(
      _value=mgpu.FragmentedArray(
          _registers=new_acc_regs,
          _layout=mgpu.WGMMA_LAYOUT,
          _is_signed=acc.value.is_signed,
      ),
      _sync=False,
  )


def wgmma_fence(array: mgpu.FragmentedArray):
  """Fences the array construction from WGMMA instructions.

  LLVM treats in-register computation as pure and can move it after the fence,
  which is explicitly disallowed by the PTX programming model. For that reason,
  we insert an LLVM optimization barrier before the fence.
  """
  array = mgpu.optimization_barrier(array)
  nvvm.wgmma_fence_aligned()
  return array


def _as_fragmented_reg_ndarray(flat_regs, dtype: ir.Type, shape: tuple[int, ...]):
  vec_regs = []
  for first, second in zip(flat_regs[::2], flat_regs[1::2]):
    vec = llvm.mlir_undef(ir.VectorType.get((2,), dtype))
    vec = llvm.insertelement(vec, first, position=_lc(0))
    vec = llvm.insertelement(vec, second, position=_lc(1))
    vec_regs.append(vec)
  return np.asarray(vec_regs, dtype=object).reshape(shape)


def _as_i32_reg(v):
  i32 = ir.IntegerType.get_signless(32)
  return llvm.extractelement(
      vector.bitcast(ir.VectorType.get((1,), i32), v), _lc(0)
  )


def _lc(x):
  i32 = ir.IntegerType.get_signless(32)
  return llvm.ConstantOp(i32, ir.IntegerAttr.get(i32, x)).result
