# Copyright 2018 The JAX Authors.
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

# Shims that allow the XLA CPU backend to call scipy-provided LAPACK kernels
# via CustomCallWithLayout.

from collections.abc import Sequence
from enum import Enum
from typing import Optional

import numpy as np

import jaxlib.mlir.ir as ir  # pylint: disable=consider-using-from-import
import jaxlib.mlir.dialects.stablehlo as hlo

from jaxlib import xla_client

from .cpu import _lapack
from .cpu._lapack import schur
from .cpu._lapack import eig
from .hlo_helpers import (
    custom_call, hlo_u8, hlo_s32,
    ensure_hlo_s32, hlo_add,
    DimensionSize, ShapeTypePair, mk_result_types_and_shapes,
)

for _name, _value in _lapack.registrations().items():
  xla_client.register_custom_call_target(
      _name,
      _value,
      platform="cpu",
      api_version=(1 if _name.endswith("_ffi") else 0),
  )


def _char_attr(c):
  return ir.IntegerAttr.get(ir.IntegerType.get_unsigned(8), ord(c))


def _lapack_int_attr(value):
  return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), value)


def _enum_to_char_attr(e: Enum):
  return ir.IntegerAttr.get(ir.IntegerType.get_unsigned(8), e.value)


def _matrix_side_attr(*, left_side: bool):
  return _char_attr("L" if left_side else "R")


def _matrix_uplo_attr(*, lower: bool):
  return _char_attr("L" if lower else "U")


def _matrix_transpose_attr(*, transpose: bool, conjugate: bool):
  return _char_attr(("C" if conjugate else "T") if transpose else "N")


def _matrix_diagonal_attr(*, unit_diag: bool):
  return _char_attr("U" if unit_diag else "N")


def _svd_computation_attr(
    *, compute_uv: bool, full_matrices: Optional[bool] = True
):
  mode = "A"
  if full_matrices is None:
    full_matrices = True
  if not compute_uv:
    # We should assert that `full_matrices` is never True here.
    # This should never happen because `full_matrices` can only be computed when
    # `compute_uv` is True. However, at this point there are too many tests that
    # rely on this behavior.
    mode = "N"
  elif not full_matrices:
    mode = "S"
  return _char_attr(mode)


LAPACK_DTYPE_PREFIX = {
    np.float32: "s",
    np.float64: "d",
    np.complex64: "c",
    np.complex128: "z",
}


def prepare_lapack_call(fn_base, dtype):
  """Initializes the LAPACK library and returns the LAPACK target name."""
  _lapack.initialize()
  return build_lapack_fn_target(fn_base, dtype)


def build_lapack_fn_target(fn_base: str, dtype) -> str:
  """Builds the target name for a LAPACK function custom call."""
  try:
    prefix = (
        LAPACK_DTYPE_PREFIX.get(dtype, None) or LAPACK_DTYPE_PREFIX[dtype.type]
    )
    return f"lapack_{prefix}{fn_base}"
  except KeyError as err:
    raise NotImplementedError(err, f"Unsupported dtype {dtype}.") from err


# TODO(phawkins): it would be nice to avoid duplicating code for each type.

# ?trsm(left_side, lower, trans_a, diag, m, n, alpha, a, b):
# triangular solve
def trsm_hlo(ctx, dtype, alpha, a, b,
             left_side=False, lower=False, trans_a=False,
             conj_a=False, diag=False, *,
             b_shape_vals: tuple[DimensionSize, ...]):
  if conj_a and not trans_a:
    raise NotImplementedError("Conjugation without transposition not supported")
  fn_base = prepare_lapack_call(fn_base="trsm", dtype=dtype)
  b_type = ir.RankedTensorType(b.type)

  batch_dims_vals = b_shape_vals[:-2]
  num_bd = len(batch_dims_vals)
  scalar_layout = []
  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  result_types, result_shapes = mk_result_types_and_shapes(
      [(b_shape_vals, b_type.element_type)])

  if ctx.is_forward_compat():
    # The old TRSM kernel name is prefixed with "blas"
    fn = fn_base.replace("lapack", "blas", 1)
    m, n = b_shape_vals[-2:]
    batch_size_val = hlo_s32(1)
    for b_v in batch_dims_vals:
      batch_size_val = hlo.multiply(batch_size_val, ensure_hlo_s32(b_v))
    result_types, result_shapes = mk_result_types_and_shapes(
        [(b_shape_vals, b_type.element_type)]
    )
    return custom_call(
        fn,
        result_types=result_types,
        operands=[hlo_s32(int(left_side)), hlo_s32(int(lower)),
                  hlo_s32((2 if conj_a else 1) if trans_a else 0), hlo_s32(int(diag)),
                  ensure_hlo_s32(m), ensure_hlo_s32(n), batch_size_val,
                  alpha, a, b],
        operand_layouts=[scalar_layout] * 8 + [layout] * 2,
        result_layouts=[layout],
        operand_output_aliases={9: 0},
        result_shapes=result_shapes,
    ).results

  fn = fn_base + "_ffi"
  return custom_call(
      fn,
      result_types=result_types,
      operands=[a, b, alpha],
      operand_layouts=[layout] * 2 + [scalar_layout],
      result_layouts=[layout],
      operand_output_aliases={1: 0},
      result_shapes=result_shapes,
      backend_config={
          "side": _matrix_side_attr(left_side=left_side),
          "uplo": _matrix_uplo_attr(lower=lower),
          "trans_x": _matrix_transpose_attr(
              transpose=trans_a, conjugate=conj_a
          ),
          "diag": _matrix_diagonal_attr(unit_diag=diag),
      },
      api_version=4,
  ).results


# ?potrf: Cholesky decomposition

def potrf_hlo(ctx, dtype, a: ir.Value, *, lower=False,
              a_shape_vals: tuple[DimensionSize, ...]):
  a_type = ir.RankedTensorType(a.type)
  fn_base = prepare_lapack_call(fn_base="potrf", dtype=dtype)
  batch_dims_vals = a_shape_vals[:-2]
  num_bd = len(batch_dims_vals)
  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  info_layout = tuple(range(num_bd - 1, -1, -1))

  shape_type_pairs: Sequence[ShapeTypePair] = [
      (a_shape_vals, a_type.element_type),
      (batch_dims_vals, ir.IntegerType.get_signless(32))
  ]
  result_types, result_shapes = mk_result_types_and_shapes(shape_type_pairs)
  if ctx.is_forward_compat():
    fn = fn_base
    scalar_layout = []
    n = a_shape_vals[-1]
    batch_size_val = hlo_s32(1)
    for b_v in batch_dims_vals:
      batch_size_val = hlo.multiply(batch_size_val, ensure_hlo_s32(b_v))
    out = custom_call(
      fn,
      result_types=result_types,
      operands=[hlo_s32(int(lower)), batch_size_val, ensure_hlo_s32(n), a],
      operand_layouts=[scalar_layout] * 3 + [layout],
      result_layouts=[layout, info_layout],
      operand_output_aliases={3: 0},
      result_shapes=result_shapes,
  ).results
  else:
    fn = fn_base + "_ffi"
    out = custom_call(
        fn,
        result_types=result_types,
        operands=[a],
        operand_layouts=[layout],
        result_layouts=[layout, info_layout],
        operand_output_aliases={0: 0},
        result_shapes=result_shapes,
        backend_config={
            "uplo": _matrix_uplo_attr(lower=lower),
        },
        api_version=4,
    ).results
  return out[:2]


# # geev: Nonsymmetric eigendecomposition (eig)

def geev_hlo(ctx, dtype, input, *,
             input_shape_vals: tuple[DimensionSize, ...],  # input.shape as ir.Values
             jobvl=True, jobvr=True):
  # input_shape_vals are used for when input has dynamic shapes.
  _lapack.initialize()
  input_shape = ir.RankedTensorType(input.type).shape
  assert len(input_shape) >= 2
  n = input_shape_vals[-1]
  batch_dims_vals = input_shape_vals[:-2]
  num_bd = len(batch_dims_vals)

  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))

  compute_left = (
      eig.ComputationMode.kComputeEigenvectors
      if jobvl
      else eig.ComputationMode.kNoEigenvectors
  )

  compute_right = (
      eig.ComputationMode.kComputeEigenvectors
      if jobvr
      else eig.ComputationMode.kNoEigenvectors
  )
  fn_base = build_lapack_fn_target(fn_base="geev", dtype=dtype)

  i32_type = ir.IntegerType.get_signless(32)
  f32_type = ir.F32Type.get()
  f64_type = ir.F64Type.get()
  c64_type = ir.ComplexType.get(ir.F32Type.get())
  c128_type = ir.ComplexType.get(ir.F64Type.get())
  if ctx.is_forward_compat():
    fn = fn_base
    workspaces: list[ShapeTypePair]
    eigvals: list[ShapeTypePair]
    if dtype == np.float32:
      real = True
      eigvecs_type = c64_type
      workspaces = [([n, n], f32_type)] * 3
      workspace_layouts = [[0, 1]] * 3
      eigvals = [(batch_dims_vals + (n,), f32_type)] * 2
      eigvals_layouts = [tuple(range(num_bd, -1, -1))] * 2
    elif dtype == np.float64:
      real = True
      eigvecs_type = c128_type
      workspaces = [([n, n], f64_type)] * 3
      workspace_layouts = [[0, 1]] * 3
      eigvals = [(batch_dims_vals + (n,), f64_type)] * 2
      eigvals_layouts = [tuple(range(num_bd, -1, -1))] * 2
    elif dtype == np.complex64:
      real = False
      eigvecs_type = c64_type
      workspaces = [([n, n], c64_type), ([hlo_add(n, n)], f32_type)]
      workspace_layouts = [[0, 1], [0]]
      eigvals = [(batch_dims_vals + (n,), c64_type)]
      eigvals_layouts = [tuple(range(num_bd, -1, -1))]
    elif dtype == np.complex128:
      real = False
      eigvecs_type = c128_type
      workspaces = [([n, n], c128_type), ([hlo_add(n, n)], f64_type)]
      workspace_layouts = [[0, 1], [0]]
      eigvals = [(batch_dims_vals + (n,), c128_type)]
      eigvals_layouts = [tuple(range(num_bd, -1, -1))]
    else:
      raise NotImplementedError(f"Unsupported dtype {dtype}")

    scalar_layout = []
    info_layout = tuple(range(num_bd - 1, -1, -1))

    batch_size_val = hlo_s32(1)
    for b_v in batch_dims_vals:
      batch_size_val = hlo.multiply(batch_size_val, ensure_hlo_s32(b_v))

    shape_type_pairs: Sequence[ShapeTypePair] = workspaces + eigvals + [
        (input_shape_vals, eigvecs_type),
        (input_shape_vals, eigvecs_type),
        (batch_dims_vals, i32_type)]
    result_types, result_shapes = mk_result_types_and_shapes(shape_type_pairs)
    out = custom_call(
        fn,
        result_types=result_types,
        operands=[batch_size_val, ensure_hlo_s32(n),
        hlo_u8(compute_left.value),
        hlo_u8(compute_right.value),
        input],
        operand_layouts=[scalar_layout] * 4 + [layout],
        result_layouts=(workspace_layouts + eigvals_layouts + [layout] * 2 +
                        [info_layout]),
        result_shapes=result_shapes,
    ).results
    if real:
      return (hlo.complex(out[3], out[4]), out[5], out[6], out[7])
    else:
      return out[2:6]
  fn = fn_base + "_ffi"
  real = dtype == np.float32 or dtype == np.float64
  eigvecs_type = (
      c64_type if dtype == np.float32 or dtype == np.complex64 else c128_type
  )
  input_type = ir.RankedTensorType(input.type)
  eigvals = [(batch_dims_vals + (n,), input_type.element_type)]
  eigvals_layouts = [tuple(range(num_bd, -1, -1))]
  if real:
    eigvals = eigvals * 2
    eigvals_layouts = eigvals_layouts * 2
  info_layout = tuple(range(num_bd - 1, -1, -1))
  shape_type_pairs: Sequence[ShapeTypePair] = [
      *eigvals,
      (input_shape_vals, eigvecs_type),
      (input_shape_vals, eigvecs_type),
      (batch_dims_vals, i32_type),
  ]
  result_types, result_shapes = mk_result_types_and_shapes(shape_type_pairs)
  out = custom_call(
      fn,
      result_types=result_types,
      operands=[input],
      operand_layouts=[layout],
      result_layouts=(
          *eigvals_layouts,
          layout,
          layout,
          info_layout,
      ),
      result_shapes=result_shapes,
      backend_config={
          "compute_left": _enum_to_char_attr(compute_left),
          "compute_right": _enum_to_char_attr(compute_right),
      },
      api_version=4,
  ).results
  if real:
    return (hlo.complex(out[0], out[1]), out[2], out[3], out[4])
  else:
    return out[:4]

# # gees : Schur factorization

def gees_hlo(ctx, dtype, a, *, jobvs=True, sort=False, select=None,
             a_shape_vals: tuple[DimensionSize, ...]):
  fn_base = prepare_lapack_call(fn_base="gees", dtype=dtype)
  a_type = ir.RankedTensorType(a.type)
  etype = a_type.element_type
  assert len(a_shape_vals) >= 2
  n = a_shape_vals[-1]
  batch_dims_vals = a_shape_vals[:-2]
  num_bd = len(batch_dims_vals)
  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))

  if sort:
    raise NotImplementedError(
        "The sort feature of LAPACK's gees routine is not implemented.")

  mode = (
      schur.ComputationMode.kComputeSchurVectors
      if jobvs
      else schur.ComputationMode.kNoComputeSchurVectors
  )
  sort = schur.Sort.kSortEigenvalues if sort else schur.Sort.kNoSortEigenvalues
  if ctx.is_forward_compat():
    fn = fn_base
    workspaces: list[ShapeTypePair]
    eigvals: list[ShapeTypePair]
    if not np.issubdtype(dtype, np.complexfloating):
      workspaces = [(a_shape_vals, etype)]
      workspace_layouts = [layout]
      eigvals = [(batch_dims_vals + (n,), etype)] * 2
      eigvals_layouts = [tuple(range(num_bd, -1, -1))] * 2
    else:
      workspaces = [(a_shape_vals, etype),
                    ([n], ir.ComplexType(etype).element_type),
      ]
      workspace_layouts = [layout, [0]]
      eigvals = [(batch_dims_vals + (n,), etype)]
      eigvals_layouts = [tuple(range(num_bd, -1, -1))]

    i32_type = ir.IntegerType.get_signless(32)

    scalar_layout = []
    batch_size_val = hlo_s32(1)
    for b_v in batch_dims_vals:
      batch_size_val = hlo.multiply(batch_size_val, ensure_hlo_s32(b_v))
    shape_type_pairs = workspaces + eigvals + [
      (a_shape_vals, etype),
      (batch_dims_vals, i32_type),
      (batch_dims_vals, i32_type)]
    result_types, result_shapes = mk_result_types_and_shapes(shape_type_pairs)
    out = custom_call(
        fn,
        result_types=result_types,
        operands=[
            batch_size_val,
            ensure_hlo_s32(n),
            hlo_u8(mode.value),
            hlo_u8(sort.value),
            # TODO: figure out how to put the callable select function here
            a
        ],
        operand_layouts=[scalar_layout] * 4 + [layout],
        result_layouts=workspace_layouts + eigvals_layouts + [
          layout,
          tuple(range(num_bd - 1, -1, -1)),
          tuple(range(num_bd - 1, -1, -1)),
        ],
        operand_output_aliases={4: 0},
        result_shapes=result_shapes,
    ).results
    if sort == schur.Sort.kSortEigenvalues:
      return (out[0], out[3], out[4], out[5])
    else:
      return (out[0], out[3], out[5])
  fn = fn_base + "_ffi"
  eigvals: list[ShapeTypePair]
  is_complex = np.issubdtype(dtype, np.complexfloating)
  eigvals = [(batch_dims_vals + (n,), etype)]
  eigvals_layouts = [tuple(range(num_bd, -1, -1))]
  if not is_complex:
    eigvals = eigvals * 2
    eigvals_layouts = eigvals_layouts * 2

  i32_type = ir.IntegerType.get_signless(32)
  shape_type_pairs = [
      (a_shape_vals, etype),
      (a_shape_vals, etype),
      *eigvals,
      (batch_dims_vals, i32_type),
      (batch_dims_vals, i32_type),
  ]
  result_types, result_shapes = mk_result_types_and_shapes(shape_type_pairs)
  out = custom_call(
      fn,
      result_types=result_types,
      operands=[a],
      # TODO(paruzelp): Use FFI execution context to put `select`
      operand_layouts=[layout],
      result_layouts=[
          layout,
          layout,
          *eigvals_layouts,
          tuple(range(num_bd - 1, -1, -1)),
          tuple(range(num_bd - 1, -1, -1)),
      ],
      operand_output_aliases={0: 0},
      result_shapes=result_shapes,
      backend_config={
          "mode": _enum_to_char_attr(mode),
          "sort": _enum_to_char_attr(sort),
      },
      api_version=4,
  ).results
  # out: Schur Form, Schur Vectors, Eigenvalues, Selected Eigenvalues, Info
  if is_complex:
    return out[0], out[1], out[2], out[3], out[4]
  else:
    return out[0], out[1], (out[2], out[3]), out[4], out[5]


# gehrd: Reduction of a non-symmetric square matrix to upper Hessenberg form.
def gehrd_hlo(ctx, dtype, a):
  fn_base = prepare_lapack_call(fn_base="gehrd", dtype=dtype)
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  assert m == n, (m, n)
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  if ctx.is_forward_compat():
    fn = fn_base
    b = 1
    for d in batch_dims:
      b *= d

    if dtype == np.float32:
      lwork = _lapack.lapack_sgehrd_workspace(n, n, 1, n)
    elif dtype == np.float64:
      lwork = _lapack.lapack_dgehrd_workspace(n, n, 1, n)
    elif dtype == np.complex64:
      lwork = _lapack.lapack_cgehrd_workspace(n, n, 1, n)
    elif dtype == np.complex128:
      lwork = _lapack.lapack_zgehrd_workspace(n, n, 1, n)
    else:
      raise NotImplementedError(f"Unsupported dtype {dtype}")

    layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
    i32_type = ir.IntegerType.get_signless(32)
    return custom_call(
        fn,
        result_types=[
          a.type,
          ir.RankedTensorType.get(batch_dims + (n - 1,), a_type.element_type),
          ir.RankedTensorType.get(batch_dims, i32_type),
          ir.RankedTensorType.get([lwork], a_type.element_type),
        ],
        operands=[hlo_s32(n), hlo_s32(1), hlo_s32(n), hlo_s32(n), hlo_s32(b),
        hlo_s32(lwork), a],
        operand_layouts=[[]] * 6 + [layout],
        result_layouts=[
          layout,
          (num_bd,) + tuple(range(num_bd - 1, -1, -1)),
          tuple(range(num_bd - 1, -1, -1)),
          [0],
        ],
        operand_output_aliases={6: 0},
    ).results[:3]
  fn = fn_base + "_ffi"
  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  i32_type = ir.IntegerType.get_signless(32)
  return custom_call(
      fn,
      result_types=[
          a.type,
          ir.RankedTensorType.get(batch_dims + (n - 1,), a_type.element_type),
          ir.RankedTensorType.get(batch_dims, i32_type),
      ],
      operands=[a],
      operand_layouts=[layout],
      result_layouts=[
          layout,
          (num_bd,) + tuple(range(num_bd - 1, -1, -1)),
          tuple(range(num_bd - 1, -1, -1)),
      ],
      operand_output_aliases={0: 0},
      backend_config={
          "low": _lapack_int_attr(1),
          "high": _lapack_int_attr(n),
      },
      api_version=4,
  ).results


# sytrd: Reduction of a symmetric (Hermitian) matrix to tridiagonal form.
def sytrd_hlo(ctx, dtype, a, *, lower):
  fn_base = "he" if dtype == np.complex64 or dtype == np.complex128 else "sy"
  fn_base = prepare_lapack_call(fn_base=fn_base + "trd", dtype=dtype)
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  assert m == n, (m, n)
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  i32_type = ir.IntegerType.get_signless(32)

  if ctx.is_forward_compat():
    fn = fn_base
    b = 1
    for d in batch_dims:
      b *= d

    if dtype == np.float32:
      lwork = _lapack.lapack_ssytrd_workspace(n, n)
      diag_type = a_type.element_type
    elif dtype == np.float64:
      lwork = _lapack.lapack_dsytrd_workspace(n, n)
      diag_type = a_type.element_type
    elif dtype == np.complex64:
      lwork = _lapack.lapack_chetrd_workspace(n, n)
      diag_type = ir.F32Type.get()
    elif dtype == np.complex128:
      lwork = _lapack.lapack_zhetrd_workspace(n, n)
      diag_type = ir.F64Type.get()
    else:
      raise NotImplementedError(f"Unsupported dtype {dtype}")

    return custom_call(
        fn,
        result_types=[
          a.type,
          ir.RankedTensorType.get(batch_dims + (n,), diag_type),
          ir.RankedTensorType.get(batch_dims + (n - 1,), diag_type),
          ir.RankedTensorType.get(batch_dims + (n - 1,), a_type.element_type),
          ir.RankedTensorType.get(batch_dims, i32_type),
          ir.RankedTensorType.get([lwork], a_type.element_type),
        ],
        operands=[hlo_s32(n), hlo_s32(1 if lower else 0), hlo_s32(max(1, n)),
        hlo_s32(b), hlo_s32(lwork), a],
        operand_layouts=[[]] * 5 + [layout],
        result_layouts=[
          layout,
          (num_bd,) + tuple(range(num_bd - 1, -1, -1)),
          (num_bd,) + tuple(range(num_bd - 1, -1, -1)),
          (num_bd,) + tuple(range(num_bd - 1, -1, -1)),
          tuple(range(num_bd - 1, -1, -1)),
          [0],
        ],
        operand_output_aliases={5: 0},
    ).results[:5]
  fn = fn_base + "_ffi"
  if dtype == np.float32 or dtype == np.complex64:
    diag_type = ir.F32Type.get()
  elif dtype == np.float64 or dtype == np.complex128:
    diag_type = ir.F64Type.get()
  else:
    raise NotImplementedError(f"Unsupported dtype {dtype}")

  # Returns x_out, on_diag, off_diag, tau, info
  return custom_call(
      fn,
      result_types=[
          a.type,
          ir.RankedTensorType.get(batch_dims + (n,), diag_type),
          ir.RankedTensorType.get(batch_dims + (n - 1,), diag_type),
          ir.RankedTensorType.get(batch_dims + (n - 1,), a_type.element_type),
          ir.RankedTensorType.get(batch_dims, i32_type),
      ],
      operands=[a],
      operand_layouts=[layout],
      result_layouts=[
          layout,
          tuple(range(num_bd, -1, -1)),
          tuple(range(num_bd, -1, -1)),
          tuple(range(num_bd, -1, -1)),
          tuple(range(num_bd - 1, -1, -1)),
      ],
      operand_output_aliases={0: 0},
      backend_config={
          "uplo": _matrix_uplo_attr(lower=lower),
      },
      api_version=4,
  ).results
