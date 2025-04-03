# Copyright 2019 The JAX Authors.
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
"""
cusparse wrappers for performing sparse matrix computations in JAX
"""

import math
from functools import partial
import importlib

import jaxlib.mlir.ir as ir

import numpy as np

from jaxlib import xla_client

from .hlo_helpers import custom_call, mk_result_types_and_shapes

for cuda_module_name in [".cuda", "jax_cuda12_plugin"]:
  try:
    _cusparse = importlib.import_module(
        f"{cuda_module_name}._sparse", package="jaxlib"
    )
  except ImportError:
    _cusparse = None
  else:
    break

if _cusparse:
  for _name, _value in _cusparse.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="CUDA")

for rocm_module_name in [".rocm", "jax_rocm60_plugin"]:
  try:
    _hipsparse = importlib.import_module(
        f"{rocm_module_name}._sparse", package="jaxlib"
    )
  except ImportError:
    _hipsparse = None
  else:
    break

if _hipsparse:
  for _name, _value in _hipsparse.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="ROCM")


cuda_is_supported = bool(_cusparse and _cusparse.sparse_supported)
rocm_is_supported = bool(_hipsparse and _hipsparse.sparse_supported)


def _validate_csr_hlo(data, indices, indptr, shape):
  data_type = ir.RankedTensorType(data.type)
  indices_type = ir.RankedTensorType(indices.type)
  indptr_type = ir.RankedTensorType(indptr.type)

  nnz, = data_type.shape
  assert indices_type.shape == [nnz]
  assert indptr_type.element_type == indices_type.element_type
  assert indptr_type.shape == [shape[0] + 1]
  return data_type.element_type, indices_type.element_type, nnz

def _validate_coo_hlo(data, row, col):
  data_type = ir.RankedTensorType(data.type)
  row_type = ir.RankedTensorType(row.type)
  col_type = ir.RankedTensorType(col.type)

  nnz, = data_type.shape
  assert row_type.shape == [nnz]
  assert col_type.element_type == row_type.element_type
  assert col_type.shape == [nnz]
  return data_type.element_type, row_type.element_type, nnz


def _csr_todense_hlo(platform, gpu_sparse, data, indices, indptr, *, shape,
                     data_dtype, index_dtype):
  """CSR to dense matrix."""
  data_type, index_type, nnz = _validate_csr_hlo(data, indices, indptr, shape)
  rows, cols = shape

  buffer_size, opaque = gpu_sparse.build_csr_todense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  out = custom_call(
      f"{platform}sparse_csr_todense",
      result_types=[
          ir.RankedTensorType.get(shape, data_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ],
      operands=[data, indices, indptr],
      backend_config=opaque,
      operand_layouts=[[0]] * 3,
      result_layouts=[[1, 0], [0]]).results
  return out[0]

cuda_csr_todense = partial(_csr_todense_hlo, "cu", _cusparse)
rocm_csr_todense = partial(_csr_todense_hlo, "hip", _hipsparse)


def _csr_fromdense_hlo(platform, gpu_sparse, mat, *, nnz, index_dtype,
                       data_dtype, index_type):
  """CSR from dense matrix."""
  mat_type = ir.RankedTensorType(mat.type)
  rows, cols = mat_type.shape

  buffer_size, opaque = gpu_sparse.build_csr_fromdense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  out = custom_call(
      f"{platform}sparse_csr_fromdense",
      result_types=[
          ir.RankedTensorType.get([nnz], mat_type.element_type),
          ir.RankedTensorType.get([nnz], index_type),
          ir.RankedTensorType.get([rows + 1], index_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ],
      operands=[mat],
      backend_config=opaque,
      operand_layouts=[[1, 0]],
      result_layouts=[[0]] * 4).results
  return out[:3]

cuda_csr_fromdense = partial(_csr_fromdense_hlo, "cu", _cusparse)
rocm_csr_fromdense = partial(_csr_fromdense_hlo, "hip", _hipsparse)


def _csr_matvec_hlo(platform, gpu_sparse, data, indices, indptr, x, *, shape,
                    transpose=False, compute_dtype=None, compute_type=None,
                    data_dtype, index_dtype, x_dtype):
  """CSR matrix/vector multiply."""
  data_type, index_type, nnz = _validate_csr_hlo(data, indices, indptr, shape)
  rows, cols = shape

  if compute_dtype is None:
    compute_dtype = data_dtype
    compute_type = data_type

  buffer_size, opaque = gpu_sparse.build_csr_matvec_descriptor(
      data_dtype, x_dtype, compute_dtype, index_dtype,
      rows, cols, nnz, transpose)
  out_size = cols if transpose else rows

  out = custom_call(
      f"{platform}sparse_csr_matvec",
      result_types=[
          ir.RankedTensorType.get([out_size], compute_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ],
      operands=[data, indices, indptr, x],
      backend_config=opaque,
      operand_layouts=[[0]] * 4,
      result_layouts=[[0]] * 2).results
  return out[0]

cuda_csr_matvec = partial(_csr_matvec_hlo, "cu", _cusparse)
rocm_csr_matvec = partial(_csr_matvec_hlo, "hip", _hipsparse)


def _csr_matmat_hlo(platform, gpu_sparse, data, indices, indptr, B, *, shape,
                    transpose=False, compute_dtype=None, compute_type=None,
                    index_dtype, data_dtype, B_dtype):
  """CSR from dense matrix."""
  data_type, index_type, nnz = _validate_csr_hlo(data, indices, indptr, shape)
  rows, cols = shape
  B_shape = ir.RankedTensorType(B.type).shape
  _, Ccols = B_shape

  if compute_dtype is None:
    compute_dtype = data_dtype
    compute_type = data_type

  buffer_size, opaque = gpu_sparse.build_csr_matmat_descriptor(
      data_dtype, B_dtype, compute_dtype, index_dtype,
      rows, cols, Ccols, nnz, transpose)
  out_size = cols if transpose else rows

  out = custom_call(
      f"{platform}sparse_csr_matmat",
      result_types=[
          ir.RankedTensorType.get([out_size, Ccols], compute_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ],
      operands=[data, indices, indptr, B],
      backend_config=opaque,
      operand_layouts=[[0], [0], [0], [1, 0]],
      result_layouts=[[1, 0], [0]]).results
  return out[0]

cuda_csr_matmat = partial(_csr_matmat_hlo, "cu", _cusparse)
rocm_csr_matmat = partial(_csr_matmat_hlo, "hip", _hipsparse)


def _coo_todense_hlo(platform, gpu_sparse, data, row, col, *, shape,
                     data_dtype, index_dtype):
  """COO to dense matrix."""
  data_type, _, nnz = _validate_coo_hlo(data, row, col)
  rows, cols = shape

  buffer_size, opaque = gpu_sparse.build_coo_todense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  out = custom_call(
      f"{platform}sparse_coo_todense",
      result_types=[
          ir.RankedTensorType.get(shape, data_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ],
      operands=[data, row, col],
      backend_config=opaque,
      operand_layouts=[[0]] * 3,
      result_layouts=[[1, 0], [0]]).results
  return out[0]

cuda_coo_todense = partial(_coo_todense_hlo, "cu", _cusparse)
rocm_coo_todense = partial(_coo_todense_hlo, "hip", _hipsparse)


def _coo_fromdense_hlo(platform, gpu_sparse, mat, *, nnz, data_dtype,
                       index_dtype, index_type):
  """COO from dense matrix."""
  mat_type = ir.RankedTensorType(mat.type)
  rows, cols = mat_type.shape

  buffer_size, opaque = gpu_sparse.build_coo_fromdense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  out = custom_call(
      f"{platform}sparse_coo_fromdense",
      result_types=[
          ir.RankedTensorType.get([nnz], mat_type.element_type),
          ir.RankedTensorType.get([nnz], index_type),
          ir.RankedTensorType.get([nnz], index_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ],
      operands=[mat],
      backend_config=opaque,
      operand_layouts=[[1, 0]],
      result_layouts=[[0]] * 4).results
  return out[:3]

cuda_coo_fromdense = partial(_coo_fromdense_hlo, "cu", _cusparse)
rocm_coo_fromdense = partial(_coo_fromdense_hlo, "hip", _hipsparse)


def _coo_matvec_hlo(platform, gpu_sparse, data, row, col, x, *, shape,
                    transpose=False, compute_dtype=None, compute_type=None,
                    index_dtype, data_dtype, x_dtype):
  """COO matrix/vector multiply."""
  data_type, _, nnz = _validate_coo_hlo(data, row, col)
  rows, cols = shape

  if compute_dtype is None:
    compute_dtype = data_dtype
    compute_type = data_type

  buffer_size, opaque = gpu_sparse.build_coo_matvec_descriptor(
      data_dtype, x_dtype, compute_dtype, index_dtype,
      rows, cols, nnz, transpose)
  out_size = cols if transpose else rows

  out = custom_call(
      f"{platform}sparse_coo_matvec",
      result_types=[
          ir.RankedTensorType.get([out_size], compute_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ],
      operands=[data, row, col, x],
      backend_config=opaque,
      operand_layouts=[[0]] * 4,
      result_layouts=[[0]] * 2).results
  return out[0]

cuda_coo_matvec = partial(_coo_matvec_hlo, "cu", _cusparse)
rocm_coo_matvec = partial(_coo_matvec_hlo, "hip", _hipsparse)


def _coo_matmat_hlo(platform, gpu_sparse, data, row, col, B, *, shape,
                    transpose=False, compute_dtype=None, compute_type=None,
                    x_dtype, data_dtype, index_dtype):
  """COO from dense matrix."""
  data_type, _, nnz = _validate_coo_hlo(data, row, col)
  is_batched_matmat = False
  batch_count = 1
  if len(shape) == 2:
    rows, cols = shape
  elif len(shape) == 3:
    is_batched_matmat = True
    batch_count, rows, cols = shape
    # Redefine nnz as nnz per batch.
    nnz = nnz // batch_count

  B_shape = ir.RankedTensorType(B.type).shape
  _, Ccols = B_shape

  if compute_dtype is None:
    compute_dtype = data_dtype
    compute_type = data_type

  # TODO(tianjianlu): use batch stride to trigger different mode of batch
  # computation. Currently batch_stride = 0 is not allowed because of the issue
  # in cusparse https://github.com/NVIDIA/CUDALibrarySamples/issues/81#issuecomment-1205562643
  # Set batch stride to be the matrix size for now.
  lhs_batch_stride = nnz
  B_rows = rows if transpose else cols
  rhs_batch_stride =  B_rows * Ccols

  buffer_size, opaque = gpu_sparse.build_coo_matmat_descriptor(
      data_dtype, x_dtype, compute_dtype, index_dtype,
      rows, cols, Ccols, nnz, transpose, batch_count, lhs_batch_stride,
      rhs_batch_stride)
  out_size = cols if transpose else rows

  if is_batched_matmat:
    out_shape = [batch_count, out_size, Ccols]
    out_layout = [2, 1, 0]
  else:
    out_shape = [out_size, Ccols]
    out_layout = [1, 0]

  out = custom_call(
      f"{platform}sparse_coo_matmat",
      result_types=[
          ir.RankedTensorType.get(out_shape, compute_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ],
      operands=[data, row, col, B],
      backend_config=opaque,
      operand_layouts=[[0], [0], [0], [1, 0]],
      result_layouts=[out_layout, [0]]).results
  return out[0]

cuda_coo_matmat = partial(_coo_matmat_hlo, "cu", _cusparse)
rocm_coo_matmat = partial(_coo_matmat_hlo, "hip", _hipsparse)


def _gtsv2_hlo(
    platform, gpu_sparse, dl, d, du, B, *, m, n, ldb, t, b_shape_vals=None):
  """Calls `cusparse<t>gtsv2(dl, d, du, B, m, n, ldb)`."""
  assert len(b_shape_vals) >= 2
  batch_dim_vals = b_shape_vals[:-2]
  batch_size = math.prod(batch_dim_vals)
  num_bd = len(b_shape_vals) - 2
  f32 = (t == np.float32)
  if f32:
    buffer_size = gpu_sparse.gtsv2_f32_buffer_size(m, n, ldb)
  else:
    buffer_size = gpu_sparse.gtsv2_f64_buffer_size(m, n, ldb)

  b_layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  d_layout = (num_bd,) + tuple(range(num_bd - 1, -1, -1))
  b_type = ir.RankedTensorType(B.type)

  shape_type_pairs = [
      (batch_dim_vals + (ldb, n), b_type.element_type),
      ((buffer_size,), ir.IntegerType.get_signless(8))
  ]
  result_types, result_shapes = mk_result_types_and_shapes(shape_type_pairs)
  out = custom_call(
      f"{platform}sparse_gtsv2_" + ("f32" if f32 else "f64"),
      result_types=result_types,
      operands=[dl, d, du, B],
      backend_config=gpu_sparse.build_gtsv2_descriptor(batch_size, m, n, ldb),
      operand_layouts=[d_layout] * 3 + [b_layout],
      result_layouts=[b_layout, [0]],
      operand_output_aliases={3: 0},
      result_shapes=result_shapes).results
  return out[0]

cuda_gtsv2 = partial(_gtsv2_hlo, "cu", _cusparse)
rocm_gtsv2 = partial(_gtsv2_hlo, "hip", _hipsparse)
