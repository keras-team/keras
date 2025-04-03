# Copyright 2021 The JAX Authors.
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

"""COO (coordinate format) matrix object and associated primitives."""
from __future__ import annotations

from collections.abc import Sequence
from functools import partial
import operator
from typing import Any, NamedTuple
import warnings

import numpy as np

import jax
from jax import lax
from jax.interpreters import mlir
from jax.experimental.sparse._base import JAXSparse
from jax.experimental.sparse.util import _coo_extract, CuSparseEfficiencyWarning
from jax import tree_util
from jax._src import core
from jax._src import dispatch
from jax._src.interpreters import ad
from jax._src.lax.lax import _const
from jax._src.lib.mlir.dialects import hlo
from jax._src.lib import gpu_sparse
from jax._src.numpy.util import promote_dtypes
from jax._src.typing import Array, ArrayLike, DTypeLike
import jax.numpy as jnp


Dtype = Any
Shape = tuple[int, ...]

class COOInfo(NamedTuple):
  shape: Shape
  rows_sorted: bool = False
  cols_sorted: bool = False


@tree_util.register_pytree_node_class
class COO(JAXSparse):
  """Experimental COO matrix implemented in JAX.

  Note: this class has minimal compatibility with JAX transforms such as
  grad and autodiff, and offers very little functionality. In general you
  should prefer :class:`jax.experimental.sparse.BCOO`.

  Additionally, there are known failures in the case that `nse` is larger
  than the true number of nonzeros in the represented matrix. This situation
  is better handled in BCOO.
  """
  data: jax.Array
  row: jax.Array
  col: jax.Array
  shape: tuple[int, int]
  nse = property(lambda self: self.data.size)
  dtype = property(lambda self: self.data.dtype)
  _info = property(lambda self: COOInfo(
      shape=self.shape, rows_sorted=self._rows_sorted,
      cols_sorted=self._cols_sorted))
  _bufs = property(lambda self: (self.data, self.row, self.col))
  _rows_sorted: bool
  _cols_sorted: bool

  def __init__(self, args: tuple[Array, Array, Array], *, shape: Shape,
               rows_sorted: bool = False, cols_sorted: bool = False):
    self.data, self.row, self.col = map(jnp.asarray, args)
    self._rows_sorted = rows_sorted
    self._cols_sorted = cols_sorted
    super().__init__(args, shape=shape)

  @classmethod
  def fromdense(cls, mat: Array, *, nse: int | None = None, index_dtype: DTypeLike = np.int32) -> COO:
    return coo_fromdense(mat, nse=nse, index_dtype=index_dtype)

  def _sort_indices(self) -> COO:
    """Return a copy of the COO matrix with sorted indices.

    The matrix is sorted by row indices and column indices per row.
    If self._rows_sorted is True, this returns ``self`` without a copy.
    """
    # TODO(jakevdp): would be benefit from lowering this to cusparse sort_rows utility?
    if self._rows_sorted:
      return self
    row, col, data = lax.sort((self.row, self.col, self.data), num_keys=2)
    return self.__class__((data, row, col), shape=self.shape,
                          rows_sorted=True)

  @classmethod
  def _empty(cls, shape: Sequence[int], *, dtype: DTypeLike | None = None,
             index_dtype: DTypeLike = 'int32') -> COO:
    """Create an empty COO instance. Public method is sparse.empty()."""
    shape = tuple(shape)
    if len(shape) != 2:
      raise ValueError(f"COO must have ndim=2; got {shape=}")
    data = jnp.empty(0, dtype)
    row = col = jnp.empty(0, index_dtype)
    return cls((data, row, col), shape=shape, rows_sorted=True,
               cols_sorted=True)

  @classmethod
  def _eye(cls, N: int, M: int, k: int, *, dtype: DTypeLike | None = None,
           index_dtype: DTypeLike = 'int32') -> COO:
    if k > 0:
      diag_size = min(N, M - k)
    else:
      diag_size = min(N + k, M)

    if diag_size <= 0:
      # if k is out of range, return an empty matrix.
      return cls._empty((N, M), dtype=dtype, index_dtype=index_dtype)

    data = jnp.ones(diag_size, dtype=dtype)
    idx = jnp.arange(diag_size, dtype=index_dtype)
    zero = _const(idx, 0)
    k = _const(idx, k)
    row = lax.sub(idx, lax.cond(k >= 0, lambda: zero, lambda: k))
    col = lax.add(idx, lax.cond(k <= 0, lambda: zero, lambda: k))
    return cls((data, row, col), shape=(N, M), rows_sorted=True, cols_sorted=True)

  def todense(self) -> Array:
    return coo_todense(self)

  def transpose(self, axes: tuple[int, ...] | None = None) -> COO:
    if axes is not None:
      raise NotImplementedError("axes argument to transpose()")
    return COO((self.data, self.col, self.row), shape=self.shape[::-1],
               rows_sorted=self._cols_sorted, cols_sorted=self._rows_sorted)

  def tree_flatten(self) -> tuple[tuple[Array, Array, Array], dict[str, Any]]:
    return (self.data, self.row, self.col), self._info._asdict()

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    obj = object.__new__(cls)
    obj.data, obj.row, obj.col = children
    if aux_data.keys() != {'shape', 'rows_sorted', 'cols_sorted'}:
      raise ValueError(f"COO.tree_unflatten: invalid {aux_data=}")
    obj.shape = aux_data['shape']
    obj._rows_sorted = aux_data['rows_sorted']
    obj._cols_sorted = aux_data['cols_sorted']
    return obj

  def __matmul__(self, other: ArrayLike) -> Array:
    if isinstance(other, JAXSparse):
      raise NotImplementedError("matmul between two sparse objects.")
    other = jnp.asarray(other)
    data, other = promote_dtypes(self.data, other)
    self_promoted = COO((data, self.row, self.col), **self._info._asdict())
    if other.ndim == 1:
      return coo_matvec(self_promoted, other)
    elif other.ndim == 2:
      return coo_matmat(self_promoted, other)
    else:
      raise NotImplementedError(f"matmul with object of shape {other.shape}")

#--------------------------------------------------------------------
# coo_todense

coo_todense_p = core.Primitive('coo_todense')

def coo_todense(mat: COO) -> Array:
  """Convert a COO-format sparse matrix to a dense matrix.

  Args:
    mat : COO matrix
  Returns:
    mat_dense: dense version of ``mat``
  """
  return _coo_todense(mat.data, mat.row, mat.col, spinfo=mat._info)

def _coo_todense(data: Array, row: Array, col: Array, *, spinfo: COOInfo) -> Array:
  """Convert CSR-format sparse matrix to a dense matrix.

  Args:
    data : array of shape ``(nse,)``.
    row : array of shape ``(nse,)``
    col : array of shape ``(nse,)`` and dtype ``row.dtype``
    spinfo : COOInfo object containing matrix metadata

  Returns:
    mat : array with specified shape and dtype matching ``data``
  """
  return coo_todense_p.bind(data, row, col, spinfo=spinfo)

def _coo_todense_impl(data, row, col, *, spinfo):
  return jnp.zeros(spinfo.shape, data.dtype).at[row, col].add(data)

@coo_todense_p.def_abstract_eval
def _coo_todense_abstract_eval(data, row, col, *, spinfo):
  return core.ShapedArray(spinfo.shape, data.dtype)

_coo_todense_lowering = mlir.lower_fun(
    _coo_todense_impl, multiple_results=False)

def _coo_todense_gpu_lowering(coo_todense_hlo, ctx, data, row, col, *, spinfo):
  data_aval, row_aval, _ = ctx.avals_in
  dtype = data_aval.dtype
  if not (np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating)):
    warnings.warn(f"coo_todense cusparse/hipsparse lowering not available for {dtype=}. "
                  "Falling back to default implementation.", CuSparseEfficiencyWarning)
    return _coo_todense_lowering(ctx, data, row, col, spinfo=spinfo)

  if spinfo.rows_sorted:
    shape = spinfo.shape
    transpose = False
  elif spinfo.cols_sorted:
    row, col = col, row
    transpose = True
    shape = spinfo.shape[::-1]
  else:
    warnings.warn("coo_todense GPU lowering requires matrices with sorted rows or sorted cols. "
                  "To sort the rows in your matrix, use e.g. mat = mat._sort_indices(). Falling "
                  "back to the default implementation.", CuSparseEfficiencyWarning)
    return _coo_todense_lowering(ctx, data, row, col, spinfo=spinfo)

  result = coo_todense_hlo(
      data, row, col, shape=shape, data_dtype=dtype, index_dtype=row_aval.dtype)
  return (
      [hlo.transpose(result, mlir.dense_int_array([1, 0]))]
      if transpose else [result])


def _coo_todense_jvp(data_dot, data, row, col, *, spinfo):
  return _coo_todense(data_dot, row, col, spinfo=spinfo)

def _coo_todense_transpose(ct, data, row, col, *, spinfo):
  # Note: we assume that transpose has the same sparsity pattern.
  # Can we check this?
  assert ad.is_undefined_primal(data)
  if ad.is_undefined_primal(row) or ad.is_undefined_primal(col):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ct.shape == spinfo.shape
  assert row.aval.dtype == col.aval.dtype
  assert ct.dtype == data.aval.dtype
  return _coo_extract(row, col, ct), row, col

ad.defjvp(coo_todense_p, _coo_todense_jvp, None, None)
ad.primitive_transposes[coo_todense_p] = _coo_todense_transpose
mlir.register_lowering(coo_todense_p, _coo_todense_lowering)
dispatch.simple_impl(coo_todense_p)

if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(
      coo_todense_p,
      partial(_coo_todense_gpu_lowering, gpu_sparse.cuda_coo_todense),
      platform='cuda')
if gpu_sparse.rocm_is_supported:
  mlir.register_lowering(
      coo_todense_p,
      partial(_coo_todense_gpu_lowering, gpu_sparse.rocm_coo_todense),
      platform='rocm')

#--------------------------------------------------------------------
# coo_fromdense

coo_fromdense_p = core.Primitive('coo_fromdense')
coo_fromdense_p.multiple_results = True

def coo_fromdense(mat: Array, *, nse: int | None = None, index_dtype: DTypeLike = jnp.int32) -> COO:
  """Create a COO-format sparse matrix from a dense matrix.

  Args:
    mat : array to be converted to COO.
    nse : number of specified entries in ``mat``. If not specified,
      it will be computed from the input matrix.
    index_dtype : dtype of sparse indices

  Returns:
    mat_coo : COO representation of the matrix.
  """
  if nse is None:
    nse = int((mat != 0).sum())
  nse_int = core.concrete_or_error(operator.index, nse, "coo_fromdense nse argument")
  return COO(_coo_fromdense(mat, nse=nse_int, index_dtype=index_dtype),
             shape=mat.shape, rows_sorted=True)

def _coo_fromdense(mat: Array, *, nse: int, index_dtype: DTypeLike = jnp.int32) -> tuple[Array, Array, Array]:
  """Create COO-format sparse matrix from a dense matrix.

  Args:
    mat : array to be converted to COO.
    nse : number of specified entries in ``mat``
    index_dtype : dtype of sparse indices

  Returns:
    data : array of shape ``(nse,)`` and dtype ``mat.dtype``
    row : array of shape ``(nse,)`` and dtype ``index_dtype``
    col : array of shape ``(nse,)`` and dtype ``index_dtype``
  """
  mat = jnp.asarray(mat)
  nse = core.concrete_or_error(operator.index, nse, "nse argument of coo_fromdense()")
  return coo_fromdense_p.bind(mat, nse=nse, index_dtype=index_dtype)

def _coo_fromdense_impl(mat, *, nse, index_dtype):
  mat = jnp.asarray(mat)
  assert mat.ndim == 2

  row, col = jnp.nonzero(mat, size=nse)
  data = mat[row, col]

  true_nonzeros = jnp.arange(nse) < (mat != 0).sum()
  data = jnp.where(true_nonzeros, data, 0)

  return data, row.astype(index_dtype), col.astype(index_dtype)

@coo_fromdense_p.def_abstract_eval
def _coo_fromdense_abstract_eval(mat, *, nse, index_dtype):
  data = core.ShapedArray((nse,), mat.dtype)
  row = col = core.ShapedArray((nse,), index_dtype)
  return data, row, col

_coo_fromdense_lowering = mlir.lower_fun(
    _coo_fromdense_impl, multiple_results=True)

def _coo_fromdense_gpu_lowering(coo_fromdense_hlo, ctx, mat, *, nse,
                                index_dtype):
  dtype = ctx.avals_in[0].dtype
  if not (np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating)):
    warnings.warn(f"coo_fromdense cusparse/hipsparse lowering not available for {dtype=}. "
                  "Falling back to default implementation.", CuSparseEfficiencyWarning)
    return _coo_fromdense_lowering(ctx, mat, nse=nse, index_dtype=index_dtype)
  data, row, col = coo_fromdense_hlo(
      mat, nnz=nse,
      data_dtype=dtype,
      index_dtype=np.dtype(index_dtype),
      index_type=mlir.dtype_to_ir_type(np.dtype(index_dtype)))
  return [data, row, col]


def _coo_fromdense_jvp(primals, tangents, *, nse, index_dtype):
  M, = primals
  Mdot, = tangents

  primals_out = _coo_fromdense(M, nse=nse, index_dtype=index_dtype)
  data, row, col = primals_out

  if type(Mdot) is ad.Zero:
    data_dot = ad.Zero.from_primal_value(data)
  else:
    data_dot = _coo_extract(row, col, Mdot)

  tangents_out = (data_dot, ad.Zero.from_primal_value(row), ad.Zero.from_primal_value(col))

  return primals_out, tangents_out

def _coo_fromdense_transpose(ct, M, *, nse, index_dtype):
  data, row, col = ct
  assert len(data) == nse
  assert row.dtype == col.dtype == index_dtype
  if isinstance(row, ad.Zero) or isinstance(col, ad.Zero):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ad.is_undefined_primal(M)
  return _coo_todense(data, row, col, spinfo=COOInfo(shape=M.aval.shape))

ad.primitive_jvps[coo_fromdense_p] = _coo_fromdense_jvp
ad.primitive_transposes[coo_fromdense_p] = _coo_fromdense_transpose
mlir.register_lowering(coo_fromdense_p, _coo_fromdense_lowering)
dispatch.simple_impl(coo_fromdense_p)

if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(
      coo_fromdense_p,
      partial(_coo_fromdense_gpu_lowering, gpu_sparse.cuda_coo_fromdense),
      platform='cuda')
if gpu_sparse.rocm_is_supported:
  mlir.register_lowering(
      coo_fromdense_p,
      partial(_coo_fromdense_gpu_lowering, gpu_sparse.rocm_coo_fromdense),
      platform='rocm')

#--------------------------------------------------------------------
# coo_matvec

coo_matvec_p = core.Primitive('coo_matvec')

def coo_matvec(mat: COO, v: Array, transpose: bool = False) -> Array:
  """Product of COO sparse matrix and a dense vector.

  Args:
    mat : COO matrix
    v : one-dimensional array of size ``(shape[0] if transpose else shape[1],)`` and
      dtype ``mat.dtype``
    transpose : boolean specifying whether to transpose the sparse matrix
      before computing.

  Returns:
    y : array of shape ``(mat.shape[1] if transpose else mat.shape[0],)`` representing
      the matrix vector product.
  """
  data, row, col = mat._bufs
  return _coo_matvec(data, row, col, v, spinfo=mat._info, transpose=transpose)

def _coo_matvec(data: Array, row: Array, col: Array, v: Array, *, spinfo: COOInfo, transpose: bool = False) -> Array:
  """Product of COO sparse matrix and a dense vector.

  Args:
    data : array of shape ``(nse,)``.
    row : array of shape ``(nse,)``
    col : array of shape ``(nse,)`` and dtype ``row.dtype``
    v : array of shape ``(shape[0] if transpose else shape[1],)`` and
      dtype ``data.dtype``
    shape : length-2 tuple representing the matrix shape
    transpose : boolean specifying whether to transpose the sparse matrix
      before computing.

  Returns:
    y : array of shape ``(shape[1] if transpose else shape[0],)`` representing
      the matrix vector product.
  """
  return coo_matvec_p.bind(data, row, col, v, spinfo=spinfo, transpose=transpose)

def _coo_matvec_impl(data, row, col, v, *, spinfo, transpose):
  v = jnp.asarray(v)
  if transpose:
    row, col = col, row
  out_shape = spinfo.shape[1] if transpose else spinfo.shape[0]
  dv = data * v[col]
  return jnp.zeros(out_shape, dv.dtype).at[row].add(dv)

@coo_matvec_p.def_abstract_eval
def _coo_matvec_abstract_eval(data, row, col, v, *, spinfo, transpose):
  assert data.shape == row.shape == col.shape
  assert data.dtype == v.dtype
  assert row.dtype == col.dtype
  assert len(spinfo.shape) == 2
  assert v.ndim == 1
  assert v.shape[0] == (spinfo.shape[0] if transpose else spinfo.shape[1])
  out_shape = spinfo.shape[1] if transpose else spinfo.shape[0]
  return core.ShapedArray((out_shape,), data.dtype)

_coo_matvec_lowering = mlir.lower_fun(
    _coo_matvec_impl, multiple_results=False)

def _coo_matvec_gpu_lowering(coo_matvec_hlo, ctx, data, row, col, v, *, spinfo,
                             transpose):
  data_aval, row_aval, _, x_aval = ctx.avals_in
  dtype = data_aval.dtype
  if dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
    warnings.warn(f"coo_matvec cusparse/hipsparse lowering not available for {dtype=}. "
                  "Falling back to default implementation.", CuSparseEfficiencyWarning)
    return _coo_matvec_lowering(ctx, data, row, col, v, spinfo=spinfo, transpose=transpose)

  if spinfo.rows_sorted:
    shape = spinfo.shape
  elif spinfo.cols_sorted:
    row, col = col, row
    transpose = not transpose
    shape = spinfo.shape[::-1]
  else:
    warnings.warn("coo_matvec GPU lowering requires matrices with sorted rows or sorted cols. "
                  "To sort the rows in your matrix, use e.g. mat = mat._sort_indices(). Falling "
                  "back to the default implementation.", CuSparseEfficiencyWarning)
    return _coo_matvec_lowering(ctx, data, row, col, v, spinfo=spinfo,
                                transpose=transpose)

  return [coo_matvec_hlo(
      data, row, col, v, shape=shape, transpose=transpose,
      index_dtype=row_aval.dtype, data_dtype=dtype, x_dtype=x_aval.dtype)]


def _coo_matvec_jvp_mat(data_dot, data, row, col, v, *, spinfo, transpose):
  return _coo_matvec(data_dot, row, col, v, spinfo=spinfo, transpose=transpose)

def _coo_matvec_jvp_vec(v_dot, data, row, col, v, *, spinfo, transpose):
  return _coo_matvec(data, row, col, v_dot, spinfo=spinfo, transpose=transpose)

def _coo_matvec_transpose(ct, data, row, col, v, *, spinfo, transpose):
  assert not ad.is_undefined_primal(row)
  assert not ad.is_undefined_primal(col)

  if ad.is_undefined_primal(v):
    return data, row, col, _coo_matvec(data, row, col, ct, spinfo=spinfo, transpose=not transpose)
  else:
    v = jnp.asarray(v)
    # The following line does this, but more efficiently:
    # return _coo_extract(row, col, jnp.outer(ct, v)), row, col, v
    return ct[row] * v[col], row, col, v

ad.defjvp(coo_matvec_p, _coo_matvec_jvp_mat, None, None, _coo_matvec_jvp_vec)
ad.primitive_transposes[coo_matvec_p] = _coo_matvec_transpose
mlir.register_lowering(coo_matvec_p, _coo_matvec_lowering)
dispatch.simple_impl(coo_matvec_p)

if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(
      coo_matvec_p,
      partial(_coo_matvec_gpu_lowering, gpu_sparse.cuda_coo_matvec),
      platform='cuda')
if gpu_sparse.rocm_is_supported:
  mlir.register_lowering(
      coo_matvec_p,
      partial(_coo_matvec_gpu_lowering, gpu_sparse.rocm_coo_matvec),
      platform='rocm')


#--------------------------------------------------------------------
# coo_matmat

coo_matmat_p = core.Primitive('coo_matmat')

def coo_matmat(mat: COO, B: Array, *, transpose: bool = False) -> Array:
  """Product of COO sparse matrix and a dense matrix.

  Args:
    mat : COO matrix
    B : array of shape ``(mat.shape[0] if transpose else mat.shape[1], cols)`` and
      dtype ``mat.dtype``
    transpose : boolean specifying whether to transpose the sparse matrix
      before computing.

  Returns:
    C : array of shape ``(mat.shape[1] if transpose else mat.shape[0], cols)``
      representing the matrix vector product.
  """
  data, row, col = mat._bufs
  return _coo_matmat(data, row, col, B, spinfo=mat._info, transpose=transpose)

def _coo_matmat(data: Array, row: Array, col: Array, B: Array, *, spinfo: COOInfo, transpose: bool = False) -> Array:
  """Product of COO sparse matrix and a dense matrix.

  Args:
    data : array of shape ``(nse,)``.
    row : array of shape ``(nse,)``
    col : array of shape ``(nse,)`` and dtype ``row.dtype``
    B : array of shape ``(shape[0] if transpose else shape[1], cols)`` and
      dtype ``data.dtype``
    shape : length-2 tuple representing the matrix shape
    transpose : boolean specifying whether to transpose the sparse matrix
      before computing.

  Returns:
    C : array of shape ``(shape[1] if transpose else shape[0], cols)``
      representing the matrix vector product.
  """
  return coo_matmat_p.bind(data, row, col, B, spinfo=spinfo, transpose=transpose)

def _coo_matmat_impl(data, row, col, B, *, spinfo, transpose):
  B = jnp.asarray(B)
  if transpose:
    row, col = col, row
  out_shape = spinfo.shape[1] if transpose else spinfo.shape[0]
  dB = data[:, None] * B[col]
  return jnp.zeros((out_shape, B.shape[1]), dB.dtype).at[row].add(dB)

@coo_matmat_p.def_abstract_eval
def _coo_matmat_abstract_eval(data, row, col, B, *, spinfo, transpose):
  assert data.shape == row.shape == col.shape
  assert data.dtype == B.dtype
  assert B.ndim == 2
  assert len(spinfo.shape) == 2
  assert B.shape[0] == (spinfo.shape[0] if transpose else spinfo.shape[1])
  out_shape = spinfo.shape[1] if transpose else spinfo.shape[0]
  return core.ShapedArray((out_shape, B.shape[1]), data.dtype)

_coo_matmat_lowering = mlir.lower_fun(_coo_matmat_impl, multiple_results=False)

def _coo_matmat_gpu_lowering(coo_matmat_hlo, ctx, data, row, col, B, *, spinfo,
                             transpose):
  data_aval, row_aval, _, B_aval = ctx.avals_in
  dtype = data_aval.dtype
  if dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
    warnings.warn(f"coo_matmat cusparse/hipsprse lowering not available for {dtype=}. "
                  "Falling back to default implementation.", CuSparseEfficiencyWarning)
    return _coo_matmat_lowering(ctx, data, row, col, B, spinfo=spinfo, transpose=transpose)

  if spinfo.rows_sorted:
    shape = spinfo.shape
  elif spinfo.cols_sorted:
    row, col = col, row
    transpose = not transpose
    shape = spinfo.shape[::-1]
  else:
    warnings.warn("coo_matmat GPU lowering requires matrices with sorted rows or sorted cols. "
                  "To sort the rows in your matrix, use e.g. mat = mat._sort_indices(). Falling "
                  "back to the default implementation.", CuSparseEfficiencyWarning)
    return _coo_matmat_lowering(ctx, data, row, col, B, spinfo=spinfo,
                                transpose=transpose)

  return [coo_matmat_hlo(data, row, col, B, shape=shape,
                                      transpose=transpose, x_dtype=B_aval.dtype,
                                      data_dtype=data_aval.dtype,
                                      index_dtype=row_aval.dtype)]


def _coo_matmat_jvp_left(data_dot, data, row, col, B, *, spinfo, transpose):
  return _coo_matmat(data_dot, row, col, B, spinfo=spinfo, transpose=transpose)

def _coo_matmat_jvp_right(B_dot, data, row, col, B, *, spinfo, transpose):
  return _coo_matmat(data, row, col, B_dot, spinfo=spinfo, transpose=transpose)

def _coo_matmat_transpose(ct, data, row, col, B, *, spinfo, transpose):
  assert not ad.is_undefined_primal(row)
  assert not ad.is_undefined_primal(col)
  if ad.is_undefined_primal(B):
    return data, row, col, _coo_matmat(data, row, col, ct, spinfo=spinfo, transpose=not transpose)
  else:
    B = jnp.asarray(B)
    return (ct[row] * B[col]).sum(1), row, col, B

ad.defjvp(coo_matmat_p, _coo_matmat_jvp_left, None, None, _coo_matmat_jvp_right)
ad.primitive_transposes[coo_matmat_p] = _coo_matmat_transpose
mlir.register_lowering(coo_matmat_p, _coo_matmat_lowering)
dispatch.simple_impl(coo_matmat_p)

if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(
      coo_matmat_p,
      partial(_coo_matmat_gpu_lowering, gpu_sparse.cuda_coo_matmat),
      platform='cuda')
if gpu_sparse.rocm_is_supported:
  mlir.register_lowering(
      coo_matmat_p,
      partial(_coo_matmat_gpu_lowering, gpu_sparse.rocm_coo_matmat),
      platform='rocm')
