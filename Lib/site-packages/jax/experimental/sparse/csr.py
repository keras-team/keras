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

"""CSR (compressed sparse row) matrix object and associated primitives."""
from __future__ import annotations

from functools import partial
import operator
import warnings

import numpy as np

import jax
from jax.interpreters import mlir
from jax.experimental.sparse._base import JAXSparse
from jax.experimental.sparse.coo import _coo_matmat, _coo_matvec, _coo_todense, COOInfo
from jax.experimental.sparse.util import _csr_to_coo, _csr_extract, CuSparseEfficiencyWarning
from jax import lax
from jax import tree_util
from jax._src import core
from jax._src import dispatch
from jax._src.interpreters import ad
from jax._src.lax.lax import _const
from jax._src.lib import gpu_sparse
from jax._src.numpy.util import promote_dtypes
from jax._src.typing import Array, DTypeLike
import jax.numpy as jnp


Shape = tuple[int, ...]


@tree_util.register_pytree_node_class
class CSR(JAXSparse):
  """Experimental CSR matrix implemented in JAX.

  Note: this class has minimal compatibility with JAX transforms such as
  grad and autodiff, and offers very little functionality. In general you
  should prefer :class:`jax.experimental.sparse.BCOO`.

  Additionally, there are known failures in the case that `nse` is larger
  than the true number of nonzeros in the represented matrix. This situation
  is better handled in BCOO.
  """
  data: jax.Array
  indices: jax.Array
  indptr: jax.Array
  shape: tuple[int, int]
  nse = property(lambda self: self.data.size)
  dtype = property(lambda self: self.data.dtype)
  _bufs = property(lambda self: (self.data, self.indices, self.indptr))

  def __init__(self, args, *, shape):
    self.data, self.indices, self.indptr = map(jnp.asarray, args)
    super().__init__(args, shape=shape)

  @classmethod
  def fromdense(cls, mat, *, nse=None, index_dtype=np.int32):
    if nse is None:
      nse = (mat != 0).sum()
    return csr_fromdense(mat, nse=nse, index_dtype=index_dtype)

  @classmethod
  def _empty(cls, shape, *, dtype=None, index_dtype='int32'):
    """Create an empty CSR instance. Public method is sparse.empty()."""
    shape = tuple(shape)
    if len(shape) != 2:
      raise ValueError(f"CSR must have ndim=2; got {shape=}")
    data = jnp.empty(0, dtype)
    indices = jnp.empty(0, index_dtype)
    indptr = jnp.zeros(shape[0] + 1, index_dtype)
    return cls((data, indices, indptr), shape=shape)

  @classmethod
  def _eye(cls, N, M, k, *, dtype=None, index_dtype='int32'):
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
    col = lax.add(idx, lax.cond(k <= 0, lambda: zero, lambda: k))
    indices = col.astype(index_dtype)
    # TODO(jakevdp): this can be done more efficiently.
    row = lax.sub(idx, lax.cond(k >= 0, lambda: zero, lambda: k))
    indptr = jnp.zeros(N + 1, dtype=index_dtype).at[1:].set(
        jnp.cumsum(jnp.bincount(row, length=N).astype(index_dtype)))
    return cls((data, indices, indptr), shape=(N, M))

  def todense(self):
    return csr_todense(self)

  def transpose(self, axes=None):
    assert axes is None
    return CSC((self.data, self.indices, self.indptr), shape=self.shape[::-1])

  def __matmul__(self, other):
    if isinstance(other, JAXSparse):
      raise NotImplementedError("matmul between two sparse objects.")
    other = jnp.asarray(other)
    data, other = promote_dtypes(self.data, other)
    if other.ndim == 1:
      return _csr_matvec(data, self.indices, self.indptr, other, shape=self.shape)
    elif other.ndim == 2:
      return _csr_matmat(data, self.indices, self.indptr, other, shape=self.shape)
    else:
      raise NotImplementedError(f"matmul with object of shape {other.shape}")

  def tree_flatten(self):
    return (self.data, self.indices, self.indptr), {"shape": self.shape}

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    obj = object.__new__(cls)
    obj.data, obj.indices, obj.indptr = children
    if aux_data.keys() != {'shape'}:
      raise ValueError(f"CSR.tree_unflatten: invalid {aux_data=}")
    obj.__dict__.update(**aux_data)
    return obj


@tree_util.register_pytree_node_class
class CSC(JAXSparse):
  """Experimental CSC matrix implemented in JAX; API subject to change."""
  data: jax.Array
  indices: jax.Array
  indptr: jax.Array
  shape: tuple[int, int]
  nse = property(lambda self: self.data.size)
  dtype = property(lambda self: self.data.dtype)

  def __init__(self, args, *, shape):
    self.data, self.indices, self.indptr = map(jnp.asarray, args)
    super().__init__(args, shape=shape)

  @classmethod
  def fromdense(cls, mat, *, nse=None, index_dtype=np.int32):
    if nse is None:
      nse = (mat != 0).sum()
    return csr_fromdense(mat.T, nse=nse, index_dtype=index_dtype).T

  @classmethod
  def _empty(cls, shape, *, dtype=None, index_dtype='int32'):
    """Create an empty CSC instance. Public method is sparse.empty()."""
    shape = tuple(shape)
    if len(shape) != 2:
      raise ValueError(f"CSC must have ndim=2; got {shape=}")
    data = jnp.empty(0, dtype)
    indices = jnp.empty(0, index_dtype)
    indptr = jnp.zeros(shape[1] + 1, index_dtype)
    return cls((data, indices, indptr), shape=shape)

  @classmethod
  def _eye(cls, N, M, k, *, dtype=None, index_dtype='int32'):
    return CSR._eye(M, N, -k, dtype=dtype, index_dtype=index_dtype).T

  def todense(self):
    return csr_todense(self.T).T

  def transpose(self, axes=None):
    assert axes is None
    return CSR((self.data, self.indices, self.indptr), shape=self.shape[::-1])

  def __matmul__(self, other):
    if isinstance(other, JAXSparse):
      raise NotImplementedError("matmul between two sparse objects.")
    other = jnp.asarray(other)
    data, other = promote_dtypes(self.data, other)
    if other.ndim == 1:
      return _csr_matvec(data, self.indices, self.indptr, other,
                         shape=self.shape[::-1], transpose=True)
    elif other.ndim == 2:
      return _csr_matmat(data, self.indices, self.indptr, other,
                         shape=self.shape[::-1], transpose=True)
    else:
      raise NotImplementedError(f"matmul with object of shape {other.shape}")

  def tree_flatten(self):
    return (self.data, self.indices, self.indptr), {"shape": self.shape}

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    obj = object.__new__(cls)
    obj.data, obj.indices, obj.indptr = children
    if aux_data.keys() != {'shape'}:
      raise ValueError(f"CSC.tree_unflatten: invalid {aux_data=}")
    obj.__dict__.update(**aux_data)
    return obj


#--------------------------------------------------------------------
# csr_todense

csr_todense_p = core.Primitive('csr_todense')

def csr_todense(mat: CSR) -> Array:
  """Convert a CSR-format sparse matrix to a dense matrix.

  Args:
    mat : CSR matrix
  Returns:
    mat_dense: dense version of ``mat``
  """
  return _csr_todense(mat.data, mat.indices, mat.indptr, shape=mat.shape)

def _csr_todense(data: Array, indices: Array, indptr: Array, *, shape: Shape) -> Array:
  """Convert CSR-format sparse matrix to a dense matrix.

  Args:
    data : array of shape ``(nse,)``.
    indices : array of shape ``(nse,)``
    indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
    shape : length-2 tuple representing the matrix shape

  Returns:
    mat : array with specified shape and dtype matching ``data``
  """
  return csr_todense_p.bind(data, indices, indptr, shape=shape)

def _csr_todense_impl(data, indices, indptr, *, shape):
  return _coo_todense(data, *_csr_to_coo(indices, indptr), spinfo=COOInfo(shape=shape))

@csr_todense_p.def_abstract_eval
def _csr_todense_abstract_eval(data, indices, indptr, *, shape):
  assert data.ndim == indices.ndim == indptr.ndim == 1
  assert indices.dtype == indptr.dtype
  assert data.shape == indices.shape
  assert indptr.shape[0] == shape[0] + 1
  return core.ShapedArray(shape, data.dtype)

_csr_todense_lowering = mlir.lower_fun(
    _csr_todense_impl, multiple_results=False)

def _csr_todense_gpu_lowering(csr_todense_hlo, ctx, data, indices, indptr, *,
                              shape):
  data_aval, indices_aval, _ = ctx.avals_in
  dtype = data_aval.dtype
  if not (np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating)):
    warnings.warn(f"csr_todense cusparse/hipsparse lowering not available for {dtype=}. "
                  "Falling back to default implementation.", CuSparseEfficiencyWarning)
    return _csr_todense_lowering(ctx, data, indices, indptr, shape=shape)
  return [csr_todense_hlo(
      data, indices, indptr, shape=shape, data_dtype=dtype,
      index_dtype=indices_aval.dtype)]


def _csr_todense_jvp(data_dot, data, indices, indptr, *, shape):
  return _csr_todense(data_dot, indices, indptr, shape=shape)

def _csr_todense_transpose(ct, data, indices, indptr, *, shape):
  # Note: we assume that transpose has the same sparsity pattern.
  # Can we check this?
  assert ad.is_undefined_primal(data)
  if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ct.shape == shape
  assert indices.aval.dtype == indptr.aval.dtype
  assert ct.dtype == data.aval.dtype
  return _csr_extract(indices, indptr, ct), indices, indptr

ad.defjvp(csr_todense_p, _csr_todense_jvp, None, None)
ad.primitive_transposes[csr_todense_p] = _csr_todense_transpose
mlir.register_lowering(csr_todense_p, _csr_todense_lowering)
dispatch.simple_impl(csr_todense_p)

if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(
      csr_todense_p,
      partial(_csr_todense_gpu_lowering, gpu_sparse.cuda_csr_todense),
      platform='cuda')
if gpu_sparse.rocm_is_supported:
  mlir.register_lowering(
      csr_todense_p,
      partial(_csr_todense_gpu_lowering, gpu_sparse.rocm_csr_todense),
      platform='rocm')


#--------------------------------------------------------------------
# csr_fromdense

csr_fromdense_p = core.Primitive('csr_fromdense')
csr_fromdense_p.multiple_results = True

def csr_fromdense(mat: Array, *, nse: int | None = None, index_dtype: DTypeLike = np.int32) -> CSR:
  """Create a CSR-format sparse matrix from a dense matrix.

  Args:
    mat : array to be converted to CSR.
    nse : number of specified entries in ``mat``. If not specified,
      it will be computed from the input matrix.
    index_dtype : dtype of sparse indices

  Returns:
    mat_coo : CSR representation of the matrix.
  """
  if nse is None:
    nse = int((mat != 0).sum())
  nse_int = core.concrete_or_error(operator.index, nse, "coo_fromdense nse argument")
  return CSR(_csr_fromdense(mat, nse=nse_int, index_dtype=index_dtype), shape=mat.shape)

def _csr_fromdense(mat: Array, *, nse: int, index_dtype: DTypeLike = np.int32) -> tuple[Array, Array, Array]:
  """Create CSR-format sparse matrix from a dense matrix.

  Args:
    mat : array to be converted to CSR.
    nse : number of specified entries in ``mat``
    index_dtype : dtype of sparse indices

  Returns:
    data : array of shape ``(nse,)`` and dtype ``mat.dtype``.
    indices : array of shape ``(nse,)`` and dtype ``index_dtype``
    indptr : array of shape ``(mat.shape[0] + 1,)`` and dtype ``index_dtype``
  """
  mat = jnp.asarray(mat)
  nse = core.concrete_or_error(operator.index, nse, "nse argument of csr_fromdense()")
  return csr_fromdense_p.bind(mat, nse=nse, index_dtype=np.dtype(index_dtype))

def _csr_fromdense_impl(mat, *, nse, index_dtype):
  mat = jnp.asarray(mat)
  assert mat.ndim == 2
  m = mat.shape[0]

  row, col = jnp.nonzero(mat, size=nse)
  data = mat[row, col]

  true_nonzeros = jnp.arange(nse) < (mat != 0).sum()
  data = jnp.where(true_nonzeros, data, 0)
  row = jnp.where(true_nonzeros, row, m)
  indices = col.astype(index_dtype)
  indptr = jnp.zeros(m + 1, dtype=index_dtype).at[1:].set(
      jnp.cumsum(jnp.bincount(row, length=m).astype(index_dtype)))
  return data, indices, indptr

@csr_fromdense_p.def_abstract_eval
def _csr_fromdense_abstract_eval(mat, *, nse, index_dtype):
  data = core.ShapedArray((nse,), mat.dtype)
  indices = core.ShapedArray((nse,), index_dtype)
  indptr = core.ShapedArray((mat.shape[0] + 1,), index_dtype)
  return data, indices, indptr

_csr_fromdense_lowering = mlir.lower_fun(_csr_fromdense_impl,
                                         multiple_results=True)

def _csr_fromdense_gpu_lowering(csr_fromdense_hlo, ctx, mat, *, nse, index_dtype):
  dtype = ctx.avals_in[0].dtype
  if not (np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating)):
    warnings.warn(f"csr_fromdense cusparse/hipsparse lowering not available for {dtype=}. "
                  "Falling back to default implementation.", CuSparseEfficiencyWarning)
    return _csr_fromdense_lowering(ctx, mat, nse=nse, index_dtype=index_dtype)
  data, indices, indptr = csr_fromdense_hlo(
      mat, nnz=nse, index_dtype=np.dtype(index_dtype),
      data_dtype=dtype, index_type=mlir.dtype_to_ir_type(np.dtype(index_dtype)))
  return [data, indices, indptr]


def _csr_fromdense_jvp(primals, tangents, *, nse, index_dtype):
  M, = primals
  Mdot, = tangents

  primals_out = _csr_fromdense(M, nse=nse, index_dtype=index_dtype)
  data, indices, indptr = primals_out

  if type(Mdot) is ad.Zero:
    data_dot = ad.Zero.from_primal_value(data)
  else:
    data_dot = _csr_extract(indices, indptr, Mdot)

  tangents_out = (data_dot, ad.Zero.from_primal_value(indices), ad.Zero.from_primal_value(indptr))

  return primals_out, tangents_out

def _csr_fromdense_transpose(ct, M, *, nse, index_dtype):
  data, indices, indptr = ct
  assert len(data) == nse
  assert indices.dtype == indptr.dtype == index_dtype
  if isinstance(indices, ad.Zero) or isinstance(indptr, ad.Zero):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ad.is_undefined_primal(M)
  return _csr_todense(data, indices, indptr, shape=M.aval.shape)

ad.primitive_jvps[csr_fromdense_p] = _csr_fromdense_jvp
ad.primitive_transposes[csr_fromdense_p] = _csr_fromdense_transpose
mlir.register_lowering(csr_fromdense_p, _csr_fromdense_lowering)
dispatch.simple_impl(csr_fromdense_p)

if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(
      csr_fromdense_p,
      partial(_csr_fromdense_gpu_lowering, gpu_sparse.cuda_csr_fromdense),
      platform='cuda')
if gpu_sparse.rocm_is_supported:
  mlir.register_lowering(
      csr_fromdense_p,
      partial(_csr_fromdense_gpu_lowering, gpu_sparse.rocm_csr_fromdense),
      platform='rocm')

#--------------------------------------------------------------------
# csr_matvec

csr_matvec_p = core.Primitive('csr_matvec')

def csr_matvec(mat: CSR, v: Array, transpose: bool = False) -> Array:
  """Product of CSR sparse matrix and a dense vector.

  Args:
    mat : CSR matrix
    v : one-dimensional array of size ``(shape[0] if transpose else shape[1],)`` and
      dtype ``mat.dtype``
    transpose : boolean specifying whether to transpose the sparse matrix
      before computing.

  Returns:
    y : array of shape ``(mat.shape[1] if transpose else mat.shape[0],)`` representing
      the matrix vector product.
  """
  data, indices, indptr = mat._bufs
  return _csr_matvec(data, indices, indptr, v, shape=mat.shape, transpose=transpose)

def _csr_matvec(data, indices, indptr, v, *, shape, transpose=False):
  """Product of CSR sparse matrix and a dense vector.

  Args:
    data : array of shape ``(nse,)``.
    indices : array of shape ``(nse,)``
    indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
    v : array of shape ``(shape[0] if transpose else shape[1],)``
      and dtype ``data.dtype``
    shape : length-2 tuple representing the matrix shape
    transpose : boolean specifying whether to transpose the sparse matrix
      before computing.

  Returns:
    y : array of shape ``(shape[1] if transpose else shape[0],)`` representing
      the matrix vector product.
  """
  return csr_matvec_p.bind(data, indices, indptr, v, shape=shape, transpose=transpose)

def _csr_matvec_impl(data, indices, indptr, v, *, shape, transpose):
  return _coo_matvec(data, *_csr_to_coo(indices, indptr), v, spinfo=COOInfo(shape=shape), transpose=transpose)

@csr_matvec_p.def_abstract_eval
def _csr_matvec_abstract_eval(data, indices, indptr, v, *, shape, transpose):
  assert len(shape) == 2
  assert v.ndim == data.ndim == indices.ndim == indptr.ndim == 1
  assert data.shape == indices.shape
  assert data.dtype == v.dtype
  assert indices.dtype == indptr.dtype
  assert indptr.shape[0] == shape[0] + 1
  out_shape = shape[1] if transpose else shape[0]
  assert v.shape[0] == (shape[0] if transpose else shape[1])
  return core.ShapedArray((out_shape,), data.dtype)

_csr_matvec_lowering = mlir.lower_fun(_csr_matvec_impl, multiple_results=False)

def _csr_matvec_gpu_lowering(csr_matvec_hlo, ctx, data, indices, indptr, v, *,
                             shape, transpose):
  data_aval, indices_aval, _, v_aval = ctx.avals_in
  dtype = data_aval.dtype
  if dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
    warnings.warn(f"csr_matvec cusparse/hipsparse lowering not available for {dtype=}. "
                  "Falling back to default implementation.", CuSparseEfficiencyWarning)
    return _csr_matvec_lowering(ctx, data, indices, indptr, v, shape=shape,
                                transpose=transpose)
  return [csr_matvec_hlo(
      data, indices, indptr, v, shape=shape, transpose=transpose,
      data_dtype=dtype, index_dtype=indices_aval.dtype, x_dtype=v_aval.dtype)]


def _csr_matvec_jvp_mat(data_dot, data, indices, indptr, v, *, shape, transpose):
  return _csr_matvec(data_dot, indices, indptr, v, shape=shape, transpose=transpose)

def _csr_matvec_jvp_vec(v_dot, data, indices, indptr, v, *, shape, transpose):
  return _csr_matvec(data, indices, indptr, v_dot, shape=shape, transpose=transpose)

def _csr_matvec_transpose(ct, data, indices, indptr, v, *, shape, transpose):
  assert not ad.is_undefined_primal(indices)
  assert not ad.is_undefined_primal(indptr)

  if ad.is_undefined_primal(v):
    return data, indices, indptr, _csr_matvec(data, indices, indptr, ct, shape=shape, transpose=not transpose)
  else:
    v = jnp.asarray(v)
    # The following lines do this, but more efficiently.
    # return _csr_extract(indices, indptr, jnp.outer(ct, v)), indices, indptr, v
    row, col = _csr_to_coo(indices, indptr)
    return ct[row] * v[col], indices, indptr, v

ad.defjvp(csr_matvec_p, _csr_matvec_jvp_mat, None, None, _csr_matvec_jvp_vec)
ad.primitive_transposes[csr_matvec_p] = _csr_matvec_transpose
mlir.register_lowering(csr_matvec_p, _csr_matvec_lowering)
dispatch.simple_impl(csr_matvec_p)

if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(
      csr_matvec_p,
      partial(_csr_matvec_gpu_lowering, gpu_sparse.cuda_csr_matvec),
      platform='cuda')
if gpu_sparse.rocm_is_supported:
  mlir.register_lowering(
      csr_matvec_p,
      partial(_csr_matvec_gpu_lowering, gpu_sparse.rocm_csr_matvec),
      platform='rocm')


#--------------------------------------------------------------------
# csr_matmat

csr_matmat_p = core.Primitive('csr_matmat')

def csr_matmat(mat: CSR, B: Array, *, transpose: bool = False) -> Array:
  """Product of CSR sparse matrix and a dense matrix.

  Args:
    mat : CSR matrix
    B : array of shape ``(mat.shape[0] if transpose else mat.shape[1], cols)`` and
      dtype ``mat.dtype``
    transpose : boolean specifying whether to transpose the sparse matrix
      before computing.

  Returns:
    C : array of shape ``(mat.shape[1] if transpose else mat.shape[0], cols)``
      representing the matrix vector product.
  """
  data, indices, indptr = mat._bufs
  return _csr_matmat(data, indices, indptr, B, shape=mat.shape, transpose=transpose)

def _csr_matmat(data: Array, indices: Array, indptr: Array, B: Array,
                *, shape: Shape, transpose: bool = False) -> Array:
  """Product of CSR sparse matrix and a dense matrix.

  Args:
    data : array of shape ``(nse,)``.
    indices : array of shape ``(nse,)``
    indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
    B : array of shape ``(shape[0] if transpose else shape[1], cols)`` and
      dtype ``data.dtype``
    shape : length-2 tuple representing the matrix shape
    transpose : boolean specifying whether to transpose the sparse matrix
      before computing.

  Returns:
    C : array of shape ``(shape[1] if transpose else shape[0], cols)``
      representing the matrix-matrix product.
  """
  return csr_matmat_p.bind(data, indices, indptr, B, shape=shape, transpose=transpose)

def _csr_matmat_impl(data, indices, indptr, B, *, shape, transpose):
  return _coo_matmat(data, *_csr_to_coo(indices, indptr), B, spinfo=COOInfo(shape=shape), transpose=transpose)

@csr_matmat_p.def_abstract_eval
def _csr_matmat_abstract_eval(data, indices, indptr, B, *, shape, transpose):
  assert len(shape) == 2
  assert data.ndim == indices.ndim == indptr.ndim == 1
  assert B.ndim == 2
  assert data.shape == indices.shape
  assert data.dtype == B.dtype
  assert indices.dtype == indptr.dtype
  assert indptr.shape[0] == shape[0] + 1
  out_shape = shape[1] if transpose else shape[0]
  assert B.shape[0] == (shape[0] if transpose else shape[1])
  return core.ShapedArray((out_shape, B.shape[1]), data.dtype)

_csr_matmat_lowering = mlir.lower_fun(_csr_matmat_impl, multiple_results=False)

def _csr_matmat_gpu_lowering(csr_matmat_hlo, ctx, data, indices, indptr, B, *,
                             shape, transpose):
  data_aval, indices_aval, _, B_aval = ctx.avals_in
  dtype = data_aval.dtype
  if dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
    warnings.warn(f"csr_matmat cusparse/hipsparse lowering not available for {dtype=}. "
                  "Falling back to default implementation.", CuSparseEfficiencyWarning)
    return _csr_matmat_lowering(ctx, data, indices, indptr, B, shape=shape,
                                transpose=transpose)
  return [csr_matmat_hlo(
      data, indices, indptr, B, shape=shape, transpose=transpose,
      index_dtype=indices_aval.dtype, data_dtype=data_aval.dtype,
      B_dtype=B_aval.dtype)]


def _csr_matmat_jvp_left(data_dot, data, indices, indptr, B, *, shape, transpose):
  return _csr_matmat(data_dot, indices, indptr, B, shape=shape, transpose=transpose)

def _csr_matmat_jvp_right(B_dot, data, indices, indptr, B, *, shape, transpose):
  return _csr_matmat(data, indices, indptr, B_dot, shape=shape, transpose=transpose)

def _csr_matmat_transpose(ct, data, indices, indptr, B, *, shape, transpose):
  assert not ad.is_undefined_primal(indices)
  assert not ad.is_undefined_primal(indptr)

  if ad.is_undefined_primal(B):
    return data, indices, indptr, _csr_matmat(data, indices, indptr, ct, shape=shape, transpose=not transpose)
  else:
    B = jnp.asarray(B)
    row, col = _csr_to_coo(indices, indptr)
    return (ct[row] * B[col]).sum(1), indices, indptr, B

ad.defjvp(csr_matmat_p, _csr_matmat_jvp_left, None, None, _csr_matmat_jvp_right)
ad.primitive_transposes[csr_matmat_p] = _csr_matmat_transpose
mlir.register_lowering(csr_matmat_p, _csr_matmat_lowering)
dispatch.simple_impl(csr_matmat_p)

if gpu_sparse:
  if gpu_sparse.cuda_is_supported:
    mlir.register_lowering(
        csr_matmat_p,
        partial(_csr_matmat_gpu_lowering, gpu_sparse.cuda_csr_matmat),
        platform='cuda')
  if gpu_sparse.rocm_is_supported:
    mlir.register_lowering(
        csr_matmat_p,
        partial(_csr_matmat_gpu_lowering, gpu_sparse.rocm_csr_matmat),
        platform='rocm')
