# Copyright 2022 The JAX Authors.
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

"""BCSR (Bached compressed row) matrix object and associated primitives."""
from __future__ import annotations

from collections.abc import Sequence
from functools import partial
import operator
from typing import Any, NamedTuple
import warnings

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax import tree_util
from jax.experimental.sparse._base import JAXSparse
from jax.experimental.sparse import bcoo
from jax.experimental.sparse.util import (
    nfold_vmap, _count_stored_elements,
    _csr_to_coo, CuSparseEfficiencyWarning, SparseInfo, Shape)
from jax.util import split_list, safe_zip

from jax._src import api_util
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src.lax.lax import DotDimensionNumbers, _dot_general_batch_dim_nums
from jax._src.lib import gpu_sparse
from jax._src.lib.mlir.dialects import hlo
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.typing import Array, ArrayLike, DTypeLike


def bcsr_eliminate_zeros(mat: BCSR, nse: int | None = None) -> BCSR:
  """Eliminate zeros in BCSR representation."""
  return BCSR.from_bcoo(bcoo.bcoo_eliminate_zeros(mat.to_bcoo(), nse=nse))


def bcsr_sum_duplicates(mat: BCSR, nse: int | None = None) -> BCSR:
  """Sums duplicate indices within a BCSR array, returning an array with sorted indices.

  Args:
    mat : BCSR array
    nse : integer (optional). The number of specified elements in the output matrix. This must
      be specified for bcoo_sum_duplicates to be compatible with JIT and other JAX transformations.
      If not specified, the optimal nse will be computed based on the contents of the data and
      index arrays. If specified nse is larger than necessary, data and index arrays will be padded
      with standard fill values. If smaller than necessary, data elements will be dropped from the
      output matrix.

  Returns:
    mat_out : BCSR array with sorted indices and no duplicate indices.
  """
  return BCSR.from_bcoo(bcoo.bcoo_sum_duplicates(mat.to_bcoo(), nse=nse))


def _bcsr_batch_dims_to_front(batched_args, batch_dims, spinfo, batch_size=None):
  data, indices, indptr = batched_args
  data_bdim, indices_bdim, indptr_bdim = batch_dims
  n_batch = indices.ndim - 1 + int(indices_bdim is None)
  if not all(b is None or 0 <= b < n_batch for b in batch_dims):
    raise NotImplementedError("batch_dims must be None or satisfy 0 < dim < n_batch. "
                              f"Got {batch_dims=} for {n_batch=}.")
  batched_data, batched_indices, batched_indptr = (
      lax.expand_dims(arg, [0]) if bdim is None else jnp.moveaxis(arg, bdim, 0)
      for arg, bdim in [(data, data_bdim), (indices, indices_bdim), (indptr, indptr_bdim)])
  if batch_size is None:
    batch_size = max(arg.shape[dim] for arg, dim in zip(batched_args, batch_dims) if dim is not None)
  batched_spinfo = SparseInfo((batch_size, *spinfo.shape),
                              indices_sorted=spinfo.indices_sorted,
                              unique_indices=spinfo.unique_indices)
  return batched_data, batched_indices, batched_indptr, batched_spinfo


class BCSRProperties(NamedTuple):
  n_batch: int
  n_dense: int
  nse: int


def _compatible(shape1: Sequence[int], shape2: Sequence[int]) -> bool:
  return all(s1 in (1, s2) for s1, s2 in safe_zip(shape1, shape2))


def _validate_bcsr_indices(indices: jax.Array, indptr: jax.Array,
                           shape: Sequence[int]) -> BCSRProperties:
  assert jnp.issubdtype(indices.dtype, jnp.integer)
  assert jnp.issubdtype(indptr.dtype, jnp.integer)
  shape = tuple(shape)

  nse = indices.shape[-1]
  n_batch = indices.ndim - 1
  n_dense = len(shape) - n_batch - 2
  assert n_dense >= 0

  if not _compatible(indices.shape[:n_batch], shape[:n_batch]):
    raise ValueError(f"indices batch dimensions not compatible for {indices.shape=}, {shape=}")
  if not _compatible(indptr.shape[:n_batch], shape[:n_batch]):
    raise ValueError(f"indptr batch dimensions not compatible for {indptr.shape=}, {shape=}")
  if indptr.shape[n_batch:] != (shape[n_batch] + 1,):
    raise ValueError("indptr shape must match the matrix shape plus 1.")

  return BCSRProperties(n_batch=n_batch, n_dense=n_dense, nse=nse)


def _validate_bcsr(data: jax.Array, indices: jax.Array,
                   indptr: jax.Array, shape: Sequence[int]) -> BCSRProperties:
  props = _validate_bcsr_indices(indices, indptr, shape)
  shape = tuple(shape)
  n_batch, n_dense, nse = props.n_batch, props.n_dense, props.nse
  n_sparse = len(shape) - n_batch - n_dense
  if n_sparse != 2:
    raise ValueError("BCSR array must have 2 sparse dimensions; "
                     f"{n_sparse} is given.")
  if not _compatible(data.shape[:n_batch], shape[:n_batch]):
    raise ValueError(f"data batch dimensions not compatible for {data.shape=}, {shape=}")
  if data.shape[-(n_dense + 1):] != (nse,) + shape[n_batch + 2:]:
    raise ValueError(f"Invalid {data.shape=} for {nse=}, {n_batch=}, {n_dense=}")
  return props


def _bcsr_to_bcoo(indices: jax.Array, indptr: jax.Array, *,
                  shape: Sequence[int]) -> jax.Array:
  """Given BCSR (indices, indptr), return BCOO (indices)."""
  n_batch, _, _ = _validate_bcsr_indices(indices, indptr, shape)
  csr_to_coo = nfold_vmap(_csr_to_coo, n_batch)
  return jnp.stack(csr_to_coo(indices, indptr), axis=indices.ndim)


def _bcoo_to_bcsr(indices: Array, *, shape: Sequence[int],
                  index_dtype: DTypeLike = jnp.int32) -> tuple[Array, Array]:
  """Given BCOO (indices), return BCSR (indices, indptr).

  Note: this assumes that ``indices`` are lexicographically sorted within each batch.
  """
  n_batch, n_sparse, _, _ = bcoo._validate_bcoo_indices(indices, shape)

  if n_sparse != 2:
    raise ValueError("Must have 2 sparse dimensions to be converted to BCSR.")

  n_rows = shape[n_batch]

  @partial(nfold_vmap, N=n_batch, broadcasted=False)
  def get_ptr(i):
    indptr = jnp.zeros(n_rows + 1, index_dtype)
    return indptr.at[1:].set(jnp.cumsum(
        jnp.bincount(i, length=n_rows).astype(index_dtype)))

  return indices[..., 1], get_ptr(indices[..., 0])


#--------------------------------------------------------------------
# bcsr_fromdense
bcsr_fromdense_p = core.Primitive('bcsr_fromdense')
bcsr_fromdense_p.multiple_results = True


_TRACED_NSE_ERROR = """
The error arose for the nse argument of bcsr_fromdense. In order for
BCSR.fromdense() to be used in traced/compiled code, you must pass a concrete
value to the nse (number of stored elements) argument.
"""


def bcsr_fromdense(mat: ArrayLike, *, nse: int | None = None, n_batch: int = 0,
                   n_dense:int = 0, index_dtype: DTypeLike = jnp.int32) -> BCSR:
  """Create BCSR-format sparse matrix from a dense matrix.

  Args:
    mat : array to be converted to BCOO.
    nse : number of stored elements in each batch
    n_batch : number of batch dimensions (default: 0)
    n_dense : number of dense dimensions (default: 0)
    index_dtype : dtype of sparse indices (default: int32)

  Returns:
    mat_bcsr: BCSR representation of the matrix.
  """
  mat_array = jnp.asarray(mat)
  nse_arr: int | Array | None = nse
  if nse_arr is None:
    nse_arr = _count_stored_elements(mat_array, n_batch, n_dense)
  nse_int: int = core.concrete_or_error(operator.index, nse_arr, _TRACED_NSE_ERROR)
  return BCSR(_bcsr_fromdense(mat_array, nse=nse_int, n_batch=n_batch,
                              n_dense=n_dense, index_dtype=index_dtype),
              shape=mat_array.shape)


def _bcsr_fromdense(mat: ArrayLike, *, nse: int, n_batch: int = 0, n_dense: int = 0,
                    index_dtype: DTypeLike = jnp.int32) -> tuple[Array, Array, Array]:
  """Create BCSR-format sparse matrix from a dense matrix.

  Args:
    mat : array to be converted to BCSR, with
      ``ndim = n_batch + n_sparse + n_dense``.
    nse : number of stored elements in each batch.
    n_batch : number of batch dimensions (default: 0)
    n_dense : number of dense dimensions (default: 0)
    index_dtype : dtype of sparse indices (default: int32)

  Returns:
    data : array of shape
    ``mat.shape[:n_batch] + (nse,) + mat.shape[mat.ndim - n_dense:]``
      and dtype ``mat.dtype``
    indices : array of shape ``mat.shape[:n_batch] + (nse,)`` and dtype of
      ``index_type``.
    indptr: array of shape ``mat.shape[:n_batch] + (mat.shape[n_batch] + 1,)``
      and dtype of ``index_type``.
  """
  mat = jnp.asarray(mat)
  nse = core.concrete_or_error(operator.index, nse, _TRACED_NSE_ERROR)
  return bcsr_fromdense_p.bind(mat, nse=nse, n_batch=n_batch, n_dense=n_dense,
                               index_dtype=index_dtype)


@bcsr_fromdense_p.def_impl
def _bcsr_fromdense_impl(mat, *, nse, n_batch, n_dense, index_dtype):
  mat = jnp.asarray(mat)
  n_sparse = mat.ndim - n_dense - n_batch
  if n_sparse != 2:
    raise ValueError("bcsr_fromdense: must have 2 sparse dimensions.")
  bcoo_mat = bcoo.bcoo_fromdense(mat, nse=nse, index_dtype=index_dtype,
                                 n_dense=n_dense, n_batch=n_batch)
  indices, indptr = _bcoo_to_bcsr(bcoo_mat.indices, shape=mat.shape)
  return bcoo_mat.data, indices, indptr


@bcsr_fromdense_p.def_abstract_eval
def _bcsr_fromdense_abstract_eval(mat, *, nse, n_batch, n_dense, index_dtype):
  n_sparse = mat.ndim - n_batch - n_dense
  if n_sparse != 2:
    raise ValueError("bcsr_fromdense: must have 2 sparse dimensions.")
  data_shape = mat.shape[:n_batch] + (nse,) + mat.shape[n_batch + n_sparse:]
  index_shape = mat.shape[:n_batch] + (nse,)
  indptr_shape = mat.shape[:n_batch] + (mat.shape[n_batch] + 1,)
  return (core.ShapedArray(data_shape, mat.dtype),
          core.ShapedArray(index_shape, index_dtype),
          core.ShapedArray(indptr_shape, index_dtype))


def _bcsr_fromdense_batching_rule(batched_args, batch_dims, *, nse, n_batch,
                                  n_dense, index_dtype):
  M, = batched_args
  bdim, = batch_dims
  if not (0 <= bdim <= n_batch):
    raise ValueError(f"Expected 0 < bdim <= n_batch; got {bdim=}, {n_batch=}")
  return _bcsr_fromdense(M, nse=nse, n_batch=n_batch + 1, n_dense=n_dense, index_dtype=index_dtype), (bdim, bdim, bdim)


def _bcsr_fromdense_jvp(primals, tangents, *, nse, n_batch, n_dense, index_dtype):
  M, = primals
  Mdot, = tangents

  primals_out = _bcsr_fromdense(M, nse=nse, n_batch=n_batch, n_dense=n_dense, index_dtype=index_dtype)
  data, indices, indptr = primals_out

  if type(Mdot) is ad.Zero:
    data_dot = ad.Zero.from_primal_value(data)
  else:
    data_dot = bcsr_extract(indices, indptr, Mdot)

  tangents_out = (data_dot, ad.Zero.from_primal_value(indices), ad.Zero.from_primal_value(indptr))

  return primals_out, tangents_out


def _bcsr_fromdense_transpose(ct, M, *, nse, n_batch, n_dense, index_dtype):
  data, indices, indptr = ct
  n_sparse = M.ndim - n_batch - n_dense
  assert data.shape == M.shape[:n_batch] + (nse,) + M.shape[n_batch + n_sparse:]
  assert indices.shape == M.shape[:n_batch] + (n_sparse, nse)
  assert indptr.shape == M.shape[:n_batch] + (M.shape[n_batch] + 1,)
  assert indices.dtype == index_dtype
  assert indptr.dtype == index_dtype
  if isinstance(indices, ad.Zero) or isinstance(indptr, ad.Zero):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ad.is_undefined_primal(M)
  return _bcsr_todense(data, indices, indptr, spinfo=SparseInfo(M.aval.shape))


ad.primitive_jvps[bcsr_fromdense_p] = _bcsr_fromdense_jvp
ad.primitive_transposes[bcsr_fromdense_p] = _bcsr_fromdense_transpose
batching.primitive_batchers[bcsr_fromdense_p] = _bcsr_fromdense_batching_rule
mlir.register_lowering(bcsr_fromdense_p, mlir.lower_fun(
    _bcsr_fromdense_impl, multiple_results=True))


#----------------------------------------------------------------------
# bcsr_todense
bcsr_todense_p = core.Primitive('bcsr_todense')


def bcsr_todense(mat: BCSR) -> Array:
  """Convert batched sparse matrix to a dense matrix.

  Args:
    mat: BCSR matrix.

  Returns:
    The dense version of ``mat``.
  """
  return _bcsr_todense(mat.data, mat.indices, mat.indptr, spinfo=mat._info)


def _bcsr_todense(data: ArrayLike, indices: ArrayLike, indptr: ArrayLike, *,
                  spinfo: SparseInfo) -> Array:
  """Convert batched sparse matrix to a dense matrix.

  Args:
    data : array of shape ``batch_dims + (nse,) + dense_dims``.
    indices : array of shape ``batch_dims + (nse,)``.
    indptr : array of shape ``batch_dims + (shape[len(batch_dims)] + 1,).
    spinfo : SparseInfo. In particular, this includes the shape
      of the matrix, which is equal to
      ``batch_dims + 2(sparse_dims) + block_dims`` where
      ``len(sparse_dims) == 2``.
  Returns:
    mat : array with specified shape and dtype matching ``data``
  """
  return bcsr_todense_p.bind(jnp.asarray(data), jnp.asarray(indices),
                             jnp.asarray(indptr), spinfo=spinfo)


@bcsr_todense_p.def_impl
def _bcsr_todense_impl(data, indices, indptr, *, spinfo):
  shape = spinfo.shape
  bcoo_indices = _bcsr_to_bcoo(indices, indptr, shape=shape)
  return (bcoo.BCOO((data, bcoo_indices), shape=shape)).todense()


@bcsr_todense_p.def_abstract_eval
def _bcsr_todense_abstract_eval(data, indices, indptr, *, spinfo):
  shape = spinfo.shape
  _validate_bcsr(data, indices, indptr, shape)
  return core.ShapedArray(shape, data.dtype)


def _bcsr_todense_batching_rule(batched_args, batch_dims, *, spinfo):
  data, indices, indptr, spinfo = _bcsr_batch_dims_to_front(batched_args, batch_dims, spinfo)
  return _bcsr_todense(data, indices, indptr, spinfo=spinfo), 0


def _bcsr_todense_jvp(data_dot, data, indices, indptr, *, spinfo):
  del data
  return _bcsr_todense(data_dot, indices, indptr, spinfo=spinfo)


def _bcsr_todense_transpose(ct, data, indices, indptr, *, spinfo):
  shape = spinfo.shape
  assert ad.is_undefined_primal(data)
  if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ct.shape == shape
  assert ct.dtype == data.aval.dtype
  return bcsr_extract(indices, indptr, ct), indices, indptr


ad.defjvp(bcsr_todense_p, _bcsr_todense_jvp, None, None)
ad.primitive_transposes[bcsr_todense_p] = _bcsr_todense_transpose
batching.primitive_batchers[bcsr_todense_p] = _bcsr_todense_batching_rule
mlir.register_lowering(bcsr_todense_p, mlir.lower_fun(
    _bcsr_todense_impl, multiple_results=False))


#--------------------------------------------------------------------
# bcsr_extract
bcsr_extract_p = core.Primitive('bcsr_extract')


def bcsr_extract(indices: ArrayLike, indptr: ArrayLike, mat: ArrayLike) -> Array:
  """Extract values from a dense matrix at given BCSR (indices, indptr).

  Args:
    indices: An ndarray; see BCSR indices.
    indptr: An ndarray; see BCSR indptr.
    mat: A dense matrix.

  Returns:
    An ndarray; see BCSR data.
  """
  return bcsr_extract_p.bind(indices, indptr, mat)


@bcsr_extract_p.def_impl
def _bcsr_extract_impl(indices, indptr, mat):
  mat = jnp.asarray(mat)
  bcoo_indices = _bcsr_to_bcoo(indices, indptr, shape=mat.shape)
  return bcoo._bcoo_extract(bcoo_indices, mat)


@bcsr_extract_p.def_abstract_eval
def _bcsr_extract_abstract_eval(indices, indptr, mat):
  n_batch, n_dense, nse = _validate_bcsr_indices(indices, indptr, mat.shape)
  out_shape = mat.shape[:n_batch] + (nse,) + mat.shape[mat.ndim - n_dense:]
  return core.ShapedArray(out_shape, mat.dtype)


def _bcsr_extract_jvp(arr_dot, indices, indptr, arr):
  assert arr_dot.shape == arr.shape
  return bcsr_extract(indices, indptr, arr_dot)


def _bcsr_extract_transpose(ct, indices, indptr, arr):
  assert ad.is_undefined_primal(arr)
  if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ct.dtype == arr.aval.dtype
  return indices, indptr, _bcsr_todense(ct, indices, indptr, spinfo=SparseInfo(arr.aval.shape))


def _bcsr_extract_batching_rule(batched_args, batch_dims):
  indices, indptr, arr = batched_args
  bdim_set = {b for b in batch_dims if b is not None}
  if len(bdim_set) != 1:
    # TODO(jakevdp): handle this by moving bdim to front?
    raise NotImplementedError("bcoo_extract with unequal batch dimensions.")
  bdim = next(iter(bdim_set))
  if batch_dims[0] is None:
    indices = lax.expand_dims(indices, (bdim,))
  if batch_dims[1] is None:
    indptr = lax.expand_dims(indptr, (bdim,))
  if batch_dims[2] is None:
    # TODO(jakevdp) can we handle this case without explicit broadcasting?
    result_shape = list(arr.shape)
    result_shape.insert(bdim, indices.shape[bdim])
    arr = lax.broadcast_in_dim(arr, result_shape, (bdim,))
  n_batch = indices.ndim - 1
  if bdim >= n_batch:
    raise ValueError(f"{batch_dims=} out of range for indices with {n_batch=}")
  return bcsr_extract(indices, indptr, arr), bdim

ad.defjvp(bcsr_extract_p, None, None, _bcsr_extract_jvp)
ad.primitive_transposes[bcsr_extract_p] = _bcsr_extract_transpose
batching.primitive_batchers[bcsr_extract_p] = _bcsr_extract_batching_rule
mlir.register_lowering(bcsr_extract_p, mlir.lower_fun(
    _bcsr_extract_impl, multiple_results=False))


#----------------------------------------------------------------------
# bcsr_dot_general


bcsr_dot_general_p = core.Primitive('bcsr_dot_general')


def bcsr_dot_general(lhs: BCSR | Array, rhs: Array, *,
                     dimension_numbers: DotDimensionNumbers,
                     precision: None = None,
                     preferred_element_type: None = None,
                     out_sharding=None) -> Array:
  """A general contraction operation.

  Args:
    lhs: An ndarray or BCSR-format sparse array.
    rhs: An ndarray or BCSR-format sparse array..
    dimension_numbers: a tuple of tuples of the form
      `((lhs_contracting_dims, rhs_contracting_dims),
      (lhs_batch_dims, rhs_batch_dims))`.
    precision: unused
    preferred_element_type: unused

  Returns:
    An ndarray or BCSR-format sparse array containing the result. If both inputs
    are sparse, the result will be sparse, of type BCSR. If either input is
    dense, the result will be dense, of type ndarray.
  """
  del precision, out_sharding  # unused
  if isinstance(rhs, (np.ndarray, jax.Array)):
    if isinstance(lhs, (np.ndarray, jax.Array)):
      return lax.dot_general(lhs, rhs, dimension_numbers=dimension_numbers,
                             preferred_element_type=preferred_element_type)

    if isinstance(lhs, BCSR):
      lhs_data, lhs_indices, lhs_indptr = lhs._bufs
      return _bcsr_dot_general(lhs_data, lhs_indices, lhs_indptr, rhs,
                               dimension_numbers=dimension_numbers,
                               preferred_element_type=preferred_element_type,
                               lhs_spinfo=lhs._info)

  raise NotImplementedError("bcsr_dot_general currently implemented for BCSR "
                            "lhs and ndarray rhs.")


def _bcsr_dot_general(lhs_data: jax.Array, lhs_indices: jax.Array,
                      lhs_indptr: jax.Array, rhs: Array, *,
                      dimension_numbers: DotDimensionNumbers,
                      preferred_element_type: Any,
                      lhs_spinfo: SparseInfo) -> Array:
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  cdims = (api_util._ensure_index_tuple(lhs_contract),
           api_util._ensure_index_tuple(rhs_contract))
  bdims = (api_util._ensure_index_tuple(lhs_batch),
           api_util._ensure_index_tuple(rhs_batch))
  return bcsr_dot_general_p.bind(jnp.asarray(lhs_data),
                                 jnp.asarray(lhs_indices),
                                 jnp.asarray(lhs_indptr), jnp.asarray(rhs),
                                 dimension_numbers=(cdims, bdims),
                                 preferred_element_type=preferred_element_type,
                                 lhs_spinfo=lhs_spinfo)


def _bcsr_dot_general_impl(lhs_data, lhs_indices, lhs_indptr, rhs, *,
                           dimension_numbers, preferred_element_type, lhs_spinfo):
  lhs_data = jnp.asarray(lhs_data)
  lhs_bcsr_indices = jnp.asarray(lhs_indices)
  lhs_bcsr_indptr = jnp.asarray(lhs_indptr)
  rhs = jnp.asarray(rhs)
  lhs_bcoo_indices = _bcsr_to_bcoo(lhs_bcsr_indices, lhs_bcsr_indptr,
                                   shape=lhs_spinfo.shape)
  return bcoo._bcoo_dot_general_impl(lhs_data, lhs_bcoo_indices, rhs,
                                     dimension_numbers=dimension_numbers,
                                     preferred_element_type=preferred_element_type,
                                     lhs_spinfo=lhs_spinfo)


@bcsr_dot_general_p.def_abstract_eval
def _bcsr_dot_general_abstract_eval(lhs_data, lhs_indices, lhs_indptr, rhs, *,
                                    dimension_numbers, preferred_element_type, lhs_spinfo):
  (lhs_contracting, _), (lhs_batch, _) = dimension_numbers
  props = _validate_bcsr_indices(lhs_indices, lhs_indptr, lhs_spinfo.shape)
  out_aval = jax.eval_shape(
    partial(lax.dot_general,
            dimension_numbers=dimension_numbers,
            preferred_element_type=preferred_element_type),
    jax.ShapeDtypeStruct(lhs_spinfo.shape, lhs_data.dtype),
    jax.ShapeDtypeStruct(rhs.shape, rhs.dtype))

  if lhs_batch and max(lhs_batch) >= props.n_batch:
    raise NotImplementedError(
      "bcsr_dot_general batch dimensions must be among the batch dimensions in the sparse representtaion.\n"
      f"got {lhs_batch=}, {props.n_batch=}")

  # TODO: support contraction of dense dimensions?
  if any(d >= props.n_batch + 2 for d in lhs_contracting):
    raise NotImplementedError("bcsr_dot_general: contracting over dense dimensions.")

  return core.ShapedArray(out_aval.shape, out_aval.dtype)


def _bcsr_dot_general_jvp_lhs(lhs_data_dot, lhs_data, lhs_indices, lhs_indptr, rhs, *,
                              dimension_numbers, preferred_element_type, lhs_spinfo):
  del lhs_data
  return _bcsr_dot_general(lhs_data_dot, lhs_indices, lhs_indptr, rhs,
                           dimension_numbers=dimension_numbers,
                           preferred_element_type=preferred_element_type,
                           lhs_spinfo=lhs_spinfo)


def _bcsr_dot_general_jvp_rhs(rhs_dot, lhs_data, lhs_indices, lhs_indptr, rhs, *,
                              dimension_numbers, preferred_element_type, lhs_spinfo):
  del rhs
  return _bcsr_dot_general(lhs_data, lhs_indices, lhs_indptr, rhs_dot,
                           dimension_numbers=dimension_numbers,
                           preferred_element_type=preferred_element_type,
                           lhs_spinfo=lhs_spinfo)


def _bcsr_dot_general_transpose(ct, lhs_data, lhs_indices, lhs_indptr, rhs, *,
                                 dimension_numbers, preferred_element_type, lhs_spinfo):
  # TODO(jakevdp): implement this in terms of bcsr_dot_general
  lhs_bcoo_indices = _bcsr_to_bcoo(
    lhs_indices, lhs_indptr, shape=lhs_spinfo.shape)
  data_out, _, rhs_out = bcoo._bcoo_dot_general_transpose(
      ct, lhs_data, lhs_bcoo_indices, rhs, dimension_numbers=dimension_numbers,
      preferred_element_type=preferred_element_type, lhs_spinfo=lhs_spinfo)
  return data_out, lhs_indices, lhs_indptr, rhs_out


def _bcsr_dot_general_batch_rule(batched_args, batch_dims, *,
                                 dimension_numbers, preferred_element_type,
                                 lhs_spinfo):
  *lhs_args, rhs = batched_args
  *lhs_dims, rhs_bdim = batch_dims
  *new_lhs_args, new_lhs_spinfo = _bcsr_batch_dims_to_front(
    lhs_args, lhs_dims, lhs_spinfo,
    batch_size=None if rhs_bdim is None else rhs.shape[rhs_bdim])
  new_dimension_numbers, result_batch_dim = _dot_general_batch_dim_nums(
      (len(lhs_spinfo.shape), rhs.ndim), (0, rhs_bdim), dimension_numbers)
  batched_out = _bcsr_dot_general(*new_lhs_args, rhs, lhs_spinfo=new_lhs_spinfo,
                                  dimension_numbers=new_dimension_numbers,
                                  preferred_element_type=preferred_element_type)
  return batched_out, result_batch_dim


ad.defjvp(bcsr_dot_general_p, _bcsr_dot_general_jvp_lhs, None, None,
          _bcsr_dot_general_jvp_rhs)
ad.primitive_transposes[bcsr_dot_general_p] = _bcsr_dot_general_transpose
batching.primitive_batchers[bcsr_dot_general_p] = _bcsr_dot_general_batch_rule


def _bcsr_correct_out_of_bound_indices(data, indices, indptr, rhs, *, shape):
  props = _validate_bcsr(data, indices, indptr, shape)
  if props.n_batch:
    f = partial(_bcsr_correct_out_of_bound_indices, rhs=rhs, shape=shape[props.n_batch:])
    return nfold_vmap(f, props.n_batch)(data, indices, indptr)
  extent = indptr[-1]
  i_data = lax.broadcasted_iota(indptr.dtype, data.shape, 0)
  data = jnp.where(i_data < extent, data, 0)
  i_indices = lax.broadcasted_iota(indptr.dtype, indices.shape, 0)
  indices = jnp.where(i_indices < extent, indices, 0)
  return [data, indices]

_bcsr_correct_out_of_bound_indices_lowered = mlir.lower_fun(
    _bcsr_correct_out_of_bound_indices, multiple_results=True)

def _bcsr_dot_general_gpu_lowering(
    csr_matvec_lowering, csr_matmat_lowering,
    ctx, lhs_data, lhs_indices, lhs_indptr, rhs, *, dimension_numbers,
    preferred_element_type, lhs_spinfo: SparseInfo):

  if not config.bcoo_cusparse_lowering.value:
    return _bcsr_dot_general_default_lowering(
      ctx, lhs_data, lhs_indices, lhs_indptr, rhs,
      dimension_numbers=dimension_numbers,
      preferred_element_type=preferred_element_type,
      lhs_spinfo=lhs_spinfo)

  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_data_aval, lhs_indices_aval, lhs_indptr_aval, rhs_aval = ctx.avals_in
  props = _validate_bcsr(
      lhs_data_aval, lhs_indices_aval, lhs_indptr_aval, lhs_spinfo.shape)

  use_default_lowering = False
  dtype = lhs_data_aval.dtype
  # TODO(vanderplas, tianjianlu): lower batched matmuls to GPU
  if lhs_batch or rhs_batch:
    # batch dimensions in dot_general are not supported
    use_default_lowering = True
  elif (lhs_data_aval.dtype != rhs_aval.dtype):
    use_default_lowering = True
  elif preferred_element_type is not None and preferred_element_type != lhs_data_aval.dtype:
    use_default_lowering = True
  elif len(lhs_spinfo.shape) != 2 or rhs_aval.ndim not in [1, 2]:
    # only matmat / matvec supported
    use_default_lowering = True
  elif props.n_batch or props.n_dense:
    # batch and dense dimensions in BCSR not supported
    use_default_lowering = True
  elif list(lhs_contract) != [1]:
    # cusparse cannot contract over more than one dimension
    use_default_lowering = True
  elif dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
    # This would be supported if not for the dtype.
    warnings.warn(f'bcsr_dot_general cusparse/hipsparse lowering not available '
                  f'for {dtype=}. Falling back to default implementation.',
                  CuSparseEfficiencyWarning)
    use_default_lowering = True

  if use_default_lowering:
    return _bcsr_dot_general_default_lowering(
      ctx, lhs_data, lhs_indices, lhs_indptr, rhs,
      dimension_numbers=dimension_numbers,
      preferred_element_type=preferred_element_type,
      lhs_spinfo=lhs_spinfo)

  # Account for a bug in cusparse: it references indices and data beyond
  # the extent of indptr.
  lhs_data, lhs_indices = _bcsr_correct_out_of_bound_indices_lowered(
    ctx, lhs_data, lhs_indices, lhs_indptr, rhs, shape=lhs_spinfo.shape)

  if rhs_aval.ndim == 1:
    dot_general_fn = csr_matvec_lowering
    x_dtype = 'x_dtype'
  elif rhs_aval.ndim == 2:
    dot_general_fn = csr_matmat_lowering
    x_dtype = 'B_dtype'
    if rhs_contract[0] == 1:
      rhs = hlo.transpose(rhs, permutation=mlir.dense_int_array([1, 0]))
  else:
    raise ValueError(f"rhs has to be 1d or 2d; get {rhs_aval.ndim}d.")

  return [dot_general_fn(lhs_data, lhs_indices, lhs_indptr, rhs,
                         shape=lhs_spinfo.shape, transpose=False,
                         data_dtype=lhs_data_aval.dtype,
                         index_dtype=lhs_indices_aval.dtype,
                         **{x_dtype: rhs_aval.dtype})]

_bcsr_dot_general_default_lowering = mlir.lower_fun(
    _bcsr_dot_general_impl, multiple_results=False)
mlir.register_lowering(
    bcsr_dot_general_p, _bcsr_dot_general_default_lowering)
dispatch.simple_impl(bcsr_dot_general_p)

if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(bcsr_dot_general_p,
                          partial(_bcsr_dot_general_gpu_lowering,
                                  gpu_sparse.cuda_csr_matvec,
                                  gpu_sparse.cuda_csr_matmat),
                          platform='cuda')
if gpu_sparse.rocm_is_supported:
  mlir.register_lowering(bcsr_dot_general_p,
                          partial(_bcsr_dot_general_gpu_lowering,
                                  gpu_sparse.rocm_csr_matvec,
                                  gpu_sparse.rocm_csr_matmat),
                          platform='rocm')


#----------------------------------------------------------------------
# BCOO functions that maybe should be primitives?

def bcsr_broadcast_in_dim(mat: BCSR, *, shape: Shape, broadcast_dimensions: Sequence[int],
                          sharding=None) -> BCSR:
  result_bcoo = bcoo.bcoo_broadcast_in_dim(
    mat.to_bcoo(), shape=shape, broadcast_dimensions=broadcast_dimensions)
  return BCSR.from_bcoo(result_bcoo)

def bcsr_concatenate(operands: Sequence[BCSR], *, dimension: int) -> BCSR:
  """Sparse implementation of :func:`jax.lax.concatenate`

  Args:
    operands : Sequence of BCSR arrays to concatenate. The arrays must have equal
      shapes, except in the `dimension` axis. Additionally, the arrays must have
      have equivalent batch, sparse, and dense dimensions.
    dimension : Positive integer specifying the dimension along which to concatenate
      the arrays. The dimension must be among batch or sparse dimensions of the input;
      concatenation along dense dimensions is not supported.

  Returns:
    A BCSR array containing the concatenation of the inputs.
  """
  return BCSR.from_bcoo(
    bcoo.bcoo_concatenate([mat.to_bcoo() for mat in operands], dimension=dimension))

@tree_util.register_pytree_node_class
class BCSR(JAXSparse):
  """Experimental batched CSR matrix implemented in JAX."""

  data: jax.Array
  indices: jax.Array
  indptr: jax.Array
  shape: Shape
  nse = property(lambda self: self.indices.shape[-1])
  dtype = property(lambda self: self.data.dtype)
  n_batch = property(lambda self: self.indices.ndim - 1)
  n_sparse = property(lambda _: 2)
  n_dense = property(lambda self: self.data.ndim - self.indices.ndim)
  indices_sorted: bool
  unique_indices: bool
  _bufs = property(lambda self: (self.data, self.indices, self.indptr))
  _info = property(lambda self: SparseInfo(self.shape, self.indices_sorted,
                                           self.unique_indices))

  @property
  def _sparse_shape(self):
    return tuple(self.shape[self.n_batch:self.n_batch + 2])

  def __init__(self, args: tuple[Array, Array, Array], *, shape: Sequence[int],
               indices_sorted: bool = False, unique_indices: bool = False):
    self.data, self.indices, self.indptr = map(jnp.asarray, args)
    self.indices_sorted = indices_sorted
    self.unique_indices = unique_indices
    super().__init__(args, shape=shape)
    _validate_bcsr(self.data, self.indices, self.indptr, self.shape)

  def __repr__(self):
    name = self.__class__.__name__
    try:
      nse = self.nse
      n_batch = self.n_batch
      n_dense = self.n_dense
      dtype = self.dtype
      shape = list(self.shape)
    except Exception:  # pylint: disable=broad-except
      repr_ = f"{name}(<invalid>)"
    else:
      extra = f", {nse=}"
      if n_batch: extra += f", {n_batch=}"
      if n_dense: extra += f", {n_dense=}"
      repr_ = f"{name}({dtype}{shape}{extra})"
    if isinstance(self.data, core.Tracer):
      repr_ = f"{type(self.data).__name__}[{repr_}]"
    return repr_

  def transpose(self, *args, **kwargs):
    raise NotImplementedError("Transpose is not implemented.")

  def tree_flatten(self):
    return (self.data, self.indices, self.indptr), self._info._asdict()

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    obj = object.__new__(cls)
    obj.data, obj.indices, obj.indptr = children
    if aux_data.keys() != {'shape', 'indices_sorted', 'unique_indices'}:
      raise ValueError(f"BCSR.tree_unflatten: invalid {aux_data=}")
    obj.__dict__.update(**aux_data)
    return obj

  @classmethod
  def _empty(cls, shape, *, dtype=None, index_dtype='int32', n_dense=0,
             n_batch=0, nse=0):
    """Create an empty BCSR instance. Public method is sparse.empty()."""
    shape = tuple(shape)
    if n_dense < 0 or n_batch < 0 or nse < 0:
      raise ValueError(f"Invalid inputs: {shape=}, {n_dense=}, {n_batch=}, {nse=}")
    n_sparse = len(shape) - n_dense - n_batch
    if n_sparse != 2:
      raise ValueError("BCSR sparse.empty: must have 2 sparse dimensions.")
    batch_shape, sparse_shape, dense_shape = split_list(shape,
                                                        [n_batch, n_sparse])
    data = jnp.zeros((*batch_shape, nse, *dense_shape), dtype)
    indices = jnp.full((*batch_shape, nse), jnp.array(sparse_shape[1]),
                       index_dtype)
    indptr = jnp.zeros((*batch_shape, sparse_shape[0] + 1), index_dtype)
    return cls((data, indices, indptr), shape=shape)

  def sum_duplicates(self, nse: int | None = None, remove_zeros: bool = True) -> BCSR:
    """Return a copy of the array with duplicate indices summed.

    Additionally, this operation will result in explicit zero entries removed, and
    indices being sorted in lexicographic order.

    Because the size of the resulting representation depends on the values in the
    arrays, this operation is not compatible with JIT or other transforms. To use
    ``sum_duplicates`` in such cases, you may pass a value to `nse` to specify the
    desired size of the output representation.

    Args:
      nse : integer (optional), if specified, gives the number of specified elements in
        the output sparse representation; if it is larger than the number required, data
        will be padded with zeros and indices will be padded with out-of-bounds values.
        If it is smaller than the number required, data will be silently discarded.
      remove_zeros : bool (default=True). If True, remove explicit zeros from the data
        as part of summing duplicates. If False, then explicit zeros at unique indices
        will remain among the specified elements. Note: remove_zeros=True is incompatible
        with autodiff.
    """
    if remove_zeros:
      return bcsr_eliminate_zeros(self, nse=nse)
    else:
      return bcsr_sum_duplicates(self, nse=nse)

  @classmethod
  def fromdense(cls, mat, *, nse=None, index_dtype=np.int32, n_dense=0,
                n_batch=0):
    """Create a BCSR array from a (dense) :class:`Array`."""
    return bcsr_fromdense(mat, nse=nse, index_dtype=index_dtype,
                          n_dense=n_dense, n_batch=n_batch)

  def todense(self):
    """Create a dense version of the array."""
    return bcsr_todense(self)

  def to_bcoo(self) -> bcoo.BCOO:
    coo_indices = _bcsr_to_bcoo(self.indices, self.indptr, shape=self.shape)
    return bcoo.BCOO((self.data, coo_indices), shape=self.shape)

  @classmethod
  def from_bcoo(cls, arr: bcoo.BCOO) -> BCSR:
    if arr.n_sparse != 2:
      raise NotImplementedError(f"BSCR.from_bcoo requires n_sparse=2; got {arr.n_sparse=}")
    if not arr.indices_sorted:
      arr = arr.sort_indices()
    indices, indptr = _bcoo_to_bcsr(arr.indices, shape=arr.shape)
    return cls((arr.data, indices, indptr), shape=arr.shape)

  @classmethod
  def from_scipy_sparse(cls, mat, *, index_dtype=None, n_dense=0, n_batch=0):
    """Create a BCSR array from a :mod:`scipy.sparse` array."""
    if n_dense != 0 or n_batch != 0:
      raise NotImplementedError("BCSR from_scipy_sparse with nonzero n_dense/n_batch.")

    if mat.ndim != 2:
      raise ValueError(f"BCSR from_scipy_sparse requires 2D array; {mat.ndim}D is given.")

    mat = mat.tocsr()
    data = jnp.asarray(mat.data)
    indices = jnp.asarray(mat.indices).astype(index_dtype or jnp.int32)
    indptr = jnp.asarray(mat.indptr).astype(index_dtype or jnp.int32)
    return cls((data, indices, indptr), shape=mat.shape)

#--------------------------------------------------------------------
# vmappable handlers
def _bcsr_to_elt(cont, _, val, axis):
  if axis is None:
    return val
  if axis >= val.n_batch:
    raise ValueError(f"Cannot map in_axis={axis} for BCSR array with n_batch="
                     f"{val.n_batch}. in_axes for batched BCSR operations must "
                     "correspond to a batched dimension.")
  return BCSR((cont(val.data, axis),
               cont(val.indices, axis),
               cont(val.indptr, axis)),
              shape=val.shape[:axis] + val.shape[axis + 1:])


def _bcsr_from_elt(cont, axis_size, elt, axis):
  if axis is None:
    return elt
  if axis > elt.n_batch:
    raise ValueError(f"BCSR: cannot add out_axis={axis} for BCSR array with "
                     f"n_batch={elt.n_batch}. BCSR batch axes must be a "
                     "contiguous block of leading dimensions.")
  return BCSR((cont(axis_size, elt.data, axis),
               cont(axis_size, elt.indices, axis),
               cont(axis_size, elt.indptr, axis)),
              shape=elt.shape[:axis] + (axis_size,) + elt.shape[axis:])

batching.register_vmappable(BCSR, int, int, _bcsr_to_elt, _bcsr_from_elt, None)
