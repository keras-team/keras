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

"""N:M-sparsity associated primitives."""

from jax._src import core
from jax._src import dispatch
from jax._src.lax.lax import DotDimensionNumbers
from jax._src.lib import gpu_sparse
from jax._src.lib.mlir.dialects import mhlo
from jax._src.typing import Array, DTypeLike
from jax.interpreters import mlir
import jax.numpy as jnp
import numpy as np

# --------------------------------------------------------------------
# nm_spmm

nm_spmm_p = core.Primitive("sparse_dense_matmul")

_supported_input_types = (jnp.int8, jnp.int16, jnp.float16, jnp.bfloat16)
_supported_output_types = (jnp.bfloat16, jnp.float32)


def nm_spmm(
    lhs: Array,
    rhs: Array,
    metadata: Array,
    dimension_numbers: DotDimensionNumbers = (((1,), (0,)), (tuple(), tuple())),
    sparse_operand_idx: int = 0,
    output_dtype: DTypeLike = jnp.bfloat16,
) -> Array:
  """Dot operation where one of the operands has N:M sparsity.

  Args:
    lhs: An ndarray (first dot operand).
    rhs: An ndarray (second dot operand).
    metadata: An ndarray with structured sparsity metadata for the contracting
      dimension. For 2:4 sparsity it should contain (N=2) two-bit index values
      for each (M=4) element group.
    dimension_numbers: a tuple of tuples of the form `((lhs_contracting_dims,
      rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims))`.
    sparse_operand_idx: index of the sparse operand (0 or 1).
    output_dtype: result type.

  Returns:
    An ndarray dense array containing the result.
  """
  return nm_spmm_p.bind(
      lhs,
      rhs,
      metadata,
      dimension_numbers=dimension_numbers,
      sparse_operand_idx=sparse_operand_idx,
      output_dtype=output_dtype,
  )


def _calc_groups_per_element(n, m):
  group_bits = n * (m.bit_length() - 1)  # 4 bits per group for 2:4
  return 16 // group_bits


def _validate_dnums(rank, contract, batch, name):
  non_contract = tuple(sorted(set(range(rank)) - set(contract + batch)))
  if sorted(non_contract + contract + batch) != list(range(rank)):
    raise TypeError(f"Incorrect dimension numbers for {name}")
  return non_contract


def _validate_metadata(lhs, rhs, metadata, dimension_numbers, index, n=2, m=4):
  assert index in (0, 1)
  size_factor = n * _calc_groups_per_element(n, m)

  sparse = [lhs, rhs][index]
  sparse_contract = dimension_numbers[0][index]
  if metadata.dtype != np.uint16:
    raise TypeError(f"Metadata must be uint16, got {metadata.dtype}")
  if sparse_contract[0] != sparse.ndim - 1:
    raise TypeError("Contracting dimension must be the minor one")
  if metadata.shape[:-1] != sparse.shape[:-1]:
    raise TypeError(
        "Metadata shape must match the operand shape (except for the"
        " contracting dimension)"
    )
  if metadata.shape[-1] * size_factor != sparse.shape[-1]:
    raise TypeError(
        f"Metadata must be exactly {size_factor} times less than the"
        f" contracting dimension for {n}:{m} structured sparsity (expected"
        f" {sparse.shape[-1] // size_factor}, got {metadata.shape[-1]})"
    )
  if sparse.shape[-1] % size_factor != 0:
    raise NotImplementedError("Metadata with padding is not supported")

  dense = [lhs, rhs][1 - index]
  dense_contract = dimension_numbers[0][1 - index]
  a, b = sparse.shape[sparse_contract[0]], dense.shape[dense_contract[0]]
  if n * b != m * a:
    raise TypeError(
        f"Contracting dimension sizes should have {n}:{m} ratio, got {a}:{b}"
    )


def _infer_result_shape(lhs, rhs, dimension_numbers):
  ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers
  if len(lhs_contract) != 1 or len(rhs_contract) != 1:
    raise TypeError("Only single contracting dimension is supported")
  lhs_dims = _validate_dnums(lhs.ndim, lhs_contract, lhs_batch, "lhs")
  rhs_dims = _validate_dnums(rhs.ndim, rhs_contract, rhs_batch, "rhs")
  if len(lhs_dims) != 1 or len(rhs_dims) != 1:
    raise TypeError("Only single non-contracting dimension is supported")
  batch = [lhs.shape[i] for i in lhs_batch]
  if batch != [rhs.shape[i] for i in rhs_batch]:
    raise TypeError("Batch dimension sizes do not match")
  return tuple(batch + [lhs.shape[lhs_dims[0]], rhs.shape[rhs_dims[0]]])


def _nm_spmm_default_lowering(*_args, **_kwargs):
  raise NotImplementedError("Sparse N:M matmul is only implemented on GPU")


def _nm_spmm_gpu_lowering(
    ctx,
    lhs,
    rhs,
    metadata,
    *,
    dimension_numbers,
    sparse_operand_idx,
    output_dtype,
):
  assert sparse_operand_idx in (0, 1)
  sparsity_descriptor = mhlo.SparsityDescriptor.get(
      dimension=dimension_numbers[0][sparse_operand_idx][0], n=2, m=4
  )
  dot_dnums = mhlo.DotDimensionNumbers.get(
      lhs_batching_dimensions=dimension_numbers[1][sparse_operand_idx],
      rhs_batching_dimensions=dimension_numbers[1][1 - sparse_operand_idx],
      lhs_contracting_dimensions=dimension_numbers[0][sparse_operand_idx],
      rhs_contracting_dimensions=dimension_numbers[0][1 - sparse_operand_idx],
  )
  dot_type = ctx.avals_out[0]
  key = ["lhs_sparsity", "rhs_sparsity"][sparse_operand_idx]
  kwargs = {key: sparsity_descriptor}
  op = mhlo.SparseDotOp(
      mlir.aval_to_ir_type(dot_type), lhs, rhs, [metadata], dot_dnums, **kwargs
  )
  return op.results


@nm_spmm_p.def_abstract_eval
def _nm_spmm_abstract_eval(
    lhs, rhs, metadata, *, dimension_numbers, sparse_operand_idx, output_dtype
):
  if lhs.dtype not in _supported_input_types:
    raise TypeError(f"Unsupported lhs input type: {lhs.dtype}")
  if rhs.dtype not in _supported_input_types:
    raise TypeError(f"Unsupported rhs input type: {rhs.dtype}")
  if output_dtype not in _supported_output_types:
    raise TypeError(f"Unsupported output type: {output_dtype}")

  res_shape = _infer_result_shape(lhs, rhs, dimension_numbers)
  _validate_metadata(lhs, rhs, metadata, dimension_numbers, sparse_operand_idx)
  return core.ShapedArray(res_shape, output_dtype)


mlir.register_lowering(nm_spmm_p, _nm_spmm_default_lowering)
dispatch.simple_impl(nm_spmm_p)

if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(nm_spmm_p, _nm_spmm_gpu_lowering, platform="cuda")

if gpu_sparse.rocm_is_supported:
  mlir.register_lowering(nm_spmm_p, _nm_spmm_gpu_lowering, platform="rocm")

# --------------------------------------------------------------------
# nm_pack

nm_pack_p = core.Primitive("sparse_pack_nm")


def nm_pack(mask: Array, n=2, m=4) -> Array:
  """Generate metadata tensor for an N:M mask.

  Args:
    mask: Predicates for the input tensor, where the elements are grouped in the
      minor dimension. In each group of size M there should be exactly N true
      values, which mark the data elements to keep.
    n: Number of non-zero elements in a group.
    m: Group size.

  Returns:
    An ndarray containing only the masked input elements.
  """
  return nm_pack_p.bind(mask, n=n, m=m)


def _compress(data, n, m, k):
  result = []
  expected = n * (k // m)
  for i in range(0, len(data), k):
    index = tuple(jnp.nonzero(data[i : i + k], size=expected)[0] % m)
    value = sum(j * pow(m, i) for i, j in enumerate(index))
    result.append(value)
  return jnp.array(result, dtype=np.uint16)


@nm_pack_p.def_impl
def _nm_pack_impl(mask, *, n, m):
  batch_size = m * _calc_groups_per_element(n, m)
  return jnp.apply_along_axis(
      lambda x: _compress(x, n, m, batch_size), -1, mask
  )


@nm_pack_p.def_abstract_eval
def _nm_pack_abstract_eval(mask, *, n, m):
  size_factor = m * _calc_groups_per_element(n, m)
  if mask.dtype != bool:
    raise TypeError(f"Mask should be bool, got {mask.dtype}")
  if mask.shape[-1] % size_factor != 0:
    raise TypeError(
        f"Inner dimension size should be divisible by {size_factor}, got"
        f" {mask.shape}"
    )
  res_shape = list(mask.shape)
  res_shape[-1] //= size_factor
  return core.ShapedArray(res_shape, np.uint16)


_nm_pack_lowering = mlir.lower_fun(_nm_pack_impl, multiple_results=False)
mlir.register_lowering(nm_pack_p, _nm_pack_lowering)
dispatch.simple_impl(nm_pack_p)
