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

"""Sparse utilities."""

import functools
from typing import NamedTuple

import numpy as np
import jax
from jax import lax
from jax import tree_util
from jax import vmap
from jax._src import core
from jax._src.api_util import flatten_axes
import jax.numpy as jnp
from jax.util import safe_zip
from jax._src.lax.lax import _dot_general_shape_rule, DotDimensionNumbers
from jax._src.typing import Array

class SparseEfficiencyError(ValueError):
  pass

class SparseEfficiencyWarning(UserWarning):
  pass

class CuSparseEfficiencyWarning(SparseEfficiencyWarning):
  pass

Shape = tuple[int, ...]

class SparseInfo(NamedTuple):
  shape: Shape
  indices_sorted: bool = False
  unique_indices: bool = False

#--------------------------------------------------------------------
# utilities
# TODO: possibly make these primitives, targeting cusparse routines
#       csr2coo/coo2csr/SPDDMM

def nfold_vmap(fun, N, *, broadcasted=True, in_axes=0):
  """Convenience function to apply (broadcasted) vmap N times."""
  _vmap = broadcasting_vmap if broadcasted else vmap
  for _ in range(N):
    fun = _vmap(fun, in_axes=in_axes)
  return fun

def broadcasting_vmap(fun, in_axes=0, out_axes=0):
  @functools.wraps(fun)
  def batched_fun(*args):
    args_flat, in_tree  = tree_util.tree_flatten(args)
    in_axes_flat = flatten_axes("vmap in_axes", in_tree, in_axes, kws=False)
    size = max(arg.shape[i] for arg, i in safe_zip(args_flat, in_axes_flat) if i is not None)
    if size > 1:
      if any(i is not None and arg.shape[i] not in (1, size)
             for arg, i in safe_zip(args_flat, in_axes_flat)):
        raise ValueError("broadcasting_vmap: mismatched input shapes")
      args_flat, in_axes_flat = zip(*(
          (arg, None) if i is None else (lax.squeeze(arg, (i,)), None) if arg.shape[i] == 1 else (arg, i)
          for arg, i in zip(args_flat, in_axes_flat)
      ))
    new_args = tree_util.tree_unflatten(in_tree, args_flat)
    new_in_axes = tree_util.tree_unflatten(in_tree, in_axes_flat)
    return vmap(fun, in_axes=new_in_axes, out_axes=out_axes)(*new_args)
  return batched_fun

@jax.jit
def _csr_to_coo(indices: Array, indptr: Array) -> tuple[Array, Array]:
  """Given CSR (indices, indptr) return COO (row, col)"""
  return jnp.cumsum(jnp.zeros_like(indices).at[indptr].add(1)) - 1, indices

def _csr_extract(indices: Array, indptr: Array, mat: Array) -> Array:
  """Extract values of dense matrix mat at given CSR indices."""
  row, col = _csr_to_coo(indices, indptr)
  return _coo_extract(row, col, mat)

def _coo_extract(row: Array, col: Array, mat: Array) -> Array:
  """Extract values of dense matrix mat at given COO indices."""
  return mat[row, col]

def _count_stored_elements_per_batch(mat: Array, n_batch: int = 0, n_dense: int = 0) -> Array:
  """Return per-batch number of stored elements (nse) of a dense matrix."""
  mat = jnp.asarray(mat)
  mask = (mat != 0)
  if n_dense > 0:
    mask = mask.any(tuple(-(i + 1) for i in range(n_dense)))
  mask = mask.sum(tuple(range(n_batch, mask.ndim)))
  return mask

def _count_stored_elements(mat: Array, n_batch: int = 0, n_dense: int = 0) -> Array:
  """Return the number of stored elements (nse) of the given dense matrix."""
  return _count_stored_elements_per_batch(mat, n_batch, n_dense).max(initial=0)

def _dot_general_validated_shape(
    lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...],
    dimension_numbers: DotDimensionNumbers) -> tuple[int, ...]:
  """Validate the inputs and return the output shape."""
  lhs = core.ShapedArray(lhs_shape, np.float32)
  rhs = core.ShapedArray(rhs_shape, np.float32)
  return _dot_general_shape_rule(
    lhs, rhs, dimension_numbers=dimension_numbers,
    precision=None, preferred_element_type=None, out_sharding=None)
