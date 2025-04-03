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
"""Sparse test utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
import functools
import itertools
import math
from typing import Any, NamedTuple

import jax
from jax import lax
from jax import tree_util
from jax._src import test_util as jtu
from jax._src.lax.lax import DotDimensionNumbers
from jax._src.typing import DTypeLike
from jax.experimental import sparse
import jax.numpy as jnp
from jax.util import safe_zip, split_list
import numpy as np

MATMUL_TOL = {
    np.float32: 1e-5,
    np.float64: 1e-10,
    np.complex64: 1e-5,
    np.complex128: 1e-10,
}


def is_sparse(x):
  return isinstance(x, sparse.JAXSparse)


class BatchedDotGeneralProperties(NamedTuple):
  lhs_shape: tuple[int, ...]
  rhs_shape: tuple[int, ...]
  n_batch: int
  n_dense: int
  dimension_numbers: DotDimensionNumbers


class SparseLayout(NamedTuple):
  n_batch: int
  n_dense: int
  n_sparse: int


class SparseTestCase(jtu.JaxTestCase):
  def assertSparseArraysEquivalent(self, x, y, *, check_dtypes=True, atol=None,
                                   rtol=None, canonicalize_dtypes=True, err_msg=''):
    x_bufs, x_tree = tree_util.tree_flatten(x)
    y_bufs, y_tree = tree_util.tree_flatten(y)

    self.assertEqual(x_tree, y_tree)
    self.assertAllClose(x_bufs, y_bufs, check_dtypes=check_dtypes, atol=atol, rtol=rtol,
                        canonicalize_dtypes=canonicalize_dtypes, err_msg=err_msg)

  def _CheckAgainstDense(self, dense_op, sparse_op, args_maker, check_jit=True,
                         check_dtypes=True, tol=None, atol=None, rtol=None,
                         canonicalize_dtypes=True):
    """Check an operation against a dense equivalent"""
    sparse_args = args_maker()
    dense_args = tree_util.tree_map(sparse.todense, sparse_args, is_leaf=is_sparse)
    expected = dense_op(*dense_args)

    sparse_ans = sparse_op(*sparse_args)
    actual = tree_util.tree_map(sparse.todense, sparse_ans, is_leaf=is_sparse)

    self.assertAllClose(expected, actual, check_dtypes=check_dtypes,
                        atol=atol or tol, rtol=rtol or tol,
                        canonicalize_dtypes=canonicalize_dtypes)
    if check_jit:
      sparse_ans_jit = jax.jit(sparse_op)(*sparse_args)
      self.assertSparseArraysEquivalent(sparse_ans, sparse_ans_jit,
                                        atol=atol or tol, rtol=rtol or tol)

  def _CheckGradsSparse(self, dense_fun, sparse_fun, args_maker, *,
                        argnums=None, modes=('fwd', 'rev'), atol=None, rtol=None):
    assert all(mode in ['fwd', 'rev'] for mode in modes)

    args = args_maker()
    args_flat, tree = tree_util.tree_flatten(args)
    num_bufs = [len(tree_util.tree_flatten(arg)[0]) for arg in args]
    argnums_flat = np.cumsum([0, *num_bufs[:-1]]).tolist()
    if argnums is not None:
      argnums_flat = [argnums_flat[n] for n in argnums]

    def dense_fun_flat(*args_flat):
      args = tree_util.tree_unflatten(tree, args_flat)
      args_dense = tree_util.tree_map(sparse.todense, args, is_leaf=is_sparse)
      return dense_fun(*args_dense)

    def sparse_fun_flat(*args_flat):
      out = sparse_fun(*tree_util.tree_unflatten(tree, args_flat))
      return tree_util.tree_map(sparse.todense, out, is_leaf=is_sparse)

    if 'rev' in modes:
      result_de = jax.jacrev(dense_fun_flat, argnums=argnums_flat)(*args_flat)
      result_sp = jax.jacrev(sparse_fun_flat, argnums=argnums_flat)(*args_flat)
      self.assertAllClose(result_de, result_sp, atol=atol, rtol=rtol)

    if 'fwd' in modes:
      result_de = jax.jacfwd(dense_fun_flat, argnums=argnums_flat)(*args_flat)
      result_sp = jax.jacfwd(sparse_fun_flat, argnums=argnums_flat)(*args_flat)
      self.assertAllClose(result_de, result_sp, atol=atol, rtol=rtol)

  def _random_bdims(self, *args):
    rng = self.rng()
    return [rng.randint(0, arg + 1) for arg in args]

  def _CheckBatchingSparse(self, dense_fun, sparse_fun, args_maker, *, batch_size=3, bdims=None,
                           check_jit=False, check_dtypes=True, tol=None, atol=None, rtol=None,
                           canonicalize_dtypes=True):
    if bdims is None:
      bdims = self._random_bdims(*(arg.n_batch if is_sparse(arg) else arg.ndim
                                   for arg in args_maker()))
    def concat(args, bdim):
      return sparse.sparsify(functools.partial(lax.concatenate, dimension=bdim))(args)
    def expand(arg, bdim):
      return sparse.sparsify(functools.partial(lax.expand_dims, dimensions=[bdim]))(arg)
    def batched_args_maker():
      args = list(zip(*(args_maker() for _ in range(batch_size))))
      return [arg[0] if bdim is None else concat([expand(x, bdim) for x in arg], bdim)
              for arg, bdim in safe_zip(args, bdims)]
    self._CheckAgainstDense(jax.vmap(dense_fun, bdims), jax.vmap(sparse_fun, bdims), batched_args_maker,
                            check_dtypes=check_dtypes, tol=tol, atol=atol, rtol=rtol, check_jit=check_jit,
                            canonicalize_dtypes=canonicalize_dtypes)

def _rand_sparse(shape: Sequence[int], dtype: DTypeLike, *,
                 rng: np.random.RandomState, rand_method: Callable[..., Any],
                 nse: int | float, n_batch: int, n_dense: int,
                 sparse_format: str) -> sparse.BCOO | sparse.BCSR:
  if sparse_format not in ['bcoo', 'bcsr']:
    raise ValueError(f"Sparse format {sparse_format} not supported.")

  n_sparse = len(shape) - n_batch - n_dense

  if n_sparse < 0 or n_batch < 0 or n_dense < 0:
    raise ValueError(f"Invalid parameters: {shape=} {n_batch=} {n_sparse=}")

  if sparse_format == 'bcsr' and n_sparse != 2:
    raise ValueError("bcsr array must have 2 sparse dimensions; "
                     f"{n_sparse} is given.")

  batch_shape, sparse_shape, dense_shape = split_list(shape,
                                                      [n_batch, n_sparse])
  if 0 <= nse < 1:
    nse = int(np.ceil(nse * np.prod(sparse_shape)))
  nse_int = int(nse)
  data_rng = rand_method(rng)
  data_shape = (*batch_shape, nse_int, *dense_shape)
  data = jnp.array(data_rng(data_shape, dtype))

  int32 = np.dtype('int32')
  if sparse_format == 'bcoo':
    index_shape = (*batch_shape, nse_int, n_sparse)
    indices = jnp.array(
      rng.randint(0, sparse_shape, size=index_shape, dtype=int32))
    return sparse.BCOO((data, indices), shape=shape)
  else:
    index_shape = (*batch_shape, nse_int)
    indptr_shape = (*batch_shape, sparse_shape[0] + 1)
    indices = jnp.array(
      rng.randint(0, sparse_shape[1], size=index_shape, dtype=int32))
    indptr = jnp.sort(
      rng.randint(0, nse_int + 1, size=indptr_shape, dtype=int32), axis=-1)
    indptr = indptr.at[..., 0].set(0)
    return sparse.BCSR((data, indices, indptr), shape=shape)

def rand_bcoo(rng: np.random.RandomState,
              rand_method: Callable[..., Any]=jtu.rand_default,
              nse: int | float=0.5, n_batch: int=0, n_dense: int=0):
  """Generates a random BCOO array."""
  return functools.partial(_rand_sparse, rng=rng, rand_method=rand_method,
                           nse=nse, n_batch=n_batch, n_dense=n_dense,
                           sparse_format='bcoo')

def rand_bcsr(rng: np.random.RandomState,
              rand_method: Callable[..., Any]=jtu.rand_default,
              nse: int | float=0.5, n_batch: int=0, n_dense: int=0):
  """Generates a random BCSR array."""
  return functools.partial(_rand_sparse, rng=rng, rand_method=rand_method,
                           nse=nse, n_batch=n_batch, n_dense=n_dense,
                           sparse_format='bcsr')


def iter_subsets(s: Sequence) -> Iterable[tuple]:
  """Return an iterator over all subsets of a sequence s"""
  return itertools.chain.from_iterable(
      itertools.combinations(s, n) for n in range(len(s) + 1)
  )


def iter_sparse_layouts(
    shape: Sequence[int], min_n_batch=0
) -> Iterator[SparseLayout]:
  for n_batch in range(min_n_batch, len(shape) + 1):
    for n_dense in range(len(shape) + 1 - n_batch):
      n_sparse = len(shape) - n_batch - n_dense
      yield SparseLayout(n_batch=n_batch, n_sparse=n_sparse, n_dense=n_dense)


def iter_bcsr_layouts(
    shape: Sequence[int], min_n_batch=0
) -> Iterator[SparseLayout]:
  n_sparse = 2
  for n_batch in range(min_n_batch, len(shape) - 1):
    n_dense = len(shape) - n_sparse - n_batch
    yield SparseLayout(n_batch=n_batch, n_sparse=n_sparse, n_dense=n_dense)


def rand_sparse(rng, nse=0.5, post=lambda x: x, rand_method=jtu.rand_default):
  def _rand_sparse(shape, dtype, nse=nse):
    rand = rand_method(rng)
    size = math.prod(shape)
    if 0 <= nse < 1:
      nse = nse * size
    nse = min(size, int(nse))
    M = rand(shape, dtype)
    indices = rng.choice(size, size - nse, replace=False)
    M.flat[indices] = 0
    return post(M)

  return _rand_sparse
