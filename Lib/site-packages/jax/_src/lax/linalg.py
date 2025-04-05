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

from __future__ import annotations

from collections.abc import Callable
import enum
from functools import partial
import math
from typing import Any, Literal, TypeVar, overload

import numpy as np

from jax import lax

from jax._src import ad_util
from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import util
from jax._src.core import (
    Primitive, ShapedArray, is_constant_dim, is_constant_shape)
from jax._src import ffi
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lax import control_flow
from jax._src.lax import eigh as lax_eigh
from jax._src.lax import lax as lax_internal
from jax._src.lax import svd as lax_svd
from jax._src.lax.lax import (
    standard_primitive, standard_unop, naryop_dtype_rule, _float, _complex,
    _input_dtype)
from jax._src.lib import gpu_solver
from jax._src.lib import gpu_sparse
from jax._src.lib import lapack
from jax._src.lib import version as jaxlib_version
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import chlo
from jax._src.lib.mlir.dialects import hlo
from jax._src.typing import Array, ArrayLike

# The following import is unused but needed to register the custom_call targets
# in the gpu_linalg module.
from jax._src.lib import gpu_linalg  # noqa: F401

TFun = TypeVar('TFun', bound=Callable[..., Any])

def _broadcasted_iotas(*sizes):
  ones = (1,) * (len(sizes) - 1)
  shapes = (util.tuple_insert(ones, i, s) for i, s in enumerate(sizes))
  return [lax.broadcasted_iota('int32', shape, i) for i, shape in enumerate(shapes)]

def _tril(m: Array, k:int = 0) -> Array:
  *_, N, M = m.shape
  mask = lax_internal._tri(bool, (N, M), k)
  return lax.select(lax.broadcast(mask, m.shape[:-2]), m, lax.zeros_like_array(m))

def _triu(m: Array, k:int = 0) -> Array:
  *_, N, M = m.shape
  mask = lax_internal._tri(bool, (N, M), k - 1)
  return lax.select(lax.broadcast(mask, m.shape[:-2]), lax.zeros_like_array(m), m)

def _construct_diagonal(s: Array) -> Array:
  """Construct a (batched) diagonal matrix"""
  i = lax.iota('int32', s.shape[-1])
  return lax.full((*s.shape, s.shape[-1]), 0, s.dtype).at[..., i, i].set(s)

def _extract_diagonal(s: Array) -> Array:
  """Extract the diagonal from a batched matrix"""
  i = lax.iota('int32', min(s.shape[-2], s.shape[-1]))
  return s[..., i, i]

def _broadcast_to(x: Array, shape: tuple[int, ...]) -> Array:
  assert x.ndim <= len(shape)
  return lax.broadcast_in_dim(x, shape, range(len(shape) - x.ndim, len(shape)))

# traceables

def cholesky(x: Array, *, symmetrize_input: bool = True) -> Array:
  """Cholesky decomposition.

  Computes the Cholesky decomposition

  .. math::
    A = L . L^H

  of square matrices, :math:`A`, such that :math:`L`
  is lower triangular. The matrices of :math:`A` must be positive-definite and
  either Hermitian, if complex, or symmetric, if real.

  Args:
    x: A batch of square Hermitian (symmetric if real) positive-definite
      matrices with shape ``[..., n, n]``.
    symmetrize_input: If ``True``, the matrix is symmetrized before Cholesky
      decomposition by computing :math:`\\frac{1}{2}(x + x^H)`. If ``False``,
      only the lower triangle of ``x`` is used; the upper triangle is ignored
      and not accessed.

  Returns:
    The Cholesky decomposition as a matrix with the same dtype as ``x`` and
    shape ``[..., n, n]``. If Cholesky decomposition fails, returns a matrix
    full of NaNs. The behavior on failure may change in the future.
  """
  if symmetrize_input:
    x = symmetrize(x)
  return _tril(cholesky_p.bind(x))


def eig(x: ArrayLike, *, compute_left_eigenvectors: bool = True,
        compute_right_eigenvectors: bool = True,
        use_magma: bool | None = None) -> list[Array]:
  """Eigendecomposition of a general matrix.

  Nonsymmetric eigendecomposition is only implemented on CPU and GPU. On GPU,
  the default implementation calls LAPACK directly on the host CPU, but an
  experimental GPU implementation using `MAGMA <https://icl.utk.edu/magma/>`_
  is also available. The MAGMA implementation is typically slower than the
  equivalent LAPACK implementation for small matrices (less than about 2048),
  but it may perform better for larger matrices.

  To enable the MAGMA implementation, you must install MAGMA yourself (there
  are Debian and conda-forge packages, or you can build from source). Then set
  the ``use_magma`` argument to ``True``, or set the ``jax_use_magma``
  configuration variable to ``"on"`` or ``"auto"``:

  .. code-block:: python

      jax.config.update('jax_use_magma', 'on')

  JAX will try to ``dlopen`` the installed MAGMA shared library, raising an
  error if it is not found. To explicitly specify the path to the MAGMA
  library, set the environment variable `JAX_GPU_MAGMA_PATH` to the full
  installation path.

  If ``jax_use_magma`` is set to ``"auto"``, the MAGMA implementation will
  be used if the library can be found, and the input matrix is sufficiently
  large (>= 2048x2048).

  Args:
    x: A batch of square matrices with shape ``[..., n, n]``.
    compute_left_eigenvectors: If true, the left eigenvectors will be computed.
    compute_right_eigenvectors: If true, the right eigenvectors will be
      computed.
    use_magma: Locally override the ``jax_use_magma`` flag. If ``True``, the
      eigendecomposition is computed using MAGMA. If ``False``, the computation
      is done using LAPACK on to the host CPU. If ``None`` (default), the
      behavior is controlled by the ``jax_use_magma`` flag. This argument
      is only used on GPU.

  Returns:
    The eigendecomposition of ``x``, which is a tuple of the form
    ``(w, vl, vr)`` where ``w`` are the eigenvalues, ``vl`` are the left
    eigenvectors, and ``vr`` are the right eigenvectors. ``vl`` and ``vr`` are
    optional and will only be included if ``compute_left_eigenvectors`` or
    ``compute_right_eigenvectors`` respectively are ``True``.

    If the eigendecomposition fails, then arrays full of NaNs will be returned
    for that batch element.
  """
  return eig_p.bind(x, compute_left_eigenvectors=compute_left_eigenvectors,
                    compute_right_eigenvectors=compute_right_eigenvectors,
                    use_magma=use_magma)


def eigh(
    x: Array,
    *,
    lower: bool = True,
    symmetrize_input: bool = True,
    sort_eigenvalues: bool = True,
    subset_by_index: tuple[int, int] | None = None,
) -> tuple[Array, Array]:
  r"""Eigendecomposition of a Hermitian matrix.

  Computes the eigenvectors and eigenvalues of a complex Hermitian or real
  symmetric square matrix.

  Args:
    x: A batch of square complex Hermitian or real symmetric matrices with shape
      ``[..., n, n]``.
    lower: If ``symmetrize_input`` is ``False``, describes which triangle of the
      input matrix to use. If ``symmetrize_input`` is ``False``, only the
      triangle given by ``lower`` is accessed; the other triangle is ignored and
      not accessed.
    symmetrize_input: If ``True``, the matrix is symmetrized before the
      eigendecomposition by computing :math:`\frac{1}{2}(x + x^H)`.
    sort_eigenvalues: If ``True``, the eigenvalues will be sorted in ascending
      order. If ``False`` the eigenvalues are returned in an
      implementation-defined order.
     subset_by_index: Optional 2-tuple [start, end] indicating the range of
       indices of eigenvalues to compute. For example, is ``range_select`` =
       [n-2,n], then ``eigh`` computes the two largest eigenvalues and their
       eigenvectors.

  Returns:
    A tuple ``(v, w)``.

    ``v`` is an array with the same dtype as ``x`` such that ``v[..., :, i]`` is
    the normalized eigenvector corresponding to eigenvalue ``w[..., i]``.

    ``w`` is an array with the same dtype as ``x`` (or its real counterpart if
    complex) with shape ``[..., d]`` containing the eigenvalues of ``x`` in
    ascending order(each repeated according to its multiplicity).
    If ``subset_by_index`` is ``None`` then ``d`` is equal to ``n``. Otherwise
    ``d`` is equal to ``subset_by_index[1] - subset_by_index[0]``.
  """
  if symmetrize_input:
    x = symmetrize(x)
  v, w = eigh_p.bind(
      x,
      lower=lower,
      sort_eigenvalues=sort_eigenvalues,
      subset_by_index=subset_by_index,
  )
  return v, w


def cholesky_update(r_matrix: ArrayLike, w_vector: ArrayLike) -> Array:
  """Given a Cholesky decomposition A = R.T @ R and a vector w,
  computes the Cholesky decomposition of A + w @ w.T in O(N^2) time.

  Args:
    r_matrix: An upper-triangular matrix (R) such that A = R.T @ R.
    w_vector: A vector (w) for rank-1 update.

  Returns:
    A new R' matrix being the Cholesky decomposition of A + w @ w.T.
  """
  return cholesky_update_p.bind(r_matrix, w_vector)


def symmetric_product(
    a_matrix: ArrayLike, c_matrix: ArrayLike,
    alpha: float = 1., beta: float = 0.,
    symmetrize_output=False):
  """Computes C = alpha * A @ A.T + beta * C (where C is symmetric)."""
  result = symmetric_product_p.bind(a_matrix, c_matrix, alpha=alpha, beta=beta)
  if symmetrize_output:
    upper_half = lax.transpose(
        _tril(result, k=-1),
        (*range(result.ndim - 2), result.ndim - 1, result.ndim - 2))
    result = _tril(result, k=0) + upper_half
  return result


def lu_pivots_to_permutation(pivots: ArrayLike, permutation_size: int) -> Array:
  """Converts the pivots (row swaps) returned by LU to a permutation.

  We build a permutation rather than applying `pivots` directly to the rows
  of a matrix because lax loops aren't differentiable.

  Args:
    pivots: an int32 array of shape (..., k) of row swaps to perform
    permutation_size: the size of the output permutation. Has to be >= k.

  Returns:
    An int32 array of shape (..., permutation_size).
  """
  permutation = lu_pivots_to_permutation_p.bind(
      pivots, permutation_size=permutation_size)
  return permutation


def lu(x: ArrayLike) -> tuple[Array, Array, Array]:
  """LU decomposition with partial pivoting.

  Computes the matrix decomposition:

  .. math::
    P.A = L.U

  where :math:`P` is a permutation of the rows of :math:`A`, :math:`L` is a
  lower-triangular matrix with unit-diagonal elements, and :math:`U` is an
  upper-triangular matrix.

  Args:
    x: A batch of matrices with shape ``[..., m, n]``.

  Returns:
    A tuple ``(lu, pivots, permutation)``.

    ``lu`` is a batch of matrices with the same shape and dtype as ``x``
    containing the :math:`L` matrix in its lower triangle and the :math:`U`
    matrix in its upper triangle. The (unit) diagonal elements of :math:`L` are
    not represented explicitly.

    ``pivots`` is an int32 array with shape ``[..., min(m, n)]`` representing a
    sequence of row swaps that should be performed on :math:`A`.

    ``permutation`` is an alternative representation of the sequence of row
    swaps as a permutation, represented as an int32 array with shape
    ``[..., m]``.
  """
  lu, pivots, permutation = lu_p.bind(x)
  return lu, pivots, permutation


@overload
def qr(x: ArrayLike, *, pivoting: Literal[False], full_matrices: bool = True,
      ) -> tuple[Array, Array]:
  ...

@overload
def qr(x: ArrayLike, *, pivoting: Literal[True], full_matrices: bool = True,
      ) -> tuple[Array, Array, Array]:
  ...

@overload
def qr(x: ArrayLike, *, pivoting: bool = False, full_matrices: bool = True,
      ) -> tuple[Array, Array] | tuple[Array, Array, Array]:
  ...

def qr(x: ArrayLike, *, pivoting: bool = False, full_matrices: bool = True,
      ) -> tuple[Array, Array] | tuple[Array, Array, Array]:
  """QR decomposition.

  Computes the QR decomposition

  .. math::
    A = Q . R

  of matrices :math:`A`, such that :math:`Q` is a unitary (orthogonal) matrix,
  and :math:`R` is an upper-triangular matrix.

  Args:
    x: A batch of matrices with shape ``[..., m, n]``.
    pivoting: Allows the QR decomposition to be rank-revealing. If ``True``,
      compute the column pivoted decomposition ``A[:, P] = Q @ R``, where ``P``
      is chosen such that the diagonal of ``R`` is non-increasing. Currently
      supported on CPU backends only.
    full_matrices: Determines if full or reduced matrices are returned; see
      below.

  Returns:
    A pair of arrays ``(q, r)``, if ``pivoting=False``, otherwise ``(q, r, p)``.

    Array ``q`` is a unitary (orthogonal) matrix,
    with shape ``[..., m, m]`` if ``full_matrices=True``, or
    ``[..., m, min(m, n)]`` if ``full_matrices=False``.

    Array ``r`` is an upper-triangular matrix with shape ``[..., m, n]`` if
    ``full_matrices=True``, or ``[..., min(m, n), n]`` if
    ``full_matrices=False``.

    Array ``p`` is an index vector with shape [..., n]
  """
  q, r, *p = qr_p.bind(x, pivoting=pivoting, full_matrices=full_matrices)
  if pivoting:
    return q, r, p[0]
  return q, r


class SvdAlgorithm(enum.Enum):
  """Enum for SVD algorithm."""
  DEFAULT = "default"
  QR = "QR"
  JACOBI = "Jacobi"


@overload
def svd(
    x: ArrayLike,
    *,
    full_matrices: bool = True,
    compute_uv: Literal[True],
    subset_by_index: tuple[int, int] | None = None,
    algorithm: SvdAlgorithm | None = None,
) -> tuple[Array, Array, Array]:
  ...


@overload
def svd(
    x: ArrayLike,
    *,
    full_matrices: bool = True,
    compute_uv: Literal[False],
    subset_by_index: tuple[int, int] | None = None,
    algorithm: SvdAlgorithm | None = None,
) -> Array:
  ...


@overload
def svd(
    x: ArrayLike,
    *,
    full_matrices: bool = True,
    compute_uv: bool = True,
    subset_by_index: tuple[int, int] | None = None,
    algorithm: SvdAlgorithm | None = None,
) -> Array | tuple[Array, Array, Array]:
  ...


# TODO: Add `max_qdwh_iterations` to the function signature for TPU SVD.
def svd(
    x: ArrayLike,
    *,
    full_matrices: bool = True,
    compute_uv: bool = True,
    subset_by_index: tuple[int, int] | None = None,
    algorithm: SvdAlgorithm | None = None,
) -> Array | tuple[Array, Array, Array]:
  """Singular value decomposition.

  Returns the singular values if compute_uv is False, otherwise returns a triple
  containing the left singular vectors, the singular values and the adjoint of
  the right singular vectors.
  """
  result = svd_p.bind(
      x,
      full_matrices=full_matrices,
      compute_uv=compute_uv,
      subset_by_index=subset_by_index,
      algorithm=algorithm,
  )
  if compute_uv:
    s, u, v = result
    return u, s, v
  else:
    s, = result
    return s


def triangular_solve(a: ArrayLike, b: ArrayLike, *,
                     left_side: bool = False, lower: bool = False,
                     transpose_a: bool = False, conjugate_a: bool = False,
                     unit_diagonal: bool = False) -> Array:
  r"""Triangular solve.

  Solves either the matrix equation

  .. math::
    \mathit{op}(A) . X = B

  if ``left_side`` is ``True`` or

  .. math::
    X . \mathit{op}(A) = B

  if ``left_side`` is ``False``.

  ``A`` must be a lower or upper triangular square matrix, and where
  :math:`\mathit{op}(A)` may either transpose :math:`A` if ``transpose_a``
  is ``True`` and/or take its complex conjugate if ``conjugate_a`` is ``True``.

  Args:
    a: A batch of matrices with shape ``[..., m, m]``.
    b: A batch of matrices with shape ``[..., m, n]`` if ``left_side`` is
      ``True`` or shape ``[..., n, m]`` otherwise.
    left_side: describes which of the two matrix equations to solve; see above.
    lower: describes which triangle of ``a`` should be used. The other triangle
      is ignored.
    transpose_a: if ``True``, the value of ``a`` is transposed.
    conjugate_a: if ``True``, the complex conjugate of ``a`` is used in the
      solve. Has no effect if ``a`` is real.
    unit_diagonal: if ``True``, the diagonal of ``a`` is assumed to be unit
      (all 1s) and not accessed.

  Returns:
    A batch of matrices the same shape and dtype as ``b``.
  """
  conjugate_a = conjugate_a and dtypes.issubdtype(lax.dtype(a), np.complexfloating)
  singleton = np.ndim(b) == np.ndim(a) - 1
  if singleton:
    b = lax.expand_dims(b, (-1 if left_side else -2,))
  out = triangular_solve_p.bind(
      a, b, left_side=left_side, lower=lower, transpose_a=transpose_a,
      conjugate_a=conjugate_a, unit_diagonal=unit_diagonal)
  if singleton:
    out = out[..., 0] if left_side else out[..., 0, :]
  return out


# utilities
def _broadcasted_matvec(a: Array, b: Array) -> Array:
  # This is a broadcasted dot_general with signature (...,n,m),(...,m)->(...,n)
  assert a.ndim >= 2
  assert b.ndim >= 1
  batch_shape = lax.broadcast_shapes(a.shape[:-2], b.shape[:-1])
  n_batch = len(batch_shape)
  a = _broadcast_to(a, (*batch_shape, *a.shape[-2:]))
  b = _broadcast_to(b, (*batch_shape, b.shape[-1]))

  dimension_numbers = (([a.ndim - 1], [b.ndim - 1]), (list(range(n_batch)), list(range(n_batch))))
  return lax.dot_general(a, b, dimension_numbers=dimension_numbers, precision=lax.Precision.HIGHEST)

def _check_solve_shapes(a: Array, b: Array):
  if not (a.ndim >= 2 and b.ndim in [a.ndim, a.ndim - 1] and
          a.shape[-1] == a.shape[-2] == b.shape[a.ndim - 2]):
    raise ValueError(
        "The arguments to solve must have shapes a=[..., m, m] and "
        f"b=[..., m, k] or b=[..., m]; got a={a.shape} and b={b.shape}")

def _solve(a: Array, b: Array) -> Array:
  _check_solve_shapes(a, b)

  # Broadcast leading dimensions of b to the shape of a, as is required by
  # custom_linear_solve.
  out_shape = tuple(d_a if d_b == 1 else d_b
                    for d_a, d_b in zip(a.shape[:-1] + (1,), b.shape))
  b = lax.broadcast_in_dim(b, out_shape, range(b.ndim))

  # With custom_linear_solve, we can reuse the same factorization when
  # computing sensitivities. This is considerably faster.
  lu_, _, permutation = lu(lax.stop_gradient(a))
  custom_solve = partial(
      lax.custom_linear_solve,
      lambda x: _broadcasted_matvec(a, x),
      solve=lambda _, x: lu_solve(lu_, permutation, x, trans=0),
      transpose_solve=lambda _, x: lu_solve(lu_, permutation, x, trans=1))
  if a.ndim == b.ndim + 1:
    # b.shape == [..., m]
    return custom_solve(b)
  else:
    # b.shape == [..., m, k]
    return api.vmap(custom_solve, b.ndim - 1, max(a.ndim, b.ndim) - 1)(b)

def _T(x: Array) -> Array:
  return lax.transpose(x, (*range(x.ndim - 2), x.ndim - 1, x.ndim - 2))
def _H(x: Array) -> Array:
  return _T(x).conj()
def symmetrize(x: Array) -> Array: return (x + _H(x)) / 2

# primitives

_cpu_lapack_types = {np.dtype(np.float32), np.dtype(np.float64),
                     np.dtype(np.complex64), np.dtype(np.complex128)}

# Cholesky decomposition

def _cholesky_jvp_rule(primals, tangents):
  x, = primals
  sigma_dot, = tangents
  L = _tril(cholesky_p.bind(x))

  # Forward-mode rule from https://arxiv.org/pdf/1602.07527.pdf
  def phi(X):
    l = _tril(X)
    return l / lax.expand_dims(
        lax_internal._const(X, 1) + lax_internal._eye(X.dtype, (X.shape[-1], X.shape[-1])),
        range(l.ndim - 2))

  tmp = triangular_solve(L, sigma_dot, left_side=False, transpose_a=True,
                         conjugate_a=True, lower=True)
  L_dot = lax.batch_matmul(L, phi(triangular_solve(
      L, tmp, left_side=True, transpose_a=False, lower=True)),
      precision=lax.Precision.HIGHEST)
  return L, L_dot

def _cholesky_batching_rule(batched_args, batch_dims):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return cholesky(x), 0

cholesky_p = standard_unop(_float | _complex, 'cholesky')
ad.primitive_jvps[cholesky_p] = _cholesky_jvp_rule
batching.primitive_batchers[cholesky_p] = _cholesky_batching_rule

def _cholesky_lowering(ctx, x):
  return [hlo.cholesky(x, lower=ir.BoolAttr.get(True))]

mlir.register_lowering(cholesky_p, _cholesky_lowering)

def _cholesky_cpu_lowering(ctx, operand):
  operand_aval, = ctx.avals_in
  out_aval, = ctx.avals_out
  batch_dims = operand_aval.shape[:-2]
  op_shape_vals = mlir.eval_dynamic_shape_as_ivals(ctx, operand_aval.shape)
  ctx_arg = (ctx,)
  result, info = lapack.potrf_hlo(*ctx_arg, operand_aval.dtype, operand,
                                  lower=True, a_shape_vals=op_shape_vals)

  ok = mlir.compare_hlo(
      info, mlir.full_like_aval(ctx, 0, ShapedArray(batch_dims, np.dtype(np.int32))),
      "EQ", "SIGNED")
  select_aval = ShapedArray(batch_dims + (1, 1), np.dtype(np.bool_))
  return [_broadcasting_select_hlo(
      ctx,
      mlir.broadcast_in_dim(ctx, ok,
                            select_aval,
                            broadcast_dimensions=range(len(batch_dims))),
      select_aval,
      result, out_aval, _nan_like_hlo(ctx, out_aval), out_aval)]

mlir.register_lowering(
    cholesky_p, _cholesky_cpu_lowering, platform='cpu')

# Cholesky update

def _cholesky_update_abstract_eval(r_matrix, w_vector):
  r_dtype = dtypes.canonicalize_dtype(r_matrix.dtype)
  w_dtype = dtypes.canonicalize_dtype(w_vector.dtype)
  if not (r_dtype == w_dtype and r_dtype in (np.float32, np.float64)):
    raise NotImplementedError(
        "Rank-1 Cholesky update is only implemented for float32 and float64.")
  if not (r_matrix.ndim == 2 and w_vector.ndim == 1
          and r_matrix.shape[-2] == r_matrix.shape[-1]
          and r_matrix.shape[-2] == w_vector.shape[-1]):
    raise ValueError(
        "Rank-1 update to Cholesky decomposition takes a square matrix "
        "and a vector as inputs. Got shapes {}, {} instead".format(
            r_matrix.shape, w_vector.shape))
  return ShapedArray(r_matrix.shape, r_matrix.dtype)

def _cholesky_update_gpu_lowering_rule(target_name_prefix, ctx, r_matrix, w_vector):
  rule = ffi.ffi_lowering(f"{target_name_prefix}_cholesky_update_ffi",
                          operand_output_aliases={0: 0, 1: 1})
  sub_ctx = ctx.replace(avals_out=ctx.avals_in)
  return rule(sub_ctx, r_matrix, w_vector)[:1]


def _cholesky_update_jax_fn(R, z):
  def _drotg(x, y):
    """Get coefs for Givens rotation in a numerically stable way."""
    def _drotg_nonzero(x, y):
      abs_x = abs(x)
      abs_y = abs(y)
      denominator = lax.select(abs_x > abs_y, abs_x, abs_y)
      x /= denominator
      y /= denominator
      rh = 1 / lax.sqrt(x ** 2 + y ** 2)
      return x * rh, -y * rh
    one_and_zero = (
        np.array(1., dtype=x.dtype),
        np.array(0., dtype=x.dtype),
    )
    return lax.cond(y == 0, lambda x, y: one_and_zero, _drotg_nonzero, x, y)

  def _drot(
      first_vector: Array, second_vector: Array,
      c_coef: float, s_coef: float) -> tuple[Array, Array]:
    return (
        c_coef * first_vector - s_coef * second_vector,
        c_coef * second_vector + s_coef * first_vector)
  n = z.shape[0]
  for k in range(n):
    c, s = _drotg(R[k, k], z[k])
    row_k, z = _drot(R[k, :], z, c, s)
    R = R.at[k, :].set(row_k)
  return R


cholesky_update_p = Primitive('cholesky_update')
cholesky_update_p.multiple_results = False
cholesky_update_p.def_abstract_eval(_cholesky_update_abstract_eval)
cholesky_update_p.def_impl(partial(dispatch.apply_primitive, cholesky_update_p))

mlir.register_lowering(
    cholesky_update_p, partial(_cholesky_update_gpu_lowering_rule, "cu"),
    platform='cuda')
mlir.register_lowering(
    cholesky_update_p,
    mlir.lower_fun(_cholesky_update_jax_fn, multiple_results=False))

# symmetric_update

def _symmetric_product_abstract_eval(a, c, *, alpha, beta):
  a_dtype = dtypes.canonicalize_dtype(a.dtype)
  c_dtype = dtypes.canonicalize_dtype(c.dtype)
  if not (a_dtype == c_dtype and a_dtype in (np.float32, np.float64)):
    raise NotImplementedError(
        "Symmetric update is only implemented for float32 and float64.")
  if not (a.ndim >= 2 and c.ndim >= 2
          and a.shape[-2] == c.shape[-1]
          and c.shape[-1] == c.shape[-2]):
    raise ValueError(
        "Symmetric update takes (maybe batched) matrices of matching shapes. "
        "Got shapes {}, {} instead".format(a.shape, c.shape))
  return ShapedArray(c.shape, c.dtype)


def _symmetric_product_batching_rule(batched_args, batch_dims, *, alpha, beta):
  a_tensor, c_tensor = batched_args
  a_bd, c_bd = batch_dims
  a_tensor = batching.moveaxis(a_tensor, a_bd, 0)
  c_tensor = batching.moveaxis(c_tensor, c_bd, 0)
  return (
      symmetric_product_p.bind(a_tensor, c_tensor, alpha=alpha, beta=beta), 0)

symmetric_product_p = Primitive('symmetric_update')
symmetric_product_p.multiple_results = False
symmetric_product_p.def_abstract_eval(_symmetric_product_abstract_eval)
symmetric_product_p.def_impl(
    partial(dispatch.apply_primitive, symmetric_product_p))
batching.primitive_batchers[
    symmetric_product_p] = _symmetric_product_batching_rule


def _symmetric_product_gpu_lowering(
    platform, ctx, a_tensor, c_tensor, alpha, beta):
  a_aval, c_aval = ctx.avals_in[:2]
  dtype = a_aval.dtype
  alpha_aval = beta_aval = ShapedArray((), dtype)

  alpha_array = mlir.full_like_aval(ctx, alpha, alpha_aval)
  beta_array = mlir.full_like_aval(ctx, beta, beta_aval)

  rule = ffi.ffi_lowering(f"{platform}solver_syrk_ffi",
                          operand_output_aliases={1: 0})
  ctx = ctx.replace(avals_in=[a_aval, c_aval, alpha_aval, beta_aval])
  return rule(ctx, a_tensor, c_tensor, alpha_array, beta_array, transpose=False)


def _symmetric_product_jax_fn(a, c, *, alpha, beta):
  a_T = lax.transpose(a, (*range(a.ndim - 2), a.ndim - 1, a.ndim - 2))
  return alpha * lax.batch_matmul(
      a, a_T, precision=lax.Precision.HIGHEST) + beta * c


mlir.register_lowering(
    symmetric_product_p,
    partial(_symmetric_product_gpu_lowering, 'cu'), platform='cuda')
mlir.register_lowering(
    symmetric_product_p,
    mlir.lower_fun(_symmetric_product_jax_fn, multiple_results=False))

# Asymmetric eigendecomposition

def eig_impl(operand, *, compute_left_eigenvectors, compute_right_eigenvectors,
             use_magma):
  return dispatch.apply_primitive(
      eig_p,
      operand,
      compute_left_eigenvectors=compute_left_eigenvectors,
      compute_right_eigenvectors=compute_right_eigenvectors,
      use_magma=use_magma,
  )

def eig_lower(*args, **kw):
  raise NotImplementedError(
    "Nonsymmetric eigendecomposition is only implemented on the CPU backend. "
    "If your matrix is symmetric or Hermitian, you should use eigh instead.")

def eig_abstract_eval(operand, *, compute_left_eigenvectors,
                      compute_right_eigenvectors, use_magma):
  del use_magma  # unused
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2 or operand.shape[-2] != operand.shape[-1]:
      raise ValueError("Argument to nonsymmetric eigendecomposition must have "
                       "shape [..., n, n], got shape {}".format(operand.shape))

    batch_dims = operand.shape[:-2]
    n = operand.shape[-1]
    dtype = np.complex64 if dtypes.finfo(operand.dtype).bits == 32 else np.complex128
    dtype = dtypes.canonicalize_dtype(dtype)
    vl = vr = operand.update(shape=batch_dims + (n, n), dtype=dtype)
    w = operand.update(shape=batch_dims + (n,), dtype=dtype)
  else:
    raise NotImplementedError

  output = [w]
  if compute_left_eigenvectors:
    output.append(vl)
  if compute_right_eigenvectors:
    output.append(vr)

  return tuple(output)

def _eig_cpu_lowering(ctx, operand, *, compute_left_eigenvectors,
                      compute_right_eigenvectors, use_magma):
  del use_magma  # unused
  operand_aval, = ctx.avals_in
  out_aval = ctx.avals_out[0]
  batch_dims = operand_aval.shape[:-2]
  op_shape_vals = mlir.eval_dynamic_shape_as_ivals(ctx, operand_aval.shape)
  w, vl, vr, info = lapack.geev_hlo(ctx, operand_aval.dtype, operand,
                                    input_shape_vals=op_shape_vals,
                                    jobvl=compute_left_eigenvectors,
                                    jobvr=compute_right_eigenvectors)

  ok = mlir.compare_hlo(
      info, mlir.full_like_aval(ctx, 0, ShapedArray(batch_dims, np.dtype(np.int32))),
      "EQ", "SIGNED")
  select_w_aval = ShapedArray(batch_dims + (1,), np.dtype(np.bool_))
  w = _broadcasting_select_hlo(
      ctx,
      mlir.broadcast_in_dim(ctx, ok, select_w_aval,
                            broadcast_dimensions=range(len(batch_dims))),
      select_w_aval,
      w, out_aval, _nan_like_hlo(ctx, out_aval), out_aval)
  output = [w]

  if compute_left_eigenvectors:
    aval = ctx.avals_out[len(output)]
    select_vl_aval = ShapedArray(batch_dims + (1, 1), np.dtype(np.bool_))
    vl = _broadcasting_select_hlo(
        ctx,
        mlir.broadcast_in_dim(ctx, ok, select_vl_aval,
                              broadcast_dimensions=range(len(batch_dims))),
        select_vl_aval,
        vl, aval, _nan_like_hlo(ctx, aval), aval)
    output.append(vl)

  if compute_right_eigenvectors:
    aval = ctx.avals_out[len(output)]
    select_vr_aval = ShapedArray(batch_dims + (1, 1), np.dtype(np.bool_))
    vr = _broadcasting_select_hlo(
        ctx,
        mlir.broadcast_in_dim(ctx, ok, select_vr_aval,
                              broadcast_dimensions=range(len(batch_dims))),
        select_vr_aval,
        vr, aval, _nan_like_hlo(ctx, aval), aval)
    output.append(vr)

  return output


def _eig_gpu_impl(target_name_prefix, x, *, compute_left_eigenvectors,
                  compute_right_eigenvectors, use_magma):
  gpu_solver.initialize_hybrid_kernels()
  dtype = x.dtype
  is_real = dtype == np.float32 or dtype == np.float64
  if is_real:
    target_name = f"{target_name_prefix}hybrid_eig_real"
    complex_dtype = np.complex64 if dtype == np.float32 else np.complex128
  else:
    target_name = f"{target_name_prefix}hybrid_eig_comp"
    assert dtype == np.complex64 or dtype == np.complex128
    complex_dtype = dtype

  batch_dims = x.shape[:-2]
  n, m = x.shape[-2:]
  assert n == m
  num_batch_dims = len(batch_dims)

  layout = tuple(range(num_batch_dims)) + (num_batch_dims + 1, num_batch_dims)
  out_types = [
      api.ShapeDtypeStruct(batch_dims + (n,), dtype),
      api.ShapeDtypeStruct(batch_dims + (n, n), complex_dtype),
      api.ShapeDtypeStruct(batch_dims + (n, n), complex_dtype),
      api.ShapeDtypeStruct(batch_dims, np.int32),
  ]
  out_layouts = [None, layout, layout, None]
  if is_real:
    out_types = [api.ShapeDtypeStruct(batch_dims + (n,), dtype)] + out_types
    out_layouts = [None] + out_layouts

  magma = config.gpu_use_magma.value
  if use_magma is not None:
    magma = "on" if use_magma else "off"
  fun = ffi.ffi_call(target_name, out_types, input_layouts=[layout],
                     output_layouts=out_layouts)
  *w, vl, vr, info = fun(x, magma=magma, left=compute_left_eigenvectors,
                         right=compute_right_eigenvectors)
  if is_real:
    assert len(w) == 2
    w = lax.complex(*w)
  else:
    assert len(w) == 1
    w = w[0]
  ok = lax.eq(info, lax.zeros_like_array(info))
  ok = _broadcast_to(ok[..., None], w.shape)
  w = lax.select(ok, w, lax.full_like(w, np.nan + np.nan * 1j))
  ok = _broadcast_to(ok[..., None], x.shape)
  output = [w]
  if compute_left_eigenvectors:
    vl = lax.select(ok, vl, lax.full_like(vl, np.nan + np.nan * 1j))
    output.append(vl)
  if compute_right_eigenvectors:
    vr = lax.select(ok, vr, lax.full_like(vr, np.nan + np.nan * 1j))
    output.append(vr)
  return output


def _eig_gpu_lowering(target_name_prefix, ctx, operand, *,
                      compute_left_eigenvectors, compute_right_eigenvectors,
                      use_magma):
  if ctx.is_forward_compat():
    raise NotImplementedError(
        "Export of nonsymmetric eigendecomposition on GPU is not supported "
        "because of forward compatibility. The "
        "'jax_export_ignore_forward_compatibility' configuration option can be "
        "used to disable this check.")
  rule = mlir.lower_fun(partial(
      _eig_gpu_impl, target_name_prefix,
      compute_left_eigenvectors=compute_left_eigenvectors,
      compute_right_eigenvectors=compute_right_eigenvectors,
      use_magma=use_magma), multiple_results=True)
  return rule(ctx, operand)


def eig_batching_rule(batched_args, batch_dims, *, compute_left_eigenvectors,
                      compute_right_eigenvectors, use_magma):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)

  return (eig_p.bind(x, compute_left_eigenvectors=compute_left_eigenvectors,
                     compute_right_eigenvectors=compute_right_eigenvectors,
                     use_magma=use_magma),
          (0,) * (1 + compute_left_eigenvectors + compute_right_eigenvectors))

def eig_jvp_rule(primals, tangents, *, compute_left_eigenvectors,
                 compute_right_eigenvectors, use_magma):
  del use_magma  # unused
  if compute_left_eigenvectors or compute_right_eigenvectors:
    raise NotImplementedError(
        'The derivatives of eigenvectors are not implemented, only '
        'eigenvalues. See '
        'https://github.com/jax-ml/jax/issues/2748 for discussion.')
  # Formula for derivative of eigenvalues w.r.t. a is eqn 4.60 in
  # https://arxiv.org/abs/1701.00392
  a, = primals
  da, = tangents
  l, v = eig(a, compute_left_eigenvectors=False)
  return [l], [(_solve(v, da.astype(v.dtype)) * _T(v)).sum(-1)]

eig_p = Primitive('eig')
eig_p.multiple_results = True
eig_p.def_impl(eig_impl)
eig_p.def_abstract_eval(eig_abstract_eval)
mlir.register_lowering(eig_p, eig_lower)
mlir.register_lowering(eig_p, _eig_cpu_lowering, platform='cpu')
mlir.register_lowering(eig_p, partial(_eig_gpu_lowering, 'cu'),
                       platform='cuda')
mlir.register_lowering(eig_p, partial(_eig_gpu_lowering, 'hip'),
                       platform='rocm')
batching.primitive_batchers[eig_p] = eig_batching_rule
ad.primitive_jvps[eig_p] = eig_jvp_rule


# Symmetric/Hermitian eigendecomposition


def eigh_jacobi(x: ArrayLike, *, lower: bool = True,
                sort_eigenvalues: bool = True) -> tuple[Array, Array]:
  """Helper Jacobi eigendecomposition implemented by XLA.

  Used as a subroutine of QDWH-eig on TPU."""
  w, v = eigh_jacobi_p.bind(x, lower=lower, sort_eigenvalues=sort_eigenvalues)
  return w, v

def _eigh_jacobi_impl(operand, *, lower, sort_eigenvalues):
  w, v = dispatch.apply_primitive(eigh_jacobi_p, operand, lower=lower,
                                  sort_eigenvalues=sort_eigenvalues)
  return w, v

def _eigh_jacobi_abstract_eval(operand, *, lower, sort_eigenvalues):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2 or operand.shape[-2] != operand.shape[-1]:
      raise ValueError(
        "Argument to symmetric eigendecomposition must have shape [..., n, n],"
        "got shape {}".format(operand.shape))

    batch_dims = operand.shape[:-2]
    n = operand.shape[-1]
    w = operand.update(shape=batch_dims + (n,),
                       dtype=lax_internal._complex_basetype(operand.dtype))
    v = operand.update(shape=batch_dims + (n, n))
  else:
    w, v = operand, operand
  return w, v


def _eigh_jacobi_lowering_rule(ctx, operand, lower, sort_eigenvalues):
  operand_aval, = ctx.avals_in
  if operand_aval.shape[-1] == 0:
    reshape_aval = operand_aval.update(shape=operand_aval.shape[:-1])
    return [
        hlo.real(mlir.reshape(ctx, operand, reshape_aval)),
        operand,
    ]

  eigvals_type = mlir.aval_to_ir_type(ctx.avals_out[0])
  eigvecs_type = mlir.aval_to_ir_type(ctx.avals_out[1])
  result_types = [eigvecs_type, eigvals_type]

  backend_config = f"{int(lower)},{int(sort_eigenvalues)},100,1e-6"

  if any(not is_constant_shape(aval_out.shape)
         for aval_out in ctx.avals_out):
    result_shapes = [
        mlir.eval_dynamic_shape_as_tensor(ctx, aval_out.shape)
        # The custom call returns the results swapped
        for aval_out in list(reversed(ctx.avals_out))
    ]
  else:
    result_shapes = None
  op = mlir.custom_call(
      "Eigh",
      result_types=result_types,
      operands=[operand],
      backend_config=backend_config,
      api_version=1,
      result_shapes=result_shapes,
  )
  return op.results[1], op.results[0]

eigh_jacobi_p = Primitive('eigh_jacobi')
eigh_jacobi_p.multiple_results = True
eigh_jacobi_p.def_impl(_eigh_jacobi_impl)
eigh_jacobi_p.def_abstract_eval(_eigh_jacobi_abstract_eval)
mlir.register_lowering(eigh_jacobi_p, _eigh_jacobi_lowering_rule)


def _eigh_impl(operand, *, lower, sort_eigenvalues, subset_by_index):
  v, w = dispatch.apply_primitive(
      eigh_p,
      operand,
      lower=lower,
      sort_eigenvalues=sort_eigenvalues,
      subset_by_index=subset_by_index,
  )
  return v, w


def _eigh_abstract_eval(operand, *, lower, sort_eigenvalues, subset_by_index):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2 or operand.shape[-2] != operand.shape[-1]:
      raise ValueError(
        "Argument to symmetric eigendecomposition must have shape [..., n, n], "
        "got shape {}".format(operand.shape))

    batch_dims = operand.shape[:-2]
    n = operand.shape[-1]
    d = (
        n
        if subset_by_index is None
        else subset_by_index[1] - subset_by_index[0]
    )
    v = operand.update(shape=batch_dims + (n, d))
    w = operand.update(
        shape=batch_dims + (d,),
        dtype=lax_internal._complex_basetype(operand.dtype),
    )
  else:
    v, w = operand, operand
  return v, w


def _eigh_cpu_gpu_lowering(
    ctx, operand, *, lower, sort_eigenvalues, subset_by_index,
    target_name_prefix: str
):
  del sort_eigenvalues  # The CPU/GPU implementations always sort.
  operand_aval, = ctx.avals_in
  v_aval, w_aval = ctx.avals_out
  n = operand_aval.shape[-1]
  if not (subset_by_index is None or subset_by_index == (0, n)):
    raise NotImplementedError("subset_by_index not supported on CPU and GPU")
  batch_dims = operand_aval.shape[:-2]
  nb = len(batch_dims)
  layout = (nb, nb + 1) + tuple(range(nb - 1, -1, -1))
  result_layouts = [layout, tuple(range(nb, -1, -1)),
                    tuple(range(nb - 1, -1, -1))]
  if target_name_prefix == "cpu":
    dtype = operand_aval.dtype
    prefix = "he" if dtypes.issubdtype(dtype, np.complexfloating) else "sy"
    target_name = lapack.prepare_lapack_call(f"{prefix}evd_ffi",
                                             operand_aval.dtype)
    kwargs = {
      "mode": np.uint8(ord("V")),
      "uplo": np.uint8(ord("L" if lower else "U")),
    }
  else:
    target_name = f"{target_name_prefix}solver_syevd_ffi"
    kwargs = {"lower": lower, "algorithm": np.uint8(0)}

  rule = ffi.ffi_lowering(target_name, operand_layouts=[layout],
                          result_layouts=result_layouts,
                          operand_output_aliases={0: 0})
  info_aval = ShapedArray(batch_dims, np.dtype(np.int32))
  sub_ctx = ctx.replace(avals_out=[v_aval, w_aval, info_aval])
  v, w, info = rule(sub_ctx, operand, **kwargs)

  zeros = mlir.full_like_aval(ctx, 0, ShapedArray(batch_dims, np.dtype(np.int32)))
  ok = mlir.compare_hlo(info, zeros, "EQ", "SIGNED")
  select_v_aval = ShapedArray(batch_dims + (1, 1), np.dtype(np.bool_))
  v = _broadcasting_select_hlo(
      ctx,
      mlir.broadcast_in_dim(ctx, ok, select_v_aval,
                            broadcast_dimensions=range(len(batch_dims))),
      select_v_aval,
      v, v_aval, _nan_like_hlo(ctx, v_aval), v_aval)
  select_w_aval = ShapedArray(batch_dims + (1,), np.dtype(np.bool_))
  w = _broadcasting_select_hlo(
      ctx,
      mlir.broadcast_in_dim(ctx, ok, select_w_aval,
                            broadcast_dimensions=range(len(batch_dims))),
      select_w_aval,
      w, w_aval, _nan_like_hlo(ctx, w_aval), w_aval)
  return [v, w]


def _eigh_tpu_impl(x, *, lower, sort_eigenvalues, subset_by_index):
  *_, m, n = x.shape
  assert m == n, (m, n)

  termination_size = 256
  if not is_constant_dim(m):
    # TODO: maybe we can relax the check below for shape polymorphism?
    raise NotImplementedError(
        "Shape polymorphism for native lowering for eigh is implemented "
        f"only for the batch dimensions: {x.shape}")
  if m <= termination_size and (
      subset_by_index is None or subset_by_index == (0, n)
  ):
    eig_vals, eig_vecs = eigh_jacobi(x, lower=lower,
                                     sort_eigenvalues=sort_eigenvalues)
    return eig_vecs, eig_vals

  def eigh_qdwh(x):
    if len(x.shape) > 2:
      return control_flow.map(eigh_qdwh, x)

    # We should only look at elements from the lower/upper triangle. Reflects
    # that triangle into the other triangle to form a Hermitian matrix.
    if lower:
      mask = lax_internal._tri(bool, (n, n), 0)
    else:
      mask = lax.bitwise_not(lax_internal._tri(bool, (n, n), -1))
    if dtypes.issubdtype(x.dtype, np.complexfloating):
      re = lax.select(mask, lax.real(x), _T(lax.real(x)))
      if lower:
        im_mask = lax_internal._tri(bool, (n, n), -1)
      else:
        im_mask = lax.bitwise_not(lax_internal._tri(bool, (n, n), 0))
      im = lax.imag(x)
      im = lax.select(im_mask, im, lax.full_like(im, 0))
      im = lax.select(mask, im, -_T(im))
      x = lax.complex(re, im)
    else:
      x = lax.select(mask, x, _T(x))

    return lax_eigh.eigh(
        x,
        sort_eigenvalues=sort_eigenvalues,
        termination_size=termination_size,
        subset_by_index=subset_by_index,
    )

  eig_vals, eig_vecs = eigh_qdwh(x)
  return eig_vecs, eig_vals


def _eigh_jvp_rule(
    primals, tangents, *, lower, sort_eigenvalues, subset_by_index
):
  (a,) = primals
  n = a.shape[-1]
  if not (subset_by_index is None or subset_by_index == (0, n)):
    raise NotImplementedError(
        "Derivatives not defined for partial eigen decomposition."
    )
  # Derivative for eigh in the simplest case of distinct eigenvalues.
  # This is classic nondegenerate perurbation theory, but also see
  # https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
  # The general solution treating the case of degenerate eigenvalues is
  # considerably more complicated. Ambitious readers may refer to the general
  # methods below or refer to degenerate perturbation theory in physics.
  # https://www.win.tue.nl/analysis/reports/rana06-33.pdf and
  # https://people.orie.cornell.edu/aslewis/publications/99-clarke.pdf
  a_dot, = tangents

  v, w_real = eigh_p.bind(
      symmetrize(a),
      lower=lower,
      sort_eigenvalues=sort_eigenvalues,
      subset_by_index=subset_by_index,
  )

  # for complex numbers we need eigenvalues to be full dtype of v, a:
  w = w_real.astype(a.dtype)
  eye_n = lax_internal._eye(a.dtype, (n, n))
  # carefully build reciprocal delta-eigenvalue matrix, avoiding NaNs.
  Fmat = lax.integer_pow(eye_n + w[..., np.newaxis, :] - w[..., np.newaxis], -1) - eye_n
  # eigh impl doesn't support batch dims, but future-proof the grad.
  dot = partial(lax.dot if a.ndim == 2 else lax.batch_matmul,
                precision=lax.Precision.HIGHEST)
  vdag_adot_v = dot(dot(_H(v), a_dot), v)
  dv = dot(v, Fmat * vdag_adot_v)
  dw = _extract_diagonal(vdag_adot_v.real)
  return (v, w_real), (dv, dw)


def _eigh_batching_rule(
    batched_args, batch_dims, *, lower, sort_eigenvalues, subset_by_index
):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return eigh_p.bind(
      x,
      lower=lower,
      sort_eigenvalues=sort_eigenvalues,
      subset_by_index=subset_by_index,
  ), (0, 0)


eigh_p = Primitive('eigh')
eigh_p.multiple_results = True
eigh_p.def_impl(_eigh_impl)
eigh_p.def_abstract_eval(_eigh_abstract_eval)
ad.primitive_jvps[eigh_p] = _eigh_jvp_rule
batching.primitive_batchers[eigh_p] = _eigh_batching_rule

mlir.register_lowering(
    eigh_p, partial(_eigh_cpu_gpu_lowering, target_name_prefix='cpu'),
    platform='cpu')
mlir.register_lowering(
  eigh_p, partial(_eigh_cpu_gpu_lowering, target_name_prefix='cu'),
  platform='cuda')
mlir.register_lowering(
  eigh_p, partial(_eigh_cpu_gpu_lowering, target_name_prefix='hip'),
  platform='rocm')
mlir.register_lowering(
    eigh_p, mlir.lower_fun(_eigh_tpu_impl, multiple_results=True),
    platform='tpu')


_triangular_solve_dtype_rule = partial(
    naryop_dtype_rule, _input_dtype, (_float | _complex, _float | _complex),
    'triangular_solve')

def _triangular_solve_shape_rule(a, b, *, left_side=False, **unused_kwargs):
  if a.ndim < 2:
    msg = "triangular_solve requires a.ndim to be at least 2, got {}."
    raise TypeError(msg.format(a.ndim))
  if b.ndim < 2:
    msg = "triangular_solve requires b.ndim to be at least 2, got {}."
    raise TypeError(msg.format(b.ndim))
  if a.shape[-1] != a.shape[-2]:
    msg = ("triangular_solve requires the last two dimensions of a to be equal "
           "in size, got a.shape of {}.")
    raise TypeError(msg.format(a.shape))
  if a.shape[:-2] != b.shape[:-2]:
    msg = ("triangular_solve requires both arguments to have the same number "
           "of dimensions and equal batch dimensions, got {} and {}.")
    raise TypeError(msg.format(a.shape, b.shape))
  common_dim = -2 if left_side else -1
  if a.shape[-1] != b.shape[common_dim]:
    msg = "Incompatible shapes for arguments to triangular_solve: {} and {}."
    raise TypeError(msg.format(a.shape, b.shape))
  return b.shape

def _triangular_solve_jvp_rule_a(
    g_a, ans, a, b, *, left_side, lower, transpose_a, conjugate_a,
    unit_diagonal):
  m, n = b.shape[-2:]
  k = 1 if unit_diagonal else 0
  g_a = _tril(g_a, k=-k) if lower else _triu(g_a, k=k)
  g_a = lax.neg(g_a)
  g_a = _T(g_a) if transpose_a else g_a
  g_a = g_a.conj() if conjugate_a else g_a
  dot = partial(lax.dot if g_a.ndim == 2 else lax.batch_matmul,
                precision=lax.Precision.HIGHEST)

  def a_inverse(rhs):
    return triangular_solve(a, rhs, left_side=left_side, lower=lower,
                            transpose_a=transpose_a, conjugate_a=conjugate_a,
                            unit_diagonal=unit_diagonal)

  # triangular_solve is about the same cost as matrix multplication (~n^2 FLOPs
  # for matrix/vector inputs). Order these operations in whichever order is
  # cheaper.
  if left_side:
    assert g_a.shape[-2:] == a.shape[-2:] == (m, m) and ans.shape[-2:] == (m, n)
    if m > n:
      return a_inverse(dot(g_a, ans))  # A^{-1} (∂A X)
    else:
      return dot(a_inverse(g_a), ans)  # (A^{-1} ∂A) X
  else:
    assert g_a.shape[-2:] == a.shape[-2:] == (n, n) and ans.shape[-2:] == (m, n)
    if m < n:
      return a_inverse(dot(ans, g_a))  # (X ∂A) A^{-1}
    else:
      return dot(ans, a_inverse(g_a))  # X (∂A A^{-1})

def _triangular_solve_transpose_rule(
    cotangent, a, b, *, left_side, lower, transpose_a, conjugate_a,
    unit_diagonal):
  # Triangular solve is nonlinear in its first argument and linear in its second
  # argument, analogous to `div` but swapped.
  assert not ad.is_undefined_primal(a) and ad.is_undefined_primal(b)
  if type(cotangent) is ad_util.Zero:
    cotangent_b = ad_util.Zero(b.aval)
  else:
    cotangent_b = triangular_solve(a, cotangent, left_side=left_side,
                                   lower=lower, transpose_a=not transpose_a,
                                   conjugate_a=conjugate_a,
                                   unit_diagonal=unit_diagonal)
  return [None, cotangent_b]


def _triangular_solve_batching_rule(batched_args, batch_dims, *, left_side,
                                   lower, transpose_a, conjugate_a,
                                   unit_diagonal):
  x, y = batched_args
  bx, by = batch_dims
  if bx is batching.not_mapped:
    if left_side:
      y = batching.moveaxis(y, by, -1)
      y_flat = y.reshape(y.shape[:-2] + (y.shape[-2] * y.shape[-1],))
      bdim_out = y.ndim - 1
    else:
      y = batching.moveaxis(y, by, -2)
      y_flat = y.reshape(y.shape[:-3]  + (y.shape[-3] * y.shape[-2], y.shape[-1]))
      bdim_out = y.ndim - 2
    out_flat = triangular_solve(
        x, y_flat, left_side=left_side, lower=lower,
        transpose_a=transpose_a, conjugate_a=conjugate_a,
        unit_diagonal=unit_diagonal)
    return out_flat.reshape(y.shape), bdim_out
  else:
    size = next(t.shape[i] for t, i in zip(batched_args, batch_dims)
                if i is not None)
    x = batching.bdim_at_front(x, bx, size)
    y = batching.bdim_at_front(y, by, size)
    return triangular_solve(x, y, left_side=left_side, lower=lower,
                            transpose_a=transpose_a, conjugate_a=conjugate_a,
                            unit_diagonal=unit_diagonal), 0

triangular_solve_p = standard_primitive(
    _triangular_solve_shape_rule, _triangular_solve_dtype_rule,
    'triangular_solve')
ad.defjvp2(triangular_solve_p,
           _triangular_solve_jvp_rule_a,
           lambda g_b, _, a, b, **kws: triangular_solve(a, g_b, **kws))
ad.primitive_transposes[triangular_solve_p] = _triangular_solve_transpose_rule
batching.primitive_batchers[triangular_solve_p] = _triangular_solve_batching_rule


def _triangular_solve_lowering(
    ctx, a, b, *, left_side, lower, transpose_a, conjugate_a, unit_diagonal):
  out_aval, = ctx.avals_out
  if conjugate_a and not transpose_a:
    a = chlo.ConjOp(a)
    conjugate_a = False
  if not transpose_a:
    transpose = "NO_TRANSPOSE"
  else:
    transpose = "ADJOINT" if conjugate_a else "TRANSPOSE"
  return [hlo.triangular_solve(
      a, b, ir.BoolAttr.get(left_side),
      ir.BoolAttr.get(lower), ir.BoolAttr.get(unit_diagonal),
      hlo.TransposeAttr.get(transpose))]


def _triangular_solve_cpu_lower(
    ctx, a, b, *, left_side, lower, transpose_a,
    conjugate_a, unit_diagonal):
  a_aval, b_aval = ctx.avals_in

  if conjugate_a and not transpose_a:
    a = chlo.conj(a)
    conjugate_a = False
  if len(a_aval.shape) == 2 and np.dtype(a_aval.dtype) in _cpu_lapack_types:
    alpha = mlir.ir_constant(np.array(1, dtype=a_aval.dtype))
    b_shape_vals = mlir.eval_dynamic_shape_as_ivals(ctx, b_aval.shape)
    # TODO(b/344892332): Remove the conditional after the compatibility period.
    ctx_args = (ctx,) if jaxlib_version >= (0, 4, 37) else ()
    return lapack.trsm_hlo(
        *ctx_args, a_aval.dtype, alpha,
        a, b, left_side, lower, transpose_a, conjugate_a, unit_diagonal,
        b_shape_vals=b_shape_vals)
  else:
    # Fall back to the HLO implementation for unsupported types or batching.
    # TODO: Consider swapping XLA for LAPACK in batched case
    if transpose_a:
      transpose = "ADJOINT" if conjugate_a else "TRANSPOSE"
    else:
      transpose = "NO_TRANSPOSE"
    return [hlo.triangular_solve(a, b, ir.BoolAttr.get(left_side),
                                 ir.BoolAttr.get(lower),
                                 ir.BoolAttr.get(unit_diagonal),
                                 hlo.TransposeAttr.get(transpose))]


mlir.register_lowering(triangular_solve_p, _triangular_solve_lowering)
mlir.register_lowering(triangular_solve_p, _triangular_solve_cpu_lower,
                       platform='cpu')


# Support operation for LU decomposition: Transformation of the pivots returned
# by LU decomposition into permutations.

# Define this outside lu_pivots_to_permutation to ensure fori_loop cache hits
def _lu_pivots_body_fn(i, permutation_and_swaps):
  permutation, swaps = permutation_and_swaps
  batch_dims = swaps.shape[:-1]
  j = swaps[..., i]
  iotas = _broadcasted_iotas(*batch_dims)
  x = permutation[..., i]
  y = permutation[(*iotas, j)]
  permutation = permutation.at[..., i].set(y)
  return permutation.at[(*iotas, j)].set(x), swaps


def _generic_lu_pivots_to_permutation(swaps, permutation_size):
  """Converts the pivots (row swaps) returned by LU to a permutation.

  We build a permutation rather than applying `swaps` directly to the rows
  of a matrix because lax loops aren't differentiable.

  Args:
    swaps: an array of shape (..., k) of row swaps to perform
    permutation_size: the size of the output permutation. Should be >= k.
  Returns:
    An int32 array of shape (..., m).
  """
  assert len(swaps.shape) >= 1
  batch_dims = swaps.shape[:-1]
  k = swaps.shape[-1]
  m = permutation_size

  permutation = lax.broadcasted_iota(np.int32, batch_dims + (m,),
                                     len(batch_dims))
  if m == 0 or k == 0:
    return permutation
  upper = np.array(k, np.int32) if is_constant_dim(k) else k
  result, _ = lax.fori_loop(np.array(0, np.int32), upper, _lu_pivots_body_fn,
                            (permutation, swaps))
  return result


def _lu_pivots_to_permutation_abstract_eval(pivots, *, permutation_size):
  if isinstance(pivots, ShapedArray):
    if pivots.ndim < 1 or pivots.dtype != np.dtype(np.int32):
      raise ValueError(
          'Argument to lu_pivots_to_permutation must have rank >= 1 and dtype '
          'int32. Got shape={} and dtype={}'.format(pivots.shape, pivots.dtype))
    pivots_size = pivots.shape[-1]
    if not permutation_size >= pivots_size:
      raise ValueError(
          'Output permutation size {} has to exceed the trailing dimension of '
          'the pivots. Got pivots size {}'.format(permutation_size, pivots_size))
    return pivots.update(shape=(*pivots.shape[:-1], permutation_size))
  else:
    return pivots


def _lu_pivots_to_permutation_batching_rule(batched_args, batch_dims, *,
                                            permutation_size):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return lu_pivots_to_permutation_p.bind(
      x, permutation_size=permutation_size), 0

def _lu_pivots_to_permutation_gpu_lowering(platform, ctx, pivots, *,
                                           permutation_size):
  del permutation_size  # unused
  rule = ffi.ffi_lowering(f"{platform}_lu_pivots_to_permutation")
  return rule(ctx, pivots)


lu_pivots_to_permutation_p = Primitive('lu_pivots_to_permutation')
lu_pivots_to_permutation_p.multiple_results = False
lu_pivots_to_permutation_p.def_impl(
    partial(dispatch.apply_primitive, lu_pivots_to_permutation_p))
lu_pivots_to_permutation_p.def_abstract_eval(
    _lu_pivots_to_permutation_abstract_eval)
batching.primitive_batchers[lu_pivots_to_permutation_p] = (
    _lu_pivots_to_permutation_batching_rule)
mlir.register_lowering(
    lu_pivots_to_permutation_p,
    mlir.lower_fun(_generic_lu_pivots_to_permutation, multiple_results=False))
mlir.register_lowering(
    lu_pivots_to_permutation_p,
    partial(_lu_pivots_to_permutation_gpu_lowering, "cu"),
    platform='cuda')
mlir.register_lowering(
    lu_pivots_to_permutation_p,
    partial(_lu_pivots_to_permutation_gpu_lowering, "hip"),
    platform='rocm')

# LU decomposition

# Computes a pivoted LU decomposition such that
# PA = LU
# In the style of LAPACK, LU are stored in the same matrix.

def _lu_unblocked(a):
  """Unblocked LU decomposition, as a rolled loop."""
  m, n = a.shape
  def body(k, state):
    pivot, perm, a = state
    m_idx = lax.iota('int32', m)
    n_idx = lax.iota('int32', n)

    if dtypes.issubdtype(a.dtype, np.complexfloating):
      t = a[:, k]
      magnitude = abs(t.real) + abs(t.imag)
    else:
      magnitude = abs(a[:, k])
    i = lax.argmax(lax.select(m_idx >= k, magnitude, lax.full_like(magnitude, -np.inf)),
                   axis=0, index_dtype=pivot.dtype)
    pivot = pivot.at[k].set(i)
    a = a.at[[k, i],].set(a[[i, k],])
    perm = perm.at[[i, k],].set(perm[[k, i],])

    # a[k+1:, k] /= a[k, k], adapted for loop-invariant shapes
    x = a[k, k]
    a = a.at[:, k].set(lax.select((m_idx > k) & (x != 0), a[:, k] / x, a[:, k]))

    # a[k+1:, k+1:] -= jnp.outer(a[k+1:, k], a[k, k+1:])
    a_outer = a[:, k, None] * a[k, None]
    a = a - lax.select((m_idx[:, None] > k) & (n_idx[None, :] > k),
                       a_outer, lax_internal._zeros(a_outer))
    return pivot, perm, a

  pivot = lax.full((min(m, n),), 0, dtype=np.int32)
  perm = lax.iota('int32', m)
  if m == 0 and n == 0:
    # If the array is empty, the loop body never executes but tracing it to a
    # jaxpr fails because the indexing cannot succeed.
    return (pivot, perm, a)
  return lax.fori_loop(0, min(m, n), body, (pivot, perm, a))


def _lu_blocked(a, block_size=128):
  """Blocked LU decomposition, as an unrolled loop."""
  m, n = a.shape
  r = min(m, n)
  pivot = lax.full((r,), 0, dtype=np.int32)
  perm = lax.iota('int32', m)
  for k in range(0, r, block_size):
    b = min(r - k, block_size)
    block_pivot, block_perm, lu_block = _lu_unblocked(a[k:, k:k+b])

    pivot = pivot.at[k:k+b].set(block_pivot + k)
    perm = perm.at[k:].set(perm[block_perm + k])
    a = a.at[k:, :].set(a[block_perm + k, :])
    a = a.at[k:, k:k+b].set(lu_block)

    if k + b < n:
      a = a.at[k:k+b, k+b:].set(
        triangular_solve(a[k:k+b, k:k+b], a[k:k+b, k+b:], left_side=True,
                         lower=True, unit_diagonal=True))
      a = a.at[k+b:, k+b:].add(-lax.dot(a[k+b:, k:k+b], a[k:k+b, k+b:],
                                        precision=lax.Precision.HIGHEST))
  return a, pivot, perm

def _lu_python(x):
  """Default LU decomposition in Python, where no better version exists."""
  batch_dims = x.shape[:-2]
  fn = _lu_blocked
  for _ in range(len(batch_dims)):
    fn = api.vmap(fn)

  return fn(x)

def _lu_impl(operand):
  lu, pivot, perm = dispatch.apply_primitive(lu_p, operand)
  return lu, pivot, perm

def _lu_abstract_eval(operand):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2:
      raise ValueError("Argument to LU decomposition must have ndims >= 2")

    batch_dims = operand.shape[:-2]
    m = operand.shape[-2]
    n = operand.shape[-1]
    pivot = operand.update(shape=batch_dims + (core.min_dim(m, n),),
                           dtype=np.int32)
    perm = operand.update(shape=batch_dims + (m,), dtype=np.int32)
  else:
    pivot = operand
    perm = operand
  return operand, pivot, perm

def _lu_jvp_rule(primals, tangents):
  a, = primals
  a_dot, = tangents
  lu, pivots, permutation = lu_p.bind(a)

  a_shape = np.shape(a)
  m, n = a_shape[-2:]
  dtype = lax.dtype(a)
  k = min(m, n)

  batch_dims = a_shape[:-2]
  iotas = _broadcasted_iotas(*batch_dims, 1)
  x = a_dot[(*iotas[:-1], permutation, slice(None))]

  # Differentiation of Matrix Functionals Using Triangular Factorization
  # F. R. De Hoog, R. S. Anderssen, and M. A. Lukas
  #
  #     LU = A
  # ==> L'U + LU' = A'
  # ==> inv(L) . L' + U' . inv(U) = inv(L) A' inv(U)
  # ==> L' = L . tril(inv(L) . A' . inv(U), -1)
  #     U' = triu(inv(L) . A' . inv(U)) . U

  ndims = len(a_shape)
  l_padding = [(0, 0, 0)] * ndims
  l_padding[-1] = (0, m - k, 0)
  zero = lax_internal._const(lu, 0)
  l = lax.pad(_tril(lu[..., :, :k], -1), zero, l_padding)
  l = l + lax.expand_dims(lax_internal._eye(dtype, (m, m)), range(l.ndim - 2))
  u_eye = lax.pad(lax_internal._eye(dtype, (n - k, n - k)), zero,
                  ((k, 0, 0), (k, 0, 0)))
  u_padding = [(0, 0, 0)] * ndims
  u_padding[-2] = (0, n - k, 0)
  u = (lax.pad(_triu(lu[..., :k, :]), zero, u_padding) +
       lax.expand_dims(u_eye, range(lu.ndim - 2)))

  la = triangular_solve(l, x, left_side=True, transpose_a=False, lower=True,
                        unit_diagonal=True)
  lau = triangular_solve(u, la, left_side=False, transpose_a=False,
                         lower=False)

  with config.default_matmul_precision("highest"):
    l_dot = l @ _tril(lau, -1)
    u_dot = _triu(lau) @ u
  lu_dot = l_dot + u_dot
  return (lu, pivots, permutation), (lu_dot, ad_util.Zero.from_primal_value(pivots),
                                     ad_util.Zero.from_primal_value(permutation))


def _lu_batching_rule(batched_args, batch_dims):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return lu_p.bind(x), (0, 0, 0)

def _lu_cpu_gpu_lowering(ctx, operand, *, target_name_prefix: str):
  operand_aval, = ctx.avals_in
  out_aval, pivot_aval, perm_aval = ctx.avals_out
  batch_dims = operand_aval.shape[:-2]
  info_aval = ShapedArray(batch_dims, np.dtype(np.int32))
  m = operand_aval.shape[-2]

  if target_name_prefix == "cpu":
    target_name = lapack.prepare_lapack_call("getrf_ffi", operand_aval.dtype)
  else:
    target_name = f"{target_name_prefix}solver_getrf_ffi"

  # We manually construct the layouts because the input and output are
  # expected to be in Fortran order.
  nb = len(batch_dims)
  layout = (nb, nb + 1) + tuple(range(nb - 1, -1, -1))
  result_layouts = [layout, tuple(range(nb, -1, -1)),
                    tuple(range(nb - 1, -1, -1))]
  rule = ffi.ffi_lowering(target_name, operand_layouts=[layout],
                          result_layouts=result_layouts,
                          operand_output_aliases={0: 0})
  sub_ctx = ctx.replace(avals_out=[out_aval, pivot_aval, info_aval])
  lu, pivot, info = rule(sub_ctx, operand)

  # Subtract 1 from the pivot to get 0-based indices.
  pivot = hlo.subtract(pivot, mlir.full_like_aval(ctx, 1, pivot_aval))
  ok = mlir.compare_hlo(info, mlir.full_like_aval(ctx, 0, info_aval),
      "GE", "SIGNED")
  select_lu_aval = ShapedArray(batch_dims + (1, 1), np.dtype(np.bool_))
  lu = _broadcasting_select_hlo(
      ctx,
      mlir.broadcast_in_dim(ctx, ok, select_lu_aval,
                            broadcast_dimensions=range(len(batch_dims))),
      select_lu_aval,
      lu, out_aval, _nan_like_hlo(ctx, out_aval), out_aval)
  sub_ctx = ctx.replace(primitive=None, avals_in=[pivot_aval], avals_out=[perm_aval])
  perm_fn = mlir.lower_fun(lambda x: lu_pivots_to_permutation(x, m),
                           multiple_results=False)
  perm, = perm_fn(sub_ctx, pivot)
  return [lu, pivot, perm]


def _lu_tpu_lowering_rule(ctx, operand):
  result_types = [
    mlir.aval_to_ir_type(ctx.avals_out[0]),
    mlir.aval_to_ir_type(ctx.avals_out[1]),
    mlir.aval_to_ir_type(ctx.avals_out[2])]
  if any(not is_constant_shape(a.shape) for a in ctx.avals_out):
    result_shapes = [
      mlir.eval_dynamic_shape_as_tensor(ctx, a.shape)
      for a in ctx.avals_out]
  else:
    result_shapes = None
  op = mlir.custom_call(
    "LuDecomposition",
    result_types=result_types,
    operands=[operand],
    result_shapes=result_shapes)
  return op.results


lu_p = Primitive('lu')
lu_p.multiple_results = True
lu_p.def_impl(_lu_impl)
lu_p.def_abstract_eval(_lu_abstract_eval)
mlir.register_lowering(lu_p, mlir.lower_fun(_lu_python, multiple_results=True))
ad.primitive_jvps[lu_p] = _lu_jvp_rule
batching.primitive_batchers[lu_p] = _lu_batching_rule

mlir.register_lowering(
    lu_p, partial(_lu_cpu_gpu_lowering, target_name_prefix="cpu"),
    platform="cpu")

mlir.register_lowering(
    lu_p, partial(_lu_cpu_gpu_lowering, target_name_prefix="cu"),
    platform="cuda")
mlir.register_lowering(
    lu_p, partial(_lu_cpu_gpu_lowering, target_name_prefix="hip"),
    platform="rocm")

mlir.register_lowering(lu_p, _lu_tpu_lowering_rule, platform='tpu')


def _lu_solve_core(lu: Array, permutation: Array, b: Array, trans: int) -> Array:
  m = lu.shape[0]
  x = lax.reshape(b, (m, math.prod(b.shape[1:])))
  if trans == 0:
    x = x[permutation, :]
    x = triangular_solve(lu, x, left_side=True, lower=True, unit_diagonal=True)
    x = triangular_solve(lu, x, left_side=True, lower=False)
  elif trans == 1 or trans == 2:
    conj = trans == 2
    x = triangular_solve(lu, x, left_side=True, lower=False, transpose_a=True,
                         conjugate_a=conj)
    x = triangular_solve(lu, x, left_side=True, lower=True, unit_diagonal=True,
                         transpose_a=True, conjugate_a=conj)
    _, ind = lax.sort_key_val(permutation, lax.iota('int32', permutation.shape[0]))
    x = x[ind, :]
  else:
    raise ValueError(f"'trans' value must be 0, 1, or 2, got {trans}")
  return lax.reshape(x, b.shape)


@partial(api.jit, static_argnums=(3,))
def _lu_solve(lu: Array, permutation: Array, b: Array, trans: int) -> Array:
  if len(lu.shape) < 2 or lu.shape[-1] != lu.shape[-2]:
    raise ValueError("last two dimensions of LU decomposition must be equal, "
                     "got shape {}".format(lu.shape))
  if len(b.shape) < 1:
    raise ValueError("b matrix must have rank >= 1, got shape {}"
                     .format(b.shape))
  # Broadcasting follows NumPy's convention for linalg.solve: the RHS is
  # treated as a (batched) vector if the number of dimensions differ by 1.
  # Otherwise, broadcasting rules apply.
  rhs_vector = lu.ndim == b.ndim + 1
  if rhs_vector:
    if b.shape[-1] != lu.shape[-1]:
      raise ValueError("When LU decomposition matrix and b have the same "
                       "number of dimensions, last axis of LU decomposition "
                       "matrix (shape {}) and b array (shape {}) must match"
                       .format(lu.shape, b.shape))
    b = b[..., np.newaxis]
  else:
    if b.shape[-2] != lu.shape[-1]:
      raise ValueError("When LU decomposition matrix and b different "
                       "numbers of dimensions, last axis of LU decomposition "
                       "matrix (shape {}) and second to last axis of b array "
                       "(shape {}) must match"
                       .format(lu.shape, b.shape))

  batch_shape = lax.broadcast_shapes(lu.shape[:-2], permutation.shape[:-1], b.shape[:-2])
  lu = _broadcast_to(lu, (*batch_shape, *lu.shape[-2:]))
  permutation = _broadcast_to(permutation, (*batch_shape, permutation.shape[-1]))
  b = _broadcast_to(b, (*batch_shape, *b.shape[-2:]))
  fn = _lu_solve_core
  for _ in batch_shape:
    fn = api.vmap(fn, in_axes=(0, 0, 0, None))
  x = fn(lu, permutation, b, trans)
  return x[..., 0] if rhs_vector else x


def lu_solve(lu: ArrayLike, permutation: ArrayLike, b: ArrayLike,
             trans: int = 0) -> Array:
  """LU solve with broadcasting."""
  return _lu_solve(lu, permutation, b, trans)


# QR decomposition

# QR decomposition is implemented as a composition of two lower-level primitives
# geqrf and orgqr. The names, while cryptic Fortran alphabet soup, are LAPACK's
# names for the primitives, and we stick with them for consistency.

def geqrf(a: ArrayLike) -> tuple[Array, Array]:
  """Computes the QR decomposition of a matrix.

  Args:
    a: an ``[..., m, n]`` batch of matrices, with floating-point or complex type.
  Returns:
    An ``(a, taus)`` pair where ``r`` is in the upper triangle of ``a``,
    ``q`` is represented in the lower triangle of ``a`` and in ``taus`` as
    elementary Householder reflectors.
  """
  a_out, taus = geqrf_p.bind(a)
  return a_out, taus

def _geqrf_abstract_eval(operand):
  if not isinstance(operand, ShapedArray):
    raise NotImplementedError("Unsupported aval in geqrf_abstract_eval: "
                              f"{operand.aval}")
  if operand.ndim < 2:
    raise ValueError("Argument to QR decomposition must have ndims >= 2")
  *batch_dims, m, n = operand.shape
  taus = operand.update(shape=(*batch_dims, core.min_dim(m, n)))
  return operand, taus

def _geqrf_batching_rule(batched_args, batch_dims):
  x, = batched_args
  bd, = batch_dims
  return geqrf(batching.moveaxis(x, bd, 0)), (0, 0)

def _geqrf_lowering_rule(ctx, operand):
  ts_type = mlir.aval_to_ir_type(ctx.avals_out[0])
  r_type = mlir.aval_to_ir_type(ctx.avals_out[1])
  result_types = [ts_type, r_type]
  if any(not is_constant_shape(aval_out.shape)
         for aval_out in ctx.avals_out):
    result_shapes = [
        mlir.eval_dynamic_shape_as_tensor(ctx, aval_out.shape)
        for aval_out in ctx.avals_out
    ]
  else:
    result_shapes = None
  op = mlir.custom_call(
      "Qr",
      result_types=result_types,
      operands=[operand],
      api_version=1,
      result_shapes=result_shapes
  )
  return op.results

def _geqrf_cpu_gpu_lowering(ctx, a, *, target_name_prefix: str):
  operand_aval, = ctx.avals_in
  batch_dims = operand_aval.shape[:-2]
  nb = len(batch_dims)
  layout = (nb, nb + 1) + tuple(range(nb - 1, -1, -1))
  result_layouts = [layout, tuple(range(nb, -1, -1))]
  if target_name_prefix == "cpu":
    target_name = lapack.prepare_lapack_call("geqrf_ffi", operand_aval.dtype)
  else:
    target_name = f"{target_name_prefix}solver_geqrf_ffi"
  rule = ffi.ffi_lowering(target_name, operand_layouts=[layout],
                          result_layouts=result_layouts,
                          operand_output_aliases={0: 0})
  return rule(ctx, a)

geqrf_p = Primitive('geqrf')
geqrf_p.multiple_results = True
geqrf_p.def_impl(partial(dispatch.apply_primitive, geqrf_p))
geqrf_p.def_abstract_eval(_geqrf_abstract_eval)
batching.primitive_batchers[geqrf_p] = _geqrf_batching_rule
mlir.register_lowering(geqrf_p, _geqrf_lowering_rule)

mlir.register_lowering(
    geqrf_p, partial(_geqrf_cpu_gpu_lowering, target_name_prefix='cpu'),
    platform='cpu')
mlir.register_lowering(
    geqrf_p,
    partial(_geqrf_cpu_gpu_lowering, target_name_prefix='cu'),
    platform='cuda')
mlir.register_lowering(
    geqrf_p,
    partial(_geqrf_cpu_gpu_lowering, target_name_prefix='hip'),
    platform='rocm')


def geqp3(a: ArrayLike, jpvt: ArrayLike) -> tuple[Array, Array, Array]:
  """Computes the column-pivoted QR decomposition of a matrix.

  Args:
    a: a ``[..., m, n]`` batch of matrices, with floating-point or complex type.
    jpvt: a ``[..., n]`` batch of column-pivot index vectors with integer type,
  Returns:
    A ``(a, jpvt, taus)`` triple, where ``r`` is in the upper triangle of ``a``,
    ``q`` is represented in the lower triangle of ``a`` and in ``taus`` as
    elementary Householder reflectors, and ``jpvt`` is the column-pivot indices
    such that ``a[:, jpvt] = q @ r``.
  """
  a_out, jpvt_out, taus = geqp3_p.bind(a, jpvt)
  return a_out, jpvt_out, taus

def _geqp3_abstract_eval(a, jpvt):
  if not isinstance(a, ShapedArray) or not isinstance(jpvt, ShapedArray):
    raise NotImplementedError("Unsupported aval in geqp3_abstract_eval: "
                              f"{a.aval}, {jpvt.aval}")
  if a.ndim < 2:
    raise ValueError("Argument to column-pivoted QR decomposition must have ndims >= 2")
  *batch_dims, m, n = a.shape
  *jpvt_batch_dims, jpvt_n = jpvt.shape
  if batch_dims != jpvt_batch_dims or jpvt_n != n:
    raise ValueError(f"Type mismatch for pivoted QR decomposition: {a=} {jpvt=}")
  taus = a.update(shape=(*batch_dims, core.min_dim(m, n)))
  return a, jpvt, taus

def _geqp3_batching_rule(batched_args, batch_dims):
  a, jpvt = batched_args
  b_a, b_jpvt = batch_dims
  a = batching.moveaxis(a, b_a, 0)
  jpvt = batching.moveaxis(jpvt, b_jpvt, 0)
  return geqp3(a, jpvt), (0, 0, 0)

def _geqp3_cpu_lowering(ctx, a, jpvt):
  a_aval, jpvt_aval = ctx.avals_in
  batch_dims = a_aval.shape[:-2]
  nb = len(batch_dims)
  layout = [(nb, nb + 1) + tuple(range(nb - 1, -1, -1)), tuple(range(nb, -1, -1))]
  result_layouts = layout + [tuple(range(nb, -1, -1))]
  target_name = lapack.prepare_lapack_call("geqp3_ffi", a_aval.dtype)
  rule = ffi.ffi_lowering(target_name, operand_layouts=layout,
                          result_layouts=result_layouts,
                          operand_output_aliases={0: 0, 1: 1})
  return rule(ctx, a, jpvt)


geqp3_p = Primitive('geqp3')
geqp3_p.multiple_results = True
geqp3_p.def_impl(partial(dispatch.apply_primitive, geqp3_p))
geqp3_p.def_abstract_eval(_geqp3_abstract_eval)
batching.primitive_batchers[geqp3_p] = _geqp3_batching_rule
mlir.register_lowering(geqp3_p, _geqp3_cpu_lowering, platform="cpu")

# householder_product: product of elementary Householder reflectors

def householder_product(a: ArrayLike, taus: ArrayLike) -> Array:
  """Product of elementary Householder reflectors.

  Args:
    a: A matrix with shape ``[..., m, n]``, whose lower triangle contains
      elementary Householder reflectors.
    taus: A vector with shape ``[..., k]``, where ``k < min(m, n)``, containing
      the scalar factors of the elementary Householder reflectors.

  Returns:
    A batch of orthogonal (unitary) matrices with the same shape as ``a``,
    containing the products of the elementary Householder reflectors.
  """
  return householder_product_p.bind(a, taus)


def _householder_product_abstract_eval(a, taus):
  if not isinstance(a, ShapedArray) or not isinstance(taus, ShapedArray):
    raise NotImplementedError("Unsupported aval in householder_product_abstract_eval: "
                              f"{a.aval} {taus.aval}")
  if a.ndim < 2:
    raise ValueError("Argument to Householder product must have ndims >= 2")
  *batch_dims, m, n = a.shape
  *taus_batch_dims, k = taus.shape
  if a.dtype != taus.dtype or batch_dims != taus_batch_dims or k > core.min_dim(m, n):
    raise ValueError(f"Type mismatch for Householder product: {a=} {taus=}")
  if m < n:
    raise ValueError("Householder product inputs must have at least as many "
                     f"rows as columns, got shape {a.shape}")
  return a

def _householder_product_batching_rule(batched_args, batch_dims):
  a, taus = batched_args
  b_a, b_taus, = batch_dims
  return householder_product(batching.moveaxis(a, b_a, 0),
               batching.moveaxis(taus, b_taus, 0)), (0,)

def _householder_product_lowering_rule(ctx, a, taus):
  aval_out, = ctx.avals_out
  if not is_constant_shape(aval_out.shape):
    result_shapes = [
        mlir.eval_dynamic_shape_as_tensor(ctx, aval_out.shape)]
  else:
    result_shapes = None
  op = mlir.custom_call(
      "ProductOfElementaryHouseholderReflectors",
      result_types=[mlir.aval_to_ir_type(aval_out)],
      operands=[a, taus],
      api_version=1,
      result_shapes=result_shapes)
  return [op.result]

def _householder_product_cpu_gpu_lowering(ctx, a, taus, *,
                                          target_name_prefix: str):
  a_aval, _ = ctx.avals_in
  batch_dims = a_aval.shape[:-2]
  nb = len(batch_dims)
  layout = (nb, nb + 1) + tuple(range(nb - 1, -1, -1))
  tau_layout = tuple(range(nb, -1, -1))
  if target_name_prefix == "cpu":
    dtype = a_aval.dtype
    prefix = "un" if dtypes.issubdtype(dtype, np.complexfloating) else "or"
    target_name = lapack.prepare_lapack_call(f"{prefix}gqr_ffi", dtype)
  else:
    target_name = f"{target_name_prefix}solver_orgqr_ffi"
  rule = ffi.ffi_lowering(target_name, operand_layouts=[layout, tau_layout],
                          result_layouts=[layout],
                          operand_output_aliases={0: 0})
  return rule(ctx, a, taus)

householder_product_p = Primitive('householder_product')
householder_product_p.def_impl(partial(dispatch.apply_primitive, householder_product_p))
householder_product_p.def_abstract_eval(_householder_product_abstract_eval)
batching.primitive_batchers[householder_product_p] = _householder_product_batching_rule
mlir.register_lowering(householder_product_p, _householder_product_lowering_rule)

mlir.register_lowering(
    householder_product_p,
    partial(_householder_product_cpu_gpu_lowering, target_name_prefix='cpu'),
    platform='cpu')
mlir.register_lowering(
    householder_product_p,
    partial(_householder_product_cpu_gpu_lowering, target_name_prefix='cu'),
    platform='cuda')
mlir.register_lowering(
    householder_product_p,
    partial(_householder_product_cpu_gpu_lowering, target_name_prefix='hip'),
    platform='rocm')


def _qr_impl(operand, *, pivoting, full_matrices):
  q, r, *p = dispatch.apply_primitive(qr_p, operand, pivoting=pivoting,
                                      full_matrices=full_matrices)
  return (q, r, p[0]) if pivoting else (q, r)

def _qr_abstract_eval(operand, *, pivoting, full_matrices):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2:
      raise ValueError("Argument to QR decomposition must have ndims >= 2")
    *batch_dims, m, n = operand.shape
    k = m if full_matrices else core.min_dim(m, n)
    q = operand.update(shape=(*batch_dims, m, k))
    r = operand.update(shape=(*batch_dims, k, n))
    p = operand.update(shape=(*batch_dims, n), dtype=np.dtype(np.int32))
  else:
    q = operand
    r = operand
    p = operand
  return (q, r, p) if pivoting else (q, r)

def qr_jvp_rule(primals, tangents, *, pivoting, full_matrices):
  # See j-towns.github.io/papers/qr-derivative.pdf for a terse derivation.
  x, = primals
  dx, = tangents
  q, r, *p = qr_p.bind(x, pivoting=pivoting, full_matrices=False)
  *_, m, n = x.shape
  if m < n or (full_matrices and m != n):
    raise NotImplementedError(
      "Unimplemented case of QR decomposition derivative")
  if pivoting:
    dx = dx[..., p[0]]
  dx_rinv = triangular_solve(r, dx)  # Right side solve by default
  qt_dx_rinv = _H(q) @ dx_rinv
  qt_dx_rinv_lower = _tril(qt_dx_rinv, -1)
  do = qt_dx_rinv_lower - _H(qt_dx_rinv_lower)  # This is skew-symmetric
  # The following correction is necessary for complex inputs
  I = lax.expand_dims(lax_internal._eye(do.dtype, (n, n)), range(qt_dx_rinv.ndim - 2))
  do = do + I * (qt_dx_rinv - qt_dx_rinv.real.astype(qt_dx_rinv.dtype))
  dq = q @ (do - qt_dx_rinv) + dx_rinv
  dr = (qt_dx_rinv - do) @ r
  if pivoting:
    dp = ad_util.Zero.from_primal_value(p[0])
    return (q, r, p[0]), (dq, dr, dp)
  return (q, r), (dq, dr)

def _qr_batching_rule(batched_args, batch_dims, *, pivoting, full_matrices):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  out_axes = (0, 0, 0) if pivoting else (0, 0)
  return qr_p.bind(x, pivoting=pivoting, full_matrices=full_matrices), out_axes

def _qr_lowering(a, *, pivoting, full_matrices):
  *batch_dims, m, n = a.shape
  if m == 0 or n == 0:
    k = m if full_matrices else core.min_dim(m, n)
    q = lax.broadcast_in_dim(lax_internal._eye(a.dtype, (m, k)),
                             (*batch_dims, m, k),
                             (len(batch_dims), len(batch_dims) + 1))
    r = lax.full((*batch_dims, k, n), 0, dtype=a.dtype)
    if pivoting:
      p = lax.full((*batch_dims, n), 0, dtype=np.dtype(np.int32))
      return q, r, p
    return q, r

  if pivoting:
    jpvt = lax.full((*batch_dims, n), 0, dtype=np.dtype(np.int32))
    r, p, taus = geqp3(a, jpvt)
    p -= 1  # Convert geqp3's 1-based indices to 0-based indices by subtracting 1.
  else:
    r, taus = geqrf(a)

  if m < n:
    q = householder_product(r[..., :m, :m], taus)
  elif full_matrices:
    pads = [(0, 0, 0)] * (len(batch_dims) + 1) + [(0, m - n, 0)]
    q = lax.pad(r, lax_internal._zero(r), pads)
    q = householder_product(q, taus)
  else:
    q = householder_product(r, taus)
    r = r[..., :n, :n]
  r = _triu(r)
  if pivoting:
    return q, r, p
  return q, r


qr_p = Primitive('qr')
qr_p.multiple_results = True
qr_p.def_impl(_qr_impl)
qr_p.def_abstract_eval(_qr_abstract_eval)

ad.primitive_jvps[qr_p] = qr_jvp_rule
batching.primitive_batchers[qr_p] = _qr_batching_rule

mlir.register_lowering(qr_p, mlir.lower_fun(_qr_lowering))

# Singular value decomposition
def _svd_impl(operand, *, full_matrices, compute_uv, subset_by_index=None,
              algorithm=None):
  return dispatch.apply_primitive(
      svd_p,
      operand,
      full_matrices=full_matrices,
      compute_uv=compute_uv,
      subset_by_index=subset_by_index,
      algorithm=algorithm,
  )


def _svd_abstract_eval(operand, *, full_matrices, compute_uv, subset_by_index,
                       algorithm=None):
  del algorithm  # unused
  if isinstance(operand, ShapedArray):
    batch_dims = operand.shape[:-2]
    m = operand.shape[-2]
    n = operand.shape[-1]
    rank = core.min_dim(m, n)
    if subset_by_index is not None:
      if full_matrices and subset_by_index != (0, rank):
        raise ValueError("full_matrices and subset_by_index cannot both be set")
      rank = min(rank, subset_by_index[1] - subset_by_index[0])

    s = operand.update(
        shape=batch_dims + (rank,),
        dtype=lax_internal._complex_basetype(operand.dtype),
    )
    if compute_uv:
      u = operand.update(shape=batch_dims + (m, m if full_matrices else rank))
      vt = operand.update(shape=batch_dims + (n if full_matrices else rank, n))
      return s, u, vt
    else:
      return s,
  else:
    raise NotImplementedError


@config.default_matmul_precision("float32")
def _svd_jvp_rule(
    primals, tangents, *, full_matrices, compute_uv, subset_by_index,
    algorithm=None,
):
  A, = primals
  dA, = tangents
  s, U, Vt = svd_p.bind(
      A, full_matrices=False, compute_uv=True, subset_by_index=subset_by_index,
      algorithm=algorithm,
  )

  if compute_uv and full_matrices:
    # TODO: implement full matrices case, documented here: https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
    raise NotImplementedError(
      "Singular value decomposition JVP not implemented for full matrices")

  Ut, V = _H(U), _H(Vt)
  s_dim = s[..., None, :]
  dS = Ut @ dA @ V
  ds = _extract_diagonal(dS.real)

  if not compute_uv:
    return (s,), (ds,)

  s_diffs = (s_dim + _T(s_dim)) * (s_dim - _T(s_dim))
  s_diffs_zeros = lax_internal._eye(s.dtype, (s.shape[-1], s.shape[-1]))  # jnp.ones((), dtype=A.dtype) * (s_diffs == 0.)  # is 1. where s_diffs is 0. and is 0. everywhere else
  s_diffs_zeros = lax.expand_dims(s_diffs_zeros, range(s_diffs.ndim - 2))
  F = 1 / (s_diffs + s_diffs_zeros) - s_diffs_zeros
  dSS = s_dim.astype(A.dtype) * dS  # dS.dot(jnp.diag(s))
  SdS = _T(s_dim.astype(A.dtype)) * dS  # jnp.diag(s).dot(dS)

  s_zeros = (s == 0).astype(s.dtype)
  s_inv = 1 / (s + s_zeros) - s_zeros
  s_inv_mat = _construct_diagonal(s_inv)
  dUdV_diag = .5 * (dS - _H(dS)) * s_inv_mat.astype(A.dtype)
  dU = U @ (F.astype(A.dtype) * (dSS + _H(dSS)) + dUdV_diag)
  dV = V @ (F.astype(A.dtype) * (SdS + _H(SdS)))

  m, n = A.shape[-2:]
  if m > n:
    dAV = dA @ V
    dU = dU + (dAV - U @ (Ut @ dAV)) / s_dim.astype(A.dtype)
  if n > m:
    dAHU = _H(dA) @ U
    dV = dV + (dAHU - V @ (Vt @ dAHU)) / s_dim.astype(A.dtype)

  return (s, U, Vt), (ds, dU, _H(dV))


def _empty_svd(a, *, full_matrices, compute_uv):
  batch_shape = a.shape[:-2]
  m, n = a.shape[-2:]
  s = lax.full(batch_shape + (0,), 0, dtype=lax_internal._complex_basetype(a.dtype))
  if not compute_uv:
    return (s,)
  if full_matrices:
    size = max(m, n)
    u = lax.broadcast_in_dim(lax_internal._eye(a.dtype, (size, size)),
                             (*batch_shape, size, size),
                             (len(batch_shape), len(batch_shape) + 1))
  else:
    u = lax.full(batch_shape + (m, n), 0, dtype=a.dtype)
  v = lax.full(batch_shape + (0, 0), 0, dtype=a.dtype)
  if m < n:
    u, v = v, u
  return s, u, v


def _svd_cpu_gpu_lowering(
    ctx,
    operand,
    *,
    full_matrices,
    compute_uv,
    subset_by_index,
    target_name_prefix: str,
    algorithm=None,
):
  operand_aval, = ctx.avals_in
  s_aval = ctx.avals_out[0]
  m, n = operand_aval.shape[-2:]
  batch_dims = operand_aval.shape[:-2]

  if not (subset_by_index is None or subset_by_index == (0, min(m, n))):
    raise NotImplementedError("subset_by_index not implemented for CPU and GPU")

  if m == 0 or n == 0:
    return mlir.lower_fun(_empty_svd, multiple_results=True)(
        ctx,
        operand,
        full_matrices=full_matrices,
        compute_uv=compute_uv,
    )
  if target_name_prefix == "cpu":
    if algorithm is not None and algorithm != SvdAlgorithm.DEFAULT:
      raise NotImplementedError(
          "The SVD algorithm parameter is not implemented on CPU.")
    target_name = lapack.prepare_lapack_call("gesdd_ffi", operand_aval.dtype)
    nb = len(batch_dims)
    layout = (nb, nb + 1) + tuple(range(nb - 1, -1, -1))
    result_layouts = [layout, tuple(range(nb, -1, -1)), layout, layout,
                      tuple(range(nb - 1, -1, -1))]
    mode = lapack._svd_computation_attr(compute_uv=compute_uv,
                                        full_matrices=full_matrices)
    rule = ffi.ffi_lowering(target_name, operand_layouts=[layout],
                            result_layouts=result_layouts,
                            operand_output_aliases={0: 0})
    info_aval = ShapedArray(batch_dims, np.dtype(np.int32))
    if compute_uv:
      s_aval, u_aval, vt_aval = ctx.avals_out
    else:
      s_aval, = ctx.avals_out
      # TODO(danfm): It should be possible to skip instantiating these arrays
      # when they are not used.
      u_aval = ShapedArray((*batch_dims, m,
                            m if full_matrices else core.min_dim(m, n)),
                           operand_aval.dtype)
      vt_aval = ShapedArray((*batch_dims,
                             n if full_matrices else core.min_dim(m, n), n),
                            operand_aval.dtype)
    sub_ctx = ctx.replace(avals_out=[operand_aval, s_aval, u_aval, vt_aval,
                                     info_aval])
    _, s, u, vt, info = rule(sub_ctx, operand, mode=mode)
  else:
    s, u, vt, info = _svd_gpu_sub_lowering(ctx, operand,
                                           full_matrices=full_matrices,
                                           compute_uv=compute_uv,
                                           target_name_prefix=target_name_prefix,
                                           algorithm=algorithm)

  zeros = mlir.full_like_aval(ctx, 0, ShapedArray(batch_dims, np.dtype(np.int32)))
  ok = mlir.compare_hlo(info, zeros, "EQ", "SIGNED")
  select_s_aval = ShapedArray(batch_dims + (1,), np.dtype(np.bool_))
  s = _broadcasting_select_hlo(
      ctx,
      mlir.broadcast_in_dim(ctx, ok, select_s_aval,
                            broadcast_dimensions=range(len(batch_dims))),
      select_s_aval,
      s, s_aval, _nan_like_hlo(ctx, s_aval), s_aval)
  result = [s]

  if compute_uv:
    u_aval, vt_aval = ctx.avals_out[1:]
    select_u_aval = ShapedArray(batch_dims + (1, 1), np.dtype(np.bool_))
    u = _broadcasting_select_hlo(
        ctx,
        mlir.broadcast_in_dim(ctx, ok, select_u_aval,
                              broadcast_dimensions=range(len(batch_dims))),
        select_u_aval,
        u, u_aval, _nan_like_hlo(ctx, u_aval), u_aval)
    select_v_aval = ShapedArray(batch_dims + (1, 1), np.dtype(np.bool_))
    vt = _broadcasting_select_hlo(
        ctx,
        mlir.broadcast_in_dim(ctx, ok, select_v_aval,
                              broadcast_dimensions=range(len(batch_dims))),
        select_v_aval,
        vt, vt_aval, _nan_like_hlo(ctx, vt_aval), vt_aval)
    result += [u, vt]

  return result


def _svd_gpu_sub_lowering(ctx, operand, *, full_matrices, compute_uv,
                          target_name_prefix, algorithm):
  operand_aval, = ctx.avals_in
  if compute_uv:
    s_aval, u_aval, vt_aval = ctx.avals_out
  else:
    s_aval, = ctx.avals_out
    u_aval = vt_aval = ShapedArray((), operand_aval.dtype)
  batch_dims = operand_aval.shape[:-2]
  info_aval = ShapedArray(batch_dims, np.dtype(np.int32))
  nb = len(batch_dims)
  m, n = operand_aval.shape[-2:]
  k = core.min_dim(m, n)

  transposed = False
  kwargs = {}

  # The Jacobi algorithm appears to outperform the default QR algorithm for
  # small to medium sized matrices. See:
  # https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9226-fast-singular-value-decomposition-on-gpus-v2.pdf
  # slide 5. With this in mind, we default to using the Jacobi algorithm for
  # matrices smaller than 1024x1024.
  #
  # Note that the Jacobi algorithm is only used by default for matrices with
  # concrete matrix dimensions. When using dynamic shapes, we always use the
  # default QR algorithm, but users can (in principle) override this behavior
  # by passing `use_jacobi=True`.
  #
  # TODO(danfm): Since this was originally implemented, hipSolver appers to
  # have added support for the Jacobi algorithm, so we should investigate
  # removing this condition.
  if algorithm is None or algorithm == SvdAlgorithm.DEFAULT:
    try:
      use_jacobi = target_name_prefix == "cu" and m <= 1024 and n <= 1024
    except core.InconclusiveDimensionOperation:
      use_jacobi = False
  else:
    use_jacobi = algorithm == SvdAlgorithm.JACOBI
  if use_jacobi:
    target_name = f"{target_name_prefix}solver_gesvdj_ffi"
    # The gesvdjbatched kernel doesn't support "econ" mode, but it also only
    # supports matrices up to 32x32, so it's always worth using the batched
    # version and then slicing afterwards when the matrix is small enough.
    try:
      econ = not full_matrices and m > 32 and n > 32
    except core.InconclusiveDimensionOperation:
      econ = False
    layout = (nb, nb + 1) + tuple(range(nb - 1, -1, -1))
  else:
    target_name = f"{target_name_prefix}solver_gesvd_ffi"
    econ = not full_matrices
    # Because the base gesvd kernel only supports matrices where m >= n, we.
    transposed = m < n
    kwargs = {"transposed": transposed}
    if transposed:
      layout = tuple(range(nb + 1, -1, -1))
    else:
      layout = (nb, nb + 1) + tuple(range(nb - 1, -1, -1))

  result_layouts = [layout, tuple(range(nb, -1, -1)),
                    layout if use_jacobi or compute_uv else (),
                    layout if use_jacobi or compute_uv else (),
                    tuple(range(nb - 1, -1, -1))]
  rule = ffi.ffi_lowering(target_name, operand_layouts=[layout],
                          result_layouts=result_layouts,
                          operand_output_aliases={0: 0})
  if use_jacobi:
    # When using the Jacobi algorithm, the U and V matrices must always be
    # allocated even if compute_uv is False.
    u_aval = ShapedArray((*batch_dims, m, k if econ else m), u_aval.dtype)
    v_aval = ShapedArray((*batch_dims, n, k if econ else n), vt_aval.dtype)
    sub_ctx = ctx.replace(avals_out=[operand_aval, s_aval, u_aval, v_aval,
                                     info_aval])
  elif transposed:
    sub_ctx = ctx.replace(avals_out=[operand_aval, s_aval, vt_aval, u_aval,
                                     info_aval])
  else:
    sub_ctx = ctx.replace(avals_out=[operand_aval, s_aval, u_aval, vt_aval,
                                     info_aval])
  _, s, u, vt, info = rule(sub_ctx, operand, full_matrices=not econ,
                           compute_uv=compute_uv, **kwargs)
  if use_jacobi and compute_uv:
    vt = hlo.transpose(
        vt,
        mlir.dense_int_array(np.array(tuple(range(nb)) + (nb + 1, nb))))
    if np.issubdtype(operand_aval.dtype, np.complexfloating):
      vt = hlo.complex(hlo.real(vt), hlo.negate(hlo.imag(vt)))
    if not full_matrices and not econ:
      nd = len(operand_aval.shape)
      u = mlir.slice_op(ctx, u, ctx.avals_out[1],
                        start_indices=np.zeros([nd], np.int64),
                        limit_indices=batch_dims + (m, k),
                        strides=np.ones([nd], np.int64))
      vt = mlir.slice_op(ctx, vt, ctx.avals_out[2],
                         start_indices=np.zeros([nd], np.int64),
                         limit_indices=batch_dims + (k, n),
                         strides=np.ones([nd], np.int64))
  if transposed:
    return s, vt, u, info
  else:
    return s, u, vt, info


def _svd_tpu(a, *, full_matrices, compute_uv, subset_by_index, algorithm=None):
  if algorithm is not None and algorithm != SvdAlgorithm.DEFAULT:
    raise NotImplementedError(
        "The SVD algorithm parameter is not implemented on TPU.")

  batch_dims = a.shape[:-2]
  fn = partial(
      lax_svd.svd,
      full_matrices=full_matrices,
      compute_uv=compute_uv,
      subset_by_index=subset_by_index,
  )
  for _ in range(len(batch_dims)):
    fn = api.vmap(fn)

  if compute_uv:
    u, s, vh = fn(a)
    return [s, u, vh]
  else:
    s = fn(a)
    return [s]


def _svd_tpu_lowering_rule(
    ctx, operand, *, full_matrices, compute_uv, subset_by_index, algorithm=None
):
  del algorithm  # unused
  operand_aval, = ctx.avals_in
  m, n = operand_aval.shape[-2:]

  if m == 0 or n == 0:
    return mlir.lower_fun(_empty_svd, multiple_results=True)(
        ctx,
        operand,
        full_matrices=full_matrices,
        compute_uv=compute_uv,
    )

  return mlir.lower_fun(_svd_tpu, multiple_results=True)(
      ctx,
      operand,
      full_matrices=full_matrices,
      compute_uv=compute_uv,
      subset_by_index=subset_by_index,
  )


def _svd_batching_rule(
    batched_args, batch_dims, *, full_matrices, compute_uv, subset_by_index,
    algorithm=None,
):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  outs = svd_p.bind(
      x,
      full_matrices=full_matrices,
      compute_uv=compute_uv,
      subset_by_index=subset_by_index,
      algorithm=algorithm,
  )

  if compute_uv:
    return outs, (0, 0, 0)
  else:
    return outs, (0,)


svd_p = Primitive('svd')
svd_p.multiple_results = True
svd_p.def_impl(_svd_impl)
svd_p.def_abstract_eval(_svd_abstract_eval)
ad.primitive_jvps[svd_p] = _svd_jvp_rule
batching.primitive_batchers[svd_p] = _svd_batching_rule

mlir.register_lowering(
    svd_p, partial(_svd_cpu_gpu_lowering, target_name_prefix='cpu'),
    platform='cpu')
mlir.register_lowering(
    svd_p, partial(_svd_cpu_gpu_lowering, target_name_prefix='cu'),
    platform='cuda')
mlir.register_lowering(
    svd_p, partial(_svd_cpu_gpu_lowering, target_name_prefix='hip'),
    platform='rocm')
mlir.register_lowering(svd_p, _svd_tpu_lowering_rule)


_tridiagonal_solve_dtype_rule = partial(
    naryop_dtype_rule, _input_dtype, (_float | _complex, _float | _complex,
                                      _float | _complex, _float | _complex),
    'tridiagonal_solve')


def _tridiagonal_solve_shape_rule(dl, d, du, b):
  if b.ndim < 2:
    raise TypeError(
        f"tridiagonal_solve requires b.ndim to be at least 2, got {b.ndim}.")
  if dl.shape != d.shape or dl.shape != du.shape:
    raise TypeError(
        "tridiagonal_solve requires that all diagonal arguments have the same "
        "shape.")
  if dl.shape != b.shape[:-1]:
    raise TypeError(
        "tridiagonal_solve requires that the leading ndim-1 dimensions of b "
        "equal the dimensions of the diagonal arguments.")
  return b.shape


def _tridiagonal_solve_gpu_lowering(lowering, ctx, dl, d, du, b):
  _, _, _, b_aval = ctx.avals_in
  if b_aval.dtype != np.float32 and b_aval.dtype != np.float64:
    raise NotImplementedError(
        "tridiagonal_solve is only implemented for float32 and float64 on GPU.")
  m, n = b_aval.shape[-2:]
  b_shape_vals = mlir.eval_dynamic_shape_as_ivals(ctx, b_aval.shape)
  return [lowering(
      dl, d, du, b, m=m, n=n, ldb=m, t=b_aval.dtype,
      b_shape_vals=b_shape_vals)]


def _tridiagonal_solve_cpu_lowering(ctx, dl, d, du, b, **kwargs):
  if jaxlib_version <= (0, 4, 38):
    rule = mlir.lower_fun(_tridiagonal_solve_jax, multiple_results=False)
    return rule(ctx, dl, d, du, b, **kwargs)
  b_aval = ctx.avals_in[-1]
  batch_dims = b_aval.shape[:-2]
  target_name = lapack.prepare_lapack_call("gtsv_ffi", b_aval.dtype)
  nb = len(batch_dims)
  b_layout = (nb, nb + 1) + tuple(range(nb - 1, -1, -1))
  d_layout = tuple(range(nb, -1, -1))
  layouts = [d_layout, d_layout, d_layout, b_layout]
  info_layout = tuple(range(nb - 1, -1, -1))
  rule = ffi.ffi_lowering(target_name, operand_layouts=layouts,
                          result_layouts=layouts + [info_layout],
                          operand_output_aliases={0: 0, 1: 1, 2: 2, 3: 3})
  info_aval = ShapedArray(batch_dims, np.dtype(np.int32))
  sub_ctx = ctx.replace(avals_out=list(ctx.avals_in) + [info_aval])
  *_, b_out, info = rule(sub_ctx, dl, d, du, b)
  zeros = mlir.full_like_aval(ctx, 0, info_aval)
  ok = mlir.compare_hlo(info, zeros, "EQ", "SIGNED")
  select_b_aval = ShapedArray(batch_dims + (1, 1), np.dtype(np.bool_))
  return [_broadcasting_select_hlo(
      ctx,
      mlir.broadcast_in_dim(ctx, ok, select_b_aval,
                            broadcast_dimensions=range(len(batch_dims))),
      select_b_aval,
      b_out, b_aval, _nan_like_hlo(ctx, b_aval), b_aval)]


def _tridiagonal_product(dl, d, du, b):
  y = lax.reshape(d, d.shape + (1,)) * b
  y = y.at[..., 1:, :].add(dl[..., 1:, None] * b[..., :-1, :])
  y = y.at[..., :-1, :].add(du[..., :-1, None] * b[..., 1:, :])
  return y


def _tridiagonal_solve_jvp_rule(primals, tangents):
  *diags, _ = primals
  *diags_dot, b_dot = tangents
  ans = tridiagonal_solve_p.bind(*primals)
  if all(type(p) is ad_util.Zero for p in diags_dot):
    rhs = b_dot
  else:
    matvec_dot = _tridiagonal_product(*map(ad.instantiate_zeros, diags_dot), ans)
    rhs = ad.add_tangents(b_dot, -matvec_dot)
  ans_dot = tridiagonal_solve_p.bind(*diags, rhs)
  return ans, ans_dot


def _tridiagonal_solve_transpose_rule(cotangent, dl, d, du, b):
  # Tridiagonal solve is nonlinear in the tridiagonal arguments and linear
  # otherwise.
  assert not (ad.is_undefined_primal(dl) or ad.is_undefined_primal(d) or
              ad.is_undefined_primal(du)) and ad.is_undefined_primal(b)
  if type(cotangent) is ad_util.Zero:
    cotangent_b = ad_util.Zero(b.aval)
  else:
    dl_trans = lax.concatenate((lax.zeros_like_array(du[..., -1:]), du[..., :-1]),
                               du.ndim-1)
    du_trans = lax.concatenate((dl[..., 1:], lax.zeros_like_array(dl[..., :1])),
                               dl.ndim-1)
    cotangent_b = tridiagonal_solve(dl_trans, d, du_trans, cotangent)
  return [None, None, None, cotangent_b]


def _tridiagonal_solve_batching_rule(batched_args, batch_dims):
  dl, d, du, b = batched_args
  bdl, bd, bdu, bb = batch_dims
  if (bdl is batching.not_mapped and
      bd is batching.not_mapped and
      bdu is batching.not_mapped):

    b = batching.moveaxis(b, bb, -2)
    b_flat = b.reshape(b.shape[:-3]  + (b.shape[-3], b.shape[-2] * b.shape[-1]))
    bdim_out = b.ndim - 2
    out_flat = tridiagonal_solve(dl, d, du, b_flat)
    return out_flat.reshape(b.shape), bdim_out
  else:
    size = next(t.shape[i] for t, i in zip(batched_args, batch_dims)
                if i is not None)
    dl = batching.bdim_at_front(dl, bdl, size)
    d = batching.bdim_at_front(d, bd, size)
    du = batching.bdim_at_front(du, bdu, size)
    b = batching.bdim_at_front(b, bb, size)
    return tridiagonal_solve(dl, d, du, b), 0


tridiagonal_solve_p = standard_primitive(
    _tridiagonal_solve_shape_rule, _tridiagonal_solve_dtype_rule,
    'tridiagonal_solve')
ad.primitive_jvps[tridiagonal_solve_p] = _tridiagonal_solve_jvp_rule
ad.primitive_transposes[tridiagonal_solve_p] = _tridiagonal_solve_transpose_rule
batching.primitive_batchers[tridiagonal_solve_p] = _tridiagonal_solve_batching_rule

mlir.register_lowering(
    tridiagonal_solve_p,
    _tridiagonal_solve_cpu_lowering,
    platform='cpu')
mlir.register_lowering(
    tridiagonal_solve_p,
    partial(_tridiagonal_solve_gpu_lowering, gpu_sparse.cuda_gtsv2),
    platform='cuda')
mlir.register_lowering(
    tridiagonal_solve_p,
    partial(_tridiagonal_solve_gpu_lowering, gpu_sparse.rocm_gtsv2),
    platform='rocm')


def _tridiagonal_solve_jax_impl(dl, d, du, b):
  def fwd(carry, args):
    cp, dp = carry
    a, b, c, d = args
    cp_next = c / (b - a * cp)
    dp_next = (d - a * dp) / (b - a * cp)
    return (cp_next, dp_next), (cp, dp)

  (_, final), (cp, dp) = lax.scan(
      fwd, (du[0] / d[0], b[0] / d[0]), (dl[1:], d[1:], du[1:], b[1:, :]),
      unroll=32)

  def bwd(xn, args):
    cp, dp = args
    x = dp - cp * xn
    return x, xn

  end, ans = lax.scan(bwd, final, (cp, dp), unroll=32, reverse=True)
  return lax.concatenate((end[None], ans), 0)


def _tridiagonal_solve_jax(dl, d, du, b, **_):
  impl = _tridiagonal_solve_jax_impl
  for _ in range(dl.ndim - 1):
    impl = api.vmap(impl)
  return impl(dl, d, du, b)


mlir.register_lowering(tridiagonal_solve_p, mlir.lower_fun(
    _tridiagonal_solve_jax, multiple_results=False))


def tridiagonal_solve(dl: Array, d: Array, du: Array, b: Array) -> Array:
  r"""Computes the solution of a tridiagonal linear system.

  This function computes the solution of a tridiagonal linear system:

  .. math::
    A . X = B

  Args:

    dl: A batch of vectors with shape ``[..., m]``.
      The lower diagonal of A: ``dl[i] := A[i, i-1]`` for i in ``[0,m)``.
      Note that ``dl[0] = 0``.
    d: A batch of vectors with shape ``[..., m]``.
      The middle diagonal of A: ``d[i]  := A[i, i]`` for i in ``[0,m)``.
    du: A batch of vectors with shape ``[..., m]``.
      The upper diagonal of A: ``du[i] := A[i, i+1]`` for i in ``[0,m)``.
      Note that ``dl[m - 1] = 0``.
    b: Right hand side matrix.

  Returns:
    Solution ``X`` of tridiagonal system.
  """
  return tridiagonal_solve_p.bind(dl, d, du, b)


# Schur Decomposition


def schur(x: ArrayLike, *,
          compute_schur_vectors: bool = True,
          sort_eig_vals: bool = False,
          select_callable: Callable[..., Any] | None = None) -> tuple[Array, Array]:
  return schur_p.bind(
      x,
      compute_schur_vectors=compute_schur_vectors,
      sort_eig_vals=sort_eig_vals,
      select_callable=select_callable)


def _schur_impl(operand, *, compute_schur_vectors, sort_eig_vals,
                select_callable):
  return dispatch.apply_primitive(
      schur_p,
      operand,
      compute_schur_vectors=compute_schur_vectors,
      sort_eig_vals=sort_eig_vals,
      select_callable=select_callable)

def _schur_lowering(ctx, *args, **kwargs):
  raise NotImplementedError(
      "Schur decomposition is only implemented on the CPU backend.")

def _schur_abstract_eval(operand, *, compute_schur_vectors, sort_eig_vals,
                         select_callable):

  if operand.ndim < 2 or operand.shape[-2] != operand.shape[-1]:
    raise ValueError("Argument to Schur decomposition must have "
                     "shape [..., n, n], got shape {}".format(operand.shape))

  batch_dims = operand.shape[:-2]
  n = operand.shape[-1]
  dtype = operand.dtype
  dtype = dtypes.canonicalize_dtype(dtype)
  T = operand.update(shape=batch_dims + (n, n), dtype=dtype)
  vs = operand.update(shape=batch_dims + (n, n), dtype=dtype)

  return (T, vs) if compute_schur_vectors else (T,)

def _schur_cpu_lowering(ctx, operand, *, compute_schur_vectors, sort_eig_vals,
                        select_callable):
  operand_aval, = ctx.avals_in
  batch_dims = operand_aval.shape[:-2]

  a_shape_vals = mlir.eval_dynamic_shape_as_ivals(ctx, operand_aval.shape)
  # TODO(b/344892332): Remove the conditional after the compatibility period.
  ctx_args = (ctx,) if jaxlib_version >= (0, 4, 37) else ()
  gees_result = lapack.gees_hlo(*ctx_args, operand_aval.dtype, operand,
                                jobvs=compute_schur_vectors,
                                sort=sort_eig_vals,
                                select=select_callable,
                                a_shape_vals=a_shape_vals)
  if jaxlib_version >= (0, 4, 37) and not ctx.is_forward_compat():
    schur_form, schur_vectors, _eig_vals, _selected_eig_vals, info = gees_result
  else:
    # Number of return values depends on value of sort_eig_vals.
    schur_form, schur_vectors, *_, info = gees_result

  ok = mlir.compare_hlo(
      info, mlir.full_like_aval(ctx, 0, ShapedArray(batch_dims, np.dtype(np.int32))),
      "EQ", "SIGNED")

  select_schur_form_aval = ShapedArray(batch_dims + (1, 1), np.dtype(np.bool_))
  schur_form = _broadcasting_select_hlo(
      ctx,
      mlir.broadcast_in_dim(
          ctx,
          ok,
          select_schur_form_aval,
          broadcast_dimensions=range(len(batch_dims)),
      ),
      select_schur_form_aval,
      schur_form,
      ctx.avals_out[0],
      _nan_like_hlo(ctx, ctx.avals_out[0]),
      ctx.avals_out[0],
  )
  output = [schur_form]
  if compute_schur_vectors:
    select_vs_aval = ShapedArray(batch_dims + (1, 1), np.dtype(np.bool_))
    schur_vectors = _broadcasting_select_hlo(
        ctx,
        mlir.broadcast_in_dim(
            ctx, ok, select_vs_aval, broadcast_dimensions=range(len(batch_dims))
        ),
        select_vs_aval,
        schur_vectors,
        ctx.avals_out[1],
        _nan_like_hlo(ctx, ctx.avals_out[1]),
        ctx.avals_out[1],
    )

    output.append(schur_vectors)

  return output


def _schur_batching_rule(batched_args, batch_dims, *, compute_schur_vectors,
                         sort_eig_vals, select_callable):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)

  return schur_p.bind(
      x,
      compute_schur_vectors=compute_schur_vectors,
      sort_eig_vals=sort_eig_vals,
      select_callable=select_callable), (0,) * (1 + compute_schur_vectors)


def _schur_jvp_rule(primals, tangents, **kwds):
  raise NotImplementedError(
      'The differentiation rules for the Schur factorization have not been implemented.'
  )


schur_p = Primitive('schur')
schur_p.multiple_results = True
schur_p.def_impl(_schur_impl)
schur_p.def_abstract_eval(_schur_abstract_eval)
mlir.register_lowering(schur_p, _schur_lowering)
mlir.register_lowering(schur_p, _schur_cpu_lowering, platform='cpu')
batching.primitive_batchers[schur_p] = _schur_batching_rule
ad.primitive_jvps[schur_p] = _schur_jvp_rule


# hessenberg: Upper Hessenberg reduction

def hessenberg(a: ArrayLike) -> tuple[Array, Array]:
  """Reduces a square matrix to upper Hessenberg form.

  Currently implemented on CPU only.

  Args:
    a: A floating point or complex square matrix or batch of matrices.

  Returns:
  A ``(a, taus)`` pair, where the upper triangle and first subdiagonal of ``a``
  contain the upper Hessenberg matrix, and the elements below the first
  subdiagonal contain the Householder reflectors. For each Householder
  reflector ``taus`` contains the scalar factors of the elementary Householder
  reflectors.
  """
  return hessenberg_p.bind(a)

def _hessenberg_abstract_eval(a):
  if a.dtype not in (np.float32, np.float64, np.complex64, np.complex128):
    raise TypeError("hessenberg requires a.dtype to be float32, float64, "
                    f"complex64, or complex128, got {a.dtype}.")
  if a.ndim < 2:
    raise TypeError("hessenberg requires a.ndim to be at least 2, got "
                    f"{a.ndim}.")
  if a.shape[-1] != a.shape[-2]:
    raise TypeError("hessenberg requires the last two dimensions of a to be "
                    f"equal in size, got a.shape of {a.shape}.")
  return [a, ShapedArray(a.shape[:-2] + (a.shape[-1] - 1,), a.dtype)]

hessenberg_p = Primitive("hessenberg")
hessenberg_p.def_impl(partial(dispatch.apply_primitive, hessenberg_p))
hessenberg_p.def_abstract_eval(_hessenberg_abstract_eval)
hessenberg_p.multiple_results = True

def _hessenberg_batching_rule(batched_args, batch_dims):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return hessenberg(x), 0

batching.primitive_batchers[hessenberg_p] = _hessenberg_batching_rule

def _hessenberg_cpu_hlo(ctx, a):
  a_aval, = ctx.avals_in
  batch_dims = a_aval.shape[:-2]
  a, taus, info = lapack.gehrd_hlo(ctx, a_aval.dtype, a)
  ok = mlir.compare_hlo(
      info, mlir.full_like_aval(ctx, 0, ShapedArray(batch_dims, np.dtype(np.int32))),
      "EQ", "SIGNED")
  select_a_aval = ShapedArray(batch_dims + (1, 1), np.dtype(np.bool_))
  select_taus_aval = ShapedArray(batch_dims + (1,), np.dtype(np.bool_))
  return [
    _broadcasting_select_hlo(
      ctx,
      mlir.broadcast_in_dim(ctx, ok, select_a_aval,
                            broadcast_dimensions=range(len(batch_dims))),
      select_a_aval,
      a, ctx.avals_out[0], _nan_like_hlo(ctx, ctx.avals_out[0]), ctx.avals_out[0]),
    _broadcasting_select_hlo(
      ctx,
      mlir.broadcast_in_dim(ctx, ok, select_taus_aval,
                            broadcast_dimensions=range(len(batch_dims))),
      select_taus_aval,
      taus, ctx.avals_out[1], _nan_like_hlo(ctx, ctx.avals_out[1]), ctx.avals_out[1]),
    ]

mlir.register_lowering(hessenberg_p, _hessenberg_cpu_hlo, platform='cpu')


# tridiagonal: Upper Hessenberg reduction

def tridiagonal(a: ArrayLike, *, lower=True
               ) -> tuple[Array, Array, Array, Array]:
  """Reduces a symmetric/Hermitian matrix to tridiagonal form.

  Currently implemented on CPU and GPU only.

  Args:
    a: A floating point or complex matrix or batch of matrices.
    lower: Describes which triangle of the input matrices to use.
      The other triangle is ignored and not accessed.

  Returns:
  A ``(a, d, e, taus)`` pair. If ``lower=True``, the diagonal and first subdiagonal of
  matrix (or batch of matrices) ``a`` contain the tridiagonal representation,
  and elements below the first subdiagonal contain the elementary Householder
  reflectors, where additionally ``d`` contains the diagonal of the matrix and ``e`` contains
  the first subdiagonal.If ``lower=False`` the diagonal and first superdiagonal of the
  matrix contains the tridiagonal representation, and elements above the first
  superdiagonal contain the elementary Householder reflectors, where
  additionally ``d`` contains the diagonal of the matrix and ``e`` contains the
  first superdiagonal. ``taus`` contains the scalar factors of the elementary
  Householder reflectors.
  """
  arr, d, e, taus, info = tridiagonal_p.bind(lax_internal.asarray(a), lower=lower)
  def nans_like(arr):
    if dtypes.issubdtype(arr.dtype, np.complexfloating):
      return lax.full_like(arr, np.nan + 1j * np.nan)
    return lax.full_like(arr, np.nan)
  mask = lambda x: lax.broadcast_in_dim(info == 0, x.shape, range(info.ndim))
  arr = lax.select(mask(arr), arr, nans_like(arr))
  d = lax.select(mask(d), d, nans_like(d))
  e = lax.select(mask(e), e, nans_like(e))
  taus = lax.select(mask(taus), taus, nans_like(taus))
  return arr, d, e, taus

def _tridiagonal_abstract_eval(a, *, lower):
  if a.dtype not in (np.float32, np.float64, np.complex64, np.complex128):
    raise TypeError("tridiagonal requires a.dtype to be float32, float64, "
                    f"complex64, or complex128, got {a.dtype}.")
  if a.ndim < 2:
    raise TypeError("tridiagonal requires a.ndim to be at least 2, got "
                    f"{a.ndim}.")
  if a.shape[-1] != a.shape[-2]:
    raise TypeError("tridiagonal requires the last two dimensions of a to be "
                    f"equal in size, got a.shape of {a.shape}.")
  if a.shape[-1] == 0:
    raise TypeError("tridiagonal requires the last two dimensions of a to be "
                    f"non-zero, got a.shape of {a.shape}.")
  real_dtype = dtypes.finfo(a.dtype).dtype
  return [
      a,
      ShapedArray(a.shape[:-2] + (a.shape[-1],), real_dtype),
      ShapedArray(a.shape[:-2] + (a.shape[-1] - 1,), real_dtype),
      ShapedArray(a.shape[:-2] + (a.shape[-1] - 1,), a.dtype),
      ShapedArray(a.shape[:-2], np.int32)
  ]

tridiagonal_p = Primitive("tridiagonal")
tridiagonal_p.def_impl(partial(dispatch.apply_primitive, tridiagonal_p))
tridiagonal_p.def_abstract_eval(_tridiagonal_abstract_eval)
tridiagonal_p.multiple_results = True

def _tridiagonal_batching_rule(batched_args, batch_dims, *, lower):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return tridiagonal(x, lower=lower), 0

batching.primitive_batchers[tridiagonal_p] = _tridiagonal_batching_rule

def _tridiagonal_cpu_hlo(ctx, a, *, lower):
  a_aval, = ctx.avals_in
  return lapack.sytrd_hlo(ctx, a_aval.dtype, a, lower=lower)

def _tridiagonal_gpu_hlo(ctx, a, *, lower, target_name_prefix):
  operand_aval, = ctx.avals_in
  dims = operand_aval.shape
  batch_dims = dims[:-2]
  nb = len(batch_dims)
  layout = (nb, nb + 1) + tuple(range(nb - 1, -1, -1))
  result_layouts = [layout, tuple(range(nb, -1, -1)), tuple(range(nb, -1, -1)),
                    tuple(range(nb, -1, -1)), tuple(range(nb - 1, -1, -1))]
  rule = ffi.ffi_lowering(f"{target_name_prefix}solver_sytrd_ffi",
                          operand_layouts=[layout],
                          result_layouts=result_layouts,
                          operand_output_aliases={0: 0})
  return rule(ctx, a, lower=lower)


mlir.register_lowering(
    tridiagonal_p, _tridiagonal_cpu_hlo, platform="cpu")
mlir.register_lowering(
    tridiagonal_p,
    partial(_tridiagonal_gpu_hlo, target_name_prefix="cu"),
    platform="cuda",
)
mlir.register_lowering(
    tridiagonal_p,
    partial(_tridiagonal_gpu_hlo, target_name_prefix="hip"),
    platform="rocm",
)

# Utilities

def _nan_like_hlo(ctx: mlir.LoweringRuleContext, aval) -> ir.Value:
  if dtypes.issubdtype(aval.dtype, np.complexfloating):
    return mlir.full_like_aval(ctx, np.nan + np.nan * 1j, aval)
  else:
    return mlir.full_like_aval(ctx, np.nan, aval)

def _broadcasting_select_hlo(ctx, which, which_aval, x, x_aval, y, y_aval) -> ir.Value:
  """Wrapper around XLA `Select` that broadcasts its arguments."""
  out_shapes = list(lax_internal.broadcast_shapes(
      tuple(which_aval.shape), tuple(x_aval.shape), tuple(y_aval.shape)))
  which, x, y = mlir.multi_broadcast_in_dim(ctx, (which, x, y),
                                            (which_aval, x_aval, y_aval),
                                            out_shapes)
  return hlo.select(which, x, y)
