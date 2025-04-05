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

from collections.abc import Sequence
from functools import partial
import itertools
import math

import numpy as np
import operator
from typing import Literal, NamedTuple, overload

import jax
from jax import jit, custom_jvp
from jax import lax

from jax._src import deprecations
from jax._src.lax import lax as lax_internal
from jax._src.lax.lax import PrecisionLike
from jax._src.lax import linalg as lax_linalg
from jax._src.numpy import lax_numpy as jnp
from jax._src.numpy import reductions, ufuncs
from jax._src.numpy.util import promote_dtypes_inexact, check_arraylike
from jax._src.util import canonicalize_axis, set_module
from jax._src.typing import ArrayLike, Array, DTypeLike, DeprecatedArg


export = set_module('jax.numpy.linalg')


class EighResult(NamedTuple):
  eigenvalues: jax.Array
  eigenvectors: jax.Array


class QRResult(NamedTuple):
  Q: jax.Array
  R: jax.Array


class SlogdetResult(NamedTuple):
  sign: jax.Array
  logabsdet: jax.Array


class SVDResult(NamedTuple):
  U: jax.Array
  S: jax.Array
  Vh: jax.Array


def _H(x: ArrayLike) -> Array:
  return ufuncs.conjugate(jnp.matrix_transpose(x))


def _symmetrize(x: Array) -> Array: return (x + _H(x)) / 2


@export
@partial(jit, static_argnames=['upper'])
def cholesky(a: ArrayLike, *, upper: bool = False) -> Array:
  """Compute the Cholesky decomposition of a matrix.

  JAX implementation of :func:`numpy.linalg.cholesky`.

  The Cholesky decomposition of a matrix `A` is:

  .. math::

     A = U^HU

  or

  .. math::

    A = LL^H

  where `U` is an upper-triangular matrix and `L` is a lower-triangular matrix, and
  :math:`X^H` is the Hermitian transpose of `X`.

  Args:
    a: input array, representing a (batched) positive-definite hermitian matrix.
      Must have shape ``(..., N, N)``.
    upper: if True, compute the upper Cholesky decomposition `U`. if False
      (default), compute the lower Cholesky decomposition `L`.

  Returns:
    array of shape ``(..., N, N)`` representing the Cholesky decomposition
    of the input. If the input is not Hermitian positive-definite, The result
    will contain NaN entries.


  See also:
    - :func:`jax.scipy.linalg.cholesky`: SciPy-style Cholesky API
    - :func:`jax.lax.linalg.cholesky`: XLA-style Cholesky API

  Examples:
    A small real Hermitian positive-definite matrix:

    >>> x = jnp.array([[2., 1.],
    ...                [1., 2.]])

    Lower Cholesky factorization:

    >>> jnp.linalg.cholesky(x)
    Array([[1.4142135 , 0.        ],
           [0.70710677, 1.2247449 ]], dtype=float32)

    Upper Cholesky factorization:

    >>> jnp.linalg.cholesky(x, upper=True)
    Array([[1.4142135 , 0.70710677],
           [0.        , 1.2247449 ]], dtype=float32)

    Reconstructing ``x`` from its factorization:

    >>> L = jnp.linalg.cholesky(x)
    >>> jnp.allclose(x, L @ L.T)
    Array(True, dtype=bool)
  """
  check_arraylike("jnp.linalg.cholesky", a)
  a, = promote_dtypes_inexact(jnp.asarray(a))
  L = lax_linalg.cholesky(a)
  return L.mT.conj() if upper else L


@overload
def svd(
    a: ArrayLike,
    full_matrices: bool = True,
    *,
    compute_uv: Literal[True],
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> SVDResult:
  ...


@overload
def svd(
    a: ArrayLike,
    full_matrices: bool,
    compute_uv: Literal[True],
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> SVDResult:
  ...


@overload
def svd(
    a: ArrayLike,
    full_matrices: bool = True,
    *,
    compute_uv: Literal[False],
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> Array:
  ...


@overload
def svd(
    a: ArrayLike,
    full_matrices: bool,
    compute_uv: Literal[False],
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> Array:
  ...


@overload
def svd(
    a: ArrayLike,
    full_matrices: bool = True,
    compute_uv: bool = True,
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> Array | SVDResult:
  ...


@export
@partial(
    jit,
    static_argnames=(
        "full_matrices",
        "compute_uv",
        "hermitian",
        "subset_by_index",
    ),
)
def svd(
    a: ArrayLike,
    full_matrices: bool = True,
    compute_uv: bool = True,
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> Array | SVDResult:
  r"""Compute the singular value decomposition.

  JAX implementation of :func:`numpy.linalg.svd`, implemented in terms of
  :func:`jax.lax.linalg.svd`.

  The SVD of a matrix `A` is given by

  .. math::

     A = U\Sigma V^H

  - :math:`U` contains the left singular vectors and satisfies :math:`U^HU=I`
  - :math:`V` contains the right singular vectors and satisfies :math:`V^HV=I`
  - :math:`\Sigma` is a diagonal matrix of singular values.

  Args:
    a: input array, of shape ``(..., N, M)``
    full_matrices: if True (default) compute the full matrices; i.e. ``u`` and ``vh`` have
      shape ``(..., N, N)`` and ``(..., M, M)``. If False, then the shapes are
      ``(..., N, K)`` and ``(..., K, M)`` with ``K = min(N, M)``.
    compute_uv: if True (default), return the full SVD ``(u, s, vh)``. If False then return
      only the singular values ``s``.
    hermitian: if True, assume the matrix is hermitian, which allows for a more efficient
      implementation (default=False)
    subset_by_index: (TPU-only) Optional 2-tuple [start, end] indicating the range of
      indices of singular values to compute. For example, if ``[n-2, n]`` then
      ``svd`` computes the two largest singular values and their singular vectors.
      Only compatible with ``full_matrices=False``.

  Returns:
    A tuple of arrays ``(u, s, vh)`` if ``compute_uv`` is True, otherwise the array ``s``.

    - ``u``: left singular vectors of shape ``(..., N, N)`` if ``full_matrices`` is True
      or ``(..., N, K)`` otherwise.
    - ``s``: singular values of shape ``(..., K)``
    - ``vh``: conjugate-transposed right singular vectors of shape ``(..., M, M)``
      if ``full_matrices`` is True or ``(..., K, M)`` otherwise.

    where ``K = min(N, M)``.

  See also:
    - :func:`jax.scipy.linalg.svd`: SciPy-style SVD API
    - :func:`jax.lax.linalg.svd`: XLA-style SVD API

  Examples:
    Consider the SVD of a small real-valued array:

    >>> x = jnp.array([[1., 2., 3.],
    ...                [6., 5., 4.]])
    >>> u, s, vt = jnp.linalg.svd(x, full_matrices=False)
    >>> s  # doctest: +SKIP
    Array([9.361919 , 1.8315067], dtype=float32)

    The singular vectors are in the columns of ``u`` and ``v = vt.T``. These vectors are
    orthonormal, which can be demonstrated by comparing the matrix product with the
    identity matrix:

    >>> jnp.allclose(u.T @ u, jnp.eye(2), atol=1E-5)
    Array(True, dtype=bool)
    >>> v = vt.T
    >>> jnp.allclose(v.T @ v, jnp.eye(2), atol=1E-5)
    Array(True, dtype=bool)

    Given the SVD, ``x`` can be reconstructed via matrix multiplication:

    >>> x_reconstructed = u @ jnp.diag(s) @ vt
    >>> jnp.allclose(x_reconstructed, x)
    Array(True, dtype=bool)
  """
  check_arraylike("jnp.linalg.svd", a)
  a, = promote_dtypes_inexact(jnp.asarray(a))
  if hermitian:
    w, v = lax_linalg.eigh(a, subset_by_index=subset_by_index)
    s = lax.abs(v)
    if compute_uv:
      sign = lax.sign(v)
      idxs = lax.broadcasted_iota(np.int64, s.shape, dimension=s.ndim - 1)
      s, idxs, sign = lax.sort((s, idxs, sign), dimension=-1, num_keys=1)
      s = lax.rev(s, dimensions=[s.ndim - 1])
      idxs = lax.rev(idxs, dimensions=[s.ndim - 1])
      sign = lax.rev(sign, dimensions=[s.ndim - 1])
      u = jnp.take_along_axis(w, idxs[..., None, :], axis=-1)
      vh = _H(u * sign[..., None, :].astype(u.dtype))
      return SVDResult(u, s, vh)
    else:
      return lax.rev(lax.sort(s, dimension=-1), dimensions=[s.ndim-1])

  if compute_uv:
    u, s, vh = lax_linalg.svd(
        a,
        full_matrices=full_matrices,
        compute_uv=True,
        subset_by_index=subset_by_index,
    )
    return SVDResult(u, s, vh)
  else:
    return lax_linalg.svd(
        a,
        full_matrices=full_matrices,
        compute_uv=False,
        subset_by_index=subset_by_index,
    )


@export
@partial(jit, static_argnames=('n',))
def matrix_power(a: ArrayLike, n: int) -> Array:
  """Raise a square matrix to an integer power.

  JAX implementation of :func:`numpy.linalg.matrix_power`, implemented via
  repeated squarings.

  Args:
    a: array of shape ``(..., M, M)`` to be raised to the power `n`.
    n: the integer exponent to which the matrix should be raised.

  Returns:
    Array of shape ``(..., M, M)`` containing the matrix power of a to the n.

  Examples:
    >>> a = jnp.array([[1., 2.],
    ...                [3., 4.]])
    >>> jnp.linalg.matrix_power(a, 3)
    Array([[ 37.,  54.],
           [ 81., 118.]], dtype=float32)
    >>> a @ a @ a  # equivalent evaluated directly
    Array([[ 37.,  54.],
           [ 81., 118.]], dtype=float32)

    This also supports zero powers:

    >>> jnp.linalg.matrix_power(a, 0)
    Array([[1., 0.],
           [0., 1.]], dtype=float32)

    and also supports negative powers:

    >>> with jnp.printoptions(precision=3):
    ...   jnp.linalg.matrix_power(a, -2)
    Array([[ 5.5 , -2.5 ],
           [-3.75,  1.75]], dtype=float32)

    Negative powers are equivalent to matmul of the inverse:

    >>> inv_a = jnp.linalg.inv(a)
    >>> with jnp.printoptions(precision=3):
    ...   inv_a @ inv_a
    Array([[ 5.5 , -2.5 ],
           [-3.75,  1.75]], dtype=float32)
  """
  check_arraylike("jnp.linalg.matrix_power", a)
  arr, = promote_dtypes_inexact(jnp.asarray(a))

  if arr.ndim < 2:
    raise TypeError("{}-dimensional array given. Array must be at least "
                    "two-dimensional".format(arr.ndim))
  if arr.shape[-2] != arr.shape[-1]:
    raise TypeError("Last 2 dimensions of the array must be square")
  try:
    n = operator.index(n)
  except TypeError as err:
    raise TypeError(f"exponent must be an integer, got {n}") from err

  if n == 0:
    return jnp.broadcast_to(jnp.eye(arr.shape[-2], dtype=arr.dtype), arr.shape)
  elif n < 0:
    arr = inv(arr)
    n = abs(n)

  if n == 1:
    return arr
  elif n == 2:
    return arr @ arr
  elif n == 3:
    return (arr @ arr) @ arr

  z = result = None
  while n > 0:
    z = arr if z is None else (z @ z)  # type: ignore[operator]
    n, bit = divmod(n, 2)
    if bit:
      result = z if result is None else (result @ z)
  assert result is not None
  return result


@export
@jit
def matrix_rank(
  M: ArrayLike, rtol: ArrayLike | None = None, *,
  tol: ArrayLike | DeprecatedArg | None = DeprecatedArg()) -> Array:
  """Compute the rank of a matrix.

  JAX implementation of :func:`numpy.linalg.matrix_rank`.

  The rank is calculated via the Singular Value Decomposition (SVD), and determined
  by the number of singular values greater than the specified tolerance.

  Args:
    M: array of shape ``(..., N, K)`` whose rank is to be computed.
    rtol: optional array of shape ``(...)`` specifying the tolerance. Singular values
      smaller than `rtol * largest_singular_value` are considered to be zero. If
      ``rtol`` is None (the default), a reasonable default is chosen based the
      floating point precision of the input.
    tol: deprecated alias of the ``rtol`` argument. Will result in a
      :class:`DeprecationWarning` if used.

  Returns:
    array of shape ``a.shape[-2]`` giving the matrix rank.

  Notes:
    The rank calculation may be inaccurate for matrices with very small singular
    values or those that are numerically ill-conditioned. Consider adjusting the
    ``rtol`` parameter or using a more specialized rank computation method in such cases.

  Examples:
    >>> a = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> jnp.linalg.matrix_rank(a)
    Array(2, dtype=int32)

    >>> b = jnp.array([[1, 0],  # Rank-deficient matrix
    ...                [0, 0]])
    >>> jnp.linalg.matrix_rank(b)
    Array(1, dtype=int32)
  """
  check_arraylike("jnp.linalg.matrix_rank", M)
  # TODO(micky774): deprecated 2024-5-14, remove after deprecation expires.
  if not isinstance(tol, DeprecatedArg):
    rtol = tol
    del tol
    deprecations.warn(
      "jax-numpy-linalg-matrix_rank-tol",
      ("The tol argument for linalg.matrix_rank is deprecated. "
       "Please use rtol instead."),
      stacklevel=2
    )
  M, = promote_dtypes_inexact(jnp.asarray(M))
  if M.ndim < 2:
    return (M != 0).any().astype(jnp.int32)
  S = svd(M, full_matrices=False, compute_uv=False)
  if rtol is None:
    rtol = S.max(-1) * np.max(M.shape[-2:]).astype(S.dtype) * jnp.finfo(S.dtype).eps
  rtol = jnp.expand_dims(rtol, np.ndim(rtol))
  return reductions.sum(S > rtol, axis=-1)


@custom_jvp
def _slogdet_lu(a: Array) -> tuple[Array, Array]:
  dtype = lax.dtype(a)
  lu, pivot, _ = lax_linalg.lu(a)
  diag = jnp.diagonal(lu, axis1=-2, axis2=-1)
  is_zero = reductions.any(diag == jnp.array(0, dtype=dtype), axis=-1)
  iota = lax.expand_dims(jnp.arange(a.shape[-1], dtype=pivot.dtype),
                         range(pivot.ndim - 1))
  parity = reductions.count_nonzero(pivot != iota, axis=-1)
  if jnp.iscomplexobj(a):
    sign = reductions.prod(diag / ufuncs.abs(diag).astype(diag.dtype), axis=-1)
  else:
    sign = jnp.array(1, dtype=dtype)
    parity = parity + reductions.count_nonzero(diag < 0, axis=-1)
  sign = jnp.where(is_zero,
                  jnp.array(0, dtype=dtype),
                  sign * jnp.array(-2 * (parity % 2) + 1, dtype=dtype))
  logdet = jnp.where(
      is_zero, jnp.array(-jnp.inf, dtype=dtype),
      reductions.sum(ufuncs.log(ufuncs.abs(diag)).astype(dtype), axis=-1))
  return sign, ufuncs.real(logdet)

@custom_jvp
def _slogdet_qr(a: Array) -> tuple[Array, Array]:
  # Implementation of slogdet using QR decomposition. One reason we might prefer
  # QR decomposition is that it is more amenable to a fast batched
  # implementation on TPU because of the lack of row pivoting.
  if jnp.issubdtype(lax.dtype(a), jnp.complexfloating):
    raise NotImplementedError("slogdet method='qr' not implemented for complex "
                              "inputs")
  n = a.shape[-1]
  a, taus = lax_linalg.geqrf(a)
  # The determinant of a triangular matrix is the product of its diagonal
  # elements. We are working in log space, so we compute the magnitude as the
  # the trace of the log-absolute values, and we compute the sign separately.
  a_diag = jnp.diagonal(a, axis1=-2, axis2=-1)
  log_abs_det = reductions.sum(ufuncs.log(ufuncs.abs(a_diag)), axis=-1)
  sign_diag = reductions.prod(ufuncs.sign(a_diag), axis=-1)
  # The determinant of a Householder reflector is -1. So whenever we actually
  # made a reflection (tau != 0), multiply the result by -1.
  sign_taus = reductions.prod(jnp.where(taus[..., :(n-1)] != 0, -1, 1), axis=-1).astype(sign_diag.dtype)
  return sign_diag * sign_taus, log_abs_det


@export
@partial(jit, static_argnames=('method',))
def slogdet(a: ArrayLike, *, method: str | None = None) -> SlogdetResult:
  """
  Compute the sign and (natural) logarithm of the determinant of an array.

  JAX implementation of :func:`numpy.linalg.slogdet`.

  Args:
    a: array of shape ``(..., M, M)`` for which to compute the sign and log determinant.
    method: the method to use for determinant computation. Options are

      - ``'lu'`` (default): use the LU decomposition.
      - ``'qr'``: use the QR decomposition.

  Returns:
    A tuple of arrays ``(sign, logabsdet)``, each of shape ``a.shape[:-2]``

    - ``sign`` is the sign of the determinant.
    - ``logabsdet`` is the natural log of the determinant's absolute value.

  See also:
    :func:`jax.numpy.linalg.det`: direct computation of determinant

  Examples:
    >>> a = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> sign, logabsdet = jnp.linalg.slogdet(a)
    >>> sign  # -1 indicates negative determinant
    Array(-1., dtype=float32)
    >>> jnp.exp(logabsdet)  # Absolute value of determinant
    Array(2., dtype=float32)
  """
  check_arraylike("jnp.linalg.slogdet", a)
  a, = promote_dtypes_inexact(jnp.asarray(a))
  a_shape = jnp.shape(a)
  if len(a_shape) < 2 or a_shape[-1] != a_shape[-2]:
    raise ValueError(f"Argument to slogdet() must have shape [..., n, n], got {a_shape}")
  if method is None or method == "lu":
    return SlogdetResult(*_slogdet_lu(a))
  elif method == "qr":
    return SlogdetResult(*_slogdet_qr(a))
  else:
    raise ValueError(f"Unknown slogdet method '{method}'. Supported methods "
                     "are 'lu' (`None`), and 'qr'.")

def _slogdet_jvp(primals, tangents):
  x, = primals
  g, = tangents
  sign, ans = slogdet(x)
  ans_dot = jnp.trace(solve(x, g), axis1=-1, axis2=-2)
  if jnp.issubdtype(jnp._dtype(x), jnp.complexfloating):
    sign_dot = (ans_dot - ufuncs.real(ans_dot).astype(ans_dot.dtype)) * sign
    ans_dot = ufuncs.real(ans_dot)
  else:
    sign_dot = jnp.zeros_like(sign)
  return (sign, ans), (sign_dot, ans_dot)

_slogdet_lu.defjvp(_slogdet_jvp)
_slogdet_qr.defjvp(_slogdet_jvp)

def _cofactor_solve(a: ArrayLike, b: ArrayLike) -> tuple[Array, Array]:
  """Equivalent to det(a)*solve(a, b) for nonsingular mat.

  Intermediate function used for jvp and vjp of det.
  This function borrows heavily from jax.numpy.linalg.solve and
  jax.numpy.linalg.slogdet to compute the gradient of the determinant
  in a way that is well defined even for low rank matrices.

  This function handles two different cases:
  * rank(a) == n or n-1
  * rank(a) < n-1

  For rank n-1 matrices, the gradient of the determinant is a rank 1 matrix.
  Rather than computing det(a)*solve(a, b), which would return NaN, we work
  directly with the LU decomposition. If a = p @ l @ u, then
  det(a)*solve(a, b) =
  prod(diag(u)) * u^-1 @ l^-1 @ p^-1 b =
  prod(diag(u)) * triangular_solve(u, solve(p @ l, b))
  If a is rank n-1, then the lower right corner of u will be zero and the
  triangular_solve will fail.
  Let x = solve(p @ l, b) and y = det(a)*solve(a, b).
  Then y_{n}
  x_{n} / u_{nn} * prod_{i=1...n}(u_{ii}) =
  x_{n} * prod_{i=1...n-1}(u_{ii})
  So by replacing the lower-right corner of u with prod_{i=1...n-1}(u_{ii})^-1
  we can avoid the triangular_solve failing.
  To correctly compute the rest of y_{i} for i != n, we simply multiply
  x_{i} by det(a) for all i != n, which will be zero if rank(a) = n-1.

  For the second case, a check is done on the matrix to see if `solve`
  returns NaN or Inf, and gives a matrix of zeros as a result, as the
  gradient of the determinant of a matrix with rank less than n-1 is 0.
  This will still return the correct value for rank n-1 matrices, as the check
  is applied *after* the lower right corner of u has been updated.

  Args:
    a: A square matrix or batch of matrices, possibly singular.
    b: A matrix, or batch of matrices of the same dimension as a.

  Returns:
    det(a) and cofactor(a)^T*b, aka adjugate(a)*b
  """
  a, = promote_dtypes_inexact(jnp.asarray(a))
  b, = promote_dtypes_inexact(jnp.asarray(b))
  a_shape = jnp.shape(a)
  b_shape = jnp.shape(b)
  a_ndims = len(a_shape)
  if not (a_ndims >= 2 and a_shape[-1] == a_shape[-2]
    and b_shape[-2:] == a_shape[-2:]):
    msg = ("The arguments to _cofactor_solve must have shapes "
           "a=[..., m, m] and b=[..., m, m]; got a={} and b={}")
    raise ValueError(msg.format(a_shape, b_shape))
  if a_shape[-1] == 1:
    return a[..., 0, 0], b
  # lu contains u in the upper triangular matrix and l in the strict lower
  # triangular matrix.
  # The diagonal of l is set to ones without loss of generality.
  lu, pivots, permutation = lax_linalg.lu(a)
  dtype = lax.dtype(a)
  batch_dims = lax.broadcast_shapes(lu.shape[:-2], b.shape[:-2])
  x = jnp.broadcast_to(b, batch_dims + b.shape[-2:])
  lu = jnp.broadcast_to(lu, batch_dims + lu.shape[-2:])
  # Compute (partial) determinant, ignoring last diagonal of LU
  diag = jnp.diagonal(lu, axis1=-2, axis2=-1)
  iota = lax.expand_dims(jnp.arange(a_shape[-1], dtype=pivots.dtype),
                         range(pivots.ndim - 1))
  parity = reductions.count_nonzero(pivots != iota, axis=-1)
  sign = jnp.asarray(-2 * (parity % 2) + 1, dtype=dtype)
  # partial_det[:, -1] contains the full determinant and
  # partial_det[:, -2] contains det(u) / u_{nn}.
  partial_det = reductions.cumprod(diag, axis=-1) * sign[..., None]
  lu = lu.at[..., -1, -1].set(1.0 / partial_det[..., -2])
  permutation = jnp.broadcast_to(permutation, (*batch_dims, a_shape[-1]))
  iotas = jnp.ix_(*(lax.iota(jnp.int32, b) for b in (*batch_dims, 1)))
  # filter out any matrices that are not full rank
  d = jnp.ones(x.shape[:-1], x.dtype)
  d = lax_linalg.triangular_solve(lu, d, left_side=True, lower=False)
  d = reductions.any(ufuncs.logical_or(ufuncs.isnan(d), ufuncs.isinf(d)), axis=-1)
  d = jnp.tile(d[..., None, None], d.ndim*(1,) + x.shape[-2:])
  x = jnp.where(d, jnp.zeros_like(x), x)  # first filter
  x = x[iotas[:-1] + (permutation, slice(None))]
  x = lax_linalg.triangular_solve(lu, x, left_side=True, lower=True,
                                  unit_diagonal=True)
  x = jnp.concatenate((x[..., :-1, :] * partial_det[..., -1, None, None],
                      x[..., -1:, :]), axis=-2)
  x = lax_linalg.triangular_solve(lu, x, left_side=True, lower=False)
  x = jnp.where(d, jnp.zeros_like(x), x)  # second filter

  return partial_det[..., -1], x


def _det_2x2(a: Array) -> Array:
  return (a[..., 0, 0] * a[..., 1, 1] -
           a[..., 0, 1] * a[..., 1, 0])


def _det_3x3(a: Array) -> Array:
  return (a[..., 0, 0] * a[..., 1, 1] * a[..., 2, 2] +
          a[..., 0, 1] * a[..., 1, 2] * a[..., 2, 0] +
          a[..., 0, 2] * a[..., 1, 0] * a[..., 2, 1] -
          a[..., 0, 2] * a[..., 1, 1] * a[..., 2, 0] -
          a[..., 0, 0] * a[..., 1, 2] * a[..., 2, 1] -
          a[..., 0, 1] * a[..., 1, 0] * a[..., 2, 2])


@custom_jvp
def _det(a):
  sign, logdet = slogdet(a)
  return sign * ufuncs.exp(logdet).astype(sign.dtype)


@_det.defjvp
def _det_jvp(primals, tangents):
  x, = primals
  g, = tangents
  y, z = _cofactor_solve(x, g)
  return y, jnp.trace(z, axis1=-1, axis2=-2)


@export
@jit
def det(a: ArrayLike) -> Array:
  """
  Compute the determinant of an array.

  JAX implementation of :func:`numpy.linalg.det`.

  Args:
    a: array of shape ``(..., M, M)`` for which to compute the determinant.

  Returns:
    An array of determinants of shape ``a.shape[:-2]``.

  See also:
    :func:`jax.scipy.linalg.det`: Scipy-style API for determinant.

  Examples:
    >>> a = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> jnp.linalg.det(a)
    Array(-2., dtype=float32)
  """
  check_arraylike("jnp.linalg.det", a)
  a, = promote_dtypes_inexact(jnp.asarray(a))
  a_shape = jnp.shape(a)
  if len(a_shape) >= 2 and a_shape[-1] == 2 and a_shape[-2] == 2:
    return _det_2x2(a)
  elif len(a_shape) >= 2 and a_shape[-1] == 3 and a_shape[-2] == 3:
    return _det_3x3(a)
  elif len(a_shape) >= 2 and a_shape[-1] == a_shape[-2]:
    return _det(a)
  else:
    msg = "Argument to _det() must have shape [..., n, n], got {}"
    raise ValueError(msg.format(a_shape))


@export
def eig(a: ArrayLike) -> tuple[Array, Array]:
  """
  Compute the eigenvalues and eigenvectors of a square array.

  JAX implementation of :func:`numpy.linalg.eig`.

  Args:
    a: array of shape ``(..., M, M)`` for which to compute the eigenvalues and vectors.

  Returns:
    A tuple ``(eigenvalues, eigenvectors)`` with

    - ``eigenvalues``: an array of shape ``(..., M)`` containing the eigenvalues.
    - ``eigenvectors``: an array of shape ``(..., M, M)``, where column ``v[:, i]`` is the
      eigenvector corresponding to the eigenvalue ``w[i]``.

  Notes:
    - This differs from :func:`numpy.linalg.eig` in that the return type of
      :func:`jax.numpy.linalg.eig` is always complex64 for 32-bit input, and complex128
      for 64-bit input.
    - At present, non-symmetric eigendecomposition is only implemented on the CPU and
      GPU backends. For more details about the GPU implementation, see the
      documentation for :func:`jax.lax.linalg.eig`.

  See also:
    - :func:`jax.numpy.linalg.eigh`: eigenvectors and eigenvalues of a Hermitian matrix.
    - :func:`jax.numpy.linalg.eigvals`: compute eigenvalues only.

  Examples:
    >>> a = jnp.array([[1., 2.],
    ...                [2., 1.]])
    >>> w, v = jnp.linalg.eig(a)
    >>> with jax.numpy.printoptions(precision=4):
    ...   w
    Array([ 3.+0.j, -1.+0.j], dtype=complex64)
    >>> v
    Array([[ 0.70710677+0.j, -0.70710677+0.j],
           [ 0.70710677+0.j,  0.70710677+0.j]], dtype=complex64)
  """
  check_arraylike("jnp.linalg.eig", a)
  a, = promote_dtypes_inexact(jnp.asarray(a))
  w, v = lax_linalg.eig(a, compute_left_eigenvectors=False)
  return w, v


@export
@jit
def eigvals(a: ArrayLike) -> Array:
  """
  Compute the eigenvalues of a general matrix.

  JAX implementation of :func:`numpy.linalg.eigvals`.

  Args:
    a: array of shape ``(..., M, M)`` for which to compute the eigenvalues.

  Returns:
    An array of shape ``(..., M)`` containing the eigenvalues.

  See also:
    - :func:`jax.numpy.linalg.eig`: computes eigenvalues eigenvectors of a general matrix.
    - :func:`jax.numpy.linalg.eigh`: computes eigenvalues eigenvectors of a Hermitian matrix.

  Notes:
    - This differs from :func:`numpy.linalg.eigvals` in that the return type of
      :func:`jax.numpy.linalg.eigvals` is always complex64 for 32-bit input, and
      complex128 for 64-bit input.
    - At present, non-symmetric eigendecomposition is only implemented on the CPU backend.

  Examples:
    >>> a = jnp.array([[1., 2.],
    ...                [2., 1.]])
    >>> w = jnp.linalg.eigvals(a)
    >>> with jnp.printoptions(precision=2):
    ...  w
    Array([ 3.+0.j, -1.+0.j], dtype=complex64)
  """
  check_arraylike("jnp.linalg.eigvals", a)
  a, = promote_dtypes_inexact(jnp.asarray(a))
  return lax_linalg.eig(a, compute_left_eigenvectors=False,
                        compute_right_eigenvectors=False)[0]


@export
@partial(jit, static_argnames=('UPLO', 'symmetrize_input'))
def eigh(a: ArrayLike, UPLO: str | None = None,
         symmetrize_input: bool = True) -> EighResult:
  """
  Compute the eigenvalues and eigenvectors of a Hermitian matrix.

  JAX implementation of :func:`numpy.linalg.eigh`.

  Args:
    a: array of shape ``(..., M, M)``, containing the Hermitian (if complex)
      or symmetric (if real) matrix.
    UPLO: specifies whether the calculation is done with the lower triangular
      part of ``a`` (``'L'``, default) or the upper triangular part (``'U'``).
    symmetrize_input: if True (default) then input is symmetrized, which leads
      to better behavior under automatic differentiation.

  Returns:
    A namedtuple ``(eigenvalues, eigenvectors)`` where

    - ``eigenvalues``: an array of shape ``(..., M)`` containing the eigenvalues,
      sorted in ascending order.
    - ``eigenvectors``: an array of shape ``(..., M, M)``, where column ``v[:, i]`` is the
      normalized eigenvector corresponding to the eigenvalue ``w[i]``.

  See also:
    - :func:`jax.numpy.linalg.eig`: general eigenvalue decomposition.
    - :func:`jax.numpy.linalg.eigvalsh`: compute eigenvalues only.
    - :func:`jax.scipy.linalg.eigh`: SciPy API for Hermitian eigendecomposition.
    - :func:`jax.lax.linalg.eigh`: XLA API for Hermitian eigendecomposition.

  Examples:
    >>> a = jnp.array([[1, -2j],
    ...                [2j, 1]])
    >>> w, v = jnp.linalg.eigh(a)
    >>> w
    Array([-1.,  3.], dtype=float32)
    >>> with jnp.printoptions(precision=3):
    ...   v
    Array([[-0.707+0.j   , -0.707+0.j   ],
           [ 0.   +0.707j,  0.   -0.707j]], dtype=complex64)
  """
  check_arraylike("jnp.linalg.eigh", a)
  if UPLO is None or UPLO == "L":
    lower = True
  elif UPLO == "U":
    lower = False
  else:
    msg = f"UPLO must be one of None, 'L', or 'U', got {UPLO}"
    raise ValueError(msg)

  a, = promote_dtypes_inexact(jnp.asarray(a))
  v, w = lax_linalg.eigh(a, lower=lower, symmetrize_input=symmetrize_input)
  return EighResult(w, v)


@export
@partial(jit, static_argnames=('UPLO',))
def eigvalsh(a: ArrayLike, UPLO: str | None = 'L') -> Array:
  """
  Compute the eigenvalues of a Hermitian matrix.

  JAX implementation of :func:`numpy.linalg.eigvalsh`.

  Args:
    a: array of shape ``(..., M, M)``, containing the Hermitian (if complex)
      or symmetric (if real) matrix.
    UPLO: specifies whether the calculation is done with the lower triangular
      part of ``a`` (``'L'``, default) or the upper triangular part (``'U'``).

  Returns:
    An array of shape ``(..., M)`` containing the eigenvalues, sorted in
    ascending order.

  See also:
    - :func:`jax.numpy.linalg.eig`: general eigenvalue decomposition.
    - :func:`jax.numpy.linalg.eigh`: computes eigenvalues and eigenvectors of a
      Hermitian matrix.

  Examples:
    >>> a = jnp.array([[1, -2j],
    ...                [2j, 1]])
    >>> w = jnp.linalg.eigvalsh(a)
    >>> w
    Array([-1.,  3.], dtype=float32)
  """
  check_arraylike("jnp.linalg.eigvalsh", a)
  a, = promote_dtypes_inexact(jnp.asarray(a))
  w, _ = eigh(a, UPLO)
  return w


# TODO(micky774): deprecated 2024-5-14, remove wrapper after deprecation expires.
@export
def pinv(a: ArrayLike, rtol: ArrayLike | None = None,
         hermitian: bool = False, *,
         rcond: ArrayLike | DeprecatedArg | None = DeprecatedArg()) -> Array:
  """Compute the (Moore-Penrose) pseudo-inverse of a matrix.

  JAX implementation of :func:`numpy.linalg.pinv`.

  Args:
    a: array of shape ``(..., M, N)`` containing matrices to pseudo-invert.
    rtol: float or array_like of shape ``a.shape[:-2]``. Specifies the cutoff
      for small singular values.of shape ``(...,)``.
      Cutoff for small singular values; singular values smaller
      ``rtol * largest_singular_value`` are treated as zero. The default is
      determined based on the floating point precision of the dtype.
    hermitian: if True, then the input is assumed to be Hermitian, and a more
      efficient algorithm is used (default: False)
    rcond: deprecated alias of the ``rtol`` argument. Will result in a
      :class:`DeprecationWarning` if used.

  Returns:
    An array of shape ``(..., N, M)`` containing the pseudo-inverse of ``a``.

  See also:
    - :func:`jax.numpy.linalg.inv`: multiplicative inverse of a square matrix.

  Notes:
    :func:`jax.numpy.linalg.pinv` differs from :func:`numpy.linalg.pinv` in the
    default value of `rcond``: in NumPy, the default  is `1e-15`. In JAX, the
    default is ``10. * max(num_rows, num_cols) * jnp.finfo(dtype).eps``.

  Examples:
    >>> a = jnp.array([[1, 2],
    ...                [3, 4],
    ...                [5, 6]])
    >>> a_pinv = jnp.linalg.pinv(a)
    >>> a_pinv  # doctest: +SKIP
    Array([[-1.333332  , -0.33333257,  0.6666657 ],
           [ 1.0833322 ,  0.33333272, -0.41666582]], dtype=float32)

    The pseudo-inverse operates as a multiplicative inverse so long as the
    output is not rank-deficient:

    >>> jnp.allclose(a_pinv @ a, jnp.eye(2), atol=1E-4)
    Array(True, dtype=bool)
  """
  if not isinstance(rcond, DeprecatedArg):
    rtol = rcond
    del rcond
    deprecations.warn(
      "jax-numpy-linalg-pinv-rcond",
      ("The rcond argument for linalg.pinv is deprecated. "
       "Please use rtol instead."),
       stacklevel=2
    )

  return _pinv(a, rtol, hermitian)


@partial(custom_jvp, nondiff_argnums=(1, 2))
@partial(jit, static_argnames=('hermitian'))
def _pinv(a: ArrayLike, rtol: ArrayLike | None = None, hermitian: bool = False) -> Array:
  # Uses same algorithm as
  # https://github.com/numpy/numpy/blob/v1.17.0/numpy/linalg/linalg.py#L1890-L1979
  check_arraylike("jnp.linalg.pinv", a)
  arr, = promote_dtypes_inexact(jnp.asarray(a))
  m, n = arr.shape[-2:]
  if m == 0 or n == 0:
    return jnp.empty(arr.shape[:-2] + (n, m), arr.dtype)
  arr = ufuncs.conj(arr)
  if rtol is None:
    max_rows_cols = max(arr.shape[-2:])
    rtol = 10. * max_rows_cols * jnp.array(jnp.finfo(arr.dtype).eps)
  rtol = jnp.asarray(rtol)
  u, s, vh = svd(arr, full_matrices=False, hermitian=hermitian)
  # Singular values less than or equal to ``rtol * largest_singular_value``
  # are set to zero.
  rtol = lax.expand_dims(rtol[..., jnp.newaxis], range(s.ndim - rtol.ndim - 1))
  cutoff = rtol * s[..., 0:1]
  s = jnp.where(s > cutoff, s, jnp.inf).astype(u.dtype)
  res = jnp.matmul(vh.mT, ufuncs.divide(u.mT, s[..., jnp.newaxis]),
                   precision=lax.Precision.HIGHEST)
  return lax.convert_element_type(res, arr.dtype)


@_pinv.defjvp
@jax.default_matmul_precision("float32")
def _pinv_jvp(rtol, hermitian, primals, tangents):
  # The Differentiation of Pseudo-Inverses and Nonlinear Least Squares Problems
  # Whose Variables Separate. Author(s): G. H. Golub and V. Pereyra. SIAM
  # Journal on Numerical Analysis, Vol. 10, No. 2 (Apr., 1973), pp. 413-432.
  # (via https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Derivative)
  a, = primals  # m x n
  a_dot, = tangents
  p = pinv(a, rtol=rtol, hermitian=hermitian)  # n x m
  if hermitian:
    # svd(..., hermitian=True) symmetrizes its input, and the JVP must match.
    a = _symmetrize(a)
    a_dot = _symmetrize(a_dot)

  # TODO(phawkins): this could be simplified in the Hermitian case if we
  # supported triangular matrix multiplication.
  m, n = a.shape[-2:]
  if m >= n:
    s = (p @ _H(p)) @ _H(a_dot)  # nxm
    t = (_H(a_dot) @ _H(p)) @ p  # nxm
    p_dot = -(p @ a_dot) @ p + s - (s @ a) @ p + t - (p @ a) @ t
  else:  # m < n
    s = p @ (_H(p) @ _H(a_dot))
    t = _H(a_dot) @ (_H(p) @ p)
    p_dot = -p @ (a_dot @ p) + s - s @ (a @ p) + t - p @ (a @ t)
  return p, p_dot


@export
@jit
def inv(a: ArrayLike) -> Array:
  """Return the inverse of a square matrix

  JAX implementation of :func:`numpy.linalg.inv`.

  Args:
    a: array of shape ``(..., N, N)`` specifying square array(s) to be inverted.

  Returns:
    Array of shape ``(..., N, N)`` containing the inverse of the input.

  Notes:
    In most cases, explicitly computing the inverse of a matrix is ill-advised. For
    example, to compute ``x = inv(A) @ b``, it is more performant and numerically
    precise to use a direct solve, such as :func:`jax.scipy.linalg.solve`.

  See Also:
    - :func:`jax.scipy.linalg.inv`: SciPy-style API for matrix inverse
    - :func:`jax.numpy.linalg.solve`: direct linear solver

  Examples:
    Compute the inverse of a 3x3 matrix

    >>> a = jnp.array([[1., 2., 3.],
    ...                [2., 4., 2.],
    ...                [3., 2., 1.]])
    >>> a_inv = jnp.linalg.inv(a)
    >>> a_inv  # doctest: +SKIP
    Array([[ 0.        , -0.25      ,  0.5       ],
           [-0.25      ,  0.5       , -0.25000003],
           [ 0.5       , -0.25      ,  0.        ]], dtype=float32)

    Check that multiplying with the inverse gives the identity:

    >>> jnp.allclose(a @ a_inv, jnp.eye(3), atol=1E-5)
    Array(True, dtype=bool)

    Multiply the inverse by a vector ``b``, to find a solution to ``a @ x = b``

    >>> b = jnp.array([1., 4., 2.])
    >>> a_inv @ b
    Array([ 0.  ,  1.25, -0.5 ], dtype=float32)

    Note, however, that explicitly computing the inverse in such a case can lead
    to poor performance and loss of precision as the size of the problem grows.
    Instead, you should use a direct solver like :func:`jax.numpy.linalg.solve`:

    >>> jnp.linalg.solve(a, b)
     Array([ 0.  ,  1.25, -0.5 ], dtype=float32)
  """
  check_arraylike("jnp.linalg.inv", a)
  arr = jnp.asarray(a)
  if arr.ndim < 2 or arr.shape[-1] != arr.shape[-2]:
    raise ValueError(
      f"Argument to inv must have shape [..., n, n], got {arr.shape}.")
  return solve(
    arr, lax.broadcast(jnp.eye(arr.shape[-1], dtype=arr.dtype), arr.shape[:-2]))


@export
@partial(jit, static_argnames=('ord', 'axis', 'keepdims'))
def norm(x: ArrayLike, ord: int | str | None = None,
         axis: None | tuple[int, ...] | int = None,
         keepdims: bool = False) -> Array:
  """Compute the norm of a matrix or vector.

  JAX implementation of :func:`numpy.linalg.norm`.

  Args:
    x: N-dimensional array for which the norm will be computed.
    ord: specify the kind of norm to take. Default is Frobenius norm for matrices,
      and the 2-norm for vectors. For other options, see Notes below.
    axis: integer or sequence of integers specifying the axes over which the norm
      will be computed. Defaults to all axes of ``x``.
    keepdims: if True, the output array will have the same number of dimensions as
      the input, with the size of reduced axes replaced by ``1`` (default: False).

  Returns:
    array containing the specified norm of x.

  Notes:
    The flavor of norm computed depends on the value of ``ord`` and the number of
    axes being reduced.

    For **vector norms** (i.e. a single axis reduction):

    - ``ord=None`` (default) computes the 2-norm
    - ``ord=inf`` computes ``max(abs(x))``
    - ``ord=-inf`` computes min(abs(x))``
    - ``ord=0`` computes ``sum(x!=0)``
    - for other numerical values, computes ``sum(abs(x) ** ord)**(1/ord)``

    For **matrix norms** (i.e. two axes reductions):

    - ``ord='fro'`` or ``ord=None`` (default) computes the Frobenius norm
    - ``ord='nuc'`` computes the nuclear norm, or the sum of the singular values
    - ``ord=1`` computes ``max(abs(x).sum(0))``
    - ``ord=-1`` computes ``min(abs(x).sum(0))``
    - ``ord=2`` computes the 2-norm, i.e. the largest singular value
    - ``ord=-2`` computes the smallest singular value

  Examples:
    Vector norms:

    >>> x = jnp.array([3., 4., 12.])
    >>> jnp.linalg.norm(x)
    Array(13., dtype=float32)
    >>> jnp.linalg.norm(x, ord=1)
    Array(19., dtype=float32)
    >>> jnp.linalg.norm(x, ord=0)
    Array(3., dtype=float32)

    Matrix norms:

    >>> x = jnp.array([[1., 2., 3.],
    ...                [4., 5., 7.]])
    >>> jnp.linalg.norm(x)  # Frobenius norm
    Array(10.198039, dtype=float32)
    >>> jnp.linalg.norm(x, ord='nuc')  # nuclear norm
    Array(10.762535, dtype=float32)
    >>> jnp.linalg.norm(x, ord=1)  # 1-norm
    Array(10., dtype=float32)

    Batched vector norm:

    >>> jnp.linalg.norm(x, axis=1)
    Array([3.7416575, 9.486833 ], dtype=float32)
  """
  check_arraylike("jnp.linalg.norm", x)
  x, = promote_dtypes_inexact(jnp.asarray(x))
  x_shape = jnp.shape(x)
  ndim = len(x_shape)

  if axis is None:
    # NumPy has an undocumented behavior that admits arbitrary rank inputs if
    # `ord` is None: https://github.com/numpy/numpy/issues/14215
    if ord is None:
      return ufuncs.sqrt(reductions.sum(ufuncs.real(x * ufuncs.conj(x)), keepdims=keepdims))
    axis = tuple(range(ndim))
  elif isinstance(axis, tuple):
    axis = tuple(canonicalize_axis(x, ndim) for x in axis)
  else:
    axis = (canonicalize_axis(axis, ndim),)

  num_axes = len(axis)
  if num_axes == 1:
    return vector_norm(x, ord=2 if ord is None else ord, axis=axis, keepdims=keepdims)

  elif num_axes == 2:
    row_axis, col_axis = axis  # pytype: disable=bad-unpacking
    if ord is None or ord in ('f', 'fro'):
      return ufuncs.sqrt(reductions.sum(ufuncs.real(x * ufuncs.conj(x)), axis=axis,
                                        keepdims=keepdims))
    elif ord == 1:
      if not keepdims and col_axis > row_axis:
        col_axis -= 1
      return reductions.amax(reductions.sum(ufuncs.abs(x), axis=row_axis, keepdims=keepdims),
                             axis=col_axis, keepdims=keepdims)
    elif ord == -1:
      if not keepdims and col_axis > row_axis:
        col_axis -= 1
      return reductions.amin(reductions.sum(ufuncs.abs(x), axis=row_axis, keepdims=keepdims),
                             axis=col_axis, keepdims=keepdims)
    elif ord == jnp.inf:
      if not keepdims and row_axis > col_axis:
        row_axis -= 1
      return reductions.amax(reductions.sum(ufuncs.abs(x), axis=col_axis, keepdims=keepdims),
                     axis=row_axis, keepdims=keepdims)
    elif ord == -jnp.inf:
      if not keepdims and row_axis > col_axis:
        row_axis -= 1
      return reductions.amin(reductions.sum(ufuncs.abs(x), axis=col_axis, keepdims=keepdims),
                     axis=row_axis, keepdims=keepdims)
    elif ord in ('nuc', 2, -2):
      x = jnp.moveaxis(x, axis, (-2, -1))
      if ord == 2:
        reducer = reductions.amax
      elif ord == -2:
        reducer = reductions.amin
      else:
        # `sum` takes an extra dtype= argument, unlike `amax` and `amin`.
        reducer = reductions.sum  # type: ignore[assignment]
      y = reducer(svd(x, compute_uv=False), axis=-1)
      if keepdims:
        y = jnp.expand_dims(y, axis)
      return y
    else:
      raise ValueError(f"Invalid order '{ord}' for matrix norm.")
  else:
    raise ValueError(
        f"Invalid axis values ({axis}) for jnp.linalg.norm.")

@overload
def qr(a: ArrayLike, mode: Literal["r"]) -> Array: ...
@overload
def qr(a: ArrayLike, mode: str = "reduced") -> Array | QRResult: ...

@export
@partial(jit, static_argnames=('mode',))
def qr(a: ArrayLike, mode: str = "reduced") -> Array | QRResult:
  """Compute the QR decomposition of an array

  JAX implementation of :func:`numpy.linalg.qr`.

  The QR decomposition of a matrix `A` is given by

  .. math::

     A = QR

  Where `Q` is a unitary matrix (i.e. :math:`Q^HQ=I`) and `R` is an upper-triangular
  matrix.

  Args:
    a: array of shape (..., M, N)
    mode: Computational mode. Supported values are:

      - ``"reduced"`` (default): return `Q` of shape ``(..., M, K)`` and `R` of shape
        ``(..., K, N)``, where ``K = min(M, N)``.
      - ``"complete"``: return `Q` of shape ``(..., M, M)`` and `R` of shape ``(..., M, N)``.
      - ``"raw"``: return lapack-internal representations of shape ``(..., M, N)`` and ``(..., K)``.
      - ``"r"``: return `R` only.

  Returns:
    A tuple ``(Q, R)`` (if ``mode`` is not ``"r"``) otherwise an array ``R``,
    where:

    - ``Q`` is an orthogonal matrix of shape ``(..., M, K)`` (if ``mode`` is ``"reduced"``)
      or ``(..., M, M)`` (if ``mode`` is ``"complete"``).
    - ``R`` is an upper-triangular matrix of shape ``(..., M, N)`` (if ``mode`` is
      ``"r"`` or ``"complete"``) or ``(..., K, N)`` (if ``mode`` is ``"reduced"``)

    with ``K = min(M, N)``.

  See also:
    - :func:`jax.scipy.linalg.qr`: SciPy-style QR decomposition API
    - :func:`jax.lax.linalg.qr`: XLA-style QR decomposition API

  Examples:
    Compute the QR decomposition of a matrix:

    >>> a = jnp.array([[1., 2., 3., 4.],
    ...                [5., 4., 2., 1.],
    ...                [6., 3., 1., 5.]])
    >>> Q, R = jnp.linalg.qr(a)
    >>> Q  # doctest: +SKIP
    Array([[-0.12700021, -0.7581426 , -0.6396022 ],
           [-0.63500065, -0.43322435,  0.63960224],
           [-0.7620008 ,  0.48737738, -0.42640156]], dtype=float32)
    >>> R  # doctest: +SKIP
    Array([[-7.8740077, -5.080005 , -2.4130025, -4.953006 ],
           [ 0.       , -1.7870499, -2.6534991, -1.028908 ],
           [ 0.       ,  0.       , -1.0660033, -4.050814 ]], dtype=float32)

    Check that ``Q`` is orthonormal:

    >>> jnp.allclose(Q.T @ Q, jnp.eye(3), atol=1E-5)
    Array(True, dtype=bool)

    Reconstruct the input:

    >>> jnp.allclose(Q @ R, a)
    Array(True, dtype=bool)
  """
  check_arraylike("jnp.linalg.qr", a)
  a, = promote_dtypes_inexact(jnp.asarray(a))
  if mode == "raw":
    a, taus = lax_linalg.geqrf(a)
    return QRResult(a.mT, taus)
  if mode in ("reduced", "r", "full"):
    full_matrices = False
  elif mode == "complete":
    full_matrices = True
  else:
    raise ValueError(f"Unsupported QR decomposition mode '{mode}'")
  q, r = lax_linalg.qr(a, pivoting=False, full_matrices=full_matrices)
  if mode == "r":
    return r
  return QRResult(q, r)


@export
@jit
def solve(a: ArrayLike, b: ArrayLike) -> Array:
  """Solve a linear system of equations

  JAX implementation of :func:`numpy.linalg.solve`.

  This solves a (batched) linear system of equations ``a @ x = b``
  for ``x`` given ``a`` and ``b``.

  Args:
    a: array of shape ``(..., N, N)``.
    b: array of shape ``(N,)`` (for 1-dimensional right-hand-side) or
      ``(..., N, M)`` (for batched 2-dimensional right-hand-side).

  Returns:
    An array containing the result of the linear solve. The result has shape ``(..., N)``
    if ``b`` is of shape ``(N,)``, and has shape ``(..., N, M)`` otherwise.

  See also:
    - :func:`jax.scipy.linalg.solve`: SciPy-style API for solving linear systems.
    - :func:`jax.lax.custom_linear_solve`: matrix-free linear solver.

  Examples:
    A simple 3x3 linear system:

    >>> A = jnp.array([[1., 2., 3.],
    ...                [2., 4., 2.],
    ...                [3., 2., 1.]])
    >>> b = jnp.array([14., 16., 10.])
    >>> x = jnp.linalg.solve(A, b)
    >>> x
    Array([1., 2., 3.], dtype=float32)

    Confirming that the result solves the system:

    >>> jnp.allclose(A @ x, b)
    Array(True, dtype=bool)
  """
  check_arraylike("jnp.linalg.solve", a, b)
  a, b = promote_dtypes_inexact(jnp.asarray(a), jnp.asarray(b))

  if a.ndim < 2:
    raise ValueError(
      f"left hand array must be at least two dimensional; got {a.shape=}")

  # Check for invalid inputs that previously would have led to a batched 1D solve:
  if (b.ndim > 1 and a.ndim == b.ndim + 1 and
      a.shape[-1] == b.shape[-1] and a.shape[-1] != b.shape[-2]):
    raise ValueError(
      f"Invalid shapes for solve: {a.shape}, {b.shape}. Prior to JAX v0.5.0,"
      " this would have been treated as a batched 1-dimensional solve."
      " To recover this behavior, use solve(a, b[..., None]).squeeze(-1).")

  signature = "(m,m),(m)->(m)" if b.ndim == 1 else "(m,m),(m,n)->(m,n)"
  return jnp.vectorize(lax_linalg._solve, signature=signature)(a, b)


def _lstsq(a: ArrayLike, b: ArrayLike, rcond: float | None, *,
           numpy_resid: bool = False) -> tuple[Array, Array, Array, Array]:
  # TODO: add lstsq to lax_linalg and implement this function via those wrappers.
  # TODO: add custom jvp rule for more robust lstsq differentiation
  a, b = promote_dtypes_inexact(a, b)
  if a.shape[0] != b.shape[0]:
    raise ValueError("Leading dimensions of input arrays must match")
  b_orig_ndim = b.ndim
  if b_orig_ndim == 1:
    b = b[:, None]
  if a.ndim != 2:
    raise TypeError(
      f"{a.ndim}-dimensional array given. Array must be two-dimensional")
  if b.ndim != 2:
    raise TypeError(
      f"{b.ndim}-dimensional array given. Array must be one or two-dimensional")
  m, n = a.shape
  dtype = a.dtype
  if a.size == 0:
    s = jnp.empty(0, dtype=a.dtype)
    rank = jnp.array(0, dtype=int)
    x = jnp.empty((n, *b.shape[1:]), dtype=a.dtype)
  else:
    if rcond is None:
      rcond = float(jnp.finfo(dtype).eps) * max(n, m)
    else:
      rcond = jnp.where(rcond < 0, jnp.finfo(dtype).eps, rcond)
    u, s, vt = svd(a, full_matrices=False)
    mask = s >= jnp.array(rcond, dtype=s.dtype) * s[0]
    rank = mask.sum()
    safe_s = jnp.where(mask, s, 1).astype(a.dtype)
    s_inv = jnp.where(mask, 1 / safe_s, 0)[:, jnp.newaxis]
    uTb = jnp.matmul(u.conj().T, b, precision=lax.Precision.HIGHEST)
    x = jnp.matmul(vt.conj().T, s_inv * uTb, precision=lax.Precision.HIGHEST)
  # Numpy returns empty residuals in some cases. To allow compilation, we
  # default to returning full residuals in all cases.
  if numpy_resid and (rank < n or m <= n):
    resid = jnp.asarray([])
  else:
    b_estimate = jnp.matmul(a, x, precision=lax.Precision.HIGHEST)
    resid = norm(b - b_estimate, axis=0) ** 2
  if b_orig_ndim == 1:
    x = x.ravel()
  return x, resid, rank, s

_jit_lstsq = jit(partial(_lstsq, numpy_resid=False))


@export
def lstsq(a: ArrayLike, b: ArrayLike, rcond: float | None = None, *,
          numpy_resid: bool = False) -> tuple[Array, Array, Array, Array]:
  """
  Return the least-squares solution to a linear equation.

  JAX implementation of :func:`numpy.linalg.lstsq`.

  Args:
    a: array of shape ``(M, N)`` representing the coefficient matrix.
    b: array of shape ``(M,)`` or ``(M, K)`` representing the right-hand side.
    rcond: Cut-off ratio for small singular values. Singular values smaller than
      ``rcond * largest_singular_value`` are treated as zero. If None (default),
      the optimal value will be used to reduce floating point errors.
    numpy_resid: If True, compute and return residuals in the same way as NumPy's
      `linalg.lstsq`. This is necessary if you want to precisely replicate NumPy's
      behavior. If False (default), a more efficient method is used to compute residuals.

  Returns:
    Tuple of arrays ``(x, resid, rank, s)`` where

    - ``x`` is a shape ``(N,)`` or ``(N, K)`` array containing the least-squares solution.
    - ``resid`` is the sum of squared residual of shape ``()`` or ``(K,)``.
    - ``rank`` is the rank of the matrix ``a``.
    - ``s`` is the singular values of the matrix ``a``.

  Examples:
    >>> a = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> b = jnp.array([5, 6])
    >>> x, _, _, _ = jnp.linalg.lstsq(a, b)
    >>> with jnp.printoptions(precision=3):
    ...   print(x)
    [-4.   4.5]
  """
  check_arraylike("jnp.linalg.lstsq", a, b)
  if numpy_resid:
    return _lstsq(a, b, rcond, numpy_resid=True)
  return _jit_lstsq(a, b, rcond)


@export
def cross(x1: ArrayLike, x2: ArrayLike, /, *, axis=-1):
  r"""Compute the cross-product of two 3D vectors

  JAX implementation of :func:`numpy.linalg.cross`

  Args:
    x1: N-dimensional array, with ``x1.shape[axis] == 3``
    x2: N-dimensional array, with ``x2.shape[axis] == 3``, and other axes
      broadcast-compatible with ``x1``.
    axis: axis along which to take the cross product (default: -1).

  Returns:
    array containing the result of the cross-product

  See Also:
    :func:`jax.numpy.cross`: more flexible cross-product API.

  Examples:

    Showing that :math:`\hat{x} \times \hat{y} = \hat{z}`:

    >>> x = jnp.array([1., 0., 0.])
    >>> y = jnp.array([0., 1., 0.])
    >>> jnp.linalg.cross(x, y)
    Array([0., 0., 1.], dtype=float32)

    Cross product of :math:`\hat{x}` with all three standard unit vectors,
    via broadcasting:

    >>> xyz = jnp.eye(3)
    >>> jnp.linalg.cross(x, xyz, axis=-1)
    Array([[ 0.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0., -1.,  0.]], dtype=float32)
  """
  check_arraylike("jnp.linalg.outer", x1, x2)
  x1, x2 = jnp.asarray(x1), jnp.asarray(x2)
  if x1.shape[axis] != 3 or x2.shape[axis] != 3:
    raise ValueError(
        "Both input arrays must be (arrays of) 3-dimensional vectors, "
        f"but they have {x1.shape[axis]=} and {x2.shape[axis]=}"
    )
  return jnp.cross(x1, x2, axis=axis)


@export
def outer(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  """Compute the outer product of two 1-dimensional arrays.

  JAX implementation of :func:`numpy.linalg.outer`.

  Args:
    x1: array
    x2: array

  Returns:
    array containing the outer product of ``x1`` and ``x2``

  See also:
    :func:`jax.numpy.outer`: similar function in the main :mod:`jax.numpy` module.

  Examples:
    >>> x1 = jnp.array([1, 2, 3])
    >>> x2 = jnp.array([4, 5, 6])
    >>> jnp.linalg.outer(x1, x2)
    Array([[ 4,  5,  6],
           [ 8, 10, 12],
           [12, 15, 18]], dtype=int32)
  """
  check_arraylike("jnp.linalg.outer", x1, x2)
  x1, x2 = jnp.asarray(x1), jnp.asarray(x2)
  if x1.ndim != 1 or x2.ndim != 1:
    raise ValueError(f"Input arrays must be one-dimensional, but they are {x1.ndim=} {x2.ndim=}")
  return x1[:, None] * x2[None, :]


@export
def matrix_norm(x: ArrayLike, /, *, keepdims: bool = False, ord: str | int = 'fro') -> Array:
  """Compute the norm of a matrix or stack of matrices.

  JAX implementation of :func:`numpy.linalg.matrix_norm`

  Args:
    x: array of shape ``(..., M, N)`` for which to take the norm.
    keepdims: if True, keep the reduced dimensions in the output.
    ord: A string or int specifying the type of norm; default is the Frobenius norm.
      See :func:`numpy.linalg.norm` for details on available options.

  Returns:
    array containing the norm of ``x``. Has shape ``x.shape[:-2]`` if ``keepdims`` is
    False, or shape ``(..., 1, 1)`` if ``keepdims`` is True.

  See also:
    - :func:`jax.numpy.linalg.vector_norm`: Norm of a vector or stack of vectors.
    - :func:`jax.numpy.linalg.norm`: More general matrix or vector norm.

  Examples:
    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6],
    ...                [7, 8, 9]])
    >>> jnp.linalg.matrix_norm(x)
    Array(16.881943, dtype=float32)
  """
  check_arraylike('jnp.linalg.matrix_norm', x)
  return norm(x, ord=ord, keepdims=keepdims, axis=(-2, -1))


@export
def matrix_transpose(x: ArrayLike, /) -> Array:
  """Transpose a matrix or stack of matrices.

  JAX implementation of :func:`numpy.linalg.matrix_transpose`.

  Args:
    x: array of shape ``(..., M, N)``

  Returns:
    array of shape ``(..., N, M)`` containing the matrix transpose of ``x``.

  See also:
    :func:`jax.numpy.transpose`: more general transpose operation.

  Examples:
    Transpose of a single matrix:

    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> jnp.linalg.matrix_transpose(x)
    Array([[1, 4],
           [2, 5],
           [3, 6]], dtype=int32)

    Transpose of a stack of matrices:

    >>> x = jnp.array([[[1, 2],
    ...                 [3, 4]],
    ...                [[5, 6],
    ...                 [7, 8]]])
    >>> jnp.linalg.matrix_transpose(x)
    Array([[[1, 3],
            [2, 4]],
    <BLANKLINE>
           [[5, 7],
            [6, 8]]], dtype=int32)

    For convenience, the same computation can be done via the
    :attr:`~jax.Array.mT` property of JAX array objects:

    >>> x.mT
    Array([[[1, 3],
            [2, 4]],
    <BLANKLINE>
           [[5, 7],
            [6, 8]]], dtype=int32)
  """
  check_arraylike('jnp.linalg.matrix_transpose', x)
  x_arr = jnp.asarray(x)
  ndim = x_arr.ndim
  if ndim < 2:
    raise ValueError(f"matrix_transpose requres at least 2 dimensions; got {ndim=}")
  return jax.lax.transpose(x_arr, (*range(ndim - 2), ndim - 1, ndim - 2))


@export
def vector_norm(x: ArrayLike, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False,
                ord: int | str = 2) -> Array:
  """Compute the vector norm of a vector or batch of vectors.

  JAX implementation of :func:`numpy.linalg.vector_norm`.

  Args:
    x: N-dimensional array for which to take the norm.
    axis: optional axis along which to compute the vector norm. If None (default)
      then ``x`` is flattened and the norm is taken over all values.
    keepdims: if True, keep the reduced dimensions in the output.
    ord: A string or int specifying the type of norm; default is the 2-norm.
      See :func:`numpy.linalg.norm` for details on available options.

  Returns:
    array containing the norm of ``x``.

  See also:
    - :func:`jax.numpy.linalg.matrix_norm`: Norm of a matrix or stack of matrices.
    - :func:`jax.numpy.linalg.norm`: More general matrix or vector norm.

  Examples:
    Norm of a single vector:

    >>> x = jnp.array([1., 2., 3.])
    >>> jnp.linalg.vector_norm(x)
    Array(3.7416575, dtype=float32)

    Norm of a batch of vectors:

    >>> x = jnp.array([[1., 2., 3.],
    ...                [4., 5., 7.]])
    >>> jnp.linalg.vector_norm(x, axis=1)
    Array([3.7416575, 9.486833 ], dtype=float32)
  """
  check_arraylike('jnp.linalg.vector_norm', x)
  if ord is None or ord == 2:
    return ufuncs.sqrt(reductions.sum(ufuncs.real(x * ufuncs.conj(x)), axis=axis,
                                      keepdims=keepdims))
  elif ord == jnp.inf:
    return reductions.amax(ufuncs.abs(x), axis=axis, keepdims=keepdims)
  elif ord == -jnp.inf:
    return reductions.amin(ufuncs.abs(x), axis=axis, keepdims=keepdims)
  elif ord == 0:
    return reductions.sum(x != 0, dtype=jnp.finfo(lax.dtype(x)).dtype,
                          axis=axis, keepdims=keepdims)
  elif ord == 1:
    # Numpy has a special case for ord == 1 as an optimization. We don't
    # really need the optimization (XLA could do it for us), but the Numpy
    # code has slightly different type promotion semantics, so we need a
    # special case too.
    return reductions.sum(ufuncs.abs(x), axis=axis, keepdims=keepdims)
  elif isinstance(ord, str):
    msg = f"Invalid order '{ord}' for vector norm."
    if ord == "inf":
      msg += "Use 'jax.numpy.inf' instead."
    if ord == "-inf":
      msg += "Use '-jax.numpy.inf' instead."
    raise ValueError(msg)
  else:
    abs_x = ufuncs.abs(x)
    ord_arr = lax_internal._const(abs_x, ord)
    ord_inv = lax_internal._const(abs_x, 1. / ord_arr)
    out = reductions.sum(abs_x ** ord_arr, axis=axis, keepdims=keepdims)
    return ufuncs.power(out, ord_inv)

@export
def vecdot(x1: ArrayLike, x2: ArrayLike, /, *, axis: int = -1,
           precision: PrecisionLike = None,
           preferred_element_type: DTypeLike | None = None) -> Array:
  """Compute the (batched) vector conjugate dot product of two arrays.

  JAX implementation of :func:`numpy.linalg.vecdot`.

  Args:
    x1: left-hand side array.
    x2: right-hand side array. Size of ``x2[axis]`` must match size of ``x1[axis]``,
      and remaining dimensions must be broadcast-compatible.
    axis: axis along which to compute the dot product (default: -1)
    precision: either ``None`` (default), which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      such values indicating precision of ``x1`` and ``x2``.
    preferred_element_type: either ``None`` (default), which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    array containing the conjugate dot product of ``x1`` and ``x2`` along ``axis``.
    The non-contracted dimensions are broadcast together.

  See also:
    - :func:`jax.numpy.vecdot`: similar API in the ``jax.numpy`` namespace.
    - :func:`jax.numpy.linalg.matmul`: matrix multiplication.
    - :func:`jax.numpy.linalg.tensordot`: general tensor dot product.

  Examples:
    Vector dot product of two 1D arrays:

    >>> x1 = jnp.array([1, 2, 3])
    >>> x2 = jnp.array([4, 5, 6])
    >>> jnp.linalg.vecdot(x1, x2)
    Array(32, dtype=int32)

    Batched vector dot product of two 2D arrays:

    >>> x1 = jnp.array([[1, 2, 3],
    ...                 [4, 5, 6]])
    >>> x2 = jnp.array([[2, 3, 4]])
    >>> jnp.linalg.vecdot(x1, x2, axis=-1)
    Array([20, 47], dtype=int32)
  """
  check_arraylike('jnp.linalg.vecdot', x1, x2)
  return jnp.vecdot(x1, x2, axis=axis, precision=precision,
                    preferred_element_type=preferred_element_type)


@export
def matmul(x1: ArrayLike, x2: ArrayLike, /, *,
           precision: PrecisionLike = None,
           preferred_element_type: DTypeLike | None = None) -> Array:
  """Perform a matrix multiplication.

  JAX implementation of :func:`numpy.linalg.matmul`.

  Args:
    x1: first input array, of shape ``(..., N)``.
    x2: second input array. Must have shape ``(N,)`` or ``(..., N, M)``.
      In the multi-dimensional case, leading dimensions must be broadcast-compatible
      with the leading dimensions of ``x1``.
    precision: either ``None`` (default), which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      such values indicating precision of ``x1`` and ``x2``.
    preferred_element_type: either ``None`` (default), which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    array containing the matrix product of the inputs. Shape is ``x1.shape[:-1]``
    if ``x2.ndim == 1``, otherwise the shape is ``(..., M)``.

  See Also:
    :func:`jax.numpy.matmul`: NumPy API for this function.
    :func:`jax.numpy.linalg.vecdot`: batched vector product.
    :func:`jax.numpy.linalg.tensordot`: batched tensor product.

  Examples:
    Vector dot products:

    >>> x1 = jnp.array([1, 2, 3])
    >>> x2 = jnp.array([4, 5, 6])
    >>> jnp.linalg.matmul(x1, x2)
    Array(32, dtype=int32)

    Matrix dot product:

    >>> x1 = jnp.array([[1, 2, 3],
    ...                 [4, 5, 6]])
    >>> x2 = jnp.array([[1, 2],
    ...                 [3, 4],
    ...                 [5, 6]])
    >>> jnp.linalg.matmul(x1, x2)
    Array([[22, 28],
           [49, 64]], dtype=int32)

    For convenience, in all cases you can do the same computation using
    the ``@`` operator:

    >>> x1 @ x2
    Array([[22, 28],
           [49, 64]], dtype=int32)
  """
  check_arraylike('jnp.linalg.matmul', x1, x2)
  return jnp.matmul(x1, x2, precision=precision,
                    preferred_element_type=preferred_element_type)


@export
def tensordot(x1: ArrayLike, x2: ArrayLike, /, *,
              axes: int | tuple[Sequence[int], Sequence[int]] = 2,
              precision: PrecisionLike = None,
              preferred_element_type: DTypeLike | None = None) -> Array:
  """Compute the tensor dot product of two N-dimensional arrays.

  JAX implementation of :func:`numpy.linalg.tensordot`.

  Args:
    x1: N-dimensional array
    x2: M-dimensional array
    axes: integer or tuple of sequences of integers. If an integer `k`, then
      sum over the last `k` axes of ``x1`` and the first `k` axes of ``x2``,
      in order. If a tuple, then ``axes[0]`` specifies the axes of ``x1`` and
      ``axes[1]`` specifies the axes of ``x2``.
    precision: either ``None`` (default), which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      such values indicating precision of ``x1`` and ``x2``.
    preferred_element_type: either ``None`` (default), which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    array containing the tensor dot product of the inputs

  See also:
    - :func:`jax.numpy.tensordot`: equivalent API in the :mod:`jax.numpy` namespace.
    - :func:`jax.numpy.einsum`: NumPy API for more general tensor contractions.
    - :func:`jax.lax.dot_general`: XLA API for more general tensor contractions.

  Examples:
    >>> x1 = jnp.arange(24.).reshape(2, 3, 4)
    >>> x2 = jnp.ones((3, 4, 5))
    >>> jnp.linalg.tensordot(x1, x2)
    Array([[ 66.,  66.,  66.,  66.,  66.],
           [210., 210., 210., 210., 210.]], dtype=float32)

    Equivalent result when specifying the axes as explicit sequences:

    >>> jnp.linalg.tensordot(x1, x2, axes=([1, 2], [0, 1]))
    Array([[ 66.,  66.,  66.,  66.,  66.],
           [210., 210., 210., 210., 210.]], dtype=float32)

    Equivalent result via :func:`~jax.numpy.einsum`:

    >>> jnp.einsum('ijk,jkm->im', x1, x2)
    Array([[ 66.,  66.,  66.,  66.,  66.],
           [210., 210., 210., 210., 210.]], dtype=float32)

    Setting ``axes=1`` for two-dimensional inputs is equivalent to a matrix
    multiplication:

    >>> x1 = jnp.array([[1, 2],
    ...                 [3, 4]])
    >>> x2 = jnp.array([[1, 2, 3],
    ...                 [4, 5, 6]])
    >>> jnp.linalg.tensordot(x1, x2, axes=1)
    Array([[ 9, 12, 15],
           [19, 26, 33]], dtype=int32)
    >>> x1 @ x2
    Array([[ 9, 12, 15],
           [19, 26, 33]], dtype=int32)

    Setting ``axes=0`` for one-dimensional inputs is equivalent to
    :func:`jax.numpy.linalg.outer`:

    >>> x1 = jnp.array([1, 2])
    >>> x2 = jnp.array([1, 2, 3])
    >>> jnp.linalg.tensordot(x1, x2, axes=0)
    Array([[1, 2, 3],
           [2, 4, 6]], dtype=int32)
    >>> jnp.linalg.outer(x1, x2)
    Array([[1, 2, 3],
           [2, 4, 6]], dtype=int32)
  """
  check_arraylike('jnp.linalg.tensordot', x1, x2)
  return jnp.tensordot(x1, x2, axes=axes, precision=precision,
                       preferred_element_type=preferred_element_type)


@export
def svdvals(x: ArrayLike, /) -> Array:
  """Compute the singular values of a matrix.

  JAX implementation of :func:`numpy.linalg.svdvals`.

  Args:
    x: array of shape ``(..., M, N)`` for which singular values will be computed.

  Returns:
    array of singular values of shape ``(..., K)`` with ``K = min(M, N)``.

  See also:
    :func:`jax.numpy.linalg.svd`: compute singular values and singular vectors

  Examples:
    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> jnp.linalg.svdvals(x)
    Array([9.508031 , 0.7728694], dtype=float32)
  """
  check_arraylike('jnp.linalg.svdvals', x)
  return svd(x, compute_uv=False, hermitian=False)


@export
def diagonal(x: ArrayLike, /, *, offset: int = 0) -> Array:
  """Extract the diagonal of an matrix or stack of matrices.

  JAX implementation of :func:`numpy.linalg.diagonal`.

  Args:
    x: array of shape ``(..., M, N)`` from which the diagonal will be extracted.
    offset: positive or negative offset from the main diagonal.

  Returns:
    Array of shape ``(..., K)`` where ``K`` is the length of the specified diagonal.

  See Also:
    - :func:`jax.numpy.diagonal`: more general functionality for extracting diagonals.
    - :func:`jax.numpy.diag`: create a diagonal matrix from values.

  Examples:
    Diagonals of a single matrix:

    >>> x = jnp.array([[1,  2,  3,  4],
    ...                [5,  6,  7,  8],
    ...                [9, 10, 11, 12]])
    >>> jnp.linalg.diagonal(x)
    Array([ 1,  6, 11], dtype=int32)
    >>> jnp.linalg.diagonal(x, offset=1)
    Array([ 2,  7, 12], dtype=int32)
    >>> jnp.linalg.diagonal(x, offset=-1)
    Array([ 5, 10], dtype=int32)

    Batched diagonals:

    >>> x = jnp.arange(24).reshape(2, 3, 4)
    >>> jnp.linalg.diagonal(x)
    Array([[ 0,  5, 10],
           [12, 17, 22]], dtype=int32)
  """
  check_arraylike('jnp.linalg.diagonal', x)
  return jnp.diagonal(x, offset=offset, axis1=-2, axis2=-1)


@export
def tensorinv(a: ArrayLike, ind: int = 2) -> Array:
  """Compute the tensor inverse of an array.

  JAX implementation of :func:`numpy.linalg.tensorinv`.

  This computes the inverse of the :func:`~jax.numpy.linalg.tensordot`
  operation with the same ``ind`` value.

  Args:
    a: array to be inverted. Must have ``prod(a.shape[:ind]) == prod(a.shape[ind:])``
    ind: positive integer specifying the number of indices in the tensor product.

  Returns:
    array of shape ``(*a.shape[ind:], *a.shape[:ind])`` containing the
    tensor inverse of ``a``.

  See also:
    - :func:`jax.numpy.linalg.tensordot`
    - :func:`jax.numpy.linalg.tensorsolve`

  Examples:
    >>> key = jax.random.key(1337)
    >>> x = jax.random.normal(key, shape=(2, 2, 4))
    >>> xinv = jnp.linalg.tensorinv(x, 2)
    >>> xinv_x = jnp.linalg.tensordot(xinv, x, axes=2)
    >>> jnp.allclose(xinv_x, jnp.eye(4), atol=1E-4)
    Array(True, dtype=bool)
  """
  check_arraylike("tensorinv", a)
  arr = jnp.asarray(a)
  ind = operator.index(ind)
  if ind <= 0:
    raise ValueError(f"ind must be a positive integer; got {ind=}")
  contracting_shape, batch_shape = arr.shape[:ind], arr.shape[ind:]
  flatshape = (math.prod(contracting_shape), math.prod(batch_shape))
  if flatshape[0] != flatshape[1]:
    raise ValueError("tensorinv is only possible when the product of the first"
                     " `ind` dimensions equals that of the remaining dimensions."
                     f" got {arr.shape=} with {ind=}.")
  return inv(arr.reshape(flatshape)).reshape(*batch_shape, *contracting_shape)


@export
def tensorsolve(a: ArrayLike, b: ArrayLike, axes: tuple[int, ...] | None = None) -> Array:
  """Solve the tensor equation a x = b for x.

  JAX implementation of :func:`numpy.linalg.tensorsolve`.

  Args:
    a: input array. After reordering via ``axes`` (see below), shape must be
      ``(*b.shape, *x.shape)``.
    b: right-hand-side array.
    axes: optional tuple specifying axes of ``a`` that should be moved to the end

  Returns:
    array x such that after reordering of axes of ``a``, ``tensordot(a, x, x.ndim)``
    is equivalent to ``b``.

  See also:
    - :func:`jax.numpy.linalg.tensordot`
    - :func:`jax.numpy.linalg.tensorinv`

  Examples:
    >>> key1, key2 = jax.random.split(jax.random.key(8675309))
    >>> a = jax.random.normal(key1, shape=(2, 2, 4))
    >>> b = jax.random.normal(key2, shape=(2, 2))
    >>> x = jnp.linalg.tensorsolve(a, b)
    >>> x.shape
    (4,)

    Now show that ``x`` can be used to reconstruct ``b`` using
    :func:`~jax.numpy.linalg.tensordot`:

    >>> b_reconstructed = jnp.linalg.tensordot(a, x, axes=x.ndim)
    >>> jnp.allclose(b, b_reconstructed)
    Array(True, dtype=bool)
  """
  check_arraylike("tensorsolve", a, b)
  a_arr, b_arr = jnp.asarray(a), jnp.asarray(b)
  if axes is not None:
    a_arr = jnp.moveaxis(a_arr, axes, len(axes) * (a_arr.ndim - 1,))
  out_shape = a_arr.shape[b_arr.ndim:]
  if a_arr.shape[:b_arr.ndim] != b_arr.shape:
    raise ValueError("After moving axes to end, leading shape of a must match shape of b."
                     f" got a.shape={a_arr.shape}, b.shape={b_arr.shape}")
  if b_arr.size != math.prod(out_shape):
    raise ValueError("Input arrays must have prod(a.shape[:b.ndim]) == prod(a.shape[b.ndim:]);"
                     f" got a.shape={a_arr.shape}, b.ndim={b_arr.ndim}.")
  a_arr = a_arr.reshape(b_arr.size, math.prod(out_shape))
  return solve(a_arr, b_arr.ravel()).reshape(out_shape)


@export
def multi_dot(arrays: Sequence[ArrayLike], *, precision: PrecisionLike = None) -> Array:
  """Efficiently compute matrix products between a sequence of arrays.

  JAX implementation of :func:`numpy.linalg.multi_dot`.

  JAX internally uses the opt_einsum library to compute the most efficient
  operation order.

  Args:
    arrays: sequence of arrays. All must be two-dimensional, except the first
      and last which may be one-dimensional.
    precision: either ``None`` (default), which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``).

  Returns:
    an array representing the equivalent of ``reduce(jnp.matmul, arrays)``, but
    evaluated in the optimal order.

  This function exists because the cost of computing sequences of matmul operations
  can differ vastly depending on the order in which the operations are evaluated.
  For a single matmul, the number of floating point operations (flops) required to
  compute a matrix product can be approximated this way:

  >>> def approx_flops(x, y):
  ...   # for 2D x and y, with x.shape[1] == y.shape[0]
  ...   return 2 * x.shape[0] * x.shape[1] * y.shape[1]

  Suppose we have three matrices that we'd like to multiply in sequence:

  >>> key1, key2, key3 = jax.random.split(jax.random.key(0), 3)
  >>> x = jax.random.normal(key1, shape=(200, 5))
  >>> y = jax.random.normal(key2, shape=(5, 100))
  >>> z = jax.random.normal(key3, shape=(100, 10))

  Because of associativity of matrix products, there are two orders in which we might
  evaluate the product ``x @ y @ z``, and both produce equivalent outputs up to floating
  point precision:

  >>> result1 = (x @ y) @ z
  >>> result2 = x @ (y @ z)
  >>> jnp.allclose(result1, result2, atol=1E-4)
  Array(True, dtype=bool)

  But the computational cost of these differ greatly:

  >>> print("(x @ y) @ z flops:", approx_flops(x, y) + approx_flops(x @ y, z))
  (x @ y) @ z flops: 600000
  >>> print("x @ (y @ z) flops:", approx_flops(y, z) + approx_flops(x, y @ z))
  x @ (y @ z) flops: 30000

  The second approach is about 20x more efficient in terms of estimated flops!

  ``multi_dot`` is a function that will automatically choose the fastest
  computational path for such problems:

  >>> result3 = jnp.linalg.multi_dot([x, y, z])
  >>> jnp.allclose(result1, result3, atol=1E-4)
  Array(True, dtype=bool)

  We can use JAX's :ref:`ahead-of-time-lowering` tools to estimate the total flops
  of each approach, and confirm that ``multi_dot`` is choosing the more efficient
  option:

  >>> jax.jit(lambda x, y, z: (x @ y) @ z).lower(x, y, z).cost_analysis()['flops']
  600000.0
  >>> jax.jit(lambda x, y, z: x @ (y @ z)).lower(x, y, z).cost_analysis()['flops']
  30000.0
  >>> jax.jit(jnp.linalg.multi_dot).lower([x, y, z]).cost_analysis()['flops']
  30000.0
  """
  check_arraylike('jnp.linalg.multi_dot', *arrays)
  arrs: list[Array] = list(map(jnp.asarray, arrays))
  if len(arrs) < 2:
    raise ValueError(f"multi_dot requires at least two arrays; got len(arrays)={len(arrs)}")
  if not (arrs[0].ndim in (1, 2) and arrs[-1].ndim in (1, 2) and
          all(a.ndim == 2 for a in arrs[1:-1])):
    raise ValueError("multi_dot: input arrays must all be two-dimensional, except for"
                     " the first and last array which may be 1 or 2 dimensional."
                     f" Got array shapes {[a.shape for a in arrs]}")
  if any(a.shape[-1] != b.shape[0] for a, b in zip(arrs[:-1], arrs[1:])):
    raise ValueError("multi_dot: last dimension of each array must match first dimension"
                     f" of following array. Got array shapes {[a.shape for a in arrs]}")
  einsum_axes: list[tuple[int, ...]] = [(i, i+1) for i in range(len(arrs))]
  if arrs[0].ndim == 1:
    einsum_axes[0] = einsum_axes[0][1:]
  if arrs[-1].ndim == 1:
    einsum_axes[-1] = einsum_axes[-1][:1]
  return jnp.einsum(*itertools.chain(*zip(arrs, einsum_axes)),  # type: ignore[call-overload]
                    optimize='auto', precision=precision)


@export
@partial(jit, static_argnames=['p'])
def cond(x: ArrayLike, p=None):
  """Compute the condition number of a matrix.

  JAX implementation of :func:`numpy.linalg.cond`.

  The condition number is defined as ``norm(x, p) * norm(inv(x), p)``. For ``p = 2``
  (the default), the condition number is the ratio of the largest to the smallest
  singular value.

  Args:
    x: array of shape ``(..., M, N)`` for which to compute the condition number.
    p: the order of the norm to use. One of ``{None, 1, -1, 2, -2, inf, -inf, 'fro'}``;
      see :func:`jax.numpy.linalg.norm` for the meaning of these. The default is ``p = None``,
      which is equivalent to ``p = 2``. If not in ``{None, 2, -2}`` then ``x`` must be square,
      i.e. ``M = N``.

  Returns:
    array of shape ``x.shape[:-2]`` containing the condition number.

  See also:
    :func:`jax.numpy.linalg.norm`

  Examples:

    Well-conditioned matrix:

    >>> x = jnp.array([[1, 2],
    ...                [2, 1]])
    >>> jnp.linalg.cond(x)
    Array(3., dtype=float32)

    Ill-conditioned matrix:

    >>> x = jnp.array([[1, 2],
    ...                [0, 0]])
    >>> jnp.linalg.cond(x)
    Array(inf, dtype=float32)
  """
  check_arraylike("cond", x)
  arr = jnp.asarray(x)
  if arr.ndim < 2:
    raise ValueError(f"jnp.linalg.cond: input array must be at least 2D; got {arr.shape=}")
  if arr.shape[-1] == 0 or arr.shape[-2] == 0:
    raise ValueError(f"jnp.linalg.cond: input array must not be empty; got {arr.shape=}")
  if p is None or p == 2:
    s = svdvals(x)
    return s[..., 0] / s[..., -1]
  elif p == -2:
    s = svdvals(x)
    r = s[..., -1] / s[..., 0]
  else:
    if arr.shape[-2] != arr.shape[-1]:
      raise ValueError(f"jnp.linalg.cond: for {p=}, array must be square; got {arr.shape=}")
    r = norm(x, ord=p, axis=(-2, -1)) * norm(inv(x), ord=p, axis=(-2, -1))
  # Convert NaNs to infs where original array has no NaNs.
  return jnp.where(ufuncs.isnan(r) & ~ufuncs.isnan(x).any(axis=(-2, -1)), jnp.inf, r)


@export
def trace(x: ArrayLike, /, *,
          offset: int = 0, dtype: DTypeLike | None = None) -> Array:
  """Compute the trace of a matrix.

  JAX implementation of :func:`numpy.linalg.trace`.

  Args:
    x: array of shape ``(..., M, N)`` and whose innermost two
      dimensions form MxN matrices for which to take the trace.
    offset: positive or negative offset from the main diagonal
      (default: 0).
    dtype: data type of the returned array (default: ``None``). If ``None``,
      then output dtype will match the dtype of ``x``, promoted to default
      precision in the case of integer types.

  Returns:
    array of batched traces with shape ``x.shape[:-2]``

  See also:
    - :func:`jax.numpy.trace`: similar API in the ``jax.numpy`` namespace.

  Examples:
    Trace of a single matrix:

    >>> x = jnp.array([[1,  2,  3,  4],
    ...                [5,  6,  7,  8],
    ...                [9, 10, 11, 12]])
    >>> jnp.linalg.trace(x)
    Array(18, dtype=int32)
    >>> jnp.linalg.trace(x, offset=1)
    Array(21, dtype=int32)
    >>> jnp.linalg.trace(x, offset=-1, dtype="float32")
    Array(15., dtype=float32)

    Batched traces:

    >>> x = jnp.arange(24).reshape(2, 3, 4)
    >>> jnp.linalg.trace(x)
    Array([15, 51], dtype=int32)
  """
  check_arraylike('jnp.linalg.trace', x)
  return jnp.trace(x, offset=offset, axis1=-2, axis2=-1, dtype=dtype)
