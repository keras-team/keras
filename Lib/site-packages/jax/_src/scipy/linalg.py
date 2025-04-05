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

from functools import partial

import numpy as np
import textwrap
from typing import overload, Any, Literal

import jax
import jax.numpy as jnp
from jax import jit, vmap, jvp
from jax import lax
from jax._src import dtypes
from jax._src.lax import linalg as lax_linalg
from jax._src.lax import qdwh
from jax._src.numpy.util import (
    check_arraylike, promote_dtypes, promote_dtypes_inexact,
    promote_dtypes_complex)
from jax._src.typing import Array, ArrayLike


_no_chkfinite_doc = textwrap.dedent("""
Does not support the Scipy argument ``check_finite=True``,
because compiled JAX code cannot perform checks of array values at runtime.
""")
_no_overwrite_and_chkfinite_doc = _no_chkfinite_doc + "\nDoes not support the Scipy argument ``overwrite_*=True``."

@partial(jit, static_argnames=('lower',))
def _cholesky(a: ArrayLike, lower: bool) -> Array:
  a, = promote_dtypes_inexact(jnp.asarray(a))
  l = lax_linalg.cholesky(a if lower else jnp.conj(a.mT), symmetrize_input=False)
  return l if lower else jnp.conj(l.mT)


def cholesky(a: ArrayLike, lower: bool = False, overwrite_a: bool = False,
             check_finite: bool = True) -> Array:
  """Compute the Cholesky decomposition of a matrix.

  JAX implementation of :func:`scipy.linalg.cholesky`.

  The Cholesky decomposition of a matrix `A` is:

  .. math::

     A = U^HU = LL^H

  where `U` is an upper-triangular matrix and `L` is a lower-triangular matrix.

  Args:
    a: input array, representing a (batched) positive-definite hermitian matrix.
      Must have shape ``(..., N, N)``.
    lower: if True, compute the lower Cholesky decomposition `L`. if False
      (default), compute the upper Cholesky decomposition `U`.
    overwrite_a: unused by JAX
    check_finite: unused by JAX

  Returns:
    array of shape ``(..., N, N)`` representing the cholesky decomposition
    of the input.

  See Also:
   - :func:`jax.numpy.linalg.cholesky`: NumPy-stype Cholesky API
   - :func:`jax.lax.linalg.cholesky`: XLA-style Cholesky API
   - :func:`jax.scipy.linalg.cho_factor`
   - :func:`jax.scipy.linalg.cho_solve`

  Examples:
    A small real Hermitian positive-definite matrix:

    >>> x = jnp.array([[2., 1.],
    ...                [1., 2.]])

    Upper Cholesky factorization:

    >>> jax.scipy.linalg.cholesky(x)
    Array([[1.4142135 , 0.70710677],
           [0.        , 1.2247449 ]], dtype=float32)

    Lower Cholesky factorization:

    >>> jax.scipy.linalg.cholesky(x, lower=True)
    Array([[1.4142135 , 0.        ],
           [0.70710677, 1.2247449 ]], dtype=float32)

    Reconstructing ``x`` from its factorization:

    >>> L = jax.scipy.linalg.cholesky(x, lower=True)
    >>> jnp.allclose(x, L @ L.T)
    Array(True, dtype=bool)
  """
  del overwrite_a, check_finite  # Unused
  return _cholesky(a, lower)


def cho_factor(a: ArrayLike, lower: bool = False, overwrite_a: bool = False,
               check_finite: bool = True) -> tuple[Array, bool]:
  """Factorization for Cholesky-based linear solves

  JAX implementation of :func:`scipy.linalg.cho_factor`. This function returns
  a result suitable for use with :func:`jax.scipy.linalg.cho_solve`. For direct
  Cholesky decompositions, prefer :func:`jax.scipy.linalg.cholesky`.

  Args:
    a: input array, representing a (batched) positive-definite hermitian matrix.
      Must have shape ``(..., N, N)``.
    lower: if True, compute the lower triangular Cholesky decomposition (default: False).
    overwrite_a: unused by JAX
    check_finite: unused by JAX

  Returns:
    ``(c, lower)``: ``c`` is an array of shape ``(..., N, N)`` representing the lower or
    upper cholesky decomposition of the input; ``lower`` is a boolean specifying whether
    this is the lower or upper decomposition.

  See Also:
    - :func:`jax.scipy.linalg.cholesky`
    - :func:`jax.scipy.linalg.cho_solve`

  Examples:
    A small real Hermitian positive-definite matrix:

    >>> x = jnp.array([[2., 1.],
    ...                [1., 2.]])

    Compute the cholesky factorization via :func:`~jax.scipy.linalg.cho_factor`,
    and use it to solve a linear equation via :func:`~jax.scipy.linalg.cho_solve`.

    >>> b = jnp.array([3., 4.])
    >>> cfac = jax.scipy.linalg.cho_factor(x)
    >>> y = jax.scipy.linalg.cho_solve(cfac, b)
    >>> y
    Array([0.6666666, 1.6666666], dtype=float32)

    Check that the result is consistent:

    >>> jnp.allclose(x @ y, b)
    Array(True, dtype=bool)
  """
  del overwrite_a, check_finite  # Unused
  return (cholesky(a, lower=lower), lower)

@partial(jit, static_argnames=('lower',))
def _cho_solve(c: ArrayLike, b: ArrayLike, lower: bool) -> Array:
  c, b = promote_dtypes_inexact(jnp.asarray(c), jnp.asarray(b))
  lax_linalg._check_solve_shapes(c, b)
  b = lax_linalg.triangular_solve(c, b, left_side=True, lower=lower,
                                  transpose_a=not lower, conjugate_a=not lower)
  b = lax_linalg.triangular_solve(c, b, left_side=True, lower=lower,
                                  transpose_a=lower, conjugate_a=lower)
  return b


def cho_solve(c_and_lower: tuple[ArrayLike, bool], b: ArrayLike,
              overwrite_b: bool = False, check_finite: bool = True) -> Array:
  """Solve a linear system using a Cholesky factorization

  JAX implementation of :func:`scipy.linalg.cho_solve`. Uses the output
  of :func:`jax.scipy.linalg.cho_factor`.

  Args:
    c_and_lower: ``(c, lower)``, where ``c`` is an array of shape ``(..., N, N)``
      representing the lower or upper cholesky decomposition of the matrix, and
      ``lower`` is a boolean specifying whether this is the lower or upper decomposition.
    b: right-hand-side of linear system. Must have shape ``(..., N)``
    overwrite_a: unused by JAX
    check_finite: unused by JAX

  Returns:
    Array of shape ``(..., N)`` representing the solution of the linear system.

  See Also:
    - :func:`jax.scipy.linalg.cholesky`
    - :func:`jax.scipy.linalg.cho_factor`

  Examples:
    A small real Hermitian positive-definite matrix:

    >>> x = jnp.array([[2., 1.],
    ...                [1., 2.]])

    Compute the cholesky factorization via :func:`~jax.scipy.linalg.cho_factor`,
    and use it to solve a linear equation via :func:`~jax.scipy.linalg.cho_solve`.

    >>> b = jnp.array([3., 4.])
    >>> cfac = jax.scipy.linalg.cho_factor(x)
    >>> y = jax.scipy.linalg.cho_solve(cfac, b)
    >>> y
    Array([0.6666666, 1.6666666], dtype=float32)

    Check that the result is consistent:

    >>> jnp.allclose(x @ y, b)
    Array(True, dtype=bool)
  """
  del overwrite_b, check_finite  # Unused
  c, lower = c_and_lower
  return _cho_solve(c, b, lower)

@overload
def _svd(x: ArrayLike, *, full_matrices: bool, compute_uv: Literal[True]) -> tuple[Array, Array, Array]: ...

@overload
def _svd(x: ArrayLike, *, full_matrices: bool, compute_uv: Literal[False]) -> Array: ...

@overload
def _svd(x: ArrayLike, *, full_matrices: bool, compute_uv: bool) -> Array | tuple[Array, Array, Array]: ...

@partial(jit, static_argnames=('full_matrices', 'compute_uv'))
def _svd(a: ArrayLike, *, full_matrices: bool, compute_uv: bool) -> Array | tuple[Array, Array, Array]:
  a, = promote_dtypes_inexact(jnp.asarray(a))
  return lax_linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)

@overload
def svd(a: ArrayLike, full_matrices: bool = True, compute_uv: Literal[True] = True,
        overwrite_a: bool = False, check_finite: bool = True,
        lapack_driver: str = 'gesdd') -> tuple[Array, Array, Array]: ...

@overload
def svd(a: ArrayLike, full_matrices: bool, compute_uv: Literal[False],
        overwrite_a: bool = False, check_finite: bool = True,
        lapack_driver: str = 'gesdd') -> Array: ...

@overload
def svd(a: ArrayLike, full_matrices: bool = True, *, compute_uv: Literal[False],
        overwrite_a: bool = False, check_finite: bool = True,
        lapack_driver: str = 'gesdd') -> Array: ...

@overload
def svd(a: ArrayLike, full_matrices: bool = True, compute_uv: bool = True,
        overwrite_a: bool = False, check_finite: bool = True,
        lapack_driver: str = 'gesdd') -> Array | tuple[Array, Array, Array]: ...


def svd(a: ArrayLike, full_matrices: bool = True, compute_uv: bool = True,
        overwrite_a: bool = False, check_finite: bool = True,
        lapack_driver: str = 'gesdd') -> Array | tuple[Array, Array, Array]:
  r"""Compute the singular value decomposition.

  JAX implementation of :func:`scipy.linalg.svd`.

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
    overwrite_a: unused by JAX
    check_finite: unused by JAX
    lapack_driver: unused by JAX

  Returns:
    A tuple of arrays ``(u, s, vh)`` if ``compute_uv`` is True, otherwise the array ``s``.

    - ``u``: left singular vectors of shape ``(..., N, N)`` if ``full_matrices`` is True
      or ``(..., N, K)`` otherwise.
    - ``s``: singular values of shape ``(..., K)``
    - ``vh``: conjugate-transposed right singular vectors of shape ``(..., M, M)``
      if ``full_matrices`` is True or ``(..., K, M)`` otherwise.

    where ``K = min(N, M)``.

  See also:
    - :func:`jax.numpy.linalg.svd`: NumPy-style SVD API
    - :func:`jax.lax.linalg.svd`: XLA-style SVD API

  Examples:
    Consider the SVD of a small real-valued array:

    >>> x = jnp.array([[1., 2., 3.],
    ...                [6., 5., 4.]])
    >>> u, s, vt = jax.scipy.linalg.svd(x, full_matrices=False)
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
  del overwrite_a, check_finite, lapack_driver  # unused
  return _svd(a, full_matrices=full_matrices, compute_uv=compute_uv)


def det(a: ArrayLike, overwrite_a: bool = False, check_finite: bool = True) -> Array:
  """Compute the determinant of a matrix

  JAX implementation of :func:`scipy.linalg.det`.

  Args:
    a: input array, of shape ``(..., N, N)``
    overwrite_a: unused by JAX
    check_finite: unused by JAX

  Returns
    Determinant of shape ``a.shape[:-2]``

  See Also:
    :func:`jax.numpy.linalg.det`: NumPy-style determinant API

  Examples:
    Determinant of a small 2D array:

    >>> x = jnp.array([[1., 2.],
    ...                [3., 4.]])
    >>> jax.scipy.linalg.det(x)
    Array(-2., dtype=float32)

    Batch-wise determinant of multiple 2D arrays:

    >>> x = jnp.array([[[1., 2.],
    ...                 [3., 4.]],
    ...                [[8., 5.],
    ...                 [7., 9.]]])
    >>> jax.scipy.linalg.det(x)
    Array([-2., 37.], dtype=float32)
  """
  del overwrite_a, check_finite  # unused
  return jnp.linalg.det(a)


@overload
def _eigh(a: ArrayLike, b: ArrayLike | None, lower: bool, eigvals_only: Literal[True],
          eigvals: None, type: int) -> Array: ...

@overload
def _eigh(a: ArrayLike, b: ArrayLike | None, lower: bool, eigvals_only: Literal[False],
          eigvals: None, type: int) -> tuple[Array, Array]: ...

@overload
def _eigh(a: ArrayLike, b: ArrayLike | None, lower: bool, eigvals_only: bool,
          eigvals: None, type: int) -> Array | tuple[Array, Array]: ...

@partial(jit, static_argnames=('lower', 'eigvals_only', 'eigvals', 'type'))
def _eigh(a: ArrayLike, b: ArrayLike | None, lower: bool, eigvals_only: bool,
          eigvals: None, type: int) -> Array | tuple[Array, Array]:
  if b is not None:
    raise NotImplementedError("Only the b=None case of eigh is implemented")
  if type != 1:
    raise NotImplementedError("Only the type=1 case of eigh is implemented.")
  if eigvals is not None:
    raise NotImplementedError(
        "Only the eigvals=None case of eigh is implemented.")

  a, = promote_dtypes_inexact(jnp.asarray(a))
  v, w = lax_linalg.eigh(a, lower=lower)

  if eigvals_only:
    return w
  else:
    return w, v

@overload
def eigh(a: ArrayLike, b: ArrayLike | None = None, lower: bool = True,
         eigvals_only: Literal[False] = False, overwrite_a: bool = False,
         overwrite_b: bool = False, turbo: bool = True, eigvals: None = None,
         type: int = 1, check_finite: bool = True) -> tuple[Array, Array]: ...

@overload
def eigh(a: ArrayLike, b: ArrayLike | None = None, lower: bool = True, *,
         eigvals_only: Literal[True], overwrite_a: bool = False,
         overwrite_b: bool = False, turbo: bool = True, eigvals: None = None,
         type: int = 1, check_finite: bool = True) -> Array: ...

@overload
def eigh(a: ArrayLike, b: ArrayLike | None, lower: bool,
         eigvals_only: Literal[True], overwrite_a: bool = False,
         overwrite_b: bool = False, turbo: bool = True, eigvals: None = None,
         type: int = 1, check_finite: bool = True) -> Array: ...

@overload
def eigh(a: ArrayLike, b: ArrayLike | None = None, lower: bool = True,
         eigvals_only: bool = False, overwrite_a: bool = False,
         overwrite_b: bool = False, turbo: bool = True, eigvals: None = None,
         type: int = 1, check_finite: bool = True) -> Array | tuple[Array, Array]: ...

def eigh(a: ArrayLike, b: ArrayLike | None = None, lower: bool = True,
         eigvals_only: bool = False, overwrite_a: bool = False,
         overwrite_b: bool = False, turbo: bool = True, eigvals: None = None,
         type: int = 1, check_finite: bool = True) -> Array | tuple[Array, Array]:
  """Compute eigenvalues and eigenvectors for a Hermitian matrix

  JAX implementation of :func:`scipy.linalg.eigh`.

  Args:
    a: Hermitian input array of shape ``(..., N, N)``
    b: optional Hermitian input of shape ``(..., N, N)``. If specified, compute
      the generalized eigenvalue problem.
    lower: if True (default) access only the lower portion of the input matrix.
      Otherwise access only the upper portion.
    eigvals_only: If True, compute only the eigenvalues. If False (default) compute
      both eigenvalues and eigenvectors.
    type: if ``b`` is specified, ``type`` gives the type of generalized eigenvalue
      problem to be computed. Denoting ``(λ, v)`` as an eigenvalue, eigenvector pair:

      - ``type = 1`` solves ``a @ v = λ * b @ v`` (default)
      - ``type = 2`` solves ``a @ b @ v = λ * v``
      - ``type = 3`` solves ``b @ a @ v = λ * v``

    eigvals: a ``(low, high)`` tuple specifying which eigenvalues to compute.
    overwrite_a: unused by JAX.
    overwrite_b: unused by JAX.
    turbo: unused by JAX.
    check_finite: unused by JAX.

  Returns:
    A tuple of arrays ``(eigvals, eigvecs)`` if ``eigvals_only`` is False, otherwise
    an array ``eigvals``.

    - ``eigvals``: array of shape ``(..., N)`` containing the eigenvalues.
    - ``eigvecs``: array of shape ``(..., N, N)`` containing the eigenvectors.

  See also:
    - :func:`jax.numpy.linalg.eigh`: NumPy-style eigh API.
    - :func:`jax.lax.linalg.eigh`: XLA-style eigh API.
    - :func:`jax.numpy.linalg.eig`: non-hermitian eigenvalue problem.
    - :func:`jax.scipy.linalg.eigh_tridiagonal`: tri-diagonal eigenvalue problem.

  Examples:
    Compute the standard eigenvalue decomposition of a simple 2x2 matrix:

    >>> a = jnp.array([[2., 1.],
    ...                [1., 2.]])
    >>> eigvals, eigvecs = jax.scipy.linalg.eigh(a)
    >>> eigvals
    Array([1., 3.], dtype=float32)
    >>> eigvecs
    Array([[-0.70710677,  0.70710677],
           [ 0.70710677,  0.70710677]], dtype=float32)

    Eigenvectors are orthonormal:

    >>> jnp.allclose(eigvecs.T @ eigvecs, jnp.eye(2), atol=1E-5)
    Array(True, dtype=bool)

    Solution satisfies the eigenvalue problem:

    >>> jnp.allclose(a @ eigvecs, eigvecs @ jnp.diag(eigvals))
    Array(True, dtype=bool)
  """
  del overwrite_a, overwrite_b, turbo, check_finite  # unused
  return _eigh(a, b, lower, eigvals_only, eigvals, type)

@partial(jit, static_argnames=('output',))
def _schur(a: Array, output: str) -> tuple[Array, Array]:
  if output == "complex":
    a = a.astype(dtypes.to_complex_dtype(a.dtype))
  return lax_linalg.schur(a)

def schur(a: ArrayLike, output: str = 'real') -> tuple[Array, Array]:
  """Compute the Schur decomposition

  JAX implementation of :func:`scipy.linalg.schur`.

  The Schur form `T` of a matrix `A` satisfies:

  .. math::

     A = Z T Z^H

  where `Z` is unitary, and `T` is upper-triangular for the complex-valued Schur
  decomposition (i.e. ``output="complex"``) and is quasi-upper-triangular for the
  real-valued Schur decomposition (i.e. ``output="real"``). In the quasi-triangular
  case, the diagonal may include 2x2 blocks associated with complex-valued
  eigenvalue pairs of `A`.

  Args:
    a: input array of shape ``(..., N, N)``
    output: Specify whether to compute the ``"real"`` (default) or ``"complex"``
      Schur decomposition.

  Returns:
    A tuple of arrays ``(T, Z)``

    - ``T`` is a shape ``(..., N, N)`` array containing the upper-triangular
      Schur form of the input.
    - ``Z`` is a shape ``(..., N, N)`` array containing the unitary Schur
      transformation matrix.

  See also:
    - :func:`jax.scipy.linalg.rsf2csf`: convert real Schur form to complex Schur form.
    - :func:`jax.lax.linalg.schur`: XLA-style API for Schur decomposition.

  Examples:
    A Schur decomposition of a 3x3 matrix:

    >>> a = jnp.array([[1., 2., 3.],
    ...                [1., 4., 2.],
    ...                [3., 2., 1.]])
    >>> T, Z = jax.scipy.linalg.schur(a)

    The Schur form ``T`` is quasi-upper-triangular in general, but is truly
    upper-triangular in this case because the input matrix is symmetric:

    >>> T  # doctest: +SKIP
    Array([[-2.0000005 ,  0.5066295 , -0.43360388],
           [ 0.        ,  1.5505103 ,  0.74519426],
           [ 0.        ,  0.        ,  6.449491  ]], dtype=float32)

    The transformation matrix ``Z`` is unitary:

    >>> jnp.allclose(Z.T @ Z, jnp.eye(3), atol=1E-5)
    Array(True, dtype=bool)

    The input can be reconstructed from the outputs:

    >>> jnp.allclose(Z @ T @ Z.T, a)
    Array(True, dtype=bool)
  """
  if output not in ('real', 'complex'):
    raise ValueError(
      f"Expected 'output' to be either 'real' or 'complex', got {output=}.")
  return _schur(a, output)


def inv(a: ArrayLike, overwrite_a: bool = False, check_finite: bool = True) -> Array:
  """Return the inverse of a square matrix

  JAX implementation of :func:`scipy.linalg.inv`.

  Args:
    a: array of shape ``(..., N, N)`` specifying square array(s) to be inverted.
    overwrite_a: unused in JAX
    check_finite: unused in JAX

  Returns:
    Array of shape ``(..., N, N)`` containing the inverse of the input.

  Notes:
    In most cases, explicitly computing the inverse of a matrix is ill-advised. For
    example, to compute ``x = inv(A) @ b``, it is more performant and numerically
    precise to use a direct solve, such as :func:`jax.scipy.linalg.solve`.

  See Also:
    - :func:`jax.numpy.linalg.inv`: NumPy-style API for matrix inverse
    - :func:`jax.scipy.linalg.solve`: direct linear solver

  Examples:
    Compute the inverse of a 3x3 matrix

    >>> a = jnp.array([[1., 2., 3.],
    ...                [2., 4., 2.],
    ...                [3., 2., 1.]])
    >>> a_inv = jax.scipy.linalg.inv(a)
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
    Instead, you should use a direct solver like :func:`jax.scipy.linalg.solve`:

    >>> jax.scipy.linalg.solve(a, b)
     Array([ 0.  ,  1.25, -0.5 ], dtype=float32)
  """
  del overwrite_a, check_finite  # unused
  return jnp.linalg.inv(a)


@partial(jit, static_argnames=('overwrite_a', 'check_finite'))
def lu_factor(a: ArrayLike, overwrite_a: bool = False, check_finite: bool = True) -> tuple[Array, Array]:
  """Factorization for LU-based linear solves

  JAX implementation of :func:`scipy.linalg.lu_factor`.

  This function returns a result suitable for use with :func:`jax.scipy.linalg.lu_solve`.
  For direct LU decompositions, prefer :func:`jax.scipy.linalg.lu`.

  Args:
    a: input array of shape ``(..., M, N)``.
    overwrite_a: unused by JAX
    check_finite: unused by JAX

  Returns:
    A tuple ``(lu, piv)``

    - ``lu`` is an array of shape ``(..., M, N)``, containing ``L`` in its
      lower triangle and ``U`` in its upper.
    - ``piv`` is an array of shape ``(..., K)`` with ``K = min(M, N)``,
      which encodes the pivots.

  See Also:
    - :func:`jax.scipy.linalg.lu`
    - :func:`jax.scipy.linalg.lu_solve`

  Examples:
    Solving a small linear system via LU factorization:

    >>> a = jnp.array([[2., 1.],
    ...                [1., 2.]])

    Compute the lu factorization via :func:`~jax.scipy.linalg.lu_factor`,
    and use it to solve a linear equation via :func:`~jax.scipy.linalg.lu_solve`.

    >>> b = jnp.array([3., 4.])
    >>> lufac = jax.scipy.linalg.lu_factor(a)
    >>> y = jax.scipy.linalg.lu_solve(lufac, b)
    >>> y
    Array([0.6666666, 1.6666667], dtype=float32)

    Check that the result is consistent:

    >>> jnp.allclose(a @ y, b)
    Array(True, dtype=bool)
  """
  del overwrite_a, check_finite  # unused
  a, = promote_dtypes_inexact(jnp.asarray(a))
  lu, pivots, _ = lax_linalg.lu(a)
  return lu, pivots


@partial(jit, static_argnames=('trans', 'overwrite_b', 'check_finite'))
def lu_solve(lu_and_piv: tuple[Array, ArrayLike], b: ArrayLike, trans: int = 0,
             overwrite_b: bool = False, check_finite: bool = True) -> Array:
  """Solve a linear system using an LU factorization

  JAX implementation of :func:`scipy.linalg.lu_solve`. Uses the output
  of :func:`jax.scipy.linalg.lu_factor`.

  Args:
    lu_and_piv: ``(lu, piv)``, output of :func:`~jax.scipy.linalg.lu_factor`.
      ``lu`` is an array of shape ``(..., M, N)``, containing ``L`` in its lower
      triangle and ``U`` in its upper. ``piv`` is an array of shape ``(..., K)``,
      with ``K = min(M, N)``, which encodes the pivots.
    b: right-hand-side of linear system. Must have shape ``(..., M)``
    trans: type of system to solve. Options are:

      - ``0``: :math:`A x = b`
      - ``1``: :math:`A^Tx = b`
      - ``2``: :math:`A^Hx = b`

    overwrite_b: unused by JAX
    check_finite: unused by JAX

  Returns:
    Array of shape ``(..., N)`` representing the solution of the linear system.

  See Also:
    - :func:`jax.scipy.linalg.lu`
    - :func:`jax.scipy.linalg.lu_factor`

  Examples:
    Solving a small linear system via LU factorization:

    >>> a = jnp.array([[2., 1.],
    ...                [1., 2.]])

    Compute the lu factorization via :func:`~jax.scipy.linalg.lu_factor`,
    and use it to solve a linear equation via :func:`~jax.scipy.linalg.lu_solve`.

    >>> b = jnp.array([3., 4.])
    >>> lufac = jax.scipy.linalg.lu_factor(a)
    >>> y = jax.scipy.linalg.lu_solve(lufac, b)
    >>> y
    Array([0.6666666, 1.6666667], dtype=float32)

    Check that the result is consistent:

    >>> jnp.allclose(a @ y, b)
    Array(True, dtype=bool)
  """
  del overwrite_b, check_finite  # unused
  lu, pivots = lu_and_piv
  m, _ = lu.shape[-2:]
  perm = lax_linalg.lu_pivots_to_permutation(pivots, m)
  return lax_linalg.lu_solve(lu, perm, b, trans)

@overload
def _lu(a: ArrayLike, permute_l: Literal[True]) -> tuple[Array, Array]: ...

@overload
def _lu(a: ArrayLike, permute_l: Literal[False]) -> tuple[Array, Array, Array]: ...

@overload
def _lu(a: ArrayLike, permute_l: bool) -> tuple[Array, Array] | tuple[Array, Array, Array]: ...

@partial(jit, static_argnums=(1,))
def _lu(a: ArrayLike, permute_l: bool) -> tuple[Array, Array] | tuple[Array, Array, Array]:
  a, = promote_dtypes_inexact(jnp.asarray(a))
  lu, _, permutation = lax_linalg.lu(a)
  dtype = lax.dtype(a)
  m, n = jnp.shape(a)
  p = jnp.real(jnp.array(permutation[None, :] == jnp.arange(m, dtype=permutation.dtype)[:, None], dtype=dtype))
  k = min(m, n)
  l = jnp.tril(lu, -1)[:, :k] + jnp.eye(m, k, dtype=dtype)
  u = jnp.triu(lu)[:k, :]
  if permute_l:
    return jnp.matmul(p, l, precision=lax.Precision.HIGHEST), u
  else:
    return p, l, u

@overload
def lu(a: ArrayLike, permute_l: Literal[False] = False, overwrite_a: bool = False,
       check_finite: bool = True) -> tuple[Array, Array, Array]: ...

@overload
def lu(a: ArrayLike, permute_l: Literal[True], overwrite_a: bool = False,
       check_finite: bool = True) -> tuple[Array, Array]: ...

@overload
def lu(a: ArrayLike, permute_l: bool = False, overwrite_a: bool = False,
       check_finite: bool = True) -> tuple[Array, Array] | tuple[Array, Array, Array]: ...


@partial(jit, static_argnames=('permute_l', 'overwrite_a', 'check_finite'))
def lu(a: ArrayLike, permute_l: bool = False, overwrite_a: bool = False,
       check_finite: bool = True) -> tuple[Array, Array] | tuple[Array, Array, Array]:
  """Compute the LU decomposition

  JAX implementation of :func:`scipy.linalg.lu`.

  The LU decomposition of a matrix `A` is:

  .. math::

     A = P L U

  where `P` is a permutation matrix, `L` is lower-triangular and `U` is upper-triangular.

  Args:
    a: array of shape ``(..., M, N)`` to decompose.
    permute_l: if True, then permute ``L`` and return ``(P @ L, U)`` (default: False)
    overwrite_a: not used by JAX
    check_finite: not used by JAX

  Returns:
    A tuple of arrays ``(P @ L, U)`` if ``permute_l`` is True, else ``(P, L, U)``:

    - ``P`` is a permutation matrix of shape ``(..., M, M)``
    - ``L`` is a lower-triangular matrix of shape ``(... M, K)``
    - ``U`` is an upper-triangular matrix of shape ``(..., K, N)``

    with ``K = min(M, N)``

  See also:
    - :func:`jax.numpy.linalg.lu`: NumPy-style API for LU decomposition.
    - :func:`jax.lax.linalg.lu`: XLA-style API for LU decomposition.
    - :func:`jax.scipy.linalg.lu_solve`: LU-based linear solver.

  Examples:
    An LU decomposition of a 3x3 matrix:

    >>> a = jnp.array([[1., 2., 3.],
    ...                [5., 4., 2.],
    ...                [3., 2., 1.]])
    >>> P, L, U = jax.scipy.linalg.lu(a)

    ``P`` is a permutation matrix: i.e. each row and column has a single ``1``:

    >>> P
    Array([[0., 1., 0.],
           [1., 0., 0.],
           [0., 0., 1.]], dtype=float32)

    ``L`` and ``U`` are lower-triangular and upper-triangular matrices:

    >>> with jnp.printoptions(precision=3):
    ...   print(L)
    ...   print(U)
    [[ 1.     0.     0.   ]
     [ 0.2    1.     0.   ]
     [ 0.6   -0.333  1.   ]]
    [[5.    4.    2.   ]
     [0.    1.2   2.6  ]
     [0.    0.    0.667]]

    The original matrix can be reconstructed by multiplying the three together:

    >>> a_reconstructed = P @ L @ U
    >>> jnp.allclose(a, a_reconstructed)
    Array(True, dtype=bool)
  """
  del overwrite_a, check_finite  # unused
  return _lu(a, permute_l)


@overload
def _qr(a: ArrayLike, mode: Literal["r"], pivoting: Literal[False]
       ) -> tuple[Array]: ...

@overload
def _qr(a: ArrayLike, mode: Literal["r"], pivoting: Literal[True]
       ) -> tuple[Array, Array]: ...

@overload
def _qr(a: ArrayLike, mode: Literal["full", "economic"], pivoting: Literal[False]
       ) -> tuple[Array, Array]: ...

@overload
def _qr(a: ArrayLike, mode: Literal["full", "economic"], pivoting: Literal[True]
       ) -> tuple[Array, Array, Array]: ...

@overload
def _qr(a: ArrayLike, mode: str, pivoting: Literal[False]
       ) -> tuple[Array] | tuple[Array, Array]: ...

@overload
def _qr(a: ArrayLike, mode: str, pivoting: Literal[True]
       ) -> tuple[Array, Array] | tuple[Array, Array, Array]: ...

@overload
def _qr(a: ArrayLike, mode: str, pivoting: bool
       ) -> tuple[Array] | tuple[Array, Array] | tuple[Array, Array, Array]: ...


@partial(jit, static_argnames=('mode', 'pivoting'))
def _qr(a: ArrayLike, mode: str, pivoting: bool
       ) -> tuple[Array] | tuple[Array, Array] | tuple[Array, Array, Array]:
  if mode in ("full", "r"):
    full_matrices = True
  elif mode == "economic":
    full_matrices = False
  else:
    raise ValueError(f"Unsupported QR decomposition mode '{mode}'")
  a, = promote_dtypes_inexact(jnp.asarray(a))
  q, r, *p = lax_linalg.qr(a, pivoting=pivoting, full_matrices=full_matrices)
  if mode == "r":
    if pivoting:
      return r, p[0]
    return (r,)
  if pivoting:
    return q, r, p[0]
  return q, r


@overload
def qr(a: ArrayLike,  overwrite_a: bool = False, lwork: Any = None, *,
       mode: Literal["full", "economic"], pivoting: Literal[False] = False,
       check_finite: bool = True) -> tuple[Array, Array]: ...

@overload
def qr(a: ArrayLike,  overwrite_a: bool = False, lwork: Any = None, *,
       mode: Literal["full", "economic"], pivoting: Literal[True] = True,
       check_finite: bool = True) -> tuple[Array, Array, Array]: ...

@overload
def qr(a: ArrayLike,  overwrite_a: bool = False, lwork: Any = None, *,
       mode: Literal["full", "economic"], pivoting: bool = False,
       check_finite: bool = True
      ) -> tuple[Array, Array] | tuple[Array, Array, Array]: ...

@overload
def qr(a: ArrayLike,  overwrite_a: bool = False, lwork: Any = None, *,
       mode: Literal["r"], pivoting: Literal[False] = False, check_finite: bool = True
      ) -> tuple[Array]: ...

@overload
def qr(a: ArrayLike,  overwrite_a: bool = False, lwork: Any = None, *,
       mode: Literal["r"], pivoting: Literal[True] = True, check_finite: bool = True
      ) -> tuple[Array, Array]: ...

@overload
def qr(a: ArrayLike,  overwrite_a: bool = False, lwork: Any = None, *,
       mode: Literal["r"], pivoting: bool = False, check_finite: bool = True
      ) -> tuple[Array] | tuple[Array, Array]: ...

@overload
def qr(a: ArrayLike, overwrite_a: bool = False, lwork: Any = None, mode: str = "full",
       pivoting: bool = False, check_finite: bool = True
      ) -> tuple[Array] | tuple[Array, Array] | tuple[Array, Array, Array]: ...


def qr(a: ArrayLike, overwrite_a: bool = False, lwork: Any = None, mode: str = "full",
       pivoting: bool = False, check_finite: bool = True
      ) -> tuple[Array] | tuple[Array, Array] | tuple[Array, Array, Array]:
  """Compute the QR decomposition of an array

  JAX implementation of :func:`scipy.linalg.qr`.

  The QR decomposition of a matrix `A` is given by

  .. math::

     A = QR

  Where `Q` is a unitary matrix (i.e. :math:`Q^HQ=I`) and `R` is an upper-triangular
  matrix.

  Args:
    a: array of shape (..., M, N)
    mode: Computational mode. Supported values are:

      - ``"full"`` (default): return `Q` of shape ``(M, M)`` and `R` of shape ``(M, N)``.
      - ``"r"``: return only `R`
      - ``"economic"``: return `Q` of shape ``(M, K)`` and `R` of shape ``(K, N)``,
        where K = min(M, N).

    pivoting: Allows the QR decomposition to be rank-revealing. If ``True``, compute
      the column-pivoted decomposition ``A[:, P] = Q @ R``, where ``P`` is chosen such
      that the diagonal of ``R`` is non-increasing.
    overwrite_a: unused in JAX
    lwork: unused in JAX
    check_finite: unused in JAX

  Returns:
    A tuple ``(Q, R)`` or ``(Q, R, P)``, if ``mode`` is not ``"r"`` and ``pivoting`` is
    respectively ``False`` or ``True``, otherwise an array ``R`` or tuple ``(R, P)`` if
    mode is ``"r"``, and ``pivoting`` is respectively ``False`` or ``True``, where:

    - ``Q`` is an orthogonal matrix of shape ``(..., M, M)`` (if ``mode`` is ``"full"``)
      or ``(..., M, K)`` (if ``mode`` is ``"economic"``),
    - ``R`` is an upper-triangular matrix of shape ``(..., M, N)`` (if ``mode`` is
      ``"r"`` or ``"full"``) or ``(..., K, N)`` (if ``mode`` is ``"economic"``),
    - ``P`` is an index vector of shape ``(..., N)``.

    with ``K = min(M, N)``.

  Notes:
    - At present, pivoting is only implemented on CPU backends.

  See also:
    - :func:`jax.numpy.linalg.qr`: NumPy-style QR decomposition API
    - :func:`jax.lax.linalg.qr`: XLA-style QR decomposition API

  Examples:
    Compute the QR decomposition of a matrix:

    >>> a = jnp.array([[1., 2., 3., 4.],
    ...                [5., 4., 2., 1.],
    ...                [6., 3., 1., 5.]])
    >>> Q, R = jax.scipy.linalg.qr(a)
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
  del overwrite_a, lwork, check_finite  # unused
  return _qr(a, mode, pivoting)


@partial(jit, static_argnames=('assume_a', 'lower'))
def _solve(a: ArrayLike, b: ArrayLike, assume_a: str, lower: bool) -> Array:
  if assume_a != 'pos':
    return jnp.linalg.solve(a, b)

  a, b = promote_dtypes_inexact(jnp.asarray(a), jnp.asarray(b))
  lax_linalg._check_solve_shapes(a, b)

  # With custom_linear_solve, we can reuse the same factorization when
  # computing sensitivities. This is considerably faster.
  factors = cho_factor(lax.stop_gradient(a), lower=lower)
  custom_solve = partial(
      lax.custom_linear_solve,
      lambda x: lax_linalg._broadcasted_matvec(a, x),
      solve=lambda _, x: cho_solve(factors, x),
      symmetric=True)
  if a.ndim == b.ndim + 1:
    # b.shape == [..., m]
    return custom_solve(b)
  else:
    # b.shape == [..., m, k]
    return vmap(custom_solve, b.ndim - 1, max(a.ndim, b.ndim) - 1)(b)


def solve(a: ArrayLike, b: ArrayLike, lower: bool = False,
          overwrite_a: bool = False, overwrite_b: bool = False, debug: bool = False,
          check_finite: bool = True, assume_a: str = 'gen') -> Array:
  """Solve a linear system of equations

  JAX implementation of :func:`scipy.linalg.solve`.

  This solves a (batched) linear system of equations ``a @ x = b`` for ``x``
  given ``a`` and ``b``.

  Args:
    a: array of shape ``(..., N, N)``.
    b: array of shape ``(..., N)`` or ``(..., N, M)``
    lower: Referenced only if ``assume_a != 'gen'``. If True, only use the lower
      triangle of the input, If False (default), only use the upper triangle.
    assume_a: specify what properties of ``a`` can be assumed. Options are:

      - ``"gen"``: generic matrix (default)
      - ``"sym"``: symmetric matrix
      - ``"her"``: hermitian matrix
      - ``"pos"``: positive-definite matrix

    overwrite_a: unused by JAX
    overwrite_b: unused by JAX
    debug: unused by JAX
    check_finite: unused by JAX

  Returns:
    An array of the same shape as ``b`` containing the solution to the linear system.

  See also:
    - :func:`jax.scipy.linalg.lu_solve`: Solve via LU factorization.
    - :func:`jax.scipy.linalg.cho_solve`: Solve via Cholesky factorization.
    - :func:`jax.scipy.linalg.solve_triangular`: Solve a triangular system.
    - :func:`jax.numpy.linalg.solve`: NumPy-style API for solving linear systems.
    - :func:`jax.lax.custom_linear_solve`: matrix-free linear solver.

  Examples:
    A simple 3x3 linear system:

    >>> A = jnp.array([[1., 2., 3.],
    ...                [2., 4., 2.],
    ...                [3., 2., 1.]])
    >>> b = jnp.array([14., 16., 10.])
    >>> x = jax.scipy.linalg.solve(A, b)
    >>> x
    Array([1., 2., 3.], dtype=float32)

    Confirming that the result solves the system:

    >>> jnp.allclose(A @ x, b)
    Array(True, dtype=bool)
  """
  del overwrite_a, overwrite_b, debug, check_finite  #unused
  valid_assume_a = ['gen', 'sym', 'her', 'pos']
  if assume_a not in valid_assume_a:
    raise ValueError(f"Expected assume_a to be one of {valid_assume_a}; got {assume_a!r}")
  return _solve(a, b, assume_a, lower)

@partial(jit, static_argnames=('trans', 'lower', 'unit_diagonal'))
def _solve_triangular(a: ArrayLike, b: ArrayLike, trans: int | str,
                      lower: bool, unit_diagonal: bool) -> Array:
  if trans == 0 or trans == "N":
    transpose_a, conjugate_a = False, False
  elif trans == 1 or trans == "T":
    transpose_a, conjugate_a = True, False
  elif trans == 2 or trans == "C":
    transpose_a, conjugate_a = True, True
  else:
    raise ValueError(f"Invalid 'trans' value {trans}")

  a, b = promote_dtypes_inexact(jnp.asarray(a), jnp.asarray(b))

  # lax_linalg.triangular_solve only supports matrix 'b's at the moment.
  b_is_vector = jnp.ndim(a) == jnp.ndim(b) + 1
  if b_is_vector:
    b = b[..., None]
  out = lax_linalg.triangular_solve(a, b, left_side=True, lower=lower,
                                    transpose_a=transpose_a,
                                    conjugate_a=conjugate_a,
                                    unit_diagonal=unit_diagonal)
  if b_is_vector:
    return out[..., 0]
  else:
    return out


def solve_triangular(a: ArrayLike, b: ArrayLike, trans: int | str = 0, lower: bool = False,
                     unit_diagonal: bool = False, overwrite_b: bool = False,
                     debug: Any = None, check_finite: bool = True) -> Array:
  """Solve a triangular linear system of equations

  JAX implementation of :func:`scipy.linalg.solve_triangular`.

  This solves a (batched) linear system of equations ``a @ x = b`` for ``x``
  given a triangular matrix ``a`` and a vector or matrix ``b``.

  Args:
    a: array of shape ``(..., N, N)``. Only part of the array will be accessed,
      depending on the ``lower`` and ``unit_diagonal`` arguments.
    b: array of shape ``(..., N)`` or ``(..., N, M)``
    lower: If True, only use the lower triangle of the input, If False (default),
      only use the upper triangle.
    unit_diagonal: If True, ignore diagonal elements of ``a`` and assume they are
      ``1`` (default: False).
    trans: specify what properties of ``a`` can be assumed. Options are:

      - ``0`` or ``'N'``: solve :math:`Ax=b`
      - ``1`` or ``'T'``: solve :math:`A^Tx=b`
      - ``2`` or ``'C'``: solve :math:`A^Hx=b`

    overwrite_b: unused by JAX
    debug: unused by JAX
    check_finite: unused by JAX

  Returns:
    An array of the same shape as ``b`` containing the solution to the linear system.

  See also:
    :func:`jax.scipy.linalg.solve`: Solve a general linear system.

  Examples:
    A simple 3x3 triangular linear system:

    >>> A = jnp.array([[1., 2., 3.],
    ...                [0., 3., 2.],
    ...                [0., 0., 5.]])
    >>> b = jnp.array([10., 8., 5.])
    >>> x = jax.scipy.linalg.solve_triangular(A, b)
    >>> x
    Array([3., 2., 1.], dtype=float32)

    Confirming that the result solves the system:

    >>> jnp.allclose(A @ x, b)
    Array(True, dtype=bool)

    Computing the transposed problem:

    >>> x = jax.scipy.linalg.solve_triangular(A, b, trans='T')
    >>> x
    Array([10. , -4. , -3.4], dtype=float32)

    Confirming that the result solves the system:

    >>> jnp.allclose(A.T @ x, b)
    Array(True, dtype=bool)
  """
  del overwrite_b, debug, check_finite  # unused
  return _solve_triangular(a, b, trans, lower, unit_diagonal)


@partial(jit, static_argnames=('upper_triangular', 'max_squarings'))
def expm(A: ArrayLike, *, upper_triangular: bool = False, max_squarings: int = 16) -> Array:
  """Compute the matrix exponential

  JAX implementation of :func:`scipy.linalg.expm`.

  Args:
    A: array of shape ``(..., N, N)``
    upper_triangular: if True, then assume that ``A`` is upper-triangular. Default=False.
    max_squarings: The number of squarings in the scaling-and-squaring approximation method
     (default: 16).

  Returns:
    An array of shape ``(..., N, N)`` containing the matrix exponent of ``A``.

  Notes:
    This uses the scaling-and-squaring approximation method, with computational complexity
    controlled by the optional ``max_squarings`` argument. Theoretically, the number of
    required squarings is ``max(0, ceil(log2(norm(A))) - c)`` where ``norm(A)`` is the L1
    norm and ``c=2.42`` for float64/complex128, or ``c=1.97`` for float32/complex64.

  See Also:
    :func:`jax.scipy.linalg.expm_frechet`

  Examples:

    ``expm`` is the matrix exponential, and has similar properties to the more
    familiar scalar exponential. For scalars ``a`` and ``b``, :math:`e^{a + b}
    = e^a e^b`. However, for matrices, this property only holds when ``A`` and
    ``B`` commute (``AB = BA``). In this case, ``expm(A+B) = expm(A) @ expm(B)``

    >>> A = jnp.array([[2, 0],
    ...                [0, 1]])
    >>> B = jnp.array([[3, 0],
    ...                [0, 4]])
    >>> jnp.allclose(jax.scipy.linalg.expm(A+B),
    ...              jax.scipy.linalg.expm(A) @ jax.scipy.linalg.expm(B),
    ...              rtol=0.0001)
    Array(True, dtype=bool)

    If a matrix ``X`` is invertible, then
    ``expm(X @ A @ inv(X)) = X @ expm(A) @ inv(X)``

    >>> X = jnp.array([[3, 1],
    ...                [2, 5]])
    >>> X_inv = jax.scipy.linalg.inv(X)
    >>> jnp.allclose(jax.scipy.linalg.expm(X @ A @ X_inv),
    ...              X @ jax.scipy.linalg.expm(A) @ X_inv)
    Array(True, dtype=bool)
  """
  A, = promote_dtypes_inexact(A)

  if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
    raise ValueError(f"Expected A to be a (batched) square matrix, got {A.shape=}.")

  if A.ndim > 2:
    return jnp.vectorize(
      partial(expm, upper_triangular=upper_triangular, max_squarings=max_squarings),
      signature="(n,n)->(n,n)")(A)

  P, Q, n_squarings = _calc_P_Q(jnp.asarray(A))

  def _nan(args):
    A, *_ = args
    return jnp.full_like(A, jnp.nan)

  def _compute(args):
    A, P, Q = args
    R = _solve_P_Q(P, Q, upper_triangular)
    R = _squaring(R, n_squarings, max_squarings)
    return R

  R = lax.cond(n_squarings > max_squarings, _nan, _compute, (A, P, Q))
  return R

@jit
def _calc_P_Q(A: Array) -> tuple[Array, Array, Array]:
  if A.ndim != 2 or A.shape[0] != A.shape[1]:
    raise ValueError('expected A to be a square matrix')
  A_L1 = jnp.linalg.norm(A,1)
  n_squarings: Array
  U: Array
  V: Array
  if A.dtype == 'float64' or A.dtype == 'complex128':
   maxnorm = 5.371920351148152
   n_squarings = jnp.maximum(0, jnp.floor(jnp.log2(A_L1 / maxnorm)))
   A = A / 2 ** n_squarings.astype(A.dtype)
   conds = jnp.array([1.495585217958292e-002, 2.539398330063230e-001,
                      9.504178996162932e-001, 2.097847961257068e+000],
                      dtype=A_L1.dtype)
   idx = jnp.digitize(A_L1, conds)
   U, V = lax.switch(idx, [_pade3, _pade5, _pade7, _pade9, _pade13], A)
  elif A.dtype == 'float32' or A.dtype == 'complex64':
    maxnorm = 3.925724783138660
    n_squarings = jnp.maximum(0, jnp.floor(jnp.log2(A_L1 / maxnorm)))
    A = A / 2 ** n_squarings.astype(A.dtype)
    conds = jnp.array([4.258730016922831e-001, 1.880152677804762e+000],
                      dtype=A_L1.dtype)
    idx = jnp.digitize(A_L1, conds)
    U, V = lax.switch(idx, [_pade3, _pade5, _pade7], A)
  else:
    raise TypeError(f"A.dtype={A.dtype} is not supported.")
  P = U + V  # p_m(A) : numerator
  Q = -U + V # q_m(A) : denominator
  return P, Q, n_squarings

def _solve_P_Q(P: ArrayLike, Q: ArrayLike, upper_triangular: bool = False) -> Array:
  if upper_triangular:
    return solve_triangular(Q, P)
  else:
    return jnp.linalg.solve(Q, P)

def _precise_dot(A: ArrayLike, B: ArrayLike) -> Array:
  return jnp.dot(A, B, precision=lax.Precision.HIGHEST)

@partial(jit, static_argnums=2)
def _squaring(R: Array, n_squarings: Array, max_squarings: int) -> Array:
  # squaring step to undo scaling
  def _squaring_precise(x):
    return _precise_dot(x, x)

  def _identity(x):
    return x

  def _scan_f(c, i):
    return lax.cond(i < n_squarings, _squaring_precise, _identity, c), None
  res, _ = lax.scan(_scan_f, R, jnp.arange(max_squarings, dtype=n_squarings.dtype))

  return res

def _pade3(A: Array) -> tuple[Array, Array]:
  b = (120., 60., 12., 1.)
  M, N = A.shape
  ident = jnp.eye(M, N, dtype=A.dtype)
  A2 = _precise_dot(A, A)
  U = _precise_dot(A, (b[3]*A2 + b[1]*ident))
  V: Array = b[2]*A2 + b[0]*ident
  return U, V

def _pade5(A: Array) -> tuple[Array, Array]:
  b = (30240., 15120., 3360., 420., 30., 1.)
  M, N = A.shape
  ident = jnp.eye(M, N, dtype=A.dtype)
  A2 = _precise_dot(A, A)
  A4 = _precise_dot(A2, A2)
  U = _precise_dot(A, b[5]*A4 + b[3]*A2 + b[1]*ident)
  V: Array = b[4]*A4 + b[2]*A2 + b[0]*ident
  return U, V

def _pade7(A: Array) -> tuple[Array, Array]:
  b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
  M, N = A.shape
  ident = jnp.eye(M, N, dtype=A.dtype)
  A2 = _precise_dot(A, A)
  A4 = _precise_dot(A2, A2)
  A6 = _precise_dot(A4, A2)
  U = _precise_dot(A, b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
  V = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
  return U,V

def _pade9(A: Array) -> tuple[Array, Array]:
  b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
       2162160., 110880., 3960., 90., 1.)
  M, N = A.shape
  ident = jnp.eye(M, N, dtype=A.dtype)
  A2 = _precise_dot(A, A)
  A4 = _precise_dot(A2, A2)
  A6 = _precise_dot(A4, A2)
  A8 = _precise_dot(A6, A2)
  U = _precise_dot(A, b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
  V = b[8]*A8 + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
  return U,V

def _pade13(A: Array) -> tuple[Array, Array]:
  b = (64764752532480000., 32382376266240000., 7771770303897600.,
       1187353796428800., 129060195264000., 10559470521600., 670442572800.,
       33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.)
  M, N = A.shape
  ident = jnp.eye(M, N, dtype=A.dtype)
  A2 = _precise_dot(A, A)
  A4 = _precise_dot(A2, A2)
  A6 = _precise_dot(A4, A2)
  U = _precise_dot(A, _precise_dot(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
  V = _precise_dot(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
  return U,V


@overload
def expm_frechet(A: ArrayLike, E: ArrayLike, *, method: str | None = None,
                 compute_expm: Literal[True] = True) -> tuple[Array, Array]: ...

@overload
def expm_frechet(A: ArrayLike, E: ArrayLike, *, method: str | None = None,
                 compute_expm: Literal[False]) -> Array: ...

@overload
def expm_frechet(A: ArrayLike, E: ArrayLike, *, method: str | None = None,
                 compute_expm: bool = True) -> Array | tuple[Array, Array]: ...


@partial(jit, static_argnames=('method', 'compute_expm'))
def expm_frechet(A: ArrayLike, E: ArrayLike, *, method: str | None = None,
                 compute_expm: bool = True) -> Array | tuple[Array, Array]:
  """Compute the Frechet derivative of the matrix exponential.

  JAX implementation of :func:`scipy.linalg.expm_frechet`

  Args:
    A: array of shape ``(..., N, N)``
    E: array of shape ``(..., N, N)``; specifies the direction of the derivative.
    compute_expm: if True (default) then compute and return ``expm(A)``.
    method: ignored by JAX

  Returns:
    A tuple ``(expm_A, expm_frechet_AE)`` if ``compute_expm`` is True, else
    the array ``expm_frechet_AE``. Both returned arrays have shape ``(..., N, N)``.

  See also:
    :func:`jax.scipy.linalg.expm`

  Examples:
    We can use this API to compute the matrix exponential of ``A``, as well as its
    derivative in the direction ``E``:

    >>> key1, key2 = jax.random.split(jax.random.key(3372))
    >>> A = jax.random.normal(key1, (3, 3))
    >>> E = jax.random.normal(key2, (3, 3))
    >>> expmA, expm_frechet_AE = jax.scipy.linalg.expm_frechet(A, E)

    This can be equivalently computed using JAX's automatic differentiation methods;
    here we'll compute the derivative of :func:`~jax.scipy.linalg.expm` in the
    direction of ``E`` using :func:`jax.jvp`, and find the same results:

    >>> expmA2, expm_frechet_AE2 = jax.jvp(jax.scipy.linalg.expm, (A,), (E,))
    >>> jnp.allclose(expmA, expmA2)
    Array(True, dtype=bool)
    >>> jnp.allclose(expm_frechet_AE, expm_frechet_AE2)
    Array(True, dtype=bool)
  """
  del method  # unused
  A_arr = jnp.asarray(A)
  E_arr = jnp.asarray(E)
  if A_arr.ndim < 2 or A_arr.shape[-2] != A_arr.shape[1]:
    raise ValueError(f'expected A to be a (batched) square matrix, got A.shape={A_arr.shape}')
  if E_arr.ndim < 2 or E_arr.shape[-2] != E_arr.shape[-1]:
    raise ValueError(f'expected E to be a (batched) square matrix, got E.shape={E_arr.shape}')
  if A_arr.shape != E_arr.shape:
    raise ValueError('expected A and E to be the same shape, got '
                     f'A.shape={A_arr.shape} E.shape={E_arr.shape}')
  bound_fun = partial(expm, upper_triangular=False, max_squarings=16)
  expm_A, expm_frechet_AE = jvp(bound_fun, (A_arr,), (E_arr,))
  if compute_expm:
    return expm_A, expm_frechet_AE
  else:
    return expm_frechet_AE


@jit
def block_diag(*arrs: ArrayLike) -> Array:
  """Create a block diagonal matrix from input arrays.

  JAX implementation of :func:`scipy.linalg.block_diag`.

  Args:
    *arrs: arrays of at most two dimensions

  Returns:
    2D block-diagonal array constructed by placing the input arrays
    along the diagonal.

  Examples:
    >>> A = jnp.ones((1, 1))
    >>> B = jnp.ones((2, 2))
    >>> C = jnp.ones((3, 3))
    >>> jax.scipy.linalg.block_diag(A, B, C)
    Array([[1., 0., 0., 0., 0., 0.],
           [0., 1., 1., 0., 0., 0.],
           [0., 1., 1., 0., 0., 0.],
           [0., 0., 0., 1., 1., 1.],
           [0., 0., 0., 1., 1., 1.],
           [0., 0., 0., 1., 1., 1.]], dtype=float32)
  """
  if len(arrs) == 0:
    arrs =  (jnp.zeros((1, 0)),)
  arrs = tuple(promote_dtypes(*arrs))
  bad_shapes = [i for i, a in enumerate(arrs) if jnp.ndim(a) > 2]
  if bad_shapes:
    raise ValueError("Arguments to jax.scipy.linalg.block_diag must have at "
                     "most 2 dimensions, got {} at argument {}."
                     .format(arrs[bad_shapes[0]], bad_shapes[0]))
  converted_arrs = [jnp.atleast_2d(a) for a in arrs]
  acc = converted_arrs[0]
  dtype = lax.dtype(acc)
  for a in converted_arrs[1:]:
    _, c = a.shape
    a = lax.pad(a, dtype.type(0), ((0, 0, 0), (acc.shape[-1], 0, 0)))
    acc = lax.pad(acc, dtype.type(0), ((0, 0, 0), (0, c, 0)))
    acc = lax.concatenate([acc, a], dimension=0)
  return acc


@partial(jit, static_argnames=("eigvals_only", "select", "select_range"))
def eigh_tridiagonal(d: ArrayLike, e: ArrayLike, *, eigvals_only: bool = False,
                     select: str = 'a', select_range: tuple[float, float] | None = None,
                     tol: float | None = None) -> Array:
  """Solve the eigenvalue problem for a symmetric real tridiagonal matrix

  JAX implementation of :func:`scipy.linalg.eigh_tridiagonal`.

  Args:
    d: real-valued array of shape ``(N,)`` specifying the diagonal elements.
    e: real-valued array of shape ``(N - 1,)`` specifying the off-diagonal elements.
    eigvals_only: If True, return only the eigenvalues (default: False). Computation
      of eigenvectors is not yet implemented, so ``eigvals_only`` must be set to True.
    select: specify which eigenvalues to calculate. Supported values are:

      - ``'a'``: all eigenvalues
      - ``'i'``: eigenvalues with indices ``select_range[0] <= i <= select_range[1]``

      JAX does not currently implement ``select = 'v'``.
    select_range: range of values used when ``select='i'``.
    tol: absolute tolerance to use when solving for the eigenvalues.

  Returns:
    An array of eigenvalues with shape ``(N,)``.

  See also:
    :func:`jax.scipy.linalg.eigh`: general Hermitian eigenvalue solver

  Examples:
    >>> d = jnp.array([1., 2., 3., 4.])
    >>> e = jnp.array([1., 1., 1.])
    >>> eigvals = jax.scipy.linalg.eigh_tridiagonal(d, e, eigvals_only=True)
    >>> eigvals
    Array([0.2547188, 1.8227171, 3.1772828, 4.745281 ], dtype=float32)

    For comparison, we can construct the full matrix and compute the same result
    using :func:`~jax.scipy.linalg.eigh`:

    >>> A = jnp.diag(d) + jnp.diag(e, 1) + jnp.diag(e, -1)
    >>> A
    Array([[1., 1., 0., 0.],
           [1., 2., 1., 0.],
           [0., 1., 3., 1.],
           [0., 0., 1., 4.]], dtype=float32)
    >>> eigvals_full = jax.scipy.linalg.eigh(A, eigvals_only=True)
    >>> jnp.allclose(eigvals, eigvals_full)
    Array(True, dtype=bool)
  """
  if not eigvals_only:
    raise NotImplementedError("Calculation of eigenvectors is not implemented")

  def _sturm(alpha, beta_sq, pivmin, alpha0_perturbation, x):
    """Implements the Sturm sequence recurrence."""
    n = alpha.shape[0]
    zeros = jnp.zeros(x.shape, dtype=jnp.int32)
    ones = jnp.ones(x.shape, dtype=jnp.int32)

    # The first step in the Sturm sequence recurrence
    # requires special care if x is equal to alpha[0].
    def sturm_step0():
      q = alpha[0] - x
      count = jnp.where(q < 0, ones, zeros)
      q = jnp.where(alpha[0] == x, alpha0_perturbation, q)
      return q, count

    # Subsequent steps all take this form:
    def sturm_step(i, q, count):
      q = alpha[i] - beta_sq[i - 1] / q - x
      count = jnp.where(q <= pivmin, count + 1, count)
      q = jnp.where(q <= pivmin, jnp.minimum(q, -pivmin), q)
      return q, count

    # The first step initializes q and count.
    q, count = sturm_step0()

    # Peel off ((n-1) % blocksize) steps from the main loop, so we can run
    # the bulk of the iterations unrolled by a factor of blocksize.
    blocksize = 16
    i = 1
    peel = (n - 1) % blocksize
    unroll_cnt = peel

    def unrolled_steps(args):
      start, q, count = args
      for j in range(unroll_cnt):
        q, count = sturm_step(start + j, q, count)
      return start + unroll_cnt, q, count

    i, q, count = unrolled_steps((i, q, count))

    # Run the remaining steps of the Sturm sequence using a partially
    # unrolled while loop.
    unroll_cnt = blocksize
    def cond(iqc):
      i, q, count = iqc
      return jnp.less(i, n)
    _, _, count = lax.while_loop(cond, unrolled_steps, (i, q, count))
    return count

  alpha = jnp.asarray(d)
  beta = jnp.asarray(e)
  supported_dtypes = (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
  if alpha.dtype != beta.dtype:
    raise TypeError("diagonal and off-diagonal values must have same dtype, "
                    f"got {alpha.dtype} and {beta.dtype}")
  if alpha.dtype not in supported_dtypes or beta.dtype not in supported_dtypes:
    raise TypeError("Only float32 and float64 inputs are supported as inputs "
                    "to jax.scipy.linalg.eigh_tridiagonal, got "
                    f"{alpha.dtype} and {beta.dtype}")
  n = alpha.shape[0]
  if n <= 1:
    return jnp.real(alpha)

  if jnp.issubdtype(alpha.dtype, jnp.complexfloating):
    alpha = jnp.real(alpha)
    beta_sq = jnp.real(beta * jnp.conj(beta))
    beta_abs = jnp.sqrt(beta_sq)
  else:
    beta_abs = jnp.abs(beta)
    beta_sq = jnp.square(beta)

  # Estimate the largest and smallest eigenvalues of T using the Gershgorin
  # circle theorem.
  off_diag_abs_row_sum = jnp.concatenate(
      [beta_abs[:1], beta_abs[:-1] + beta_abs[1:], beta_abs[-1:]], axis=0)
  lambda_est_max = jnp.amax(alpha + off_diag_abs_row_sum)
  lambda_est_min = jnp.amin(alpha - off_diag_abs_row_sum)
  # Upper bound on 2-norm of T.
  t_norm = jnp.maximum(jnp.abs(lambda_est_min), jnp.abs(lambda_est_max))

  # Compute the smallest allowed pivot in the Sturm sequence to avoid
  # overflow.
  finfo = np.finfo(alpha.dtype)
  one = np.ones([], dtype=alpha.dtype)
  safemin = np.maximum(one / finfo.max, (one + finfo.eps) * finfo.tiny)
  pivmin = safemin * jnp.maximum(1, jnp.amax(beta_sq))
  alpha0_perturbation = jnp.square(finfo.eps * beta_abs[0])
  abs_tol = finfo.eps * t_norm
  if tol is not None:
    abs_tol = jnp.maximum(tol, abs_tol)

  # In the worst case, when the absolute tolerance is eps*lambda_est_max and
  # lambda_est_max = -lambda_est_min, we have to take as many bisection steps
  # as there are bits in the mantissa plus 1.
  # The proof is left as an exercise to the reader.
  max_it = finfo.nmant + 1

  # Determine the indices of the desired eigenvalues, based on select and
  # select_range.
  if select == 'a':
    target_counts = jnp.arange(n, dtype=jnp.int32)
  elif select == 'i':
    if select_range is None:
      raise ValueError("for select='i', select_range must be specified.")
    if select_range[0] > select_range[1]:
      raise ValueError('Got empty index range in select_range.')
    target_counts = jnp.arange(select_range[0], select_range[1] + 1, dtype=jnp.int32)
  elif select == 'v':
    # TODO(phawkins): requires dynamic shape support.
    raise NotImplementedError("eigh_tridiagonal(..., select='v') is not "
                              "implemented")
  else:
    raise ValueError("'select must have a value in {'a', 'i', 'v'}.")

  # Run binary search for all desired eigenvalues in parallel, starting from
  # the interval lightly wider than the estimated
  # [lambda_est_min, lambda_est_max].
  fudge = 2.1  # We widen starting interval the Gershgorin interval a bit.
  norm_slack = jnp.array(n, alpha.dtype) * fudge * finfo.eps * t_norm
  lower = lambda_est_min - norm_slack - 2 * fudge * pivmin
  upper = lambda_est_max + norm_slack + fudge * pivmin

  # Pre-broadcast the scalars used in the Sturm sequence for improved
  # performance.
  target_shape = jnp.shape(target_counts)
  lower = jnp.broadcast_to(lower, shape=target_shape)
  upper = jnp.broadcast_to(upper, shape=target_shape)
  mid = 0.5 * (upper + lower)
  pivmin = jnp.broadcast_to(pivmin, target_shape)
  alpha0_perturbation = jnp.broadcast_to(alpha0_perturbation, target_shape)

  # Start parallel binary searches.
  def cond(args):
    i, lower, _, upper = args
    return jnp.logical_and(
        jnp.less(i, max_it),
        jnp.less(abs_tol, jnp.amax(upper - lower)))

  def body(args):
    i, lower, mid, upper = args
    counts = _sturm(alpha, beta_sq, pivmin, alpha0_perturbation, mid)
    lower = jnp.where(counts <= target_counts, mid, lower)
    upper = jnp.where(counts > target_counts, mid, upper)
    mid = 0.5 * (lower + upper)
    return i + 1, lower, mid, upper

  _, _, mid, _ = lax.while_loop(cond, body, (0, lower, mid, upper))
  return mid

@partial(jit, static_argnames=('side', 'method'))
@jax.default_matmul_precision("float32")
def polar(a: ArrayLike, side: str = 'right', *, method: str = 'qdwh', eps: float | None = None,
          max_iterations: int | None = None) -> tuple[Array, Array]:
  r"""Computes the polar decomposition.

  Given the :math:`m \times n` matrix :math:`a`, returns the factors of the polar
  decomposition :math:`u` (also :math:`m \times n`) and :math:`p` such that
  :math:`a = up` (if side is ``"right"``; :math:`p` is :math:`n \times n`) or
  :math:`a = pu` (if side is ``"left"``; :math:`p` is :math:`m \times m`),
  where :math:`p` is positive semidefinite.  If :math:`a` is nonsingular,
  :math:`p` is positive definite and the
  decomposition is unique. :math:`u` has orthonormal columns unless
  :math:`n > m`, in which case it has orthonormal rows.

  Writing the SVD of :math:`a` as
  :math:`a = u_\mathit{svd} \cdot s_\mathit{svd} \cdot v^h_\mathit{svd}`, we
  have :math:`u = u_\mathit{svd} \cdot v^h_\mathit{svd}`. Thus the unitary
  factor :math:`u` can be constructed as the application of the sign function to
  the singular values of :math:`a`; or, if :math:`a` is Hermitian, the
  eigenvalues.

  Several methods exist to compute the polar decomposition. Currently two
  are supported:

  * ``method="svd"``:

    Computes the SVD of :math:`a` and then forms
    :math:`u = u_\mathit{svd} \cdot v^h_\mathit{svd}`.

  * ``method="qdwh"``:

    Applies the `QDWH`_ (QR-based Dynamically Weighted Halley) algorithm.

  Args:
    a: The :math:`m \times n` input matrix.
    side: Determines whether a right or left polar decomposition is computed.
      If ``side`` is ``"right"`` then :math:`a = up`. If ``side`` is ``"left"``
      then :math:`a = pu`. The default is ``"right"``.
    method: Determines the algorithm used, as described above.
    precision: :class:`~jax.lax.Precision` object specifying the matmul precision.
    eps: The final result will satisfy
      :math:`\left|x_k - x_{k-1}\right| < \left|x_k\right| (4\epsilon)^{\frac{1}{3}}`,
      where :math:`x_k` are the QDWH iterates. Ignored if ``method`` is not
      ``"qdwh"``.
    max_iterations: Iterations will terminate after this many steps even if the
      above is unsatisfied.  Ignored if ``method`` is not ``"qdwh"``.

  Returns:
    A ``(unitary, posdef)`` tuple, where ``unitary`` is the unitary factor
    (:math:`m \times n`), and ``posdef`` is the positive-semidefinite factor.
    ``posdef`` is either :math:`n \times n` or :math:`m \times m` depending on
    whether ``side`` is ``"right"`` or ``"left"``, respectively.

  Examples:

    Polar decomposition of a 3x3 matrix:

    >>> a = jnp.array([[1., 2., 3.],
    ...                [5., 4., 2.],
    ...                [3., 2., 1.]])
    >>> U, P = jax.scipy.linalg.polar(a)

    U is a Unitary Matrix:

    >>> jnp.round(U.T @ U)  # doctest: +SKIP
    Array([[ 1., -0., -0.],
           [-0.,  1.,  0.],
           [-0.,  0.,  1.]], dtype=float32)

    P is positive-semidefinite Matrix:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...     print(P)
    [[4.79 3.25 1.23]
     [3.25 3.06 2.01]
     [1.23 2.01 2.91]]

    The original matrix can be reconstructed by multiplying the U and P:

    >>> a_reconstructed = U @ P
    >>> jnp.allclose(a, a_reconstructed)
    Array(True, dtype=bool)

  .. _QDWH: https://epubs.siam.org/doi/abs/10.1137/090774999
  """
  arr = jnp.asarray(a)
  if arr.ndim != 2:
    raise ValueError("The input `a` must be a 2-D array.")

  if side not in ["right", "left"]:
    raise ValueError("The argument `side` must be either 'right' or 'left'.")

  m, n = arr.shape
  if method == "qdwh":
    # TODO(phawkins): return info also if the user opts in?
    if m >= n and side == "right":
      unitary, posdef, _, _ = qdwh.qdwh(arr, is_hermitian=False, eps=eps)
    elif m < n and side == "left":
      arr = arr.T.conj()
      unitary, posdef, _, _ = qdwh.qdwh(arr, is_hermitian=False, eps=eps)
      posdef = posdef.T.conj()
      unitary = unitary.T.conj()
    else:
      raise NotImplementedError("method='qdwh' only supports mxn matrices "
                                "where m < n where side='right' and m >= n "
                                f"side='left', got {arr.shape} with {side=}")
  elif method == "svd":
    u_svd, s_svd, vh_svd = lax_linalg.svd(arr, full_matrices=False)
    s_svd = s_svd.astype(u_svd.dtype)
    unitary = u_svd @ vh_svd
    if side == "right":
      # a = u * p
      posdef = (vh_svd.T.conj() * s_svd[None, :]) @ vh_svd
    else:
      # a = p * u
      posdef = (u_svd * s_svd[None, :]) @ (u_svd.T.conj())
  else:
    raise ValueError(f"Unknown polar decomposition method {method}.")

  return unitary, posdef


@jit
def _sqrtm_triu(T: Array) -> Array:
  """
  Implements Björck, Å., & Hammarling, S. (1983).
      "A Schur method for the square root of a matrix". Linear algebra and
      its applications", 52, 127-140.
  """
  diag = jnp.sqrt(jnp.diag(T))
  n = diag.size
  U = jnp.diag(diag)

  def i_loop(l, data):
    j, U = data
    i = j - 1 - l
    s = lax.fori_loop(i + 1, j, lambda k, val: val + U[i, k] * U[k, j], 0.0)
    value = jnp.where(T[i, j] == s, 0.0,
                      (T[i, j] - s) / (diag[i] + diag[j]))
    return j, U.at[i, j].set(value)

  def j_loop(j, U):
    _, U = lax.fori_loop(0, j, i_loop, (j, U))
    return U

  U = lax.fori_loop(0, n, j_loop, U)
  return U

@jit
def _sqrtm(A: ArrayLike) -> Array:
  T, Z = schur(A, output='complex')
  sqrt_T = _sqrtm_triu(T)
  return jnp.matmul(jnp.matmul(Z, sqrt_T, precision=lax.Precision.HIGHEST),
                    jnp.conj(Z.T), precision=lax.Precision.HIGHEST)


def sqrtm(A: ArrayLike, blocksize: int = 1) -> Array:
  """Compute the matrix square root

  JAX implementation of :func:`scipy.linalg.sqrtm`.

  Args:
    A: array of shape ``(N, N)``
    blocksize: Not supported in JAX; JAX always uses ``blocksize=1``.

  Returns:
    An array of shape ``(N, N)`` containing the matrix square root of ``A``

  See Also:
    :func:`jax.scipy.linalg.expm`

  Examples:
    >>> a = jnp.array([[1., 2., 3.],
    ...                [2., 4., 2.],
    ...                [3., 2., 1.]])
    >>> sqrt_a = jax.scipy.linalg.sqrtm(a)
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(sqrt_a)
    [[0.92+0.71j 0.54+0.j   0.92-0.71j]
     [0.54+0.j   1.85+0.j   0.54-0.j  ]
     [0.92-0.71j 0.54-0.j   0.92+0.71j]]

    By definition, matrix multiplication of the matrix square root with itself should
    equal the input:

    >>> jnp.allclose(a, sqrt_a @ sqrt_a)
    Array(True, dtype=bool)

  Notes:
    This function implements the complex Schur method described in [1]_.  It does not use
    recursive blocking to speed up computations as a Sylvester Equation solver is not
    yet available in JAX.

  References:
    .. [1] Björck, Å., & Hammarling, S. (1983). "A Schur method for the square root of a matrix".
           Linear algebra and its applications, 52, 127-140.
  """
  if blocksize > 1:
      raise NotImplementedError("Blocked version is not implemented yet.")
  return _sqrtm(A)


@partial(jit, static_argnames=('check_finite',))
def rsf2csf(T: ArrayLike, Z: ArrayLike, check_finite: bool = True) -> tuple[Array, Array]:
  """Convert real Schur form to complex Schur form.

  JAX implementation of :func:`scipy.linalg.rsf2csf`.

  Args:
    T: array of shape ``(..., N, N)`` containing the real Schur form of the input.
    Z: array of shape ``(..., N, N)`` containing the corresponding Schur transformation
      matrix.
    check_finite: unused by JAX

  Returns:
    A tuple of arrays ``(T, Z)`` of the same shape as the inputs, containing the
    Complex Schur form and the associated Schur transformation matrix.

  See Also:
    :func:`jax.scipy.linalg.schur`: Schur decomposition

  Examples:
    >>> A = jnp.array([[0., 3., 3.],
    ...                [0., 1., 2.],
    ...                [2., 0., 1.]])
    >>> Tr, Zr = jax.scipy.linalg.schur(A)
    >>> Tc, Zc = jax.scipy.linalg.rsf2csf(Tr, Zr)

    Both the real and complex form can be used to reconstruct the input matrix
    to float32 precision:

    >>> jnp.allclose(Zr @ Tr @ Zr.T, A, atol=1E-5)
    Array(True, dtype=bool)
    >>> jnp.allclose(Zc @ Tc @ Zc.conj().T, A, atol=1E-5)
    Array(True, dtype=bool)

    The real-valued Schur form is only quasi-upper-triangular, as we can see in this case:

    >>> with jax.numpy.printoptions(precision=2, suppress=True):
    ...   print(Tr)
    [[ 3.76 -2.17  1.38]
     [ 0.   -0.88 -0.35]
     [ 0.    2.37 -0.88]]

    By contrast, the complex form is truly upper-triangular:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(Tc)
    [[ 3.76+0.j    1.29-0.78j  2.02-0.5j ]
     [ 0.  +0.j   -0.88+0.91j -2.02+0.j  ]
     [ 0.  +0.j    0.  +0.j   -0.88-0.91j]]
  """
  del check_finite  # unused

  T_arr = jnp.asarray(T)
  Z_arr = jnp.asarray(Z)

  if T_arr.ndim != 2 or T_arr.shape[0] != T_arr.shape[1]:
    raise ValueError("Input 'T' must be square.")
  if Z_arr.ndim != 2 or Z_arr.shape[0] != Z_arr.shape[1]:
    raise ValueError("Input 'Z' must be square.")
  if T_arr.shape[0] != Z_arr.shape[0]:
    raise ValueError(f"Input array shapes must match: Z: {Z_arr.shape} vs. T: {T_arr.shape}")

  T_arr, Z_arr = promote_dtypes_complex(T_arr, Z_arr)
  eps = jnp.finfo(T_arr.dtype).eps
  N = T_arr.shape[0]

  if N == 1:
    return T_arr, Z_arr

  def _update_T_Z(m, T, Z):
    mu = jnp.linalg.eigvals(lax.dynamic_slice(T, (m-1, m-1), (2, 2))) - T[m, m]
    r = jnp.linalg.norm(jnp.array([mu[0], T[m, m-1]])).astype(T.dtype)
    c = mu[0] / r
    s = T[m, m-1] / r
    G = jnp.array([[c.conj(), s], [-s, c]], dtype=T.dtype)

    # T[m-1:m+1, m-1:] = G @ T[m-1:m+1, m-1:]
    T_rows = lax.dynamic_slice_in_dim(T, m-1, 2, axis=0)
    col_mask = jnp.arange(N) >= m-1
    G_dot_T_zeroed_cols = G @ jnp.where(col_mask, T_rows, 0)
    T_rows_new = jnp.where(~col_mask, T_rows, G_dot_T_zeroed_cols)
    T = lax.dynamic_update_slice_in_dim(T, T_rows_new, m-1, axis=0)

    # T[:m+1, m-1:m+1] = T[:m+1, m-1:m+1] @ G.conj().T
    T_cols = lax.dynamic_slice_in_dim(T, m-1, 2, axis=1)
    row_mask = jnp.arange(N)[:, jnp.newaxis] < m+1
    T_zeroed_rows_dot_GH = jnp.where(row_mask, T_cols, 0) @ G.conj().T
    T_cols_new = jnp.where(~row_mask, T_cols, T_zeroed_rows_dot_GH)
    T = lax.dynamic_update_slice_in_dim(T, T_cols_new, m-1, axis=1)

    # Z[:, m-1:m+1] = Z[:, m-1:m+1] @ G.conj().T
    Z_cols = lax.dynamic_slice_in_dim(Z, m-1, 2, axis=1)
    Z = lax.dynamic_update_slice_in_dim(Z, Z_cols @ G.conj().T, m-1, axis=1)
    return T, Z

  def _rsf2scf_iter(i, TZ):
    m = N-i
    T, Z = TZ
    T, Z = lax.cond(
      jnp.abs(T[m, m-1]) > eps*(jnp.abs(T[m-1, m-1]) + jnp.abs(T[m, m])),
      _update_T_Z,
      lambda m, T, Z: (T, Z),
      m, T, Z)
    T = T.at[m, m-1].set(0.0)
    return T, Z

  return lax.fori_loop(1, N, _rsf2scf_iter, (T_arr, Z_arr))

@overload
def hessenberg(a: ArrayLike, *, calc_q: Literal[False], overwrite_a: bool = False,
               check_finite: bool = True) -> Array: ...

@overload
def hessenberg(a: ArrayLike, *, calc_q: Literal[True], overwrite_a: bool = False,
               check_finite: bool = True) -> tuple[Array, Array]: ...


@partial(jit, static_argnames=('calc_q', 'check_finite', 'overwrite_a'))
def hessenberg(a: ArrayLike, *, calc_q: bool = False, overwrite_a: bool = False,
               check_finite: bool = True) -> Array | tuple[Array, Array]:
  """Compute the Hessenberg form of the matrix

  JAX implementation of :func:`scipy.linalg.hessenberg`.

  The Hessenberg form `H` of a matrix `A` satisfies:

  .. math::

     A = Q H Q^H

  where `Q` is unitary and `H` is zero below the first subdiagonal.

  Args:
    a : array of shape ``(..., N, N)``
    calc_q: if True, calculate the ``Q`` matrix (default: False)
    overwrite_a: unused by JAX
    check_finite: unused by JAX

  Returns:
    A tuple of arrays ``(H, Q)`` if ``calc_q`` is True, else an array ``H``

    - ``H`` has shape ``(..., N, N)`` and is the Hessenberg form of ``a``
    - ``Q`` has shape ``(..., N, N)`` and is the associated unitary matrix

  Examples:
    Computing the Hessenberg form of a 4x4 matrix

    >>> a = jnp.array([[1., 2., 3., 4.],
    ...                [1., 4., 2., 3.],
    ...                [3., 2., 1., 4.],
    ...                [2., 3., 2., 2.]])
    >>> H, Q = jax.scipy.linalg.hessenberg(a, calc_q=True)
    >>> with jnp.printoptions(suppress=True, precision=3):
    ...   print(H)
    [[ 1.    -5.078  1.167  1.361]
     [-3.742  5.786 -3.613 -1.825]
     [ 0.    -2.992  2.493 -0.577]
     [ 0.     0.    -0.043 -1.279]]

    Notice the zeros in the subdiagonal positions. The original matrix
    can be reconstructed using the ``Q`` vectors:

    >>> a_reconstructed = Q @ H @ Q.conj().T
    >>> jnp.allclose(a_reconstructed, a)
    Array(True, dtype=bool)
  """
  del overwrite_a, check_finite  # unused
  n = jnp.shape(a)[-1]
  if n == 0:
    if calc_q:
      return jnp.zeros_like(a), jnp.zeros_like(a)
    else:
      return jnp.zeros_like(a)
  a_out, taus = lax_linalg.hessenberg(a)
  h = jnp.triu(a_out, -1)
  if calc_q:
    q = lax_linalg.householder_product(a_out[..., 1:, :-1], taus)
    batch_dims = a_out.shape[:-2]
    q = jnp.block([[jnp.ones(batch_dims + (1, 1), dtype=a_out.dtype),
                    jnp.zeros(batch_dims + (1, n - 1), dtype=a_out.dtype)],
                   [jnp.zeros(batch_dims + (n - 1, 1), dtype=a_out.dtype), q]])
    return h, q
  else:
    return h


def toeplitz(c: ArrayLike, r: ArrayLike | None = None) -> Array:
  r"""Construct a Toeplitz matrix.

  JAX implementation of :func:`scipy.linalg.toeplitz`.

  A Toeplitz matrix has equal diagonals: :math:`A_{ij} = k_{i - j}`
  for :math:`0 \le i < n` and :math:`0 \le j < n`. This function
  specifies the diagonals via the first column ``c`` and the first row
  ``r``, such that for row `i` and column `j`:

  .. math::

     A_{ij} = \begin{cases}
      c_{i - j} & i \ge j \\
      r_{j - i} & i < j
     \end{cases}

  Notice this implies that :math:`r_0` is ignored.

  Args:
    c: array of shape ``(..., N)`` specifying the first column.
    r: (optional) array of shape ``(..., M)`` specifying the first row. Leading
      dimensions must be broadcast-compatible with those of ``c``. If not specified,
      ``r`` defaults to ``conj(c)``.

  Returns:
    A Toeplitz matrix of shape ``(... N, M)``.

  Examples:
    Specifying ``c`` only:

    >>> c = jnp.array([1, 2, 3])
    >>> jax.scipy.linalg.toeplitz(c)
    Array([[1, 2, 3],
           [2, 1, 2],
           [3, 2, 1]], dtype=int32)

    Specifying ``c`` and ``r``:

    >>> r = jnp.array([-1, -2, -3])
    >>> jax.scipy.linalg.toeplitz(c, r)  # Note r[0] is ignored
    Array([[ 1, -2, -3],
           [ 2,  1, -2],
           [ 3,  2,  1]], dtype=int32)

    If specifying only complex-valued ``c``, ``r`` defaults to ``c.conj()``,
    resulting in a Hermitian matrix if ``c[0].imag == 0``:

    >>> c = jnp.array([1, 2+1j, 1+2j])
    >>> M = jax.scipy.linalg.toeplitz(c)
    >>> M
    Array([[1.+0.j, 2.-1.j, 1.-2.j],
           [2.+1.j, 1.+0.j, 2.-1.j],
           [1.+2.j, 2.+1.j, 1.+0.j]], dtype=complex64)
    >>> print("M is Hermitian:", jnp.all(M == M.conj().T))
    M is Hermitian: True

    For N-dimensional ``c`` and/or ``r``, the result is a batch of Toeplitz matrices:

    >>> c = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> jax.scipy.linalg.toeplitz(c)
    Array([[[1, 2, 3],
            [2, 1, 2],
            [3, 2, 1]],
    <BLANKLINE>
           [[4, 5, 6],
            [5, 4, 5],
            [6, 5, 4]]], dtype=int32)
  """
  if r is None:
    check_arraylike("toeplitz", c)
    r = jnp.conjugate(jnp.asarray(c))
  else:
    check_arraylike("toeplitz", c, r)
  return _toeplitz(jnp.atleast_1d(jnp.asarray(c)), jnp.atleast_1d(jnp.asarray(r)))

@partial(jnp.vectorize, signature="(m),(n)->(m,n)")
def _toeplitz(c: Array, r: Array) -> Array:
  ncols, = c.shape
  nrows, = r.shape
  if ncols == 0 or nrows == 0:
    return jnp.empty((ncols, nrows), dtype=jnp.promote_types(c.dtype, r.dtype))
  nelems = ncols + nrows - 1
  elems = jnp.concatenate((c[::-1], r[1:]))
  patches = lax.conv_general_dilated_patches(
      elems.reshape((1, nelems, 1)),
      (nrows,), (1,), 'VALID', dimension_numbers=('NTC', 'IOT', 'NTC'),
      precision=lax.Precision.HIGHEST)[0]
  return jnp.flip(patches, axis=0)

@partial(jit, static_argnames=("n",))
def hilbert(n: int) -> Array:
  r"""Create a Hilbert matrix of order n.

  JAX implementation of :func:`scipy.linalg.hilbert`.

  The Hilbert matrix is defined by:

  .. math::

     H_{ij} = \frac{1}{i + j + 1}

  for :math:`1 \le i \le n` and :math:`1 \le j \le n`.

  Args:
    n: the size of the matrix to create.

  Returns:
    A Hilbert matrix of shape ``(n, n)``

  Examples:
    >>> jax.scipy.linalg.hilbert(2)
    Array([[1.        , 0.5       ],
           [0.5       , 0.33333334]], dtype=float32)
    >>> jax.scipy.linalg.hilbert(3)
    Array([[1.        , 0.5       , 0.33333334],
           [0.5       , 0.33333334, 0.25      ],
           [0.33333334, 0.25      , 0.2       ]], dtype=float32)
  """
  a = lax.broadcasted_iota(jnp.float64, (n, 1), 0)
  return 1/(a + a.T + 1)
