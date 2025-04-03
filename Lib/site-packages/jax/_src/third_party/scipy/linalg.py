from __future__ import annotations

from collections.abc import Callable

from jax import jit, lax
import jax.numpy as jnp
from jax._src.numpy.linalg import norm
from jax._src.scipy.linalg import rsf2csf, schur
from jax._src.typing import ArrayLike, Array


@jit
def _algorithm_11_1_1(F: Array, T: Array) -> tuple[Array, Array]:
  # Algorithm 11.1.1 from Golub and Van Loan "Matrix Computations"
  N = T.shape[0]
  minden = jnp.abs(T[0, 0])

  def _outer_loop(p, F_minden):
    _, F, minden = lax.fori_loop(1, N-p+1, _inner_loop, (p, *F_minden))
    return F, minden

  def _inner_loop(i, p_F_minden):
    p, F, minden = p_F_minden
    j = i+p
    s = T[i-1, j-1] * (F[j-1, j-1] - F[i-1, i-1])
    T_row, T_col = T[i-1], T[:, j-1]
    F_row, F_col = F[i-1], F[:, j-1]
    ind = (jnp.arange(N) >= i) & (jnp.arange(N) < j-1)
    val = (jnp.where(ind, T_row, 0) @ jnp.where(ind, F_col, 0) -
            jnp.where(ind, F_row, 0) @ jnp.where(ind, T_col, 0))
    s = s + val
    den = T[j-1, j-1] - T[i-1, i-1]
    s = jnp.where(den != 0, s / den, s)
    F = F.at[i-1, j-1].set(s)
    minden = jnp.minimum(minden, jnp.abs(den))
    return p, F, minden

  return lax.fori_loop(1, N, _outer_loop, (F, minden))


def funm(A: ArrayLike, func: Callable[[Array], Array],
         disp: bool = True) -> Array | tuple[Array, Array]:
  """Evaluate a matrix-valued function

  JAX implementation of :func:`scipy.linalg.funm`.

  Args:
    A: array of shape ``(N, N)`` for which the function is to be computed.
    func: Callable object that takes a scalar argument and returns a scalar result.
      Represents the function to be evaluated over the eigenvalues of A.
    disp: If true (default), error information is not returned. Unlike scipy's version JAX
      does not attempt to display information at runtime.
    compute_expm: (N, N) array_like or None, optional.
      If provided, the matrix exponential of A. This is used for improving efficiency when `func`
      is the exponential function. If not provided, it is computed internally.
      Defaults to None.

  Returns:
    Array of same shape as ``A``, containing the result of ``func`` evaluated on the
    eigenvalues of ``A``.

  Notes:
    The returned dtype of JAX's implementation may differ from that of scipy;
    specifically, in cases where all imaginary parts of the array values are
    close to zero, the SciPy function may return a real-valued array, whereas
    the JAX implementation will return a complex-valued array.

  Examples:
    Applying an arbitrary matrix function:

    >>> A = jnp.array([[1., 2.], [3., 4.]])
    >>> def func(x):
    ...   return jnp.sin(x) + 2 * jnp.cos(x)
    >>> jax.scipy.linalg.funm(A, func)  # doctest: +SKIP
    Array([[ 1.2452652 +0.j, -0.3701772 +0.j],
           [-0.55526584+0.j,  0.6899995 +0.j]], dtype=complex64)

    Comparing two ways of computing the matrix exponent:

    >>> expA_1 = jax.scipy.linalg.funm(A, jnp.exp)
    >>> expA_2 = jax.scipy.linalg.expm(A)
    >>> jnp.allclose(expA_1, expA_2, rtol=1E-4)
    Array(True, dtype=bool)
  """
  A_arr = jnp.asarray(A)
  if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
    raise ValueError('expected square array_like input')

  T, Z = schur(A_arr)
  T, Z = rsf2csf(T, Z)

  F = jnp.diag(func(jnp.diag(T)))
  F = F.astype(T.dtype.char)

  F, minden = _algorithm_11_1_1(F, T)
  F = Z @ F @ Z.conj().T

  if disp:
    return F

  if F.dtype.char.lower() == 'e':
    tol = jnp.finfo(jnp.float16).eps
  if F.dtype.char.lower() == 'f':
    tol = jnp.finfo(jnp.float32).eps
  else:
    tol = jnp.finfo(jnp.float64).eps

  minden = jnp.where(minden == 0.0, tol, minden)
  err = jnp.where(jnp.any(jnp.isinf(F)), jnp.inf, jnp.minimum(1, jnp.maximum(
          tol, (tol / minden) * norm(jnp.triu(T, 1), 1))))

  return F, err
