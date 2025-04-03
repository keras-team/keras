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

"""Sparse linear algebra routines."""

from __future__ import annotations

from collections.abc import Callable
import functools

import jax
import jax.numpy as jnp

from jax.experimental import sparse
from jax.interpreters import mlir
from jax.interpreters import xla

from jax._src import core
from jax._src.interpreters import ad
from jax._src.lib import gpu_solver

import numpy as np
from scipy.sparse import csr_matrix, linalg


def lobpcg_standard(
    A: jax.Array | Callable[[jax.Array], jax.Array],
    X: jax.Array,
    m: int = 100,
    tol: jax.Array | float | None = None):
  """Compute the top-k standard eigenvalues using the LOBPCG routine.

  LOBPCG [1] stands for Locally Optimal Block Preconditioned Conjugate Gradient.
  The method enables finding top-k eigenvectors in an accelerator-friendly
  manner.

  This initial experimental version has several caveats.

    - Only the standard eigenvalue problem `A U = lambda U` is supported,
      general eigenvalues are not.
    - Gradient code is not available.
    - f64 will only work where jnp.linalg.eigh is supported for that type.
    - Finding the smallest eigenvectors is not yet supported. As a result,
      we don't yet support preconditioning, which is mostly needed for this
      case.

  The implementation is based on [2] and [3]; however, we deviate from these
  sources in several ways to improve robustness or facilitate implementation:

    - Despite increased iteration cost, we always maintain an orthonormal basis
      for the block search directions.
    - We change the convergence criterion; see the `tol` argument.
    - Soft locking [4] is intentionally not implemented; it relies on
      choosing an appropriate problem-specific tolerance to prevent
      blow-up near convergence from catastrophic cancellation of
      near-0 residuals. Instead, the approach implemented favors
      truncating the iteration basis.

  [1]: http://ccm.ucdenver.edu/reports/rep149.pdf
  [2]: https://arxiv.org/abs/1704.07458
  [3]: https://arxiv.org/abs/0705.2626
  [4]: DOI 10.13140/RG.2.2.11794.48327

  Args:
    A : An `(n, n)` array representing a square Hermitian matrix or a
        callable with its action.
    X : An `(n, k)` array representing the initial search directions for the `k`
        desired top eigenvectors. This need not be orthogonal, but must be
        numerically linearly independent (`X` will be orthonormalized).
        Note that we must have `0 < k * 5 < n`.
    m : Maximum integer iteration count; LOBPCG will only ever explore (a
        subspace of) the Krylov basis `{X, A X, A^2 X, ..., A^m X}`.
    tol : A float convergence tolerance; an eigenpair `(lambda, v)` is converged
          when its residual L2 norm `r = |A v - lambda v|` is below
          `tol * 10 * n * (lambda + |A v|)`, which
          roughly estimates the worst-case floating point error for an ideal
          eigenvector. If all `k` eigenvectors satisfy the tolerance
          comparison, then LOBPCG exits early. If left as None, then this is set
          to the float epsilon of `A.dtype`.

  Returns:
    `theta, U, i`, where `theta` is a `(k,)` array
    of eigenvalues, `U` is a `(n, k)` array of eigenvectors, `i` is the
    number of iterations performed.

  Raises:
    ValueError : if `A,X` dtypes or `n` dimensions do not match, or `k` is too
                 large (only `k * 5 < n` supported), or `k == 0`.
  """
  # Jit-compile once per matrix shape if possible.
  if isinstance(A, (jax.Array, np.ndarray)):
    return _lobpcg_standard_matrix(A, X, m, tol, debug=False)
  return _lobpcg_standard_callable(A, X, m, tol, debug=False)

@functools.partial(jax.jit, static_argnames=['m', 'debug'])
def _lobpcg_standard_matrix(
    A: jax.Array,
    X: jax.Array,
    m: int,
    tol: jax.Array | float | None,
    debug: bool = False):
  """Computes lobpcg_standard(), possibly with debug diagnostics."""
  return _lobpcg_standard_callable(
      functools.partial(_mm, A), X, m, tol, debug)

@functools.partial(jax.jit, static_argnames=['A', 'm', 'debug'])
def _lobpcg_standard_callable(
    A: Callable[[jax.Array], jax.Array],
    X: jax.Array,
    m: int,
    tol: jax.Array | float | None,
    debug: bool = False):
  """Supports generic lobpcg_standard() callable interface."""

  # TODO(vladf): support mixed_precision flag, which allows f64 Rayleigh-Ritz
  # with f32 inputs.

  n, k = X.shape
  dt = X.dtype

  _check_inputs(A, X)

  if tol is None:
    tol = float(jnp.finfo(dt).eps)

  X = _orthonormalize(X)
  P = _extend_basis(X, X.shape[1])

  # We maintain X, our current list of best eigenvectors,
  # P, our search direction, and
  # R, our residuals, in a large joint array XPR, column-stacked, so (n, 3*k).

  AX = A(X)
  theta = jnp.sum(X * AX, axis=0, keepdims=True)
  R = AX - theta * X

  def cond(state):
    i, _X, _P, _R, converged, _ = state
    return jnp.logical_and(i < m, converged < k)

  def body(state):
    i, X, P, R, _, theta = state
    # Invariants: X, P, R kept orthonormal
    # Some R, P columns may be 0 (due to basis truncation, as decided
    # by orthogonalization routines), but not X.

    # TODO(vladf): support preconditioning for bottom-k eigenvectors
    # if M is not None:
    #   R = M(R)

    # Residual basis selection.
    R = _project_out(jnp.concatenate((X, P), axis=1), R)
    XPR = jnp.concatenate((X, P, R), axis=1)

    # Projected eigensolve.
    theta, Q = _rayleigh_ritz_orth(A, XPR)

    # Eigenvector X extraction
    B = Q[:, :k]
    normB = jnp.linalg.norm(B, ord=2, axis=0, keepdims=True)
    B /= normB
    X = _mm(XPR, B)
    normX = jnp.linalg.norm(X, ord=2, axis=0, keepdims=True)
    X /= normX

    # Difference terms P extraction
    #
    # In next step of LOBPCG, naively, we'd set
    # P = S[:, k:] @ Q[k:, :k] to achieve span(X, P) == span(X, previous X)
    # (this is not obvious, see section 4 of [1]).
    #
    # Instead we orthogonalize concat(0, Q[k:, :k]) against Q[:, :k]
    # in the standard basis before mapping with XPR. Since XPR is itself
    # orthonormal, the resulting directions are themselves orthonormalized.
    #
    # [2] leverages Q's existing orthogonality to derive
    # an analytic expression for this value based on the quadrant Q[:k,k:]
    # (see section 4.2 of [2]).
    q, _ = jnp.linalg.qr(Q[:k, k:].T)
    diff_rayleigh_ortho = _mm(Q[:, k:], q)
    P = _mm(XPR, diff_rayleigh_ortho)
    normP = jnp.linalg.norm(P, ord=2, axis=0, keepdims=True)
    P /= jnp.where(normP == 0, 1.0, normP)

    # Compute new residuals.
    AX = A(X)
    R = AX - theta[jnp.newaxis, :k] * X
    resid_norms = jnp.linalg.norm(R, ord=2, axis=0)

    # I tried many variants of hard and soft locking [3]. All of them seemed
    # to worsen performance relative to no locking.
    #
    # Further, I found a more experimental convergence formula compared to what
    # is suggested in the literature, loosely based on floating-point
    # expectations.
    #
    # [2] discusses various strategies for this in Sec 5.3. The solution
    # they end up with, which estimates operator norm |A| via Gaussian
    # products, was too crude in practice (and overly-lax). The Gaussian
    # approximation seems like an estimate of the average eigenvalue.
    #
    # Instead, we test convergence via self-consistency of the eigenpair
    # i.e., the residual norm |r| should be small, relative to the floating
    # point error we'd expect from computing just the residuals given
    # candidate vectors.
    reltol = jnp.linalg.norm(AX, ord=2, axis=0) + theta[:k]
    reltol *= n
    # Allow some margin for a few element-wise operations.
    reltol *= 10
    res_converged = resid_norms < tol * reltol
    converged = jnp.sum(res_converged)

    new_state = i + 1, X, P, R, converged, theta[jnp.newaxis, :k]
    if debug:
      diagnostics = _generate_diagnostics(
          XPR, X, P, R, theta, converged, resid_norms / reltol)
      new_state = (new_state, diagnostics)
    return new_state

  converged = 0
  state = (0, X, P, R, converged, theta)
  if debug:
    state, diagnostics = jax.lax.scan(
        lambda state, _: body(state), state, xs=None, length=m)
  else:
    state = jax.lax.while_loop(cond, body, state)
  i, X, _P, _R, _converged, theta = state

  if debug:
    return theta[0, :], X, i, diagnostics
  return theta[0, :], X, i


def _check_inputs(A, X):
  n, k = X.shape
  dt = X.dtype

  if k == 0:
    raise ValueError(f'must have search dim > 0, got {k}')

  if k * 5 >= n:
    raise ValueError(f'expected search dim * 5 < matrix dim (got {k * 5}, {n})')

  test_output = A(jnp.zeros((n, 1), dtype=X.dtype))

  if test_output.dtype != dt:
    raise ValueError(
        f'A, X must have same dtypes (were {test_output.dtype}, {dt})')

  if test_output.shape != (n, 1):
    s = test_output.shape
    raise ValueError(f'A must be ({n}, {n}) matrix A, got output {s}')


def _mm(a, b, precision=jax.lax.Precision.HIGHEST):
  return jax.lax.dot(a, b, (precision, precision))

def _generate_diagnostics(prev_XPR, X, P, R, theta, converged, adj_resid):
  k = X.shape[1]
  assert X.shape == P.shape

  diagdiag = lambda x: jnp.diag(jnp.diag(x))
  abserr = lambda x: jnp.abs(x).sum() / (k ** 2)

  XTX = _mm(X.T, X)
  DX = diagdiag(XTX)
  orthX = abserr(XTX - DX)

  PTP = _mm(P.T, P)
  DP = diagdiag(PTP)
  orthP = abserr(PTP - DP)

  PX = abserr(X.T @ P)

  prev_basis = prev_XPR.shape[1] - jnp.sum(jnp.all(prev_XPR == 0.0, axis=0))

  return {
      'basis rank': prev_basis,
      'X zeros': jnp.sum(jnp.all(X == 0.0, axis=0)),
      'P zeros': jnp.sum(jnp.all(P == 0.0, axis=0)),
      'lambda history': theta[:k],
      'residual history': jnp.linalg.norm(R, axis=0, ord=2),
      'converged': converged,
      'adjusted residual max': jnp.max(adj_resid),
      'adjusted residual p50': jnp.median(adj_resid),
      'adjusted residual min': jnp.min(adj_resid),
      'X orth': orthX,
      'P orth': orthP,
      'P.X': PX}

def _eigh_ascending(A):
  w, V = jnp.linalg.eigh(A)
  return w[::-1], V[:, ::-1]


def _svqb(X):
  """Derives a truncated orthonormal basis for `X`.

  SVQB [1] is an accelerator-friendly orthonormalization procedure, which
  squares the matrix `C = X.T @ X` and computes an eigenbasis for a smaller
  `(k, k)` system; this offloads most of the work in orthonormalization
  to the first multiply when `n` is large.

  Importantly, if diagonalizing the squared matrix `C` reveals rank deficiency
  of X (which would be evidenced by near-0 then), eigenvalues corresponding
  columns are zeroed out.

  [1]: https://sdm.lbl.gov/~kewu/ps/45577.html

  Args:
    X : An `(n, k)` array which describes a linear subspace of R^n, possibly
        numerically degenerate with some rank less than `k`.

  Returns:
    An orthonormal space `V` described by a `(n, k)` array, with trailing
    columns possibly zeroed out if `X` is of low rank.
  """

  # In [1] diagonal conditioning is explicit, but by normalizing first
  # we can simplify the formulas a bit, since then diagonal conditioning
  # becomes a no-op.
  norms = jnp.linalg.norm(X, ord=2, axis=0, keepdims=True)
  X /= jnp.where(norms == 0, 1.0, norms)

  inner = _mm(X.T, X)

  w, V = _eigh_ascending(inner)

  # All mask logic is used to avoid divide-by-zeros when input columns
  # may have been zero or new zero columns introduced from truncation.
  #
  # If an eigenvalue is less than max eigvalue * eps, then consider
  # that direction "degenerate".
  tau = jnp.finfo(X.dtype).eps * w[0]
  padded = jnp.maximum(w, tau)

  # Note the tau == 0 edge case where X was all zeros.
  sqrted = jnp.where(tau > 0, padded, 1.0) ** (-0.5)

  # X^T X = V diag(w) V^T, so
  # W = X V diag(w)^(-1/2) will yield W^T W = I (excerpting zeros).
  scaledV = V * sqrted[jnp.newaxis, :]
  orthoX = _mm(X, scaledV)

  keep = ((w > tau) * (jnp.diag(inner) > 0.0))[jnp.newaxis, :]
  orthoX *= keep.astype(orthoX.dtype)
  norms = jnp.linalg.norm(orthoX, ord=2, axis=0, keepdims=True)
  keep *= norms > 0.0
  orthoX /= jnp.where(keep, norms, 1.0)
  return orthoX


def _project_out(basis, U):
  """Derives component of U in the orthogonal complement of basis.

  This method iteratively subtracts out the basis component and orthonormalizes
  the remainder. To an extent, these two operations can oppose each other
  when the remainder norm is near-zero (since normalization enlarges a vector
  which may possibly lie in the subspace `basis` to be subtracted).

  We make sure to prioritize orthogonality between `basis` and `U`, favoring
  to return a lower-rank space thank `rank(U)`, in this tradeoff.

  Args:
    basis : An `(n, m)` array which describes a linear subspace of R^n, this
        is assumed to be orthonormal but zero columns are allowed.
    U : An `(n, k)` array representing another subspace of R^n, whose `basis`
        component is to be projected out.

  Returns:
    An `(n, k)` array, with some columns possibly zeroed out, representing
    the component of `U` in the complement of `basis`. The nonzero columns
    are mutually orthonormal.
  """

  # See Sec. 6.9 of The Symmetric Eigenvalue Problem by Beresford Parlett [1]
  # which motivates two loop iterations for basis subtraction. This
  # "twice is enough" approach is due to Kahan. See also a practical note
  # by SLEPc developers [2].
  #
  # Interspersing with orthonormalization isn't directly grounded in the
  # original analysis, but taken from Algorithm 5 of [3]. In practice, due to
  # normalization, I have noticed that the orthonormalized basis
  # does not always end up as a subspace of the starting basis in practice.
  # There may be room to refine this procedure further, but the adjustment
  # in the subsequent block handles this edge case well enough for now.
  #
  # [1]: https://epubs.siam.org/doi/abs/10.1137/1.9781611971163
  # [2]: http://slepc.upv.es/documentation/reports/str1.pdf
  # [3]: https://arxiv.org/abs/1704.07458
  for _ in range(2):
    U -= _mm(basis, _mm(basis.T, U))
    U = _orthonormalize(U)

  # It's crucial to end on a subtraction of the original basis.
  # This seems to be a detail not present in [2], possibly because of
  # of reliance on soft locking.
  #
  # Near convergence, if the residuals R are 0 and our last
  # operation when projecting (X, P) out from R is the orthonormalization
  # done above, then due to catastrophic cancellation we may re-introduce
  # (X, P) subspace components into U, which can ruin the Rayleigh-Ritz
  # conditioning.
  #
  # We zero out any columns that are even remotely suspicious, so the invariant
  # that [basis, U] is zero-or-orthogonal is ensured.
  for _ in range(2):
    U -= _mm(basis, _mm(basis.T, U))
  normU = jnp.linalg.norm(U, ord=2, axis=0, keepdims=True)
  U *= (normU >= 0.99).astype(U.dtype)

  return U

def _orthonormalize(basis):
  # Twice is enough, again.
  for _ in range(2):
    basis = _svqb(basis)
  return basis


def _rayleigh_ritz_orth(A, S):
  """Solve the Rayleigh-Ritz problem for `A` projected to `S`.

  Solves the local eigenproblem for `A` within the subspace `S`, which is
  assumed to be orthonormal (with zero columns allowed), identifying `w, V`
  satisfying

  (1) `S.T A S V ~= diag(w) V`
  (2) `V` is standard orthonormal

  Note that (2) is simplified to be standard orthonormal because `S` is.

  Args:
    A: An operator representing the action of an `n`-sized square matrix.
    S: An orthonormal subspace of R^n represented by an `(n, k)` array, with
       zero columns allowed.

  Returns:
    Eigenvectors `V` and eigenvalues `w` satisfying the size-`k` system
    described in this method doc. Note `V` will be full rank, even if `S` isn't.
  """

  SAS = _mm(S.T, A(S))

  # Solve the projected subsystem.
  # If we could tell to eigh to stop after first k, we would.
  return _eigh_ascending(SAS)


def _extend_basis(X, m):
  """Extend the basis of `X` with `m` addition dimensions.

  Given an orthonormal `X` of dimension `k`, a typical strategy for deriving
  an extended basis is to generate a random one and project it out.

  We instead generate a basis using block householder reflectors [1] [2] to
  leverage the favorable properties of determinism and avoiding the chance that
  the generated random basis has overlap with the starting basis, which may
  happen with non-negligible probability in low-dimensional cases.

  [1]: https://epubs.siam.org/doi/abs/10.1137/0725014
  [2]: https://www.jstage.jst.go.jp/article/ipsjdc/2/0/2_0_298/_article

  Args:
    X : An `(n, k)` array representing a `k`-rank orthonormal basis for a linear
        subspace of R^n.
    m : A nonnegative integer such that `k + m <= n` telling us how much to
        extend the basis by.

  Returns:
    An `(n, m)` array representing an extension to the basis of `X` such that
    their union is orthonormal.
  """
  n, k = X.shape
  # X = vstack(Xupper, Xlower), where Xupper is (k, k)
  Xupper, Xlower = jnp.split(X, [k], axis=0)
  u, s, vt = jnp.linalg.svd(Xupper)

  # Adding U V^T to Xupper won't change its row or column space, but notice
  # its singular values are all lifted by 1; we could write its upper k rows
  # as u diag(1 + s) vt.
  y = jnp.concatenate([Xupper + _mm(u, vt), Xlower], axis=0)

  # Suppose we found a full-rank (n, k) matrix w which defines the
  # perpendicular to a space we'd like to reflect over. The block householder
  # reflector H(w) would have the usual involution property.
  #
  # Consider the two definitions below:
  # H(w) = I - 2 w w^T
  # 2 w w^T = y (v diag(1+s)^(-1) vt) y^T
  #
  # After some algebra, we see H(w) X = vstack(-u vt, 0)
  # Applying H(w) to both sides since H(w)^2 = I we have
  # X = H(w) vstack(-u vt, 0). But since H(w) is unitary its action must
  # preserve rank. Thus H(w) vstack(0, eye(n - k)) must be orthogonal to
  # X; taking just the first m columns H(w) vstack(0, eye(m), 0) yields
  # an orthogonal extension to X.
  other = jnp.concatenate(
      [jnp.eye(m, dtype=X.dtype),
       jnp.zeros((n - k - m, m), dtype=X.dtype)], axis=0)
  w = _mm(y, vt.T * ((2 * (1 + s)) ** (-1/2))[jnp.newaxis, :])
  h = -2 * jnp.linalg.multi_dot(
      [w, w[k:, :].T, other], precision=jax.lax.Precision.HIGHEST)
  return h.at[k:].add(other)


# Sparse direct solve via QR factorization
def _spsolve_abstract_eval(data, indices, indptr, b, *, tol, reorder):
  if data.dtype != b.dtype:
    raise ValueError(f"data types do not match: {data.dtype=} {b.dtype=}")
  if not (jnp.issubdtype(indices.dtype, jnp.integer) and jnp.issubdtype(indptr.dtype, jnp.integer)):
    raise ValueError(f"index arrays must be integer typed; got {indices.dtype=} {indptr.dtype=}")
  if not data.ndim == indices.ndim == indptr.ndim == b.ndim == 1:
    raise ValueError("Arrays must be one-dimensional. "
                     f"Got {data.shape=} {indices.shape=} {indptr.shape=} {b.shape=}")
  if indptr.size != b.size + 1 or  data.shape != indices.shape:
    raise ValueError(f"Invalid CSR buffer sizes: {data.shape=} {indices.shape=} {indptr.shape=}")
  if reorder not in [0, 1, 2, 3]:
    raise ValueError(f"{reorder=} not valid, must be one of [1, 2, 3, 4]")
  tol = float(tol)
  return b


def _spsolve_gpu_lowering(ctx, data, indices, indptr, b, *, tol, reorder):
  data_aval, _, _, _, = ctx.avals_in
  return gpu_solver.cuda_csrlsvqr(data_aval.dtype, data, indices,
                                  indptr, b, tol, reorder)


def _spsolve_cpu_lowering(ctx, data, indices, indptr, b, tol, reorder):
  del tol, reorder
  args = [data, indices, indptr, b]

  def _callback(data, indices, indptr, b, **kwargs):
    A = csr_matrix((data, indices, indptr), shape=(b.size, b.size))
    return (linalg.spsolve(A, b).astype(b.dtype),)

  result, _, _ = mlir.emit_python_callback(
      ctx, _callback, None, args, ctx.avals_in, ctx.avals_out,
      has_side_effect=False)
  return result


def _spsolve_jvp_lhs(data_dot, data, indices, indptr, b, **kwds):
    # d/dM M^-1 b = M^-1 M_dot M^-1 b
    p = spsolve(data, indices, indptr, b, **kwds)
    q = sparse.csr_matvec_p.bind(data_dot, indices, indptr, p,
                                 shape=(indptr.size - 1, len(b)),
                                 transpose=False)
    return -spsolve(data, indices, indptr, q, **kwds)


def _spsolve_jvp_rhs(b_dot, data, indices, indptr, b, **kwds):
    # d/db M^-1 b = M^-1 b_dot
    return spsolve(data, indices, indptr, b_dot, **kwds)


def _csr_transpose(data, indices, indptr):
  # Transpose of a square CSR matrix
  m = indptr.size - 1
  row = jnp.cumsum(jnp.zeros_like(indices).at[indptr].add(1)) - 1
  row_T, indices_T, data_T = jax.lax.sort((indices, row, data), num_keys=2)
  indptr_T = jnp.zeros_like(indptr).at[1:].set(
      jnp.cumsum(jnp.bincount(row_T, length=m)).astype(indptr.dtype))
  return data_T, indices_T, indptr_T


def _spsolve_transpose(ct, data, indices, indptr, b, **kwds):
  assert not ad.is_undefined_primal(indices)
  assert not ad.is_undefined_primal(indptr)
  if ad.is_undefined_primal(b):
    # TODO(jakevdp): can we do this without an explicit transpose?
    data_T, indices_T, indptr_T = _csr_transpose(data, indices, indptr)
    ct_out = spsolve(data_T, indices_T, indptr_T, ct, **kwds)
    return data, indices, indptr, ct_out
  else:
    # Should never reach here, because JVP is linear wrt data.
    raise NotImplementedError("spsolve transpose with respect to data")


spsolve_p = core.Primitive('spsolve')
spsolve_p.def_impl(functools.partial(xla.apply_primitive, spsolve_p))
spsolve_p.def_abstract_eval(_spsolve_abstract_eval)
ad.defjvp(spsolve_p, _spsolve_jvp_lhs, None, None, _spsolve_jvp_rhs)
ad.primitive_transposes[spsolve_p] = _spsolve_transpose
mlir.register_lowering(spsolve_p, _spsolve_gpu_lowering, platform='cuda')
mlir.register_lowering(spsolve_p, _spsolve_cpu_lowering, platform='cpu')


def spsolve(data, indices, indptr, b, tol=1e-6, reorder=1):
  """A sparse direct solver using QR factorization.

  Accepts a sparse matrix in CSR format `data, indices, indptr` arrays.
  Currently only the CUDA GPU backend is implemented, the CPU backend will fall
  back to `scipy.sparse.linalg.spsolve`. Neither the CPU nor the GPU
  implementation support batching with `vmap`.

  Args:
    data : An array containing the non-zero entries of the CSR matrix.
    indices : The column indices of the CSR matrix.
    indptr : The row pointer array of the CSR matrix.
    b : The right hand side of the linear system.
    tol : Tolerance to decide if singular or not. Defaults to 1e-6.
    reorder : The reordering scheme to use to reduce fill-in. No reordering if
      ``reorder=0``. Otherwise, symrcm, symamd, or csrmetisnd (``reorder=1,2,3``),
      respectively. Defaults to symrcm.

  Returns:
    An array with the same dtype and size as b representing the solution to
    the sparse linear system.
  """
  return spsolve_p.bind(data, indices, indptr, b, tol=tol, reorder=reorder)
