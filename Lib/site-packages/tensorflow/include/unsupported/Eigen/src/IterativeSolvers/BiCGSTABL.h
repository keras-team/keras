// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 Chris Schoutrop <c.e.m.schoutrop@tue.nl>
// Copyright (C) 2020 Jens Wehner <j.wehner@esciencecenter.nl>
// Copyright (C) 2020 Jan van Dijk <j.v.dijk@tue.nl>
// Copyright (C) 2020 Adithya Vijaykumar
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*

  This implementation of BiCGStab(L) is based on the papers
      General algorithm:
      1. G.L.G. Sleijpen, D.R. Fokkema. (1993). BiCGstab(l) for linear equations
  involving unsymmetric matrices with complex spectrum. Electronic Transactions
  on Numerical Analysis. Polynomial step update:
      2. G.L.G. Sleijpen, M.B. Van Gijzen. (2010) Exploiting BiCGstab(l)
  strategies to induce dimension reduction SIAM Journal on Scientific Computing.
      3. Fokkema, Diederik R. Enhanced implementation of BiCGstab (l) for
  solving linear systems of equations. Universiteit Utrecht. Mathematisch
  Instituut, 1996
      4. Sleijpen, G. L., & van der Vorst, H. A. (1996). Reliable updated
  residuals in hybrid Bi-CG methods. Computing, 56(2), 141-163.
*/

#ifndef EIGEN_BICGSTABL_H
#define EIGEN_BICGSTABL_H

namespace Eigen {

namespace internal {
/**     \internal Low-level bi conjugate gradient stabilized algorithm with L
   additional residual minimization steps \param mat The matrix A \param rhs The
   right hand side vector b \param x On input and initial solution, on output
   the computed solution. \param precond A preconditioner being able to
   efficiently solve for an approximation of Ax=b (regardless of b) \param iters
   On input the max number of iteration, on output the number of performed
   iterations. \param tol_error On input the tolerance error, on output an
   estimation of the relative error. \param L On input Number of additional
   GMRES steps to take. If L is too large (~20) instabilities occur. \return
   false in the case of numerical issue, for example a break down of BiCGSTABL.
*/
template <typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
bool bicgstabl(const MatrixType &mat, const Rhs &rhs, Dest &x, const Preconditioner &precond, Index &iters,
               typename Dest::RealScalar &tol_error, Index L) {
  using numext::abs;
  using numext::sqrt;
  typedef typename Dest::RealScalar RealScalar;
  typedef typename Dest::Scalar Scalar;
  const Index N = rhs.size();
  L = L < x.rows() ? L : x.rows();

  Index k = 0;

  const RealScalar tol = tol_error;
  const Index maxIters = iters;

  typedef Matrix<Scalar, Dynamic, 1> VectorType;
  typedef Matrix<Scalar, Dynamic, Dynamic, ColMajor> DenseMatrixType;

  DenseMatrixType rHat(N, L + 1);
  DenseMatrixType uHat(N, L + 1);

  // We start with an initial guess x_0 and let us set r_0 as (residual
  // calculated from x_0)
  VectorType x0 = x;
  rHat.col(0) = rhs - mat * x0;  // r_0

  x.setZero();  // This will contain the updates to the solution.
  // rShadow is arbritary, but must never be orthogonal to any residual.
  VectorType rShadow = VectorType::Random(N);

  VectorType x_prime = x;

  // Redundant: x is already set to 0
  // x.setZero();
  VectorType b_prime = rHat.col(0);

  // Other vectors and scalars initialization
  Scalar rho0 = 1.0;
  Scalar alpha = 0.0;
  Scalar omega = 1.0;

  uHat.col(0).setZero();

  bool bicg_convergence = false;

  const RealScalar normb = rhs.stableNorm();
  if (internal::isApprox(normb, RealScalar(0))) {
    x.setZero();
    iters = 0;
    return true;
  }
  RealScalar normr = rHat.col(0).stableNorm();
  RealScalar Mx = normr;
  RealScalar Mr = normr;

  // Keep track of the solution with the lowest residual
  RealScalar normr_min = normr;
  VectorType x_min = x_prime + x;

  // Criterion for when to apply the group-wise update, conform ref 3.
  const RealScalar delta = 0.01;

  bool compute_res = false;
  bool update_app = false;

  while (normr > tol * normb && k < maxIters) {
    rho0 *= -omega;

    for (Index j = 0; j < L; ++j) {
      const Scalar rho1 = rShadow.dot(rHat.col(j));

      if (!(numext::isfinite)(rho1) || rho0 == RealScalar(0.0)) {
        // We cannot continue computing, return the best solution found.
        x += x_prime;

        // Check if x is better than the best stored solution thus far.
        normr = (rhs - mat * (precond.solve(x) + x0)).stableNorm();

        if (normr > normr_min || !(numext::isfinite)(normr)) {
          // x_min is a better solution than x, return x_min
          x = x_min;
          normr = normr_min;
        }
        tol_error = normr / normb;
        iters = k;
        // x contains the updates to x0, add those back to obtain the solution
        x = precond.solve(x);
        x += x0;
        return (normr < tol * normb);
      }

      const Scalar beta = alpha * (rho1 / rho0);
      rho0 = rho1;
      // Update search directions
      uHat.leftCols(j + 1) = rHat.leftCols(j + 1) - beta * uHat.leftCols(j + 1);
      uHat.col(j + 1) = mat * precond.solve(uHat.col(j));
      const Scalar sigma = rShadow.dot(uHat.col(j + 1));
      alpha = rho1 / sigma;
      // Update residuals
      rHat.leftCols(j + 1) -= alpha * uHat.middleCols(1, j + 1);
      rHat.col(j + 1) = mat * precond.solve(rHat.col(j));
      // Complete BiCG iteration by updating x
      x += alpha * uHat.col(0);
      normr = rHat.col(0).stableNorm();
      // Check for early exit
      if (normr < tol * normb) {
        /*
          Convergence was achieved during BiCG step.
          Without this check BiCGStab(L) fails for trivial matrices, such as
          when the preconditioner already is the inverse, or the input matrix is
          identity.
        */
        bicg_convergence = true;
        break;
      } else if (normr < normr_min) {
        // We found an x with lower residual, keep this one.
        x_min = x + x_prime;
        normr_min = normr;
      }
    }
    if (!bicg_convergence) {
      /*
        The polynomial/minimize residual step.

        QR Householder method for argmin is more stable than (modified)
        Gram-Schmidt, in the sense that there is less loss of orthogonality. It
        is more accurate than solving the normal equations, since the normal
        equations scale with condition number squared.
      */
      const VectorType gamma = rHat.rightCols(L).householderQr().solve(rHat.col(0));
      x += rHat.leftCols(L) * gamma;
      rHat.col(0) -= rHat.rightCols(L) * gamma;
      uHat.col(0) -= uHat.rightCols(L) * gamma;
      normr = rHat.col(0).stableNorm();
      omega = gamma(L - 1);
    }
    if (normr < normr_min) {
      // We found an x with lower residual, keep this one.
      x_min = x + x_prime;
      normr_min = normr;
    }

    k++;

    /*
      Reliable update part

      The recursively computed residual can deviate from the actual residual
      after several iterations. However, computing the residual from the
      definition costs extra MVs and should not be done at each iteration. The
      reliable update strategy computes the true residual from the definition:
      r=b-A*x at strategic intervals. Furthermore a "group wise update" strategy
      is used to combine updates, which improves accuracy.
    */

    // Maximum norm of residuals since last update of x.
    Mx = numext::maxi(Mx, normr);
    // Maximum norm of residuals since last computation of the true residual.
    Mr = numext::maxi(Mr, normr);

    if (normr < delta * normb && normb <= Mx) {
      update_app = true;
    }

    if (update_app || (normr < delta * Mr && normb <= Mr)) {
      compute_res = true;
    }

    if (bicg_convergence) {
      update_app = true;
      compute_res = true;
      bicg_convergence = false;
    }

    if (compute_res) {
      // Explicitly compute residual from the definition

      // This is equivalent to the shifted version of rhs - mat *
      // (precond.solve(x)+x0)
      rHat.col(0) = b_prime - mat * precond.solve(x);
      normr = rHat.col(0).stableNorm();
      Mr = normr;

      if (update_app) {
        // After the group wise update, the original problem is translated to a
        // shifted one.
        x_prime += x;
        x.setZero();
        b_prime = rHat.col(0);
        Mx = normr;
      }
    }
    if (normr < normr_min) {
      // We found an x with lower residual, keep this one.
      x_min = x + x_prime;
      normr_min = normr;
    }

    compute_res = false;
    update_app = false;
  }

  // Convert internal variable to the true solution vector x
  x += x_prime;

  normr = (rhs - mat * (precond.solve(x) + x0)).stableNorm();
  if (normr > normr_min || !(numext::isfinite)(normr)) {
    // x_min is a better solution than x, return x_min
    x = x_min;
    normr = normr_min;
  }
  tol_error = normr / normb;
  iters = k;

  // x contains the updates to x0, add those back to obtain the solution
  x = precond.solve(x);
  x += x0;
  return true;
}

}  // namespace internal

template <typename MatrixType_, typename Preconditioner_ = DiagonalPreconditioner<typename MatrixType_::Scalar>>
class BiCGSTABL;

namespace internal {

template <typename MatrixType_, typename Preconditioner_>
struct traits<Eigen::BiCGSTABL<MatrixType_, Preconditioner_>> {
  typedef MatrixType_ MatrixType;
  typedef Preconditioner_ Preconditioner;
};

}  // namespace internal

template <typename MatrixType_, typename Preconditioner_>
class BiCGSTABL : public IterativeSolverBase<BiCGSTABL<MatrixType_, Preconditioner_>> {
  typedef IterativeSolverBase<BiCGSTABL> Base;
  using Base::m_error;
  using Base::m_info;
  using Base::m_isInitialized;
  using Base::m_iterations;
  using Base::matrix;
  Index m_L;

 public:
  typedef MatrixType_ MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Preconditioner_ Preconditioner;

  /** Default constructor. */
  BiCGSTABL() : m_L(2) {}

  /**
  Initialize the solver with matrix \a A for further \c Ax=b solving.

  This constructor is a shortcut for the default constructor followed
  by a call to compute().

  \warning this class stores a reference to the matrix A as well as some
  precomputed values that depend on it. Therefore, if \a A is changed
  this class becomes invalid. Call compute() to update it with the new
  matrix A, or modify a copy of A.
  */
  template <typename MatrixDerived>
  explicit BiCGSTABL(const EigenBase<MatrixDerived> &A) : Base(A.derived()), m_L(2) {}

  /** \internal */
  /** Loops over the number of columns of b and does the following:
    1. sets the tolerence and maxIterations
    2. Calls the function that has the core solver routine
  */
  template <typename Rhs, typename Dest>
  void _solve_vector_with_guess_impl(const Rhs &b, Dest &x) const {
    m_iterations = Base::maxIterations();

    m_error = Base::m_tolerance;

    bool ret = internal::bicgstabl(matrix(), b, x, Base::m_preconditioner, m_iterations, m_error, m_L);
    m_info = (!ret) ? NumericalIssue : m_error <= Base::m_tolerance ? Success : NoConvergence;
  }

  /** Sets the parameter L, indicating how many minimize residual steps are
   * used. Default: 2 */
  void setL(Index L) {
    eigen_assert(L >= 1 && "L needs to be positive");
    m_L = L;
  }
};

}  // namespace Eigen

#endif /* EIGEN_BICGSTABL_H */
