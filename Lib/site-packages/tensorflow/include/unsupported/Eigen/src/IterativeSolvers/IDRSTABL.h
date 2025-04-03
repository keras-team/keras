// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 Chris Schoutrop <c.e.m.schoutrop@tue.nl>
// Copyright (C) 2020 Mischa Senders <m.j.senders@student.tue.nl>
// Copyright (C) 2020 Lex Kuijpers <l.kuijpers@student.tue.nl>
// Copyright (C) 2020 Jens Wehner <j.wehner@esciencecenter.nl>
// Copyright (C) 2020 Jan van Dijk <j.v.dijk@tue.nl>
// Copyright (C) 2020 Adithya Vijaykumar
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
/*

The IDR(S)Stab(L) method is a combination of IDR(S) and BiCGStab(L)

This implementation of IDRSTABL is based on
1. Aihara, K., Abe, K., & Ishiwata, E. (2014). A variant of IDRstab with
reliable update strategies for solving sparse linear systems. Journal of
Computational and Applied Mathematics, 259, 244-258.
   doi:10.1016/j.cam.2013.08.028
                2. Aihara, K., Abe, K., & Ishiwata, E. (2015). Preconditioned
IDRSTABL Algorithms for Solving Nonsymmetric Linear Systems. International
Journal of Applied Mathematics, 45(3).
                3. Saad, Y. (2003). Iterative Methods for Sparse Linear Systems:
Second Edition. Philadelphia, PA: SIAM.
                4. Sonneveld, P., & Van Gijzen, M. B. (2009). IDR(s): A Family
of Simple and Fast Algorithms for Solving Large Nonsymmetric Systems of Linear
Equations. SIAM Journal on Scientific Computing, 31(2), 1035-1062.
   doi:10.1137/070685804
                5. Sonneveld, P. (2012). On the convergence behavior of IDR (s)
and related methods. SIAM Journal on Scientific Computing, 34(5), A2576-A2598.

    Right-preconditioning based on Ref. 3 is implemented here.
*/

#ifndef EIGEN_IDRSTABL_H
#define EIGEN_IDRSTABL_H

namespace Eigen {

namespace internal {

template <typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
bool idrstabl(const MatrixType &mat, const Rhs &rhs, Dest &x, const Preconditioner &precond, Index &iters,
              typename Dest::RealScalar &tol_error, Index L, Index S) {
  /*
    Setup and type definitions.
  */
  using numext::abs;
  using numext::sqrt;
  typedef typename Dest::Scalar Scalar;
  typedef typename Dest::RealScalar RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;
  typedef Matrix<Scalar, Dynamic, Dynamic, ColMajor> DenseMatrixType;

  const Index N = x.rows();

  Index k = 0;  // Iteration counter
  const Index maxIters = iters;

  const RealScalar rhs_norm = rhs.stableNorm();
  const RealScalar tol = tol_error * rhs_norm;

  if (rhs_norm == 0) {
    /*
      If b==0, then the exact solution is x=0.
      rhs_norm is needed for other calculations anyways, this exit is a freebie.
    */
    x.setZero();
    tol_error = 0.0;
    return true;
  }
  // Construct decomposition objects beforehand.
  FullPivLU<DenseMatrixType> lu_solver;

  if (S >= N || L >= N) {
    /*
      The matrix is very small, or the choice of L and S is very poor
      in that case solving directly will be best.
    */
    lu_solver.compute(DenseMatrixType(mat));
    x = lu_solver.solve(rhs);
    tol_error = (rhs - mat * x).stableNorm() / rhs_norm;
    return true;
  }

  // Define maximum sizes to prevent any reallocation later on.
  DenseMatrixType u(N, L + 1);
  DenseMatrixType r(N, L + 1);

  DenseMatrixType V(N * (L + 1), S);

  VectorType alpha(S);
  VectorType gamma(L);
  VectorType update(N);

  /*
    Main IDRSTABL algorithm
  */
  // Set up the initial residual
  VectorType x0 = x;
  r.col(0) = rhs - mat * x;
  x.setZero();  // The final solution will be x0+x

  tol_error = r.col(0).stableNorm();

  // FOM = Full orthogonalisation method
  DenseMatrixType h_FOM = DenseMatrixType::Zero(S, S - 1);

  // Construct an initial U matrix of size N x S
  DenseMatrixType U(N * (L + 1), S);
  for (Index col_index = 0; col_index < S; ++col_index) {
    // Arnoldi-like process to generate a set of orthogonal vectors spanning
    // {u,A*u,A*A*u,...,A^(S-1)*u}. This construction can be combined with the
    // Full Orthogonalization Method (FOM) from Ref.3 to provide a possible
    // early exit with no additional MV.
    if (col_index != 0) {
      /*
      Modified Gram-Schmidt strategy:
      */
      VectorType w = mat * precond.solve(u.col(0));
      for (Index i = 0; i < col_index; ++i) {
        auto v = U.col(i).head(N);
        h_FOM(i, col_index - 1) = v.dot(w);
        w -= h_FOM(i, col_index - 1) * v;
      }
      u.col(0) = w;
      h_FOM(col_index, col_index - 1) = u.col(0).stableNorm();

      if (abs(h_FOM(col_index, col_index - 1)) != RealScalar(0)) {
        /*
        This only happens if u is NOT exactly zero. In case it is exactly zero
        it would imply that that this u has no component in the direction of the
        current residual.

        By then setting u to zero it will not contribute any further (as it
        should). Whereas attempting to normalize results in division by zero.

        Such cases occur if:
        1. The basis of dimension <S is sufficient to exactly solve the linear
        system. I.e. the current residual is in span{r,Ar,...A^{m-1}r}, where
        (m-1)<=S.
        2. Two vectors vectors generated from r, Ar,... are (numerically)
        parallel.

        In case 1, the exact solution to the system can be obtained from the
        "Full Orthogonalization Method" (Algorithm 6.4 in the book of Saad),
        without any additional MV.

        Contrary to what one would suspect, the comparison with ==0.0 for
        floating-point types is intended here. Any arbritary non-zero u is fine
        to continue, however if u contains either NaN or Inf the algorithm will
        break down.
        */
        u.col(0) /= h_FOM(col_index, col_index - 1);
      }
    } else {
      u.col(0) = r.col(0);
      u.col(0).normalize();
    }

    U.col(col_index).head(N) = u.col(0);
  }

  if (S > 1) {
    // Check for early FOM exit.
    Scalar beta = r.col(0).stableNorm();
    VectorType e1 = VectorType::Zero(S - 1);
    e1(0) = beta;
    lu_solver.compute(h_FOM.topLeftCorner(S - 1, S - 1));
    VectorType y = lu_solver.solve(e1);
    VectorType x2 = x + U.topLeftCorner(N, S - 1) * y;

    // Using proposition 6.7 in Saad, one MV can be saved to calculate the
    // residual
    RealScalar FOM_residual = (h_FOM(S - 1, S - 2) * y(S - 2) * U.col(S - 1).head(N)).stableNorm();

    if (FOM_residual < tol) {
      // Exit, the FOM algorithm was already accurate enough
      iters = k;
      // Convert back to the unpreconditioned solution
      x = precond.solve(x2);
      // x contains the updates to x0, add those back to obtain the solution
      x += x0;
      tol_error = FOM_residual / rhs_norm;
      return true;
    }
  }

  /*
    Select an initial (N x S) matrix R0.
    1. Generate random R0, orthonormalize the result.
    2. This results in R0, however to save memory and compute we only need the
    adjoint of R0. This is given by the matrix R_T.\ Additionally, the matrix
    (mat.adjoint()*R_tilde).adjoint()=R_tilde.adjoint()*mat by the
    anti-distributivity property of the adjoint. This results in AR_T, which is
    constant if R_T does not have to be regenerated and can be precomputed.
    Based on reference 4, this has zero probability in exact arithmetic.
  */

  // Original IDRSTABL and Kensuke choose S random vectors:
  const HouseholderQR<DenseMatrixType> qr(DenseMatrixType::Random(N, S));
  DenseMatrixType R_T = (qr.householderQ() * DenseMatrixType::Identity(N, S)).adjoint();
  DenseMatrixType AR_T = DenseMatrixType(R_T * mat);

  // Pre-allocate sigma.
  DenseMatrixType sigma(S, S);

  bool reset_while = false;  // Should the while loop be reset for some reason?

  while (k < maxIters) {
    for (Index j = 1; j <= L; ++j) {
      /*
        The IDR Step
      */
      // Construction of the sigma-matrix, and the decomposition of sigma.
      for (Index i = 0; i < S; ++i) {
        sigma.col(i).noalias() = AR_T * precond.solve(U.block(N * (j - 1), i, N, 1));
      }

      lu_solver.compute(sigma);
      // Obtain the update coefficients alpha
      if (j == 1) {
        // alpha=inverse(sigma)*(R_T*r_0);
        alpha.noalias() = lu_solver.solve(R_T * r.col(0));
      } else {
        // alpha=inverse(sigma)*(AR_T*r_{j-2})
        alpha.noalias() = lu_solver.solve(AR_T * precond.solve(r.col(j - 2)));
      }

      // Obtain new solution and residual from this update
      update.noalias() = U.topRows(N) * alpha;
      r.col(0) -= mat * precond.solve(update);
      x += update;

      for (Index i = 1; i <= j - 2; ++i) {
        // This only affects the case L>2
        r.col(i) -= U.block(N * (i + 1), 0, N, S) * alpha;
      }
      if (j > 1) {
        // r=[r;A*r_{j-2}]
        r.col(j - 1).noalias() = mat * precond.solve(r.col(j - 2));
      }
      tol_error = r.col(0).stableNorm();

      if (tol_error < tol) {
        // If at this point the algorithm has converged, exit.
        reset_while = true;
        break;
      }

      bool break_normalization = false;
      for (Index q = 1; q <= S; ++q) {
        if (q == 1) {
          // u = r;
          u.leftCols(j + 1) = r.leftCols(j + 1);
        } else {
          // u=[u_1;u_2;...;u_j]
          u.leftCols(j) = u.middleCols(1, j);
        }

        // Obtain the update coefficients beta implicitly
        // beta=lu_sigma.solve(AR_T * u.block(N * (j - 1), 0, N, 1)
        u.reshaped().head(u.rows() * j) -= U.topRows(N * j) * lu_solver.solve(AR_T * precond.solve(u.col(j - 1)));

        // u=[u;Au_{j-1}]
        u.col(j).noalias() = mat * precond.solve(u.col(j - 1));

        // Orthonormalize u_j to the columns of V_j(:,1:q-1)
        if (q > 1) {
          /*
          Modified Gram-Schmidt-like procedure to make u orthogonal to the
          columns of V from Ref. 1.

          The vector mu from Ref. 1 is obtained implicitly:
          mu=V.block(N * j, 0, N, q - 1).adjoint() * u.block(N * j, 0, N, 1).
          */
          for (Index i = 0; i <= q - 2; ++i) {
            auto v = V.col(i).segment(N * j, N);
            Scalar h = v.squaredNorm();
            h = v.dot(u.col(j)) / h;
            u.reshaped().head(u.rows() * (j + 1)) -= h * V.block(0, i, N * (j + 1), 1);
          }
        }
        // Normalize u and assign to a column of V
        Scalar normalization_constant = u.col(j).stableNorm();
        //  If u is exactly zero, this will lead to a NaN. Small, non-zero u is
        //  fine.
        if (normalization_constant == RealScalar(0.0)) {
          break_normalization = true;
          break;
        } else {
          u.leftCols(j + 1) /= normalization_constant;
        }

        V.block(0, q - 1, N * (j + 1), 1).noalias() = u.reshaped().head(u.rows() * (j + 1));
      }

      if (break_normalization == false) {
        U = V;
      }
    }
    if (reset_while) {
      break;
    }

    // r=[r;mat*r_{L-1}]
    r.col(L).noalias() = mat * precond.solve(r.col(L - 1));

    /*
            The polynomial step
    */
    ColPivHouseholderQR<DenseMatrixType> qr_solver(r.rightCols(L));
    gamma.noalias() = qr_solver.solve(r.col(0));

    // Update solution and residual using the "minimized residual coefficients"
    update.noalias() = r.leftCols(L) * gamma;
    x += update;
    r.col(0) -= mat * precond.solve(update);

    // Update iteration info
    ++k;
    tol_error = r.col(0).stableNorm();

    if (tol_error < tol) {
      // Slightly early exit by moving the criterion before the update of U,
      // after the main while loop the result of that calculation would not be
      // needed.
      break;
    }

    /*
    U=U0-sum(gamma_j*U_j)
    Consider the first iteration. Then U only contains U0, so at the start of
    the while-loop U should be U0. Therefore only the first N rows of U have to
    be updated.
    */
    for (Index i = 1; i <= L; ++i) {
      U.topRows(N) -= U.block(N * i, 0, N, S) * gamma(i - 1);
    }
  }

  /*
          Exit after the while loop terminated.
  */
  iters = k;
  // Convert back to the unpreconditioned solution
  x = precond.solve(x);
  // x contains the updates to x0, add those back to obtain the solution
  x += x0;
  tol_error = tol_error / rhs_norm;
  return true;
}

}  // namespace internal

template <typename MatrixType_, typename Preconditioner_ = DiagonalPreconditioner<typename MatrixType_::Scalar>>
class IDRSTABL;

namespace internal {

template <typename MatrixType_, typename Preconditioner_>
struct traits<IDRSTABL<MatrixType_, Preconditioner_>> {
  typedef MatrixType_ MatrixType;
  typedef Preconditioner_ Preconditioner;
};

}  // namespace internal

/** \ingroup IterativeLinearSolvers_Module
 * \brief The IDR(s)STAB(l) is a combination of IDR(s) and BiCGSTAB(l). It is a
 * short-recurrences Krylov method for sparse square problems. It can outperform
 * both IDR(s) and BiCGSTAB(l). IDR(s)STAB(l) generally closely follows the
 * optimal GMRES convergence in terms of the number of Matrix-Vector products.
 * However, without the increasing cost per iteration of GMRES. IDR(s)STAB(l) is
 * suitable for both indefinite systems and systems with complex eigenvalues.
 *
 * This class allows solving for A.x = b sparse linear problems. The vectors x
 * and b can be either dense or sparse.
 *
 * \tparam MatrixType_ the type of the sparse matrix A, can be a dense or a
 * sparse matrix. \tparam Preconditioner_ the type of the preconditioner.
 * Default is DiagonalPreconditioner
 *
 * \implsparsesolverconcept
 *
 * The maximum number of iterations and tolerance value can be controlled via
 * the setMaxIterations() and setTolerance() methods. The defaults are the size
 * of the problem for the maximum number of iterations and
 * NumTraits<Scalar>::epsilon() for the tolerance.
 *
 * The tolerance is the maximum relative residual error: |Ax-b|/|b| for which
 * the linear system is considered solved.
 *
 * \b Performance: When using sparse matrices, best performance is achieved for
 * a row-major sparse matrix format. Moreover, in this case multi-threading can
 * be exploited if the user code is compiled with OpenMP enabled. See \ref
 * TopicMultiThreading for details.
 *
 * By default the iterations start with x=0 as an initial guess of the solution.
 * One can control the start using the solveWithGuess() method.
 *
 * IDR(s)STAB(l) can also be used in a matrix-free context, see the following
 * \link MatrixfreeSolverExample example \endlink.
 *
 * \sa class SimplicialCholesky, DiagonalPreconditioner, IdentityPreconditioner
 */

template <typename MatrixType_, typename Preconditioner_>
class IDRSTABL : public IterativeSolverBase<IDRSTABL<MatrixType_, Preconditioner_>> {
  typedef IterativeSolverBase<IDRSTABL> Base;
  using Base::m_error;
  using Base::m_info;
  using Base::m_isInitialized;
  using Base::m_iterations;
  using Base::matrix;
  Index m_L;
  Index m_S;

 public:
  typedef MatrixType_ MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Preconditioner_ Preconditioner;

 public:
  /** Default constructor. */
  IDRSTABL() : m_L(2), m_S(4) {}

  /**   Initialize the solver with matrix \a A for further \c Ax=b solving.

  This constructor is a shortcut for the default constructor followed
  by a call to compute().

  \warning this class stores a reference to the matrix A as well as some
  precomputed values that depend on it. Therefore, if \a A is changed
  this class becomes invalid. Call compute() to update it with the new
  matrix A, or modify a copy of A.
          */
  template <typename MatrixDerived>
  explicit IDRSTABL(const EigenBase<MatrixDerived> &A) : Base(A.derived()), m_L(2), m_S(4) {}

  /** \internal */
  /**     Loops over the number of columns of b and does the following:
                                  1. sets the tolerance and maxIterations
                                  2. Calls the function that has the core solver
     routine
  */
  template <typename Rhs, typename Dest>
  void _solve_vector_with_guess_impl(const Rhs &b, Dest &x) const {
    m_iterations = Base::maxIterations();
    m_error = Base::m_tolerance;
    bool ret = internal::idrstabl(matrix(), b, x, Base::m_preconditioner, m_iterations, m_error, m_L, m_S);

    m_info = (!ret) ? NumericalIssue : m_error <= 10 * Base::m_tolerance ? Success : NoConvergence;
  }

  /** Sets the parameter L, indicating the amount of minimize residual steps are
   * used. */
  void setL(Index L) {
    eigen_assert(L >= 1 && "L needs to be positive");
    m_L = L;
  }
  /** Sets the parameter S, indicating the dimension of the shadow residual
   * space.. */
  void setS(Index S) {
    eigen_assert(S >= 1 && "S needs to be positive");
    m_S = S;
  }
};

}  // namespace Eigen

#endif /* EIGEN_IDRSTABL_H */
