// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2013-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_JACOBISVD_H
#define EIGEN_JACOBISVD_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// forward declaration (needed by ICC)
// the empty body is required by MSVC
template <typename MatrixType, int Options, bool IsComplex = NumTraits<typename MatrixType::Scalar>::IsComplex>
struct svd_precondition_2x2_block_to_be_real {};

/*** QR preconditioners (R-SVD)
 ***
 *** Their role is to reduce the problem of computing the SVD to the case of a square matrix.
 *** This approach, known as R-SVD, is an optimization for rectangular-enough matrices, and is a requirement for
 *** JacobiSVD which by itself is only able to work on square matrices.
 ***/

enum { PreconditionIfMoreColsThanRows, PreconditionIfMoreRowsThanCols };

template <typename MatrixType, int QRPreconditioner, int Case>
struct qr_preconditioner_should_do_anything {
  enum {
    a = MatrixType::RowsAtCompileTime != Dynamic && MatrixType::ColsAtCompileTime != Dynamic &&
        MatrixType::ColsAtCompileTime <= MatrixType::RowsAtCompileTime,
    b = MatrixType::RowsAtCompileTime != Dynamic && MatrixType::ColsAtCompileTime != Dynamic &&
        MatrixType::RowsAtCompileTime <= MatrixType::ColsAtCompileTime,
    ret = !((QRPreconditioner == NoQRPreconditioner) || (Case == PreconditionIfMoreColsThanRows && bool(a)) ||
            (Case == PreconditionIfMoreRowsThanCols && bool(b)))
  };
};

template <typename MatrixType, int Options, int QRPreconditioner, int Case,
          bool DoAnything = qr_preconditioner_should_do_anything<MatrixType, QRPreconditioner, Case>::ret>
struct qr_preconditioner_impl {};

template <typename MatrixType, int Options, int QRPreconditioner, int Case>
class qr_preconditioner_impl<MatrixType, Options, QRPreconditioner, Case, false> {
 public:
  void allocate(const JacobiSVD<MatrixType, Options>&) {}
  template <typename Xpr>
  bool run(JacobiSVD<MatrixType, Options>&, const Xpr&) {
    return false;
  }
};

/*** preconditioner using FullPivHouseholderQR ***/

template <typename MatrixType, int Options>
class qr_preconditioner_impl<MatrixType, Options, FullPivHouseholderQRPreconditioner, PreconditionIfMoreRowsThanCols,
                             true> {
 public:
  typedef typename MatrixType::Scalar Scalar;
  typedef JacobiSVD<MatrixType, Options> SVDType;

  enum { WorkspaceSize = MatrixType::RowsAtCompileTime, MaxWorkspaceSize = MatrixType::MaxRowsAtCompileTime };

  typedef Matrix<Scalar, 1, WorkspaceSize, RowMajor, 1, MaxWorkspaceSize> WorkspaceType;

  void allocate(const SVDType& svd) {
    if (svd.rows() != m_qr.rows() || svd.cols() != m_qr.cols()) {
      internal::destroy_at(&m_qr);
      internal::construct_at(&m_qr, svd.rows(), svd.cols());
    }
    if (svd.m_computeFullU) m_workspace.resize(svd.rows());
  }
  template <typename Xpr>
  bool run(SVDType& svd, const Xpr& matrix) {
    if (matrix.rows() > matrix.cols()) {
      m_qr.compute(matrix);
      svd.m_workMatrix = m_qr.matrixQR().block(0, 0, matrix.cols(), matrix.cols()).template triangularView<Upper>();
      if (svd.m_computeFullU) m_qr.matrixQ().evalTo(svd.m_matrixU, m_workspace);
      if (svd.computeV()) svd.m_matrixV = m_qr.colsPermutation();
      return true;
    }
    return false;
  }

 private:
  typedef FullPivHouseholderQR<MatrixType> QRType;
  QRType m_qr;
  WorkspaceType m_workspace;
};

template <typename MatrixType, int Options>
class qr_preconditioner_impl<MatrixType, Options, FullPivHouseholderQRPreconditioner, PreconditionIfMoreColsThanRows,
                             true> {
 public:
  typedef typename MatrixType::Scalar Scalar;
  typedef JacobiSVD<MatrixType, Options> SVDType;

  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
    MatrixOptions = traits<MatrixType>::Options
  };

  typedef typename internal::make_proper_matrix_type<Scalar, ColsAtCompileTime, RowsAtCompileTime, MatrixOptions,
                                                     MaxColsAtCompileTime, MaxRowsAtCompileTime>::type
      TransposeTypeWithSameStorageOrder;

  void allocate(const SVDType& svd) {
    if (svd.cols() != m_qr.rows() || svd.rows() != m_qr.cols()) {
      internal::destroy_at(&m_qr);
      internal::construct_at(&m_qr, svd.cols(), svd.rows());
    }
    if (svd.m_computeFullV) m_workspace.resize(svd.cols());
  }
  template <typename Xpr>
  bool run(SVDType& svd, const Xpr& matrix) {
    if (matrix.cols() > matrix.rows()) {
      m_qr.compute(matrix.adjoint());
      svd.m_workMatrix =
          m_qr.matrixQR().block(0, 0, matrix.rows(), matrix.rows()).template triangularView<Upper>().adjoint();
      if (svd.m_computeFullV) m_qr.matrixQ().evalTo(svd.m_matrixV, m_workspace);
      if (svd.computeU()) svd.m_matrixU = m_qr.colsPermutation();
      return true;
    } else
      return false;
  }

 private:
  typedef FullPivHouseholderQR<TransposeTypeWithSameStorageOrder> QRType;
  QRType m_qr;
  typename plain_row_type<MatrixType>::type m_workspace;
};

/*** preconditioner using ColPivHouseholderQR ***/

template <typename MatrixType, int Options>
class qr_preconditioner_impl<MatrixType, Options, ColPivHouseholderQRPreconditioner, PreconditionIfMoreRowsThanCols,
                             true> {
 public:
  typedef typename MatrixType::Scalar Scalar;
  typedef JacobiSVD<MatrixType, Options> SVDType;

  enum {
    WorkspaceSize = internal::traits<SVDType>::MatrixUColsAtCompileTime,
    MaxWorkspaceSize = internal::traits<SVDType>::MatrixUMaxColsAtCompileTime
  };

  typedef Matrix<Scalar, 1, WorkspaceSize, RowMajor, 1, MaxWorkspaceSize> WorkspaceType;

  void allocate(const SVDType& svd) {
    if (svd.rows() != m_qr.rows() || svd.cols() != m_qr.cols()) {
      internal::destroy_at(&m_qr);
      internal::construct_at(&m_qr, svd.rows(), svd.cols());
    }
    if (svd.m_computeFullU)
      m_workspace.resize(svd.rows());
    else if (svd.m_computeThinU)
      m_workspace.resize(svd.cols());
  }
  template <typename Xpr>
  bool run(SVDType& svd, const Xpr& matrix) {
    if (matrix.rows() > matrix.cols()) {
      m_qr.compute(matrix);
      svd.m_workMatrix = m_qr.matrixQR().block(0, 0, matrix.cols(), matrix.cols()).template triangularView<Upper>();
      if (svd.m_computeFullU)
        m_qr.householderQ().evalTo(svd.m_matrixU, m_workspace);
      else if (svd.m_computeThinU) {
        svd.m_matrixU.setIdentity(matrix.rows(), matrix.cols());
        m_qr.householderQ().applyThisOnTheLeft(svd.m_matrixU, m_workspace);
      }
      if (svd.computeV()) svd.m_matrixV = m_qr.colsPermutation();
      return true;
    }
    return false;
  }

 private:
  typedef ColPivHouseholderQR<MatrixType> QRType;
  QRType m_qr;
  WorkspaceType m_workspace;
};

template <typename MatrixType, int Options>
class qr_preconditioner_impl<MatrixType, Options, ColPivHouseholderQRPreconditioner, PreconditionIfMoreColsThanRows,
                             true> {
 public:
  typedef typename MatrixType::Scalar Scalar;
  typedef JacobiSVD<MatrixType, Options> SVDType;

  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
    MatrixOptions = internal::traits<MatrixType>::Options,
    WorkspaceSize = internal::traits<SVDType>::MatrixVColsAtCompileTime,
    MaxWorkspaceSize = internal::traits<SVDType>::MatrixVMaxColsAtCompileTime
  };

  typedef Matrix<Scalar, WorkspaceSize, 1, ColMajor, MaxWorkspaceSize, 1> WorkspaceType;

  typedef typename internal::make_proper_matrix_type<Scalar, ColsAtCompileTime, RowsAtCompileTime, MatrixOptions,
                                                     MaxColsAtCompileTime, MaxRowsAtCompileTime>::type
      TransposeTypeWithSameStorageOrder;

  void allocate(const SVDType& svd) {
    if (svd.cols() != m_qr.rows() || svd.rows() != m_qr.cols()) {
      internal::destroy_at(&m_qr);
      internal::construct_at(&m_qr, svd.cols(), svd.rows());
    }
    if (svd.m_computeFullV)
      m_workspace.resize(svd.cols());
    else if (svd.m_computeThinV)
      m_workspace.resize(svd.rows());
  }
  template <typename Xpr>
  bool run(SVDType& svd, const Xpr& matrix) {
    if (matrix.cols() > matrix.rows()) {
      m_qr.compute(matrix.adjoint());

      svd.m_workMatrix =
          m_qr.matrixQR().block(0, 0, matrix.rows(), matrix.rows()).template triangularView<Upper>().adjoint();
      if (svd.m_computeFullV)
        m_qr.householderQ().evalTo(svd.m_matrixV, m_workspace);
      else if (svd.m_computeThinV) {
        svd.m_matrixV.setIdentity(matrix.cols(), matrix.rows());
        m_qr.householderQ().applyThisOnTheLeft(svd.m_matrixV, m_workspace);
      }
      if (svd.computeU()) svd.m_matrixU = m_qr.colsPermutation();
      return true;
    } else
      return false;
  }

 private:
  typedef ColPivHouseholderQR<TransposeTypeWithSameStorageOrder> QRType;
  QRType m_qr;
  WorkspaceType m_workspace;
};

/*** preconditioner using HouseholderQR ***/

template <typename MatrixType, int Options>
class qr_preconditioner_impl<MatrixType, Options, HouseholderQRPreconditioner, PreconditionIfMoreRowsThanCols, true> {
 public:
  typedef typename MatrixType::Scalar Scalar;
  typedef JacobiSVD<MatrixType, Options> SVDType;

  enum {
    WorkspaceSize = internal::traits<SVDType>::MatrixUColsAtCompileTime,
    MaxWorkspaceSize = internal::traits<SVDType>::MatrixUMaxColsAtCompileTime
  };

  typedef Matrix<Scalar, 1, WorkspaceSize, RowMajor, 1, MaxWorkspaceSize> WorkspaceType;

  void allocate(const SVDType& svd) {
    if (svd.rows() != m_qr.rows() || svd.cols() != m_qr.cols()) {
      internal::destroy_at(&m_qr);
      internal::construct_at(&m_qr, svd.rows(), svd.cols());
    }
    if (svd.m_computeFullU)
      m_workspace.resize(svd.rows());
    else if (svd.m_computeThinU)
      m_workspace.resize(svd.cols());
  }
  template <typename Xpr>
  bool run(SVDType& svd, const Xpr& matrix) {
    if (matrix.rows() > matrix.cols()) {
      m_qr.compute(matrix);
      svd.m_workMatrix = m_qr.matrixQR().block(0, 0, matrix.cols(), matrix.cols()).template triangularView<Upper>();
      if (svd.m_computeFullU)
        m_qr.householderQ().evalTo(svd.m_matrixU, m_workspace);
      else if (svd.m_computeThinU) {
        svd.m_matrixU.setIdentity(matrix.rows(), matrix.cols());
        m_qr.householderQ().applyThisOnTheLeft(svd.m_matrixU, m_workspace);
      }
      if (svd.computeV()) svd.m_matrixV.setIdentity(matrix.cols(), matrix.cols());
      return true;
    }
    return false;
  }

 private:
  typedef HouseholderQR<MatrixType> QRType;
  QRType m_qr;
  WorkspaceType m_workspace;
};

template <typename MatrixType, int Options>
class qr_preconditioner_impl<MatrixType, Options, HouseholderQRPreconditioner, PreconditionIfMoreColsThanRows, true> {
 public:
  typedef typename MatrixType::Scalar Scalar;
  typedef JacobiSVD<MatrixType, Options> SVDType;

  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
    MatrixOptions = internal::traits<MatrixType>::Options,
    WorkspaceSize = internal::traits<SVDType>::MatrixVColsAtCompileTime,
    MaxWorkspaceSize = internal::traits<SVDType>::MatrixVMaxColsAtCompileTime
  };

  typedef Matrix<Scalar, WorkspaceSize, 1, ColMajor, MaxWorkspaceSize, 1> WorkspaceType;

  typedef typename internal::make_proper_matrix_type<Scalar, ColsAtCompileTime, RowsAtCompileTime, MatrixOptions,
                                                     MaxColsAtCompileTime, MaxRowsAtCompileTime>::type
      TransposeTypeWithSameStorageOrder;

  void allocate(const SVDType& svd) {
    if (svd.cols() != m_qr.rows() || svd.rows() != m_qr.cols()) {
      internal::destroy_at(&m_qr);
      internal::construct_at(&m_qr, svd.cols(), svd.rows());
    }
    if (svd.m_computeFullV)
      m_workspace.resize(svd.cols());
    else if (svd.m_computeThinV)
      m_workspace.resize(svd.rows());
  }

  template <typename Xpr>
  bool run(SVDType& svd, const Xpr& matrix) {
    if (matrix.cols() > matrix.rows()) {
      m_qr.compute(matrix.adjoint());

      svd.m_workMatrix =
          m_qr.matrixQR().block(0, 0, matrix.rows(), matrix.rows()).template triangularView<Upper>().adjoint();
      if (svd.m_computeFullV)
        m_qr.householderQ().evalTo(svd.m_matrixV, m_workspace);
      else if (svd.m_computeThinV) {
        svd.m_matrixV.setIdentity(matrix.cols(), matrix.rows());
        m_qr.householderQ().applyThisOnTheLeft(svd.m_matrixV, m_workspace);
      }
      if (svd.computeU()) svd.m_matrixU.setIdentity(matrix.rows(), matrix.rows());
      return true;
    } else
      return false;
  }

 private:
  typedef HouseholderQR<TransposeTypeWithSameStorageOrder> QRType;
  QRType m_qr;
  WorkspaceType m_workspace;
};

/*** 2x2 SVD implementation
 ***
 *** JacobiSVD consists in performing a series of 2x2 SVD subproblems
 ***/

template <typename MatrixType, int Options>
struct svd_precondition_2x2_block_to_be_real<MatrixType, Options, false> {
  typedef JacobiSVD<MatrixType, Options> SVD;
  typedef typename MatrixType::RealScalar RealScalar;
  static bool run(typename SVD::WorkMatrixType&, SVD&, Index, Index, RealScalar&) { return true; }
};

template <typename MatrixType, int Options>
struct svd_precondition_2x2_block_to_be_real<MatrixType, Options, true> {
  typedef JacobiSVD<MatrixType, Options> SVD;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  static bool run(typename SVD::WorkMatrixType& work_matrix, SVD& svd, Index p, Index q, RealScalar& maxDiagEntry) {
    using std::abs;
    using std::sqrt;
    Scalar z;
    JacobiRotation<Scalar> rot;
    RealScalar n = sqrt(numext::abs2(work_matrix.coeff(p, p)) + numext::abs2(work_matrix.coeff(q, p)));

    const RealScalar considerAsZero = (std::numeric_limits<RealScalar>::min)();
    const RealScalar precision = NumTraits<Scalar>::epsilon();

    if (numext::is_exactly_zero(n)) {
      // make sure first column is zero
      work_matrix.coeffRef(p, p) = work_matrix.coeffRef(q, p) = Scalar(0);

      if (abs(numext::imag(work_matrix.coeff(p, q))) > considerAsZero) {
        // work_matrix.coeff(p,q) can be zero if work_matrix.coeff(q,p) is not zero but small enough to underflow when
        // computing n
        z = abs(work_matrix.coeff(p, q)) / work_matrix.coeff(p, q);
        work_matrix.row(p) *= z;
        if (svd.computeU()) svd.m_matrixU.col(p) *= conj(z);
      }
      if (abs(numext::imag(work_matrix.coeff(q, q))) > considerAsZero) {
        z = abs(work_matrix.coeff(q, q)) / work_matrix.coeff(q, q);
        work_matrix.row(q) *= z;
        if (svd.computeU()) svd.m_matrixU.col(q) *= conj(z);
      }
      // otherwise the second row is already zero, so we have nothing to do.
    } else {
      rot.c() = conj(work_matrix.coeff(p, p)) / n;
      rot.s() = work_matrix.coeff(q, p) / n;
      work_matrix.applyOnTheLeft(p, q, rot);
      if (svd.computeU()) svd.m_matrixU.applyOnTheRight(p, q, rot.adjoint());
      if (abs(numext::imag(work_matrix.coeff(p, q))) > considerAsZero) {
        z = abs(work_matrix.coeff(p, q)) / work_matrix.coeff(p, q);
        work_matrix.col(q) *= z;
        if (svd.computeV()) svd.m_matrixV.col(q) *= z;
      }
      if (abs(numext::imag(work_matrix.coeff(q, q))) > considerAsZero) {
        z = abs(work_matrix.coeff(q, q)) / work_matrix.coeff(q, q);
        work_matrix.row(q) *= z;
        if (svd.computeU()) svd.m_matrixU.col(q) *= conj(z);
      }
    }

    // update largest diagonal entry
    maxDiagEntry = numext::maxi<RealScalar>(
        maxDiagEntry, numext::maxi<RealScalar>(abs(work_matrix.coeff(p, p)), abs(work_matrix.coeff(q, q))));
    // and check whether the 2x2 block is already diagonal
    RealScalar threshold = numext::maxi<RealScalar>(considerAsZero, precision * maxDiagEntry);
    return abs(work_matrix.coeff(p, q)) > threshold || abs(work_matrix.coeff(q, p)) > threshold;
  }
};

template <typename MatrixType_, int Options>
struct traits<JacobiSVD<MatrixType_, Options> > : svd_traits<MatrixType_, Options> {
  typedef MatrixType_ MatrixType;
};

}  // end namespace internal

/** \ingroup SVD_Module
 *
 *
 * \class JacobiSVD
 *
 * \brief Two-sided Jacobi SVD decomposition of a rectangular matrix
 *
 * \tparam MatrixType_ the type of the matrix of which we are computing the SVD decomposition
 * \tparam Options this optional parameter allows one to specify the type of QR decomposition that will be used
 * internally for the R-SVD step for non-square matrices. Additionally, it allows one to specify whether to compute thin
 * or full unitaries \a U and \a V. See discussion of possible values below.
 *
 * SVD decomposition consists in decomposing any n-by-p matrix \a A as a product
 *   \f[ A = U S V^* \f]
 * where \a U is a n-by-n unitary, \a V is a p-by-p unitary, and \a S is a n-by-p real positive matrix which is zero
 * outside of its main diagonal; the diagonal entries of S are known as the \em singular \em values of \a A and the
 * columns of \a U and \a V are known as the left and right \em singular \em vectors of \a A respectively.
 *
 * Singular values are always sorted in decreasing order.
 *
 * This JacobiSVD decomposition computes only the singular values by default. If you want \a U or \a V, you need to ask
 * for them explicitly.
 *
 * You can ask for only \em thin \a U or \a V to be computed, meaning the following. In case of a rectangular n-by-p
 * matrix, letting \a m be the smaller value among \a n and \a p, there are only \a m singular vectors; the remaining
 * columns of \a U and \a V do not correspond to actual singular vectors. Asking for \em thin \a U or \a V means asking
 * for only their \a m first columns to be formed. So \a U is then a n-by-m matrix, and \a V is then a p-by-m matrix.
 * Notice that thin \a U and \a V are all you need for (least squares) solving.
 *
 * Here's an example demonstrating basic usage:
 * \include JacobiSVD_basic.cpp
 * Output: \verbinclude JacobiSVD_basic.out
 *
 * This JacobiSVD class is a two-sided Jacobi R-SVD decomposition, ensuring optimal reliability and accuracy. The
 * downside is that it's slower than bidiagonalizing SVD algorithms for large square matrices; however its complexity is
 * still \f$ O(n^2p) \f$ where \a n is the smaller dimension and \a p is the greater dimension, meaning that it is still
 * of the same order of complexity as the faster bidiagonalizing R-SVD algorithms. In particular, like any R-SVD, it
 * takes advantage of non-squareness in that its complexity is only linear in the greater dimension.
 *
 * If the input matrix has inf or nan coefficients, the result of the computation is undefined, but the computation is
 * guaranteed to terminate in finite (and reasonable) time.
 *
 * The possible QR preconditioners that can be set with Options template parameter are:
 * \li ColPivHouseholderQRPreconditioner is the default. In practice it's very safe. It uses column-pivoting QR.
 * \li FullPivHouseholderQRPreconditioner, is the safest and slowest. It uses full-pivoting QR.
 *     Contrary to other QRs, it doesn't allow computing thin unitaries.
 * \li HouseholderQRPreconditioner is the fastest, and less safe and accurate than the pivoting variants. It uses
 * non-pivoting QR. This is very similar in safety and accuracy to the bidiagonalization process used by bidiagonalizing
 * SVD algorithms (since bidiagonalization is inherently non-pivoting). However the resulting SVD is still more reliable
 * than bidiagonalizing SVDs because the Jacobi-based iterarive process is more reliable than the optimized bidiagonal
 * SVD iterations. \li NoQRPreconditioner allows not to use a QR preconditioner at all. This is useful if you know that
 * you will only be computing JacobiSVD decompositions of square matrices. Non-square matrices require a QR
 * preconditioner. Using this option will result in faster compilation and smaller executable code. It won't
 * significantly speed up computation, since JacobiSVD is always checking if QR preconditioning is needed before
 * applying it anyway.
 *
 * One may also use the Options template parameter to specify how the unitaries should be computed. The options are
 * #ComputeThinU, #ComputeThinV, #ComputeFullU, #ComputeFullV. It is not possible to request both the thin and full
 * versions of a unitary. By default, unitaries will not be computed.
 *
 * You can set the QRPreconditioner and unitary options together: JacobiSVD<MatrixType,
 * ColPivHouseholderQRPreconditioner | ComputeThinU | ComputeFullV>
 *
 * \sa MatrixBase::jacobiSvd()
 */
template <typename MatrixType_, int Options_>
class JacobiSVD : public SVDBase<JacobiSVD<MatrixType_, Options_> > {
  typedef SVDBase<JacobiSVD> Base;

 public:
  typedef MatrixType_ MatrixType;
  typedef typename Base::Scalar Scalar;
  typedef typename Base::RealScalar RealScalar;
  enum : int {
    Options = Options_,
    QRPreconditioner = internal::get_qr_preconditioner(Options),
    RowsAtCompileTime = Base::RowsAtCompileTime,
    ColsAtCompileTime = Base::ColsAtCompileTime,
    DiagSizeAtCompileTime = Base::DiagSizeAtCompileTime,
    MaxRowsAtCompileTime = Base::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = Base::MaxColsAtCompileTime,
    MaxDiagSizeAtCompileTime = Base::MaxDiagSizeAtCompileTime,
    MatrixOptions = Base::MatrixOptions
  };

  typedef typename Base::MatrixUType MatrixUType;
  typedef typename Base::MatrixVType MatrixVType;
  typedef typename Base::SingularValuesType SingularValuesType;
  typedef Matrix<Scalar, DiagSizeAtCompileTime, DiagSizeAtCompileTime, MatrixOptions, MaxDiagSizeAtCompileTime,
                 MaxDiagSizeAtCompileTime>
      WorkMatrixType;

  /** \brief Default Constructor.
   *
   * The default constructor is useful in cases in which the user intends to
   * perform decompositions via JacobiSVD::compute(const MatrixType&).
   */
  JacobiSVD() {}

  /** \brief Default Constructor with memory preallocation
   *
   * Like the default constructor but with preallocation of the internal data
   * according to the specified problem size and \a Options template parameter.
   *
   * \sa JacobiSVD()
   */
  JacobiSVD(Index rows, Index cols) { allocate(rows, cols, internal::get_computation_options(Options)); }

  /** \brief Default Constructor with memory preallocation
   *
   * Like the default constructor but with preallocation of the internal data
   * according to the specified problem size.
   *
   * One \b cannot request unitaries using both the \a Options template parameter
   * and the constructor. If possible, prefer using the \a Options template parameter.
   *
   * \param computationOptions specify whether to compute Thin/Full unitaries U/V
   * \sa JacobiSVD()
   *
   * \deprecated Will be removed in the next major Eigen version. Options should
   * be specified in the \a Options template parameter.
   */
  EIGEN_DEPRECATED JacobiSVD(Index rows, Index cols, unsigned int computationOptions) {
    internal::check_svd_options_assertions<MatrixType, Options>(computationOptions, rows, cols);
    allocate(rows, cols, computationOptions);
  }

  /** \brief Constructor performing the decomposition of given matrix, using the custom options specified
   *         with the \a Options template paramter.
   *
   * \param matrix the matrix to decompose
   */
  explicit JacobiSVD(const MatrixType& matrix) { compute_impl(matrix, internal::get_computation_options(Options)); }

  /** \brief Constructor performing the decomposition of given matrix using specified options
   *         for computing unitaries.
   *
   *  One \b cannot request unitiaries using both the \a Options template parameter
   *  and the constructor. If possible, prefer using the \a Options template parameter.
   *
   * \param matrix the matrix to decompose
   * \param computationOptions specify whether to compute Thin/Full unitaries U/V
   *
   * \deprecated Will be removed in the next major Eigen version. Options should
   * be specified in the \a Options template parameter.
   */
  // EIGEN_DEPRECATED // TODO(cantonios): re-enable after fixing a few 3p libraries that error on deprecation warnings.
  JacobiSVD(const MatrixType& matrix, unsigned int computationOptions) {
    internal::check_svd_options_assertions<MatrixType, Options>(computationOptions, matrix.rows(), matrix.cols());
    compute_impl(matrix, computationOptions);
  }

  /** \brief Method performing the decomposition of given matrix. Computes Thin/Full unitaries U/V if specified
   *         using the \a Options template parameter or the class constructor.
   *
   * \param matrix the matrix to decompose
   */
  JacobiSVD& compute(const MatrixType& matrix) { return compute_impl(matrix, m_computationOptions); }

  /** \brief Method performing the decomposition of given matrix, as specified by
   *         the `computationOptions` parameter.
   *
   * \param matrix the matrix to decompose
   * \param computationOptions specify whether to compute Thin/Full unitaries U/V
   *
   * \deprecated Will be removed in the next major Eigen version. Options should
   * be specified in the \a Options template parameter.
   */
  EIGEN_DEPRECATED JacobiSVD& compute(const MatrixType& matrix, unsigned int computationOptions) {
    internal::check_svd_options_assertions<MatrixType, Options>(m_computationOptions, matrix.rows(), matrix.cols());
    return compute_impl(matrix, computationOptions);
  }

  using Base::cols;
  using Base::computeU;
  using Base::computeV;
  using Base::diagSize;
  using Base::rank;
  using Base::rows;

 private:
  void allocate(Index rows_, Index cols_, unsigned int computationOptions) {
    if (Base::allocate(rows_, cols_, computationOptions)) return;
    eigen_assert(!(ShouldComputeThinU && int(QRPreconditioner) == int(FullPivHouseholderQRPreconditioner)) &&
                 !(ShouldComputeThinU && int(QRPreconditioner) == int(FullPivHouseholderQRPreconditioner)) &&
                 "JacobiSVD: can't compute thin U or thin V with the FullPivHouseholderQR preconditioner. "
                 "Use the ColPivHouseholderQR preconditioner instead.");

    m_workMatrix.resize(diagSize(), diagSize());
    if (cols() > rows()) m_qr_precond_morecols.allocate(*this);
    if (rows() > cols()) m_qr_precond_morerows.allocate(*this);
  }

  JacobiSVD& compute_impl(const MatrixType& matrix, unsigned int computationOptions);

 protected:
  using Base::m_computationOptions;
  using Base::m_computeFullU;
  using Base::m_computeFullV;
  using Base::m_computeThinU;
  using Base::m_computeThinV;
  using Base::m_info;
  using Base::m_isAllocated;
  using Base::m_isInitialized;
  using Base::m_matrixU;
  using Base::m_matrixV;
  using Base::m_nonzeroSingularValues;
  using Base::m_prescribedThreshold;
  using Base::m_singularValues;
  using Base::m_usePrescribedThreshold;
  using Base::ShouldComputeThinU;
  using Base::ShouldComputeThinV;

  EIGEN_STATIC_ASSERT(!(ShouldComputeThinU && int(QRPreconditioner) == int(FullPivHouseholderQRPreconditioner)) &&
                          !(ShouldComputeThinU && int(QRPreconditioner) == int(FullPivHouseholderQRPreconditioner)),
                      "JacobiSVD: can't compute thin U or thin V with the FullPivHouseholderQR preconditioner. "
                      "Use the ColPivHouseholderQR preconditioner instead.")

  template <typename MatrixType__, int Options__, bool IsComplex_>
  friend struct internal::svd_precondition_2x2_block_to_be_real;
  template <typename MatrixType__, int Options__, int QRPreconditioner_, int Case_, bool DoAnything_>
  friend struct internal::qr_preconditioner_impl;

  internal::qr_preconditioner_impl<MatrixType, Options, QRPreconditioner, internal::PreconditionIfMoreColsThanRows>
      m_qr_precond_morecols;
  internal::qr_preconditioner_impl<MatrixType, Options, QRPreconditioner, internal::PreconditionIfMoreRowsThanCols>
      m_qr_precond_morerows;
  WorkMatrixType m_workMatrix;
};

template <typename MatrixType, int Options>
JacobiSVD<MatrixType, Options>& JacobiSVD<MatrixType, Options>::compute_impl(const MatrixType& matrix,
                                                                             unsigned int computationOptions) {
  using std::abs;

  allocate(matrix.rows(), matrix.cols(), computationOptions);

  // currently we stop when we reach precision 2*epsilon as the last bit of precision can require an unreasonable number
  // of iterations, only worsening the precision of U and V as we accumulate more rotations
  const RealScalar precision = RealScalar(2) * NumTraits<Scalar>::epsilon();

  // limit for denormal numbers to be considered zero in order to avoid infinite loops (see bug 286)
  const RealScalar considerAsZero = (std::numeric_limits<RealScalar>::min)();

  // Scaling factor to reduce over/under-flows
  RealScalar scale = matrix.cwiseAbs().template maxCoeff<PropagateNaN>();
  if (!(numext::isfinite)(scale)) {
    m_isInitialized = true;
    m_info = InvalidInput;
    m_nonzeroSingularValues = 0;
    return *this;
  }
  if (numext::is_exactly_zero(scale)) scale = RealScalar(1);

  /*** step 1. The R-SVD step: we use a QR decomposition to reduce to the case of a square matrix */

  if (rows() != cols()) {
    m_qr_precond_morecols.run(*this, matrix / scale);
    m_qr_precond_morerows.run(*this, matrix / scale);
  } else {
    m_workMatrix =
        matrix.template topLeftCorner<DiagSizeAtCompileTime, DiagSizeAtCompileTime>(diagSize(), diagSize()) / scale;
    if (m_computeFullU) m_matrixU.setIdentity(rows(), rows());
    if (m_computeThinU) m_matrixU.setIdentity(rows(), diagSize());
    if (m_computeFullV) m_matrixV.setIdentity(cols(), cols());
    if (m_computeThinV) m_matrixV.setIdentity(cols(), diagSize());
  }

  /*** step 2. The main Jacobi SVD iteration. ***/
  RealScalar maxDiagEntry = m_workMatrix.cwiseAbs().diagonal().maxCoeff();

  bool finished = false;
  while (!finished) {
    finished = true;

    // do a sweep: for all index pairs (p,q), perform SVD of the corresponding 2x2 sub-matrix

    for (Index p = 1; p < diagSize(); ++p) {
      for (Index q = 0; q < p; ++q) {
        // if this 2x2 sub-matrix is not diagonal already...
        // notice that this comparison will evaluate to false if any NaN is involved, ensuring that NaN's don't
        // keep us iterating forever. Similarly, small denormal numbers are considered zero.
        RealScalar threshold = numext::maxi<RealScalar>(considerAsZero, precision * maxDiagEntry);
        if (abs(m_workMatrix.coeff(p, q)) > threshold || abs(m_workMatrix.coeff(q, p)) > threshold) {
          finished = false;
          // perform SVD decomposition of 2x2 sub-matrix corresponding to indices p,q to make it diagonal
          // the complex to real operation returns true if the updated 2x2 block is not already diagonal
          if (internal::svd_precondition_2x2_block_to_be_real<MatrixType, Options>::run(m_workMatrix, *this, p, q,
                                                                                        maxDiagEntry)) {
            JacobiRotation<RealScalar> j_left, j_right;
            internal::real_2x2_jacobi_svd(m_workMatrix, p, q, &j_left, &j_right);

            // accumulate resulting Jacobi rotations
            m_workMatrix.applyOnTheLeft(p, q, j_left);
            if (computeU()) m_matrixU.applyOnTheRight(p, q, j_left.transpose());

            m_workMatrix.applyOnTheRight(p, q, j_right);
            if (computeV()) m_matrixV.applyOnTheRight(p, q, j_right);

            // keep track of the largest diagonal coefficient
            maxDiagEntry = numext::maxi<RealScalar>(
                maxDiagEntry, numext::maxi<RealScalar>(abs(m_workMatrix.coeff(p, p)), abs(m_workMatrix.coeff(q, q))));
          }
        }
      }
    }
  }

  /*** step 3. The work matrix is now diagonal, so ensure it's positive so its diagonal entries are the singular values
   * ***/

  for (Index i = 0; i < diagSize(); ++i) {
    // For a complex matrix, some diagonal coefficients might note have been
    // treated by svd_precondition_2x2_block_to_be_real, and the imaginary part
    // of some diagonal entry might not be null.
    if (NumTraits<Scalar>::IsComplex && abs(numext::imag(m_workMatrix.coeff(i, i))) > considerAsZero) {
      RealScalar a = abs(m_workMatrix.coeff(i, i));
      m_singularValues.coeffRef(i) = abs(a);
      if (computeU()) m_matrixU.col(i) *= m_workMatrix.coeff(i, i) / a;
    } else {
      // m_workMatrix.coeff(i,i) is already real, no difficulty:
      RealScalar a = numext::real(m_workMatrix.coeff(i, i));
      m_singularValues.coeffRef(i) = abs(a);
      if (computeU() && (a < RealScalar(0))) m_matrixU.col(i) = -m_matrixU.col(i);
    }
  }

  m_singularValues *= scale;

  /*** step 4. Sort singular values in descending order and compute the number of nonzero singular values ***/

  m_nonzeroSingularValues = diagSize();
  for (Index i = 0; i < diagSize(); i++) {
    Index pos;
    RealScalar maxRemainingSingularValue = m_singularValues.tail(diagSize() - i).maxCoeff(&pos);
    if (numext::is_exactly_zero(maxRemainingSingularValue)) {
      m_nonzeroSingularValues = i;
      break;
    }
    if (pos) {
      pos += i;
      std::swap(m_singularValues.coeffRef(i), m_singularValues.coeffRef(pos));
      if (computeU()) m_matrixU.col(pos).swap(m_matrixU.col(i));
      if (computeV()) m_matrixV.col(pos).swap(m_matrixV.col(i));
    }
  }

  m_isInitialized = true;
  return *this;
}

/** \svd_module
 *
 * \return the singular value decomposition of \c *this computed by two-sided
 * Jacobi transformations.
 *
 * \sa class JacobiSVD
 */
template <typename Derived>
template <int Options>
JacobiSVD<typename MatrixBase<Derived>::PlainObject, Options> MatrixBase<Derived>::jacobiSvd() const {
  return JacobiSVD<PlainObject, Options>(*this);
}

template <typename Derived>
template <int Options>
JacobiSVD<typename MatrixBase<Derived>::PlainObject, Options> MatrixBase<Derived>::jacobiSvd(
    unsigned int computationOptions) const {
  return JacobiSVD<PlainObject, Options>(*this, computationOptions);
}

}  // end namespace Eigen

#endif  // EIGEN_JACOBISVD_H
