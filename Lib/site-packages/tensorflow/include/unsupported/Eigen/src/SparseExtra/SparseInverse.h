// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2022 Julian Kent <jkflying@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSEINVERSE_H
#define EIGEN_SPARSEINVERSE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "../../../../Eigen/Sparse"
#include "../../../../Eigen/SparseLU"

namespace Eigen {

/**
 * @brief Kahan algorithm based accumulator
 *
 * The Kahan sum algorithm guarantees to bound the error from floating point
 * accumulation to a fixed value, regardless of the number of accumulations
 * performed. Naive accumulation accumulates errors O(N), and pairwise O(logN).
 * However pairwise also requires O(logN) memory while Kahan summation requires
 * O(1) memory, but 4x the operations / latency.
 *
 * NB! Do not enable associative math optimizations, they may cause the Kahan
 * summation to be optimized out leaving you with naive summation again.
 *
 */
template <typename Scalar>
class KahanSum {
  // Straighforward Kahan summation for accurate accumulation of a sum of numbers
  Scalar _sum{};
  Scalar _correction{};

 public:
  Scalar value() { return _sum; }

  void operator+=(Scalar increment) {
    const Scalar correctedIncrement = increment + _correction;
    const Scalar previousSum = _sum;
    _sum += correctedIncrement;
    _correction = correctedIncrement - (_sum - previousSum);
  }
};
template <typename Scalar, Index Width = 16>
class FABSum {
  // https://epubs.siam.org/doi/pdf/10.1137/19M1257780
  // Fast and Accurate Blocked Summation
  // Uses naive summation for the fast sum, and Kahan summation for the accurate sum
  // Theoretically SIMD sum could be changed to a tree sum which would improve accuracy
  // over naive summation
  KahanSum<Scalar> _totalSum;
  Matrix<Scalar, Width, 1> _block;
  Index _blockUsed{};

 public:
  Scalar value() { return _block.topRows(_blockUsed).sum() + _totalSum.value(); }

  void operator+=(Scalar increment) {
    _block(_blockUsed++, 0) = increment;
    if (_blockUsed == Width) {
      _totalSum += _block.sum();
      _blockUsed = 0;
    }
  }
};

/**
 * @brief computes an accurate dot product on two sparse vectors
 *
 * Uses an accurate summation algorithm for the accumulator in order to
 * compute an accurate dot product for two sparse vectors.
 *
 */
template <typename Derived, typename OtherDerived>
typename Derived::Scalar accurateDot(const SparseMatrixBase<Derived>& A, const SparseMatrixBase<OtherDerived>& other) {
  typedef typename Derived::Scalar Scalar;
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(OtherDerived)
  EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(Derived, OtherDerived)
  static_assert(internal::is_same<Scalar, typename OtherDerived::Scalar>::value, "mismatched types");

  internal::evaluator<Derived> thisEval(A.derived());
  typename Derived::ReverseInnerIterator i(thisEval, 0);

  internal::evaluator<OtherDerived> otherEval(other.derived());
  typename OtherDerived::ReverseInnerIterator j(otherEval, 0);

  FABSum<Scalar> res;
  while (i && j) {
    if (i.index() == j.index()) {
      res += numext::conj(i.value()) * j.value();
      --i;
      --j;
    } else if (i.index() > j.index())
      --i;
    else
      --j;
  }
  return res.value();
}

/**
 * @brief calculate sparse subset of inverse of sparse matrix
 *
 * This class returns a sparse subset of the inverse of the input matrix.
 * The nonzeros correspond to the nonzeros of the input, plus any additional
 * elements required due to fill-in of the internal LU factorization. This is
 * is minimized via a applying a fill-reducing permutation as part of the LU
 * factorization.
 *
 * If there are specific entries of the input matrix which you need inverse
 * values for, which are zero for the input, you need to insert entries into
 * the input sparse matrix for them to be calculated.
 *
 * Due to the sensitive nature of matrix inversion, particularly on large
 * matrices which are made possible via sparsity, high accuracy dot products
 * based on Kahan summation are used to reduce numerical error. If you still
 * encounter numerical errors you may with to equilibrate your matrix before
 * calculating the inverse, as well as making sure it is actually full rank.
 */
template <typename Scalar>
class SparseInverse {
 public:
  typedef SparseMatrix<Scalar, ColMajor> MatrixType;
  typedef SparseMatrix<Scalar, RowMajor> RowMatrixType;

  SparseInverse() {}

  /**
   * @brief This Constructor is for if you already have a factored SparseLU and would like to use it to calculate a
   * sparse inverse.
   *
   * Just call this constructor with your already factored SparseLU class and you can directly call the .inverse()
   * method to get the result.
   */
  SparseInverse(const SparseLU<MatrixType>& slu) { _result = computeInverse(slu); }

  /**
   * @brief Calculate the sparse inverse from a given sparse input
   */
  SparseInverse& compute(const SparseMatrix<Scalar>& A) {
    SparseLU<MatrixType> slu;
    slu.compute(A);
    _result = computeInverse(slu);
    return *this;
  }

  /**
   * @brief return the already-calculated sparse inverse, or a 0x0 matrix if it could not be computed
   */
  const MatrixType& inverse() const { return _result; }

  /**
   * @brief Internal function to calculate the sparse inverse in a functional way
   * @return A sparse inverse representation, or, if the decomposition didn't complete, a 0x0 matrix.
   */
  static MatrixType computeInverse(const SparseLU<MatrixType>& slu) {
    if (slu.info() != Success) {
      return MatrixType(0, 0);
    }

    // Extract from SparseLU and decompose into L, inverse D and U terms
    Matrix<Scalar, Dynamic, 1> invD;
    RowMatrixType Upper;
    {
      RowMatrixType DU = slu.matrixU().toSparse();
      invD = DU.diagonal().cwiseInverse();
      Upper = (invD.asDiagonal() * DU).template triangularView<StrictlyUpper>();
    }
    MatrixType Lower = slu.matrixL().toSparse().template triangularView<StrictlyLower>();

    // Compute the inverse and reapply the permutation matrix from the LU decomposition
    return slu.colsPermutation().transpose() * computeInverse(Upper, invD, Lower) * slu.rowsPermutation();
  }

  /**
   * @brief Internal function to calculate the inverse from strictly upper, diagonal and strictly lower components
   */
  static MatrixType computeInverse(const RowMatrixType& Upper, const Matrix<Scalar, Dynamic, 1>& inverseDiagonal,
                                   const MatrixType& Lower) {
    // Calculate the 'minimal set', which is the nonzeros of (L+U).transpose()
    // It could be zeroed, but we will overwrite all non-zeros anyways.
    MatrixType colInv = Lower.transpose().template triangularView<UnitUpper>();
    colInv += Upper.transpose();

    // We also need rowmajor representation in order to do efficient row-wise dot products
    RowMatrixType rowInv = Upper.transpose().template triangularView<UnitLower>();
    rowInv += Lower.transpose();

    // Use the Takahashi algorithm to build the supporting elements of the inverse
    // upwards and to the left, from the bottom right element, 1 col/row at a time
    for (Index recurseLevel = Upper.cols() - 1; recurseLevel >= 0; recurseLevel--) {
      const auto& col = Lower.col(recurseLevel);
      const auto& row = Upper.row(recurseLevel);

      // Calculate the inverse values for the nonzeros in this column
      typename MatrixType::ReverseInnerIterator colIter(colInv, recurseLevel);
      for (; recurseLevel < colIter.index(); --colIter) {
        const Scalar element = -accurateDot(col, rowInv.row(colIter.index()));
        colIter.valueRef() = element;
        rowInv.coeffRef(colIter.index(), recurseLevel) = element;
      }

      // Calculate the inverse values for the nonzeros in this row
      typename RowMatrixType::ReverseInnerIterator rowIter(rowInv, recurseLevel);
      for (; recurseLevel < rowIter.index(); --rowIter) {
        const Scalar element = -accurateDot(row, colInv.col(rowIter.index()));
        rowIter.valueRef() = element;
        colInv.coeffRef(recurseLevel, rowIter.index()) = element;
      }

      // And finally the diagonal, which corresponds to both row and col iterator now
      const Scalar diag = inverseDiagonal(recurseLevel) - accurateDot(row, colInv.col(recurseLevel));
      rowIter.valueRef() = diag;
      colIter.valueRef() = diag;
    }

    return colInv;
  }

 private:
  MatrixType _result;
};

}  // namespace Eigen
#endif
