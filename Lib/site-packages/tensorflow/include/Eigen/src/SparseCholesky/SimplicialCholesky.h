// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2012 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SIMPLICIAL_CHOLESKY_H
#define EIGEN_SIMPLICIAL_CHOLESKY_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

enum SimplicialCholeskyMode { SimplicialCholeskyLLT, SimplicialCholeskyLDLT };

namespace internal {
template <typename CholMatrixType, typename InputMatrixType>
struct simplicial_cholesky_grab_input {
  typedef CholMatrixType const* ConstCholMatrixPtr;
  static void run(const InputMatrixType& input, ConstCholMatrixPtr& pmat, CholMatrixType& tmp) {
    tmp = input;
    pmat = &tmp;
  }
};

template <typename MatrixType>
struct simplicial_cholesky_grab_input<MatrixType, MatrixType> {
  typedef MatrixType const* ConstMatrixPtr;
  static void run(const MatrixType& input, ConstMatrixPtr& pmat, MatrixType& /*tmp*/) { pmat = &input; }
};
}  // end namespace internal

/** \ingroup SparseCholesky_Module
 * \brief A base class for direct sparse Cholesky factorizations
 *
 * This is a base class for LL^T and LDL^T Cholesky factorizations of sparse matrices that are
 * selfadjoint and positive definite. These factorizations allow for solving A.X = B where
 * X and B can be either dense or sparse.
 *
 * In order to reduce the fill-in, a symmetric permutation P is applied prior to the factorization
 * such that the factorized matrix is P A P^-1.
 *
 * \tparam Derived the type of the derived class, that is the actual factorization type.
 *
 */
template <typename Derived>
class SimplicialCholeskyBase : public SparseSolverBase<Derived> {
  typedef SparseSolverBase<Derived> Base;
  using Base::m_isInitialized;

 public:
  typedef typename internal::traits<Derived>::MatrixType MatrixType;
  typedef typename internal::traits<Derived>::OrderingType OrderingType;
  enum { UpLo = internal::traits<Derived>::UpLo };
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef typename internal::traits<Derived>::DiagonalScalar DiagonalScalar;
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef SparseMatrix<Scalar, ColMajor, StorageIndex> CholMatrixType;
  typedef CholMatrixType const* ConstCholMatrixPtr;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;
  typedef Matrix<StorageIndex, Dynamic, 1> VectorI;

  enum { ColsAtCompileTime = MatrixType::ColsAtCompileTime, MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime };

 public:
  using Base::derived;

  /** Default constructor */
  SimplicialCholeskyBase()
      : m_info(Success), m_factorizationIsOk(false), m_analysisIsOk(false), m_shiftOffset(0), m_shiftScale(1) {}

  explicit SimplicialCholeskyBase(const MatrixType& matrix)
      : m_info(Success), m_factorizationIsOk(false), m_analysisIsOk(false), m_shiftOffset(0), m_shiftScale(1) {
    derived().compute(matrix);
  }

  ~SimplicialCholeskyBase() {}

  Derived& derived() { return *static_cast<Derived*>(this); }
  const Derived& derived() const { return *static_cast<const Derived*>(this); }

  inline Index cols() const { return m_matrix.cols(); }
  inline Index rows() const { return m_matrix.rows(); }

  /** \brief Reports whether previous computation was successful.
   *
   * \returns \c Success if computation was successful,
   *          \c NumericalIssue if the matrix.appears to be negative.
   */
  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "Decomposition is not initialized.");
    return m_info;
  }

  /** \returns the permutation P
   * \sa permutationPinv() */
  const PermutationMatrix<Dynamic, Dynamic, StorageIndex>& permutationP() const { return m_P; }

  /** \returns the inverse P^-1 of the permutation P
   * \sa permutationP() */
  const PermutationMatrix<Dynamic, Dynamic, StorageIndex>& permutationPinv() const { return m_Pinv; }

  /** Sets the shift parameters that will be used to adjust the diagonal coefficients during the numerical
   * factorization.
   *
   * During the numerical factorization, the diagonal coefficients are transformed by the following linear model:\n
   * \c d_ii = \a offset + \a scale * \c d_ii
   *
   * The default is the identity transformation with \a offset=0, and \a scale=1.
   *
   * \returns a reference to \c *this.
   */
  Derived& setShift(const DiagonalScalar& offset, const DiagonalScalar& scale = 1) {
    m_shiftOffset = offset;
    m_shiftScale = scale;
    return derived();
  }

#ifndef EIGEN_PARSED_BY_DOXYGEN
  /** \internal */
  template <typename Stream>
  void dumpMemory(Stream& s) {
    int total = 0;
    s << "  L:        "
      << ((total += (m_matrix.cols() + 1) * sizeof(int) + m_matrix.nonZeros() * (sizeof(int) + sizeof(Scalar))) >> 20)
      << "Mb"
      << "\n";
    s << "  diag:     " << ((total += m_diag.size() * sizeof(Scalar)) >> 20) << "Mb"
      << "\n";
    s << "  tree:     " << ((total += m_parent.size() * sizeof(int)) >> 20) << "Mb"
      << "\n";
    s << "  nonzeros: " << ((total += m_nonZerosPerCol.size() * sizeof(int)) >> 20) << "Mb"
      << "\n";
    s << "  perm:     " << ((total += m_P.size() * sizeof(int)) >> 20) << "Mb"
      << "\n";
    s << "  perm^-1:  " << ((total += m_Pinv.size() * sizeof(int)) >> 20) << "Mb"
      << "\n";
    s << "  TOTAL:    " << (total >> 20) << "Mb"
      << "\n";
  }

  /** \internal */
  template <typename Rhs, typename Dest>
  void _solve_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& dest) const {
    eigen_assert(m_factorizationIsOk &&
                 "The decomposition is not in a valid state for solving, you must first call either compute() or "
                 "symbolic()/numeric()");
    eigen_assert(m_matrix.rows() == b.rows());

    if (m_info != Success) return;

    if (m_P.size() > 0)
      dest = m_P * b;
    else
      dest = b;

    if (m_matrix.nonZeros() > 0)  // otherwise L==I
      derived().matrixL().solveInPlace(dest);

    if (m_diag.size() > 0) dest = m_diag.asDiagonal().inverse() * dest;

    if (m_matrix.nonZeros() > 0)  // otherwise U==I
      derived().matrixU().solveInPlace(dest);

    if (m_P.size() > 0) dest = m_Pinv * dest;
  }

  template <typename Rhs, typename Dest>
  void _solve_impl(const SparseMatrixBase<Rhs>& b, SparseMatrixBase<Dest>& dest) const {
    internal::solve_sparse_through_dense_panels(derived(), b, dest);
  }

#endif  // EIGEN_PARSED_BY_DOXYGEN

 protected:
  /** Computes the sparse Cholesky decomposition of \a matrix */
  template <bool DoLDLT, bool NonHermitian>
  void compute(const MatrixType& matrix) {
    eigen_assert(matrix.rows() == matrix.cols());
    Index size = matrix.cols();
    CholMatrixType tmp(size, size);
    ConstCholMatrixPtr pmat;
    ordering<NonHermitian>(matrix, pmat, tmp);
    analyzePattern_preordered(*pmat, DoLDLT);
    factorize_preordered<DoLDLT, NonHermitian>(*pmat);
  }

  template <bool DoLDLT, bool NonHermitian>
  void factorize(const MatrixType& a) {
    eigen_assert(a.rows() == a.cols());
    Index size = a.cols();
    CholMatrixType tmp(size, size);
    ConstCholMatrixPtr pmat;

    if (m_P.size() == 0 && (int(UpLo) & int(Upper)) == Upper) {
      // If there is no ordering, try to directly use the input matrix without any copy
      internal::simplicial_cholesky_grab_input<CholMatrixType, MatrixType>::run(a, pmat, tmp);
    } else {
      internal::permute_symm_to_symm<UpLo, Upper, NonHermitian>(a, tmp, m_P.indices().data());
      pmat = &tmp;
    }

    factorize_preordered<DoLDLT, NonHermitian>(*pmat);
  }

  template <bool DoLDLT, bool NonHermitian>
  void factorize_preordered(const CholMatrixType& a);

  template <bool DoLDLT, bool NonHermitian>
  void analyzePattern(const MatrixType& a) {
    eigen_assert(a.rows() == a.cols());
    Index size = a.cols();
    CholMatrixType tmp(size, size);
    ConstCholMatrixPtr pmat;
    ordering<NonHermitian>(a, pmat, tmp);
    analyzePattern_preordered(*pmat, DoLDLT);
  }
  void analyzePattern_preordered(const CholMatrixType& a, bool doLDLT);

  template <bool NonHermitian>
  void ordering(const MatrixType& a, ConstCholMatrixPtr& pmat, CholMatrixType& ap);

  inline DiagonalScalar getDiag(Scalar x) { return internal::traits<Derived>::getDiag(x); }
  inline Scalar getSymm(Scalar x) { return internal::traits<Derived>::getSymm(x); }

  /** keeps off-diagonal entries; drops diagonal entries */
  struct keep_diag {
    inline bool operator()(const Index& row, const Index& col, const Scalar&) const { return row != col; }
  };

  mutable ComputationInfo m_info;
  bool m_factorizationIsOk;
  bool m_analysisIsOk;

  CholMatrixType m_matrix;
  VectorType m_diag;  // the diagonal coefficients (LDLT mode)
  VectorI m_parent;   // elimination tree
  VectorI m_nonZerosPerCol;
  PermutationMatrix<Dynamic, Dynamic, StorageIndex> m_P;     // the permutation
  PermutationMatrix<Dynamic, Dynamic, StorageIndex> m_Pinv;  // the inverse permutation

  DiagonalScalar m_shiftOffset;
  DiagonalScalar m_shiftScale;
};

template <typename MatrixType_, int UpLo_ = Lower,
          typename Ordering_ = AMDOrdering<typename MatrixType_::StorageIndex> >
class SimplicialLLT;
template <typename MatrixType_, int UpLo_ = Lower,
          typename Ordering_ = AMDOrdering<typename MatrixType_::StorageIndex> >
class SimplicialLDLT;
template <typename MatrixType_, int UpLo_ = Lower,
          typename Ordering_ = AMDOrdering<typename MatrixType_::StorageIndex> >
class SimplicialNonHermitianLLT;
template <typename MatrixType_, int UpLo_ = Lower,
          typename Ordering_ = AMDOrdering<typename MatrixType_::StorageIndex> >
class SimplicialNonHermitianLDLT;
template <typename MatrixType_, int UpLo_ = Lower,
          typename Ordering_ = AMDOrdering<typename MatrixType_::StorageIndex> >
class SimplicialCholesky;

namespace internal {

template <typename MatrixType_, int UpLo_, typename Ordering_>
struct traits<SimplicialLLT<MatrixType_, UpLo_, Ordering_> > {
  typedef MatrixType_ MatrixType;
  typedef Ordering_ OrderingType;
  enum { UpLo = UpLo_ };
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar DiagonalScalar;
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef SparseMatrix<Scalar, ColMajor, StorageIndex> CholMatrixType;
  typedef TriangularView<const CholMatrixType, Eigen::Lower> MatrixL;
  typedef TriangularView<const typename CholMatrixType::AdjointReturnType, Eigen::Upper> MatrixU;
  static inline MatrixL getL(const CholMatrixType& m) { return MatrixL(m); }
  static inline MatrixU getU(const CholMatrixType& m) { return MatrixU(m.adjoint()); }
  static inline DiagonalScalar getDiag(Scalar x) { return numext::real(x); }
  static inline Scalar getSymm(Scalar x) { return numext::conj(x); }
};

template <typename MatrixType_, int UpLo_, typename Ordering_>
struct traits<SimplicialLDLT<MatrixType_, UpLo_, Ordering_> > {
  typedef MatrixType_ MatrixType;
  typedef Ordering_ OrderingType;
  enum { UpLo = UpLo_ };
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar DiagonalScalar;
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef SparseMatrix<Scalar, ColMajor, StorageIndex> CholMatrixType;
  typedef TriangularView<const CholMatrixType, Eigen::UnitLower> MatrixL;
  typedef TriangularView<const typename CholMatrixType::AdjointReturnType, Eigen::UnitUpper> MatrixU;
  static inline MatrixL getL(const CholMatrixType& m) { return MatrixL(m); }
  static inline MatrixU getU(const CholMatrixType& m) { return MatrixU(m.adjoint()); }
  static inline DiagonalScalar getDiag(Scalar x) { return numext::real(x); }
  static inline Scalar getSymm(Scalar x) { return numext::conj(x); }
};

template <typename MatrixType_, int UpLo_, typename Ordering_>
struct traits<SimplicialNonHermitianLLT<MatrixType_, UpLo_, Ordering_> > {
  typedef MatrixType_ MatrixType;
  typedef Ordering_ OrderingType;
  enum { UpLo = UpLo_ };
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::Scalar DiagonalScalar;
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef SparseMatrix<Scalar, ColMajor, StorageIndex> CholMatrixType;
  typedef TriangularView<const CholMatrixType, Eigen::Lower> MatrixL;
  typedef TriangularView<const typename CholMatrixType::ConstTransposeReturnType, Eigen::Upper> MatrixU;
  static inline MatrixL getL(const CholMatrixType& m) { return MatrixL(m); }
  static inline MatrixU getU(const CholMatrixType& m) { return MatrixU(m.transpose()); }
  static inline DiagonalScalar getDiag(Scalar x) { return x; }
  static inline Scalar getSymm(Scalar x) { return x; }
};

template <typename MatrixType_, int UpLo_, typename Ordering_>
struct traits<SimplicialNonHermitianLDLT<MatrixType_, UpLo_, Ordering_> > {
  typedef MatrixType_ MatrixType;
  typedef Ordering_ OrderingType;
  enum { UpLo = UpLo_ };
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::Scalar DiagonalScalar;
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef SparseMatrix<Scalar, ColMajor, StorageIndex> CholMatrixType;
  typedef TriangularView<const CholMatrixType, Eigen::UnitLower> MatrixL;
  typedef TriangularView<const typename CholMatrixType::ConstTransposeReturnType, Eigen::UnitUpper> MatrixU;
  static inline MatrixL getL(const CholMatrixType& m) { return MatrixL(m); }
  static inline MatrixU getU(const CholMatrixType& m) { return MatrixU(m.transpose()); }
  static inline DiagonalScalar getDiag(Scalar x) { return x; }
  static inline Scalar getSymm(Scalar x) { return x; }
};

template <typename MatrixType_, int UpLo_, typename Ordering_>
struct traits<SimplicialCholesky<MatrixType_, UpLo_, Ordering_> > {
  typedef MatrixType_ MatrixType;
  typedef Ordering_ OrderingType;
  enum { UpLo = UpLo_ };
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar DiagonalScalar;
  static inline DiagonalScalar getDiag(Scalar x) { return numext::real(x); }
  static inline Scalar getSymm(Scalar x) { return numext::conj(x); }
};

}  // namespace internal

/** \ingroup SparseCholesky_Module
 * \class SimplicialLLT
 * \brief A direct sparse LLT Cholesky factorizations
 *
 * This class provides a LL^T Cholesky factorizations of sparse matrices that are
 * selfadjoint and positive definite. The factorization allows for solving A.X = B where
 * X and B can be either dense or sparse.
 *
 * In order to reduce the fill-in, a symmetric permutation P is applied prior to the factorization
 * such that the factorized matrix is P A P^-1.
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ the triangular part that will be used for the computations. It can be Lower
 *               or Upper. Default is Lower.
 * \tparam Ordering_ The ordering method to use, either AMDOrdering<> or NaturalOrdering<>. Default is AMDOrdering<>
 *
 * \implsparsesolverconcept
 *
 * \sa class SimplicialLDLT, class AMDOrdering, class NaturalOrdering
 */
template <typename MatrixType_, int UpLo_, typename Ordering_>
class SimplicialLLT : public SimplicialCholeskyBase<SimplicialLLT<MatrixType_, UpLo_, Ordering_> > {
 public:
  typedef MatrixType_ MatrixType;
  enum { UpLo = UpLo_ };
  typedef SimplicialCholeskyBase<SimplicialLLT> Base;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef SparseMatrix<Scalar, ColMajor, Index> CholMatrixType;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;
  typedef internal::traits<SimplicialLLT> Traits;
  typedef typename Traits::MatrixL MatrixL;
  typedef typename Traits::MatrixU MatrixU;

 public:
  /** Default constructor */
  SimplicialLLT() : Base() {}
  /** Constructs and performs the LLT factorization of \a matrix */
  explicit SimplicialLLT(const MatrixType& matrix) : Base(matrix) {}

  /** \returns an expression of the factor L */
  inline const MatrixL matrixL() const {
    eigen_assert(Base::m_factorizationIsOk && "Simplicial LLT not factorized");
    return Traits::getL(Base::m_matrix);
  }

  /** \returns an expression of the factor U (= L^*) */
  inline const MatrixU matrixU() const {
    eigen_assert(Base::m_factorizationIsOk && "Simplicial LLT not factorized");
    return Traits::getU(Base::m_matrix);
  }

  /** Computes the sparse Cholesky decomposition of \a matrix */
  SimplicialLLT& compute(const MatrixType& matrix) {
    Base::template compute<false, false>(matrix);
    return *this;
  }

  /** Performs a symbolic decomposition on the sparcity of \a matrix.
   *
   * This function is particularly useful when solving for several problems having the same structure.
   *
   * \sa factorize()
   */
  void analyzePattern(const MatrixType& a) { Base::template analyzePattern<false, false>(a); }

  /** Performs a numeric decomposition of \a matrix
   *
   * The given matrix must has the same sparcity than the matrix on which the symbolic decomposition has been performed.
   *
   * \sa analyzePattern()
   */
  void factorize(const MatrixType& a) { Base::template factorize<false, false>(a); }

  /** \returns the determinant of the underlying matrix from the current factorization */
  Scalar determinant() const {
    Scalar detL = Base::m_matrix.diagonal().prod();
    return numext::abs2(detL);
  }
};

/** \ingroup SparseCholesky_Module
 * \class SimplicialLDLT
 * \brief A direct sparse LDLT Cholesky factorizations without square root.
 *
 * This class provides a LDL^T Cholesky factorizations without square root of sparse matrices that are
 * selfadjoint and positive definite. The factorization allows for solving A.X = B where
 * X and B can be either dense or sparse.
 *
 * In order to reduce the fill-in, a symmetric permutation P is applied prior to the factorization
 * such that the factorized matrix is P A P^-1.
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ the triangular part that will be used for the computations. It can be Lower
 *               or Upper. Default is Lower.
 * \tparam Ordering_ The ordering method to use, either AMDOrdering<> or NaturalOrdering<>. Default is AMDOrdering<>
 *
 * \implsparsesolverconcept
 *
 * \sa class SimplicialLLT, class AMDOrdering, class NaturalOrdering
 */
template <typename MatrixType_, int UpLo_, typename Ordering_>
class SimplicialLDLT : public SimplicialCholeskyBase<SimplicialLDLT<MatrixType_, UpLo_, Ordering_> > {
 public:
  typedef MatrixType_ MatrixType;
  enum { UpLo = UpLo_ };
  typedef SimplicialCholeskyBase<SimplicialLDLT> Base;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef SparseMatrix<Scalar, ColMajor, StorageIndex> CholMatrixType;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;
  typedef internal::traits<SimplicialLDLT> Traits;
  typedef typename Traits::MatrixL MatrixL;
  typedef typename Traits::MatrixU MatrixU;

 public:
  /** Default constructor */
  SimplicialLDLT() : Base() {}

  /** Constructs and performs the LLT factorization of \a matrix */
  explicit SimplicialLDLT(const MatrixType& matrix) : Base(matrix) {}

  /** \returns a vector expression of the diagonal D */
  inline const VectorType vectorD() const {
    eigen_assert(Base::m_factorizationIsOk && "Simplicial LDLT not factorized");
    return Base::m_diag;
  }
  /** \returns an expression of the factor L */
  inline const MatrixL matrixL() const {
    eigen_assert(Base::m_factorizationIsOk && "Simplicial LDLT not factorized");
    return Traits::getL(Base::m_matrix);
  }

  /** \returns an expression of the factor U (= L^*) */
  inline const MatrixU matrixU() const {
    eigen_assert(Base::m_factorizationIsOk && "Simplicial LDLT not factorized");
    return Traits::getU(Base::m_matrix);
  }

  /** Computes the sparse Cholesky decomposition of \a matrix */
  SimplicialLDLT& compute(const MatrixType& matrix) {
    Base::template compute<true, false>(matrix);
    return *this;
  }

  /** Performs a symbolic decomposition on the sparcity of \a matrix.
   *
   * This function is particularly useful when solving for several problems having the same structure.
   *
   * \sa factorize()
   */
  void analyzePattern(const MatrixType& a) { Base::template analyzePattern<true, false>(a); }

  /** Performs a numeric decomposition of \a matrix
   *
   * The given matrix must has the same sparcity than the matrix on which the symbolic decomposition has been performed.
   *
   * \sa analyzePattern()
   */
  void factorize(const MatrixType& a) { Base::template factorize<true, false>(a); }

  /** \returns the determinant of the underlying matrix from the current factorization */
  Scalar determinant() const { return Base::m_diag.prod(); }
};

/** \ingroup SparseCholesky_Module
 * \class SimplicialNonHermitianLLT
 * \brief A direct sparse LLT Cholesky factorizations, for symmetric non-hermitian matrices.
 *
 * This class provides a LL^T Cholesky factorizations of sparse matrices that are
 * symmetric but not hermitian. For real matrices, this is equivalent to the regular LLT factorization.
 * The factorization allows for solving A.X = B where X and B can be either dense or sparse.
 *
 * In order to reduce the fill-in, a symmetric permutation P is applied prior to the factorization
 * such that the factorized matrix is P A P^-1.
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ the triangular part that will be used for the computations. It can be Lower
 *               or Upper. Default is Lower.
 * \tparam Ordering_ The ordering method to use, either AMDOrdering<> or NaturalOrdering<>. Default is AMDOrdering<>
 *
 * \implsparsesolverconcept
 *
 * \sa class SimplicialNonHermitianLDLT, SimplicialLLT, class AMDOrdering, class NaturalOrdering
 */
template <typename MatrixType_, int UpLo_, typename Ordering_>
class SimplicialNonHermitianLLT
    : public SimplicialCholeskyBase<SimplicialNonHermitianLLT<MatrixType_, UpLo_, Ordering_> > {
 public:
  typedef MatrixType_ MatrixType;
  enum { UpLo = UpLo_ };
  typedef SimplicialCholeskyBase<SimplicialNonHermitianLLT> Base;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef SparseMatrix<Scalar, ColMajor, StorageIndex> CholMatrixType;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;
  typedef internal::traits<SimplicialNonHermitianLLT> Traits;
  typedef typename Traits::MatrixL MatrixL;
  typedef typename Traits::MatrixU MatrixU;

 public:
  /** Default constructor */
  SimplicialNonHermitianLLT() : Base() {}

  /** Constructs and performs the LLT factorization of \a matrix */
  explicit SimplicialNonHermitianLLT(const MatrixType& matrix) : Base(matrix) {}

  /** \returns an expression of the factor L */
  inline const MatrixL matrixL() const {
    eigen_assert(Base::m_factorizationIsOk && "Simplicial LLT not factorized");
    return Traits::getL(Base::m_matrix);
  }

  /** \returns an expression of the factor U (= L^*) */
  inline const MatrixU matrixU() const {
    eigen_assert(Base::m_factorizationIsOk && "Simplicial LLT not factorized");
    return Traits::getU(Base::m_matrix);
  }

  /** Computes the sparse Cholesky decomposition of \a matrix */
  SimplicialNonHermitianLLT& compute(const MatrixType& matrix) {
    Base::template compute<false, true>(matrix);
    return *this;
  }

  /** Performs a symbolic decomposition on the sparcity of \a matrix.
   *
   * This function is particularly useful when solving for several problems having the same structure.
   *
   * \sa factorize()
   */
  void analyzePattern(const MatrixType& a) { Base::template analyzePattern<false, true>(a); }

  /** Performs a numeric decomposition of \a matrix
   *
   * The given matrix must has the same sparcity than the matrix on which the symbolic decomposition has been performed.
   *
   * \sa analyzePattern()
   */
  void factorize(const MatrixType& a) { Base::template factorize<false, true>(a); }

  /** \returns the determinant of the underlying matrix from the current factorization */
  Scalar determinant() const {
    Scalar detL = Base::m_matrix.diagonal().prod();
    return detL * detL;
  }
};

/** \ingroup SparseCholesky_Module
 * \class SimplicialNonHermitianLDLT
 * \brief A direct sparse LDLT Cholesky factorizations without square root, for symmetric non-hermitian matrices.
 *
 * This class provides a LDL^T Cholesky factorizations without square root of sparse matrices that are
 * symmetric but not hermitian. For real matrices, this is equivalent to the regular LDLT factorization.
 * The factorization allows for solving A.X = B where X and B can be either dense or sparse.
 *
 * In order to reduce the fill-in, a symmetric permutation P is applied prior to the factorization
 * such that the factorized matrix is P A P^-1.
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ the triangular part that will be used for the computations. It can be Lower
 *               or Upper. Default is Lower.
 * \tparam Ordering_ The ordering method to use, either AMDOrdering<> or NaturalOrdering<>. Default is AMDOrdering<>
 *
 * \implsparsesolverconcept
 *
 * \sa class SimplicialNonHermitianLLT, SimplicialLDLT, class AMDOrdering, class NaturalOrdering
 */
template <typename MatrixType_, int UpLo_, typename Ordering_>
class SimplicialNonHermitianLDLT
    : public SimplicialCholeskyBase<SimplicialNonHermitianLDLT<MatrixType_, UpLo_, Ordering_> > {
 public:
  typedef MatrixType_ MatrixType;
  enum { UpLo = UpLo_ };
  typedef SimplicialCholeskyBase<SimplicialNonHermitianLDLT> Base;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef SparseMatrix<Scalar, ColMajor, StorageIndex> CholMatrixType;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;
  typedef internal::traits<SimplicialNonHermitianLDLT> Traits;
  typedef typename Traits::MatrixL MatrixL;
  typedef typename Traits::MatrixU MatrixU;

 public:
  /** Default constructor */
  SimplicialNonHermitianLDLT() : Base() {}

  /** Constructs and performs the LLT factorization of \a matrix */
  explicit SimplicialNonHermitianLDLT(const MatrixType& matrix) : Base(matrix) {}

  /** \returns a vector expression of the diagonal D */
  inline const VectorType vectorD() const {
    eigen_assert(Base::m_factorizationIsOk && "Simplicial LDLT not factorized");
    return Base::m_diag;
  }
  /** \returns an expression of the factor L */
  inline const MatrixL matrixL() const {
    eigen_assert(Base::m_factorizationIsOk && "Simplicial LDLT not factorized");
    return Traits::getL(Base::m_matrix);
  }

  /** \returns an expression of the factor U (= L^*) */
  inline const MatrixU matrixU() const {
    eigen_assert(Base::m_factorizationIsOk && "Simplicial LDLT not factorized");
    return Traits::getU(Base::m_matrix);
  }

  /** Computes the sparse Cholesky decomposition of \a matrix */
  SimplicialNonHermitianLDLT& compute(const MatrixType& matrix) {
    Base::template compute<true, true>(matrix);
    return *this;
  }

  /** Performs a symbolic decomposition on the sparcity of \a matrix.
   *
   * This function is particularly useful when solving for several problems having the same structure.
   *
   * \sa factorize()
   */
  void analyzePattern(const MatrixType& a) { Base::template analyzePattern<true, true>(a); }

  /** Performs a numeric decomposition of \a matrix
   *
   * The given matrix must has the same sparcity than the matrix on which the symbolic decomposition has been performed.
   *
   * \sa analyzePattern()
   */
  void factorize(const MatrixType& a) { Base::template factorize<true, true>(a); }

  /** \returns the determinant of the underlying matrix from the current factorization */
  Scalar determinant() const { return Base::m_diag.prod(); }
};

/** \deprecated use SimplicialLDLT or class SimplicialLLT
 * \ingroup SparseCholesky_Module
 * \class SimplicialCholesky
 *
 * \sa class SimplicialLDLT, class SimplicialLLT
 */
template <typename MatrixType_, int UpLo_, typename Ordering_>
class SimplicialCholesky : public SimplicialCholeskyBase<SimplicialCholesky<MatrixType_, UpLo_, Ordering_> > {
 public:
  typedef MatrixType_ MatrixType;
  enum { UpLo = UpLo_ };
  typedef SimplicialCholeskyBase<SimplicialCholesky> Base;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef SparseMatrix<Scalar, ColMajor, StorageIndex> CholMatrixType;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;
  typedef internal::traits<SimplicialLDLT<MatrixType, UpLo> > LDLTTraits;
  typedef internal::traits<SimplicialLLT<MatrixType, UpLo> > LLTTraits;

 public:
  SimplicialCholesky() : Base(), m_LDLT(true) {}

  explicit SimplicialCholesky(const MatrixType& matrix) : Base(), m_LDLT(true) { compute(matrix); }

  SimplicialCholesky& setMode(SimplicialCholeskyMode mode) {
    switch (mode) {
      case SimplicialCholeskyLLT:
        m_LDLT = false;
        break;
      case SimplicialCholeskyLDLT:
        m_LDLT = true;
        break;
      default:
        break;
    }

    return *this;
  }

  inline const VectorType vectorD() const {
    eigen_assert(Base::m_factorizationIsOk && "Simplicial Cholesky not factorized");
    return Base::m_diag;
  }
  inline const CholMatrixType rawMatrix() const {
    eigen_assert(Base::m_factorizationIsOk && "Simplicial Cholesky not factorized");
    return Base::m_matrix;
  }

  /** Computes the sparse Cholesky decomposition of \a matrix */
  SimplicialCholesky& compute(const MatrixType& matrix) {
    if (m_LDLT)
      Base::template compute<true, false>(matrix);
    else
      Base::template compute<false, false>(matrix);
    return *this;
  }

  /** Performs a symbolic decomposition on the sparcity of \a matrix.
   *
   * This function is particularly useful when solving for several problems having the same structure.
   *
   * \sa factorize()
   */
  void analyzePattern(const MatrixType& a) {
    if (m_LDLT)
      Base::template analyzePattern<true, false>(a);
    else
      Base::template analyzePattern<false, false>(a);
  }

  /** Performs a numeric decomposition of \a matrix
   *
   * The given matrix must has the same sparcity than the matrix on which the symbolic decomposition has been performed.
   *
   * \sa analyzePattern()
   */
  void factorize(const MatrixType& a) {
    if (m_LDLT)
      Base::template factorize<true, false>(a);
    else
      Base::template factorize<false, false>(a);
  }

  /** \internal */
  template <typename Rhs, typename Dest>
  void _solve_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& dest) const {
    eigen_assert(Base::m_factorizationIsOk &&
                 "The decomposition is not in a valid state for solving, you must first call either compute() or "
                 "symbolic()/numeric()");
    eigen_assert(Base::m_matrix.rows() == b.rows());

    if (Base::m_info != Success) return;

    if (Base::m_P.size() > 0)
      dest = Base::m_P * b;
    else
      dest = b;

    if (Base::m_matrix.nonZeros() > 0)  // otherwise L==I
    {
      if (m_LDLT)
        LDLTTraits::getL(Base::m_matrix).solveInPlace(dest);
      else
        LLTTraits::getL(Base::m_matrix).solveInPlace(dest);
    }

    if (Base::m_diag.size() > 0) dest = Base::m_diag.real().asDiagonal().inverse() * dest;

    if (Base::m_matrix.nonZeros() > 0)  // otherwise I==I
    {
      if (m_LDLT)
        LDLTTraits::getU(Base::m_matrix).solveInPlace(dest);
      else
        LLTTraits::getU(Base::m_matrix).solveInPlace(dest);
    }

    if (Base::m_P.size() > 0) dest = Base::m_Pinv * dest;
  }

  /** \internal */
  template <typename Rhs, typename Dest>
  void _solve_impl(const SparseMatrixBase<Rhs>& b, SparseMatrixBase<Dest>& dest) const {
    internal::solve_sparse_through_dense_panels(*this, b, dest);
  }

  Scalar determinant() const {
    if (m_LDLT) {
      return Base::m_diag.prod();
    } else {
      Scalar detL = Diagonal<const CholMatrixType>(Base::m_matrix).prod();
      return numext::abs2(detL);
    }
  }

 protected:
  bool m_LDLT;
};

template <typename Derived>
template <bool NonHermitian>
void SimplicialCholeskyBase<Derived>::ordering(const MatrixType& a, ConstCholMatrixPtr& pmat, CholMatrixType& ap) {
  eigen_assert(a.rows() == a.cols());
  const Index size = a.rows();
  pmat = &ap;
  // Note that ordering methods compute the inverse permutation
  if (!internal::is_same<OrderingType, NaturalOrdering<Index> >::value) {
    {
      CholMatrixType C;
      internal::permute_symm_to_fullsymm<UpLo, NonHermitian>(a, C, NULL);

      OrderingType ordering;
      ordering(C, m_Pinv);
    }

    if (m_Pinv.size() > 0)
      m_P = m_Pinv.inverse();
    else
      m_P.resize(0);

    ap.resize(size, size);
    internal::permute_symm_to_symm<UpLo, Upper, NonHermitian>(a, ap, m_P.indices().data());
  } else {
    m_Pinv.resize(0);
    m_P.resize(0);
    if (int(UpLo) == int(Lower) || MatrixType::IsRowMajor) {
      // we have to transpose the lower part to to the upper one
      ap.resize(size, size);
      internal::permute_symm_to_symm<UpLo, Upper, NonHermitian>(a, ap, NULL);
    } else
      internal::simplicial_cholesky_grab_input<CholMatrixType, MatrixType>::run(a, pmat, ap);
  }
}

}  // end namespace Eigen

#endif  // EIGEN_SIMPLICIAL_CHOLESKY_H
