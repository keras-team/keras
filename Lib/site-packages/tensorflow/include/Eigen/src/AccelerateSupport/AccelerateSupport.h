#ifndef EIGEN_ACCELERATESUPPORT_H
#define EIGEN_ACCELERATESUPPORT_H

#include <Accelerate/Accelerate.h>

#include <Eigen/Sparse>

namespace Eigen {

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
class AccelerateImpl;

/** \ingroup AccelerateSupport_Module
 * \class AccelerateLLT
 * \brief A direct Cholesky (LLT) factorization and solver based on Accelerate
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ additional information about the matrix structure. Default is Lower.
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateLLT
 */
template <typename MatrixType, int UpLo = Lower>
using AccelerateLLT = AccelerateImpl<MatrixType, UpLo | Symmetric, SparseFactorizationCholesky, true>;

/** \ingroup AccelerateSupport_Module
 * \class AccelerateLDLT
 * \brief The default Cholesky (LDLT) factorization and solver based on Accelerate
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ additional information about the matrix structure. Default is Lower.
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateLDLT
 */
template <typename MatrixType, int UpLo = Lower>
using AccelerateLDLT = AccelerateImpl<MatrixType, UpLo | Symmetric, SparseFactorizationLDLT, true>;

/** \ingroup AccelerateSupport_Module
 * \class AccelerateLDLTUnpivoted
 * \brief A direct Cholesky-like LDL^T factorization and solver based on Accelerate with only 1x1 pivots and no pivoting
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ additional information about the matrix structure. Default is Lower.
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateLDLTUnpivoted
 */
template <typename MatrixType, int UpLo = Lower>
using AccelerateLDLTUnpivoted = AccelerateImpl<MatrixType, UpLo | Symmetric, SparseFactorizationLDLTUnpivoted, true>;

/** \ingroup AccelerateSupport_Module
 * \class AccelerateLDLTSBK
 * \brief A direct Cholesky (LDLT) factorization and solver based on Accelerate with Supernode Bunch-Kaufman and static
 * pivoting
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ additional information about the matrix structure. Default is Lower.
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateLDLTSBK
 */
template <typename MatrixType, int UpLo = Lower>
using AccelerateLDLTSBK = AccelerateImpl<MatrixType, UpLo | Symmetric, SparseFactorizationLDLTSBK, true>;

/** \ingroup AccelerateSupport_Module
 * \class AccelerateLDLTTPP
 * \brief A direct Cholesky (LDLT) factorization and solver based on Accelerate with full threshold partial pivoting
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ additional information about the matrix structure. Default is Lower.
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateLDLTTPP
 */
template <typename MatrixType, int UpLo = Lower>
using AccelerateLDLTTPP = AccelerateImpl<MatrixType, UpLo | Symmetric, SparseFactorizationLDLTTPP, true>;

/** \ingroup AccelerateSupport_Module
 * \class AccelerateQR
 * \brief A QR factorization and solver based on Accelerate
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateQR
 */
template <typename MatrixType>
using AccelerateQR = AccelerateImpl<MatrixType, 0, SparseFactorizationQR, false>;

/** \ingroup AccelerateSupport_Module
 * \class AccelerateCholeskyAtA
 * \brief A QR factorization and solver based on Accelerate without storing Q (equivalent to A^TA = R^T R)
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateCholeskyAtA
 */
template <typename MatrixType>
using AccelerateCholeskyAtA = AccelerateImpl<MatrixType, 0, SparseFactorizationCholeskyAtA, false>;

namespace internal {
template <typename T>
struct AccelFactorizationDeleter {
  void operator()(T* sym) {
    if (sym) {
      SparseCleanup(*sym);
      delete sym;
      sym = nullptr;
    }
  }
};

template <typename DenseVecT, typename DenseMatT, typename SparseMatT, typename NumFactT>
struct SparseTypesTraitBase {
  typedef DenseVecT AccelDenseVector;
  typedef DenseMatT AccelDenseMatrix;
  typedef SparseMatT AccelSparseMatrix;

  typedef SparseOpaqueSymbolicFactorization SymbolicFactorization;
  typedef NumFactT NumericFactorization;

  typedef AccelFactorizationDeleter<SymbolicFactorization> SymbolicFactorizationDeleter;
  typedef AccelFactorizationDeleter<NumericFactorization> NumericFactorizationDeleter;
};

template <typename Scalar>
struct SparseTypesTrait {};

template <>
struct SparseTypesTrait<double> : SparseTypesTraitBase<DenseVector_Double, DenseMatrix_Double, SparseMatrix_Double,
                                                       SparseOpaqueFactorization_Double> {};

template <>
struct SparseTypesTrait<float>
    : SparseTypesTraitBase<DenseVector_Float, DenseMatrix_Float, SparseMatrix_Float, SparseOpaqueFactorization_Float> {
};

}  // end namespace internal

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
class AccelerateImpl : public SparseSolverBase<AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_> > {
 protected:
  using Base = SparseSolverBase<AccelerateImpl>;
  using Base::derived;
  using Base::m_isInitialized;

 public:
  using Base::_solve_impl;

  typedef MatrixType_ MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::StorageIndex StorageIndex;
  enum { ColsAtCompileTime = Dynamic, MaxColsAtCompileTime = Dynamic };
  enum { UpLo = UpLo_ };

  using AccelDenseVector = typename internal::SparseTypesTrait<Scalar>::AccelDenseVector;
  using AccelDenseMatrix = typename internal::SparseTypesTrait<Scalar>::AccelDenseMatrix;
  using AccelSparseMatrix = typename internal::SparseTypesTrait<Scalar>::AccelSparseMatrix;
  using SymbolicFactorization = typename internal::SparseTypesTrait<Scalar>::SymbolicFactorization;
  using NumericFactorization = typename internal::SparseTypesTrait<Scalar>::NumericFactorization;
  using SymbolicFactorizationDeleter = typename internal::SparseTypesTrait<Scalar>::SymbolicFactorizationDeleter;
  using NumericFactorizationDeleter = typename internal::SparseTypesTrait<Scalar>::NumericFactorizationDeleter;

  AccelerateImpl() {
    m_isInitialized = false;

    auto check_flag_set = [](int value, int flag) { return ((value & flag) == flag); };

    if (check_flag_set(UpLo_, Symmetric)) {
      m_sparseKind = SparseSymmetric;
      m_triType = (UpLo_ & Lower) ? SparseLowerTriangle : SparseUpperTriangle;
    } else if (check_flag_set(UpLo_, UnitLower)) {
      m_sparseKind = SparseUnitTriangular;
      m_triType = SparseLowerTriangle;
    } else if (check_flag_set(UpLo_, UnitUpper)) {
      m_sparseKind = SparseUnitTriangular;
      m_triType = SparseUpperTriangle;
    } else if (check_flag_set(UpLo_, StrictlyLower)) {
      m_sparseKind = SparseTriangular;
      m_triType = SparseLowerTriangle;
    } else if (check_flag_set(UpLo_, StrictlyUpper)) {
      m_sparseKind = SparseTriangular;
      m_triType = SparseUpperTriangle;
    } else if (check_flag_set(UpLo_, Lower)) {
      m_sparseKind = SparseTriangular;
      m_triType = SparseLowerTriangle;
    } else if (check_flag_set(UpLo_, Upper)) {
      m_sparseKind = SparseTriangular;
      m_triType = SparseUpperTriangle;
    } else {
      m_sparseKind = SparseOrdinary;
      m_triType = (UpLo_ & Lower) ? SparseLowerTriangle : SparseUpperTriangle;
    }

    m_order = SparseOrderDefault;
  }

  explicit AccelerateImpl(const MatrixType& matrix) : AccelerateImpl() { compute(matrix); }

  ~AccelerateImpl() {}

  inline Index cols() const { return m_nCols; }
  inline Index rows() const { return m_nRows; }

  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "Decomposition is not initialized.");
    return m_info;
  }

  void analyzePattern(const MatrixType& matrix);

  void factorize(const MatrixType& matrix);

  void compute(const MatrixType& matrix);

  template <typename Rhs, typename Dest>
  void _solve_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& dest) const;

  /** Sets the ordering algorithm to use. */
  void setOrder(SparseOrder_t order) { m_order = order; }

 private:
  template <typename T>
  void buildAccelSparseMatrix(const SparseMatrix<T>& a, AccelSparseMatrix& A, std::vector<long>& columnStarts) {
    const Index nColumnsStarts = a.cols() + 1;

    columnStarts.resize(nColumnsStarts);

    for (Index i = 0; i < nColumnsStarts; i++) columnStarts[i] = a.outerIndexPtr()[i];

    SparseAttributes_t attributes{};
    attributes.transpose = false;
    attributes.triangle = m_triType;
    attributes.kind = m_sparseKind;

    SparseMatrixStructure structure{};
    structure.attributes = attributes;
    structure.rowCount = static_cast<int>(a.rows());
    structure.columnCount = static_cast<int>(a.cols());
    structure.blockSize = 1;
    structure.columnStarts = columnStarts.data();
    structure.rowIndices = const_cast<int*>(a.innerIndexPtr());

    A.structure = structure;
    A.data = const_cast<T*>(a.valuePtr());
  }

  void doAnalysis(AccelSparseMatrix& A) {
    m_numericFactorization.reset(nullptr);

    SparseSymbolicFactorOptions opts{};
    opts.control = SparseDefaultControl;
    opts.orderMethod = m_order;
    opts.order = nullptr;
    opts.ignoreRowsAndColumns = nullptr;
    opts.malloc = malloc;
    opts.free = free;
    opts.reportError = nullptr;

    m_symbolicFactorization.reset(new SymbolicFactorization(SparseFactor(Solver_, A.structure, opts)));

    SparseStatus_t status = m_symbolicFactorization->status;

    updateInfoStatus(status);

    if (status != SparseStatusOK) m_symbolicFactorization.reset(nullptr);
  }

  void doFactorization(AccelSparseMatrix& A) {
    SparseStatus_t status = SparseStatusReleased;

    if (m_symbolicFactorization) {
      m_numericFactorization.reset(new NumericFactorization(SparseFactor(*m_symbolicFactorization, A)));

      status = m_numericFactorization->status;

      if (status != SparseStatusOK) m_numericFactorization.reset(nullptr);
    }

    updateInfoStatus(status);
  }

 protected:
  void updateInfoStatus(SparseStatus_t status) const {
    switch (status) {
      case SparseStatusOK:
        m_info = Success;
        break;
      case SparseFactorizationFailed:
      case SparseMatrixIsSingular:
        m_info = NumericalIssue;
        break;
      case SparseInternalError:
      case SparseParameterError:
      case SparseStatusReleased:
      default:
        m_info = InvalidInput;
        break;
    }
  }

  mutable ComputationInfo m_info;
  Index m_nRows, m_nCols;
  std::unique_ptr<SymbolicFactorization, SymbolicFactorizationDeleter> m_symbolicFactorization;
  std::unique_ptr<NumericFactorization, NumericFactorizationDeleter> m_numericFactorization;
  SparseKind_t m_sparseKind;
  SparseTriangle_t m_triType;
  SparseOrder_t m_order;
};

/** Computes the symbolic and numeric decomposition of matrix \a a */
template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
void AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::compute(const MatrixType& a) {
  if (EnforceSquare_) eigen_assert(a.rows() == a.cols());

  m_nRows = a.rows();
  m_nCols = a.cols();

  AccelSparseMatrix A{};
  std::vector<long> columnStarts;

  buildAccelSparseMatrix(a, A, columnStarts);

  doAnalysis(A);

  if (m_symbolicFactorization) doFactorization(A);

  m_isInitialized = true;
}

/** Performs a symbolic decomposition on the sparsity pattern of matrix \a a.
 *
 * This function is particularly useful when solving for several problems having the same structure.
 *
 * \sa factorize()
 */
template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
void AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::analyzePattern(const MatrixType& a) {
  if (EnforceSquare_) eigen_assert(a.rows() == a.cols());

  m_nRows = a.rows();
  m_nCols = a.cols();

  AccelSparseMatrix A{};
  std::vector<long> columnStarts;

  buildAccelSparseMatrix(a, A, columnStarts);

  doAnalysis(A);

  m_isInitialized = true;
}

/** Performs a numeric decomposition of matrix \a a.
 *
 * The given matrix must have the same sparsity pattern as the matrix on which the symbolic decomposition has been
 * performed.
 *
 * \sa analyzePattern()
 */
template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
void AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::factorize(const MatrixType& a) {
  eigen_assert(m_symbolicFactorization && "You must first call analyzePattern()");
  eigen_assert(m_nRows == a.rows() && m_nCols == a.cols());

  if (EnforceSquare_) eigen_assert(a.rows() == a.cols());

  AccelSparseMatrix A{};
  std::vector<long> columnStarts;

  buildAccelSparseMatrix(a, A, columnStarts);

  doFactorization(A);
}

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
template <typename Rhs, typename Dest>
void AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::_solve_impl(const MatrixBase<Rhs>& b,
                                                                              MatrixBase<Dest>& x) const {
  if (!m_numericFactorization) {
    m_info = InvalidInput;
    return;
  }

  eigen_assert(m_nRows == b.rows());
  eigen_assert(((b.cols() == 1) || b.outerStride() == b.rows()));

  SparseStatus_t status = SparseStatusOK;

  Scalar* b_ptr = const_cast<Scalar*>(b.derived().data());
  Scalar* x_ptr = const_cast<Scalar*>(x.derived().data());

  AccelDenseMatrix xmat{};
  xmat.attributes = SparseAttributes_t();
  xmat.columnCount = static_cast<int>(x.cols());
  xmat.rowCount = static_cast<int>(x.rows());
  xmat.columnStride = xmat.rowCount;
  xmat.data = x_ptr;

  AccelDenseMatrix bmat{};
  bmat.attributes = SparseAttributes_t();
  bmat.columnCount = static_cast<int>(b.cols());
  bmat.rowCount = static_cast<int>(b.rows());
  bmat.columnStride = bmat.rowCount;
  bmat.data = b_ptr;

  SparseSolve(*m_numericFactorization, bmat, xmat);

  updateInfoStatus(status);
}

}  // end namespace Eigen

#endif  // EIGEN_ACCELERATESUPPORT_H
