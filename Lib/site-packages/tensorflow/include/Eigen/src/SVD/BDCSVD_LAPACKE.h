// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2022 Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
// Copyright (c) 2011, Intel Corporation. All rights reserved.
//
// This file is based on the JacobiSVD_LAPACKE.h originally from Intel -
// see license notice below:
/*
 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of Intel Corporation nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 ********************************************************************************
 *   Content : Eigen bindings to LAPACKe
 *    Singular Value Decomposition - SVD (divide and conquer variant)
 ********************************************************************************
*/
#ifndef EIGEN_BDCSVD_LAPACKE_H
#define EIGEN_BDCSVD_LAPACKE_H

namespace Eigen {

namespace internal {

namespace lapacke_helpers {

/** \internal Specialization for the data types supported by LAPACKe */

// defining a derived class to allow access to protected members
template <typename MatrixType_, int Options>
class BDCSVD_LAPACKE : public BDCSVD<MatrixType_, Options> {
  typedef BDCSVD<MatrixType_, Options> SVD;
  typedef typename SVD::MatrixType MatrixType;
  typedef typename SVD::Scalar Scalar;
  typedef typename SVD::RealScalar RealScalar;

 public:
  // construct this by moving from a parent object
  BDCSVD_LAPACKE(SVD&& svd) : SVD(std::move(svd)) {}

  void compute_impl_lapacke(const MatrixType& matrix, unsigned int computationOptions) {
    SVD::allocate(matrix.rows(), matrix.cols(), computationOptions);

    SVD::m_nonzeroSingularValues = SVD::m_diagSize;

    // prepare arguments to ?gesdd
    const lapack_int matrix_order = lapack_storage_of(matrix);
    const char jobz = (SVD::m_computeFullU || SVD::m_computeFullV)   ? 'A'
                      : (SVD::m_computeThinU || SVD::m_computeThinV) ? 'S'
                                                                     : 'N';
    const lapack_int u_cols = (jobz == 'A') ? to_lapack(SVD::rows()) : (jobz == 'S') ? to_lapack(SVD::diagSize()) : 1;
    const lapack_int vt_rows = (jobz == 'A') ? to_lapack(SVD::cols()) : (jobz == 'S') ? to_lapack(SVD::diagSize()) : 1;
    lapack_int ldu, ldvt;
    Scalar *u, *vt, dummy;
    MatrixType localU;
    if (SVD::computeU() && !(SVD::m_computeThinU && SVD::m_computeFullV)) {
      ldu = to_lapack(SVD::m_matrixU.outerStride());
      u = SVD::m_matrixU.data();
    } else if (SVD::computeV()) {
      localU.resize(SVD::rows(), u_cols);
      ldu = to_lapack(localU.outerStride());
      u = localU.data();
    } else {
      ldu = 1;
      u = &dummy;
    }
    MatrixType localV;
    if (SVD::computeU() || SVD::computeV()) {
      localV.resize(vt_rows, SVD::cols());
      ldvt = to_lapack(localV.outerStride());
      vt = localV.data();
    } else {
      ldvt = 1;
      vt = &dummy;
    }
    MatrixType temp;
    temp = matrix;

    // actual call to ?gesdd
    lapack_int info = gesdd(matrix_order, jobz, to_lapack(SVD::rows()), to_lapack(SVD::cols()), to_lapack(temp.data()),
                            to_lapack(temp.outerStride()), (RealScalar*)SVD::m_singularValues.data(), to_lapack(u), ldu,
                            to_lapack(vt), ldvt);

    // Check the result of the LAPACK call
    if (info < 0 || !SVD::m_singularValues.allFinite()) {
      // this includes info == -4 => NaN entry in A
      SVD::m_info = InvalidInput;
    } else if (info > 0) {
      SVD::m_info = NoConvergence;
    } else {
      SVD::m_info = Success;
      if (SVD::m_computeThinU && SVD::m_computeFullV) {
        SVD::m_matrixU = localU.leftCols(SVD::m_matrixU.cols());
      }
      if (SVD::computeV()) {
        SVD::m_matrixV = localV.adjoint().leftCols(SVD::m_matrixV.cols());
      }
    }
    SVD::m_isInitialized = true;
  }
};

template <typename MatrixType_, int Options>
BDCSVD<MatrixType_, Options>& BDCSVD_wrapper(BDCSVD<MatrixType_, Options>& svd, const MatrixType_& matrix,
                                             int computationOptions) {
  // we need to move to the wrapper type and back
  BDCSVD_LAPACKE<MatrixType_, Options> tmpSvd(std::move(svd));
  tmpSvd.compute_impl_lapacke(matrix, computationOptions);
  svd = std::move(tmpSvd);
  return svd;
}

}  // end namespace lapacke_helpers

}  // end namespace internal

#define EIGEN_LAPACKE_SDD(EIGTYPE, EIGCOLROW, OPTIONS)                                                                 \
  template <>                                                                                                          \
  inline BDCSVD<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic>, OPTIONS>&                              \
  BDCSVD<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic>, OPTIONS>::compute_impl(                       \
      const Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic>& matrix, unsigned int computationOptions) { \
    return internal::lapacke_helpers::BDCSVD_wrapper(*this, matrix, computationOptions);                               \
  }

#define EIGEN_LAPACK_SDD_OPTIONS(OPTIONS)        \
  EIGEN_LAPACKE_SDD(double, ColMajor, OPTIONS)   \
  EIGEN_LAPACKE_SDD(float, ColMajor, OPTIONS)    \
  EIGEN_LAPACKE_SDD(dcomplex, ColMajor, OPTIONS) \
  EIGEN_LAPACKE_SDD(scomplex, ColMajor, OPTIONS) \
                                                 \
  EIGEN_LAPACKE_SDD(double, RowMajor, OPTIONS)   \
  EIGEN_LAPACKE_SDD(float, RowMajor, OPTIONS)    \
  EIGEN_LAPACKE_SDD(dcomplex, RowMajor, OPTIONS) \
  EIGEN_LAPACKE_SDD(scomplex, RowMajor, OPTIONS)

EIGEN_LAPACK_SDD_OPTIONS(0)
EIGEN_LAPACK_SDD_OPTIONS(ComputeThinU)
EIGEN_LAPACK_SDD_OPTIONS(ComputeThinV)
EIGEN_LAPACK_SDD_OPTIONS(ComputeFullU)
EIGEN_LAPACK_SDD_OPTIONS(ComputeFullV)
EIGEN_LAPACK_SDD_OPTIONS(ComputeThinU | ComputeThinV)
EIGEN_LAPACK_SDD_OPTIONS(ComputeFullU | ComputeFullV)
EIGEN_LAPACK_SDD_OPTIONS(ComputeThinU | ComputeFullV)
EIGEN_LAPACK_SDD_OPTIONS(ComputeFullU | ComputeThinV)

#undef EIGEN_LAPACK_SDD_OPTIONS

#undef EIGEN_LAPACKE_SDD

}  // end namespace Eigen

#endif  // EIGEN_BDCSVD_LAPACKE_H
