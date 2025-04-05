/*
 Copyright (c) 2011, Intel Corporation. All rights reserved.

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
 *     LLt decomposition based on LAPACKE_?potrf function.
 ********************************************************************************
*/

#ifndef EIGEN_LLT_LAPACKE_H
#define EIGEN_LLT_LAPACKE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

namespace lapacke_helpers {
// -------------------------------------------------------------------------------------------------------------------
//        Dispatch for rank update handling upper and lower parts
// -------------------------------------------------------------------------------------------------------------------

template <UpLoType Mode>
struct rank_update {};

template <>
struct rank_update<Lower> {
  template <typename MatrixType, typename VectorType>
  static Index run(MatrixType &mat, const VectorType &vec, const typename MatrixType::RealScalar &sigma) {
    return Eigen::internal::llt_rank_update_lower(mat, vec, sigma);
  }
};

template <>
struct rank_update<Upper> {
  template <typename MatrixType, typename VectorType>
  static Index run(MatrixType &mat, const VectorType &vec, const typename MatrixType::RealScalar &sigma) {
    Transpose<MatrixType> matt(mat);
    return Eigen::internal::llt_rank_update_lower(matt, vec.conjugate(), sigma);
  }
};

// -------------------------------------------------------------------------------------------------------------------
//        Generic lapacke llt implementation that hands of to the dispatches
// -------------------------------------------------------------------------------------------------------------------

template <typename Scalar, UpLoType Mode>
struct lapacke_llt {
  EIGEN_STATIC_ASSERT(((Mode == Lower) || (Mode == Upper)), MODE_MUST_BE_UPPER_OR_LOWER)
  template <typename MatrixType>
  static Index blocked(MatrixType &m) {
    eigen_assert(m.rows() == m.cols());
    if (m.rows() == 0) {
      return -1;
    }
    /* Set up parameters for ?potrf */
    lapack_int size = to_lapack(m.rows());
    lapack_int matrix_order = lapack_storage_of(m);
    constexpr char uplo = Mode == Upper ? 'U' : 'L';
    Scalar *a = &(m.coeffRef(0, 0));
    lapack_int lda = to_lapack(m.outerStride());

    lapack_int info = potrf(matrix_order, uplo, size, to_lapack(a), lda);
    info = (info == 0) ? -1 : info > 0 ? info - 1 : size;
    return info;
  }

  template <typename MatrixType, typename VectorType>
  static Index rankUpdate(MatrixType &mat, const VectorType &vec, const typename MatrixType::RealScalar &sigma) {
    return rank_update<Mode>::run(mat, vec, sigma);
  }
};
}  // namespace lapacke_helpers
// end namespace lapacke_helpers

/*
 * Here, we just put the generic implementation from lapacke_llt into a full specialization of the llt_inplace
 * type. By being a full specialization, the versions defined here thus get precedence over the generic implementation
 * in LLT.h for double, float and complex double, complex float types.
 */

#define EIGEN_LAPACKE_LLT(EIGTYPE)                                                             \
  template <>                                                                                  \
  struct llt_inplace<EIGTYPE, Lower> : public lapacke_helpers::lapacke_llt<EIGTYPE, Lower> {}; \
  template <>                                                                                  \
  struct llt_inplace<EIGTYPE, Upper> : public lapacke_helpers::lapacke_llt<EIGTYPE, Upper> {};

EIGEN_LAPACKE_LLT(double)
EIGEN_LAPACKE_LLT(float)
EIGEN_LAPACKE_LLT(std::complex<double>)
EIGEN_LAPACKE_LLT(std::complex<float>)

#undef EIGEN_LAPACKE_LLT

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_LLT_LAPACKE_H
