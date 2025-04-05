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
 *    Householder QR decomposition of a matrix w/o pivoting based on
 *    LAPACKE_?geqrf function.
 ********************************************************************************
*/

#ifndef EIGEN_QR_LAPACKE_H
#define EIGEN_QR_LAPACKE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

namespace lapacke_helpers {

template <typename MatrixQR, typename HCoeffs>
struct lapacke_hqr {
  static void run(MatrixQR& mat, HCoeffs& hCoeffs, Index = 32, typename MatrixQR::Scalar* = 0) {
    lapack_int m = to_lapack(mat.rows());
    lapack_int n = to_lapack(mat.cols());
    lapack_int lda = to_lapack(mat.outerStride());
    lapack_int matrix_order = lapack_storage_of(mat);
    geqrf(matrix_order, m, n, to_lapack(mat.data()), lda, to_lapack(hCoeffs.data()));
    hCoeffs.adjointInPlace();
  }
};

}  // namespace lapacke_helpers

/** \internal Specialization for the data types supported by LAPACKe */
#define EIGEN_LAPACKE_HH_QR(EIGTYPE)                                      \
  template <typename MatrixQR, typename HCoeffs>                          \
  struct householder_qr_inplace_blocked<MatrixQR, HCoeffs, EIGTYPE, true> \
      : public lapacke_helpers::lapacke_hqr<MatrixQR, HCoeffs> {};

EIGEN_LAPACKE_HH_QR(double)
EIGEN_LAPACKE_HH_QR(float)
EIGEN_LAPACKE_HH_QR(std::complex<double>)
EIGEN_LAPACKE_HH_QR(std::complex<float>)

#undef EIGEN_LAPACKE_HH_QR

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_QR_LAPACKE_H
