/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_GEMM_OS_BLAS_HPP
#define CPU_GEMM_OS_BLAS_HPP

/* oneDNN provides gemm functionality on its own using jit generated
 * kernels. This is the only official supported option.
 *
 * However, for the debugging purposes we keep (internally) an ability
 * to use cblas functions from the other libraries. The following macros
 * affect the behavior:
 * - USE_CBLAS allow using sgemm and other regular BLAS functionality
 * - USE_MKL (implies USE_CBLAS) same as above + allow using igemm and
 *   packed gemm from Intel MKL library.
 */

#if defined(USE_MKL)

#if !defined(USE_CBLAS)
#define USE_CBLAS
#endif

#include "mkl_cblas.h"
#include "mkl_version.h"

#define USE_MKL_PACKED_GEMM (INTEL_MKL_VERSION >= 20190001)
#define USE_MKL_IGEMM \
    (INTEL_MKL_VERSION >= 20180000 && __INTEL_MKL_BUILD_DATE >= 20170628)

#else /* defined(USE_MKL) */

#define USE_MKL_PACKED_GEMM 0
#define USE_MKL_IGEMM 0

#if defined(USE_ACCELERATE)
#include "Accelerate.h"
#else

#if defined(USE_CBLAS)

#if defined(_SX)
extern "C" {
#endif

#include "cblas.h"

#if defined(_SX)
}
#endif

#endif /* defined(USE_CBLAS) */
#endif /* defined(USE_ACCELERATE) */
#endif /* defined(USE_MKL) */

#endif /* CPU_GEMM_OS_BLAS_HPP */

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
