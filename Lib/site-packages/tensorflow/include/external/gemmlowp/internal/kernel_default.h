// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// kernel_default.h: Chooses default GEMM and GEMV kernels for the
// host platform.

#ifndef GEMMLOWP_INTERNAL_KERNEL_DEFAULT_H_
#define GEMMLOWP_INTERNAL_KERNEL_DEFAULT_H_

#include "../public/bit_depth.h"
#include "common.h"
#include "kernel.h"
#include "kernel_reference.h"

namespace gemmlowp {

template <bool MaxProductIsLessThan4096, bool IsUnsigned, bool LhsNonZero>
struct DefaultKernelImpl {};

// Partial specialization implementing the logic that if we want to use
// a kernel for MaxProductIsLessThan4096 but do not have such a kernel, then we
// fall back to a generic kernel not taking advantage of
// MaxProductIsLessThan4096.
template <bool LhsNonZero>
struct DefaultKernelImpl<true, true, LhsNonZero>
    : DefaultKernelImpl<false, true, LhsNonZero> {};

// Partial specialization implementing the logic that if we want to use
// a kernel for LhsNonZero but do not have such a kernel, then we fall
// back to a generic kernel not taking advantage of LhsNonZero.
template <bool MaxProductIsLessThan4096>
struct DefaultKernelImpl<MaxProductIsLessThan4096, true, true>
    : DefaultKernelImpl<MaxProductIsLessThan4096, true, false> {};

template <typename BitDepthParams>
struct DefaultKernel
    : DefaultKernelImpl<(BitDepthParams::LhsRange::kMaxValue *
                             BitDepthParams::RhsRange::kMaxValue <
                         4096),
                        (BitDepthParams::LhsRange::kMinValue >= 0),
                        (BitDepthParams::LhsRange::kMinValue > 0 ||
                         (BitDepthParams::LhsRange::kMaxValue <= 127 &&
                          BitDepthParams::LhsRange::kMinValue > -128))> {};

}  // end namespace gemmlowp

#define GEMMLOWP_SET_DEFAULT_KERNEL(MaxProductIsLessThan4096, IsUnsigned, \
                                    LhsAlwaysNonZero, Kernel)             \
  namespace gemmlowp {                                                    \
  template <>                                                             \
  struct DefaultKernelImpl<MaxProductIsLessThan4096, IsUnsigned,          \
                           LhsAlwaysNonZero> : Kernel {};                 \
  }

// User-provided int8 inputs is only supported in the NEON path currently.
#if defined GEMMLOWP_NEON_32
#include "kernel_neon.h"
GEMMLOWP_SET_DEFAULT_KERNEL(false, true, false, NEON_32_Kernel12x4Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(true, true, false,
                            NEON_32_Kernel12x4Depth2Assuming12BitProducts)
GEMMLOWP_SET_DEFAULT_KERNEL(false, true, true,
                            NEON_32bit_GEMM_Int8Operands_LhsNonzero)
GEMMLOWP_SET_DEFAULT_KERNEL(false, false, true,
                            NEON_32bit_GEMM_Int8Operands_LhsNonzero_Int8Inputs)
#elif defined GEMMLOWP_NEON_64
#include "kernel_neon.h"
#if defined GEMMLOWP_DOTPROD_KERNEL
GEMMLOWP_SET_DEFAULT_KERNEL(false, true, false,
                            NEON_64_Kernel12x8Depth4_dotprod)
#else
GEMMLOWP_SET_DEFAULT_KERNEL(false, true, false, NEON_64_Kernel12x8Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(false, true, true,
                            NEON_64bit_GEMM_Int8Operands_LhsNonzero)
#endif
GEMMLOWP_SET_DEFAULT_KERNEL(false, false, true,
                            NEON_64bit_GEMM_Int8Operands_LhsNonzero_Int8Inputs)
#elif defined(GEMMLOWP_MSA)
#include "kernel_msa.h"
GEMMLOWP_SET_DEFAULT_KERNEL(false, true, false, MSA_Kernel12x8Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(false, true, true, MSA_GEMM_Int8Operands_LhsNonzero)
#elif defined GEMMLOWP_SSE4_32
#include "kernel_sse.h"
GEMMLOWP_SET_DEFAULT_KERNEL(false, true, false, SSE4_32_Kernel4x4Depth2)
#elif defined GEMMLOWP_SSE4_64
#include "kernel_sse.h"
GEMMLOWP_SET_DEFAULT_KERNEL(false, true, false, SSE4_64_Kernel12x4Depth2)
#elif defined GEMMLOWP_AVX2_64
#include "kernel_avx.h"
GEMMLOWP_SET_DEFAULT_KERNEL(false, true, false, AVX2_64_Kernel24x8Depth2)
#else
#include "kernel_reference.h"
namespace gemmlowp {
typedef ReferenceKernel<KernelFormat<
    KernelSideFormat<CellFormat<4, 16, CellOrder::WidthMajor>, 1>,
    KernelSideFormat<CellFormat<4, 16, CellOrder::WidthMajor>, 1> > >
    DefaultReferenceKernel;
}
GEMMLOWP_SET_DEFAULT_KERNEL(false, true, false, DefaultReferenceKernel)
#endif

#endif  // GEMMLOWP_INTERNAL_KERNEL_DEFAULT_H_
