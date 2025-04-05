/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#ifndef CPU_GEMM_S8X8S32_REF_GEMM_S8X8S32_HPP
#define CPU_GEMM_S8X8S32_REF_GEMM_S8X8S32_HPP

#include <cstdint>

#include "oneapi/dnnl/dnnl_types.h"

#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <typename b_dt>
dnnl_status_t ref_gemm_s8x8s32(const char *transa, const char *transb,
        const char *offsetc, const dim_t *M, const dim_t *N, const dim_t *K,
        const float *alpha, const int8_t *A, const dim_t *LDA, const int8_t *ao,
        const b_dt *B, const dim_t *LDB, const b_dt *bo, const float *beta,
        int32_t *C, const dim_t *LDC, const int32_t *co);
}
} // namespace impl
} // namespace dnnl
#endif // CPU_GEMM_S8X8S32_REF_GEMM_S8X8S32_HPP
