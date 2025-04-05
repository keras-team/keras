/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef CPU_GEMM_BF16_REF_GEMM_BF16_HPP
#define CPU_GEMM_BF16_REF_GEMM_BF16_HPP

#include "oneapi/dnnl/dnnl_types.h"

#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

dnnl_status_t ref_gemm_bf16bf16f32(const char *transa, const char *transb,
        const dim_t *M, const dim_t *N, const dim_t *K, const float *alpha,
        const bfloat16_t *A, const dim_t *lda, const bfloat16_t *B,
        const dim_t *ldb, const float *beta, float *C, const dim_t *ldc);

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_GEMM_F32_REF_GEMM_F32_HPP
